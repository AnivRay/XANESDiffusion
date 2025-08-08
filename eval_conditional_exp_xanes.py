import argparse
from os.path import join
import torch
import numpy as np
import pickle
from qm9.models import get_model, get_autoencoder, get_latent_diffusion
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from xanes.sampling import sample_chain, sample, sample_sweep_conditional
import qm9.visualizer as vis

def get_exp_values():
    expXanes = np.load("data/exp/Fe_all_morphed_xanes_interp.npy")
    values = {'xanes': torch.tensor(expXanes)}
    nodesxsample = torch.tensor([7, 5, 5, 5, 5]).int()
    return values, nodesxsample

def get_and_save_values_from_dataloader(args, dataloader, properties, dataset_info):
    values = {}
    nodesxsample = []
    for property_key in properties:
        values[property_key] = []
    id_from = 0
    for data in dataloader:
        for property_key in properties:
            values[property_key].append(data[property_key][:, 0, :])
        nodesxsample.append(torch.sum(data['atom_mask'], dim=1))
        # Save batch of ground truth data starting from where previous batch left off
        vis.save_xyz_file('outputs/%s/analysis/groundtruth/' % (args.exp_name), data['one_hot'].float(), data['charges'], data['positions'], dataset_info, id_from, name='gt', node_mask=data['atom_mask'])
        id_from += len(data['positions'])
    for key in values:
        values[key] = torch.cat(values[key])
    nodesxsample = torch.cat(nodesxsample).int()

    # Save xanes specta
    np.save('outputs/{}/analysis/xanes_conditioning.npy'.format(args.exp_name), values['xanes'].numpy())
    return values, nodesxsample

def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    print("Dataset name: ", args_gen.dataset)
    # assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_generator(dir_path, dataloaders, device, args_gen, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model, nodes_dist, prop_dist = get_latent_diffusion(args_gen, device, dataset_info, dataloaders['train'])
    fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


class DiffusionDataloader:
    def __init__(self, args_gen, model, nodes_dist, prop_dist, device, unkown_labels=False,
                 batch_size=1, iterations=200):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)

        node_mask = node_mask.squeeze(2)
        context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        prop_key = self.prop_dist.properties[0]
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        else:
            context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        data = {
            'positions': x.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            'one_hot': one_hot.detach(),
            prop_key: context.detach()
        }
        return data

    def __next__(self):
        if self.i <= self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations


def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, _ = get_generator(args.generators_path, dataloaders,
                                                    args.device, args_gen, property_norms)

    # Create a dataloader with the generator

    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    if args.task == 'edm':
        diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist,
                                                   args.device, batch_size=args.batch_size, iterations=args.iterations)
        print("EDM: We evaluate the classifier on our generated samples")
        loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
        print("Loss classifier on Generated samples: %.4f" % loss)
    elif args.task == 'qm9_second_half':
        print("qm9_second_half: We evaluate the classifier on QM9")
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on qm9_second_half: %.4f" % loss)
    elif args.task == 'naive':
        print("Naive: We evaluate the classifier on QM9")
        length = dataloaders['train'].dataset.data[args.property].size(0)
        idxs = torch.randperm(length)
        dataloaders['train'].dataset.data[args.property] = dataloaders['train'].dataset.data[args.property][idxs]
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on naive: %.4f" % loss)
    #elif args.task == 'numnodes':
    #    print("Numnodes: We evaluate the numnodes classifier on EDM samples")
    #    diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist, device,
    #                                               batch_size=args.batch_size, iterations=args.iterations)
    #    loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
    #    print("Loss numnodes classifier on EDM generated samples: %.4f" % loss)


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, values, property_norms, nodesxsample, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist, values, property_norms, nodesxsample)
    vis.save_xyz_file(
        'outputs/%s/analysis_exp/run%s/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    # vis.visualize_chain("outputs/%s/analysis/run%s/" % (args.exp_name, epoch), dataset_info,
    #                     wandb=None, mode='conditional', spheres_3d=True)

    return one_hot, charges, x

def modify_for_num_nodes(values, nodesxsample, dataset_info):
    n_nodes = list(dataset_info["n_nodes"].keys())
    print("n_nodes:", n_nodes)
    new_nodesxsample = torch.tensor(n_nodes).repeat(nodesxsample.size(0))
    values['xanes'] = values['xanes'].repeat_interleave(len(n_nodes), dim=0)
    print("GT Nodes:", nodesxsample[:20])
    return values, new_nodesxsample

def get_conditional_num_nodes(args):
    metal = args.generators_path.split("/")[-1].split("_")[0].title()
    nodesxsample_new = torch.from_numpy(np.load('outputs/CN_classifier/{}_CN_rf_model_pred.npy'.format(metal))) + 1
    return nodesxsample_new

def get_unconditional_num_nodes(nodes_dist, nodesxsample):
    nodesxsample_new = nodes_dist.sample(nodesxsample.size(0))
    return nodesxsample_new
    
def main_qualitative(args):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    args_gen = get_args_gen(args.generators_path)
    args_gen.device = args.device
    dataloaders = get_dataloader(args_gen)
    # This is to account for loading in models that were not trained with xanes conditioning
    if "xanes" not in args_gen.conditioning:
        args_gen.conditioning.append("xanes")
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, "xanes_test") # args_gen.dataset) NOTE: This is due to xanes dataset using geom dataset name. Fix later.
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path,
                                                               dataloaders, args.device, args_gen,
                                                               property_norms)
    values, nodesxsample_gt = get_exp_values()
    # if args_gen.context_node_nf > 0: # conditional
    #     nodesxsample = get_conditional_num_nodes(args)
    # else: # unconditional
    #     nodesxsample = get_unconditional_num_nodes(nodes_dist, nodesxsample_gt)
    nodesxsample = nodesxsample_gt
    # values, nodesxsample = modify_for_num_nodes(values, nodesxsample, dataset_info)
    
    for i in range(args.n_sweeps):
        print("Sampling sweep %d/%d" % (i+1, args.n_sweeps))
        # if args_gen.context_node_nf == 0: # unconditional
        #     nodesxsample = get_unconditional_num_nodes(nodes_dist, nodesxsample_gt)
        save_and_sample_conditional(args_gen, device, model, prop_dist, dataset_info, values, property_norms, nodesxsample, epoch=i, id_from=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_xanes')
    parser.add_argument('--generators_path', type=str, default='outputs/xanes_cond')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--property', type=str, default='xames',
                        help="'xanes', 'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=20,
                        help='break point or not')
    parser.add_argument('--task', type=str, default='qualitative',
                        help='naive, edm, qm9_second_half, qualitative')
    parser.add_argument('--n_sweeps', type=int, default=10,
                        help='number of sweeps for the qualitative conditional experiment')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    if args.task == 'qualitative':
        main_qualitative(args)
    else:
        main_quantitative(args)
