import argparse
from os.path import join
from os import makedirs
import torch
import numpy as np
import pickle
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction import main_qm9_prop
import qm9.visualizer as vis
from linear_baseline_trainer import LinearRegressor

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
    model = LinearRegressor() # get_latent_diffusion(args_gen, device, dataset_info, dataloaders['train'])
    fn = 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    return model.to(device), dataset_info


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

def sample_sweep_conditional(args, device, generative_model, dataset_info, property_values, property_norms, nodesxsample, n_nodes=7, n_frames=100):
    context = []
    for key in property_values:
        values = property_values[key]
        mean, mad = property_norms[key]['mean'], property_norms[key]['mad']
        values = (values - mean) / mad
        context.append(values)
    context = torch.cat(context, dim=1).float().to(device)

    print(context.size())
    pred = generative_model(context)
    neighborPos = torch.reshape(pred, (-1, 12, 3))

    return neighborPos

def save_linear_xyz_file(rootPath, neighborPos, nodesxsample, name='conditional'):
    try:
        makedirs(rootPath)
    except OSError:
        pass
    for i in range(neighborPos.size(0)):
        numAtoms = nodesxsample[i].item()
        numOxygen = numAtoms - 1
        neighbors = neighborPos[i]
        with open(join(rootPath, "{}_{:03d}.txt".format(name, i)), 'w') as outFile:
            outFile.write(str(numAtoms) + "\n\n")
            outFile.write("Ti {:.9f} {:.9f} {:.9f}\n".format(0, 0, 0))
            for j in range(numOxygen):
                outFile.write("O {:.9f} {:.9f} {:.9f}\n".format(*neighbors[j].tolist()))


def save_and_sample_conditional(args, device, model, dataset_info, values, property_norms, nodesxsample, id_from=0):
    neighborPos = sample_sweep_conditional(args, device, model, dataset_info, values, property_norms, nodesxsample)

    save_linear_xyz_file('outputs/%s/analysis/run/' % (args.exp_name), neighborPos, nodesxsample, name='conditional')


def main(args):
    args_gen = get_args_gen(args.generators_path)
    args_gen.device = args.device
    dataloaders = get_dataloader(args_gen)
    # This is to account for loading in models that were not trained with xanes conditioning
    if "xanes" not in args_gen.conditioning:
        args_gen.conditioning.append("xanes")
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, "xanes_test") # args_gen.dataset) NOTE: This is due to xanes dataset using geom dataset name. Fix later.
    model, dataset_info = get_generator(args.generators_path, dataloaders, args.device, args_gen, property_norms)
    values, nodesxsample = get_and_save_values_from_dataloader(args_gen, dataloaders["test"], args_gen.conditioning, dataset_info)
    
    save_and_sample_conditional(args_gen, device, model, dataset_info, values, property_norms, nodesxsample, id_from=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_xanes')
    parser.add_argument('--generators_path', type=str, default='outputs/ti_xanes_large_linear')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--property', type=str, default='xanes',
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

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    main(args)
