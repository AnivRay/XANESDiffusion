import argparse
from os.path import join
from os import makedirs
import torch
import numpy as np
import pickle
from configs.datasets_config import get_dataset_info
import qm9.visualizer as vis

class kNN:
    def __init__(self, k=1):
        self.k = k
        self.xas_data = None  # shape: (num_samples, 200)
        self.labels = None    # could be any metadata, e.g., molecule IDs

    def fit(self, xas_vectors, labels=None):
        self.xas_data = np.array(xas_vectors)
        self.labels = np.array(labels) if labels is not None else None

    def query(self, xas_queries):
        # xas_queries: (BS, 200)
        diffs = self.xas_data[None, :, :] - xas_queries[:, None, :]  # (BS, N, 200)
        dists = np.linalg.norm(diffs, axis=2)                        # (BS, N)
        nn_indices = np.argsort(dists, axis=1)[:, :self.k]           # (BS, k)
        neighbor_labels = (self.labels[nn_indices] if self.labels is not None else None)
        if neighbor_labels:
            return nn_indices, neighbor_labels
        else:
            return nn_indices

def retrieve_dataloaders(cfg):
    if 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/XANES_dataset.npy' # Fix this later: changing to xanes file for now
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            # We want the dataloader to be sequential for the kNN baseline and return the whole dataset in a single batch
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=len(dataset),
                shuffle=False)
            
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale

def get_and_save_values_from_dataloader(args, dataloader, dataset_info, save=True):
    """
    Extracts values from the dataloader and saves ground truth data to files.
    Args:
        args: Arguments containing experiment name and other configurations.
        dataloader: Dataloader from which to extract values.
        dataset_info: Information about the dataset.
        save_gt: Whether to save ground truth data to files.
    Returns:
        values: A dictionary containing concatenated values from the dataloader.
    """
    values = {}
    id_from = 0
    for data in dataloader:
        for value_key in data:
            if value_key not in values:
                values[value_key] = []
            values[value_key].append(data[value_key])
        # Save batch of ground truth data starting from where previous batch left off
        if save:
            vis.save_xyz_file('outputs/%s/analysis/groundtruth/' % (args.exp_name), data['one_hot'].float(), data['charges'], data['positions'], dataset_info, id_from, name='gt', node_mask=data['atom_mask'])
        id_from += len(data['positions'])
    print([key for key in values])
    for key in values:
        values[key] = torch.cat(values[key])
    # print("Number of samples in the dataset: ", values['positions'].shape)
    exit(0)
    return values


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


def get_generator(device, args_gen):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model = kNN(k=10)
    return model, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = retrieve_dataloaders(args_gen)
    return dataloaders


def sample_sweep_conditional(args, device, generative_model, dataset_info, property_values, trainVals):
    predIndices = generative_model.query(property_values)
    return predIndices


def save_and_sample_conditional(args, device, model, dataset_info, property_values, trainVals, id_from=0):
    allNeighbors = sample_sweep_conditional(args, device, model, dataset_info, property_values, trainVals)

    for i in range(allNeighbors.shape[1]):
        predIndices = allNeighbors[:, i]
        one_hot, charges, x, node_mask = trainVals["one_hot"][predIndices], \
            trainVals["charges"][predIndices], \
            trainVals["positions"][predIndices], \
            trainVals["atom_mask"][predIndices]
        vis.save_xyz_file(
            'outputs/%s/analysis_knn/run%s/' % (args.exp_name, i), one_hot.float(), charges, x, dataset_info,
            id_from, name='conditional', node_mask=node_mask)
    

def main(args):
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
    model, dataset_info = get_generator(args.device, args_gen)
    # trainVals = get_and_save_values_from_dataloader(args_gen, dataloaders["train"], dataset_info, save=False)
    trainVals = next(iter(dataloaders["train"]))
    trainXANES = trainVals["xanes"][:, 0, :].numpy()
    testVals = next(iter(dataloaders["test"]))
    vis.save_xyz_file('outputs/%s/analysis_knn/groundtruth/' % (args_gen.exp_name), testVals['one_hot'].float(), testVals['charges'], testVals['positions'], dataset_info, name='gt', node_mask=testVals['atom_mask'])
    testXANES = testVals["xanes"][:, 0, :].numpy()
    model.fit(trainXANES)
    
    save_and_sample_conditional(args_gen, device, model, dataset_info, testXANES, trainVals, id_from=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_xanes')
    parser.add_argument('--generators_path', type=str, default='outputs/fe_xanes_cond_final')
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
