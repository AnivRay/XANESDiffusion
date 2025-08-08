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
        self.pdf_data = None  # shape: (num_samples, 600, 2)

    def fit(self, pdf_vectors, labels=None):
        self.pdf_data = np.array(pdf_vectors)

    def query(self, pdf_queries):
        """
        Find the k nearest neighbors of each query in X_query among X_train, 
        using the Rw distance:
        
            Rw(expected, observed, r) =
            sum(r * (observed - expected)**2) / sum(r * expected**2)
        
        Parameters
        ----------
        self.pdf_data : np.ndarray, shape (n_train, 600, 2)
            Training samples.  [:, :, 0] is the common r-range; [:, :, 1] are measurements.
        pdf_queries : np.ndarray, shape (n_query, 600, 2)
            Query samples, same format.
        k : int
            Number of neighbors to return.
        
        Returns
        -------
        neigh_inds : np.ndarray, shape (n_query, k)
            Indices into X_train of the k nearest neighbors for each query.
        neigh_dists : np.ndarray, shape (n_query, k)
            The corresponding Rw distances.
        """
        # Extract r-vector (common for all samples) and measurements
        # We assume r is identical in every sample, so just grab from the first train example:
        r = self.pdf_data[0, :, 0]                     # shape (600,)
        T = self.pdf_data[..., 1]                      # shape (n_train, 600)
        Q = pdf_queries[..., 1]                      # shape (n_query, 600)
        
        # Precompute denominator per training sample: sum(r * expected^2)
        denom = np.sum(r * T**2, axis=1)         # shape (n_train,)
        
        # Compute squared‐difference weighted by r via broadcasting:
        #   diff[i,j,l] = Q[i,l] - T[j,l]
        # then numerator[i,j] = sum_l r[l] * diff[i,j,l]^2
        diff = Q[:, None, :] - T[None, :, :]     # shape (n_query, n_train, 600)
        numer = np.sum(r[None, None, :] * diff**2, axis=2)  # shape (n_query, n_train)
        
        # Full distance matrix
        D = numer / denom[None, :]               # shape (n_query, n_train)
        
        # Find k smallest per query:
        # 1) argpartition for speed, then 2) sort those k to order them
        idx_part = np.argpartition(D, self.k, axis=1)[:, :self.k]  # (n_query, k), unordered
        rows = np.arange(D.shape[0])[:, None]
        # distances of those k
        d_part = D[rows, idx_part]                      # (n_query, k)
        order = np.argsort(d_part, axis=1)              # order within each row
        neigh_inds  = idx_part[rows, order]              # (n_query, k)
        neigh_dists = D[rows, neigh_inds]                # (n_query, k)
        
        return neigh_inds, neigh_dists

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


def get_dataloader(args_gen):
    dataloaders, charge_scale = retrieve_dataloaders(args_gen)
    return dataloaders


def get_generator(args_gen):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    model = kNN(k=10)
    return model, dataset_info


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


def sample_sweep_conditional(generative_model, property_values):
    predIndices, predDists = generative_model.query(property_values)
    return predIndices, predDists


def save_and_sample_conditional(model, property_values, id_from=0):
    allNeighbors, allDists = sample_sweep_conditional(model, property_values)
    print(allNeighbors.shape, allDists.shape)

    # Save the neighbors and distances
    makedirs(join('outputs', 'analysis_pdf'), exist_ok=True)
    np.save(join('outputs', 'analysis_pdf', 'neighbors.npy'), allNeighbors)
    np.save(join('outputs', 'analysis_pdf', 'distances.npy'), allDists)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    rootDir = '/home/user_aniv/XANES-evaluation/pdf_analysis_gt'

    args_gen = get_args_gen(args.generators_path)
    args_gen.device = args.device
    dataloaders = get_dataloader(args_gen)
    model, dataset_info = get_generator(args_gen)
    # trainVals = next(iter(dataloaders["train"]))
    # vis.save_xyz_file('outputs/%s/analysis_knn/train/' % (args_gen.exp_name), trainVals['one_hot'].float(), trainVals['charges'], trainVals['positions'], dataset_info, name='gt', node_mask=trainVals['atom_mask'])
    # testVals = next(iter(dataloaders["test"]))
    # vis.save_xyz_file('outputs/%s/analysis_knn/test/' % (args_gen.exp_name), testVals['one_hot'].float(), testVals['charges'], testVals['positions'], dataset_info, name='gt', node_mask=testVals['atom_mask'])

    trainPDFs = np.load(join(rootDir, 'train.npy'))  # shape: (num_train_samples, 2, 600)
    testPDFs = np.load(join(rootDir, 'test.npy'))  # shape: (num_test_samples, 2, 600)
    
    model.fit(trainPDFs)
    
    save_and_sample_conditional(model, testPDFs, id_from=0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generators_path', type=str, default='outputs/fe_xanes_cond_final')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    main()
