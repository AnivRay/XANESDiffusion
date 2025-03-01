# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import build_geom_dataset
from configs.datasets_config import geom_with_h
import utils
import argparse
from os.path import join
from qm9.models import get_optim

import torch
from torch import nn
import torch.nn.functional as F
import time
import pickle

from qm9.utils import prepare_context, compute_mean_mad

parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')

parser.add_argument('--n_epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')

# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='geom',
                    help='dataset name')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=50)
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
parser.add_argument('--save_model', type=eval, default=True, help='save model')
parser.add_argument('--generate_epochs', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=0,
                    help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=1)
parser.add_argument('--data_augmentation', type=eval, default=False,
                    help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='multiple arguments can be passed, '
                         'including: homo | onehot | lumo | num_atoms | etc. '
                         'usage: "--conditioning H_thermo homo onehot H_thermo"')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=20,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=False, help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=5000)
parser.add_argument('--normalization_factor', type=float,
                    default=100, help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean" aggregation for the graph network')
parser.add_argument('--filter_molecule_size', type=int, default=None,
                    help="Only use molecules below this size.")
parser.add_argument('--sequential', action='store_true',
                    help='Organize data by size to reduce average memory usage.')
args = parser.parse_args()

data_file = './data/geom/Ti_w_XANES_large.npy' # './data/geom/geom_drugs_30.npy'

if args.remove_h:
    raise NotImplementedError()
else:
    dataset_info = geom_with_h

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

split_data = build_geom_dataset.load_split_data(data_file, val_proportion=0.1, test_proportion=0.1, filter_size=args.filter_molecule_size)
transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, device, args.sequential)
dataloaders = {}
for key, data_list in zip(['train', 'val', 'test'], split_data):
    dataset = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
    shuffle = (key == 'train') and not args.sequential

    # Sequential dataloading disabled for now.
    dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
        sequential=args.sequential, dataset=dataset, batch_size=args.batch_size,
        shuffle=shuffle)
del split_data

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name
    args.start_epoch = start_epoch

utils.create_folders(args)

data_dummy = next(iter(dataloaders['train']))

if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, "xanes") # Update this
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf

class LinearRegressor(nn.Module):
    def __init__(self):
        super(LinearRegressor, self).__init__()
        self.layer = nn.Linear(200, 3*(dataset_info["max_n_nodes"]-1))

    def forward(self, x):
        return self.layer(x)

model = LinearRegressor()

model = model.to(device)
numParams = sum(p.numel() for p in model.parameters())
numTrainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Model has\n", numParams, "Parameters\n", numTrainableParams, "Trainable Parameters")
optim = get_optim(args, model)

gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.



def train_epoch(model, loader, epoch, optim, loss_fn):
    model.train()
    loss_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        neighbor_x = x[:, 1:, :]
        seq_len = x.shape[1]
        pad_amount = dataset_info["max_n_nodes"] - seq_len
        neighbor_x_padded = F.pad(neighbor_x, pad=(0, 0, 0, pad_amount))
        # print(x.size(), x_padded.size())

        if len(args.conditioning) > 0:
            xanes = prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            xanes = xanes[:, 0, :]
        else:
            print("No XANES!")
            exit(1)

        optim.zero_grad()

        pred = model(xanes)
        # print(xanes.size())
        # print(pred.size())
        # print(neighbor_x_padded.size())

        loss = loss_fn(pred, neighbor_x_padded.flatten(start_dim=1))
        loss.backward()

        optim.step()


        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss.item():.2f}")
        loss_epoch.append(loss.item())
        

def test(model, loader, loss_fn):
    model.eval()
    loss_epoch = []
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        neighbor_x = x[:, 1:, :]
        seq_len = x.shape[1]
        pad_amount = dataset_info["max_n_nodes"] - seq_len
        neighbor_x_padded = F.pad(neighbor_x, pad=(0, 0, 0, pad_amount))

        if len(args.conditioning) > 0:
            xanes = prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            xanes = xanes[:, 0, :]
        else:
            print("No XANES!")
            exit(1)

        pred = model(xanes)

        loss = loss_fn(pred, neighbor_x_padded.flatten(start_dim=1))
        loss_epoch.append(loss.item())
    avgLoss = sum(loss_epoch) / len(loss_epoch)
    return avgLoss

def main():
    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'flow.npy'))
        dequantizer_state_dict = torch.load(join(args.resume, 'dequantizer.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1 and args.cuda:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    loss_fn = nn.MSELoss()

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(model, dataloaders['train'], epoch, optim, loss_fn)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            nll_val = test(model, dataloaders['val'], loss_fn)
            nll_test = test(model, dataloaders['test'], loss_fn)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))


if __name__ == "__main__":
    main()
