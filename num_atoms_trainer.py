from configs.datasets_config import geom_with_h
import argparse
import wandb
import utils
from os.path import join, exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

parser = argparse.ArgumentParser(description='e3_diffusion')
parser.add_argument('--exp_name', type=str, default='num_nodes_classifier')

parser.add_argument('--n_epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--filter_molecule_size', type=int, default=None,
                    help="Only use molecules below this size.")

parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')

args = parser.parse_args()

data_file = './data/geom/Ti_w_XANES_large.npy'

def load_split_data(conformation_file, val_proportion=0.1, test_proportion=0.1,
                    filter_size=None, permfile=None):
    from pathlib import Path
    path = Path(conformation_file)
    base_path = path.parent.absolute()

    # base_path = os.path.dirname(conformation_file)
    all_data = np.load(conformation_file)  # 2d array: num_atoms x 5

    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]
    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)

    # Filter based on molecule size.
    if filter_size is not None:
        # Keep only molecules <= filter_size
        data_list = [molecule for molecule in data_list
                     if molecule.shape[0] <= filter_size]

        assert len(data_list) > 0, 'No molecules left after filter.'

    # CAREFUL! Only for first time run:
    if permfile is None:
        permfile = 'Ti_w_XANES_large_permutation.npy'
        perm = np.random.permutation(len(data_list)).astype('int32')
        print('Warning, currently taking a random permutation for '
            'train/val/test partitions, this needs to be fixed for'
            'reproducibility.')
        assert not exists(join(base_path, permfile))
        np.save(join(base_path, permfile), perm)
        del perm

    perm = np.load(join(base_path, permfile))
    data_list = [data_list[i] for i in perm]

    num_mol = len(data_list)
    val_index = int(num_mol * val_proportion)
    test_index = val_index + int(num_mol * test_proportion)
    val_data, test_data, train_data = data_list[:val_index], data_list[val_index:test_index], data_list[test_index:] # np.split(data_list, [val_index, test_index])
    return train_data, val_data, test_data

class XASDataset(Dataset):
    def __init__(self, rawdata, n_nodes, input_dim=200):
        """
        Args:
            dataset_config (dict): A dictionary where keys are the number of atoms (labels)
                                   and values are the count of molecules with that number of atoms.
            input_dim (int): Dimension of the XAS vector.
        """
        self.dataset_config = n_nodes
        self.input_dim = input_dim

        # Generate synthetic data based on the dataset_config
        self.data = []
        self.labels = []
        self.classes = sorted(n_nodes.keys())
        
        for datum in rawdata:
            num_atoms = datum.shape[0]
            xas_vector = datum[0,-self.input_dim:].astype(np.float32)
            self.data.append(xas_vector)
            self.labels.append(self.classes.index(num_atoms))
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

class XASClassifier(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=512, num_layers=8, skip_connection_interval=4, num_classes=9):
        super(XASClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.skip_connection_interval = skip_connection_interval
        self.num_classes = num_classes

        # Initial input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim if not (i + 1) % skip_connection_interval == 0 else hidden_dim + input_dim, hidden_dim) for i in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
        # Normalization layers (useful for stability)
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x_input = x
        x = F.relu(self.input_layer(x))
        
        for i in range(self.num_layers):
            # Apply skip connection if needed
            if (i + 1) % self.skip_connection_interval == 0:
                x = torch.cat([x, x_input], dim=-1)
            
            # Apply hidden layer
            x = self.hidden_layers[i](x)
            x = self.norm_layers[i](x)
            x = F.relu(x)
        
        # Output layer with softmax activation for classification
        x = self.output_layer(x)
        
        return x

def evaluate_per_class_accuracy(model, dataloader, device, num_classes):
    """
    Evaluates the model on the given dataloader and
    returns overall accuracy and a list of per-class accuracies.
    """
    model.eval()
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(x_batch)                    # shape: [batch_size, num_classes]
            _, predicted = torch.max(outputs, dim=1)    # shape: [batch_size]
            
            # Update counts for overall accuracy
            for label, pred in zip(y_batch, predicted):
                class_total[label.item()] += 1
                if label.item() == pred.item():
                    class_correct[label.item()] += 1

    # Compute overall accuracy
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Compute per-class accuracy
    per_class_accuracy = []
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
        else:
            acc = 0.0  # No samples for this class
        per_class_accuracy.append(acc)

    return overall_accuracy, per_class_accuracy

def main():

    wandb.init(
        project="e3_diffusion_geom",       # Change to your project name
        name=args.exp_name,          # Give your run a custom name
        config=args,
        mode='disabled' if args.no_wandb else 'online'
    )
    utils.create_folders(args)

    dataset_info = geom_with_h

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32

    train_data, val_data, test_data = load_split_data(data_file, val_proportion=0.1, test_proportion=0.1, filter_size=args.filter_molecule_size, permfile="Ti_w_XANES_large_permutation.npy")
    print("Made split data", "\nTrain:", len(train_data), "\nVal:", len(val_data), "\nTest:", len(test_data))

    n_nodes = dataset_info['n_nodes']
    trainDataset = XASDataset(train_data, n_nodes)
    valDataset = XASDataset(val_data, n_nodes)
    testDataset = XASDataset(test_data, n_nodes)

    # Build DataLoaders
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valDataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False)

    # ========== Model, Loss, Optimizer ==========
    num_classes = len(n_nodes.keys())  # each unique number of atoms is a class
    model = XASClassifier(
        input_dim=200,
        hidden_dim=512,
        num_layers=8,
        skip_connection_interval=4,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()           # for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ========== Training Loop ==========
    best_val_accuracy = 0.0
    for epoch in range(args.n_epochs):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, dtype=dtype)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)             # logits of shape (batch_size, num_classes)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            # Calculate training accuracy
            _, predicted = torch.max(output, dim=1)
            correct_preds += (predicted == y_batch).sum().item()
            total_samples += x_batch.size(0)

        train_loss = running_loss / total_samples
        train_accuracy = correct_preds / total_samples

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, dtype=dtype)
                y_val = y_val.to(device)

                output_val = model(x_val)
                loss_val = criterion(output_val, y_val)

                val_loss += loss_val.item() * x_val.size(0)
                _, predicted_val = torch.max(output_val, dim=1)
                val_correct += (predicted_val == y_val).sum().item()
                val_total += x_val.size(0)

        val_loss = val_loss / val_total
        val_accuracy = val_correct / val_total

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # Keep track of the best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Optionally, save the best model
            torch.save(model.state_dict(), "outputs/{}/best_model.pt".format(args.exp_name))

        print(f"[Epoch {epoch+1}/{args.n_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # ========== Test Evaluation ==========
    model.load_state_dict(torch.load("outputs/{}/best_model.pt".format(args.exp_name), map_location=device))
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device, dtype=dtype)
            y_test = y_test.to(device)

            output_test = model(x_test)
            loss_test = criterion(output_test, y_test)

            test_loss += loss_test.item() * x_test.size(0)
            _, predicted_test = torch.max(output_test, dim=1)
            test_correct += (predicted_test == y_test).sum().item()
            test_total += x_test.size(0)

    test_loss = test_loss / test_total
    test_accuracy = test_correct / test_total

    _, per_class_accuracy = evaluate_per_class_accuracy(model, test_loader, device, num_classes)
    for i, class_accuracy in enumerate(per_class_accuracy):
        print("Accuracy for Molecules with ", testDataset.classes[i], "atoms =", class_accuracy)

    # Log final test results to W&B
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

    print(f"\nFinal Test Loss: {test_loss:.4f}, Final Test Acc: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
