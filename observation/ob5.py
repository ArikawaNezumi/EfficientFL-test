import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import matplotlib.pyplot as plt
import logging
import datetime
import os
import random

# --- Configuration ---
ARGS = {
    'dataset': 'cifar10',  # Options: 'mnist', 'cifar10'
    'num_client': 40,
    'num_round': 50,  # Increase rounds for CIFAR-10 as it's harder
    'clients_per_round': 5,
    'local_epoch': 3,
    'batch_size': 64,
    'lr': 0.01,
    'q_level': 4,  # Quantization level (bits)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset_root': './data',
    'log_dir': './logs',
    'seed': 42
}

# --- Setup Logging ---
if not os.path.exists(ARGS['log_dir']):
    os.makedirs(ARGS['log_dir'])
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(ARGS['log_dir'], f'fl_{ARGS["dataset"]}_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --- Model Definitions ---
class SimpleCNN_MNIST(nn.Module):
    def __init__(self):
        super(SimpleCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN_CIFAR10(nn.Module):
    def __init__(self):
        super(SimpleCNN_CIFAR10, self).__init__()
        # Input: 3x32x32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # After 2 pools: 32 -> 16 -> 8. Feature map size: 64 * 8 * 8
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(dataset_name):
    if dataset_name == 'mnist':
        return SimpleCNN_MNIST()
    elif dataset_name == 'cifar10':
        return SimpleCNN_CIFAR10()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# --- Data Helper ---
def get_data(dataset_name, num_clients):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root=ARGS['dataset_root'], train=True, download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=ARGS['dataset_root'], train=False, download=True,
                                                  transform=transform)

    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=ARGS['dataset_root'], train=True, download=True,
                                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=ARGS['dataset_root'], train=False, download=True,
                                                    transform=transform_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Random IID split
    data_len = len(train_dataset)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)

    client_loaders = []
    for idxs in client_indices:
        client_loaders.append(DataLoader(Subset(train_dataset, idxs), batch_size=ARGS['batch_size'], shuffle=True))
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return client_loaders, test_loader


# --- Quantization Utils ---
def quantize_tensor(tensor, bits):
    if bits >= 32: return tensor
    min_val, max_val = tensor.min(), tensor.max()
    if min_val == max_val: return tensor
    levels = 2 ** bits - 1
    scale = (max_val - min_val) / levels
    zero_point = -min_val / scale
    q_tensor = torch.round(tensor / scale + zero_point)
    q_tensor = torch.clamp(q_tensor, 0, levels)
    return (q_tensor - zero_point) * scale


class Compressor:
    def __init__(self, model_params, bits, use_ef=True):
        self.bits = bits
        self.use_ef = use_ef
        self.error_buffer = {name: torch.zeros_like(param) for name, param in model_params.items()}

    def compress(self, new_gradients):
        compressed_gradients = {}
        for name, grad in new_gradients.items():
            if grad is None: continue

            if self.use_ef:
                corrected_grad = grad + self.error_buffer[name]
            else:
                corrected_grad = grad

            compressed_grad = quantize_tensor(corrected_grad, self.bits)

            if self.use_ef:
                self.error_buffer[name] = corrected_grad - compressed_grad
            else:
                self.error_buffer[name].zero_()

            compressed_gradients[name] = compressed_grad
        return compressed_gradients


# --- Local Training ---
def train_client(model, train_loader, device, lr, local_epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    epoch_loss, correct, total = 0, 0, 0
    for _ in range(local_epoch):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = epoch_loss / (len(train_loader) * local_epoch)
    accuracy = 100 * correct / total if total > 0 else 0

    final_weights = model.state_dict()
    updates = {name: final_weights[name] - initial_weights[name] for name in initial_weights}
    return updates, avg_loss, accuracy


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return test_loss / len(test_loader), 100 * correct / total


# --- Experiment Runner ---
def run_experiment(exp_name, use_ef, client_loaders, test_loader, device):
    logging.info(f"STARTING EXPERIMENT: {exp_name} (EF={use_ef}, Bits={ARGS['q_level']}, Dataset={ARGS['dataset']})")
    set_seed(ARGS['seed'])

    global_model = get_model(ARGS['dataset']).to(device)

    client_compressors = [
        Compressor(global_model.state_dict(), ARGS['q_level'], use_ef=use_ef)
        for _ in range(ARGS['num_client'])
    ]

    history = {'server_acc': [], 'server_loss': []}

    for round_idx in range(ARGS['num_round']):
        selected_clients = np.random.choice(range(ARGS['num_client']), ARGS['clients_per_round'], replace=False)
        round_updates = []

        for client_idx in selected_clients:
            local_model = copy.deepcopy(global_model)
            updates, loss, acc = train_client(local_model, client_loaders[client_idx], device, ARGS['lr'],
                                              ARGS['local_epoch'])

            compressed_updates = client_compressors[client_idx].compress(updates)
            round_updates.append(compressed_updates)

        global_dict = global_model.state_dict()
        for key in global_dict:
            key_updates = torch.stack([upd[key] for upd in round_updates])
            global_dict[key] += torch.mean(key_updates, dim=0)
        global_model.load_state_dict(global_dict)

        s_loss, s_acc = evaluate(global_model, test_loader, device)
        history['server_loss'].append(s_loss)
        history['server_acc'].append(s_acc)

        if (round_idx + 1) % 1 == 0:
            logging.info(f"[{exp_name}] Round {round_idx + 1}: Acc={s_acc:.2f}%, Loss={s_loss:.4f}")

    return history


# --- Main ---
def main():
    device = torch.device(ARGS['device'])
    logging.info(f"Using device: {device}, Dataset: {ARGS['dataset']}")

    set_seed(ARGS['seed'])
    client_loaders, test_loader = get_data(ARGS['dataset'], ARGS['num_client'])

    hist_ef = run_experiment("With EF", True, client_loaders, test_loader, device)
    hist_no_ef = run_experiment("No EF", False, client_loaders, test_loader, device)

    # --- Plotting ---
    epochs = range(1, ARGS['num_round'] + 1)

    plt.figure(figsize=(14, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_ef['server_acc'], label='With Error Feedback', marker='o', markersize=4)
    plt.plot(epochs, hist_no_ef['server_acc'], label='No Error Feedback', marker='x', markersize=4, linestyle='--')
    plt.title(f'Server Accuracy ({ARGS["dataset"].upper()}, Q={ARGS["q_level"]} bits)')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist_ef['server_loss'], label='With Error Feedback', marker='o', markersize=4)
    plt.plot(epochs, hist_no_ef['server_loss'], label='No Error Feedback', marker='x', markersize=4, linestyle='--')
    plt.title(f'Server Loss ({ARGS["dataset"].upper()}, Q={ARGS["q_level"]} bits)')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(ARGS['log_dir'], f'comparison_{ARGS["dataset"]}_q{ARGS["q_level"]}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Comparison plots saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    main()