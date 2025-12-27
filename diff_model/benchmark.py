import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random
import copy
import time
import logging
import os
from quant_utils import sym_quantize, get_model_size_bits
from dbs import BitWidthSearcher

# ==========================================
# 0. 日志配置
# ==========================================
if not os.path.exists('./logs'):
    os.makedirs('./logs')

timestamp = time.strftime("%Y%m%d_%H%M")
log_filename = f'./logs/benchmark_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


def log(msg):
    logger.info(msg)


# ==========================================
# 全局配置
# ==========================================
NUM_CLIENTS = 20
CLIENTS_PER_ROUND = 4
NUM_ROUNDS = 40
LOCAL_EPOCHS = 3
SEARCH_EPOCHS = 5
BATCH_SIZE = 32
LR = 0.01
DATA_DISTRIBUTION = "iid"
DIRICHLET_ALPHA = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 基础设施
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DataPartitioner:
    def __init__(self, num_clients, alpha=0.5):
        self.num_clients = num_clients
        log(f"Loading CIFAR-10... (Alpha={alpha})")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        subset_indices = list(range(10000))
        subset = torch.utils.data.Subset(train_dataset, subset_indices)

        loader = DataLoader(subset, batch_size=len(subset))
        self.X, self.Y = next(iter(loader))
        self.X, self.Y = self.X.to(DEVICE), self.Y.to(DEVICE)
        self.num_samples = len(self.Y)
        self.alpha = alpha

    def get_partition(self):
        client_data = []
        min_size = 0
        while min_size < 10:
            idx_batch = [[] for _ in range(self.num_clients)]
            for k in range(10):
                idx_k = np.where(self.Y.cpu() == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
                proportions = np.array([p * (len(idx_j) < self.num_samples / self.num_clients) for p, idx_j in
                                        zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for i in range(self.num_clients):
            indices = torch.tensor(idx_batch[i]).to(DEVICE)
            client_data.append((self.X[indices], self.Y[indices]))
        return client_data


class Client:
    def __init__(self, client_id, train_data, train_label):
        self.id = client_id
        self.model = SimpleCNN().to(DEVICE)
        dataset = TensorDataset(train_data, train_label)
        self.train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.searcher = BitWidthSearcher(self.model, candidate_bits=[2, 4, 8])

    def run(self, global_weights, experiment_mode, bw_target_bits=None):
        self.model.load_state_dict(global_weights)
        optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # 1. 本地训练
        self.model.train()
        for epoch in range(LOCAL_EPOCHS):
            for data, target in self.train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 2. 上传策略
        compressed_w = {}
        total_bits = 0

        state_dict = self.model.state_dict()

        if experiment_mode == "FedAvg (FP32)":
            for name, param in state_dict.items():
                compressed_w[name] = param.cpu()
                total_bits += param.numel() * 32

        elif experiment_mode == "Fixed 4-bit":
            for name, param in state_dict.items():
                if 'weight' in name and ('conv' in name or 'fc' in name):
                    compressed_w[name] = sym_quantize(param, 4).cpu()
                    total_bits += param.numel() * 4
                else:
                    compressed_w[name] = param.cpu()
                    total_bits += param.numel() * 32

        elif experiment_mode == "Ours (DBS)":
            bit_config = self.searcher.search(
                self.train_loader, bw_target_bits,
                lambda_coeff=5.0, iterations=SEARCH_EPOCHS
            )
            for name, param in state_dict.items():
                layer_name = name.split('.')[0]
                if layer_name in bit_config and 'weight' in name:
                    bit = bit_config[layer_name]
                    compressed_w[name] = sym_quantize(param, bit).cpu()
                    total_bits += param.numel() * bit
                else:
                    compressed_w[name] = param.cpu()
                    total_bits += param.numel() * 32

        return compressed_w, total_bits


class Server:
    def __init__(self):
        self.global_model = SimpleCNN().to(DEVICE)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    def aggregate(self, updates):
        avg_weights = copy.deepcopy(updates[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
        n = len(updates)
        for w in updates:
            for key in avg_weights.keys():
                avg_weights[key] += w[key]
        for key in avg_weights.keys():
            avg_weights[key] = avg_weights[key] / n
        self.global_model.load_state_dict(avg_weights)

    def evaluate(self):
        self.global_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        return 100. * correct / len(self.test_loader.dataset)


# ==========================================
# 2. 实验控制器
# ==========================================
def run_experiment(mode_name, clients, full_size_bits):
    log(f"\n>>> Starting Experiment: {mode_name}")
    server = Server()
    acc_history = []
    total_comm_bits = 0

    dbs_target = full_size_bits / (32 / 4.5)

    for round_idx in range(1, NUM_ROUNDS + 1):
        random.seed(round_idx)
        selected_indices = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)

        client_updates = []
        round_bits = 0

        for i in selected_indices:
            w_up, bits = clients[i].run(server.global_model.state_dict(), mode_name, dbs_target)
            client_updates.append(w_up)
            round_bits += bits

        server.aggregate(client_updates)
        acc = server.evaluate()
        acc_history.append(acc)
        total_comm_bits += round_bits

        log(f"  Round {round_idx}: Acc={acc:.2f}%, Round Comm={round_bits / 8 / 1024:.1f} KB")

    return acc_history[-1], total_comm_bits


# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    log(f"--- Benchmark Experiment (Log: {log_filename}) ---")

    partitioner = DataPartitioner(NUM_CLIENTS, alpha=DIRICHLET_ALPHA)
    client_datasets = partitioner.get_partition()
    clients = [Client(i, *client_datasets[i]) for i in range(NUM_CLIENTS)]

    temp_server = Server()
    full_size_bits = get_model_size_bits(temp_server.global_model)
    log(f"Model Full Size: {full_size_bits / 8 / 1024:.2f} KB\n")

    modes = ["FedAvg (FP32)", "Fixed 4-bit", "Ours (DBS)"]
    results = {}

    for mode in modes:
        start_time = time.time()
        final_acc, total_comm = run_experiment(mode, clients, full_size_bits)
        duration = time.time() - start_time
        results[mode] = {
            "acc": final_acc,
            "comm_mb": total_comm / 8 / 1024 / 1024,
            "time": duration
        }

    log("\n" + "=" * 60)
    log(f"{'Method':<15} | {'Final Acc':<10} | {'Total Comm (MB)':<15} | {'Time (s)':<10}")
    log("-" * 60)
    for mode, res in results.items():
        log(f"{mode:<15} | {res['acc']:.2f}%      | {res['comm_mb']:.2f}            | {res['time']:.0f}")
    log("=" * 60)