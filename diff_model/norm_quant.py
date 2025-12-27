import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random
import copy
import logging
import time
import os
from quant_utils import sym_quantize, get_model_size_bits

# ==========================================
# 0. 日志配置
# ==========================================
if not os.path.exists('./logs'):
    os.makedirs('./logs')

# 可以在这里修改是 4bit 还是 8bit
FIXED_BIT = 2

timestamp = time.strftime("%Y%m%d_%H%M")
log_filename = f'./logs/fl_fixed_{FIXED_BIT}bit_{timestamp}.log'

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
# 配置参数
# ==========================================
NUM_CLIENTS = 10
CLIENTS_PER_ROUND = 4
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LR = 0.01

DATA_DISTRIBUTION = "non-iid"  # 保持和 DBS 实验一致
DIRICHLET_ALPHA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 模型定义 (SimpleCNN)
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


# ==========================================
# 2. 数据分区工具 (保持一致)
# ==========================================
class DataPartitioner:
    def __init__(self, num_clients, alpha=0.5, root='./data'):
        self.num_clients = num_clients
        self.alpha = alpha

        log(f"Loading CIFAR-10... (Alpha={alpha})")
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

        train_dataset = datasets.CIFAR10(root, train=True, download=True)
        data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2).float() / 255.0
        self.X = (data - self.mean) / self.std
        self.Y = torch.tensor(train_dataset.targets)
        self.num_samples = len(self.Y)
        self.num_classes = 10

    def get_partition(self, dist_type="non-iid"):
        client_data = []
        log(f"Partitioning data ({dist_type})...")

        if dist_type == "iid":
            idxs = np.random.permutation(self.num_samples)
            batch_idxs = np.array_split(idxs, self.num_clients)
            for i in range(self.num_clients):
                client_data.append((self.X[batch_idxs[i]], self.Y[batch_idxs[i]]))
        elif dist_type == "non-iid":
            min_size = 0
            while min_size < 10:
                idx_batch = [[] for _ in range(self.num_clients)]
                for k in range(self.num_classes):
                    idx_k = np.where(self.Y == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
                    proportions = np.array([p * (len(idx_j) < self.num_samples / self.num_clients) for p, idx_j in
                                            zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for i in range(self.num_clients):
                np.random.shuffle(idx_batch[i])
                indices = torch.tensor(idx_batch[i])
                client_data.append((self.X[indices], self.Y[indices]))

        return client_data


# ==========================================
# 3. Client 定义 (关键修改：固定量化)
# ==========================================
class Client:
    def __init__(self, client_id, train_data, train_label):
        self.id = client_id
        self.model = SimpleCNN().to(DEVICE)
        dataset = TensorDataset(train_data, train_label)
        self.train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        # 这里不需要 BitWidthSearcher 了

    def train_and_upload(self, global_weights):
        self.model.load_state_dict(global_weights)
        optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # A. 本地训练 (完全一样)
        self.model.train()
        for epoch in range(LOCAL_EPOCHS):
            correct = 0
            total = 0
            epoch_loss = 0
            for data, target in self.train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        final_train_acc = 100. * correct / (total + 1e-6)
        final_train_loss = epoch_loss / len(self.train_loader)

        # B. 强制量化 (Fixed Quantization)
        compressed_state_dict = {}
        total_bits = 0
        state_dict = self.model.state_dict()

        for name, param in state_dict.items():
            # 只有 Conv 和 Linear 的权重才量化，Bias 保持 FP32
            if 'weight' in name and ('conv' in name or 'fc' in name):
                # 核心：直接用 FIXED_BIT
                compressed_state_dict[name] = sym_quantize(param, FIXED_BIT).cpu()
                total_bits += param.numel() * FIXED_BIT
            else:
                compressed_state_dict[name] = param.cpu()
                total_bits += param.numel() * 32

        return compressed_state_dict, total_bits, final_train_loss, final_train_acc


# ==========================================
# 4. Server 定义
# ==========================================
class Server:
    def __init__(self):
        self.global_model = SimpleCNN().to(DEVICE)

        log("Loading CIFAR-10 Test Set...")
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
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / len(self.test_loader.dataset)
        return test_loss, test_acc


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    # torch.manual_seed(42)

    log(f"--- Federated Learning (Fixed {FIXED_BIT}-bit) ---")
    log(f"Log File Saved to: {log_filename}")

    partitioner = DataPartitioner(NUM_CLIENTS, alpha=DIRICHLET_ALPHA)
    client_datasets = partitioner.get_partition(DATA_DISTRIBUTION)

    clients = []
    log("\n[Data Distribution]")
    for i in range(5):
        counts = np.bincount(client_datasets[i][1].numpy(), minlength=10)
        log(f"Client {i}: {counts}")

    server = Server()
    full_size_bits = get_model_size_bits(server.global_model)
    log(f"\nFull Model Size (FP32): {full_size_bits / 8 / 1024:.2f} KB\n")

    for round_idx in range(1, NUM_ROUNDS + 1):
        log(f"=== Round {round_idx}/{NUM_ROUNDS} ===")

        selected_indices = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        selected_clients = [Client(i, *client_datasets[i]) for i in selected_indices]

        client_updates = []

        for client in selected_clients:
            # Fixed 模式下，不需要 bandwidth_limit
            w_up, size_bits, loss, acc = client.train_and_upload(
                server.global_model.state_dict()
            )

            comp_rate = full_size_bits / size_bits

            # 打印日志
            log(f"  [C{client.id:02d}] Acc:{acc:5.1f}% | Fixed {FIXED_BIT}-bit | Rate: {comp_rate:.1f}x")

            client_updates.append(w_up)

        server.aggregate(client_updates)
        g_loss, g_acc = server.evaluate()
        log(f"  >> Global Result: Loss={g_loss:.4f}, Acc={g_acc:.2f}%\n")