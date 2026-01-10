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

# --- 全局配置参数 ---
ARGS = {
    'dataset': 'cifar10',  # 数据集: 'mnist' 或 'cifar10'
    'num_client': 20,  # 客户端总数
    'num_round': 30,  # 训练轮数 (CIFAR-10 建议多跑几轮看效果)
    'clients_per_round': 5,  # 每轮参与的客户端数
    'local_epoch': 2,  # 本地训练轮数
    'batch_size': 64,  # 批大小
    'lr': 0.01,  # 学习率
    'q_level': 2,  # 量化比特数 (2bit 最能体现 EF 的作用)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset_root': './data',
    'log_dir': './logs',
    'seed': 42
}

# --- 初始化日志 ---
if not os.path.exists(ARGS['log_dir']):
    os.makedirs(ARGS['log_dir'])
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(ARGS['log_dir'], f'fl_ef_{ARGS["dataset"]}_{ARGS["q_level"]}bit_{timestamp}.log')

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


# --- 模型定义 ---
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
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


# --- 数据准备 ---
def get_data(dataset_name, num_clients):
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.MNIST(root=ARGS['dataset_root'], train=True, download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=ARGS['dataset_root'], train=False, download=True,
                                                  transform=transform)
    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=ARGS['dataset_root'], train=True, download=True,
                                                     transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=ARGS['dataset_root'], train=False, download=True,
                                                    transform=transform_test)

    # IID 数据划分
    data_len = len(train_dataset)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)

    client_loaders = []
    for idxs in client_indices:
        client_loaders.append(DataLoader(Subset(train_dataset, idxs), batch_size=ARGS['batch_size'], shuffle=True))
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return client_loaders, test_loader


# --- 量化函数 ---
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


# --- 误差反馈压缩器 ---
class ErrorFeedbackCompressor:
    """
    实现论文 'Error Feedback Fixes SignSGD' 中的逻辑:
    p_t = g_t + e_{t-1}
    delta_t = Compress(p_t)
    e_t = p_t - delta_t
    """

    def __init__(self, model_params, bits):
        self.bits = bits
        # e_0 = 0 (初始化误差缓存)
        self.error_buffer = {name: torch.zeros_like(param) for name, param in model_params.items()}

    def compress(self, new_gradients):
        compressed_gradients = {}
        for name, grad in new_gradients.items():
            if grad is None: continue

            # 1. Error Correction: p_t = g_t + e_{t-1}
            # 将上一轮未发送的误差加到当前梯度上
            corrected_grad = grad + self.error_buffer[name]

            # 2. Compression: delta_t = Q(p_t)
            # 对修正后的梯度进行量化
            compressed_grad = quantize_tensor(corrected_grad, self.bits)

            # 3. Update Residual: e_t = p_t - delta_t
            # 计算本次量化丢失的信息，存入 buffer 供下一轮使用
            self.error_buffer[name] = corrected_grad - compressed_grad

            compressed_gradients[name] = compressed_grad
        return compressed_gradients


# --- 客户端训练 ---
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
    acc = 100 * correct / total if total > 0 else 0

    # 计算更新量 (g_t)
    final_weights = model.state_dict()
    updates = {name: final_weights[name] - initial_weights[name] for name in initial_weights}
    return updates, avg_loss, acc


# --- 评估 ---
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


# --- 主程序 ---
def main():
    set_seed(ARGS['seed'])
    device = torch.device(ARGS['device'])
    logging.info(f"Start FL with Error Feedback. Settings: {ARGS}")

    # 1. 初始化
    global_model = get_model(ARGS['dataset']).to(device)
    client_loaders, test_loader = get_data(ARGS['dataset'], ARGS['num_client'])

    # 每个客户端拥有独立的误差反馈压缩器
    client_compressors = [
        ErrorFeedbackCompressor(global_model.state_dict(), ARGS['q_level'])
        for _ in range(ARGS['num_client'])
    ]

    history = {'client_loss': [], 'client_acc': [], 'server_loss': [], 'server_acc': []}

    # 2. 训练循环
    for round_idx in range(ARGS['num_round']):
        selected_clients = np.random.choice(range(ARGS['num_client']), ARGS['clients_per_round'], replace=False)
        round_updates = []
        round_loss = []
        round_acc = []

        # --- 客户端阶段 ---
        for client_idx in selected_clients:
            local_model = copy.deepcopy(global_model)
            # 本地训练
            updates, loss, acc = train_client(local_model, client_loaders[client_idx], device, ARGS['lr'],
                                              ARGS['local_epoch'])

            # 压缩 + 误差反馈
            compressed_updates = client_compressors[client_idx].compress(updates)

            round_updates.append(compressed_updates)
            round_loss.append(loss)
            round_acc.append(acc)

        # 记录客户端平均指标
        avg_c_loss = sum(round_loss) / len(round_loss)
        avg_c_acc = sum(round_acc) / len(round_acc)
        history['client_loss'].append(avg_c_loss)
        history['client_acc'].append(avg_c_acc)

        # --- 服务器阶段 ---
        global_dict = global_model.state_dict()
        for key in global_dict:
            # 聚合压缩后的梯度
            key_updates = torch.stack([upd[key] for upd in round_updates])
            global_dict[key] += torch.mean(key_updates, dim=0)
        global_model.load_state_dict(global_dict)

        # 评估
        s_loss, s_acc = evaluate(global_model, test_loader, device)
        history['server_loss'].append(s_loss)
        history['server_acc'].append(s_acc)

        logging.info(f"Round {round_idx + 1}: Client Loss {avg_c_loss:.4f}, Server Acc {s_acc:.2f}%")

    # --- 绘图 ---
    epochs = range(1, ARGS['num_round'] + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['client_loss'], label='Avg Client Loss', linestyle='--')
    plt.plot(epochs, history['server_loss'], label='Server Loss')
    plt.title(f'Loss (EF Enabled, {ARGS["q_level"]} bits)')
    plt.xlabel('Round');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['client_acc'], label='Avg Client Acc', linestyle='--')
    plt.plot(epochs, history['server_acc'], label='Server Acc')
    plt.title(f'Accuracy (EF Enabled, {ARGS["q_level"]} bits)')
    plt.xlabel('Round');
    plt.ylabel('Accuracy (%)');
    plt.legend();
    plt.grid(True)

    save_path = os.path.join(ARGS['log_dir'], f'result_{ARGS["dataset"]}_ef_q{ARGS["q_level"]}.png')
    plt.savefig(save_path)
    logging.info(f"Done. Plot saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    main()