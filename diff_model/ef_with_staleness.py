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
    'num_client': 20,  # 增加客户端总数以制造更明显的稀疏参与 (Staleness)
    'num_round': 300,  # 训练轮数
    'clients_per_round': 5,  # 每轮参与较少，增加 Staleness 的概率
    'local_epoch': 2,
    'batch_size': 64,
    'lr': 0.01,
    'q_level': 2,  # 2bit 量化

    # --- 创新点开关 ---
    'enable_staleness_decay': True,  # 是否开启“时间衰减因子”创新
    'decay_lambda': 0.6,  # 衰减基数 (0 < lambda <= 1)，值越小衰减越快

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dataset_root': './data',
    'log_dir': './logs',
    'seed': 42
}

# --- 初始化日志 ---
if not os.path.exists(ARGS['log_dir']):
    os.makedirs(ARGS['log_dir'])
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
decay_str = "DecayOn" if ARGS['enable_staleness_decay'] else "DecayOff"
log_file = os.path.join(ARGS['log_dir'], f'fl_{ARGS["dataset"]}_{decay_str}_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
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


# --- 创新点：支持陈旧性衰减的压缩器 ---
class StaleAdaptiveCompressor:
    def __init__(self, model_params, bits, enable_decay, decay_lambda):
        self.bits = bits
        self.enable_decay = enable_decay
        self.decay_lambda = decay_lambda

        # 本地误差缓存
        self.error_buffer = {name: torch.zeros_like(param) for name, param in model_params.items()}

        # 记录该客户端上一次参与训练的轮次
        self.last_participated_round = -1

    def compress(self, new_gradients, current_round):
        compressed_gradients = {}

        # --- 1. 计算陈旧性 (Staleness) ---
        if self.last_participated_round == -1:
            staleness = 1  # 第一次参与，视为无延迟
        else:
            staleness = current_round - self.last_participated_round

        # 更新参与记录
        self.last_participated_round = current_round

        # --- 2. 计算衰减因子 (Gamma) ---
        if self.enable_decay:
            # 公式: gamma = lambda ^ (tau - 1)
            # tau=1 -> gamma=1 (新鲜误差，全额利用)
            # tau大 -> gamma变小 (陈旧误差，降低权重)
            gamma = self.decay_lambda ** (staleness - 1)
        else:
            gamma = 1.0  # 传统 EF，不衰减

        # 为了打印日志，这里简单取一个参数的名称
        sample_key = list(new_gradients.keys())[0]
        # logging.debug(f"Staleness: {staleness}, Gamma: {gamma:.4f}")

        for name, grad in new_gradients.items():
            if grad is None: continue

            # --- 3. 应用衰减的误差修正 ---
            # Corrected = Gradient + Gamma * Old_Error
            old_error = self.error_buffer[name]
            corrected_grad = grad + gamma * old_error

            # --- 4. 量化 ---
            compressed_grad = quantize_tensor(corrected_grad, self.bits)

            # --- 5. 更新误差缓存 ---
            # 新误差 = 修正后的梯度 - 实际发送的量化值
            self.error_buffer[name] = corrected_grad - compressed_grad

            compressed_gradients[name] = compressed_grad

        return compressed_gradients, gamma


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

    final_weights = model.state_dict()
    updates = {name: final_weights[name] - initial_weights[name] for name in initial_weights}
    return updates, avg_loss, acc


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


# --- 实验运行器 ---
def run_experiment(exp_name, enable_decay, client_loaders, test_loader, device):
    logging.info(f"START: {exp_name} (Decay={enable_decay}, Lambda={ARGS['decay_lambda']})")
    set_seed(ARGS['seed'])  # 重置种子，确保公平对比

    global_model = get_model(ARGS['dataset']).to(device)

    # 初始化带衰减功能的压缩器
    client_compressors = [
        StaleAdaptiveCompressor(
            global_model.state_dict(),
            ARGS['q_level'],
            enable_decay=enable_decay,
            decay_lambda=ARGS['decay_lambda']
        )
        for _ in range(ARGS['num_client'])
    ]

    history = {'server_acc': [], 'server_loss': []}

    for round_idx in range(ARGS['num_round']):
        selected_clients = np.random.choice(range(ARGS['num_client']), ARGS['clients_per_round'], replace=False)
        round_updates = []

        # 记录本轮平均 gamma 值，用于观察
        gammas = []

        for client_idx in selected_clients:
            local_model = copy.deepcopy(global_model)
            updates, loss, acc = train_client(local_model, client_loaders[client_idx], device, ARGS['lr'],
                                              ARGS['local_epoch'])

            # 压缩 + 自适应误差反馈
            # 传入 current_round 用于计算 staleness
            compressed_updates, gamma = client_compressors[client_idx].compress(updates, round_idx)
            round_updates.append(compressed_updates)
            gammas.append(gamma)

        # 聚合
        global_dict = global_model.state_dict()
        for key in global_dict:
            key_updates = torch.stack([upd[key] for upd in round_updates])
            global_dict[key] += torch.mean(key_updates, dim=0)
        global_model.load_state_dict(global_dict)

        # 评估
        s_loss, s_acc = evaluate(global_model, test_loader, device)
        history['server_loss'].append(s_loss)
        history['server_acc'].append(s_acc)

        avg_gamma = sum(gammas) / len(gammas)
        if (round_idx + 1) % 1 == 0:
            logging.info(f"[{exp_name}] R{round_idx + 1}: Acc={s_acc:.2f}%, AvgGamma={avg_gamma:.2f}")

    return history


# --- 主函数 ---
def main():
    device = torch.device(ARGS['device'])
    logging.info(f"Using device: {device}, Dataset: {ARGS['dataset']}")

    # 加载数据
    set_seed(ARGS['seed'])
    client_loaders, test_loader = get_data(ARGS['dataset'], ARGS['num_client'])

    # 实验 1: 开启时间衰减 (Proposed Method)
    hist_decay = run_experiment("Proposed(Decay)", True, client_loaders, test_loader, device)

    # 实验 2: 关闭时间衰减 (Baseline: Standard EF)
    hist_std = run_experiment("Baseline(Std EF)", False, client_loaders, test_loader, device)

    # --- 绘图 ---
    epochs = range(1, ARGS['num_round'] + 1)
    plt.figure(figsize=(14, 6))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist_decay['server_acc'], label='Proposed (Staleness Decay)', marker='o', markersize=4)
    plt.plot(epochs, hist_std['server_acc'], label='Baseline (Standard EF)', marker='x', markersize=4, linestyle='--')
    plt.title(f'Accuracy ({ARGS["dataset"].upper()}, Q={ARGS["q_level"]}bit)')
    plt.xlabel('Round');
    plt.ylabel('Accuracy (%)');
    plt.legend();
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist_decay['server_loss'], label='Proposed (Staleness Decay)', marker='o', markersize=4)
    plt.plot(epochs, hist_std['server_loss'], label='Baseline (Standard EF)', marker='x', markersize=4, linestyle='--')
    plt.title(f'Loss ({ARGS["dataset"].upper()}, Q={ARGS["q_level"]}bit)')
    plt.xlabel('Round');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)

    save_path = os.path.join(ARGS['log_dir'], f'comparison_decay_{ARGS["dataset"]}.png')
    plt.savefig(save_path)
    logging.info(f"Plots saved to {save_path}")
    plt.show()


if __name__ == '__main__':
    main()