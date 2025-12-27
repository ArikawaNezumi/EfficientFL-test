import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

print(f"PyTorch Version: {torch.__version__}")


# 1. 配置参数
class Config:
    NUM_CLIENTS = 10
    NUM_ROUNDS = 30  # CIFAR-10 较难，可适当增加轮数
    CLIENTS_PER_ROUND = 4
    EPOCHS_PER_CLIENT = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 我们将测试这些固定的量化水平
    QUANTIZATION_LEVELS = [2, 4, 8, 16]


# 2. 模型定义 (SimpleCNN for CIFAR-10)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # --- 修改点 1: 输入通道改为 3 (RGB) ---
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 32x32 -> 16x16

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 16x16 -> 8x8

        # --- 修改点 2: 全连接层输入维度调整 ---
        # 经过两次池化后特征图大小为 8x8
        self.fc1 = nn.Linear(8 * 8 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # --- 修改点 3: view 维度调整 ---
        x = x.view(-1, 8 * 8 * 32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. 量化函数
def stochastic_quantize(tensor, q_level):
    """
    简化的随机量化函数。
    """
    if q_level is None or not isinstance(q_level, int) or q_level <= 1:
        return tensor  # 不量化

    norm = torch.norm(tensor)
    if norm == 0:
        return tensor

    # 归一化到 [-1, 1]
    normalized_tensor = tensor / norm

    # 随机量化
    scaled_abs_tensor = torch.abs(normalized_tensor) * q_level
    rand_for_rounding = torch.rand_like(scaled_abs_tensor)
    rounded_tensor = torch.floor(scaled_abs_tensor + rand_for_rounding)
    quantized_abs_tensor = torch.clamp(rounded_tensor, 0, q_level) / q_level

    # 恢复符号和范数
    dequantized_tensor = torch.sign(tensor) * quantized_abs_tensor * norm

    return dequantized_tensor


# 4. 数据准备 (CIFAR-10)
def get_data_loaders(num_clients):
    # --- 修改点 4: 适配 CIFAR-10 的 normalization ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # --- 修改点 5: 加载 CIFAR-10 数据集 ---
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # IID 划分数据
    num_samples_per_client = len(full_train_dataset) // num_clients
    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)

    client_loaders = []
    for i in range(num_clients):
        client_indices = indices[i * num_samples_per_client: (i + 1) * num_samples_per_client]
        client_dataset = Subset(full_train_dataset, client_indices)
        client_loader = DataLoader(client_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        client_loaders.append(client_loader)

    test_loader = DataLoader(test_dataset, batch_size=1000)
    return client_loaders, test_loader


# 5. 客户端
class Client:
    def __init__(self, client_id, data_loader, device):
        self.id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = SimpleCNN().to(device)
        # 建议加 momentum，SimpleCNN 在 CIFAR-10 上收敛较慢
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def train(self, q_level):
        self.model.train()
        initial_weights = copy.deepcopy(self.model.state_dict())

        for epoch in range(Config.EPOCHS_PER_CLIENT):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

        # 计算模型更新 (delta)
        final_weights = self.model.state_dict()
        model_update = {key: final_weights[key] - initial_weights[key] for key in initial_weights}

        # 量化模型更新
        quantized_update = {key: stochastic_quantize(model_update[key], q_level) for key in model_update}

        return quantized_update


# 6. 服务端/主训练循环
def server_round(global_model, clients, selected_client_ids, q_level):
    global_weights = global_model.state_dict()
    quantized_updates = []

    # 分发模型并进行本地训练
    for client_id in selected_client_ids:
        client = clients[client_id]
        client.set_weights(global_weights)
        update = client.train(q_level)
        quantized_updates.append(update)

    # 聚合更新
    aggregated_update = {key: torch.zeros_like(global_weights[key]) for key in global_weights}
    for update in quantized_updates:
        for key in aggregated_update:
            aggregated_update[key] += update[key] / len(selected_client_ids)

    # 更新全局模型
    new_global_weights = {key: global_weights[key] + aggregated_update[key] for key in global_weights}
    global_model.load_state_dict(new_global_weights)


def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


# 7. 主程序
if __name__ == "__main__":
    print(f"Running on device: {Config.DEVICE}")

    # 准备数据
    client_loaders, test_loader = get_data_loaders(Config.NUM_CLIENTS)

    # 存储所有实验的结果
    results = {}

    # 获取模型参数数量用于计算通信成本
    temp_model = SimpleCNN()
    num_params = sum(p.numel() for p in temp_model.parameters())

    for q in Config.QUANTIZATION_LEVELS:
        print(f"\n--- Running experiment with Quantization Level q = {q} ---")

        # 为每个实验重置模型和客户端
        global_model = SimpleCNN().to(Config.DEVICE)
        clients = [Client(i, client_loaders[i], Config.DEVICE) for i in range(Config.NUM_CLIENTS)]

        round_losses = []
        communication_costs = []
        cumulative_comm = 0

        # 计算每轮的通信成本
        bits_per_param = math.ceil(math.log2(q)) if q > 1 else 32
        comm_per_round = Config.CLIENTS_PER_ROUND * num_params * bits_per_param

        # 联邦学习训练过程
        for round_num in range(Config.NUM_ROUNDS):
            selected_clients = np.random.choice(range(Config.NUM_CLIENTS), Config.CLIENTS_PER_ROUND, replace=False)
            server_round(global_model, clients, selected_clients, q_level=q)

            # 评估全局模型
            loss, acc = evaluate_model(global_model, test_loader, Config.DEVICE)

            # 记录数据
            round_losses.append(loss)
            cumulative_comm += comm_per_round
            communication_costs.append(cumulative_comm / 1e6)  # 转换为 Mbits

            print(
                f"Round {round_num + 1:2d}/{Config.NUM_ROUNDS} | Loss: {loss:.4f} | Accuracy: {acc:.2f}% | Comm: {communication_costs[-1]:.2f} Mbits")

        results[q] = {'loss': round_losses, 'comm': communication_costs}

    # 8. 绘图 (X 轴使用训练轮次)
    plt.figure(figsize=(10, 6))
    for q, data in results.items():
        # --- 修改: X 轴使用轮次 ---
        rounds = range(1, len(data['loss']) + 1)
        plt.plot(rounds, data['loss'], marker='o', linestyle='-', markersize=4, label=f'q = {q}')

    plt.title('CIFAR-10: Loss vs. Rounds for Different Quantization Levels')
    plt.xlabel('Training Rounds')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()