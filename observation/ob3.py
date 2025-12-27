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
    NUM_ROUNDS = 30
    CLIENTS_PER_ROUND = 4
    EPOCHS_PER_CLIENT = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 修改点 1: 固定量化位宽，改为定义要实验的层 ---
    FIXED_Q_LEVEL = 8  # 固定为 8-bit 量化 (2^8 = 256 levels)
    # 定义我们要单独测试的层名称 (对应 SimpleCNN 中的定义)
    # 还可以添加 "all" (全部量化) 或 "none" (全不量化) 作为基准
    TARGET_LAYERS = ["conv1", "conv2", "fc1", "fc2"]


# 2. 模型定义 (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # layer name: conv1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # layer name: conv2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # layer name: fc1
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.relu3 = nn.ReLU()
        # layer name: fc2
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 3. 量化函数 (保持不变)
def stochastic_quantize(tensor, q_level):
    if q_level is None or not isinstance(q_level, int) or q_level <= 1:
        return tensor

    norm = torch.norm(tensor)
    if norm == 0:
        return tensor

    normalized_tensor = tensor / norm
    scaled_abs_tensor = torch.abs(normalized_tensor) * q_level
    rand_for_rounding = torch.rand_like(scaled_abs_tensor)
    rounded_tensor = torch.floor(scaled_abs_tensor + rand_for_rounding)
    quantized_abs_tensor = torch.clamp(rounded_tensor, 0, q_level) / q_level
    dequantized_tensor = torch.sign(tensor) * quantized_abs_tensor * norm

    return dequantized_tensor


# 4. 数据准备 (保持不变)
def get_data_loaders(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    # --- 修改点 2: train 函数接收 target_layer_name ---
    def train(self, target_layer_name):
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

        # 计算模型更新
        final_weights = self.model.state_dict()
        model_update = {key: final_weights[key] - initial_weights[key] for key in initial_weights}

        # --- 修改点 3: 只量化指定的层 ---
        quantized_update = {}
        for key, value in model_update.items():
            # key 的格式通常是 "layer_name.weight" 或 "layer_name.bias"
            # 如果 target_layer_name (例如 "conv1") 在 key 中，则量化
            if target_layer_name in key:
                quantized_update[key] = stochastic_quantize(value, Config.FIXED_Q_LEVEL)
            else:
                # 其他层保持全精度 (模拟 32-bit 传输)
                quantized_update[key] = value

        return quantized_update


# 6. 服务端/主训练循环
# --- 修改点 4: 参数改为 target_layer_name ---
def server_round(global_model, clients, selected_client_ids, target_layer_name):
    global_weights = global_model.state_dict()
    quantized_updates = []

    for client_id in selected_client_ids:
        client = clients[client_id]
        client.set_weights(global_weights)
        # 传入目标层名称
        update = client.train(target_layer_name)
        quantized_updates.append(update)

    aggregated_update = {key: torch.zeros_like(global_weights[key]) for key in global_weights}
    for update in quantized_updates:
        for key in aggregated_update:
            aggregated_update[key] += update[key] / len(selected_client_ids)

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


# 辅助函数：计算特定策略下的通信位宽
def calculate_bits_per_round(model, target_layer_name, q_bits=8):
    total_bits = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        # 如果参数属于目标层，则使用低比特，否则使用 32 bit
        if target_layer_name in name:
            total_bits += num_params * math.ceil(math.log2(q_bits))
        else:
            total_bits += num_params * 32

    # 考虑选中的客户端数量
    return total_bits * Config.CLIENTS_PER_ROUND


# 7. 主程序
if __name__ == "__main__":
    print(f"Running on device: {Config.DEVICE}")
    print(f"Fixed Quantization Level: {Config.FIXED_Q_LEVEL}")

    client_loaders, test_loader = get_data_loaders(Config.NUM_CLIENTS)
    results = {}

    # 用于计算参数的临时模型
    temp_model = SimpleCNN()

    # --- 修改点 5: 遍历层名称而不是遍历量化等级 ---
    for target_layer in Config.TARGET_LAYERS:
        print(f"\n--- Experiment: Quantizing ONLY layer '{target_layer}' (others FP32) ---")

        global_model = SimpleCNN().to(Config.DEVICE)
        clients = [Client(i, client_loaders[i], Config.DEVICE) for i in range(Config.NUM_CLIENTS)]

        round_losses = []
        communication_costs = []
        cumulative_comm = 0

        # --- 修改点 6: 计算混合精度的通信开销 ---
        bits_per_round = calculate_bits_per_round(temp_model, target_layer, Config.FIXED_Q_LEVEL)
        print(f"  -> Bits per round for this config: {bits_per_round / 1e6:.4f} Mbits")

        for round_num in range(Config.NUM_ROUNDS):
            selected_clients = np.random.choice(range(Config.NUM_CLIENTS), Config.CLIENTS_PER_ROUND, replace=False)

            # 传入当前实验的目标层
            server_round(global_model, clients, selected_clients, target_layer_name=target_layer)

            loss, acc = evaluate_model(global_model, test_loader, Config.DEVICE)

            round_losses.append(loss)
            cumulative_comm += bits_per_round
            communication_costs.append(cumulative_comm / 1e6)  # Mbits

            print(
                f"Round {round_num + 1:2d}/{Config.NUM_ROUNDS} | Loss: {loss:.4f} | Accuracy: {acc:.2f}% | Comm: {communication_costs[-1]:.2f} Mbits")

        results[target_layer] = {'loss': round_losses, 'comm': communication_costs}

    # 8. 绘图
    plt.figure(figsize=(10, 6))
    for layer_name, data in results.items():
        # 这里 x 轴依然使用通信开销，可以观察不同层的参数量对总通信的影响
        # 如果想看轮次，可以改用 range(len(data['loss']))
        plt.plot(data['comm'], data['loss'], marker='o', linestyle='-', markersize=4, label=f'Quantized: {layer_name}')

    plt.title(f'Impact of Quantizing Specific Layers (Fixed q={Config.FIXED_Q_LEVEL})')
    plt.xlabel('Cumulative Communication (Mbits)')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()