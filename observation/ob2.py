import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import copy
import math


# 1. 配置参数
class Config:
    NUM_CLIENTS = 10
    NUM_ROUNDS = 10
    CLIENTS_PER_ROUND = 4
    EPOCHS_PER_CLIENT = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    QUANTIZATION_LEVELS = [2, 4, 8, 16]


# 2. 模型定义 (ResNet18 for CIFAR-10)
def get_resnet18_cifar():
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    # CIFAR-10 适配
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model


# 3. 量化函数 (跳过 BN 层)
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


# 4. 数据准备 (CIFAR-10)
def get_data_loaders(num_clients):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    num_samples_per_client = len(full_train_dataset) // num_clients
    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)

    client_loaders = []
    for i in range(num_clients):
        client_indices = indices[i * num_samples_per_client: (i + 1) * num_samples_per_client]
        client_dataset = Subset(full_train_dataset, client_indices)
        client_loader = DataLoader(client_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
        client_loaders.append(client_loader)

    test_loader = DataLoader(test_dataset, batch_size=1000)
    return client_loaders, test_loader


# 5. 客户端
class Client:
    def __init__(self, client_id, data_loader, device):
        self.id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = get_resnet18_cifar().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()

        final_weights = self.model.state_dict()
        model_update = {}

        for key in final_weights:
            diff = (final_weights[key] - initial_weights[key]).float()
            is_bn_stat = 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key

            if is_bn_stat:
                model_update[key] = diff
            else:
                model_update[key] = stochastic_quantize(diff, q_level)

        return model_update


# 6. 服务端
def server_round(global_model, clients, selected_client_ids, q_level):
    global_weights = global_model.state_dict()
    quantized_updates = []

    for client_id in selected_client_ids:
        client = clients[client_id]
        client.set_weights(global_weights)
        update = client.train(q_level)
        quantized_updates.append(update)

    aggregated_update = {key: torch.zeros_like(global_weights[key], dtype=torch.float) for key in global_weights}

    for update in quantized_updates:
        for key in aggregated_update:
            aggregated_update[key] += update[key] / len(selected_client_ids)

    new_global_weights = {}
    for key in global_weights:
        updated_val = global_weights[key] + aggregated_update[key]
        if global_weights[key].dtype == torch.long:
            new_global_weights[key] = updated_val.long()
        else:
            new_global_weights[key] = updated_val

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

    client_loaders, test_loader = get_data_loaders(Config.NUM_CLIENTS)
    results = {}

    temp_model = get_resnet18_cifar()
    num_params = sum(p.numel() for p in temp_model.parameters())
    print(f"Model: ResNet18 (Adapted for CIFAR-10), Total Parameters: {num_params}")

    for q in Config.QUANTIZATION_LEVELS:
        print(f"\n--- Running experiment with Quantization Level q = {q} ---")

        global_model = get_resnet18_cifar().to(Config.DEVICE)
        clients = [Client(i, client_loaders[i], Config.DEVICE) for i in range(Config.NUM_CLIENTS)]

        round_losses = []
        # 虽然不画图了，但保留计算供控制台打印参考
        communication_costs = []
        cumulative_comm = 0
        bits_per_param = math.ceil(math.log2(q)) if q > 1 else 32
        comm_per_round = Config.CLIENTS_PER_ROUND * num_params * bits_per_param

        for round_num in range(Config.NUM_ROUNDS):
            selected_clients = np.random.choice(range(Config.NUM_CLIENTS), Config.CLIENTS_PER_ROUND, replace=False)
            server_round(global_model, clients, selected_clients, q_level=q)

            loss, acc = evaluate_model(global_model, test_loader, Config.DEVICE)

            round_losses.append(loss)
            cumulative_comm += comm_per_round
            communication_costs.append(cumulative_comm / 1e6)

            print(
                f"Round {round_num + 1:2d}/{Config.NUM_ROUNDS} | Loss: {loss:.4f} | Accuracy: {acc:.2f}% | Comm: {communication_costs[-1]:.2f} Mbits")

            if math.isnan(loss):
                print("Error: Loss became NaN. Stopping this experiment.")
                break

        results[q] = {'loss': round_losses, 'comm': communication_costs}

    # 8. 绘图 (修改部分)
    plt.figure(figsize=(10, 6))
    for q, data in results.items():
        if len(data['loss']) > 0 and not math.isnan(data['loss'][0]):
            # 生成 x 轴数据：1 到 轮次数
            rounds = range(1, len(data['loss']) + 1)

            # 修改：X轴使用 rounds，而非 data['comm']
            plt.plot(rounds, data['loss'], marker='o', linestyle='-', markersize=4, label=f'q = {q}')

    plt.title('ResNet18 on CIFAR-10: Test Loss vs. Round Number')
    plt.xlabel('Round Number')  # 修改 X 轴标签
    plt.ylabel('Test Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()