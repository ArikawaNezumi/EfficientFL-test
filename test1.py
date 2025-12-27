import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from scipy.stats import pearsonr

SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 模型：使用预训练 ResNet18 (冻结 Backbone)
# ==========================================
def get_model():
    # 加载预训练模型，保证特征提取能力
    model = torchvision.models.resnet18(pretrained=True)

    # 冻结所有层
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层 (Reset FC)，这一层是可训练的
    # ResNet18 fc in_features=512
    model.fc = nn.Linear(512, 10)
    # 只有 model.fc 的参数 require_grad=True
    return model


# ==========================================
# 数据划分 (同前)
# ==========================================
# ... (保持之前的 get_partitioned_data 函数不变) ...
def get_partitioned_data(num_clients=20, alpha=0.1):  # alpha改小点，差异更明显
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # 为了速度，只取 CIFAR-10 的一部分子集进行验证
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # ... (其余逻辑不变) ...
    # 略写，直接复用之前的逻辑，把 Subset 的 transform 加上
    # 这里为了代码简洁，请直接复用上一版函数的逻辑
    targets = np.array(trainset.targets)
    client_idcs = [[] for _ in range(num_clients)]
    for c in range(10):
        idx_k = np.where(targets == c)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.array(
            [p * (len(idx_j) < len(targets) / num_clients) for p, idx_j in zip(proportions, client_idcs)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)
        for i in range(num_clients):
            client_idcs[i] += idx_batch[i].tolist()

    client_dists = []
    for i in range(num_clients):
        labels = targets[client_idcs[i]]
        counts = np.bincount(labels, minlength=10)
        total = counts.sum()
        client_dists.append(counts / total if total > 0 else np.zeros(10))
    return trainset, client_idcs, np.array(client_dists)


# ==========================================
# 核心实验
# ==========================================
def run_simulation(epochs=1):
    print(f"Running simulation (Pretrained ResNet, Frozen Backbone)...")
    NUM_CLIENTS = 20
    GROUP_SIZE = 2

    trainset, client_idcs, client_dists = get_partitioned_data(NUM_CLIENTS, alpha=0.1)
    p_global = np.mean(client_dists, axis=0)

    # 1. 初始化模型 (Pretrained)
    global_model = get_model().to(device)
    initial_weights = copy.deepcopy(global_model.fc.weight.data)  # 只关注 FC 层

    client_grads_norm = []  # 存储每个客户端每类梯度的范数

    criterion = nn.CrossEntropyLoss()

    # 2. 客户端训练
    for i in range(NUM_CLIENTS):
        model = get_model().to(device)
        model.fc.weight.data = copy.deepcopy(initial_weights)

        # 优化器只优化 FC 层
        optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.0)  # 去掉动量，看纯梯度

        if len(client_idcs[i]) == 0:
            client_grads_norm.append(np.zeros(10))
            continue

        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(trainset, client_idcs[i]),
            batch_size=32, shuffle=True
        )

        model.train()
        # 累积梯度
        accumulated_grad = torch.zeros_like(model.fc.weight.data)

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()

            # 累加梯度 (不更新权重，或者最后更新，这里只为了看梯度)
            accumulated_grad += model.fc.weight.grad.data

        # 计算每一类对应权重的梯度范数 (Gradient Norm per Class)
        # weight shape: [10, 512] -> row c is w_c
        # 计算每一行的 L2 Norm
        grad_norms = torch.norm(accumulated_grad, dim=1).cpu().numpy()  # shape [10]

        # 归一化 (变成能量分布)
        if grad_norms.sum() > 0:
            grad_dist = grad_norms / grad_norms.sum()
        else:
            grad_dist = np.zeros(10)

        client_grads_norm.append(grad_dist)
        print(f"\rClient {i + 1} done.", end="")

    print("\nCalculating...")

    # 全局平均梯度分布
    global_grad_dist = np.mean(client_grads_norm, axis=0)

    # --- 随机抽取 Group ---
    group_indices = random.sample(range(NUM_CLIENTS), GROUP_SIZE)

    # 1. GT: 真实标签分布差异
    p_group = np.mean([client_dists[i] for i in group_indices], axis=0)
    gt_diff = np.abs(p_group - p_global)

    # 2. Proxy: 梯度能量分布差异
    g_group = np.mean([client_grads_norm[i] for i in group_indices], axis=0)
    proxy_diff = np.abs(g_group - global_grad_dist)

    # --- 绘图 ---
    classes = np.arange(10)
    corr, _ = pearsonr(gt_diff, proxy_diff)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_ylabel('GT: Label Dist Diff', color=color)
    ax1.bar(classes, gt_diff, color=color, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Proxy: Gradient Norm Diff', color=color)
    ax2.plot(classes, proxy_diff, color=color, marker='o', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Validation (Frozen Feature Extractor)\nPearson Correlation: {corr:.4f}")
    plt.show()


if __name__ == '__main__':
    run_simulation()