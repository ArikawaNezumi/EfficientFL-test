import torch
import torch.nn as nn
import torch.optim as optim
from dbs import BitWidthSearcher
from quant_utils import get_model_size_bits

# --- 1. 定义一个简单的 CNN 模型 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)  # Layer A
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Layer B
        self.fc1 = nn.Linear(32 * 5 * 5, 128)  # Layer C (胖层)
        self.fc2 = nn.Linear(128, 10)  # Layer D

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 注意：为了配合 dbs.py 中的简化 forward 模拟，
        # 这里的结构必须和 dbs.py 中的遍历顺序一致。
        # 在真实项目中，dbs 应该通过 hook 实现，不需要关心模型具体结构。
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. 模拟环境设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()

# 伪造一些数据 (MNIST 格式)
dummy_data = torch.randn(16, 1, 28, 28).to(device)
dummy_target = torch.randint(0, 10, (16,)).to(device)

# --- 3. 阶段一: 本地训练 (FP32) ---
# 你的论文里强调：本地训练是完整的，不量化的
print(">>> Phase 1: Local Training (FP32) ...")
optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()
# 模拟训练几步
for _ in range(5):
    optimizer.zero_grad()
    output = model(dummy_data)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
print("Local Training Done. Weights are updated (FP32).")

# --- 4. 阶段二: 带宽检测与微型搜索 ---
print("\n>>> Phase 2: Bandwidth Adaptive Search ...")

# 计算一下原始 FP32 模型有多大
full_size_bits = get_model_size_bits(model)
print(f"Original Model Size (FP32): {full_size_bits / 8 / 1024:.2f} KB")

# 模拟一个受限带宽场景：假设只能传 1/5 的大小
current_bandwidth = full_size_bits * 0.2
print(f"Current Bandwidth Limit: {current_bandwidth / 8 / 1024:.2f} KB (Compression Rate ~4x)")

# 初始化搜索器
searcher = BitWidthSearcher(model, candidate_bits=[2, 4, 8])

# 执行搜索 (只用几毫秒)
# 注意：这里我们用一个 batch 的数据来探测敏感度
final_config = searcher.search(dummy_data, dummy_target, current_bandwidth, lambda_coeff=10.0, iterations=20)

# --- 5. 阶段三: 结果展示 ---
print("\n>>> Phase 3: Final Decision & Upload")
print("Selected Bit-width Configuration:")
total_compressed_size = 0
for name, bit in final_config.items():
    # 获取参数量
    num_params = dict(model.named_modules())[name].weight.numel()
    print(f"  Layer {name}: \t{bit}-bit \t(Params: {num_params})")
    total_compressed_size += num_params * bit

print(f"\nFinal Compressed Size: {total_compressed_size / 8 / 1024:.2f} KB")
print(f"Constraint Met? {'YES' if total_compressed_size <= current_bandwidth else 'NO (Need Tuning Lambda)'}")

# 你会发现：
# fc1 这种参数巨多（32*5*5*128 = 102k参数）的层，会被压到 2-bit
# conv1 这种参数少（16*1*3*3 = 144参数）且敏感的层，可能会保留 4-bit 或 8-bit