import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import logging
import copy
import os
from datetime import datetime
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. Environment & Logging Configuration
# ==========================================
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"fedavg_noniid_quantization_3_{run_timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

checkpoint_dir = os.path.join("./saved_models", run_timestamp)
os.makedirs(checkpoint_dir, exist_ok=True)


# ==========================================
# 2. Non-IID Data Partitioning (Dirichlet)
# ==========================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    """Partition data into Non-IID distribution using Dirichlet."""
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs


# ==========================================
# 3. Model Definition (ResNet-18 for CIFAR-10)
# ==========================================
def build_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ==========================================
# 4. Core Metric Calculation Tools
# ==========================================
def calculate_shannon_entropy(weight_tensor, bins=256):
    w = weight_tensor.detach().cpu().numpy()
    hist, _ = np.histogram(w, bins=bins, density=True)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return entropy


def simulate_quantization_noise(weight_tensor, bits):
    w = weight_tensor.detach()
    q_max = (2 ** (bits - 1)) - 1
    scale = w.abs().max() / q_max

    if scale == 0:
        return 0.0

    w_q = torch.round(w / scale) * scale
    noise_l2_sq = torch.norm(w_q - w, p=2) ** 2

    return noise_l2_sq.item()


def get_resnet18_layer_statistics(model):
    """
    Compute statistics for the 18 counted layers in ResNet-18:
    1 stem conv + 16 residual-block convs + 1 final fc.
    BatchNorm and downsample projection layers are not counted.
    """
    counted_layers = [
        ("layer_01", "conv1.weight"),
        ("layer_02", "layer1.0.conv1.weight"),
        ("layer_03", "layer1.0.conv2.weight"),
        ("layer_04", "layer1.1.conv1.weight"),
        ("layer_05", "layer1.1.conv2.weight"),
        ("layer_06", "layer2.0.conv1.weight"),
        ("layer_07", "layer2.0.conv2.weight"),
        ("layer_08", "layer2.1.conv1.weight"),
        ("layer_09", "layer2.1.conv2.weight"),
        ("layer_10", "layer3.0.conv1.weight"),
        ("layer_11", "layer3.0.conv2.weight"),
        ("layer_12", "layer3.1.conv1.weight"),
        ("layer_13", "layer3.1.conv2.weight"),
        ("layer_14", "layer4.0.conv1.weight"),
        ("layer_15", "layer4.0.conv2.weight"),
        ("layer_16", "layer4.1.conv1.weight"),
        ("layer_17", "layer4.1.conv2.weight"),
        ("layer_18", "fc.weight"),
    ]

    named_parameters = dict(model.named_parameters())
    layer_statistics = []

    for layer_id, param_name in counted_layers:
        if param_name not in named_parameters:
            raise KeyError(f"Missing expected ResNet-18 parameter: {param_name}")

        weight_tensor = named_parameters[param_name].detach().view(-1).cpu()
        layer_statistics.append(
            {
                "layer_id": layer_id,
                "param_name": param_name,
                "num_params": weight_tensor.numel(),
                "entropy": calculate_shannon_entropy(weight_tensor),
                "quantization_noise": {
                    bit: simulate_quantization_noise(weight_tensor, bit) for bit in range(2, 9)
                }
            }
        )

    return layer_statistics


# ==========================================
# 5. Training and Evaluation Functions
# ==========================================
def client_update(client_model, dataloader, local_epochs, lr, device):
    """Executes local training and returns the weights and average train loss."""
    client_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)

    epoch_loss = []

    for epoch in range(local_epochs):
        batch_loss = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    avg_train_loss = sum(epoch_loss) / len(epoch_loss)
    return client_model.state_dict(), avg_train_loss


def evaluate_model(model, testloader, device):
    """Evaluates the model on the global test dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ==========================================
# 6. Main execution
# ==========================================
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"--- Experiment Initialized | Target Device: {device} ---")
    logger.info(f"--- Run Timestamp: {run_timestamp} ---")
    logger.info(f"--- Log File: {log_filename} ---")
    logger.info(f"--- Checkpoint Directory: {checkpoint_dir} ---")

    # Federated Learning Hyperparameters
    num_clients = 10
    clients_per_round = 5
    global_rounds = 55
    local_epochs = 2
    lr = 0.01
    validation_rounds = [10, 20, 50]
    data_root = "/Users/chentonghao/IWQoS/newdiff/data/"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load Train Dataset (for Non-IID splitting)
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    train_labels = np.array(trainset.targets)

    # Load Global Test Dataset
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # Non-IID Dirichlet Split (alpha=0.5)
    client_idcs = dirichlet_split_noniid(train_labels, alpha=0.5, n_clients=num_clients)
    dataloaders = [DataLoader(DatasetSplit(trainset, idcs), batch_size=32, shuffle=True) for idcs in client_idcs]

    global_model = build_resnet18().to(device)
    global_weights = global_model.state_dict()

    for round_idx in range(1, global_rounds + 1):
        logger.info(f"\n[Global Round {round_idx}]")

        selected_clients = np.random.choice(range(num_clients), clients_per_round, replace=False)
        local_weights_list = []

        for client_id in selected_clients:
            # Step A: Distribute Global Model
            client_model = build_resnet18().to(device)
            client_model.load_state_dict(copy.deepcopy(global_weights))

            # Step B: Execute Local Update
            local_weights, train_loss = client_update(client_model, dataloaders[client_id], local_epochs, lr, device)
            local_weights_list.append(copy.deepcopy(local_weights))

            # Step C: Evaluate Local Model on Global Test Set
            _, local_test_acc = evaluate_model(client_model, testloader, device)

            logger.info(
                f"  Client {client_id} Local Update Completed | Local Train Loss: {train_loss:.4f} | Local Test Acc: {local_test_acc:.2f}%")

            # ==========================================
            # Step D: Validation Phase (Strictly AFTER local update)
            # ==========================================
            if round_idx in validation_rounds:
                model_path = os.path.join(
                    checkpoint_dir,
                    f"client_{client_id}_round_{round_idx}_{run_timestamp}.pth"
                )
                torch.save(local_weights, model_path)

                logger.info(
                    f"\n  >>> [Validation Phase] Analyzing Client {client_id} Post-Update Weights | Model saved to {model_path} <<<")

                client_model.load_state_dict(local_weights)
                layer_statistics = get_resnet18_layer_statistics(client_model)
                for layer_stat in layer_statistics:
                    logger.info(
                        f"    ResNet-18 {layer_stat['layer_id']} | "
                        f"Param: {layer_stat['param_name']} | "
                        f"Params: {layer_stat['num_params']} | "
                        f"Shannon Entropy: {layer_stat['entropy']:.4f}"
                    )

                    noise_str = "      Quantization Noise L2^2: "
                    for bit, abs_noise in layer_stat["quantization_noise"].items():
                        noise_str += f"[{bit}b: {abs_noise:.4f}] "
                    logger.info(noise_str)

        # Global Aggregation (FedAvg)
        avg_weights = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights_list)):
                avg_weights[key] += local_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))

        global_weights = avg_weights
        global_model.load_state_dict(global_weights)

        # Evaluate Global Model on Global Test Set
        global_test_loss, global_test_acc = evaluate_model(global_model, testloader, device)
        logger.info(
            f"Global Round {round_idx} Aggregation Completed | Global Test Loss: {global_test_loss:.4f} | Global Test Acc: {global_test_acc:.2f}%")


if __name__ == "__main__":
    main()
