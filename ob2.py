import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging
import copy
import os
from torch.utils.data import DataLoader, Dataset

# ==========================================
# 1. Environment & Logging Configuration
# ==========================================
log_filename = "fedavg_hawqv2_quantization_2.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

checkpoint_dir = "./saved_models"
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
# 3. Model Definition (3 Conv + 2 FC)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 4. Core Metric Calculation Tools (HAWQ-V2)
# ==========================================
def estimate_layer_hessian_trace(model, dataloader, param_name, device, num_batches=2, hutchinson_steps=20):
    """
    Estimates the trace of the Hessian for a specific layer using Hutchinson's method.
    Uses Rademacher distribution for random vectors and Hessian-Vector Products (HVP).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    trace_est = 0.0
    total_samples = 0

    target_param = dict(model.named_parameters())[param_name]

    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # 1st derivative (gradient) with create_graph=True to allow 2nd derivative
        grad = torch.autograd.grad(loss, target_param, create_graph=True)[0]

        batch_trace = 0.0
        # Hutchinson iterative estimation
        for _ in range(hutchinson_steps):
            # Generate random vector from Rademacher distribution {-1, 1}
            v = torch.randint_like(target_param, high=2, device=device) * 2 - 1.0

            # 2nd derivative (Hessian-Vector Product)
            hvp = torch.autograd.grad(grad, target_param, grad_outputs=v, retain_graph=True)[0]

            # z^T * H * z
            batch_trace += torch.sum(hvp * v).item()

        trace_est += (batch_trace / hutchinson_steps) * images.size(0)
        total_samples += images.size(0)

    # Free the graph memory
    grad = grad.detach()
    model.zero_grad()


    return trace_est / total_samples if total_samples > 0 else 0.0


def simulate_quantization_noise(weight_tensor, bits):
    """Calculates the L2 norm of quantization perturbation: ||Q(W) - W||_2^2"""
    w = weight_tensor.detach()
    q_max = (2 ** (bits - 1)) - 1
    scale = w.abs().max() / q_max

    if scale == 0:
        return 0.0

    w_q = torch.round(w / scale) * scale
    noise_l2_sq = torch.norm(w_q - w, p=2) ** 2

    return noise_l2_sq.item()


# ==========================================
# 5. Training and Evaluation Functions
# ==========================================
def client_update(client_model, dataloader, local_epochs, lr, device):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"--- Experiment Initialized | Target Device: {device} ---")

    num_clients = 10
    clients_per_round = 5
    global_rounds = 55
    local_epochs = 2
    lr = 0.01
    validation_rounds = [10, 20, 50]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_labels = np.array(trainset.targets)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    client_idcs = dirichlet_split_noniid(train_labels, alpha=10, n_clients=num_clients)
    dataloaders = [DataLoader(DatasetSplit(trainset, idcs), batch_size=32, shuffle=True) for idcs in client_idcs]

    global_model = SimpleCNN().to(device)
    global_weights = global_model.state_dict()

    for round_idx in range(1, global_rounds + 1):
        logger.info(f"\n[Global Round {round_idx}]")

        selected_clients = np.random.choice(range(num_clients), clients_per_round, replace=False)
        local_weights_list = []

        for client_id in selected_clients:
            # Step A: Distribute Global Model
            client_model = SimpleCNN().to(device)
            client_model.load_state_dict(copy.deepcopy(global_weights))

            # Step B: Execute Local Update
            local_weights, train_loss = client_update(client_model, dataloaders[client_id], local_epochs, lr, device)
            local_weights_list.append(copy.deepcopy(local_weights))

            # Step C: Evaluate Local Model on Global Test Set
            _, local_test_acc = evaluate_model(client_model, testloader, device)

            logger.info(
                f"  Client {client_id} Local Update Completed | Local Train Loss: {train_loss:.4f} | Local Test Acc: {local_test_acc:.2f}%")

            # ==========================================
            # Step D: HAWQ-V2 Validation Phase
            # Strictly executed AFTER local update using local_weights
            # ==========================================
            if round_idx in validation_rounds:
                model_path = os.path.join(checkpoint_dir, f"ob2_client_{client_id}_round_{round_idx}.pth")
                torch.save(local_weights, model_path)

                logger.info(
                    f"\n  >>> [Validation Phase] Analyzing Client {client_id} Post-Update Weights | Model saved to {model_path} <<<")

                client_model.load_state_dict(local_weights)

                for name, param in client_model.named_parameters():
                    if 'weight' in name:
                        # 1. Compute Hessian Trace via Hutchinson
                        trace_val = estimate_layer_hessian_trace(
                            model=client_model,
                            dataloader=dataloaders[client_id],
                            param_name=name,
                            device=device,
                            num_batches=2,  # Increase for higher fidelity
                            hutchinson_steps=20  # Paper recommends ~50 for convergence
                        )
                        logger.info(f"    Layer: {name} | Hessian Trace: {trace_val:.4f}")

                        noise_str = "      Quant Noise L2^2 : "
                        sens_str = "      HAWQ Sensitivity : "

                        # 2. Compute 2-8 bit Noise and Sensitivity Metric
                        for bit in range(2, 9):
                            abs_noise = simulate_quantization_noise(param, bit)
                            sensitivity = trace_val * abs_noise  # Omega_i metric

                            noise_str += f"[{bit}b: {abs_noise:.4f}] "
                            sens_str += f"[{bit}b: {sensitivity:.4f}] "

                        logger.info(noise_str)
                        logger.info(sens_str)

        # Global Aggregation (FedAvg)
        avg_weights = copy.deepcopy(local_weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights_list)):
                avg_weights[key] += local_weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(local_weights_list))

        global_weights = avg_weights
        global_model.load_state_dict(global_weights)

        global_test_loss, global_test_acc = evaluate_model(global_model, testloader, device)
        logger.info(
            f"Global Round {round_idx} Aggregation Completed | Global Test Loss: {global_test_loss:.4f} | Global Test Acc: {global_test_acc:.2f}%")


if __name__ == "__main__":
    main()