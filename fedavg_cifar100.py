import argparse
import copy
import logging
import os
import random
import tarfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models


RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        image, label = self.dataset[self.idxs[index]]
        return image, label


class FlexibleCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, data_dir, train, transform=None):
        data_path = Path(data_dir)
        self.base_folder = data_path.name
        super().__init__(
            root=str(data_path.parent),
            train=train,
            transform=transform,
            download=False,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Standard FedAvg on CIFAR-100 with ResNet.")
    parser.add_argument("--data-root", type=str, default="/autodl-pub/data")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=5)
    parser.add_argument("--global-rounds", type=int, default=100)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="./runs")
    parser.add_argument("--log-prefix", type=str, default="fedavg_cifar100")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    return parser.parse_args()


def setup_paths(args):
    run_dir = os.path.join(args.save_dir, RUN_TIMESTAMP)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "{}_{}.log".format(args.log_prefix, RUN_TIMESTAMP))
    return run_dir, log_path


def setup_logger(log_path):
    logger = logging.getLogger("fedavg_cifar100")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet(model_name="resnet18", num_classes=100):
    if model_name == "resnet34":
        model = models.resnet34(pretrained=False)
    else:
        model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_indices = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    client_indices = [[] for _ in range(n_clients)]

    for class_idx, fracs in zip(class_indices, label_distribution):
        split_points = (np.cumsum(fracs)[:-1] * len(class_idx)).astype(int)
        split_data = np.split(class_idx, split_points)
        for client_id, idxs in enumerate(split_data):
            client_indices[client_id].append(idxs)

    merged_indices = []
    for idxs in client_indices:
        if len(idxs) == 0:
            merged_indices.append(np.array([], dtype=np.int64))
        else:
            merged_indices.append(np.concatenate(idxs))
    return merged_indices


def load_datasets(data_root, extract_root):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4867, 0.4408),
                std=(0.2675, 0.2565, 0.2761),
            ),
        ]
    )

    resolved_root, use_direct_folder = resolve_cifar100_root(data_root, extract_root)

    if use_direct_folder:
        train_set = FlexibleCIFAR100(
            data_dir=resolved_root,
            train=True,
            transform=train_transform,
        )
        test_set = FlexibleCIFAR100(
            data_dir=resolved_root,
            train=False,
            transform=test_transform,
        )
    else:
        train_set = torchvision.datasets.CIFAR100(
            root=resolved_root,
            train=True,
            download=False,
            transform=train_transform,
        )
        test_set = torchvision.datasets.CIFAR100(
            root=resolved_root,
            train=False,
            download=False,
            transform=test_transform,
        )
    return train_set, test_set


def extract_cifar100_archive(archive_path, extract_root):
    archive_path = Path(archive_path).expanduser()
    extract_root = Path(extract_root).expanduser()
    target_root = extract_root / "cifar100_extracted"
    target_folder = target_root / "cifar-100-python"

    if all((target_folder / name).exists() for name in ["train", "test", "meta"]):
        return str(target_root), False

    target_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(archive_path), "r:gz") as tar:
        tar.extractall(path=str(target_root))
    return str(target_root), False


def resolve_cifar100_root(data_root, extract_root):
    root_path = Path(data_root).expanduser()
    direct_folder_names = ["cifar-100-python", "cifar-100", "cifar100"]
    archive_names = ["cifar-100-python.tar.gz", "cifar-100.tar.gz", "cifar100.tar.gz"]

    direct_candidates = [root_path]
    for folder_name in direct_folder_names:
        direct_candidates.append(root_path / folder_name)
    for folder_name in direct_folder_names:
        direct_candidates.append(root_path / "data" / folder_name)
    for outer_name in ["cifar-100", "cifar100"]:
        for inner_name in direct_folder_names:
            direct_candidates.append(root_path / outer_name / inner_name)

    for candidate in direct_candidates:
        if all((candidate / name).exists() for name in ["train", "test", "meta"]):
            return str(candidate), True

    archive_candidates = []
    for archive_name in archive_names:
        archive_candidates.append(root_path / archive_name)
    for folder_name in direct_folder_names:
        for archive_name in archive_names:
            archive_candidates.append(root_path / folder_name / archive_name)
    if (root_path / "data").exists():
        for folder_name in direct_folder_names:
            for archive_name in archive_names:
                archive_candidates.append(root_path / "data" / folder_name / archive_name)

    for archive_path in archive_candidates:
        if archive_path.exists():
            return extract_cifar100_archive(archive_path, extract_root)

    root_candidates = [root_path]
    if root_path.parent != root_path:
        root_candidates.append(root_path.parent)
    if (root_path / "data").exists():
        root_candidates.append(root_path / "data")
    for outer_name in ["cifar-100", "cifar100"]:
        if (root_path / outer_name).exists():
            root_candidates.append(root_path / outer_name)

    for candidate in root_candidates:
        if all((candidate / "cifar-100-python" / name).exists() for name in ["train", "test", "meta"]):
            return str(candidate), False

    if root_path.exists() and os.access(str(root_path), os.W_OK):
        return str(root_path), False

    raise FileNotFoundError(
        "CIFAR-100 dataset not found under '{root}'. "
        "Expected one of: "
        "'{root}/cifar-100-python/', '{root}/cifar-100/', or a directory containing train/test/meta directly. "
        "The script also supports '{root}/.../cifar-100-python.tar.gz'. "
        "If the dataset is stored in a read-only directory, please pass the parent directory that already contains "
        "'cifar-100-python'.".format(root=data_root)
    )


def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def client_update(model, dataloader, local_epochs, lr, momentum, weight_decay, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    epoch_losses = []
    for _ in range(local_epochs):
        batch_losses = []
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        epoch_losses.append(sum(batch_losses) / max(len(batch_losses), 1))

    avg_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
    return model.state_dict(), avg_train_loss


def fedavg_aggregate(local_states, local_sizes):
    total_size = float(sum(local_sizes))
    aggregated_state = OrderedDict()

    for key in local_states[0].keys():
        tensor_sum = None
        reference_tensor = local_states[0][key]

        if reference_tensor.is_floating_point():
            for state, size in zip(local_states, local_sizes):
                weighted_tensor = state[key].float() * (size / total_size)
                if tensor_sum is None:
                    tensor_sum = weighted_tensor
                else:
                    tensor_sum += weighted_tensor
            aggregated_state[key] = tensor_sum.type_as(reference_tensor)
        else:
            aggregated_state[key] = reference_tensor.clone()

    return aggregated_state


def log_run_configuration(logger, args, device, run_dir, log_path):
    logger.info("=== Run Configuration ===")
    logger.info("run_timestamp: {}".format(RUN_TIMESTAMP))
    logger.info("device: {}".format(device))
    logger.info("run_dir: {}".format(run_dir))
    logger.info("log_path: {}".format(log_path))
    for key in sorted(vars(args).keys()):
        logger.info("{}: {}".format(key, getattr(args, key)))


def main():
    args = parse_args()
    run_dir, log_path = setup_paths(args)
    logger = setup_logger(log_path)

    set_seed(args.seed)
    device = select_device()
    log_run_configuration(logger, args, device, run_dir, log_path)

    train_set, test_set = load_datasets(args.data_root, run_dir)
    train_labels = np.array(train_set.targets)
    client_indices = dirichlet_split_noniid(train_labels, args.alpha, args.num_clients)

    train_loaders = []
    client_data_sizes = []
    for idxs in client_indices:
        client_subset = DatasetSplit(train_set, idxs)
        train_loaders.append(
            DataLoader(
                client_subset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        )
        client_data_sizes.append(len(client_subset))

    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    global_model = build_resnet(args.model, num_classes=100).to(device)
    global_weights = copy.deepcopy(global_model.state_dict())

    for round_idx in range(1, args.global_rounds + 1):
        logger.info("\n[Global Round {}]".format(round_idx))
        selected_clients = np.random.choice(
            range(args.num_clients),
            args.clients_per_round,
            replace=False,
        )
        logger.info("Selected Clients: {}".format(list(selected_clients)))

        local_states = []
        local_sizes = []

        for client_id in selected_clients:
            client_model = build_resnet(args.model, num_classes=100).to(device)
            client_model.load_state_dict(copy.deepcopy(global_weights))

            local_state, train_loss = client_update(
                model=client_model,
                dataloader=train_loaders[client_id],
                local_epochs=args.local_epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                device=device,
            )

            client_model.load_state_dict(local_state)
            local_test_loss, local_test_acc = evaluate_model(client_model, test_loader, device)

            local_states.append(copy.deepcopy(local_state))
            local_sizes.append(client_data_sizes[client_id])

            logger.info(
                "  Client {} | Samples: {} | Train Loss: {:.4f} | Test Loss: {:.4f} | Test Acc: {:.2f}%".format(
                    client_id,
                    client_data_sizes[client_id],
                    train_loss,
                    local_test_loss,
                    local_test_acc,
                )
            )

        aggregated_weights = fedavg_aggregate(local_states, local_sizes)
        global_weights = aggregated_weights
        global_model.load_state_dict(global_weights)

        global_test_loss, global_test_acc = evaluate_model(global_model, test_loader, device)
        logger.info(
            "Global Aggregation | Test Loss: {:.4f} | Test Acc: {:.2f}%".format(
                global_test_loss,
                global_test_acc,
            )
        )

    logger.info("\nTraining completed.")
    logger.info("Log saved to: {}".format(log_path))


if __name__ == "__main__":
    main()
