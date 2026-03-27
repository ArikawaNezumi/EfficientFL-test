import argparse
import copy
import logging
import os
import random
import tarfile
import time
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
LAYER_SPECS = [
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
LAYER_IDS = [layer_id for layer_id, _ in LAYER_SPECS]
TRACKED_PARAM_NAMES = [param_name for _, param_name in LAYER_SPECS]


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
    parser = argparse.ArgumentParser(
        description="Communication-efficient federated learning with layer-wise quantization."
    )
    parser.add_argument("--data-root", type=str, default="/autodl-pub/data")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--clients-per-round", type=int, default=5)
    parser.add_argument("--communication-budget-ratio", type=float, default=0.20)
    parser.add_argument("--upload-rate-mbps", type=float, default=2.0)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--global-rounds", type=int, default=55)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--quant-min-bit", type=int, default=2)
    parser.add_argument("--quant-max-bit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--omega-scale", type=float, default=1.0)
    parser.add_argument("--entropy-bins", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="./runs")
    parser.add_argument("--log-prefix", type=str, default="layerquant_cifar100")
    parser.add_argument(
        "--enable-gdc",
        dest="enable_gdc",
        action="store_true",
        help="Enable layer-wise Gradient Dispersion Calibration.",
    )
    parser.add_argument(
        "--disable-gdc",
        dest="enable_gdc",
        action="store_false",
        help="Disable layer-wise Gradient Dispersion Calibration.",
    )
    parser.set_defaults(enable_gdc=True)
    return parser.parse_args()


def setup_paths(args):
    run_dir = os.path.join(args.save_dir, RUN_TIMESTAMP)
    os.makedirs(run_dir, exist_ok=True)
    log_filename = "{}_{}.log".format(args.log_prefix, RUN_TIMESTAMP)
    log_path = os.path.join(run_dir, log_filename)
    return run_dir, log_path


def setup_logger(log_path):
    logger = logging.getLogger("fed_layerwise_quant")
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


def build_resnet18(num_classes=100):
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
        split_indices = np.split(class_idx, split_points)
        for client_id, idxs in enumerate(split_indices):
            client_indices[client_id].append(idxs)

    merged_indices = []
    for idxs in client_indices:
        if len(idxs) == 0:
            merged_indices.append(np.array([], dtype=np.int64))
        else:
            merged_indices.append(np.concatenate(idxs))
    return merged_indices


def calculate_shannon_entropy(weight_tensor, bins=256):
    weights = weight_tensor.detach().view(-1).cpu().numpy()
    if weights.size == 0:
        return 0.0

    hist, _ = np.histogram(weights, bins=bins, density=True)
    hist_sum = np.sum(hist)
    if hist_sum <= 0:
        return 0.0

    prob = hist / hist_sum
    prob = prob[prob > 0]
    if prob.size == 0:
        return 0.0
    return float(-np.sum(prob * np.log2(prob)))


def simulate_quantization_noise(weight_tensor, bits):
    if bits >= 32:
        return 0.0

    weights = weight_tensor.detach().float()
    q_max = (2 ** (bits - 1)) - 1
    if q_max <= 0:
        return 0.0

    scale = weights.abs().max() / q_max
    if scale.item() == 0:
        return 0.0

    quantized = torch.round(weights / scale) * scale
    noise = torch.norm(quantized - weights, p=2) ** 2
    return float(noise.item())


def quantize_tensor(weight_tensor, bits):
    if bits >= 32 or not weight_tensor.is_floating_point():
        return weight_tensor.clone()

    q_max = (2 ** (bits - 1)) - 1
    if q_max <= 0:
        return weight_tensor.clone()

    scale = weight_tensor.detach().abs().max() / q_max
    if scale.item() == 0:
        return weight_tensor.clone()

    quantized = torch.round(weight_tensor / scale) * scale
    return quantized


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


def init_gradient_meter():
    meter = {}
    for layer_id, param_name in LAYER_SPECS:
        meter[param_name] = {
            "layer_id": layer_id,
            "sum": 0.0,
            "sq_sum": 0.0,
            "norm_sq_sum": 0.0,
            "count": 0,
        }
    return meter


def update_gradient_meter(gradient_meter, named_parameters):
    for _, param_name in LAYER_SPECS:
        grad = named_parameters[param_name].grad
        if grad is None:
            continue

        grad_cpu = grad.detach().view(-1).float().cpu()
        gradient_meter[param_name]["sum"] += float(grad_cpu.sum().item())
        gradient_meter[param_name]["sq_sum"] += float(torch.sum(grad_cpu * grad_cpu).item())
        gradient_meter[param_name]["norm_sq_sum"] += float(torch.norm(grad_cpu, p=2).item() ** 2)
        gradient_meter[param_name]["count"] += grad_cpu.numel()


def build_gradient_statistics(gradient_meter):
    gradient_statistics = {}
    for _, param_name in LAYER_SPECS:
        stats = gradient_meter[param_name]
        count = stats["count"]
        if count == 0:
            gradient_statistics[param_name] = {
                "variance": 0.0,
                "norm_sq": 0.0,
            }
            continue

        mean = stats["sum"] / count
        second_moment = stats["sq_sum"] / count
        variance = max(second_moment - mean * mean, 0.0)
        norm_sq = stats["norm_sq_sum"] / max(count, 1)
        gradient_statistics[param_name] = {
            "variance": float(variance),
            "norm_sq": float(norm_sq),
        }
    return gradient_statistics


def client_update(client_model, dataloader, local_epochs, lr, device, enable_gdc):
    client_model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
    gradient_meter = init_gradient_meter()
    named_parameters = dict(client_model.named_parameters())
    epoch_losses = []

    for _ in range(local_epochs):
        batch_losses = []
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = client_model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if enable_gdc:
                update_gradient_meter(gradient_meter, named_parameters)

            optimizer.step()
            batch_losses.append(loss.item())

        epoch_losses.append(sum(batch_losses) / max(len(batch_losses), 1))

    avg_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
    if enable_gdc:
        gradient_statistics = build_gradient_statistics(gradient_meter)
    else:
        gradient_statistics = {
            param_name: {"variance": 0.0, "norm_sq": 0.0}
            for param_name in TRACKED_PARAM_NAMES
        }

    return client_model.state_dict(), avg_train_loss, gradient_statistics


def get_model_size_bits(state_dict):
    total_bits = 0
    for tensor in state_dict.values():
        total_bits += int(tensor.numel()) * 32
    return total_bits


def get_fixed_overhead_bits(state_dict):
    tracked_names = set(TRACKED_PARAM_NAMES)
    fixed_bits = 0
    for name, tensor in state_dict.items():
        if name in tracked_names:
            continue
        fixed_bits += int(tensor.numel()) * 32
    return fixed_bits


def build_layer_option_table(state_dict, quant_min_bit, quant_max_bit, entropy_bins):
    layer_option_table = []
    for layer_id, param_name in LAYER_SPECS:
        weights = state_dict[param_name].detach().float().cpu()
        entropy = calculate_shannon_entropy(weights, bins=entropy_bins)
        entropy_ref = max(entropy, 1e-12)
        options = []

        for bits in range(quant_min_bit, quant_max_bit + 1):
            noise = simulate_quantization_noise(weights, bits)
            shannon_loss_index = np.log2(1.0 + (noise / entropy_ref))
            objective_value = shannon_loss_index
            options.append(
                {
                    "layer_id": layer_id,
                    "param_name": param_name,
                    "bits": bits,
                    "num_params": int(weights.numel()),
                    "cost_bits": int(weights.numel()) * bits,
                    "entropy": entropy,
                    "noise": noise,
                    "shannon_loss_index": float(shannon_loss_index),
                    "objective_value": float(objective_value),
                }
            )

        layer_option_table.append(
            {
                "layer_id": layer_id,
                "param_name": param_name,
                "num_params": int(weights.numel()),
                "entropy": entropy,
                "options": options,
            }
        )
    return layer_option_table


def prune_frontier(states):
    states = sorted(states, key=lambda item: (item["cost_bits"], item["objective_value"]))
    pruned_states = []
    best_objective = None

    for state in states:
        if best_objective is None or state["objective_value"] < best_objective - 1e-12:
            pruned_states.append(state)
            best_objective = state["objective_value"]
    return pruned_states


def solve_layerwise_bitwidths(layer_option_table, budget_bits):
    states = [
        {
            "cost_bits": 0,
            "objective_value": 0.0,
            "choices": [],
        }
    ]

    for layer_info in layer_option_table:
        next_states = []
        for state in states:
            for option in layer_info["options"]:
                new_cost = state["cost_bits"] + option["cost_bits"]
                if new_cost > budget_bits:
                    continue

                next_states.append(
                    {
                        "cost_bits": new_cost,
                        "objective_value": state["objective_value"] + option["objective_value"],
                        "choices": state["choices"] + [option],
                    }
                )

        if not next_states:
            raise ValueError("Communication budget is too small for the minimum quantization setting.")
        states = prune_frontier(next_states)

    best_state = min(states, key=lambda item: item["objective_value"])
    return best_state


def get_bitwidth_noise_scale(bits, omega_scale):
    q_max = max((2 ** (bits - 1)) - 1, 1)
    return float(omega_scale / float(q_max ** 2))


def build_local_delta(global_state_dict, local_state_dict):
    delta_state = OrderedDict()
    for name, global_tensor in global_state_dict.items():
        local_tensor = local_state_dict[name].detach().cpu()
        delta_state[name] = local_tensor - global_tensor.detach().cpu()
    return delta_state


def compute_gdc_alignment_factors(selected_options, gradient_statistics, omega_scale):
    gamma_map = {}
    for option in selected_options:
        param_name = option["param_name"]
        variance = gradient_statistics[param_name]["variance"]
        norm_sq = gradient_statistics[param_name]["norm_sq"]
        omega = get_bitwidth_noise_scale(option["bits"], omega_scale)
        denominator = variance + omega * norm_sq
        if denominator <= 0 or variance <= 0:
            gamma_value = 1.0
        else:
            gamma_value = float(np.sqrt(variance / denominator))
        gamma_map[param_name] = {
            "gamma": gamma_value,
            "variance": variance,
            "norm_sq": norm_sq,
            "omega": omega,
        }
    return gamma_map


def apply_layerwise_quantization(delta_state_dict, selected_options, gamma_map=None):
    quantized_state = OrderedDict()
    bitwidth_map = {option["param_name"]: option["bits"] for option in selected_options}

    for name, tensor in delta_state_dict.items():
        if name in bitwidth_map:
            quantized_tensor = quantize_tensor(tensor.detach().cpu(), bitwidth_map[name])
            if gamma_map is not None and name in gamma_map:
                quantized_tensor = quantized_tensor * gamma_map[name]["gamma"]
            quantized_state[name] = quantized_tensor
        else:
            quantized_state[name] = tensor.detach().cpu().clone()
    return quantized_state


def reconstruct_state_from_delta(global_state_dict, delta_state_dict):
    reconstructed_state = OrderedDict()
    for name, global_tensor in global_state_dict.items():
        reconstructed_state[name] = global_tensor.detach().cpu() + delta_state_dict[name]
    return reconstructed_state


def compute_quantized_upload_bits(state_dict, selected_options):
    tracked_bitwidths = {option["param_name"]: option["bits"] for option in selected_options}
    total_bits = 0

    for name, tensor in state_dict.items():
        if name in tracked_bitwidths:
            total_bits += int(tensor.numel()) * tracked_bitwidths[name]
        else:
            total_bits += int(tensor.numel()) * 32
    return total_bits


def aggregate_state_dicts(state_dict_list):
    aggregated = OrderedDict()
    if not state_dict_list:
        return aggregated

    first_state = state_dict_list[0]
    for key in first_state.keys():
        reference_tensor = first_state[key]
        if reference_tensor.is_floating_point():
            accumulator = torch.zeros_like(reference_tensor, dtype=torch.float32)
            for state in state_dict_list:
                accumulator += state[key].float()
            aggregated[key] = (accumulator / len(state_dict_list)).type_as(reference_tensor)
        else:
            aggregated[key] = reference_tensor.clone()
    return aggregated


def format_layer_bits(selected_options):
    return " | ".join(
        "{}={}".format(option["layer_id"], option["bits"]) for option in selected_options
    )


def format_layer_metrics(selected_options):
    formatted = []
    for option in selected_options:
        formatted.append(
            "{}({}): bit={} entropy={:.4f} noise={:.4f} sli={:.6f}".format(
                option["layer_id"],
                option["param_name"],
                option["bits"],
                option["entropy"],
                option["noise"],
                option["shannon_loss_index"],
            )
        )
    return "\n".join(formatted)


def format_gdc_metrics(selected_options, gamma_map):
    formatted = []
    for option in selected_options:
        param_name = option["param_name"]
        gdc_stats = gamma_map.get(param_name, None)
        if gdc_stats is None:
            continue
        formatted.append(
            "{}({}): gamma={:.6f} var={:.6e} norm_sq={:.6e} omega={:.6e}".format(
                option["layer_id"],
                param_name,
                gdc_stats["gamma"],
                gdc_stats["variance"],
                gdc_stats["norm_sq"],
                gdc_stats["omega"],
            )
        )
    return "\n".join(formatted)


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

    raise FileNotFoundError(
        "CIFAR-100 dataset not found under '{root}'. "
        "Expected one of: "
        "'{root}/cifar-100-python/', '{root}/cifar-100/', or a directory containing train/test/meta directly. "
        "The script also supports '{root}/.../cifar-100-python.tar.gz'.".format(root=data_root)
    )


def log_run_configuration(logger, args, device, run_dir, log_path, full_model_bits):
    logger.info("=== Run Configuration ===")
    logger.info("run_timestamp: {}".format(RUN_TIMESTAMP))
    logger.info("device: {}".format(device))
    logger.info("run_dir: {}".format(run_dir))
    logger.info("log_path: {}".format(log_path))
    for key in sorted(vars(args).keys()):
        logger.info("{}: {}".format(key, getattr(args, key)))
    logger.info("full_model_size_mbit: {:.4f}".format(full_model_bits / 1e6))


def main():
    total_start_time = time.time()
    args = parse_args()
    run_dir, log_path = setup_paths(args)
    logger = setup_logger(log_path)

    set_seed(args.seed)
    device = select_device()

    train_set, test_set = load_datasets(args.data_root, run_dir)
    train_labels = np.array(train_set.targets)
    client_indices = dirichlet_split_noniid(train_labels, args.alpha, args.num_clients)

    train_loaders = [
        DataLoader(DatasetSplit(train_set, idxs), batch_size=args.batch_size, shuffle=True)
        for idxs in client_indices
    ]
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

    global_model = build_resnet18(num_classes=100).to(device)
    global_weights = copy.deepcopy(global_model.state_dict())

    full_model_bits = get_model_size_bits(global_weights)
    fixed_overhead_bits = get_fixed_overhead_bits(global_weights)
    total_budget_bits = int(full_model_bits * args.communication_budget_ratio)
    tracked_budget_bits = total_budget_bits - fixed_overhead_bits

    if tracked_budget_bits <= 0:
        raise ValueError("Communication budget is too small after accounting for fixed non-quantized tensors.")

    log_run_configuration(logger, args, device, run_dir, log_path, full_model_bits)
    logger.info("--- Fixed Overhead Size: {:.4f} Mbit ---".format(fixed_overhead_bits / 1e6))
    logger.info("--- Total Budget: {:.4f} Mbit ---".format(total_budget_bits / 1e6))
    logger.info("--- Quantized-Layer Budget: {:.4f} Mbit ---".format(tracked_budget_bits / 1e6))
    logger.info("--- Training Start Time: {} ---".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    for round_idx in range(1, args.global_rounds + 1):
        round_start_time = time.time()
        round_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("\n[Global Round {}]".format(round_idx))
        logger.info("Round Start Time: {}".format(round_start_timestamp))
        selected_clients = np.random.choice(
            range(args.num_clients), args.clients_per_round, replace=False
        )
        logger.info("Selected Clients: {}".format(list(selected_clients)))

        quantized_local_states = []

        for client_id in selected_clients:
            client_model = build_resnet18(num_classes=100).to(device)
            client_model.load_state_dict(copy.deepcopy(global_weights))

            local_state, train_loss, gradient_statistics = client_update(
                client_model=client_model,
                dataloader=train_loaders[client_id],
                local_epochs=args.local_epochs,
                lr=args.lr,
                device=device,
                enable_gdc=args.enable_gdc,
            )
            client_model.load_state_dict(local_state)
            local_test_loss, local_test_acc = evaluate_model(client_model, test_loader, device)

            local_delta = build_local_delta(global_weights, local_state)
            layer_option_table = build_layer_option_table(
                state_dict=local_delta,
                quant_min_bit=args.quant_min_bit,
                quant_max_bit=args.quant_max_bit,
                entropy_bins=args.entropy_bins,
            )
            best_state = solve_layerwise_bitwidths(layer_option_table, tracked_budget_bits)
            selected_options = best_state["choices"]
            if args.enable_gdc:
                gamma_map = compute_gdc_alignment_factors(
                    selected_options=selected_options,
                    gradient_statistics=gradient_statistics,
                    omega_scale=args.omega_scale,
                )
            else:
                gamma_map = {}

            quantized_delta = apply_layerwise_quantization(local_delta, selected_options, gamma_map)
            quantized_upload_state = reconstruct_state_from_delta(global_weights, quantized_delta)
            quantized_upload_bits = compute_quantized_upload_bits(local_delta, selected_options)
            upload_megabytes = quantized_upload_bits / 8.0 / 1024.0 / 1024.0
            upload_time_seconds = quantized_upload_bits / (args.upload_rate_mbps * 1e6)

            quantized_local_states.append(quantized_upload_state)

            logger.info(
                "  Client {} | Train Loss: {:.4f} | Test Loss: {:.4f} | Test Acc: {:.2f}%".format(
                    client_id, train_loss, local_test_loss, local_test_acc
                )
            )
            logger.info("    Layer Bitwidths | {}".format(format_layer_bits(selected_options)))
            logger.info("    Upload Size | {:.4f} Mbit | {:.4f} MB".format(
                quantized_upload_bits / 1e6, upload_megabytes
            ))
            logger.info("    Upload Time @ {:.2f} Mbps | {:.4f} s".format(
                args.upload_rate_mbps, upload_time_seconds
            ))
        aggregated_weights = aggregate_state_dicts(quantized_local_states)
        global_weights = aggregated_weights
        global_model.load_state_dict(global_weights)

        global_test_loss, global_test_acc = evaluate_model(global_model, test_loader, device)
        logger.info(
            "Global Aggregation | Test Loss: {:.4f} | Test Acc: {:.2f}%".format(
                global_test_loss, global_test_acc
            )
        )
        round_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        round_elapsed_time = time.time() - round_start_time
        logger.info("Round End Time: {}".format(round_end_timestamp))
        logger.info("Round Duration: {:.2f} s | {:.2f} min".format(
            round_elapsed_time, round_elapsed_time / 60.0
        ))

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    logger.info("\nTraining completed.")
    logger.info("Training End Time: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    logger.info("Total Training Duration: {:.2f} s | {:.2f} min | {:.2f} h".format(
        total_elapsed_time,
        total_elapsed_time / 60.0,
        total_elapsed_time / 3600.0,
    ))
    logger.info("Log saved to: {}".format(log_path))


if __name__ == "__main__":
    main()



# python3 /Users/chentonghao/WASA/layerquant_cifar100.py \
#   --data-root /autodl-pub/data \
#   --num-clients 10 \
#   --clients-per-round 5 \
#   --communication-budget-ratio 0.2 \
#   --upload-rate-mbps 2 \
#   --local-epochs 2 \
#   --global-rounds 55 \
#   --alpha 0.5 \
#   --lr 0.01 \
#   --batch-size 32 \
#   --test-batch-size 128 \
#   --quant-min-bit 2 \
#   --quant-max-bit 8 \
#   --omega-scale 1.0 \
#   --enable-gdc
