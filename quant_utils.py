import torch


def sym_quantize(tensor, bits):
    """模拟对称量化 (Fake Quantization)"""
    if bits >= 32:
        return tensor

    levels = 2 ** bits - 1
    abs_max = tensor.abs().max()
    scale = abs_max / (levels / 2) + 1e-6

    # Quantize + Dequantize
    # 这一步是为了模拟量化带来的数值误差
    tensor_q = torch.round(tensor / scale).clamp(-levels / 2, levels / 2)
    tensor_deq = tensor_q * scale

    return tensor_deq


def get_model_size_bits(model):
    """计算模型 FP32 下的总比特数"""
    total = 0
    # 这里接收的是 nn.Module 对象
    for p in model.parameters():
        total += p.numel() * 32
    return total