import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# --- 简单的量化工具函数 (你可能有自己的 quant_utils，这里仅作占位) ---
def sym_quantize(x, bits):
    if bits >= 32: return x
    levels = 2 ** bits - 1
    # 简单的 MinMax 量化，实际可替换为更复杂的
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (levels + 1e-8)
    zero_point = -min_val / (scale + 1e-8)
    x_q = torch.round(x / (scale + 1e-8) + zero_point)
    x_q = torch.clamp(x_q, 0, levels)
    x_deq = (x_q - zero_point) * scale
    # STE (Straight-Through Estimator)
    return (x_deq - x).detach() + x


class ClientState:
    """
    维护每个客户端的状态，用于 EF 和 Staleness
    """

    def __init__(self, model):
        self.last_participated_round = -1
        # 针对每个可训练层维护一个残差 Buffer
        self.error_buffer = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 初始化为 0
                self.error_buffer[name] = torch.zeros_like(module.weight).detach()


class BitWidthSearcher:
    def __init__(self, model, client_state, current_round,
                 candidate_bits=[2, 4, 8],
                 enable_ef=True,
                 enable_staleness=True,
                 decay_lambda=0.9):

        self.model = model
        self.client_state = client_state
        self.current_round = current_round
        self.candidate_bits = candidate_bits

        # --- 开关控制 ---
        self.enable_ef = enable_ef
        self.enable_staleness = enable_staleness
        self.decay_lambda = decay_lambda

        self.device = next(model.parameters()).device

        self.target_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.target_layers[name] = module

        self.alphas = nn.ParameterDict()
        for name in self.target_layers.keys():
            # 初始化为0，代表概率均等
            self.alphas[name] = nn.Parameter(torch.zeros(len(candidate_bits)).to(self.device))

    def _calculate_decay_factor(self):
        """
        计算时间衰减因子 Gamma
        """
        if not self.enable_staleness:
            return 1.0

        last_round = self.client_state.last_participated_round
        if last_round == -1:
            staleness = 1
        else:
            staleness = self.current_round - last_round

        # Gamma = lambda ^ (tau - 1)
        gamma = self.decay_lambda ** (staleness - 1)
        return gamma

    def search_and_quantize(self, input_data, target, bandwidth_limit, lambda_coeff=2.0, iterations=5):
        """
        1. 执行微型搜索 (Micro-Search)
        2. 应用最优策略进行量化
        3. 更新误差残差
        返回: 压缩后的模型权重字典 (用于上传)
        """
        optimizer = torch.optim.Adam(self.alphas.values(), lr=0.1)
        self.model.eval()

        data = input_data.to(self.device)
        target = target.to(self.device)

        # 1. 计算衰减系数
        gamma = self._calculate_decay_factor()

        # -----------------------------------------------------------
        # Phase 1: Search (寻找最优 Alpha)
        # -----------------------------------------------------------
        for i in range(iterations):
            optimizer.zero_grad()
            total_expected_size = 0

            # 临时保存原始权重，用于搜索后的恢复
            temp_weights = {}

            for name, module in self.target_layers.items():
                probs = F.softmax(self.alphas[name], dim=0)

                # 获取当前梯度 (模拟) 和 历史残差
                # 注意：DBS 通常是在 forward 过程中对 weight 进行量化模拟
                # 这里我们需要把 Error 加到 Weight 上进行搜索吗？
                # 答：是的。因为 Weight Update = -lr * Gradient
                # Quantized(Update) = Quantized(-lr * G + e)
                # 所以我们是对 (Weight_new - Weight_old + e) 进行量化
                # 但为了简化 DBS 逻辑，通常直接对当前 Weight 进行量化搜索是等价的近似

                w_current = module.weight.detach()
                e_old = self.client_state.error_buffer[name].to(self.device)

                # *** 关键点: 修正后的待量化目标 ***
                # 如果开启 EF，我们搜索的目标不是原始权重，而是 "权重 + 衰减后的残差"
                if self.enable_ef:
                    w_target = w_current + gamma * e_old
                else:
                    w_target = w_current

                w_virtual = 0
                layer_exp_bits = 0

                # 软量化混合
                for idx, bit in enumerate(self.candidate_bits):
                    w_q = sym_quantize(w_target, bit)
                    w_virtual += probs[idx] * w_q
                    layer_exp_bits += probs[idx] * bit

                total_expected_size += layer_exp_bits * w_target.numel()

                # 临时替换
                temp_weights[name] = module.weight
                del module._parameters['weight']
                module.weight = w_virtual

            # 前向 & Loss
            output = self.model(data)
            loss_task = F.cross_entropy(output, target)

            size_excess = F.relu(total_expected_size - bandwidth_limit)
            loss_bw = lambda_coeff * (size_excess / (bandwidth_limit + 1e-6))

            total_loss = loss_task + loss_bw
            total_loss.backward()

            # 还原
            for name, module in self.target_layers.items():
                del module.weight
                module._parameters['weight'] = temp_weights[name]

            optimizer.step()

        # -----------------------------------------------------------
        # Phase 2: Apply & Update Residual (应用策略并更新残差)
        # -----------------------------------------------------------
        final_quantized_weights = {}

        with torch.no_grad():
            for name, module in self.target_layers.items():
                # 1. 确定最优位宽
                best_idx = torch.argmax(self.alphas[name]).item()
                best_bit = self.candidate_bits[best_idx]

                w_current = module.weight.detach()
                e_old = self.client_state.error_buffer[name].to(self.device)

                # 2. 计算修正后的目标权重
                if self.enable_ef:
                    w_corrected = w_current + gamma * e_old
                else:
                    w_corrected = w_current

                # 3. 执行最终量化
                w_quantized = sym_quantize(w_corrected, best_bit)

                # 4. 更新残差 (Error Feedback)
                # e_new = w_corrected - w_quantized
                if self.enable_ef:
                    new_error = w_corrected - w_quantized
                    # 更新 Buffer (存回 CPU 节省显存，或者保持在 GPU)
                    self.client_state.error_buffer[name] = new_error.detach()

                final_quantized_weights[name] = w_quantized.cpu()

        # 更新客户端状态
        self.client_state.last_participated_round = self.current_round

        return final_quantized_weights

# --- 使用示例 ---
# 1. 初始化 (在 Server 端或 Client 持久化存储)
# client_states = [ClientState(model) for _ in range(num_clients)]

# 2. 在某一轮训练中
# searcher = BitWidthSearcher(
#     model=local_model,
#     client_state=client_states[client_idx],
#     current_round=round_idx,
#     enable_ef=True,        # 开启误差反馈
#     enable_staleness=True  # 开启陈旧性衰减
# )

# quantized_dict = searcher.search_and_quantize(data_batch, label_batch, bw_limit)