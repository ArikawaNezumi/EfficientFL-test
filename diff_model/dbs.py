import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_utils import sym_quantize

class BitWidthSearcher:
    def __init__(self, model, candidate_bits=[2, 4, 8]):
        self.model = model
        self.candidate_bits = candidate_bits
        self.device = next(model.parameters()).device

        self.target_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.target_layers[name] = module

        self.alphas = nn.ParameterDict()
        for name in self.target_layers.keys():
            # 初始化为0，代表概率均等
            self.alphas[name] = nn.Parameter(torch.zeros(len(candidate_bits)).to(self.device))

    # --- 修改点 1: 参数签名改为接收 input_data 和 target ---
    def search(self, input_data, target, bandwidth_limit, lambda_coeff=2.0, iterations=10):
        """
        执行微型搜索
        """
        # 使用 Adam 优化 Alpha，收敛快
        optimizer = torch.optim.Adam(self.alphas.values(), lr=0.1)

        # 确保模型处于 Eval 模式
        self.model.eval()

        # --- 修改点 2: 直接使用传入的 Tensor，不再需要 next(iter(...)) ---
        data = input_data.to(self.device)
        target = target.to(self.device)

        for i in range(iterations):
            optimizer.zero_grad()

            total_expected_size = 0
            original_weights = {}

            # --- 1. 构建虚拟权重 (Soft Composition) ---
            for name, module in self.target_layers.items():
                probs = F.softmax(self.alphas[name], dim=0)

                original_weights[name] = module.weight
                w_orig = module.weight.detach()  # 冻结原始权重

                w_virtual = 0
                layer_exp_bits = 0

                for idx, bit in enumerate(self.candidate_bits):
                    w_q = sym_quantize(w_orig, bit)
                    w_virtual += probs[idx] * w_q
                    layer_exp_bits += probs[idx] * bit

                # 累加期望体积
                total_expected_size += layer_exp_bits * w_orig.numel()

                # Monkey Patch: 临时替换权重
                del module._parameters['weight']
                module.weight = w_virtual

            # --- 2. 前向传播 ---
            output = self.model(data)

            # --- 3. 计算 Loss ---
            loss_task = F.cross_entropy(output, target)

            # 带宽约束 Loss
            size_excess = F.relu(total_expected_size - bandwidth_limit)
            loss_bw = lambda_coeff * (size_excess / (bandwidth_limit + 1e-6))

            total_loss = loss_task + loss_bw

            # --- 4. 更新 Alpha ---
            total_loss.backward()

            # --- 5. 还原现场 ---
            for name, module in self.target_layers.items():
                del module.weight
                module._parameters['weight'] = original_weights[name]

            optimizer.step()

        # --- 6. 生成最终策略 ---
        final_config = {}
        for name, alpha in self.alphas.items():
            best_idx = torch.argmax(alpha).item()
            final_config[name] = self.candidate_bits[best_idx]

        return final_config


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from quant_utils import sym_quantize
# import logging
#
#
# class BitWidthSearcher:
#     def __init__(self, model, candidate_bits=[2, 4, 8]):
#         self.model = model
#         self.candidate_bits = candidate_bits
#         self.device = next(model.parameters()).device
#
#         self.target_layers = {}
#         for name, module in model.named_modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 self.target_layers[name] = module
#
#         self.alphas = nn.ParameterDict()
#         for name in self.target_layers.keys():
#             # 初始化为0，代表概率均等
#             self.alphas[name] = nn.Parameter(torch.zeros(len(candidate_bits)).to(self.device))
#
#     def search(self, input_data, target, bandwidth_limit, lambda_coeff=2.0, iterations=10):
#         """
#         执行微型搜索，加入 ERAT (Entropy-Rate Adaptive Termination) 机制
#         """
#         # ERAT 参数配置
#         theta_abs = 0.4  # 绝对熵阈值
#         theta_rel = 0.04  # 相对变化率阈值 (0.5%)
#         patience = 3  # 稳定窗口大小
#
#         optimizer = torch.optim.Adam(self.alphas.values(), lr=0.1)
#         self.model.eval()
#
#         data = input_data.to(self.device)
#         target = target.to(self.device)
#
#         entropy_history = []  # 记录每一轮的平均熵
#
#         for i in range(iterations):
#             optimizer.zero_grad()
#
#             total_expected_size = 0
#             original_weights = {}
#
#             # --- 1. 构建虚拟权重 (Soft Composition) ---
#             for name, module in self.target_layers.items():
#                 probs = F.softmax(self.alphas[name], dim=0)
#
#                 original_weights[name] = module.weight
#                 w_orig = module.weight.detach()
#
#                 w_virtual = 0
#                 layer_exp_bits = 0
#
#                 for idx, bit in enumerate(self.candidate_bits):
#                     w_q = sym_quantize(w_orig, bit)
#                     w_virtual += probs[idx] * w_q
#                     layer_exp_bits += probs[idx] * bit
#
#                 total_expected_size += layer_exp_bits * w_orig.numel()
#
#                 del module._parameters['weight']
#                 module.weight = w_virtual
#
#             # --- 2. 前向传播 & Loss ---
#             output = self.model(data)
#             loss_task = F.cross_entropy(output, target)
#
#             size_excess = F.relu(total_expected_size - bandwidth_limit)
#             loss_bw = lambda_coeff * (size_excess / (bandwidth_limit + 1e-6))
#
#             total_loss = loss_task + 1.2 * loss_bw
#             total_loss.backward()
#
#             # --- 3. 还原现场 ---
#             for name, module in self.target_layers.items():
#                 del module.weight
#                 module._parameters['weight'] = original_weights[name]
#
#             optimizer.step()
#
#             # ==========================================
#             # ERAT: 熵率自适应终止检测
#             # ==========================================
#             with torch.no_grad():
#                 current_entropies = []
#                 for alpha in self.alphas.values():
#                     probs = F.softmax(alpha, dim=0)
#                     # H(p) = - sum(p * log(p))
#                     entropy = -torch.sum(probs * torch.log(probs + 1e-9))
#                     current_entropies.append(entropy.item())
#
#                 # 计算全网平均熵 H_bar(t)
#                 avg_entropy = sum(current_entropies) / len(current_entropies)
#                 entropy_history.append(avg_entropy)
#
#                 # 1. 确定性收敛检测 (Confidence Convergence)
#                 if avg_entropy < theta_abs:
#                     # logging.info(f"    [DBS] Converged at iter {i+1} (Entropy {avg_entropy:.4f} < {theta_abs})")
#                     break
#
#                 # 2. 平稳性收敛检测 (Stability Convergence)
#                 if len(entropy_history) >= patience + 1:
#                     is_stable = True
#                     for k in range(1, patience + 1):
#                         h_t = entropy_history[-k]
#                         h_prev = entropy_history[-(k + 1)]
#                         # 计算变化率 delta_H
#                         change_rate = abs(h_t - h_prev) / (h_prev + 1e-9)
#                         if change_rate >= theta_rel:
#                             is_stable = False
#                             break
#
#                     if is_stable:
#                         # logging.info(f"    [DBS] Stabilized at iter {i+1} (Change Rate < {theta_rel} for {patience} steps)")
#                         break
#
#         # --- 生成最终策略 ---
#         final_config = {}
#         for name, alpha in self.alphas.items():
#             best_idx = torch.argmax(alpha).item()
#             final_config[name] = self.candidate_bits[best_idx]
#
#         return final_config, i + 1