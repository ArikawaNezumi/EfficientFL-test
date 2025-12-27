#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from models.Nets import CNNMnist, CNNCifar, MLP  # 新增导入，用于创建模型实例
import torch.quantization  # 新增导入


def FedAvg(w_locals, net_glob, args):
    """
    此函数现在是处理量化模型聚合的总入口。
    它接收量化后的 state_dicts，将其反量化，然后进行平均。
    """
    # 步骤1: 反量化所有客户端的模型
    w_locals_dequantized = []

    # 确定模型类型以创建实例
    if args.model == 'cnn' and args.dataset == 'cifar':
        ModelClass = CNNCifar
    elif args.model == 'cnn' and args.dataset == 'mnist':
        ModelClass = CNNMnist
    elif args.model == 'mlp':
        # MLP需要额外参数，这里简化处理或需要传递更多信息
        # 假设我们主要对CNN做量化
        # 如果也要支持MLP，需要正确获取其输入维度
        img_size = (1, 28, 28) if args.dataset == 'mnist' else (3, 32, 32)
        len_in = 1
        for x in img_size:
            len_in *= x
        # 用 lambda 包装起来，使其调用方式与CNN一致
        ModelClass = lambda args: MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model in FedAvg')

    for w_q in w_locals:
        # 创建一个浮点模型实例用于加载反量化后的权重
        model_float = ModelClass(args=args)

        # 创建一个量化模型实例的骨架
        # 注意：这里我们不需要完整的量化函数，因为我们只是用它来加载state_dict
        model_quantized = ModelClass(args=args)
        # 动态量化只影响特定层，我们只需确保模型结构一致
        if args.model == 'cnn':  # 通常只对CNN的线性层量化
            model_quantized = torch.quantization.quantize_dynamic(
                model=model_quantized, qconfig_spec={nn.Linear}, dtype=torch.qint8
            )

        # 加载客户端上传的量化state_dict
        model_quantized.load_state_dict(w_q)

        # 将权重从量化模型加载到浮点模型（PyTorch自动处理反量化）
        model_float.load_state_dict(model_quantized.state_dict())

        # 将反量化后的浮点state_dict存起来
        w_locals_dequantized.append(model_float.state_dict())

    # 步骤2: 对反量化后的浮点模型进行联邦平均
    return FedAvg_float(w_locals_dequantized)


def FedAvg_float(w):
    """
    原始的联邦平均函数，只处理浮点权重。
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg