#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import copy
# import numpy as np
# from torchvision import datasets, transforms
# import torch
#
# from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
# from utils.options import args_parser
# from models.Update import LocalUpdate
# from models.Nets import MLP, CNNMnist, CNNCifar
# from models.Fed import FedAvg
# from models.test import test_img
#
#
# if __name__ == '__main__':
#     # parse args
#     args = args_parser()
#     args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#
#     # load dataset and split users
#     if args.dataset == 'mnist':
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
#         dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
#         # sample users
#         if args.iid:
#             dict_users = mnist_iid(dataset_train, args.num_users)
#         else:
#             dict_users = mnist_noniid(dataset_train, args.num_users)
#     elif args.dataset == 'cifar':
#         trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
#         dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
#         if args.iid:
#             dict_users = cifar_iid(dataset_train, args.num_users)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')
#     else:
#         exit('Error: unrecognized dataset')
#     img_size = dataset_train[0][0].shape
#
#     # build model
#     if args.model == 'cnn' and args.dataset == 'cifar':
#         net_glob = CNNCifar(args=args).to(args.device)
#     elif args.model == 'cnn' and args.dataset == 'mnist':
#         net_glob = CNNMnist(args=args).to(args.device)
#     elif args.model == 'mlp':
#         len_in = 1
#         for x in img_size:
#             len_in *= x
#         net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
#     else:
#         exit('Error: unrecognized model')
#     print(net_glob)
#     net_glob.train()
#
#     # copy weights
#     w_glob = net_glob.state_dict()
#
#     # training
#     loss_train = []
#     cv_loss, cv_acc = [], []
#     val_loss_pre, counter = 0, 0
#     net_best = None
#     best_loss = None
#     val_acc_list, net_list = [], []
#
#     if args.all_clients:
#         print("Aggregation over all clients")
#         w_locals = [w_glob for i in range(args.num_users)]
#     for iter in range(args.epochs):
#         loss_locals = []
#         if not args.all_clients:
#             w_locals = []
#         m = max(int(args.frac * args.num_users), 1)
#         idxs_users = np.random.choice(range(args.num_users), m, replace=False)
#         for idx in idxs_users:
#             local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
#             w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
#             if args.all_clients:
#                 w_locals[idx] = copy.deepcopy(w)
#             else:
#                 w_locals.append(copy.deepcopy(w))
#             loss_locals.append(copy.deepcopy(loss))
#         # update global weights
#         w_glob = FedAvg(w_locals)
#
#         # copy weight to net_glob
#         net_glob.load_state_dict(w_glob)
#
#         # print loss
#         loss_avg = sum(loss_locals) / len(loss_locals)
#         print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
#         loss_train.append(loss_avg)
#
#     # plot loss curve
#     plt.figure()
#     plt.plot(range(len(loss_train)), loss_train)
#     plt.ylabel('train_loss')
#     plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
#
#     # testing
#     net_glob.eval()
#     acc_train, loss_train = test_img(net_glob, dataset_train, args)
#     acc_test, loss_test = test_img(net_glob, dataset_test, args)
#     print("Training accuracy: {:.2f}".format(acc_train))
#     print("Testing accuracy: {:.2f}".format(acc_test))


# import matplotlib
#
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import copy
# import numpy as np
# from torchvision import datasets, transforms
# import torch
# # ---- 新增导入 ----
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# # -----------------
#
# from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
# from utils.options import args_parser
# from models.Update import LocalUpdate
# from models.Nets import MLP, CNNMnist, CNNCifar
# from models.Fed import FedAvg
#
#
# # from models.test import test_img  # 我们将用下面的函数替代它
#
#
# # ---- 新增/修改：从 main_nn.py 借鉴过来的评估函数 ----
# def test_img(net_g, dataloader, args):
#     net_g.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(args.device), target.to(args.device)
#             log_probs = net_g(data)
#             test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#             y_pred = log_probs.data.max(1, keepdim=True)[1]
#             correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
#
#     test_loss /= len(dataloader.dataset)
#     accuracy = 100.00 * correct / len(dataloader.dataset)
#     # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#     #     test_loss, correct, len(dataloader.dataset), accuracy))
#     return accuracy.item(), test_loss
#
#
# # ----------------------------------------------------
#
#
# if __name__ == '__main__':
#     # parse args
#     args = args_parser()
#     args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#
#     # load dataset and split users
#     if args.dataset == 'mnist':
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
#         dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
#         # ---- 新增/修改：设置通道数 ----
#         args.num_channels = 1
#         # sample users
#         if args.iid:
#             dict_users = mnist_iid(dataset_train, args.num_users)
#         else:
#             dict_users = mnist_noniid(dataset_train, args.num_users)
#     elif args.dataset == 'cifar':
#         trans_cifar = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
#         dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
#         # ---- 新增/修改：设置通道数 ----
#         args.num_channels = 3
#         if args.iid:
#             dict_users = cifar_iid(dataset_train, args.num_users)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')
#     else:
#         exit('Error: unrecognized dataset')
#     img_size = dataset_train[0][0].shape
#
#     # build model
#     if args.model == 'cnn' and args.dataset == 'cifar':
#         net_glob = CNNCifar(args=args).to(args.device)
#     elif args.model == 'cnn' and args.dataset == 'mnist':
#         net_glob = CNNMnist(args=args).to(args.device)
#     elif args.model == 'mlp':
#         len_in = 1
#         for x in img_size:
#             len_in *= x
#         net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
#     else:
#         exit('Error: unrecognized model')
#     print(net_glob)
#     net_glob.train()
#
#     # copy weights
#     w_glob = net_glob.state_dict()
#
#     # ---- 新增：创建评估用的 DataLoader ----
#     test_loader = DataLoader(dataset_test, batch_size=68, shuffle=False)
#     # 为了计算全局训练集准确率，也创建一个 train_loader
#     train_loader = DataLoader(dataset_train, batch_size=68, shuffle=False)
#     # ------------------------------------
#
#     # training
#     loss_train_list = []
#     # ---- 新增：用于存储每轮准确率的列表 ----
#     acc_train_list, acc_test_list = [], []
#     # ---------------------------------------
#
#     cv_loss, cv_acc = [], []
#     val_loss_pre, counter = 0, 0
#     net_best = None
#     best_loss = None
#     val_acc_list, net_list = [], []
#
#     if args.all_clients:
#         print("Aggregation over all clients")
#         w_locals = [w_glob for i in range(args.num_users)]
#
#     for iter in range(args.epochs):
#         loss_locals = []
#         if not args.all_clients:
#             w_locals = []
#         m = max(int(args.frac * args.num_users), 1)
#         idxs_users = np.random.choice(range(args.num_users), m, replace=False)
#         for idx in idxs_users:
#             local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
#             w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
#             if args.all_clients:
#                 w_locals[idx] = copy.deepcopy(w)
#             else:
#                 w_locals.append(copy.deepcopy(w))
#             loss_locals.append(copy.deepcopy(loss))
#         # update global weights
#         w_glob = FedAvg(w_locals)
#
#         # copy weight to net_glob
#         net_glob.load_state_dict(w_glob)
#
#         # print loss
#         loss_avg = sum(loss_locals) / len(loss_locals)
#         print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
#         loss_train_list.append(loss_avg)
#
#         # ---- 新增/修改：每轮都进行评估 ----
#         net_glob.eval()
#         acc_train, loss_train = test_img(net_glob, train_loader, args)
#         acc_test, loss_test = test_img(net_glob, test_loader, args)
#
#         acc_train_list.append(acc_train)
#         acc_test_list.append(acc_test)
#
#         print("Round {:3d}, Training accuracy: {:.2f}%".format(iter, acc_train))
#         print("Round {:3d}, Testing accuracy: {:.2f}%".format(iter, acc_test))
#         net_glob.train()  # 评估结束后，切回训练模式
#         # -----------------------------------
#
#     # plot loss curve
#     plt.figure()
#     plt.plot(range(len(loss_train_list)), loss_train_list)
#     plt.ylabel('train_loss')
#     plt.savefig(
#         './save/fed_{}_{}_{}_C{}_iid{}_loss.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
#
#     # ---- 新增：绘制准确率曲线 ----
#     plt.figure()
#     plt.plot(range(len(acc_train_list)), acc_train_list, label='Train Acc')
#     plt.plot(range(len(acc_test_list)), acc_test_list, label='Test Acc')
#     plt.title('Accuracy vs. Communication Rounds')
#     plt.ylabel('Accuracy (%)')
#     plt.xlabel('Communication Rounds')
#     plt.legend()
#     plt.savefig(
#         './save/fed_{}_{}_{}_C{}_iid{}_acc.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
#     # ----------------------------
#
#     # testing
#     # ---- 修改：使用我们新的评估流程打印最终结果 ----
#     net_glob.eval()
#     # 这里的 acc_train 和 acc_test 已经是最后一轮的结果了，可以直接使用
#     print("\n--- Final Results ---")
#     print("Final Training accuracy: {:.2f}%".format(acc_train_list[-1]))
#     print("Final Testing accuracy: {:.2f}%".format(acc_test_list[-1]))
#     # ---------------------------------------------


# file: main_fed.py

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.quantization  # 新增导入
import os  # 新增导入，用于计算文件大小

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg  # FedAvg现在是我们的新聚合函数
from models.test import test_img  # 假设你已整合了上一版的test_img函数


# ---- 新增/修改：从上一版借鉴的评估函数 ----
def test_img(net_g, dataloader, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(dataloader.dataset)
    accuracy = 100.00 * correct / len(dataloader.dataset)
    return accuracy.item(), test_loss


# ----------------------------------------------------

# ---- 新增：量化和模型大小计算的辅助函数 ----
def quantize_model_dynamic(model):
    """对模型的线性层进行动态量化"""
    # 我们只对CNN模型中的全连接层进行量化
    # 这是动态量化最常见的应用场景
    quantized_model = torch.quantization.quantize_dynamic(
        model=model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def get_model_size(model):
    """计算模型state_dict存储到磁盘的大小(MB)"""
    torch.save(model.state_dict(), "temp_model.p")
    size = os.path.getsize("temp_model.p") / 1e6  # 转换为MB
    os.remove("temp_model.p")
    return size


# ------------------------------------------


if __name__ == '__main__':
    # ... (前面的代码保持不变, 直到 `w_glob = net_glob.state_dict()`)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        args.num_channels = 3
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # ... (创建DataLoader, list等代码保持不变, 直到主训练循环)
    test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
    train_loader = DataLoader(dataset_train, batch_size=128, shuffle=False)

    loss_train = []
    acc_train_list, acc_test_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # ---- 新增：用于记录本轮通信大小 ----
        total_size_float = 0
        total_size_quantized = 0

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

            # local.train 现在返回训练好的模型对象和损失
            # 修改 LocalUpdate.train() 返回 net 对象而不是 state_dict
            net_trained_local, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # ---- 新增：量化和大小计算 ----
            # 将模型移到CPU进行量化，因为动态量化主要支持CPU
            net_trained_local.to('cpu')

            # 计算原始模型大小
            float_size = get_model_size(net_trained_local)
            total_size_float += float_size

            # 对模型进行量化
            net_quantized_local = quantize_model_dynamic(net_trained_local)

            # 计算量化模型大小
            quantized_size = get_model_size(net_quantized_local)
            total_size_quantized += quantized_size

            # 将量化后的 state_dict "上传"
            w_locals.append(copy.deepcopy(net_quantized_local.state_dict()))
            loss_locals.append(copy.deepcopy(loss))

        # ---- 修改：调用新的聚合函数 ----
        # 将全局模型和args传入，以便在服务器端重建模型
        w_glob = FedAvg(w_locals, net_glob, args)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss and communication savings
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        print('Communication savings this round: {:.2f} MB -> {:.2f} MB ({:.2f}% reduction)'.format(
            total_size_float, total_size_quantized, (1 - total_size_quantized / total_size_float) * 100
        ))
        loss_train.append(loss_avg)

        # ... (后面的评估和画图代码保持不变)
        net_glob.eval()
        acc_train, loss_train_val = test_img(net_glob, train_loader, args)
        acc_test, loss_test = test_img(net_glob, test_loader, args)

        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)

        print("Round {:3d}, Training accuracy: {:.2f}%".format(iter, acc_train))
        print("Round {:3d}, Testing accuracy: {:.2f}%".format(iter, acc_test))
        net_glob.train()

    # ... (最后的画图和测试代码保持不变)