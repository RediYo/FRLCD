import random
from typing import Any, List, Tuple, Optional

import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_datasets(
    num_clients: int = 10,
    partition: int = 1,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42
) -> Tuple[List[DataLoader[Any]], List[DataLoader[Any]], DataLoader[Any]]:
    """
    加载FashionMNIST数据集，并将其分成训练集、验证集和测试集，然后将训练集和验证集分成num_clients份，每份分配给一个客户端。
    Args:
        num_clients (int): 客户端的数量，默认为10。
        partition (int): 数据集分割方式，1表示每个客户端都包含所有类别，但是每个类别的样本数量可能不同，2表示每个客户端只包含部分类别，但是每个类别的样本数量相同，其他值表示每个客户端的类别和样本数量都是随机的，默认为1。
        val_ratio (float): 验证集的比例，默认为0.1。
        batch_size (Optional[int]): DataLoader中每个batch的大小，默认为32。
        seed (Optional[int]): 随机数种子，用于重现实验结果，默认为42。
    Returns:
        Tuple[List[DataLoader[Any]], List[DataLoader[Any]], DataLoader[Any]]: 返回一个元组，包含训练数据集、验证数据集和测试数据集。
    """

    # 定义要应用于FashionMNIST数据集的变换
    # 定义数据预处理操作
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载FashionMNIST数据集，并将其分成训练集和测试集
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 设置随机数种子，重复实验
    np.random.seed(seed)
    random.seed(seed)

    # 将训练集分成num_clients份，分别分配给每个客户端
    clients_train_data = [None] * num_clients
    if partition == 1:
        # 如果partition为1，将数据集分成每个客户端都包含所有类别，但是每个类别的样本数量可能不同
        # 将训练集按类别划分
        class_indices = [[] for _ in range(10)]
        for i in range(len(trainset)):
            label = trainset.targets[i]
            class_indices[label].append(i)
        # 获取每个类别的样本数量
        class_counts = [len(class_indices[j]) for j in range(10)]
        print(class_counts)
        # 为每个客户端分配样本
        for i in range(num_clients):
            client_data_idx = []  # 定义一个空列表，用于存储将要分配给该客户端的样本索引

            for j in range(10):  # 对于每个类别j，循环10次
                # 计算类别j分配给该客户端的样本数量，保证每个类别的样本数在1/(2*num_clients)和1/num_clients之间随机分配, 300 - 600
                num_samples_per_class = random.randint(class_counts[j] // (2 * num_clients),
                                                       class_counts[j] // num_clients)
                print(f"num_samples_per_class:{num_samples_per_class}")
                # 随机选择num_samples_per_class个样本索引添加到client_data_idx列表中，使用min避免因数据量不够采样失败
                client_data_idx += random.sample(class_indices[j], min(num_samples_per_class, len(class_indices[j])))
                # 从class_indices[j]列表中删除已经分配的样本的索引，避免下一个客户端重复使用相同的样本
                class_indices[j] = list(set(class_indices[j]) - set(client_data_idx))
                print(f"len:{j}-{len(class_indices[j])}")
            # 将该客户端的数据存储为DataLoader的子集
            clients_train_data[i] = data.Subset(trainset, indices=client_data_idx)
    elif partition == 2:
        # 如果partition为2，将数据集分成每个客户端只包含部分类别，但是每个类别的样本数量相同
        # 将训练集按类别划分
        class_indices = [[] for _ in range(10)]
        for i in range(len(trainset)):
            label = trainset.targets[i]
            class_indices[label].append(i)
        # 获取每个类别的样本数量
        class_counts = [len(class_indices[j]) for j in range(10)]
        print(class_counts)
        # 为每个客户端分配样本
        for i in range(num_clients):
            client_data_idx = []
            num_classes = np.random.randint(low=5, high=11)  # 每个客户端随机选择5-10个类别
            classes = np.random.choice(range(10), num_classes, replace=False)  # 随机选择num_classes个类别
            print(f"num_classes:{classes}")
            for j in classes:
                # 对于每个类别，将class_counts[j]/num_clients个样本分配给该客户端
                client_data_idx += class_indices[j][:class_counts[j]//num_clients]
                print(f"num_samples_per_class:{class_counts[j]/num_clients}")
                # 从class_indices[j]列表中删除已经分配的样本的索引，避免下一个客户端重复使用相同的样本
                class_indices[j] = list(set(class_indices[j]) - set(client_data_idx))
                print(f"len:{j}-{len(class_indices[j])}")
            # 将该客户端的数据存储为DataLoader的子集
            clients_train_data[i] = data.Subset(trainset, indices=client_data_idx)
    else:
        # 如果partition为其他值，将数据集分成每个客户端的类别和样本数量都是随机的
        # 将训练集按类别划分
        class_indices = [[] for _ in range(10)]
        for i in range(len(trainset)):
            label = trainset.targets[i]
            class_indices[label].append(i)
        # 获取每个类别的样本数量
        class_counts = [len(class_indices[j]) for j in range(10)]
        print(class_counts)
        # 随机分配类别和样本数量给每个客户端
        for i in range(num_clients):
            client_data_idx = []
            num_classes = np.random.randint(low=5, high=11)  # 每个客户端随机选择5-10个类别
            classes = np.random.choice(range(10), num_classes, replace=False)  # 随机选择num_classes个类别
            print(f"num_classes:{classes}")
            for j in classes:
                # 计算类别j分配给该客户端的样本数量，保证每个类别的样本数在1/(2*num_clients)和1/num_clients之间随机分配, 300 - 600
                num_samples_per_class = random.randint(class_counts[j] // (2 * num_clients),
                                                       class_counts[j] // num_clients)
                print(f"num_samples_per_class:{num_samples_per_class}")
                # 随机选择num_samples_per_class个样本索引添加到client_data_idx列表中，使用min避免因数据量不够采样失败
                client_data_idx += random.sample(class_indices[j], min(num_samples_per_class, len(class_indices[j])))
                # 从class_indices[j]列表中删除已经分配的样本的索引，避免下一个客户端重复使用相同的样本
                class_indices[j] = list(set(class_indices[j]) - set(client_data_idx))
                print(f"len:{j}-{len(class_indices[j])}")
            # 将该客户端的数据存储为DataLoader的子集
            clients_train_data[i] = data.Subset(trainset, indices=client_data_idx)

    # 将每个客户端的训练数据分为训练集和验证集
    client_train_data_loader = []
    client_val_data_loader = []
    for client_data in clients_train_data:
        num_train = len(client_data)
        indices = list(range(num_train))
        split = int(np.floor(val_ratio * num_train))
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
        # 采样器，sampler和shuffle参数只能同时指定一个
        train_sampler = data.SubsetRandomSampler(train_idx)
        valid_sampler = data.SubsetRandomSampler(valid_idx)
        client_train_data_loader.append(DataLoader(client_data, batch_size=batch_size, sampler=train_sampler))
        client_val_data_loader.append(DataLoader(client_data, batch_size=batch_size, sampler=valid_sampler))

    # 定义测试数据集的DataLoader
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return client_train_data_loader, client_val_data_loader, test_loader
