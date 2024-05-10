"""FashionMNIST dataset utilities for federated learning."""


from typing import Optional, Tuple, Any, List

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST


# 1.每个客户端包含所有类别但是每个类别的数据量不同
def load_datasets_one(  # pylint: disable=too-many-arguments
    num_clients: int = 10,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,  # 随机数种子
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # 定义FashionMNIST数据集
    train_dataset = FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST('./data', train=False, download=True, transform=transform)

    # 定义每个类别的样本索引列表和样本数量
    class_indices = [[] for _ in range(10)]
    class_sizes = [0 for _ in range(10)]
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
        class_sizes[label] += 1

    # 定义每个客户端的样本索引列表和每个类别的样本数量
    client_indices = []
    num_samples_per_class = []  # 每个客户端的每个类别的数量
    for i in range(num_clients):
        # 对于每个客户端，从每个类别的样本中随机采样不同数量的样本，使得每个客户端包含所有类别但是每个类别的数据量不同
        indices = []
        samples_per_class = []
        for j in range(10):
            # 随机采样不同数量的样本，每个客户端的每个类的样本数量为 每类样本总数/客户端数
            np.random.seed(seed + i * 10 + j)
            num_samples = int(class_sizes[j] / num_clients) + np.random.randint(-int(class_sizes[j] / (2 * num_clients)), int(class_sizes[j] / (2 * num_clients)) + 1)
            sampled_indices = np.random.choice(class_indices[j], num_samples, replace=False)
            indices.extend(sampled_indices.tolist())
            samples_per_class.append(num_samples)
        client_indices.append(indices)
        num_samples_per_class.append(samples_per_class)

    # 创建客户端数据集
    client_datasets = []
    client_validates = []
    for indices in client_indices:
        client_subset = Subset(train_dataset, indices)
        client_dataset = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_datasets.append(client_dataset)
    client_validates = client_datasets
    return client_datasets, client_validates, DataLoader(test_dataset, batch_size=batch_size)


# 2.每个类别的数据量相同，但是不同客户端包含的类别数量不同
def load_datasets_two(  # pylint: disable=too-many-arguments
    num_clients: int = 10,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,  # 随机数种子
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # 定义FashionMNIST数据集
    train_dataset = FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST('./data', train=False, download=True, transform=transform)

    # 定义每个类别的样本索引列表和样本数量
    class_indices = [[] for _ in range(10)]
    class_sizes = [0 for _ in range(10)]
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
        class_sizes[label] += 1

    # 每个客户端的最多类别数目
    num_classes_per_client = 10

    # 定义每个客户端的样本索引列表和每个类别的样本数量
    client_indices = []
    num_samples_per_class = []  # 每个客户端的每个类别的数量
    for i in range(num_clients):
        # 随机选择一定数量的类别
        np.random.seed(seed + i)
        num_classes = np.random.randint(1, num_classes_per_client + 1)
        selected_classes = np.random.choice(range(10), num_classes, replace=False)
        indices = []
        samples_per_class = []
        for j in selected_classes:
            # 随机采样相同数量的样本
            np.random.seed(seed + i * 10 + j)
            num_samples = int(class_sizes[j] / num_clients)
            sampled_indices = np.random.choice(class_indices[j], num_samples, replace=False)
            indices.extend(sampled_indices.tolist())
            samples_per_class.append(num_samples)
        client_indices.append(indices)
        num_samples_per_class.append(samples_per_class)

    # 创建客户端数据集
    client_datasets = []
    client_validates = []
    for indices in client_indices:
        client_subset = Subset(train_dataset, indices)
        client_dataset = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_datasets.append(client_dataset)

    return client_datasets, client_datasets, DataLoader(test_dataset, batch_size=batch_size)


# 3.每个客户端的类别数量与每个类别的样本数量均不同，采用随机的方式进行分配
def load_datasets_three(  # pylint: disable=too-many-arguments
    num_clients: int = 10,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,  # 随机数种子
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # 定义FashionMNIST数据集
    train_dataset = FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST('./data', train=False, download=True, transform=transform)

    # 定义每个类别的样本索引列表和样本数量
    class_indices = [[] for _ in range(10)]
    class_sizes = [0 for _ in range(10)]
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
        class_sizes[label] += 1

    # 每个客户端的最多类别数目
    num_classes_per_client = 10

    # 定义每个客户端的样本索引列表和每个类别的样本数量
    client_indices = []
    num_samples_per_class = []  # 每个客户端的每个类别的数量
    for i in range(num_clients):
        # 随机选择一定数量的类别
        np.random.seed(seed + i)
        num_classes = np.random.randint(1, num_classes_per_client + 1)
        selected_classes = np.random.choice(range(10), num_classes, replace=False)
        indices = []
        samples_per_class = []
        for j in selected_classes:
            # 随机采样不同数量的样本
            np.random.seed(seed + i * 10 + j)
            num_samples = int(class_sizes[j] / num_clients) + np.random.randint(
                -int(class_sizes[j] / (2 * num_clients)), int(class_sizes[j] / (2 * num_clients)) + 1)
            sampled_indices = np.random.choice(class_indices[j], num_samples, replace=False)
            indices.extend(sampled_indices.tolist())
            samples_per_class.append(num_samples)
        client_indices.append(indices)
        num_samples_per_class.append(samples_per_class)

    # 创建客户端数据集
    client_datasets = []
    client_validates = []
    for indices in client_indices:
        client_subset = Subset(train_dataset, indices)
        client_dataset = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        client_datasets.append(client_dataset)

    return client_datasets, client_datasets, DataLoader(test_dataset, batch_size=batch_size)


# 增加验证集
def load_datasets_four(
    num_clients: int = 10,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader[Any]], List[DataLoader[Any]], DataLoader[Any]]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # 定义FashionMNIST数据集
    train_dataset = FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = FashionMNIST('./data', train=False, download=True, transform=transform)

    # 定义每个类别的样本索引列表和样本数量
    class_indices = [[] for _ in range(10)]
    class_sizes = [0 for _ in range(10)]
    for i, (_, label) in enumerate(train_dataset):
        class_indices[label].append(i)
        class_sizes[label] += 1

    # 定义每个客户端的样本索引列表和每个类别的样本数量
    client_indices = []
    num_samples_per_class = []  # 每个客户端的每个类别的数量
    for i in range(num_clients):
        # 对于每个客户端，从每个类别的样本中随机采样不同数量的样本，使得每个客户端包含所有类别但是每个类别的数据量不同
        indices = []
        samples_per_class = []
        for j in range(10):
            # 随机采样不同数量的样本，每个客户端的每个类的样本数量为 每类样本总数/客户端数
            np.random.seed(seed + i * 10 + j)
            num_samples = int(class_sizes[j] / num_clients) + np.random.randint(-int(class_sizes[j] / (2 * num_clients)), int(class_sizes[j] / (2 * num_clients)) + 1)
            sampled_indices = np.random.choice(class_indices[j], num_samples, replace=False)
            indices.extend(sampled_indices.tolist())
            samples_per_class.append(num_samples)
        client_indices.append(indices)
        num_samples_per_class.append(samples_per_class)

    # 创建客户端数据集
    trainloaders = []
    valloaders = []
    for indices, samples_per_class in zip(client_indices, num_samples_per_class):
        # 拆分数据集为训练集和验证集
        train_indices = []
        val_indices = []
        for j in range(10):
            num_samples = samples_per_class[j]
            sampled_indices = indices[j * num_samples:(j + 1) * num_samples]
            np.random.shuffle(sampled_indices)
            val_size = int(0.2 * num_samples)
            val_indices.extend(sampled_indices[:val_size])
            train_indices.extend(sampled_indices[val_size:])
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
        trainloaders.append(trainloader)
        valloaders.append(valloader)

    testloader = DataLoader(test_dataset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
