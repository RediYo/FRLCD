import csv
import random
from pathlib import Path
from typing import Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.utils import Bunch
from torch.utils.data import Dataset, DataLoader


def load_datasets(
        num_clients: int = 5,
        val_ratio: float = 0.1,
        batch_size: Optional[int] = 8,
        seed: Optional[int] = 42
) -> Tuple[List[DataLoader[Any]], List[DataLoader[Any]]]:
    """
    加载数据集
    """

    data_path = ["../datasets/BLE-Move/dataset/data"]

    return get_ble_data_dataloader(data_path, num_clients, val_ratio, batch_size, seed)


# 自定义数据集类，继承自Dataset
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y


def split_dataset(dataset, num_clients, seed):
    # 获取数据集大小
    dataset_size = len(dataset.data)
    # 计算每个客户端的数据集大小
    data_per_client = int(dataset_size / num_clients)
    # 设置随机数种子
    random.seed(seed)
    # 生成随机索引
    indices = list(range(dataset_size))
    random.shuffle(indices)
    # 划分数据集
    dataset_list = []
    # print(f"num_clients{num_clients}")
    for i in range(num_clients):
        start_index = i * data_per_client
        end_index = (i + 1) * data_per_client if i < num_clients - 1 else dataset_size
        client_indices = indices[start_index:end_index]
        # 构建子集对象
        client_data = np.array(dataset.data)[client_indices]
        client_target = np.array(dataset.target)[client_indices]
        client_dataset = CustomDataset(client_data, client_target)
        dataset_list.append(client_dataset)
    return dataset_list


def split_data(dataset, val_ratio, seed):
    # 获取数据集大小
    dataset_size = len(dataset.data)
    # 根据val_ratio计算验证集大小
    val_size = int(dataset_size * val_ratio)
    # 设置随机数种子
    random.seed(seed)
    # 生成随机索引
    indices = list(range(dataset_size))
    random.shuffle(indices)
    # 划分训练集和验证集
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    train_data = dataset.data[train_indices]
    train_target = dataset.target[train_indices]
    val_data = dataset.data[val_indices]
    val_target = dataset.target[val_indices]
    # 构建训练集和验证集的数据集对象
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_target = torch.tensor(train_target, dtype=torch.float32)
    val_data = torch.tensor(val_data, dtype=torch.float32)
    val_target = torch.tensor(val_target, dtype=torch.float32)
    train_dataset = CustomDataset(train_data, train_target)
    val_dataset = CustomDataset(val_data, val_target)
    return train_dataset, val_dataset


def get_ble_data_dataloader(data_path_strs, num_clients, val_ratio, batch_size, seed):
    dataset = Bunch()
    data = []
    target = []

    for data_path in data_path_strs:
        train_p = Path(data_path)
        for file in train_p.glob('*.csv'):
            df = pd.read_csv(file, parse_dates=['inter_time'], infer_datetime_format=True)
            df['interval'] = df['inter_time'].diff().dt.total_seconds().fillna(5.0)
            tempData = list(zip(df['interval'], df['rssi']))
            tag_file = Path(str(Path(data_path).parent) + "/tag/" + file.name)
            with open(tag_file) as tag_file_in:
                csv_tag_reader = csv.reader(tag_file_in)
                tag = next(csv_tag_reader)
            tag = [int(x) for x in tag]
            new_tag = [sum(tag[0:3]), sum(tag[3:6]), sum(tag[6:10])]
            dataTuple = (tempData, [x / sum(new_tag) for x in new_tag])
            data.append(dataTuple[0])
            target.append(dataTuple[1])

    # 将训练数据转换为张量对象，为使用批操作，前向填充0
    features = pad_sequence_pre([torch.tensor(x, dtype=torch.float32) for x in data], batch_first=True,
                                padding_value=0)
    # 将PyTorch张量转换为NumPy数组
    features = features.numpy()
    labels = np.array(target)

    # 将Bunch数据格式转换为PyTorch的DataLoader数据格式
    dataset.data = features
    dataset.target = labels

    # 随机划分数据集
    dataset_list = split_dataset(dataset, num_clients, seed)

    # 构建DataLoader
    train_dataloader_list = []
    val_dataloader_list = []
    for i in range(num_clients):
        # 划分训练集和验证集
        train_dataset, val_dataset = split_data(dataset_list[i], val_ratio, seed)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)

    return train_dataloader_list, val_dataloader_list


def pad_sequence_pre(sequences, batch_first=False, padding_value=0):
    max_length = max([s.size(0) for s in sequences])
    trailing_dims = sequences[0].size()[1:]
    out_dims = (len(sequences), max_length, *trailing_dims) if batch_first else (
        max_length, len(sequences), *trailing_dims)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # Pre-padding
        if batch_first:
            out_tensor[i, -length:] = tensor
        else:
            out_tensor[-length:, i] = tensor

    return out_tensor
