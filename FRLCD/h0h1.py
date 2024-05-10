import os
import numpy as np
from sklearn.utils import Bunch
from torch.utils.data import Dataset, DataLoader

def read_data_Kalman(folder_path):
    """
    读取文件夹中的数据文件，并使用卡尔曼滤波对RSSI值进行平滑处理
    """
    data_list = []  # 存储所有数据
    target_list = []  # 存储所有标签
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):  # 只读取csv文件
            continue
        file_path = os.path.join(folder_path, file_name)
        with open(file_path) as f:
            lines = f.readlines()
            target = np.array(list(map(float, lines[0].strip().split(","))))  # 获取标签
            target /= target.sum()  # 概率归一化
            data = np.array([list(map(float, line.strip().split(","))) for line in lines[1:]])  # 获取数据
            # 对数据进行归一化处理
            # data[:, 0] = min_max_normalize(data[:, 0])  # 时间秒列
            # data[:, 1] = min_max_normalize(data[:, 1])  # rssi值列

            # 使用卡尔曼滤波对RSSI值进行平滑处理
            kf = KalmanFilter()  # 创建卡尔曼滤波器
            kf.x[0] = data[0, 1]  # 使用第一个观测的RSSI值初始化状态向量
            filtered_rssi = []
            for rssi in data[:, 1]:
                kf.predict()  # 预测
                kf.update(rssi)  # 更新
                filtered_rssi.append(kf.x[0, 0])  # 存储滤波后的值
            data[:, 1] = filtered_rssi  # 用滤波后的值替换原始值

            # 对数据进行填充或截断处理
            if len(data) < 600:
                padding = np.zeros((600 - len(data), data.shape[1]))
                data = np.concatenate([padding, data], axis=0)
            else:  # 数据大于600则丢弃不匹配数据
                continue
            data_list.append(data)
            target_list.append(target)
    return np.array(data_list).astype(np.float32), np.array(target_list).astype(np.float32)


class KalmanFilter:
    def __init__(self):
        self.dt = 1.0  # 时间步长
        self.A = np.array([[1, self.dt], [0, 1]])  # 状态转移矩阵
        self.H = np.array([[1, 0]])  # 观测矩阵
        self.Q = np.array([[1e-6, 0], [0, 1e-6]])  # 系统噪声协方差矩阵
        self.R = np.array([[0.1]])  # 观测噪声协方差矩阵
        self.x = np.zeros((2, 1))  # 状态估计向量
        self.P = np.zeros((2, 2))  # 状态估计协方差矩阵

    def predict(self):
        # 预测状态和协方差
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态和协方差
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)


def read_data(folder_path):
    """
    读取文件夹中的数据文件
    """
    # print("----------------- Begin Reading Data -----------------")
    data_list = []  # 存储所有数据
    target_list = []  # 存储所有标签
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):  # 只读取csv文件
            continue
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)
        with open(file_path) as f:
            lines = f.readlines()
            target = np.array(list(map(float, lines[0].strip().split(","))))  # 获取标签
            target /= target.sum()  # 概率归一化
            # print(target)
            data = np.array([list(map(float, line.strip().split(","))) for line in lines[1:]])  # 获取数据
            # 对数据进行归一化处理
            # data[:, 0] = min_max_normalize(data[:, 0])  # 时间秒列
            # data[:, 1] = min_max_normalize(data[:, 1])  # rssi值列
            # print(data)
            # print(len(data))
            if len(data) < 600:
                padding = np.zeros((600 - len(data), data.shape[1]))
                data = np.concatenate([padding, data], axis=0)
                # print(data)
            else:  # 数据大于600则丢弃不匹配数据
                continue
            # print(data)
            data_list.append(data)
            target_list.append(target)
    # print(target_list)
    return np.array(data_list).astype(np.float32), np.array(target_list).astype(np.float32)


def read_data_gaussian(folder_path):
    """
    读取文件夹中的数据文件
    """
    # print("----------------- Begin Reading Data -----------------")
    data_list = []  # 存储所有数据
    target_list = []  # 存储所有标签
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):  # 只读取csv文件
            continue
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)
        with open(file_path) as f:
            lines = f.readlines()
            target = np.array(list(map(float, lines[0].strip().split(","))))  # 获取标签
            target /= target.sum()  # 概率归一化
            # print(target)
            data = np.array([list(map(float, line.strip().split(","))) for line in lines[1:]])  # 获取数据
            # 对数据进行归一化处理
            # data[:, 0] = min_max_normalize(data[:, 0])  # 时间秒列
            # data[:, 1] = min_max_normalize(data[:, 1])  # rssi值列
            # print(data)
            # print(len(data))
            if len(data) < 600:
                padding = np.zeros((600 - len(data), data.shape[1]))
                data = np.concatenate([padding, data], axis=0)
                # print(data)
            else:  # 数据大于600则丢弃不匹配数据
                continue
            # 添加高斯噪声
            mu, sigma = 0, 8  # 均值和标准差
            noise = np.random.normal(mu, sigma, (600, 1))  # 生成高斯噪声
            # 对rssi值列添加高斯噪声，确保值在-100到0之间
            data[:, 1] += noise[:, 0]
            data[:, 1] = np.clip(data[:, 1], -100, 0)
            # print(data)
            data_list.append(data)
            target_list.append(target)
    # print(target_list)
    return np.array(data_list).astype(np.float32), np.array(target_list).astype(np.float32)


def read_data_withmissing(folder_path):
    """
    读取文件夹中的数据文件
    """
    data_list = []  # 存储所有数据
    target_list = []  # 存储所有标签
    np.random.seed(42)
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):  # 只读取csv文件
            continue
        file_path = os.path.join(folder_path, file_name)
        with open(file_path) as f:
            lines = f.readlines()
            target = np.array(list(map(float, lines[0].strip().split(","))))  # 获取标签
            target /= target.sum()  # 概率归一化

            # 随机删除一些行
            data = []
            interval = 0  # 用于存储缺失的数据行
            for line in lines[1:]:
                if np.random.rand() <= 0.5:  # 缺失的数据行
                    interval += list(map(float, line.strip().split(",")))[0]
                else:
                    rowdata = list(map(float, line.strip().split(",")))
                    rowdata[0] += interval
                    interval = 0
                    # print(rowdata)
                    data.append(rowdata)
            data = np.array(data)

            if len(data) == 0:
                continue

            # 对数据进行处理
            if len(data) <= 600:
                padding = np.zeros((600 - len(data), data.shape[1]))
                data = np.concatenate([padding, data], axis=0)
            else:  # 数据大于600则丢弃不匹配数据
                continue

            data_list.append(data)
            target_list.append(target)

    return np.array(data_list).astype(np.float32), np.array(target_list).astype(np.float32)


def min_max_normalize(data):
    """
    最小-最大归一化
    :param data: 输入数据
    :return: 归一化后的数据
    """
    min_val = np.min(data)
    max_val = np.max(data)
    result = (data - min_val) / (max_val - min_val)
    return result


def create_dataset():
    """
    创建Bunch类型数据
    """
    train_folder_path = "./dataset/train"
    train_data, train_target = read_data_Kalman(train_folder_path)
    train_dataset = Bunch(data=train_data.reshape(train_data.shape[0], -1), target=train_target)

    test_folder_path = "./dataset/test"
    test_data, test_target = read_data_Kalman(test_folder_path)
    test_dataset = Bunch(data=test_data.reshape(test_data.shape[0], -1), target=test_target)
    return train_dataset, test_dataset


def create_miss_dataset():
    """
    创建Bunch类型数据
    """
    train_folder_path = "./dataset/train"
    train_data, train_target = read_data_withmissing(train_folder_path)
    train_dataset = Bunch(data=train_data.reshape(train_data.shape[0], -1), target=train_target)

    test_folder_path = "./dataset/test"
    test_data, test_target = read_data_withmissing(test_folder_path)
    test_dataset = Bunch(data=test_data.reshape(test_data.shape[0], -1), target=test_target)
    return train_dataset, test_dataset


def create_seq_dataset():
    """
        创建Bunch类型数据
        """
    train_folder_path = "./dataset/train"
    train_data, train_target = read_data_Kalman(train_folder_path)
    train_dataset = Bunch(data=train_data, target=train_target)

    test_folder_path = "./dataset/test"
    test_data, test_target = read_data_Kalman(test_folder_path)
    test_dataset = Bunch(data=test_data, target=test_target)
    return train_dataset, test_dataset


def create_seq_miss_dataset():
    """
        创建Bunch类型数据
        """
    train_folder_path = "./dataset/train"
    train_data, train_target = read_data_withmissing(train_folder_path)
    train_dataset = Bunch(data=train_data, target=train_target)

    test_folder_path = "./dataset/test"
    test_data, test_target = read_data_withmissing(test_folder_path)
    test_dataset = Bunch(data=test_data, target=test_target)
    return train_dataset, test_dataset


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data.copy()  # 创建数据的可写副本
        self.target = target.copy()  # 创建目标的可写副本

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.target[idx]
        return sample, label


# 定义划分数据集的函数
def split_dataset(dataset, indices, num_per_client, num_clients):
    split_datasets = []
    start = 0
    for _ in range(num_clients):
        client_indices = indices[start:start+num_per_client]
        client_data = [dataset.data[i] for i in client_indices]
        client_target = [dataset.target[i] for i in client_indices]
        client_dataset = CustomDataset(client_data, client_target)
        split_datasets.append(client_dataset)
        start += num_per_client
    return split_datasets

# 定义将数据集列表转换为 DataLoader 列表的函数
def convert_to_dataloader(datasets, batch_size):
    dataloader_list = []
    for dataset in datasets:
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader_list.append(dataloader)
    return dataloader_list

def create_dataset_loader(num_clients):
    train_dataset, test_dataset = create_seq_dataset()

    # 根据客户端数量划分训练集、验证集和测试集
    train_length = len(train_dataset.data)
    test_length = len(test_dataset.data)
    train_per_client = train_length // num_clients
    test_per_client = test_length // num_clients
    train_indices = np.random.permutation(train_length)
    test_indices = np.random.permutation(test_length)
    train_datasets = split_dataset(train_dataset, train_indices, train_per_client, num_clients)
    test_datasets = split_dataset(test_dataset, test_indices, test_per_client, num_clients)

    # 将训练集列表转换为 DataLoader 列表
    train_dataloaders = convert_to_dataloader(datasets=train_datasets, batch_size=32)

    # 将测试集列表转换为 DataLoader 列表
    test_dataloaders = convert_to_dataloader(test_datasets, batch_size=1)

    # 返回包含 DataLoader 列表的元组
    return train_dataloaders, test_dataloaders

if __name__ == "__main__":
    create_miss_dataset()
