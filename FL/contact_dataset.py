import csv
import time

from torch.utils.data import Dataset, random_split
from pathlib2 import Path


class ContactDataset(Dataset):  # 需要继承Dataset
    def __init__(self):
        super().__init__()
        # 初始化文件路径或文件名列表
        self.p = Path(r"./dataset")
        self.file_paths = [path for path in self.p.rglob("*_*[!_][!t][!a][!g].csv")]  # 除标签文件路径

    def __getitem__(self, index):
        # 1、根据list从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        # 2、预处理数据（例如torchvision.Transform）。
        # 3、返回数据对（例如图像和标签）。
        # 这里需要注意的是，这步所处理的是index所对应的一个样本
        file_path = self.file_paths[index]  # 根据查找第index条数据
        with open(file_path) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件

            header = next(csv_reader)  # 读取第一行作为表头
            temp = 0
            tempData = []
            timeLong = 0
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                # 转换成时间数组
                timeArray = time.strptime(row[2], "%Y/%m/%d %H:%M:%S")
                # 转换成时间戳
                timestamp = time.mktime(timeArray)
                if temp == 0:
                    interval = 5  # 5s
                else:
                    interval = timestamp - temp
                temp = timestamp
                timeLong += interval
                t = (interval, float(row[6]))  # 选择某几列加入到data数组中
                tempData.append(t)
            # 1.如果一次训练数据太少则不参与训练，因为很可能缺失了数据，蓝牙信号未接收到
            # 2.如果相隔时间太短则也表示数据缺失严重不参与训练
            # if csv_reader.line_num < 10 or timeLong < 400:
            # 利用tag文件数据生成标签，如果设置new_tag则将距离区间范围扩大例如 （0,2.5)(2.5,5.5)(5.5,10)
            strs = (file_path.name.replace(".csv", "")).split("_")
            filename = strs[0] + "_" + strs[1]
            # print(f"filename:{filename}")
            tag_file = Path(str(file_path.parent) + "/tag/" + filename + "_tag.csv")
            if not tag_file.exists():
                filename = strs[1] + "_" + strs[0]
                tag_file = Path(str(file_path.parent) + "/tag/" + filename + "_tag.csv")
            # print(f"tag_file: {tag_file}")
            with open(tag_file) as tag_file_in:
                csv_tag_reader = csv.reader(tag_file_in)  # 使用csv.reader读取csv
                tag = next(csv_tag_reader)  # 读取第一行标签
                # print(f"tag: {tag}")
            tag = [int(x) for x in tag]
            new_tag = []
            new_tag.append(tag[0] + tag[1] + tag[2])
            new_tag.append(tag[3] + tag[4] + tag[5])
            new_tag.append(tag[6] + tag[7] + tag[8] + tag[9])

            # dataTuple = (
            #     tempData,
            #     [float(tag[0]) / 10, float(tag[1]) / 10, float(tag[2]) / 10, float(tag[3]) / 10, float(tag[4]) / 10,
            #      float(tag[5]) / 10, float(tag[6]) / 10, float(tag[7]) / 10, float(tag[8]) / 10, float(tag[9]) / 10])

            data_tuple = (
                tempData,
                [float(new_tag[0]) / 10, float(new_tag[1]) / 10, float(new_tag[2]) / 10])
            # print(data_tuple)

        return data_tuple[0], data_tuple[1]

    def partition(self, train_size, test_size):
        # Calculate the sizes of the training and testing sets
        total_size = len(self)
        train_size = int(train_size * total_size)
        test_size = int(test_size * total_size)

        # Use random_split to partition the data
        train_data, test_data = random_split(self, [train_size, test_size])

        return train_data, test_data

    def __len__(self):
        # 返回数据集大小
        return len(self.file_paths)

