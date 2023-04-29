import csv
import time

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from pathlib import Path


def load_ble():
    """
    获取ble数据集
    """
    data_path = "../dataset_classify_min/data"
    ble = Bunch()
    ble.data, ble.target = get_ble_data(data_path)
    return ble


def get_ble_data(data_path):
    dp = Path(data_path)
    data = []
    target = []

    for file in dp.glob('*.csv'):

        with open(file) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            header = next(csv_reader)  # 读取第一行作为表头
            temp = []
            temp_group = []
            temp_tag = []
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                if int(row[0]) < 10:  # 表示为最后一行即标签行
                    for s in row:
                        temp_tag.append(int(s))
                    break
                else:
                    temp.append(row)
            # print(f"len(temp_tag) {len(temp_tag)}")
            # print(f"temp {temp}")
            # target.extend(temp_tag)
            # 转换成时间数组
            timeArray = time.strptime(temp[0][2], "%Y/%m/%d %H:%M:%S")
            # 转换成时间戳
            first_timestamp = time.mktime(timeArray)
            t = []  # 一分钟内数据
            for tmp in temp[1:]:
                # 转换成时间数组
                timeArray = time.strptime(tmp[2], "%Y/%m/%d %H:%M:%S")
                # 转换成时间戳
                timestamp = time.mktime(timeArray)
                if timestamp - first_timestamp <= 60:
                    t.append(tmp)
                else:
                    temp_group.append(t)
                    first_timestamp = first_timestamp + 60
                    while first_timestamp + 60 <= timestamp:
                        temp_group.append([])
                        first_timestamp += 60
                    t = []

            target.extend(temp_tag[0:len(temp_group)])
            print(f"len(temp_tag) {len(temp_tag[0:len(temp_group)])}")
            print(f"len(temp_group) {len(temp_group)}")
            print(f"temp_group {len(temp_group)}")

            # 每分钟数据扩充0为60
            for i in range(0, len(temp_group)):
                d_min = temp_group[i]
                d_min_ex = []
                if len(d_min) == 0:
                    for m in range(0, 60):  # 用0扩充到60条数据
                        d_min_ex.append([0, 0])
                    print(f"d_min_ex {d_min_ex}")
                    data.append(d_min_ex)
                    continue
                # 转换成时间数组
                timeArray = time.strptime(d_min[0][2], "%Y/%m/%d %H:%M:%S")
                # 转换成时间戳
                pre_timestamp = time.mktime(timeArray)
                d_min_ex.append([d_min[0][6], 5])
                for j in d_min[1:]:
                    # 转换成时间数组
                    timeArray = time.strptime(j[2], "%Y/%m/%d %H:%M:%S")
                    # 转换成时间戳
                    timestamp = time.mktime(timeArray)
                    d_min_ex.append([j[6], timestamp - pre_timestamp])
                    pre_timestamp = timestamp
                for m in range(len(d_min_ex), 60):  # 用0扩充到60条数据
                    d_min_ex.append([0, 0])
                data.append(d_min_ex)
                # print(d_min_ex)

    print(f"data:{np.array(data).reshape(np.array(data).shape[0], -1)}")
    print(f"target:{np.array(target[0:len(data)])}")

    return np.array(data, dtype=object).reshape(np.array(data).shape[0], -1), np.array(target[0:len(data)], dtype=object)

# get_ble_data("../dataset_classify_min/data")
