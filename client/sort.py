import csv
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

raw_path = "raw_data"
sorted_path = "sorted_data"
rp = Path(raw_path)
sp = Path(sorted_path)
# 直接遍历出文件绝对路径
for file_name in rp.glob('*.csv'):
    # csv文件对交互时间进行排序
    # 1.读取文件数据
    df = pd.read_csv(file_name)
    # 按照列值排序
    data = df.sort_values(by="inter_time", ascending=True)
    # 把新的数据写入文件
    data.to_csv(sp.joinpath(os.path.basename(file_name)), mode='a+', index=False)


# raw_data = []
# with open(filename) as csvfile:
#     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
#     header = next(csv_reader)  # 读取第一行每一列的标题
#     temp = 0
#     tempData = []
#     for row in csv_reader:  # 将csv 文件中的数据保存到data中
#         # 转换成时间数组
#         timeArray = time.strptime(row[5], "%Y/%m/%d %H:%M:%S")
#         # 转换成时间戳
#         timestamp = time.mktime(timeArray)
#         if temp == 0:
#             interval = 5  # 第一个数据间隔时间简单设置为5s
#         else:
#             interval = timestamp - temp
#         temp = timestamp
#         t = (interval, float(row[8]))  # 选择某几列加入到data数组中
#         tempData.append(t)
#     dataTuple = (tempData, [1, 1, 1])
#     raw_data.append(dataTuple)
#     print(raw_data)
