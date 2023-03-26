import os
import time
from pathlib import Path
import csv
import openpyxl
from collections import Counter

# 利用submit数据计算索引及所处时间


path = r"D:\Pycharm projects\FRLCD\client\dataset_2\submit"
os.chdir(path)  # 修改全局工作路径

identityCodes = "fedc20de", "dc32f1d2", "75909e89", "0926bde8"
input_path = r"活动室"
od = Path(input_path)
index_time = {"fedc20de": [],
                 "dc32f1d2": [],
                 "75909e89": [],
                 "0926bde8": [], }  # []内为每一分钟的索引值
for file in od.glob('1.csv'):
    for identityCode in identityCodes:
        with open(file) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            header = next(csv_reader)  # 读取第一行作为表头
            last_timestamp = 0
            last_index = 0
            flag = True  # 初始时间戳标志
            for row in csv_reader:
                if row[1] == identityCode + "bletest0":
                    # 转换成时间数组
                    timeArray = time.strptime(row[3], "%Y/%m/%d %H:%M:%S")
                    # 转换成时间戳
                    timestamp = time.mktime(timeArray)
                    # minutes = round(timestamp / 60)
                    # print(f"minutes:{minutes}")
                    if flag:
                        last_timestamp = timestamp
                        flag = False
                    min = round((timestamp - last_timestamp) / 60)
                    print(f"min:{min}")
                    index_time[identityCode].extend([last_index for x in range(0, min)])
                    last_index = int(row[2])
                    last_timestamp = timestamp
        csvfile.close()
        print(f"index_time:{index_time}")
        # 生成标签文件
        with open(file.name, 'w', encoding='GBK') as output:  # 在全局工作路径下
            writer = csv.writer(output)  # 用writer函数读入文件指针
            t = []
            print(result.items())
            for key in range(0, 10):
                t.append(result.get(key, 0))
            print(f"t={t}")
            writer.writerow(t)
            output.close()
