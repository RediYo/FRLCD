import os
import time
from pathlib import Path
import csv
import openpyxl
from collections import Counter

# 利用submit数据计算索引及所处时间


path = r"D:\Pycharm projects\FRLCD\client"
os.chdir(path)  # 修改全局工作路径

identityCodes = ["fedc20de", "dc32f1d2", "75909e89", "0926bde8"]
input_path = r"dataset_2\submit\走廊"
input_path_2 = r"dataset_2\走廊\3"
output_path_2 = r"dataset_2\走廊\3\tag"
in_p = Path(input_path)
in_p_2 = Path(input_path_2)
index_time = {"fedc20de": [],
              "dc32f1d2": [],
              "75909e89": [],
              "0926bde8": [], }  # []内为每一分钟的索引值
for file in in_p.glob('3.csv'):
    start_timestamp = 0  # 交互开始时间
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
                        start_timestamp = timestamp
                        last_timestamp = timestamp
                        flag = False
                    min = round((timestamp - last_timestamp) / 60)
                    print(f"min:{min}")
                    index_time[identityCode].extend([last_index for x in range(0, min)])
                    last_index = int(row[2])
                    last_timestamp = timestamp
            csvfile.close()
    print(f"index_time:{index_time}")
    # 找出时间范围区间，需要根据不同的距离区间索引时间文件进行操作
    filenames = [identityCodes[0] + '_' + identityCodes[1],
                 identityCodes[0] + '_' + identityCodes[2],
                 identityCodes[0] + '_' + identityCodes[3],
                 identityCodes[1] + '_' + identityCodes[2],
                 identityCodes[1] + '_' + identityCodes[3],
                 identityCodes[2] + '_' + identityCodes[3]]
    for filename in filenames:
        for file_2 in in_p_2.glob('*.csv'):
            strs = filename.split('_')
            filename_2 = strs[1] + '_' + strs[0]
            if (file_2.name.find(filename) != -1) or (file_2.name.find(filename_2) != -1):
                # 生成索引距离差值list
                dis_list = [abs(x - y) for x, y in zip(index_time[strs[0]], index_time[strs[1]])]
                # print(f"dis_list:{dis_list}")
                strs = file_2.name.split("_")
                start_time = float(strs[0])
                end_time = float(strs[1])
                start_min = round((start_time - start_timestamp) / 60) + 1
                if start_min < 0:
                    start_min = 0
                end_min = round((end_time - start_timestamp) / 60)
                print(start_min, end_min)
                # print(start_min + 2, end_min + 1)
                # 根据范围区间统计每个区间索引的时间
                print(f"dis_list:{dis_list[int(start_min):int(end_min+1)]}")
                result = dict(Counter(dis_list[int(start_min):int(end_min+1)]))  # 统计出现次数

                # os.makedirs(os.path.dirname(filename+".csv"), exist_ok=True)
                with open(output_path_2+"/"+file_2.name, 'w', encoding='GBK') as output:  # 在全局工作路径下
                    writer = csv.writer(output)  # 用writer函数读入文件指针
                    t = []
                    print(result.items())
                    for key in range(0, 10):
                        t.append(result.get(key, 0))
                    print(f"t={t}")
                    writer.writerow(t)
                    output.close()

