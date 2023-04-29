from pathlib import Path
import csv
import time
from shutil import copyfile

# 第2、3次数据采集
# 3.筛选完整数据 数据接收时间间隔均小于1分钟
data_path = "dataset_3/钟组/活动室/2"
data_path_is_full = data_path+"/is_full_data"
data_path_not_full = data_path+"/not_full_data"
dp = Path(data_path)
# 直接遍历出文件绝对路径
for file in dp.glob('*.csv'):
    with open(file) as csvfile:

        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)  # 读取第一行作为表头
        try:
            row_first = next(csv_reader)  # 读取第二行作为初始时间戳，时间戳单位秒
        except StopIteration as s:
            print(f"{csvfile.name} StopIteration")
            continue
        # 转换成时间数组
        timeArray = time.strptime(row_first[2], "%Y/%m/%d %H:%M:%S")
        # 转换成时间戳
        first_timestamp = time.mktime(timeArray)
        temp = first_timestamp
        data = []
        data_one = [row_first]
        flag = True  # 完整数据标记
        for row in csv_reader:  # 遍历
            # 转换成时间数组
            timeArray = time.strptime(row[2], "%Y/%m/%d %H:%M:%S")
            # 转换成时间戳
            timestamp = time.mktime(timeArray)
            # 小于1分钟即60秒，则为完整数据
            if (timestamp - temp) < 60:
                temp = timestamp
            else:  # 否则为缺失数据（但是为正常数据）
                # 复制该文件到缺失文件夹
                print(f"timestamp:{timestamp}")
                flag = False
                copyfile(data_path+"/"+file.name, data_path_not_full+"/"+file.name)
                break
        if flag:  # 复制该文件到完整数据文件夹
            copyfile(data_path+"/"+file.name, data_path_is_full+"/"+file.name)
    csvfile.close()
