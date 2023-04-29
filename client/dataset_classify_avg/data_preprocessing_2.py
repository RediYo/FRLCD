from pathlib import Path
import csv
import time

import numpy as np

# 计算每分钟平均RSSI值
data_path = "../dataset_2/活动室/3"
dp = Path(data_path)
# 遍历所有特征数据文件
for file in dp.glob('*.csv'):
    with open(file) as csvfile:
        with open("./rssi_avg/" + file.name, 'w', newline='', encoding='GBK') as output:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            writer = csv.writer(output)  # 用writer函数读入文件指针
            header = next(csv_reader)  # 读取第一行作为表头
            # writer.writerow(header)
            first_data_row = next(csv_reader)
            first_timeArray=time.strptime(first_data_row[2], "%Y/%m/%d %H:%M:%S")
            time_min = first_timeArray.tm_min
            while first_timeArray.tm_sec > 40:
                first_data_row = next(csv_reader)
                first_timeArray = time.strptime(first_data_row[2], "%Y/%m/%d %H:%M:%S")
                if first_timeArray.tm_min > time_min:
                    break
            pre_min = first_timeArray.tm_min
            rssi_s = [float(first_data_row[6])]
            for row in csv_reader:  # 将csv 文件中的数据保存到data中，列索引从0开始
                # 转换成时间数组
                timeArray = time.strptime(row[2], "%Y/%m/%d %H:%M:%S")
                current_min = timeArray.tm_min
                if current_min - pre_min >= 2:
                    rssi_avg = sum(rssi_s) / len(rssi_s)
                    writer.writerow([rssi_avg])
                    rssi_s.clear()
                    rssi_s.append(float(row[6]))
                    for i in range(current_min-pre_min-1):
                        writer.writerow([0])  # 一分钟内缺失rssi值默认填充0
                    pre_min = current_min
                elif current_min == pre_min:
                    rssi_s.append(float(row[6]))
                else:
                    rssi_avg = sum(rssi_s) / len(rssi_s)
                    writer.writerow([rssi_avg])
                    pre_min = current_min
                    rssi_s.clear()
                    rssi_s.append(float(row[6]))
                print(current_min)
            # 最后一分钟数据
            rssi_avg = sum(rssi_s) / len(rssi_s)
            writer.writerow([rssi_avg])

