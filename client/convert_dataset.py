from pathlib import Path
import csv
import time
# 1.将排序好的数据转换成LSTM模型的的训练数据（0-10min，10-20min，20-30min）
sorted_path = "sorted_data"
sp = Path(sorted_path)
# 走廊 2023-02-21 12:10:00,12:20:00,12:30:00,12:40:00
not_wifi_time_A = [1676952600, 1676953200, 1676953800, 1676954400]
# 大厅 2023-02-21 13:04:00,13:14:00,13:24:00,13:34:00
not_wifi_time_B = [1676955840, 1676956440, 1676957040, 1676957640]
# 走廊 2023-02-21 13:03:00,13:13:00,13:23:00,13:33:00
wifi_time_A = [1676955780, 1676956380, 1676956980, 1676957580]
# 大厅 2023-02-21 12:15:00,12:25:00,12:35:00,12:45:00
wifi_time_B = [1676952900, 1676953500, 1676954100, 1676954700]
# 直接遍历出文件绝对路径
for file in sp.rglob('*.csv'):
    with open(file) as csvfile:
        with open("./dataset/"+file.name, 'w', newline='', encoding='GBK') as output:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            writer = csv.writer(output)  # 用writer函数读入文件指针
            header = next(csv_reader)  # 读取第一行作为表头
            writer.writerow(header)
            for row in csv_reader:  # 将csv 文件中的数据保存到data中，列索引从0开始
                # 转换成时间数组
                timeArray = time.strptime(row[2], "%Y/%m/%d %H:%M:%S")
                # 转换成时间戳
                timestamp = time.mktime(timeArray)
                print(timestamp)
                if wifi_time_B[2] <= timestamp <= wifi_time_B[3]:
                    writer.writerow(row)
