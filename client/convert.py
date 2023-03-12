from pathlib import Path
import csv
import time
# 将排序好的数据转换成LSTM模型的的训练数据
sorted_path = "sorted_data"
sp = Path(sorted_path)
data = []
# 走廊 2023-02-21 12:10:00,12:20:00,12:30:00,12:40:00
not_wifi_time_A = [1676952600000, 1676953200000, 1676953800000, 1676954400000]
# 大厅 2023-02-21 13:04:00,13:14:00,13:24:00,13:34:00
not_wifi_time_B = [1676955840000, 1676956440000, 1676957040000, 1676957640000]
# 直接遍历出文件绝对路径
for file in sp.rglob('*.csv'):
    with open(file) as csvfile:
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        tag = next(csv_reader)  # 读取第一行作为标签
        temp = 0
        tempData = []
        for row in csv_reader:  # 将csv 文件中的数据保存到data中，列索引从0开始
            # 转换成时间数组
            timeArray = time.strptime(row[2], "%Y/%m/%d %H:%M:%S")
            # 转换成时间戳
            timestamp = time.mktime(timeArray)
            if temp == 0:
                interval = 5  # 5s
            else:
                interval = timestamp - temp
            temp = timestamp
            t = (interval, float(row[6]))  # 选择某几列加入到data数组中
            tempData.append(t)
        dataTuple = (tempData, [float(tag[0])/10, float(tag[1])/10, float(tag[2])/10, float(tag[3])/10, float(tag[4])/10])
        data.append(dataTuple)
print(data)
