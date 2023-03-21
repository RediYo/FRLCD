from pathlib import Path
import csv
import time

# A.将排序好的数据转换成LSTM模型的的训练数据（3-10min）
sorted_path = "sorted_data"
sp = Path(sorted_path)
# 直接遍历出文件绝对路径
for file in sp.glob('*.csv'):
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
        data_one = []
        for row in csv_reader:  # 将csv 文件中的数据按照3-10min保存到data中，列索引从0开始
            # 转换成时间数组
            timeArray = time.strptime(row[2], "%Y/%m/%d %H:%M:%S")
            # 转换成时间戳
            timestamp = time.mktime(timeArray)
            print(f"timestamp:{timestamp}")
            # 小于3分钟即180秒，则为完整数据, 同时一次数据需要小于10min
            if timestamp - temp < 180 and (timestamp - first_timestamp) < 600:
                data_one.append(row)
            else:  # 否则断开预测
                data.append(data_one)
                data_one = []
                first_timestamp = timestamp
            temp = timestamp

        # 遍历data列表分别写入data_one
        for data_one in data:
            if len(data_one) > 0:
                # 转换成时间数组
                timeArray = time.strptime(data_one[0][2], "%Y/%m/%d %H:%M:%S")
                # 转换成时间戳
                timestamp_first = time.mktime(timeArray)
                timeArray = time.strptime(data_one[len(data_one)-1][2], "%Y/%m/%d %H:%M:%S")
                timestamp_last = time.mktime(timeArray)
                with open("./dataset/" + str(timestamp_first) + "_" + str(timestamp_last) + "_" + file.name, 'w', newline='', encoding='GBK') as output:
                    writer = csv.writer(output)  # 用writer函数读入文件指针
                    writer.writerow(header)
                    for one in data_one:
                        writer.writerow(one)  # 用writer函数读入文件指针
                    output.close()
    csvfile.close()

