import os
from pathlib import Path
import csv
import openpyxl
from collections import Counter

# 3.距离区间索引时间.xlsx数据提取训练数据的标签，生成csv文件，距离0-9m，从0开始

# 走廊 2023-02-21 12:10:00, 12:40:00
not_wifi_time_A = [1676952600.0, 1676954400.0]
# 大厅 2023-02-21 13:04:00, 13:34:00
not_wifi_time_B = [1676955840.0, 1676957640.0]
# 走廊 2023-02-21 13:03:00,13:33:00
wifi_time_A = [1676955780.0, 1676957580.0]
# 大厅 2023-02-21 12:15:00,12:45:00
wifi_time_B = [1676952900.0, 1676954700.0]


path = r"D:\Pycharm projects\FRLCD\client\dataset"
os.chdir(path)  # 修改全局工作路径

workbook = openpyxl.load_workbook('距离区间索引时间.xlsx')  # 返回一个workbook数据类型的值
print(workbook.sheetnames)  # 打印Excel表中的所有表
sheet = workbook.active  # 获取活动表
# identityCodes = ["cb581061", "f69de6fc", "fedc20de", "defe41cc"]
identityCodes = "dc32f1d2", "164732b9", "0926bde8", "75909e89"
filename = ""
filename_sp = ""
input_dir = "input"
od = Path(input_dir)
for dis in ['F', 'G', 'H', 'I', 'J', 'K']:

    if dis == 'F':
        filename = identityCodes[0] + '_' + identityCodes[1]
        filename_sp = identityCodes[1] + '_' + identityCodes[0]
    elif dis == 'G':
        filename = identityCodes[0] + '_' + identityCodes[2]
        filename_sp = identityCodes[2] + '_' + identityCodes[0]
    elif dis == 'H':
        filename = identityCodes[0] + '_' + identityCodes[3]
        filename_sp = identityCodes[3] + '_' + identityCodes[0]
    elif dis == 'I':
        filename = identityCodes[1] + '_' + identityCodes[2]
        filename_sp = identityCodes[2] + '_' + identityCodes[1]
    elif dis == 'J':
        filename = identityCodes[1] + '_' + identityCodes[3]
        filename_sp = identityCodes[3] + '_' + identityCodes[1]
    elif dis == 'K':
        filename = identityCodes[2] + '_' + identityCodes[3]
        filename_sp = identityCodes[3] + '_' + identityCodes[2]

    for file in od.glob('*.csv'):  # 找出时间范围区间，需要根据不同的距离区间索引时间文件进行操作
        if (file.name.find(filename) != -1) or (file.name.find(filename_sp) != -1):
            strs = file.name.split("_")
            start_time = strs[0]
            end_time = strs[1]
            start = wifi_time_B[0]
            end = wifi_time_B[1]
            if float(start_time) > float(end):
                continue
            start_min = int(float(start_time) - start)//60  # 除法向下取整
            end_min = int(float(end_time) - start)//60+1  # 除法向上取整
            print(start_min, end_min)
            # print(start_min + 2, end_min + 1)
            # 根据范围区间统计每个区间索引的时间
            cells = sheet[dis]
            cell = []
            for i, j in zip(range(1, 32), cells):  # 根据时间戳范围判断需要统计的时间
                if start_min+2 <= i <= end_min+1:
                    cell.append(j.value)
            print(f"cell:{cell}")
            result = dict(Counter(cell))  # 统计出现次数

            # os.makedirs(os.path.dirname(filename+".csv"), exist_ok=True)
            with open(file.name, 'w', encoding='GBK') as output:  # 在全局工作路径下
                writer = csv.writer(output)  # 用writer函数读入文件指针
                t = []
                print(result.items())
                for key in range(0, 10):
                    t.append(result.get(key, 0))
                print(f"t={t}")
                writer.writerow(t)
                output.close()
