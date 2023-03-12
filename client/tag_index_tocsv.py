import os
from pathlib import Path
import csv
import openpyxl
from collections import Counter

# 3.距离区间索引时间.xlsx数据提取训练数据的标签，生成csv文件，距离0-9m，从0开始

path = r"D:\Pycharm projects\close_contact_prediction\client\dataset"
os.chdir(path)  # 修改工作路径

workbook = openpyxl.load_workbook('距离区间索引时间.xlsx')  # 返回一个workbook数据类型的值
print(workbook.sheetnames)  # 打印Excel表中的所有表
sheet = workbook.active  # 获取活动表
# identityCodes = ["cb581061", "f69de6fc", "fedc20de", "defe41cc"]
identityCodes = "dc32f1d2", "164732b9", "0926bde8", "75909e89"
filename = ""
for dis in ['F', 'G', 'H', 'I', 'J', 'K']:
    cells = sheet[dis]
    cell = []
    for i, j in zip(range(1, 32), cells):
        if 1 < i < 12:  # 1 < i < 12 11 < i < 22 21 < i < 32
            cell.append(j.value)
    print(cell)
    result = dict(Counter(cell))
    if dis == 'F':
        filename = identityCodes[0] + '_' + identityCodes[1] + "_tag"
    elif dis == 'G':
        filename = identityCodes[0] + '_' + identityCodes[2] + "_tag"
    elif dis == 'H':
        filename = identityCodes[0] + '_' + identityCodes[3] + "_tag"
    elif dis == 'I':
        filename = identityCodes[1] + '_' + identityCodes[2] + "_tag"
    elif dis == 'J':
        filename = identityCodes[1] + '_' + identityCodes[3] + "_tag"
    elif dis == 'K':
        filename = identityCodes[2] + '_' + identityCodes[3] + "_tag"
    # os.makedirs(os.path.dirname(filename+".csv"), exist_ok=True)
    with open(filename+".csv", 'w', encoding='GBK') as output:
        writer = csv.writer(output)  # 用writer函数读入文件指针
        t = []
        print(result.items())
        for key in range(0, 10):
            t.append(result.get(key, 0))
        print(f"t={t}")
        writer.writerow(t)
        output.close()
