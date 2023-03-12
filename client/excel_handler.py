import os
import openpyxl
# 2.索引值处理，距离差值
path = r"D:\Pycharm projects\close_contact_prediction\client\dataset"
os.chdir(path)  # 修改工作路径

workbook = openpyxl.load_workbook('距离区间索引时间.xlsx')  # 返回一个workbook数据类型的值
print(workbook.sheetnames)  # 打印Excel表中的所有表
sheet = workbook.active  # 获取活动表
# cell_1 = sheet['B2']
# cell_2 = sheet['C2']
# cell_3 = sheet['D2']
# cell_4 = sheet['E2']
cellsB = sheet["B"]
cellB = []
for i in cellsB:
    cellB.append(i.value)

cellsC = sheet["C"]
cellC = []
for i in cellsC:
    cellC.append(i.value)

cellsD = sheet["D"]
cellD = []
for i in cellsD:
    cellD.append(i.value)

cellsE = sheet["E"]
cellE = []
for i in cellsE:
    cellE.append(i.value)

for i, cell_b, cell_c, cell_d, cell_e in zip(range(1, 32), cellB, cellC, cellD, cellE):
    print(cell_b, cell_c, cell_d, cell_e)
    if i == 1:
        sheet["F" + str(i)].value = "b-c"
        sheet["G" + str(i)].value = "b-d"
        sheet["H" + str(i)].value = "b-e"
        sheet["I" + str(i)].value = "c-d"
        sheet["J" + str(i)].value = "c-e"
        sheet["K" + str(i)].value = "d-e"
    if i != 1:
        sheet["F" + str(i)].value = abs(cell_b - cell_c)
        sheet["G" + str(i)].value = abs(cell_b - cell_d)
        sheet["H" + str(i)].value = abs(cell_b - cell_e)
        sheet["I" + str(i)].value = abs(cell_c - cell_d)
        sheet["J" + str(i)].value = abs(cell_c - cell_e)
        sheet["K" + str(i)].value = abs(cell_d - cell_e)
workbook.save('距离区间索引时间.xlsx')
