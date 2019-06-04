import pywt
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math


####################一些参数和函数############
def sgn(num):
    if (num > 0.0):
        return 1.0
    elif (num == 0.0):
        return 0.0
    else:
        return -1.0


begin = 1
end = 1001
###软硬阈值折衷法 a 参数
a = 0.5
###################一些参数和函数#############

###sample###
# x = [3, 7, 1, 1, -2, 5, 4, 6]

# read data
data = pd.read_csv('F:/downloads/energydata_complete.csv')
# y_value为原信号
##########画图################################################
x1 = range(begin, end)
y_values = data['RH_6'][begin:end]
'''
scatter() 
x:横坐标 y:纵坐标 s:点的尺寸
'''
# plt.scatter(x1, y_values, s=10)

plt.plot(x1, y_values)
## 设置图表标题并给坐标轴加上标签
plt.title('before')
# plt.xlabel('xValue', fontsize=14)
# plt.ylabel('yValue', fontsize=14)

# 设置刻度标记的大小
# plt.tick_params(axis='both', which='major', labelsize=14)

# 设置每个坐标轴的取值范围
# plt.axis([0, 1000, 0, 100])
plt.show()
##############画图############################################

# print(data.shape)
# print(data)
# print(data['RH_6'])
##################去噪#########################
db1 = pywt.Wavelet('db1')
# [ca3, cd3, cd2, cd1] = pywt.wavedec(x, db1)
# print(ca3)
# print(cd3)
# print(cd2)
# print(cd1)
# 分解为三层
coeffs = pywt.wavedec(y_values, db1, level=3)
print("------------------len of coeffs---------------------")
print(len(coeffs))
# print(coeffs)
recoeffs = pywt.waverec(coeffs, db1)
# print(recoeffs)

thcoeffs = []
for i in range(1, len(coeffs)):
    tmp = coeffs[i].copy()
    Sum = 0.0
    for j in coeffs[i]:
        Sum = Sum + abs(j)
    N = len(coeffs[i])
    Sum = (1.0 / float(N)) * Sum
    sigma = (1.0 / 0.6745) * Sum
    lamda = sigma * math.sqrt(2.0 * math.log(float(N), math.e))
    for k in range(len(tmp)):
        if (abs(tmp[k]) >= lamda):
            tmp[k] = sgn(tmp[k]) * (abs(tmp[k]) - a * lamda)
        else:
            tmp[k] = 0.0
    thcoeffs.append(tmp)
# print(thcoeffs)
usecoeffs = []
usecoeffs.append(coeffs[0])
usecoeffs.extend(thcoeffs)
# print(usecoeffs)
# recoeffs为去噪后信号
recoeffs = pywt.waverec(usecoeffs, db1)
# print(recoeffs)


##########画图################################################
x1 = range(begin, end)
y_values = recoeffs
'''
scatter() 
x:横坐标 y:纵坐标 s:点的尺寸
'''
# plt.scatter(x1, y_values, s=10)

plt.plot(x1, y_values)
## 设置图表标题并给坐标轴加上标签
plt.title('after')
# plt.xlabel('xValue', fontsize=14)
# plt.ylabel('yValue', fontsize=14)

# 设置刻度标记的大小
# plt.tick_params(axis='both', which='major', labelsize=14)

# 设置每个坐标轴的取值范围
# plt.axis([0, 1000, 0, 100])
plt.show()
##############画图############################################
