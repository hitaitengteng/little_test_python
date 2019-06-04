# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# @Date     : 2019-04-02 13:27:00
# @Author   : aitt
# @Language : Python3.6
# emd联调版本
# 参数F:/emd1.csv

# 安装skfuzzy包，请在终端键入以下的命令 pip install -U scikit-fuzzy
import sys
import csv
import codecs
import numpy as np
#from skfuzzy.cluster import cmeans
import numpy as np
import scipy.signal as signal
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
import pandas as pd

# 保存数据至csv文件
def data_write_csv(file_name, datas, mode):
    try:
        file_csv = codecs.open(file_name, mode, 'utf-8')
        # file_csv = codecs.open(file_name, 'a+', 'utf-8')
        # a+ 追加模式    w 覆盖写入模式
        writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for data in datas:
            writer.writerow(data)
    except Exception as err:
        print(err)
        print("保存文件失败，处理结束")
    print("保存文件成功，处理结束")


# 读csv文件的首行
def read_csv_headline(filename):
    with open(filename, "r") as csvfile:
        # 读取csv文件，返回的是迭代类型
        read = csv.reader(csvfile)
        for i, read_line in enumerate(read):
            if i == 0:
                return read_line
            else:
                break


def write_csv_headline(file_name,title):
    try:
        headline = read_csv_headline(file_name)
        headline.append(title)
        headlines = list()
        headlines.append(headline)
        data_write_csv(file_name, headlines, 'w')
    except Exception as err:
        print(err)
        sys.exit()



# 算法函数部分
# 判定当前的时间序列是否是单调序列
def ismonotonic(x):
    max_peaks = signal.argrelextrema(x, np.greater)[0]
    min_peaks = signal.argrelextrema(x, np.less)[0]
    all_num = len(max_peaks) + len(min_peaks)
    if all_num > 0:
        return False
    else:
        return True


# 寻找当前时间序列的极值点
def findpeaks(x):
    #     df_index=np.nonzero(np.diff((np.diff(x)>=0)+0)<0)

    #     u_data=np.nonzero((x[df_index[0]+1]>x[df_index[0]]))
    #     df_index[0][u_data[0]]+=1

    #     return df_index[0]
    return signal.argrelextrema(x, np.greater)[0]


# 判断当前的序列是否为 IMF 序列
def isImf(x):
    N = np.size(x)
    pass_zero = np.sum(x[0:N - 2] * x[1:N - 1] < 0)  # 过零点的个数
    peaks_num = np.size(findpeaks(x)) + np.size(findpeaks(-x))  # 极值点的个数
    if abs(pass_zero - peaks_num) > 1:
        return False
    else:
        return True


# 获取当前样条曲线
def getspline(x):
    N = np.size(x)
    peaks = findpeaks(x)
    #     print '当前极值点个数：',len(peaks)
    peaks = np.concatenate(([0], peaks))
    peaks = np.concatenate((peaks, [N - 1]))
    if (len(peaks) <= 3):
        #         if(len(peaks)<2):
        #             peaks=np.concatenate(([0],peaks))
        #             peaks=np.concatenate((peaks,[N-1]))
        #             t=interpolate.splrep(peaks,y=x[peaks], w=None, xb=None, xe=None,k=len(peaks)-1)
        #             return interpolate.splev(np.arange(N),t)
        t = interpolate.splrep(peaks, y=x[peaks], w=None, xb=None, xe=None, k=len(peaks) - 1)
        return interpolate.splev(np.arange(N), t)
    t = interpolate.splrep(peaks, y=x[peaks])
    return interpolate.splev(np.arange(N), t)


#     f=interp1d(np.concatenate(([0,1],peaks,[N+1])),np.concatenate(([0,1],x[peaks],[0])),kind='cubic')
#     f=interp1d(peaks,x[peaks],kind='cubic')
#     return f(np.linspace(1,N,N))


# 经验模态分解方法
def emd(x):
    imf = []
    while not ismonotonic(x):
        x1 = x
        sd = np.inf
        while sd > 0.1 or (not isImf(x1)):
            #             print isImf(x1)
            s1 = getspline(x1)
            s2 = -getspline(-1 * x1)
            x2 = x1 - (s1 + s2) / 2
            sd = np.sum((x1 - x2) ** 2) / np.sum(x1 ** 2)
            x1 = x2

        imf.append(x1)
        x = x - x1
    imf.append(x)
    return imf
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)







#算法调用部分

# 控制台输入参数的个数
para_nums = len(sys.argv)
try:
    assert (para_nums >= 2)
except AssertionError:
    print('输入参数不足')
    sys.exit(-1)
try:
    assert (para_nums <= 2)
except AssertionError:
    print('输入参数过多')
    sys.exit(-1)

# 提取参数
import sys
file_name = sys.argv[1]
# 数据准备
try:
    x1 = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
except FileNotFoundError as err:
    print(err)
    # print("输入文件名的格式应为：C:/Users/admin-pc/Desktop/a/test_temp.csv")
    sys.exit(-1)


#获取数据长度
qwe=x1.size
print(qwe)
t = np.arange(qwe, dtype=float)


imf1 = emd(x1)
print(imf1)






i=0
for imf1[i] in imf1:        # 第二个实例

  # 将数据与标签列合并成一个数组
  try:
      test_data_merge = np.column_stack((x1,imf1[i]))
  except Exception as err:
      print(err)
      sys.exit(-1)
  title=i

  # 首先将表头写入csv文件
  write_csv_headline(file_name,title)

  # 将合并后的数据保存至csv文件
  data_write_csv(file_name, test_data_merge, 'a+')
