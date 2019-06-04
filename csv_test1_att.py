import pandas as pd
import numpy as np
import datetime,time

starttime = datetime.datetime.now()


#读取csv
papa=pd.read_csv('F:/test.txt',header=None) #加载papa.txt,指定它的分隔符是 \t
test=papa.loc[papa[2] == 0]
test2=test.loc[test[3] == "ALCSP"]
test3=test[3].unique()
#将dataframe中某一列中不同的值，以列表的形式返回
 
#papa.head() #显示数据的前几行
test.to_csv('998.csv', encoding='utf-8')
endtime = datetime.datetime.now()
cha=endtime - starttime
print(cha.seconds)