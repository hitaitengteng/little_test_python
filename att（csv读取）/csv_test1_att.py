import pandas as pd
import numpy as np
import datetime,time
starttime = datetime.datetime.now()
#long running

#读取csv
papa=pd.read_csv('F:/test.txt',header=None) #加载papa.txt,指定它的分隔符是 \t
test=papa.loc[papa[2] == 0]
#papa.head() #显示数据的前几行
test.to_csv('998.csv', encoding='utf-8')
endtime = datetime.datetime.now()
cha=endtime - starttime
print(cha.seconds)