import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


def create_model():
    model = Sequential()
    #输入数据的shape为(n_samples, timestamps, features)
    #隐藏层设置为256, input_shape元组第二个参数1意指features为1
    #下面还有个lstm，故return_sequences设置为True
    model.add(LSTM(units=256,input_shape=(None,1),return_sequences=True))
    model.add(LSTM(units=256))
    #后接全连接层，直接输出单个值，故units为1
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mse',optimizer='adam')
    return model
	
df = pd.read_csv('international-airline-passengers.csv',usecols=['passengers'])
#数据归一化后分成序列
scaler_minmax = MinMaxScaler()
data = scaler_minmax.fit_transform(df)
infer_seq_length = 10#用于推断的历史序列长度

d = []
for i in range(data.shape[0]-infer_seq_length):
    d.append(data[i:i+infer_seq_length+1].tolist())
d = np.array(d)


split_rate = 0.9
X_train, y_train = d[:int(d.shape[0]*split_rate),:-1], d[:int(d.shape[0]*split_rate),-1]


model =create_model()

model.fit(X_train, y_train, batch_size=20,epochs=100,validation_split=0.1)

#inverse_transform获得归一化前的原始数据
plt.plot(scaler_minmax.inverse_transform(d[:,-1]),label='true data')
plt.plot(scaler_minmax.inverse_transform(model.predict(d[:,:-1])),'r:',label='predict')
plt.legend()

plt.plot()
plt.plot(scaler_minmax.inverse_transform(d[int(len(d)*split_rate):,-1]),label='true data')
plt.plot(scaler_minmax.inverse_transform(model.predict(d[int(len(d)*split_rate):,:-1])),'r:',label='predict')
plt.legend()




