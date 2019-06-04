#################生成训练数据
#上面的代码用来生产训练数据，用sin函数生成了1000个数据点。
#rnn_data_format是为了生成[time_step, input]维度的数据。
#这里吧time_step设为5。因此可知，X的每一个样本是shape为[5,1]的矩阵，用来预测y，y的每一个样本shape为[1,1]。
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers as tflayers
 
def generator(x):
    return [np.sin(i*0.06) for i in range(x)]
 
def rnn_data_format(data,timestep=7,label=False):
    data=pd.DataFrame(data)
    rnn_data=[]
    if label:  ###label是二维数组，[样本数，1]
        for i in range(len(data) - timestep):
           rnn_data.append([x for x in data.iloc[i+timestep].as_matrix()])
    else:     ###样本是3维数组[样本数，time_step，1]
        for i in range(len(data) - timestep):
           rnn_data.append([x for x in data.iloc[i:(i+timestep)].as_matrix()])
    return np.array(rnn_data,dtype=np.float32)
	
class DataSet(object):
    def __init__(self, x,y):
        self._data_size = len(x)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._data_index = np.arange(len(x))
        self.x=x
        self.y=y
 
    def next_batch(self,batch_size):
        start=self._index_in_epoch
        if start+batch_size>=self._data_size :
            np.random.shuffle(self._data_index)
            self._index_in_epoch=0
            start=self._index_in_epoch
            end=self._index_in_epoch+batch_size
            self._index_in_epoch=end
        else:
            end = self._index_in_epoch + batch_size
            self._index_in_epoch = end
        batch_x,batch_y=self.get_data(start,end)
        return np.array(batch_x,dtype=np.float32),np.array(batch_y,dtype=np.float32)
 
    def get_data(self,start,end):
        batch_x=[]
        batch_y=[]
        for i in range(start,end):
            batch_x.append(self.x[self._data_index[i]])
            batch_y.append(self.y[self._data_index[i]])
        return batch_x,batch_y
 
##生成数据
x=generator(1000)
X=rnn_data_format(x,5)
y=rnn_data_format(x,5,label=True)
trainds = DataSet(X,y)


##################2构建网络
def weight_variable(shape):  ###这里定义的是全连接的参数w
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):   ###这里定义的是全连接的参数b
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
def lstm_cell3(model='lstm',rnn_size=[128,128],keep_prob=0.8):   ###定义LSTM层
    if model=='lstm':
        cell_func=tf.contrib.rnn.BasicLSTMCell
    elif model=='gru':
        cell_func=tf.contrib.rnn.GRUCell
    elif model=='rnn':
        cell_func=tf.contrib.rnn.BasicRNNCell
    cell=[]
    for unit in rnn_size:  ###定义多层LSTM
        cell.append(tf.contrib.rnn.DropoutWrapper(cell_func(unit, state_is_tuple = True),output_keep_prob=keep_prob))    ###使用的dropout
    return tf.contrib.rnn.MultiRNNCell(cell,state_is_tuple=True)
 
def dnn_stack(input,layers):   ###全连接层使用tflayers里面的stack，这样不用自己手动写连接
    if layers and isinstance(layers, dict):
        dnn_out=tflayers.stack(input, tflayers.fully_connected,
                              layers['layers'],
                              activation_fn=layers.get('activation')
                              )
    elif layers:
        dnn_out= tflayers.stack(input, tflayers.fully_connected, layers)
    W_fc1 = weight_variable([layers['layers'][-1], 1])
    b_fc1 = bias_variable([1])
    pred=tf.add(tf.matmul(dnn_out,W_fc1),b_fc1,name='dnnout')   ###dnn的输出结果和label对应是一个数字
    return pred

##################3，定义损失函数，梯度
input_data=tf.placeholder("float", shape=[None, 5,1])
input_label=tf.placeholder("float", shape=[None, 1])
###定义LSTM
rnncell=lstm_cell() 
initial_state = rnncell.zero_state(batch_size, tf.float32)
output, state = tf.nn.dynamic_rnn(rnncell, inputs=input_data, initial_state=initial_state, time_major=False) ##LSTM的结果
###LSTM结果输入dnn
dnn_out=dnn_stack(output[:,-1,:],layers={'layers':[32,16]}) ##
loss=tf.reduce_sum(tf.pow(dnn_out-input_label,2)) ##平方和损失
learning_rate = tf.Variable(0.0, trainable = False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5) ##计算梯度
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(grads, tvars))

#################4, 模型训练
#一般的步骤：
#1, 定义好损失函数,梯度
#2, 启动session
#3, 设置循环次数(epoch或者step )
#4, 循环中喂入数据
#在训练完成后保存模型，用于后面的预测

batch_size = 32
epoch=30
batch=len(X)//batch_size
 
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epo in range(epoch):
        sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epo)))
        all_loss = 0.0
        for bat in range(batch):
            x_,y_=trainds.next_batch(batch_size=batch_size)
            train_loss, _ = sess.run([loss, train_op], feed_dict={input_data: x_, input_label: y_})
            all_loss = all_loss + train_loss
        print epoch, ' Loss: ', all_loss * 1.0 / batch
    saver.save(sess,'./rnn/lstm_time_series.model')

##################5.预测
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers as tflayers
 
def generator(x):
    return [np.sin(i*0.06) for i in range(x)]
 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
 
def rnn_data_format(data,timestep=7,label=False):
    data=pd.DataFrame(data)
    rnn_data=[]
    if label:
        for i in range(len(data) - timestep):
           rnn_data.append([x for x in data.iloc[i+timestep].as_matrix()])
    else:
        for i in range(len(data) - timestep):
           rnn_data.append([x for x in data.iloc[i:(i+timestep)].as_matrix()])
    return np.array(rnn_data,dtype=np.float32)
 
def lstm_cell(model='lstm',rnn_size=[128,128],keep_prob=0.8):
    if model=='lstm':
        cell_func=tf.contrib.rnn.BasicLSTMCell
    elif model=='gru':
        cell_func=tf.contrib.rnn.GRUCell
    elif model=='rnn':
        cell_func=tf.contrib.rnn.BasicRNNCell
    cell=[]
    for unit in rnn_size:
        cell.append(tf.contrib.rnn.DropoutWrapper(cell_func(unit, state_is_tuple = True),output_keep_prob=keep_prob))
    return tf.contrib.rnn.MultiRNNCell(cell,state_is_tuple=True)
 
def dnn_stack(input,layers):
    if layers and isinstance(layers, dict):
        dnn_out=tflayers.stack(input, tflayers.fully_connected,
                              layers['layers'],
                              activation_fn=layers.get('activation')
                              )
    elif layers:
        dnn_out= tflayers.stack(input, tflayers.fully_connected, layers)
    W_fc1 = weight_variable([layers['layers'][-1], 1])
    b_fc1 = bias_variable([1])
    pred=tf.add(tf.matmul(dnn_out,W_fc1),b_fc1,name='dnnout')
    return pred
 

##生成数据
x=generator(1000)
X=rnn_data_format(x,5)
y=rnn_data_format(x,5,label=True)
trainds = DataSet(X,y)
 
input_data=tf.placeholder("float", shape=[batch_size, 5,1])
input_label=tf.placeholder("float", shape=[batch_size, None])
###
rnncell=lstm_cell()
initial_state = rnncell.zero_state(1, tf.float32)
 
output, state = tf.nn.dynamic_rnn(rnncell, inputs=input_data, initial_state=initial_state, time_major=False)
 
dnn_layer(output[:,-1,:])
dnn_out=dnn_stack(output[:,-1,:],layers={'layers':[32,16]})
##################读取数据
num_predict = 10
saver = tf.train.Saver(tf.global_variables())
# tf.reset_default_graph()
with tf.Session() as sess:
 
    saver.restore(sess, './rnn/layer3/lstm_time_series.model')
    #graph = tf.get_default_graph()
    prev_seq = X[-num_predict]
    #o,s=sess.run([output, state],feed_dict={input_data: [prev_seq]})
    #print o
    #print s
    predict = []
    for i in range(num_predict):
        next_seq = sess.run(dnn_out, feed_dict={input_data: [prev_seq]})
        prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        predict.append(next_seq[-1])
        print next_seq
 
    plt.figure()
    plt.plot(list(range(len(x[-(num_predict - 900):]))), x[-(num_predict - 900):], color='b')
    plt.plot(list(range(len(x[-(num_predict - 900):]) - num_predict, len(x[-(num_predict - 900):]))), predict,color='r')
    plt.show()

	
	