#tesorflow

#从包中导入
import tensorflow as tf
#创建两个常量
#没看懂的解释？？？？？？之所以要用tf.constant，这是因为tensorflow有一个特别的地方，
#那就是用tensorflow的函数创建或返回一个数据，相当于一个源op，源op相当于tensorflow中最小的单元，每一个数据都是一个源op
matrix1 = tf.constant([[3., 3.]])  #1 row by 2 column
matrix2 = tf.constant([[2.],[2.]]) # 2 row by 1 column
#开始矩阵相乘
product = tf.matmul(matrix1, matrix2)
#创建一个会话
sess = tf.Session()
#tensorflow要求所有的任务必须在会话中运行，上面这个语句就是创建一个会话
sess.run(product)



######一个回归的例子
import tensorflow as tf
import numpy as np

## prepare the original data创建原始数据集
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.3*x_data+0.1
##creat parameters 创建参量（开始为随机数
weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))
##get y_prediction 创建线性预测函数
y_prediction = weight*x_data+bias
##compute the loss定义损失函数
loss = tf.reduce_mean(tf.square(y_data-y_prediction))
##creat optimizer 使用tf自带的优化器，旨在对所有步骤中的所有变量使用恒定的学习率
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss 开始训练，使用优化器让损失函数最小化
train = optimizer.minimize(loss)
#creat init 生成初始化op,相当于所有变量初始化的开关，在sess里运行则所有变量进行初始化
init = tf.global_variables_initializer()

##creat a Session 创建对话
sess = tf.Session()
##initialize 启动图只要用tf.Variable函数，都得用这个sess.run(init)初始化，不然参数没有进行初始化，无法迭代更新
sess.run(init)


## Loop循环执行训练。备注：使用tf函数生成的源op，必须在会话中运行
for step  in  range(101):
    sess.run(train)
    if step %10==0 :
        print step ,'weight:',sess.run(weight),'bias:',sess.run(bias)