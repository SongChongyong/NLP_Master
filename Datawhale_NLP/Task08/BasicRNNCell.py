# -*- cooding:utf-8 -*-

import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
output, h1 = cell.__call__(inputs, h0) #调用__call__函数(两个'_')

print(h1.shape) # (32, 128)




#保存计算图
#使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./Lstm_stocks_Board'下
#而后可以在命令行输入: tensorboard --logdir=./Learn_rnn_Board启动tensorboard,
#然后在浏览器中查看张量的计算图(见Lstm_stocks_Board.png)
writer = tf.summary.FileWriter(logdir='./Learn_rnn_Board',graph=tf.get_default_graph())
writer.flush()
