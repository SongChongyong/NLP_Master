import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"




# A placeholder is an symbolic input
x = tf.placeholder(shape=[1,10],dtype=tf.float32,name='x') 

# Variable
W = tf.Variable(tf.random_uniform(shape=[10,5], minval=-0.1, maxval=0.1, dtype=tf.float32),name='W') 
# Variable
b = tf.Variable(tf.zeros(shape=[5],dtype=tf.float32),name='b') 

h = tf.nn.sigmoid(tf.matmul(x,W) + b) # Operation to be performed


with tf.Session() as sess:
    
    tf.global_variables_initializer().run()
    # Run the operation by providing a value to the symbolic input x
    h_eval = sess.run(h,feed_dict={x: np.random.rand(1,10)}) 
    print(h_eval)
    # [[0.4901914  0.46346903 0.51798373 0.44409066 0.44412076]]




#保存计算图
#使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./Chapter02_Board'下
#而后可以在命令行输入: tensorboard --logdir=./Chapter02_Board 启动tensorboard,
#然后在浏览器中查看张量的计算图(见3.4.3_Variable.png)
# writer = tf.summary.FileWriter(logdir='./Chapter02_Board',graph=tf.get_default_graph())
# writer.flush()