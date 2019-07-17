import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 2.2.1 在TensorFlow中定义输入，有三种方式
# 使用Python代码在算法的每个步骤中提供数据 (**占位符: tf.placeholder()**)
# 将数据预加载并存储为TensorFlow张量 (**tf.constant()**)
# 搭建输入管道

# ========搭建输入管道======
'''
步骤：
- 文件名列表
- 文件名队伍，用于为输入(记录) 读取器生成文件名
- 记录读取器，用于读取输入(记录)
- 解码器，用于解码读取的数据
- 预处理步骤(可选)
- 一个示例(即解码输入)队伍
'''
# Defining the graph and session
graph = tf.Graph() # Creates a graph
session = tf.InteractiveSession(graph=graph) # Creates a session

# The filename queue 文件名队伍
filenames = ['test%d.txt'%i for i in range(1,4)]
filename_queue = tf.train.string_input_producer(filenames, capacity=3, shuffle=True,name='string_input_producer')

# check if all files are there
for f in filenames:
    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
    else:
        print('File %s found.'%f)

# 使用TextLineReader读取器读取txt文件，Reader
reader = tf.TextLineReader()

# 使用读取器的reader()函数从文件读取数据，输出是键值对
key, value = reader.read(filename_queue, name='text_read_op')

# 定义record_defaults，如果发现任何错误记录，将输出它
record_defaults = [[-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0]]

# 将读取的9行文本行解码为数字列
# 然后把这些列拼接起来，形成单个张量（称为特征），这些张量打乱批次被传入变量x
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10])

# 用tf.train.shuffle_batch实现把张量进行打乱按批次输出
# 不需要打乱数据时，用tf.train.batch操作
x = tf.train.shuffle_batch([features], batch_size=3,
                           capacity=5, name='data_batch', 
                           min_after_dequeue=1,num_threads=1)

# 线程管理器：tf.train.Coordinator
# 执行线程管理器： tf.train.start_queue_runners()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=session)

# Building the graph by defining the variables and calculations
# 定义变量和运算
W = tf.Variable(tf.random_uniform(shape=[10,5], minval=-0.1, maxval=0.1, dtype=tf.float32),name='W') # Variable
b = tf.Variable(tf.zeros(shape=[5],dtype=tf.float32),name='b') # Variable

h = tf.nn.sigmoid(tf.matmul(x,W) + b) # Operation to be performed

# Executing operations and evaluating nodes in the graph
tf.global_variables_initializer().run() # Initialize the variables 初始化

# Calculate h with x and print the results for 5 steps
for step in range(5):
    x_eval, h_eval = session.run([x,h]) 
    print('========== Step %d =========='%step)
    print('Evaluated data (x)')
    print(x_eval)
    print('Evaluated data (h)')
    print(h_eval)
    print('')

# We also need to explicitly stop the coordinator otherwise the process will hang indefinitely
# 停止线程
coord.request_stop()
coord.join(threads)
session.close()


'''
========== Step 0 ==========
Evaluated data (x)
[[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
 [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
 [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]]
Evaluated data (h)
[[0.45768267 0.4832915  0.4832512  0.46735597 0.47615865]
 [0.45768267 0.4832915  0.4832512  0.46735597 0.47615865]
 [0.4576827  0.48329148 0.4832512  0.46735597 0.47615865]]

========== Step 1 ==========
Evaluated data (x)
[[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
 [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
 [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]]
Evaluated data (h)
[[0.49576756 0.49833485 0.49610713 0.49691007 0.4961124 ]
 [0.49576756 0.49833485 0.49610713 0.49691013 0.4961124 ]
 [0.49576753 0.49833485 0.49610713 0.49691013 0.4961124 ]]

========== Step 2 ==========
Evaluated data (x)
[[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
 [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
 [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]]
Evaluated data (h)
[[0.49576756 0.49833485 0.49610713 0.49691007 0.4961124 ]
 [0.45768267 0.4832915  0.4832512  0.46735597 0.47615865]
 [0.4576827  0.48329148 0.4832512  0.46735597 0.47615865]]

========== Step 3 ==========
Evaluated data (x)
[[0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
 [1.  0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]
 [1.  0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]]
Evaluated data (h)
[[0.49576756 0.49833485 0.49610713 0.49691007 0.4961124 ]
 [0.49586084 0.49839813 0.47395623 0.49870092 0.4811041 ]
 [0.4958608  0.4983981  0.47395623 0.49870092 0.4811041 ]]

========== Step 4 ==========
Evaluated data (x)
[[1.  0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]
 [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
 [1.  0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]]
Evaluated data (h)
[[0.49586084 0.49839813 0.47395623 0.49870095 0.4811041 ]
 [0.49576756 0.49833485 0.49610713 0.49691013 0.4961124 ]
 [0.4958608  0.4983981  0.47395623 0.49870092 0.4811041 ]]
'''



#保存计算图
#使用tf.get_default_graph()保存默认的TensorBoard,事件文件保存在'./Chapter02_Board'下
#而后可以在命令行输入: tensorboard --logdir=./Chapter02_Board 启动tensorboard,
#然后在浏览器中查看张量的计算图(见3.4.3_Variable.png)
# writer = tf.summary.FileWriter(logdir='./Chapter02_Board',graph=tf.get_default_graph())
# writer.flush()