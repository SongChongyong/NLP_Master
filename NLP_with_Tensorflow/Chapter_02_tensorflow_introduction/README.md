# Chapter02 理解TensorFlow

## 2.1 Tensorflow是什么



## 2.2输入、变量、输出和操作

### 2.2.1 在TensorFlow中定义输入

客户端接收数据的方式主要有三种：

- 使用Python代码在算法的每个步骤中提供数据 (**占位符: tf.placeholder()**)
- 将数据预加载并存储为TensorFlow张量 (**tf.constant()**)
- 搭建输入管道

#### 2.2.1.1 使用Python代码在算法的每个步骤中提供数据 (占位符: tf.placeholder())

1. 在计算图的构建过程中，用占位符: tf.placeholder()先占位

2. 在计算图执行过程中，通过session.run(..., feed_dict={placeholder: value}) 把外部数据传入

```python
# Building the graph

# x定义为占位符 
x = tf.placeholder(shape=[1,10],dtype=tf.float32,name='x') 
...
# session.run()过程中用feed_dict参数用于把需要的数据节点及对应数据传入
# feed_dict参数的形式是数据填充字典，形如 <数据节点，填充数据>
h_eval = session.run(h,feed_dict={x: np.random.rand(1,10)}) 
```



#### 2.2.1.2 将数据预加载并存储为TensorFlow张量 (**tf.constant()**)

数据预加载并存储，可以使用tf.constant()函数实现。

由于数据已经预先加载，执行期间无需提供数据，但是数据无法改变。

```python
# 定义常量输入
x = tf.constant(value=[[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]],dtype=tf.float32,name='x') 
```



#### 2.2.1.3 搭建输入管道

搭建输入管道用于需要快速处理大量数据时。一个典型的管道包含以下元素：

- 文件名列表
- 文件名队伍，用于为输入(记录) 读取器生成文件名
- 记录读取器，用于读取输入(记录)
- 解码器，用于解码读取的数据
- 预处理步骤(可选)
- 一个示例(即解码输入)队伍

