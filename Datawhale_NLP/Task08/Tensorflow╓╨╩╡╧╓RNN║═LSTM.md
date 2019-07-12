# Tensorflow中实现RNN和LSTM

参考链接：[TensorFlow中RNN实现的正确打开方式](https://zhuanlan.zhihu.com/p/28196873)

注：所有的笔记文件只是为了记录学习的过程，具体详细教程见参考链接。

## 01 tf.nn.rnn_cell--单步的RNN

RNNCell: TensorFlow中实现RNN的基本单元，每个RNNCell都有一个call方法，使用方式是：(output, next_state) = call(input, state)。

**每调用一次RNNCell的call方法，就相当于在时间上“推进了一步”**

注意：这里的output并不是x的输出，**output其实和隐状态的值是一样的**，要得到真正的输出y=g(V * output)

- **state_size**：隐层的大小
- **output_size**：输出的大小 = state_size

代码实现RNNCell，则需要用的它的两个子类BasicRNNCell (RNN的基础类)和BasicLSTMCell (LSTM的基础类)。




(1) **BasicRNNCell**

如将一个batch送入模型计算：

|                   |                           | 状态 |
| :---------------: | :-----------------------: | :--: |
|       输入        |  (batch_size,input_size)  |  h0  |
| cell.__call__方法 | (batch_size, state_size)  |      |
|       输出        | (batch_size, output_size) |  h1  |

代码验证如下：

```python
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) 		 # 通过zero_state得到一个全0的初始状态
											 # 形状为(batch_size, state_size)
output, h1 = cell.__call__(inputs, h0) #调用__call__函数(两个'_')

print(h1.shape) # (32, 128)

```
![RNNCell_TensorBoard](./pictures/Learn_RNNCell.png)


（2） **BasicLSTMCell**

对于BasicLSTMCell，因为LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple，每个都是(batch_size, state_size)的形状：

```python
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)		# state_size = 128
inputs = tf.placeholder(np.float32, shape=(32, 100)) 		# 32 是 batch_size
h0 = lstm_cell.zero_state(32, np.float32) 				   # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.__call__(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
'''
Tensor("basic_lstm_cell/Mul_2:0", shape=(32, 128), dtype=float32)
Tensor("basic_lstm_cell/Add_1:0", shape=(32, 128), dtype=float32)
'''
```

## 02 tf.nn.dynamic_rnn--一次执行多步RNN

使用该函数就相当于调用了n次call函数。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}。

```python
# 输入数据增加time_steps，time_steps表示序列本身的长度
inputs: shape = (batch_size, time_steps, input_size) 
# 定义cell方法，cell = tf.nn.rnn_cell.BasicRNNCell() or cell = tf.nn.rnn_cell.BasicLSTMCell()
cell: RNNCell
# 初始状态。一般可以取零矩阵
initial_state: shape = (batch_size, cell.state_size)
# 得到的outputs就是time_steps步里所有的输出。
# 它的形状为(batch_size, time_steps, cell.output_size)。
# state是最后一步的隐状态，它的形状为(batch_size, cell.state_size)。
outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
```



## 03 tf.nn.rnn_cell.MultiRNNCell--多层的RNN

使用tf.nn.rnn_cell.MultiRNNCell函数对RNNCell进行堆叠

```python
# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)

# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) 
# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128, 128, 128)
# (128, 128, 128)并不是128x128x128的意思
# 而是表示共有3个隐层状态，每个隐层状态的大小为128
print(cell.state_size) # (128, 128, 128)


# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = cell.__call__(inputs, h0)
print(h1) # tuple中含有3个32x128的向量
'''
(<tf.Tensor 'multi_rnn_cell/cell_0/basic_rnn_cell/Tanh:0' shape=(32, 128) dtype=float32>, 
<tf.Tensor 'multi_rnn_cell/cell_1/basic_rnn_cell/Tanh:0' shape=(32, 128) dtype=float32>, 
<tf.Tensor 'multi_rnn_cell/cell_2/basic_rnn_cell/Tanh:0' shape=(32, 128) dtype=float32>)
'''
```

