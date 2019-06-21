# Task 01 Notes
## 01 IMDB数据集下载和探索

### 1.1 IMDB数据集下载

参考Keras中文官方文档([常用数据库](<https://keras-cn.readthedocs.io/en/latest/other/datasets/>))

**IMDB影评倾向分类库**

本数据库含有来自IMDB的25,000条影评，被标记为正面/负面两种评价。影评已被预处理为词下标构成的[序列](https://keras-cn.readthedocs.io/en/latest/preprocessing/sequence)。方便起见，**单词的下标基于它在数据集中出现的频率标定**，例如整数3所编码的词为数据集中第3常出现的词。这样的组织方法使得用户可以快速完成诸如“只考虑最常出现的10,000个词，但不考虑最常出现的20个词”这样的操作.

按照惯例，0不代表任何特定的词，而用来编码任何未知单词

**使用方法**":

```python
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      test_split=0.1,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
             
```

**参数**

- path：如果你在本机上已有此数据集（位于`'~/.keras/datasets/'+path`），则载入。否则数据将下载到该目录下
- num_words：整数或None，要考虑的最常见的单词数，序列中任何出现频率更低的单词将会被编码为`oov_char`的值。(最新的新命名是num_words,老命名为nb_words)
- skip_top：整数，忽略最常出现的若干单词，这些单词将会被编码为`oov_char`的值
- maxlen：整数，最大序列长度，任何长度大于此值的序列将被截断
- seed：整数，用于数据重排的随机数种子
- start_char：字符，序列的起始将以该字符标记，默认为1因为0通常用作padding
- oov_char：整数，因`nb_words`或`skip_top`限制而cut掉的单词将被该字符代替
- index_from：整数，真实的单词（而不是类似于`start_char`的特殊占位符）将从这个下标开始

**返回值**

两个Tuple,`(X_train, y_train), (X_test, y_test)`，其中

- X_train和X_test：序列的列表，每个序列都是词下标的列表。如果指定了`nb_words`，则序列中可能的最大下标为`nb_words-1`。如果指定了`maxlen`，则序列的最大可能长度为`maxlen`
- y_train和y_test：为序列的标签，是一个二值list

**出现的问题**

- (1) <u>path这里使用绝对路径</u>才能把imdb.npz文件保存到当前目录下, 使用相对路径会直接保存文件到'~/.keras/datasets/'

- (2) 代码报告错误"ValueError: Object arrays cannot be loaded when allow_pickle=False"

  原因: numpy版本不符合当前代码,numpy版本过高, 降低版本至1.16.2

  解决方法: pip install numpy==1.16.2 安装1.16.2版numpy

  参考链接: [使用google colab运行RNN网络代码报告错误"ValueError: Object arrays cannot be loaded when allow_pickle=False"](<https://blog.csdn.net/scrence/article/details/89645854>)



### 1.2 IMDB数据集搜索

下面看一下一下数据的格式。该数据集已经过预处理：

- (1) 每个样本都是一个整数数组，表示影评中的字词。

- (2) 每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评。

如可以打印出一下数据, 对数据集有一个了解(具体见 imdb_data.py ):

```
# 训练集影评个数: 25000, 标签个数: 25000
# 测试集影评个数: 25000, 标签个数: 25000

# 第一条影评：[1, 14, 22, 16, 43, 530, 973, 1622, 1385, ..., 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]

# 第一条影评中的字词数: 218, 第二条影评中的字词数: 189

# 第一条标签: 1, 第二条标签: 0

```



## 02 THUCNews数据集下载和探索

### 2.1 THUCNews数据集下载

THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。清华大学重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

使用THUCNews的一个子集进行训练与测试, 本次训练使用了其中的10个分类，每个分类6500条数据。

类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

数据集划分如下：

- 训练集: 5000*10			 ----- cnews.train.txt  (50000条) (文件见"./THUCNews_data",下面同理)
- 验证集: 500*10               ----cnews.val.txt  (5000条)
- 测试集: 1000*10             ----cnews.test.txt  (10000条)



### 2.2 THUCNews数据集预处理

`cnews_loader.py`为数据的预处理文件。

- `read_file()`: 读取文件数据;
- `build_vocab()`: 构建词汇表，使用字符级的表示，这一函数会将词汇表存储下来，避免每一次重复处理;
- `read_vocab()`: 读取上一步存储的词汇表，转换为`{词：id}`表示;
- `read_category()`: 将分类目录固定，转换为`{类别: id}`表示;
- `to_words()`: 将一条由id表示的数据重新转换为文字;
- `process_file()`: 将数据集从文字转换为固定长度的id序列表示;
- `batch_iter()`: 为神经网络的训练准备经过shuffle的批次的数据。

经过数据预处理，数据的格式如下：

| Data    | Shape        | Data    | Shape       |
| ------- | ------------ | ------- | ----------- |
| x_train | [50000, 600] | y_train | [50000, 10] |
| x_val   | [5000, 600]  | y_val   | [5000, 10]  |
| x_test  | [10000, 600] | y_test  | [10000, 10] |



## 03 学习召回率、准确率、ROC曲线、AUC、PR曲线这些基本概念







## 04 Datawhale Task1要求:数据集探索 (2 days)

1. 数据集

数据集：中、英文数据集各一份

中文数据集：THUCNews

THUCNews数据子集：https://pan.baidu.com/s/1hugrfRu 密码：qfud

英文数据集：IMDB数据集 Sentiment Analysis

1. IMDB数据集下载和探索

参考TensorFlow官方教程：影评文本分类  |  TensorFlow

[科赛 - Kesci.com](https://www.kesci.com/home/project/5b6c05409889570010ccce90)

1. THUCNews数据集下载和探索

参考博客中的数据集部分和预处理部分：[CNN字符级中文文本分类-基于TensorFlow实现 - 一蓑烟雨 - CSDN博客](https://blog.csdn.net/u011439796/article/details/77692621)

参考代码：[text-classification-cnn-rnn/cnews_loader.py at mas...](https://github.com/gaussic/text-classification-cnn-rnn/blob/master/data/cnews_loader.py)

1. 学习召回率、准确率、ROC曲线、AUC、PR曲线这些基本概念

参考1：[机器学习之类别不平衡问题 (2) —— ROC和PR曲线_慕课手记](https://www.imooc.com/article/48072)

