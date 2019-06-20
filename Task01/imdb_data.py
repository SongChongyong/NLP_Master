# coding: utf-8
import tensorflow as tf
from tensorflow import keras
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# =============================== (1) 下载 IMDB 数据集 ===============================
# 导入Kears的datasets的imdb模块
imdb = keras.datasets.imdb
# 使用imdb的load_data函数下载数据集
# num_words=10000表示只考虑的最常见的10000单词，序列中任何出现频率更低的单词将会被编码为`oov_char`
# 数据集被保存在"/home/terence/Desktop/NLP-master/Task01//IMDB_data/imdb.npz"目录下
# 问题[保存路径这里我发现直接使用相对路径"./IMDB_dta/imdb.npz"会下载失败,或者是下载到"~/.keras/datasets/'+path"]
# 用于数据重排的随机数种子seed设置为113
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path="/home/terence/Desktop/NLP-master/Task01//IMDB_data/imdb.npz",
                                                                      num_words=10000)



# ===============================（2）探索数据 ===============================
'''
该数据集已经过预处理：每个样本都是一个整数数组，表示影评中的字词。
                  每个标签都是整数值 0 或 1，其中 0 表示负面影评，1 表示正面影评。
'''

print("\n训练集影评个数: {}, 标签个数: {}".format(len(train_data), len(train_labels)))
# 训练集影评个数: 25000, 标签个数: 25000
print("\n测试集影评个数: {}, 标签个数: {}".format(len(test_data), len(test_labels)))
# 测试集影评个数: 25000, 标签个数: 25000

# 打印第一条影评 
print("\n第一条影评："+str(train_data[0]))
'''
第一条影评：[1, 14, 22, 16, 43, 530, 973, 1622, 1385, ..., 32, 15, 16, 5345, 19, 178, 32]
'''
# 打印第一条和第二条影评中的字词数
print("\n第一条影评中的字词数: {}, 第二条影评中的字词数: {}".format(len(train_data[0]), len(train_data[1])))
# 第一条影评中的字词数: 218, 第二条影评中的字词数: 189

# 打印第一条标签,第二条标签
print("\n第一条标签: "+str(train_labels[0]))
print("第二条标签: "+str(train_labels[1]))
# 第一条标签: 1
# 第二条标签: 0
