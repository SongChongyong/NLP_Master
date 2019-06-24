# Task 01 Notes
## 01 基本文本处理技能  

此部分参考了博客[NLP系列——(2)特征提取](https://blog.csdn.net/weixin_40593658/article/details/90181471)



## 02 语言模型各种概念

## 03 文本矩阵化

### 3.1 分词--结巴分词

结巴分词的学习主要参考[“结巴”中文分词：做最好的 Python 中文分词组件](https://github.com/fxsjy/jieba)

结巴分词组件的安装我由于是用的anaconda, 所以使用conda install -c conda-forge jieba:

```
$ conda install -c conda-forge jieba
CondaIOError: Missing write permissions in: /home/terence/anaconda3
# 显示没有写入anaconda3的权限,解决error:chown更改目录或文件的用户名和用户组, 这样就有写入权限了
$ sudo chown -R terence /home/terence/anaconda3
# 修改权限后再次安装jieba组件
$ conda install -c conda-forge jieba           # 成功
```





## 04 Datawhale Task2 要求:数据集探索 (2 days)

【**任务2-自然语言处理**】时长：2天

Task2 特征提取 (2 days)

1. 基本文本处理技能  

   1.1 分词的概念（分词的正向最大、逆向最大、双向最大匹配法）； 

   1.2 词、字符频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）]

2. 

   2.1 语言模型中unigram、bigram、trigram的概念；  
   2.2 unigram、bigram频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）
   
3. 文本矩阵化：要求采用词袋模型且是词级别的矩阵化步骤有： 
   3.1 分词（可采用结巴分词来进行分词操作，其他库也可以）；  
   3.2 去停用词；构造词表。   
   
   3.3 每篇文档的向量化。