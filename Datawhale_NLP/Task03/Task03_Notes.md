# Task 01 Notes
## 01 TF-IDF原理  

此部分学习了吴军<数学之美>中[数学之美 系列九 -- 如何确定网页和查询的相关性](https://china.googleblog.com/2006/06/blog-post_3066.html)

单文本词频(Term Frequency):关键词的次数除以网页的总字数

即

需要给汉语中的每一个词给一个权重，这个权重的设定必须满足下面两个条件：

- 1. 一个词预测主题能力越强，权重就越大，反之，权重就越小。我们在网页中看到“原子能”这个词，或多或少地能了解网页的主题。我们看到“应用”一次，对主题基本上还是一无所知。因此，“原子能“的权重就应该比应用大。

- 2. 应删除词的权重应该是零。

在信息检索中，使用最多的权重是“逆文本频率指数” （Inverse document frequency 缩写为ＩＤＦ），它的公式为ｌｏｇ（Ｄ／Ｄｗ）其中Ｄ是全部网页数。





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

3.2 去停用词；构造词表

停用词就是句子中没什么必要的单词，去掉他们以后对理解整个句子的语义没有影响。文本中，会存在大量的虚词、代词或者没有特定含义的动词、名词，这些词语对文本分析起不到任何的帮助，我们往往希望能去掉这些“停用词”.

在英文中，例如，“a”，“the”,“to"，“their”等冠词，借此，代词… 我们可以直接用nltk中提供的英文停用词表.

```py
sentence = "this is a apple"
filter_sentence= [w for w in sentence.split(' ') if w not in stopwords.words('english')]
print(filter_sentence)
```





## 04 Datawhale Task3 特征选择 (2 days)

1. TF-IDF原理。
2. 文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）
3. 互信息的原理。
4. 使用第二步生成的特征矩阵，利用互信息进行特征筛选。

参考
[文本挖掘预处理之TF-IDF：文本挖掘预处理之TF-IDF - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/6693230.html
[使用不同的方法计算TF-IDF值：使用不同的方法计算TF-IDF值 - 简书](https://www.jianshu.com/p/f3b92124cd2b
[sklearn-点互信息和互信息：sklearn：点互信息和互信息 - 专注计算机体系结构 - CSDN博客](https://blog.csdn.net/u013710265/article/details/72848755
[如何进行特征选择（理论篇）机器学习你会遇到的“坑”：如何进行特征选择（理论篇）机器学习你会遇到的“坑” ](https://baijiahao.baidu.com/s?id=1604074325918456186&wfr=spider&for=pc