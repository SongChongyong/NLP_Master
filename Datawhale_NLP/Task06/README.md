

# 【**任务6-自然语言处理**】

## 01 FastText的原理


$$
p(x_1...x_n)=\prod_{i=1}^nq(x_i|x_{i-2},x_{i-1})
$$


##  02 利用FastText模型进行文本分类







##  03 DataWhale 任务6要求 

**Task6 神经网络基础 (2 days )**

 建议第一天基础，第二天FastText

- 前馈神经网络、网络层数、输入层、隐藏层、输出层、隐藏单元、9激活函数的概念。
- 感知机相关；定义简单的几层网络（激活函数sigmoid），递归使用链式法则来实现反向传播。
- 激活函数的种类以及各自的提出背景、优缺点。（和线性模型对比，线性模型的局限性，去线性化）
- 深度学习中的正则化（参数范数惩罚：L1正则化、L2正则化；数据集增强；噪声添加；early stop；Dropout层）、正则化的介绍。
- 深度模型中的优化：参数初始化策略；自适应学习率算法（梯度下降、AdaGrad、RMSProp、Adam；优化算法的选择）；batch norm层（提出背景、解决什么问题、层在训练和测试阶段的计算公式）；layer norm层。
- FastText的原理。
- 利用FastText模型进行文本分类。
- [fasttext1](https://github.com/facebookresearch/fastText#building-fasttext-for-python) [fasttext2](https://github.com/salestock/fastText.py) [fasttext3 其中的参考](https://jepsonwong.github.io/2018/05/02/fastText/)

 





---
**参考**：
1. 吴军著《数学之美》

2. **Michael Collins Notes**：[Michael Collins Notes](<http://www.cs.columbia.edu/~mcollins/lm-spring2013.pdf>)

3. **peghoty博客**： [word2vec 中的数学原理详解（三）背景知识](<https://blog.csdn.net/itplus/article/details/37969817>)

4. [特征提取方法: one-hot 和TF-IDF](https://www.cnblogs.com/lianyingteng/p/7755545.html)

5. [用gensim学习word2vec](https://www.cnblogs.com/pinard/p/7278324.html)