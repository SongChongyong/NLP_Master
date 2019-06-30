# -*- coding: utf-8 -*-


from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 文本特征向量化模块
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn import svm 

def text_vec(X_train, X_test):
    vec = CountVectorizer()
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)
    return X_train, X_test
    

def pre_news(data):
    '''
        数据预处理：
            划分训练集，测试集
            文本特征向量化
    '''
    # 随机采样20%作为测试集
    X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                        data.target,
                                                        test_size = 0.2,
                                                        random_state=33)
    # 文本特征向量化
    X_train, X_test = text_vec(X_train, X_test)
    print(X_train.shape)
    
    return X_train, X_test, y_train, y_test

def train_model_svm(data, X_train, X_test, y_train, y_test):
    '''
        SVM进行新闻分类
    '''
    svm_clf = SGDClassifier(loss='hinge',
                            penalty='l2',
                            alpha=8e-5,
                            max_iter=5,
                            random_state=50)
    svm_clf.fit(X_train, y_train)
    y_pre = svm_clf.predict(X_test)
    # 输出准确率
    print('acc:', np.mean(y_test == y_pre))
    # 获取结果报告
    print(classification_report(y_test, y_pre, target_names=data.target_names))


def read_file(path):
    with open(path, 'r', encoding="UTF-8") as f:
        data = []
        labels = []
        for line in f:
            data.append(line.split('\t')[0])
            labels.append(line.split('\t')[1])
    return data, labels


def train_model_svc(data, target_names, X_train, X_test, y_train, y_test):
    '''
        SVM进行新闻分类
    '''
    svm_clf = svm.SVC(C=10.0,           # 惩罚系数
                      cache_size=200,       # 缓冲大小，限制计算量大小
                      class_weight=None,        # 权重设置
                      coef0=0.0,            # 核函数常数，默认为0
                      decision_function_shape=None,         # 原始的SVM只适用于二分类问题
                      degree=3,         # 当指定kernel为'poly'时，表示选择的多项式的最高次数，默认为三次多项式
                      gamma='auto',     # 核函数系数，默认是'auto'，那么将会使用特征位数的倒数，即1 / n_features
                      kernel='rbf',     # 算法中采用的核函数类型， 默认的是"RBF"，即径向基核，也就是高斯核函数
                      max_iter=-1,      # 最大迭代次数，默认-1
                      probability=False,    # 是否使用概率估计，默认是False
                      random_state=None,    # 在使用SVM训练数据时，要先将训练数据打乱顺序，用来提高分类精度，这里就用到了伪随机序列。如果该参数给定的是一个整数，则该整数就是伪随机序列的种子值
                      shrinking=True,       # 是否进行启发式
                      tol=0.001,            # 残差收敛条件，默认是0.0001
                      verbose=False)        # 是否启用详细输出
    svm_clf.fit(X_train, y_train)
    y_pre = svm_clf.predict(X_test)
    # 输出准确率
    print('acc:', np.mean(y_test == y_pre))
    # 获取结果报告
    print(classification_report(y_test, y_pre, target_names=target_names))


# 获取数据
news = fetch_20newsgroups(subset='all')
# 输出数据长度
print('len(nes):', len(news.data))
# 查看新闻类别
print(list(news.target_names))

X_train, X_test, y_train, y_test = pre_news(news)
train_model_svm(news, X_train, X_test, y_train, y_test)

data, labels = read_file('./merge.txt')
X_train, X_test, y_train, y_test = train_test_split(data, 
                                                    labels,
                                                    test_size = 0.2,
                                                    random_state=33)
X_train, X_test = text_vec(X_train, X_test)
target_names = ['class0', 'class1']
train_model_svc(data, target_names, X_train, X_test, y_train, y_test)

'''
len(nes): 18846
['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
(15076, 155283)
acc: 0.849602122016
                          precision    recall  f1-score   support

             alt.atheism       0.90      0.70      0.79       154
           comp.graphics       0.80      0.77      0.78       205
 comp.os.ms-windows.misc       0.81      0.83      0.82       201
comp.sys.ibm.pc.hardware       0.73      0.71      0.72       189
   comp.sys.mac.hardware       0.73      0.86      0.79       196
          comp.windows.x       0.83      0.89      0.86       225
            misc.forsale       0.87      0.79      0.83       198
               rec.autos       0.71      0.91      0.80       187
         rec.motorcycles       0.93      0.91      0.92       227
      rec.sport.baseball       0.93      0.93      0.93       206
        rec.sport.hockey       0.95      0.98      0.96       196
               sci.crypt       0.93      0.81      0.87       183
         sci.electronics       0.89      0.72      0.80       188
                 sci.med       0.94      0.87      0.90       189
               sci.space       0.91      0.92      0.92       183
  soc.religion.christian       0.81      0.91      0.86       184
      talk.politics.guns       0.78      0.96      0.86       197
   talk.politics.mideast       0.98      0.93      0.95       189
      talk.politics.misc       0.82      0.86      0.84       162
      talk.religion.misc       0.87      0.53      0.66       111

             avg / total       0.86      0.85      0.85      3770

acc: 0.729508196721
             precision    recall  f1-score   support

     class0       0.72      1.00      0.84       421
     class1       1.00      0.13      0.23       189

avg / total       0.81      0.73      0.65       610

'''
