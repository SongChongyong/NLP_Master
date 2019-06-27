from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
# 代码来自：https://www.cnblogs.com/pinard/p/6693230.html

corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 

# 方法一：用CountVectorizer类向量化之后再调用TfidfTransformer类进行预处理
vectorizer=CountVectorizer()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
# print (tfidf)

# 方法二：直接用TfidfVectorizer完成向量化与TF-IDF预处理
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer()
re = tfidf2.fit_transform(corpus)

# 得到语料库所有不重复的词
print(tfidf2.get_feature_names())
'''
['and', 'apple', 'car', 'china', 'come', 'in', 'is', 
'love', 'papers', 'polupar', 'science', 'some', 'tea', 
'the', 'this', 'to', 'travel', 'work', 'write']
'''

# 得到每个单词对应的id值
# print(tfidf2.vocabulary_)
'''
{'come': 4, 'to': 15, 'china': 3, 'travel': 16, 'this': 14, 'is': 6, 
'car': 2, 'polupar': 9, 'in': 5,'love': 7, 'tea': 12, 'and': 0, 'apple': 1, 
'the': 13, 'work': 17, 'write': 18, 'some': 11, 'papers': 8, 'science': 10}
'''

# 得到每个句子所对应的向量
# 向量里数字的顺序是按照词语的id顺序来的
print (re.toarray())
'''
[[0.         0.         0.         0.34884223 0.44246214 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.69768446 0.44246214 0.
  0.        ]
 [0.         0.         0.4533864  0.35745504 0.         0.35745504
  0.35745504 0.         0.         0.4533864  0.         0.
  0.         0.         0.4533864  0.         0.         0.
  0.        ]
 [0.5        0.5        0.         0.         0.         0.
  0.         0.5        0.         0.         0.         0.
  0.5        0.         0.         0.         0.         0.
  0.        ]
 [0.         0.         0.         0.         0.         0.28113163
  0.28113163 0.         0.35657982 0.         0.35657982 0.35657982
  0.         0.35657982 0.         0.28113163 0.         0.35657982
  0.35657982]]
'''

'''
两种方法的输出结果都是：
  (0, 4)    0.4424621378947393
  (0, 15)    0.697684463383976
  (0, 3)    0.348842231691988
  (0, 16)    0.4424621378947393
  (1, 3)    0.3574550433419527
  (1, 14)    0.45338639737285463
  (1, 6)    0.3574550433419527
  (1, 2)    0.45338639737285463
  (1, 9)    0.45338639737285463
  (1, 5)    0.3574550433419527
  (2, 7)    0.5
  (2, 12)    0.5
  (2, 0)    0.5
  (2, 1)    0.5
  (3, 15)    0.2811316284405006
  (3, 6)    0.2811316284405006
  (3, 5)    0.2811316284405006
  (3, 13)    0.3565798233381452
  (3, 17)    0.3565798233381452
  (3, 18)    0.3565798233381452
  (3, 11)    0.3565798233381452
  (3, 8)    0.3565798233381452
  (3, 10)    0.3565798233381452

'''        