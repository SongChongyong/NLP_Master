from sklearn import preprocessing  
      
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1],[1, 0, 2]])

# enc.n_values_:每个特征值的特征数目
print("enc.n_values_ is:",enc.n_values_)
# enc.n_values_ is: [2 3 4]

# enc.feature_indices_:表明每个特征在one-hot向量中的坐标范围,0-2 是第一个特征，2-5就是第二个特征，5-9是第三个特
print("enc.feature_indices_ is:",enc.feature_indices_)
# enc.feature_indices_ is: [0 2 5 9]

# 打印特征值转化为 one-hot编码结果
print(enc.transform([[0, 1, 1]]).toarray()) #[[1. 0. 第0位   0. 1. 0.  第一位  0. 1. 0. 0. 第一位]]
print(enc.transform([[1, 1, 1]]).toarray())
print(enc.transform([[1, 2, 1]]).toarray())
'''
[[1. 0. 0. 1. 0. 0. 1. 0. 0.]]
[[0. 1. 0. 1. 0. 0. 1. 0. 0.]]
[[0. 1. 0. 0. 1. 0. 1. 0. 0.]]
'''
        