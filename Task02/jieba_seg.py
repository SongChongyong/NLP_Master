# coding:utf-8
import jieba

'''
jieba.cut("需要分词的字符串", cut_all参数, HMM参数)
cut_all=True表示采用全模式, cut_all=False表示采用精确模式(默认)
HMM 参数用来控制是否使用 HMM 模型
'''
# 全模式
seg_list1 = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list1))  
# Full Mode: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学

# 精确模式(默认)
seg_list2 = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list2))  
# Default Mode: 我/ 来到/ 北京/ 清华大学

# 搜索引擎模式
'''
jieba.cut_for_search 方法用于搜索引擎构建倒排索引的分词，粒度比较细;
接受两个参数：需要分词的字符串；是否使用 HMM 模型.
'''
seg_list3 = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在京都大学深造")  
print(", ".join(seg_list3))
# 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, ，, 后, 在, 京都, 大学, 京都大学, 深造



