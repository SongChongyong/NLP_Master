# coding:utf-8
# reference:https://www.cnblogs.com/pinard/p/6908150.html

import jieba
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)

# ======首先我们进行分词，并把分词结果分别存在nlp_test1.txt, nlp_test3.txt和 nlp_test5.txt
#第一个文档分词#
with open('./lad_data/nlp_test0.txt') as f:
    document = f.read()
    
    document_decode = document
    document_cut = jieba.cut(document_decode)
    #print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
    result = ' '.join(document_cut)
#     result = result.encode('utf-8')
    with open('./lad_data/nlp_test1.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()

#第二个文档分词#
with open('./lad_data/nlp_test2.txt') as f:
    document2 = f.read()
    
    document2_decode = document2
    document2_cut = jieba.cut(document2_decode)
    #print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
#     result = result.encode('utf-8')
    with open('./lad_data/nlp_test3.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close() 

#第三个文档分词#
jieba.suggest_freq('桓温', True)
with open('./lad_data/nlp_test4.txt') as f:
    document3 = f.read()
    
    document3_decode = document3
    document3_cut = jieba.cut(document3_decode)
    #print  ' '.join(jieba_cut)
    result = ' '.join(document3_cut)
#     result = result.encode('utf-8')
    with open('./lad_data/nlp_test5.txt', 'w') as f3:
        f3.write(result)
f.close()
f3.close()  


# ===读入分好词的数据到内存备用，并打印分词结果观察：
with open('./lad_data/nlp_test1.txt') as f3:
    res1 = f3.read()
print (res1)
with open('./lad_data/nlp_test3.txt') as f4:
    res2 = f4.read()
print (res2)
with open('./lad_data/nlp_test5.txt') as f5:
    res3 = f5.read()
print (res3)
'''
沙瑞金 赞叹 易学习 的 胸怀 ， 是 金山 的 百姓 有福 ， 可是 这件 事对 李达康 的 触动 很大 。 易学习 又 回忆起 
他们 三人 分开 的 前一晚 ， 大家 一起 喝酒 话别 ， 易学习 被 降职 到 道口 县当 县长 ， 王大路 下海经商 ， 
李达康 连连 赔礼道歉 ， 觉得 对不起 大家 ， 他 最 对不起 的 是 王大路 ， 就 和 易学习 一起 给 王大路 凑 了 
5 万块 钱 ， 王大路 自己 东挪西撮 了 5 万块 ， 开始 下海经商 。 没想到 后来 王大路 竟然 做 得 风生水 起 。 
沙瑞金 觉得 他们 三人 ， 在 困难 时期 还 能 以沫 相助 ， 很 不 容易 。 

沙瑞金 向 毛娅 打听 他们 家 在 京州 的 别墅 ， 毛娅 笑 着 说 ， 王大路 事业有成 之后 ， 要 给 欧阳 菁 和 
她 公司 的 股权 ， 她们 没有 要 ， 王大路 就 在 京州 帝豪园 买 了 三套 别墅 ， 可是 李达康 和 易学习 都 不要 ， 
这些 房子 都 在 王大路 的 名下 ， 欧阳 菁 好像 去 住 过 ， 毛娅 不想 去 ， 她 觉得 房子 太大 很 浪费 ， 
自己 家住 得 就 很 踏实 。 

347 年 （ 永和 三年 ） 三月 ， 桓温 兵至 彭模 （ 今 四川 彭山 东南 ） ， 留下 参军 周楚 、 孙盛 看守 辎重 ， 
自己 亲率 步兵 直攻 成都 。 同月 ， 成汉 将领 李福 袭击 彭模 ， 结果 被 孙盛 等 人 击退 ； 而 桓温 三 战三胜 ， 
一直 逼近 成都 。 
'''

# =================从文件导入停用词表
stpwrdpath = "./lad_data/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list  
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

# ===把词转化为词频向量，注意由于LDA是基于词频统计的，因此一般不用TF-IDF来做文档特征
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
corpus = [res1,res2,res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)
print (cntTf)   # 输出即为所有文档中各个词的词频向量
'''
 (0, 44)    1
 (0, 75)    1
  ...
 (2, 81)    1
'''

# 做LDA主题模型了，由于我们只有三个文档，所以选择主题数K=2
lda = LatentDirichletAllocation(n_topics=2,
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)
# 通过fit_transform函数，我们就可以得到文档的主题模型分布在docres中。
# 而主题词分布则在lda.components_中。我们将其打印出来
print (docres)
# [[ 0.00950072  0.99049928]
#  [ 0.0168786   0.9831214 ]
#  [ 0.98429257  0.01570743]]
print (lda.components_)
'''
[[ 1.32738199  1.24830645  0.90453117  0.7416939   0.78379936  0.89659305
   1.26874773  1.23261029  0.82094727  0.87788498  0.94980757  1.21509469
   ...
1.10391419  1.26932908  1.26207274  0.70943937  1.1236972   1.24175001
   1.27929042  1.19130408]]
'''
