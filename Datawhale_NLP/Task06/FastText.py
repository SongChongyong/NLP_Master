# coding:utf-8
# 代码来源:https://blog.csdn.net/weixin_36604953/article/details/78324834
import logging
import fasttext

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

classifier = fasttext.train_supervised('./THUCNews_data/news_fasttext_train.txt',
        							   './THUCNews_data/news_fasttext.model',label_prefix='__label__')

classifier = fasttext.test_supervised('./THUCNews_data/news_fasttext.model.bin',label_prefix='__label__')
result = classifier.test('./THUCNews_data/news_fasttext_test.txt')
print(result.precision)
print(result.recall)



