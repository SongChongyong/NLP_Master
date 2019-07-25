# coding: utf-8
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import bz2
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk # standard preprocessing
import operator # sorting items in dictionary by value
#nltk.download() #tokenizers/punkt/PY3/english.pickle
from math import ceil
import csv


# ## Dataset
# 从 http://www.evanjones.ca/software/wikipedia2text.html 下载数据
# 但是上面链接已经无法下载数据了，作者在GitHub上存储了数据 https://github.com/thushv89/data_packt_natural_language_processing_with_tensorflow


url = 'http://www.evanjones.ca/software/' # 这个链接现在失效了

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    print('Downloading file...')
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('wikipedia2text-extracted.txt.bz2', 18377035)


# ## Read Data without Preprocessing 
# Reads data as it is to a string and tokenize(分词) it using spaces and returns a list of words



def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""

  with bz2.BZ2File(filename) as f:
    data = []
    file_string = f.read().decode('utf-8')
    file_string = nltk.word_tokenize(file_string)
    data.extend(file_string)
  return data
  
words = read_data(filename)
print('Data size %d' % len(words))
print('Example words (start): ',words[:10])
print('Example words (end): ',words[-10:])
'''
Data size 11634270
Example words (start):  ['Propaganda', 'is', 'a', 'concerted', 
                       'set', 'of', 'messages', 'aimed', 'at', 'influencing']
Example words (end):  ['useless', 'for', 'cultivation', '.', 
                      'and', 'people', 'have', 'sex', 'there', '.']

'''
   
# ====Read Data with Preprocessing with NLTK 用NLTK读取数据并预处理

# Reads data as it is to a string, convert to lower-case and tokenize it using the nltk library. 
# 将数据转换为字符串，并转换为小写，同时使用nltk库进行分词、去除标点符号。 
 
def read_data(filename):
  """
  Extract the first file enclosed in a zip file as a list of words
  and pre-processes it using the nltk python library
  """
 
  with bz2.BZ2File(filename) as f:
 
    data = []
    file_size = os.stat(filename).st_size
    chunk_size = 1024 * 1024 # reading 1 MB at a time as the dataset is moderately large
    print('Reading data...')
    for i in range(ceil(file_size//chunk_size)+1):
        bytes_to_read = min(chunk_size,file_size-(i*chunk_size))
        file_string = f.read(bytes_to_read).decode('utf-8')
        file_string = file_string.lower()
        # tokenizes a string to words residing in a list
        file_string = nltk.word_tokenize(file_string)
        data.extend(file_string)
  return data
 
words = read_data(filename)
print('Data size %d' % len(words))
print('Example words (start): ',words[:10])
print('Example words (end): ',words[-10:])

'''
Data size 3361041
Example words (start):  ['propaganda', 'is', 'a', 'concerted', 'set', 
                         'of', 'messages', 'aimed', 'at', 'influencing']
Example words (end):  ['favorable', 'long-term', 'outcomes', 'for', 'around', 
                       'half', 'of', 'those', 'diagnosed', 'with']
'''

# ======== Building the Dictionaries 构建词典
# 以"I like to go to school"为例
# 
# * `dictionary`: maps a string word to an ID (e.g. {I:0, like:1, to:2, go:3, school:4})
# * `reverse_dictionary`: maps an ID to a string word (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
# * `count`: List of list of (word, frequency) elements (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
# * `data` : Contain the string of text we read, where string words are replaced with word IDs (e.g. [0, 1, 2, 3, 2, 4])
# 同时把所有不常用词汇换成代号"UNK"


# we restrict our vocabulary size to 50000
vocabulary_size = 50000 
# 使用collections.Counts统计单词列表中单词的频数
# 然后使用 most_common 方法取top50000频数的单词作为vocabulary
# 再创建一个dict，将 top5000词汇的vocabulary放入dictionary中，以便快速查询
# 接下来，将全部单词以频数排序，top50000词汇之外的单词，将其认定为Unkown(UNK)，将其编号为0，并统计这类词汇的数量
# 遍历单词列表，对其中的每一个单词，判断是否在dictionary中，在则转化为编号，不在则转化为0
# 最后返回转化后的编码(data)、每个单词的频数统计(count)、词汇表(dictionary)及其反转的形式(reverse_dictionary)。

def build_dataset(words):
  count = [['UNK', -1]]
  # Gets only the vocabulary_size most common words as the vocabulary
  # All the other words will be replaced with UNK token
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()

  # Create an ID for each word by giving the current length of the dictionary
  # And adding that item to the dictionary
  for word, _ in count:
    dictionary[word] = len(dictionary)
    
  data = list()
  unk_count = 0
  # Traverse through all the text we have and produce a list
  # where each element corresponds to the ID of the word found at that index
  for word in words:
    # If word is in the dictionary use the word ID,
    # else use the ID of the special token "UNK"
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
    
  # update the count variable with the number of UNK occurences
  count[0][1] = unk_count
  
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  # Make sure the dictionary is of size of the vocabulary
  assert len(dictionary) == vocabulary_size
    
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)


# 打印（1）vocabulary中最高频出现的词汇及其数量(包括unknown词汇)，
# 可以看到"UNK"这类一共有68859个，最常出现的"the"有226892，第二常出现的是","有184013，...；
# 打印(2) 我们的data中前10个单词为['propaganda', 'is', 'a', 'concerted', 'set', 'of', 'messages',
# 'aimed', 'at', 'influencing']，对应编号为[1721, 9, 8, 16476, 223, 4, 5166, 4457, 26, 11592]
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
del words  # 删除原始单词列表，以节约内存。
'''
Most common words (+UNK) [['UNK', 68859], ('the', 226892), 
                          (',', 184013), ('.', 120944), ('of', 116323)]
Sample data [1721, 9, 8, 16476, 223, 4, 5166, 4457, 26, 11592] ['propaganda', 'is', 
             'a', 'concerted', 'set', 'of', 'messages', 'aimed', 'at', 'influencing']
'''



