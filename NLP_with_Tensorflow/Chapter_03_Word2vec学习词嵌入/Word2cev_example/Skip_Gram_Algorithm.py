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
url = 'http://www.evanjones.ca/software/'
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


# === Read Data without Preprocessing 
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""

  with bz2.BZ2File(filename) as f:
    data = []
    file_string = f.read().decode('utf-8')
    file_string = nltk.word_tokenize(file_string)
    data.extend(file_string)
  return data
  
words = read_data(filename)


# === Read Data with Preprocessing with NLTK
# 将数据转换为字符串，并转换为小写，同时使用nltk库进行分词、去除标点符号。
def read_data(filename):
  """
  Extract the first file enclosed in a zip file as a list of words
  and pre-processes it using the nltk python library
  """

  with bz2.BZ2File(filename) as f:

    data = []
    file_size = os.stat(filename).st_size
    chunk_size = 1024 * 1024 
    print('Reading data...')
    for i in range(ceil(file_size//chunk_size)+1):
        bytes_to_read = min(chunk_size,file_size-(i*chunk_size))
        file_string = f.read(bytes_to_read).decode('utf-8')
        file_string = file_string.lower()
        file_string = nltk.word_tokenize(file_string)
        data.extend(file_string)
  return data

words = read_data(filename)

# ======== Building the Dictionaries 构建词典
# 同时把所有不常用词汇换成代号"UNK"

# we restrict our vocabulary size to 50000
vocabulary_size = 50000 

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()

  for word, _ in count:
    dictionary[word] = len(dictionary)
    
  data = list()
  unk_count = 0

  for word in words:
    # If word is in the dictionary use the word ID,
    # else use the ID of the special token "UNK"
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
    
  count[0][1] = unk_count
  
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  assert len(dictionary) == vocabulary_size
    
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

del words  # 删除原始单词列表，以节约内存。

# ========Generating Batches of Data for Skip-Gram
# 生成Word2vec的训练样本

data_index = 0

# 定义generate_batch_skip_gram函数用来生成训练用的batch数据
# 参数batch_size为batch的大小，window_size指单词最远可以联系的距离，设置为1代表上下文单词为紧邻的两个单词
def generate_batch_skip_gram(batch_size, window_size):
  # 定义单词序号data_index为global变量
  # 因为我们会反复调用generate_batch_skip_gram，所以要确保data_index可以在generate_batch_skip_gram函数中被修改
  global data_index 
    
  # 两个数组来保存 目标单词(batch) 和上下文单词 (labels)
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    
  # span 为对某个单词创建样本时会使用到的单词数量，包括目标单词本身和它的上下文单词，所以span = s*window_size+1
  span = 2 * window_size + 1 
    
  # 并创建一个最大容量为span的deque，即双向队伍，在对deque使用append方法添加变量时，只会保留最后插入的span个变量
  buffer = collections.deque(maxlen=span)
  
  # Fill the buffer and update the data_index
  # 从序号data_index开始，把span个单词顺序读入buffer作为初始值
  # 因为buffer的容量为span的deque，所以此时buffer已填满，后续数据将替换掉前面的数据
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  
  # num_samples为对每个单词生成多少个上下文单词
  num_samples = 2*window_size 

  #进入第一层循环，每次循环内对一个目标单词生成上下文单词
  for i in range(batch_size // num_samples):
    k=0
    # avoid the target word itself as a prediction
    # fill in batch and label numpy arrays
    for j in list(range(window_size))+list(range(window_size+1,2*window_size+1)):
      batch[i * num_samples + k] = buffer[window_size]
      labels[i * num_samples + k, 0] = buffer[j]
      k += 1 
    
    # 在对一个目标单词生成所有上下文单词后，我们再读入下一个单词(同时抛掉buffer中第一个单词)，即把滑窗向后移动一位
    # 这样我们的目标单词也向后移动一个，上下文单词也整体后移，便可以开始生成下一个目标单词的训练样本
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

# 分别用window_size为1和2时，测试一下对于data中前8个单词，对应的batch和labels
print('data:', [reverse_dictionary[di] for di in data[:8]])

for window_size in [1, 2]:
    data_index = 0
    batch, labels = generate_batch_skip_gram(batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' %window_size)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

'''
data: ['propaganda', 'is', 'a', 'concerted', 'set', 'of', 'messages', 'aimed']

with window_size = 1:
    batch: ['is', 'is', 'a', 'a', 'concerted', 'concerted', 'set', 'set']
    labels: ['propaganda', 'a', 'is', 'concerted', 'a', 'set', 'concerted', 'of']

with window_size = 2:
    batch: ['a', 'a', 'a', 'a', 'concerted', 'concerted', 'concerted', 'concerted']
    labels: ['propaganda', 'is', 'concerted', 'set', 'is', 'a', 'set', 'of']

如对“propaganda is a concerted set of messages aimed"
用窗口大小为1创建的(batch,labels)是 (is,progaanda),(is,a),(a,is),(a,concerted)...
用窗口大小为2创建的(batch,labels)是
    (a a,proganda is),(a a,concerted set),
    (concerted concert, is a),(concerted concert, set of)...
'''

# =========================== Skip-Gram Algorithm

#   ===(1)定义超参数
batch_size = 128 
embedding_size = 128 # 单词转为稠密向量时的维度，一般是50~1000内的值
window_size = 4 

# 生成验证数据valid_example，这里随机抽取一些频数最高的单词和一些罕见的单词，看向量空间上跟它们最近的单词是否相关性比较高。
valid_size = 16 # 指用来抽取的验证单词书
valid_window = 50 # 指从一个50的大窗口随机抽取数据

valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)

num_sampled = 32 # 训练时用来做负样本的噪声单词的数量

#  ===(2) 定义输入和输出
# 先创建一个tf.Graph()并设置为默认graph
tf.reset_default_graph()

train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
#  ===(3) 定义模型参数和变量
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    
softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=0.5 / math.sqrt(embedding_size))
                            )
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))

# ===(4) 定义模型计算
embed = tf.nn.embedding_lookup(embeddings, train_dataset)

# Compute the softmax loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(
        weights=softmax_weights, biases=softmax_biases, inputs=embed,
        labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size)
)

# ===(5) Calculating Word Similarities 计算单词的相似性

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    
# ===(6) Model Parameter Optimizer 定义优化器
# Optimizer.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

# ===(7) Running the Skip-Gram Algorithm 运行算法

num_steps = 100001  # 最大迭代次数
skip_losses = []
# ConfigProto is a way of providing various configuration settings 
# required to execute the graph
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
  # 初始化
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0
  
  # 进行num_step次迭代
  for step in range(num_steps):
    
    # 用generate_batch_skip_gram生成一个batch的data和labels数据。并用它们创建feed_dict
    batch_data, batch_labels = generate_batch_skip_gram(
      batch_size, window_size)
    
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    # 使用session.run()执行一次优化器运算和损失计算
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    
    # 把这一步训练的loss累计到average_loss
    average_loss += l
    
    # 每2000词循环，计算一下平均loss并打印
    if (step+1) % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      
      skip_losses.append(average_loss)
      # 打印平均loss
      print('Average loss at step %d: %f' % (step+1, average_loss))
      average_loss = 0
    
    # 每10000词循环，计算一次验证单词和全部单词的相似度，并将与每个验证单词最相似的8个单词打印出来
    if (step+1) % 10000 == 0:
      sim = similarity.eval()
      # Here we compute the top_k closest words for a given validation word
      # in terms of the cosine distance
      # We do this for all the words in the validation set
      # Note: This is an expensive step
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # 最相似单词数
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  skip_gram_final_embeddings = normalized_embeddings.eval()


# 保存训练的结果
np.save('skip_embeddings',skip_gram_final_embeddings)

with open('skip_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_losses)

# =============================================== 可视化Skip-Gram算法的学习

# ===(1) Finding Only the Words Clustered Together Instead of Sparsely Distributed Words
def find_clustered_embeddings(embeddings,distance_threshold,sample_threshold):
    ''' 
    Find only the closely clustered embeddings. 
    This gets rid of more sparsly distributed word embeddings and make the visualization clearer
    This is useful for t-SNE visualization
    
    distance_threshold: maximum distance between two points to qualify as neighbors
    sample_threshold: number of neighbors required to be considered a cluster
    '''
    
    # calculate cosine similarity
    cosine_sim = np.dot(embeddings,np.transpose(embeddings))
    norm = np.dot(np.sum(embeddings**2,axis=1).reshape(-1,1),np.sum(np.transpose(embeddings)**2,axis=0).reshape(1,-1))
    assert cosine_sim.shape == norm.shape
    cosine_sim /= norm
    
    # make all the diagonal entries zero otherwise this will be picked as highest
    np.fill_diagonal(cosine_sim, -1.0)
    
    argmax_cos_sim = np.argmax(cosine_sim, axis=1)
    mod_cos_sim = cosine_sim
    # find the maximums in a loop to count if there are more than n items above threshold
    for _ in range(sample_threshold-1):
        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        mod_cos_sim[np.arange(mod_cos_sim.shape[0]),argmax_cos_sim] = -1
    
    max_cosine_sim = np.max(mod_cos_sim,axis=1)

    return np.where(max_cosine_sim>distance_threshold)[0]

# ===(2) Computing the t-SNE Visualization of Word Embeddings Using Scikit-Learn
# use a large sample space to build the T-SNE manifold 
# and then prune it using cosine similarity
num_points = 1000 

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

print('Fitting embeddings to T-SNE. This can take some time ...')
# get the T-SNE manifold
selected_embeddings = skip_gram_final_embeddings[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('Pruning the T-SNE embeddings')
# prune the embeddings by getting ones only more than n-many sample above the similarity threshold
# this unclutters the visualization
selected_ids = find_clustered_embeddings(selected_embeddings,.25,10)
two_d_embeddings = two_d_embeddings[selected_ids,:]

print('Out of ',num_points,' samples, ', selected_ids.shape[0],' samples were selected by pruning')

# ===(3) Plotting the t-SNE Results with Matplotlib
def plot(embeddings, labels):
  
  n_clusters = 20 # number of clusters
  # automatically build a discrete set of colors, each for cluster
  label_colors = [pylab.cm.spectral(float(i) /n_clusters) for i in range(n_clusters)]
  
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  
  # Define K-Means
  kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(embeddings)
  kmeans_labels = kmeans.labels_
  
  pylab.figure(figsize=(15,15))  # in inches
    
  # plot all the embeddings and their corresponding words
  for i, (label,klabel) in enumerate(zip(labels,kmeans_labels)):
    x, y = embeddings[i,:]
    pylab.scatter(x, y, c=label_colors[klabel])    
        
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom',fontsize=10)

  # use for saving the figure if needed
  #pylab.savefig('word_embeddings.png')
  pylab.show()

words = [reverse_dictionary[i] for i in selected_ids]
plot(two_d_embeddings, words)



