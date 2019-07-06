import nltk

# ===从NLTK的book模块加载所有的东西
from nltk.book import *

# ===搜索文本
print("\n查一下《白鲸记》(text1)中的单词'monstrous'：")
text1.concordance("monstrous")
'''
查一下《白鲸记》(text1)中的单词'monstrous'：
Displaying 11 of 11 matches:
ong the former , one was of a most monstrous size . ... This came towards us , 
ON OF THE PSALMS . " Touching that monstrous bulk of the whale or ork we have r
ll over with a heathenish array of monstrous clubs and spears . Some were thick
d as you gazed , and wondered what monstrous cannibal and savage could ever hav
that has survived the flood ; most monstrous and most mountainous ! That Himmal
they might scout at Moby Dick as a monstrous fable , or still worse and more de
th of Radney .'" CHAPTER 55 Of the Monstrous Pictures of Whales . I shall ere l
ing Scenes . In connexion with the monstrous pictures of whales , I am strongly
ere to enter upon those still more monstrous stories of them which are to be fo
ght have been rummaged out of this monstrous cabinet there is no telling . But 
of Whale - Bones ; for Whales of a monstrous size are oftentimes cast up dead u
'''

print("\n查一下《白鲸记》(text1)中的词'monstrous'相关的词")
text1.similar("monstrous")
print("\n查一下《理智与情感》(text2)中的词'monstrous'相关的词")
text2.similar("monstrous")
print("\n查一下《理智与情感》(text2)中的词'monstrous'和'very'相关的词")
text2.common_contexts(["monstrous", "very"])
'''
查一下《白鲸记》(text1)中的词'monstrous'相关的词
true contemptible christian abundant few part mean careful puzzled
mystifying passing curious loving wise doleful gamesome singular
delightfully perilous fearless

查一下《理智与情感》(text2)中的词'monstrous'相关的词
very so exceedingly heartily a as good great extremely remarkably
sweet vast amazingly

查一下《理智与情感》(text2)中的词'monstrous'和'very'相关的词
a_pretty am_glad a_lucky is_pretty be_glad
'''

# ====比较similar函数和common_contexts函数
print("\n查一下《理智与情感》(text2)中的词'monstrous'相关的词")
text2.similar("monstrous")
print("\n查一下《理智与情感》(text2)中的词'monstrous'相关的词")
text2.common_contexts(["monstrous"])
'''
查一下《理智与情感》(text2)中的词'monstrous'相关的词
very so exceedingly heartily a as good great extremely remarkably
sweet vast amazingly

查一下《理智与情感》(text2)中的词'monstrous'相关的词
a_pretty am_glad a_lucky is_pretty be_glad is_fond was_happy a_deal
'''

# 用离散图表示单词词在文本中的位置
# 绘制《美国总统就职演说》(text4)中词汇分布图
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

