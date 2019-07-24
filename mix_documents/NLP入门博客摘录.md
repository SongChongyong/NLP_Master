# NLP入门博客摘录





学习NLP，我建议第一步学language model， 然后依次学POS tagging， 语法分析PCFG，接着接触NLP的第一个实际应用，学习机器翻译（机器翻译真是融合了各种NLP知识到里面），先从基于统计的机器翻译开始学，IBM model1， IBM model 2，再到phrase based machine translation，然后再学log linear model。 再往后就可以学习各种应用啦，情感分析，文本分类等，这个可以上斯坦福的那门NLP课程，也是非常棒的课程。  

等学完了这两门课程，然后你可以去学cs224N， 体会一下深度学习在NLP领域的不同之处，再往后就是紧跟热点，多看看经典论文了。

作者：Cheeeen

链接：https://www.zhihu.com/question/266856019/answer/319002132



NLP四大任务类型：分类、序列标注、文本匹配、文本生成，都需要完整实现一遍。

对于实验室新生，我一般让他们实现五个练习来上手NLP。可以参考：

https://github.com/FudanNLP/nlp-beginner





## 怎么能表示自己自然语言处理入门了呢？**

那就是写一个**分类器**，我大三进入NLP实验室，听到新来的研究生师兄师姐们第一个任务总是写一个分类器。而我期间干了很多杂事以及上课，并没有真正的写过一个分类器。再加上考研的原因，我真正写一个自己基本都懂各种细节的文本分类器是在考完研的那个寒假。这个的功能就是给你一句话，你给这句话分个类即可。刚开始最好用CNN这个神经网络，因为这个简单。而你得需要数据，这个你可以去github上搜索，比如cnn text classification +自己喜欢用的框架（tensorflow，pytorch等），里面有代码，也基本会有数据。

作者：zenRRan

链接：https://www.zhihu.com/question/19895141/answer/379595051



第一门是Natural Language Processing，Chris和另一个大牛Dan Jurafsky共同讲授。这门课于2012年讲授，主要是从偏统计以及传统机器学习方法来入门NLP，对于了解NLP研究领域、应用场景以及基础方法十分有效。讲义在[https://web.stanford.edu/~jurafsky/NLPCourseraSlides.html](https://link.zhihu.com/?target=https%3A//web.stanford.edu/~jurafsky/NLPCourseraSlides.html)， 视频在Stanford Online官网

[https://www.youtube.com/playlist?list=PLoROMvodv4rOFZnDyrlW3-nI7tMLtmiJZ](https://link.zhihu.com/?target=https%3A//www.youtube.com/playlist%3Flist%3DPLoROMvodv4rOFZnDyrlW3-nI7tMLtmiJZ)。

作者：川陀学者

链接：https://www.zhihu.com/question/19895141/answer/549481321







这里YY几个未来可能会热门的NLP的应用：
**语法纠错**
目前文档编辑器（比如Word）只能做单词拼写错误识别，语法级别的错误还无能为力。现在学术领域最好的语法纠错系统的正确率已经可以接近50%了，部分细分错误可以做到80%以上，转化成产品的话很有吸引力吧~无论是增强文档编辑器的功能还是作为教学软件更正英语学习者的写作错误。

**结构化信息抽取**
输入一篇文章，输出的是产品名、售价，或者活动名、时间、地点等结构化的信息。NLP相关的研究很多，不过产品目前看并不多，我也不是研究这个的，不知瓶颈在哪儿。不过想象未来互联网信息大量的结构化、语义化，那时的搜索效率绝对比现在翻番啊~

**语义理解**
这个目前做的并不好，但已经有siri等一票语音助手了，也有watson这种逆天的专家系统了。继续研究下去，虽然离人工智能还相去甚远，但是离真正好用的智能助手估计也不远了。那时生活方式会再次改变。即使做不到这么玄乎，大大改进搜索体验是肯定能做到的~搜索引擎公司在这方面的投入肯定会是巨大的。

**机器翻译**
这个不多说了，目前一直在缓慢进步中~我们已经能从中获益，看越南网页，看阿拉伯网页，猜个大概意思没问题了。此外，口语级别的简单句的翻译目前的效果已经很好了，潜在的商业价值也是巨大的。

作者：White Pillow

链接：https://www.zhihu.com/question/26391679/answer/34169968

