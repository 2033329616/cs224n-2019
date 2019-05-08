<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

# 词向量(Word Vector)

[TOC]

---

## 1. Efficient Estimation of Word Representations in Vector Space

本文目地：从大量数据中学到continuous vector representations of words(word vectors)

结果：imporvements in accuracy at much lower computational cost

简单模型在大量数据集上训练的效果比复杂模型在少量数据集上的效果好$\rightarrow$改进模型

quality of the vector representations
1. similar words tends to be close to each other
2. words have **multiple degrees of similarity**(多种程度的相似性), eg. nouns can have multiple word endings

similarity of word representations goes beyond simple syntactic regularities$\rightarrow$词向量更加注重语义

**previous work**：
1.paper:*A neural probabilistic language model*
feedforward neural network + linear projection + non-linear => **learn jointly** word vector representation and a statistical language model
2.paper:*Language Modeling for Speech Recognition in Czech*
neural network with a single hidden layer to **only** learn the word vectors

在使用嵌入矩阵W提取one-hot对应的词向量时，虽然是可以通过查表法得到第$i$个字典位置词的词向量，但实际计算时使用1乘了$d$次，得到$d$维度的词向量

$$ [0,0,\dots,1,0]_V \cdot 
\left[
\begin{matrix} a_{1,1} & a_{1,2} & \dots & a_{1,d}\\
a_{2,1} & a_{2,2} & \dots & a_{2,d}\\
\vdots & \vdots & \ddots & \vdots\\
a_{V,1} & a_{V,2} & \dots & a_{V,d}
\end{matrix}
\right]
$$

之前NNLM语言模型中最复杂的部分是非线性隐藏层，但该部分保证了模型的性能=>为了减少复杂度同时保证精度不会太差，引入了两个新模型

- continuous bag-of-words(连续词袋模型)
- continuous skip-gram

![models](imgs/models.png)

使用*Semantic-Syntactic Word Relationship test set*来评估语言模型的质量(quality)
使用更多的数据+维度更大的词向量=>提高性能，但只增加一个方面，后面随着该变量的增加，模型性能提升的幅度会下降

*Microsoft Research Sentence Completion Challenge*该任务将句子中的某个词去掉，然后选择一个词，使其与句子的其他部分更加协调

对于out-of-the-list words，可以将一些词的向量进行平均，然后得到未知词汇的词向量

**总结**：
- neural network每层的计算复杂度为$N \times D$，$N$和$D$分别表示前一层的节点和当前层节点的个数
- 数据越多+词向量维度越大=>效果越好
- 分层softmax的复杂度为$log_2(V)$，$V$是字典大小
- 负采样和分层softmax不仅提高训练效率而且提高准确率
- 词向量关注：语义关系(向量的代数运算，queen示例)+句法形式(比较级、复数，etc)

## 2. Distributed Representations of Words and Phrases and their Compositionality

high-quality distributed vector representations
- capture precise syntactic
- capture semantic word relationships

inherent limitation:
- indifference to word order不注重词序
- inability to represent idiomatic phrases不能表示常用的短语(New York不是两个单词意思的组合)

词向量重要作用：聚集**相似单词**
通过学习得到的词向量可以包括**linguistic regularities and patterns**(语言学规则和模式)的信息
```
vec('Madrid') - vec('Spain') + vec('France) => vec('Paris)
```

在skip-gram模型中，在训练时对高频词进行采样(subsampling)的优点：
- 提升训练速度
- 提高对**低频词**的表示精度

将短语phrase也作为一个单独的token来训练得到向量，注意短语的向量不等于各个单词向量的拼接

相似性推理任务analogical reasoning task，即找到单词和短语对之间的关系：
```
vec('Montreal Canadiens') - vec('Montreal') + vec('Toronto') => ('Toronto Maple Leafs')
```

### 1. Hierarchical Softmax
将所有单词放到二叉树的节点上，不去优化输出的词向量而是优化各个节点的向量，模型复杂度$log_2(V)$，$V$是字典大小

单词为$w$，$n(w,j)$表示从根节点到叶子节点$w$路径中第$j\text{-}th$个节点，$L(w)$是路径的长度，$ch(n)$是父节点固定的一个子节点(例如左节点)，$[x]$在$x$为真时是1，否则为-1，$p(w_O | w_I)$的公式定义如下：

$$
p\left(w | w_{I}\right)=\prod_{j=1}^{L(w)-1} \sigma\left([n(w, j+1)=\operatorname{ch}(n(w, j))] \cdot v_{n(w, j)}^{\prime}{v_{w_{I}}} \right)
$$

计算的复杂度是$L(w_O)$，与输出单词所在的路径长度成比例，所有单词平均后的结果不会超过$logV$

使用霍夫曼树将高频的词放到路径短的叶子节点，提高训练速度

注意：
- 标准的softmax中，每一个词$w$都有两个向量，输入词向量$v_w^{'}$和输出词向量$v_w$
- 分层softmax中除了一个词向量$v_w$，二叉树的各个节点$v_n^{'}$也是词向量表示的一部分

### 2. Negative Sampling
#### 2.1 模型的优化目标
该方法可以用到skip-gram或者cbow，但道理类似，
对于skip-gram模型，需要从中心词预测上下文，所以负采样会从噪声分布中采样得到上下文负样本

在Noise Contrastive Estimation(NCE)模型中，好的模型可以有效的区分噪声和数据，所以这里模型的优化目标使，正样本属于数据集中的概率更大，负样本属于噪声分布的概率更大，样本的分类概率更接近真实结果，用极大似然概率的乘积来描述，取对数得下面结果：

$$
\log \sigma({v_{w_O}^{\prime}}^{\top} v_{w_I})+\sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_{n}(w)}\left[ \log \sigma ({v_{w_i}^{\prime}}^{\top} v_{w_I}) \right]
$$

$w_i$是从噪声分布采样得到的负样本，$v_{w_I}$是中心词的词向量，$v_{w_O}^{\prime}$是该中心词的上下文的词向量，而$v_{w_i}^{\prime}$是负样本的词向量，在小数据集中$k$取5-20，大数据集中$k$取2-5

这里负采样用NEG表示，与NCE不同：
1. 只使用采集的样本，而NCE会使用噪声分布的数值概率
2. NEG只计算词向量内积，而NCE需要最大化softmax的对数概率

这里的噪声分布$P_n(w)$使用unigram distribution一元模型的3/4次幂，即$U(w)^{3/4}/Z$效果更好，$Z$进行归一化

$$
P\left(w_{i}\right)=\frac{f\left(w_{i}\right)^{3 / 4}}{\sum_{j=0}^{n}\left(f\left(w_{j}\right)^{3 / 4}\right)}
$$

我的理解：首先要计算数据集中每个单词的词频$f(w_i)$，然后求得负采样概率$P(w_i)$，创建一个unigram table，这是很大的数组(应该大于原数据集的单词总数)，各个元素是每个单词对应的字典索引号，所以单词会出现重复，每个单词在这个数组中出现的次数是$w_i$在该table中重复$P(w_i)*table\_size$。在实际采样中，随机生成一个0到1亿的随机数，然后选择该数字对应的table位置的单词为negative word
$ \color{red}{unigram \space table怎么生成？} $

#### 2.2 高频词的子采样(subsampling)
作用：
- 加速学习的过程
- 提高**低频词汇**准确率
  
在skip-gram模型中，遇到高频词会根据一个概率而随机将其抛弃。
1. 论文中的概率公式：

$$
P\left(w_{i}\right)=1-\sqrt{\frac{t}{f\left(w_{i}\right)}}
$$

$P(w_i)$代表**抛弃**单词的概率，$f\left(w_{i}\right)$是单词$w_i$的频率frequency，即该单词出现的次数与总单词数的比值，$t$是一个选定的阈值

1. [博客](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)中的公式(word2vec程序中的公式)：

$$
P\left(w_{i}\right)=\left(\sqrt{\frac{z\left(w_{i}\right)}{0.001}}+1\right) \cdot \frac{0.001}{z\left(w_{i}\right)}
$$

$P(w_i)$代表**保留**单词的概率，$z(w_i)$是单词$w_i$在所有语料中的频率，即该单词出现的次数除以单词总个数，该函数曲线类似于$1/x$的形式，单词频率越高，保留的概率越小

论文中与程序中定义的不同，程序中更加权威

我的理解：
每个词的保留是由概率决定的，所以每次subsampling的结果是不同的，高频词会倾向于抛弃，但有时也会保留，这时计算后可以更新高频词的词向量，当抛弃的词使上下文时直接不参与中心词的计算，当抛弃词是上下文时直接不计算，而跳转到下个中心词

### 3. learning phrases

方法：找到在一些上下文中经常一起使用的单词组合，但在其他上下文中不经常使用，然后将这些短语当作一个词来训练词向量

$$
\operatorname{score}\left(w_{i}, w_{j}\right)=\frac{\operatorname{count}\left(w_{i} w_{j}\right)-\delta}{\operatorname{count}\left(w_{i}\right) \times \operatorname{count}\left(w_{j}\right)}
$$

$count(w_i)$表示单词$w_i$出现的次数，$count(w_i,w_j)$表示两个单词同时出现的次数，$\delta$是discounting coefficient用来排除一些短语中包含低频词的情况

**总结**：
- 对高频词的subsampling可以**提升训练速度**，并且提高**低频词**的词向量的表示能力。论文中表示高频词和其他词共现的频率很高，从反方向表示就是高频词的向量表示经过很多样本训练后不会显著改变(the vector representations of frequent words do not change significantly after training on several million examples)，因此通过子采样随机抛弃一些高频词，可以使低频词和高频词实现一些平衡，不但减少了训练样本而且提高了低频词的表示能力
- 负采样提升训练速度，而且提高了**高频词**和**低维度**的词向量表示能力，$\color{red}{为什么会对高频词和低纬度向量好？}$是因为高频词采样的次数多，所以更新词向量的次数多，所以高频词的词向量好？
- hierarchical softmax(分层softmax)对**低频词**的效果好，参考[关于word2vec，我有话要说](https://zhuanlan.zhihu.com/p/29364112)，CBOW是基于上下文词汇预测中心词，虽然某些单词词频较低，但它会收到上下文的影响，上下文的词向量效果很好的话，也会提升作为中心词的低频词的词向量的表示能力
