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

# GloVe and Word Senses

[TOC]

---

## 1. GloVe: Global Vectors for Word Representation

### 1. 之前的词向量方法

1. Matrix Factorization Methods
global matrix factorization(全局的矩阵分解)
基于共现矩阵的方法：首先获取所有单词间的共现矩阵(co-occurrence)，然后使用奇异值分解(SVD)进行降维得到词向量

> 优点：可以利用全局的统计信息(golbal statistical information)
> 缺点：在单词类别任务(word analogy)上表现很差，即无法有效地捕捉单词间的语义关系

基于矩阵分解的方法都是使用了**低秩近似low-rank approximations**的原理来分解大的矩阵来获取语料中的统计信息，不过不同的方法中矩阵的形式是不同的，如在latent semantic analysis (LSA)中，矩阵是"term-doucment"的形式。

2. Shallow Window-Based Methods
local context window(局部窗口)来预测单词
Mikolov论文中提出的skip-gram和continuous bag-of-words利用上下文和中心词关系来学习词向量，这些向量可以学习到语言模式(linguistic patterns)和单词向量间的线性关系(linear relationships between word vectors)，这里语言模式可以包含句法和语义(syntasitc and semantic)。

>优点：在单词类别任务中表现良好，即可以有效地捕捉单词的语义，换言之就是在单词空间中可以包含有语义信息的子结构
>缺点：模型是通过遍历语料的形式来学习词向量，无法利用全局的语料统计信息

### 2. GloVe模型

之前的两种方法各有利弊，所以GloVe模型是结合其各自的优点得出的
思路：

- 通过直接在**共现矩阵**上计算来利用语料的全局统计信息
- 直接计算**上下文的相关概率**来实现利用局部上下文信息

global log-bilinear regression(全局对数双线性回归)

符号声明：
>$X$表示单词的共现矩阵或计数(word-word co-occurrence counts)
$X_{ij}$代表单词$j$出现在单词$i$上下文中的次数
$X_i = \sum_k X_{ik}$代表任意单词$k$出现在单词$i$上下文中的次数总和
$P_{ij}=P(j \mid i)=\frac{X_{ij}}{X_i}$是单词$j$出现在单词$i$上下文中的概率

**步骤：**
(1) 使用共现概率的比率来描述该模型
Glove模型认为使用共现概率的比值比直接使用共现概率的效果好，具体看原文中ice和steam概率和比率的表格1如下：
![ratio](imgs/ratio_co_coourrence.jpg)
使用下面的公式来表示共现概率的比值：
$$
F\left(w_{i}, w_{j}, \tilde{w}_{k}\right)=\frac{P_{i k}}{P_{j k}} \tag{1}
$$
其中$w \in R^d$是词向量，$\tilde{w} \in R^d$是**分开的上下文词向量**，即上述3个词向量来自于两个词嵌入矩阵，等号右边是直接更加共现矩阵求得的，$P_{i k}$和$P_{j k}$分别是在中心词为$i$和$j$时上下文是$k$的概率。
(2)确定函数$F$的形式
$F$函数为了为了在词向量空间表示出比率$P_{i k}/P_{j k}$，$w_i$和$w_j$做差，然后与$\tilde{w}$做内积，这种方式可以更好体现函数$F$的线性，并且更好将词向量各个维度对应起来，公式如下：
$$
F\left(\left(w_{i}-w_{j}\right)^{T} \tilde{w}_{k}\right)=\frac{P_{i k}}{P_{j k}} \tag{2}
$$
在语料中每一个词即可以是中心词也可以是上下文，所以单词共现矩阵$X$是对称矩阵的，因此模型需要保证各个词向量是可交换的，函数$F$是同态的([homomorphism](https://zh.wikipedia.org/wiki/%E5%90%8C%E6%80%81))，得到如下公式(3)：
同态满足下面性质：
```
f(a + b) = f(a) + f(b)
f(a + b) = f(a) * f(b)
```
$\color{red}{仍有疑问}$，公式(3)怎么得出的
$$
F\left(\left(w_{i}-w_{j}\right)^{T} \tilde{w}_{k}\right)=\frac{F\left(w_{i}^{T} \tilde{w}_{k}\right)}{F\left(w_{j}^{T} \tilde{w}_{k}\right)} \tag{3}
$$
根据公式(2)得下面公式：
$$
F\left(w_{i}^{T} \tilde{w}_{k}\right)=P_{i k}=\frac{X_{i k}}{X_{i}} \tag{4}
$$
则公式(3)的解为$F=exp()$：
$$
w_{i}^{T} \tilde{w}_{k}=\log \left(P_{i k}\right)=\log \left(X_{i k}\right)-\log \left(X_{i}\right) \tag{5}
$$
上述公式中$log(X_i)$的存在破坏了可交换对称性(exchange symmetry)，但该项是独立于$k$的，所以使用下面的公式来实现可交换对称：
$$
w_{i}^{T} \tilde{w}_{k}+b_{i}+\tilde{b}_{k}=\log \left(X_{i k}\right) \tag{6}
$$
其中$b_i$与$log(X_i)$等价，$\tilde{b}_k$的存在是为了保证$\tilde{w}$的对称性，因此交换$i$和$k$的值，上述公式的结果完全不变。
(3)确定损失函数形式
在公式(6)中存在两个问题：
1. 当$X_{ij}=0$时函数是无意义的，但在实际中存在大量的词直接是没有共现关系的，因此等于0的现象大量存在，可以考虑给它加个1：$log(X_{ik}) \rightarrow log(1+X_{ik})$
2. 该公式中所有的共现的结果视为相同的(weights all co-occurrences equally)
实际中有的单词间共现很少甚至没有，所以每个共现都应该使用不同的权重

综上得到下面公式：
$$
J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2} \tag{7}
$$
其中$V$是字典的大小，上述公式中损失函数趋于使公式(6)的结果成立，权重函数$f$满足下面的属性：
1. $f(0)=0$，保证在单词共现计数为0时不会产生损失
2. $f(x)$应该是非减函数，这样低频的共现性单词的权重不会偏大(rare co-occurrences are not overweighted)
3. $f(x)$对于大的共现计数应该相对较小，为了防止高频词的权重过大( frequent co-occurrences are not overweighted)

这里的$f(x)$的形式如下：
![](imgs/function.jpg)
$$
f(x)=\left\{\begin{array}{cl}{\left(x / x_{\max }\right)^{\alpha}} & {\text { if } x<x_{\max }} \\ {1} & {\text { otherwise }}\end{array}\right. \tag{8}
$$
这里的$\alpha$取$3/4$，这个值与负采样论文中用一元模型的指数一样，$x_{max}=100$。