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

# CS224n Assignment2 word2vec

### 1. Written: Understanding word2vec

(a)
$$
-\sum_{w \in \text {Vocab}} y_{w} \log \left(\hat{y}_{w}\right)=-\log \left(\hat{y}_{o}\right) \tag{1}
$$
因为 $\boldsymbol{y}$ 是独热向量(one-hot vector), 当且仅当 $i=o$ 时 $y_i=1$, 其他情况下都是$y_w=0$，如下所示:
$$
y_{i}=\left\{\begin{array}{ll}{1,} & {\text { if } i=o} \\ {0,} & {\text { otherwise }}\end{array}\right. \tag{2}
$$

(b)
$$
\begin{aligned}
\frac{\partial \boldsymbol{J}_{\text { naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{c}} &=-\boldsymbol{u}_{o} + \frac{\sum_{i=1}^{V}{u_i^Tv_cu_i}}{\sum_{w=1}^{V}{u_w^Tv_c}}
\\ &= -\boldsymbol{u}_{o}+\sum_{w=1}^{V} \hat{y}_{w} \boldsymbol{u}_{w} \tag{3}
\end{aligned}
$$
注意这里的$\hat{y}_{w}$是单词$w$归一化后的概率，所以加号右边相当于对每个上下文单词向量$u_w$进行**加权平均**，与正确的上下文向量$u_o$进行叠加，相当于用加权后的上下文向量来修正正确的上下文向量，最终修正后的上下文向量得到损失函数对中心词的梯度
上面的公式(3)进过化简得：
$$
\frac{\partial \boldsymbol{J}_{\text { naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{c}}=\boldsymbol{U}(\hat{\boldsymbol{y}}-\boldsymbol{y}) \tag{4}
$$
上面的公式中使用$\hat{\boldsymbol{y}}-\boldsymbol{y}$计算预测概率分布和真实概率分布的差，然后与上下文嵌入矩阵做内积得到对中心词的梯度，维度为$d \times 1$，其中$d$为词嵌入的维度

(c)
当$w=o$时：
$$
\begin{aligned}
\frac{\partial \boldsymbol{J}_{\text { naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_o} &=-v_c + \frac{exp(u_o^Tv_c)v_c}{\sum_{w=1}^Vexp(u_w^Tv_c)}\\
&=-v_c + \hat{y}_o v_c \tag{5}
\end{aligned}
$$
当$w \neq o$时：
$$
\begin{aligned}
\frac{\partial \boldsymbol{J}_{\text { naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_w}&=\frac{exp(u_w^Tv_c)v_c}{\sum_{w=1}^Vexp(u_w^Tv_c)}\\
&=\hat{y}_w v_c \tag{6}
\end{aligned}
$$
将上面的两个式子融合为一个式子：
$$
\frac{\partial \boldsymbol{J}_{\text { naive-softmax }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{U}}=\boldsymbol{v}_{c}(\hat{\boldsymbol{y}}-\boldsymbol{y})^{\top} \tag{7}
$$
通过计算中心词词向量$v_c$与预测分布和真实分布差的外积，得到$d \times V$大小的梯度，其中$d$为词嵌入的维度，$V$为字典大小，该维度与$U$一致

(d)
sigmoid函数的微分：
$$
\sigma^{\prime}(\boldsymbol{x})=\sigma(\boldsymbol{x})(1-\sigma(\boldsymbol{x})) \tag{8}
$$
注意 $\sigma$ 函数在$(0,1)$内，其微分后仍然为正数，所以公式(8)中一定是$1-\sigma(x)$

(e)
使用Negative Sampling loss代替Naive Softmax loss，负采样的损失如下：
$$
J_{\text { neg-sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)=-\log \left(\sigma\left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)\right)-\sum_{k=1}^{K} \log \left(\sigma\left(-\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)\right) \tag{9}
$$
该损失函数中将$u_o$和$v_c$当作正样本，负采样的上下文$u_k$和$v_c$当作负样本，所以使用$\sigma$函数的逻辑回归来进行分类，所以上面负样本对应的内积前要加个负号
$$
\frac{\partial \boldsymbol{J}_{\text { neg-sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{c}}=\left(\sigma\left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)-1\right) \boldsymbol{u}_{o}+\sum_{k=1}^{K}\left(1-\sigma\left(-\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)\right) \boldsymbol{u}_{k} \tag{10}
$$
$$
\frac{\partial \boldsymbol{J}_{\text { neg-sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{o}}=\left(\sigma\left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)-1\right) \boldsymbol{v}_{c} \tag{11}
$$
$$
\frac{\partial \boldsymbol{J}_{\text { neg-sample }}\left(\boldsymbol{v}_{c}, o, \boldsymbol{U}\right)}{\partial \boldsymbol{u}_{k}}=\left(1-\sigma\left(-\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)\right) \boldsymbol{v}_{c} \tag{12}
$$
在naive softmax的损失函数中，$\hat{y}$计算时需要归一化，如果字典很大则计算量很大，而在negative sample负采样的损失中，只需要计算正样本和负样本的部分，而且不需要归一化操作，因此可以大幅减少运算量

(f)
当中心词$c=w_{t}$时，上下文窗口内单词为$\left[w_{t-m}, \ldots, w_{t-1}, w_{t}, w_{t+1}, \ldots w_{t+m} \right]$，其中$m$是滑动窗口的大小，总的损失函数为：
$$
J_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right)=\sum_{-m \leq j \leq m \atop j \neq 0} \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right) \tag{13}
$$
i. 对上下文矩阵的梯度如下：

$$
\begin{aligned}
\frac{\partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right)}{\partial \boldsymbol{U}}&=\sum_{-m \leq j \leq m, j \neq 0} \frac{\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)}{\partial \boldsymbol{U}} \\
&= \sum_{-m \leq j \leq m, j \neq 0} \boldsymbol{v}_{c}(\hat{\boldsymbol{y}}_{t+j}-\boldsymbol{y}_{t+j})^{\top} \tag{14}
\end{aligned}
$$
上述的式子中求和号里用公式(7)替换，注意这里对**整个上下文嵌入矩阵**求导
ii. 对中心词的梯度如下：
$$
\begin{aligned}
\frac{\partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{c}}&=\sum_{-m \leq j \leq m, j \neq 0} \frac{\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{c}} \\
&= \sum_{-m \leq j \leq m, j \neq 0} \boldsymbol{U}(\hat{\boldsymbol{y}}_{t+j}-\boldsymbol{y}_{t+j}) \tag{15}
\end{aligned}
$$
上述式子中求和好里用公式(4)替换，注意这里只对**中心词**求梯度

iii. 对**非中心词**的中心词向量$v_w$求梯度结果为0，因为损失函数中没有用到该变量
$$
\frac{\partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right)}{\partial \boldsymbol{v}_{w}}=0, \text { when } w \neq c \tag{16}
$$
