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

# Backpropagation and Derivatives

[TOC]

---

## 1. Learning Representations by Backpropagating Errors

本文贡献：提出了反向传播梯度下降方法来更新参数，从而有效地学习神经网络的中间表示

隐藏单元可以表示特定领域任务(task domain)的特征，单元间的相互交互可以捕捉任务中的规则(regularities)

>自组织神经网络(self-organizing neural networks)：通过自动寻找样本中的内在规律和本质属性，自组织、自适应地改变网络的**参数**和**结构**

损失函数使用均方误差函数(mean square error,MSE)，基于链式法则求损失函数对每层网络中权重的梯度，然后使用梯度下降的方法更新参数

网络中的连接更多(weight-space添加更多的维度)，即网络更复杂，模型的局部最小值的表现也会更好一点，即使优化到了局部最小值，最终结果也可以接受

注意：论文中提出更新权重的公式如下：
$$
\Delta w(t)=-\varepsilon \partial E / \partial w(t)+\alpha \Delta w(t-1) \tag{1}
$$
该公式中与常用的梯度下降公式略有不同，在旧的权重上加了参数$\alpha$，该参数是一个在0和1之间的指数衰减因子(exponential decay factor)，来决定上一个时刻和当前时刻梯度对权重的**相对贡献**(determine the relative contribution of the current gradient and earlier gradients to the weight change)

满足下面条件后全连接网络和循环神经网络可以等价(equivalent)：

- 前向计算时存储每个中间层单元的输出，为了后续的反向传播
- 不同的层有相同的值
  we average gradient for all the weights in each set of corresponding weights and then change each weight in the set by an amount proportional to this average gradient?

基于反向传播的任务：

1. 检测输入单元的对称性(symmetry)
2. 在家谱树(family trees)中存储信息
   将家谱的关系表示为：`<person1> <relationship> <person2>`
   输入前两个词可以预测第3个词