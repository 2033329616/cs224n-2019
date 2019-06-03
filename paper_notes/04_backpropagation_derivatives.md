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

## 2. Derivatives, Backpropagation, and Vectorization

### 1. Derivatives

#### 1.1 Scalar Case (输入输出全为标量)

- 函数：$f : \mathbb{R} \rightarrow \mathbb{R}$

微分定义，$h$是一个常量：
$$
f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h} \tag{1}
$$

函数$f$在点$x$处的微分表示当$x$改变时$f$的改变量
$$
f(x+\varepsilon) \approx f(x)+\varepsilon f^{\prime}(x) \tag{2}
$$

$\frac{\partial y}{\partial x}$是变量$x$和$y$之间变化的**比率**，注意这里都是标量
$$
x \rightarrow x+\Delta x \Longrightarrow y \rightarrow \approx y+\frac{\partial y}{\partial x} \Delta x \tag{3}
$$

#### 1.2 Gradient: Vector in, scalar out (输入为向量，输出为标量)

> 这里的向量vector指一维度的列向量

- 函数：$f : \mathbb{R}^{N} \rightarrow \mathbb{R}$

梯度定义，注意$h$是一个向量:
$$
\nabla_{x} f(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{\|h\|} \tag{4}
$$
梯度表示改变$x$的每一个维度的量而对$y$产生了多少影响，下面公式仍旧成立，不过$\frac{\partial y}{\partial x}$和$\Delta x$都是向量，最终$y$的值为标量:
$$
x \rightarrow x+\Delta x \Longrightarrow y \rightarrow \approx y+\frac{\partial y}{\partial x} \cdot \Delta x \tag{5}
$$

$\frac{\partial y}{\partial x}$可以表示成下面的方式，$x$的每一维与$y$之间的改变的比率：
$$
\frac{\partial y}{\partial x}=\left(\frac{\partial y}{\partial x_{1}}, \frac{\partial y}{\partial x_{2}}, \ldots, \frac{\partial y}{\partial x_{N}}\right) \tag{6}
$$

#### 1.3 Jacobian: Vector in, Vector out (输入和输出都是向量)

- 函数：$f : \mathbb{R}^{N} \rightarrow \mathbb{R}^{M}$

雅可比矩阵(Jacobian)的维度为$output \times input $，矩阵$(i,j)$位置的元素表示第$j$个输入与第$i$个输出间改变量的比率，可表示为$\frac{\partial y_i}{\partial x_j}$：
$$
\frac{\partial y}{\partial x}=\left(\begin{array}{ccc}{\frac{\partial y_{1}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{1}}{\partial x_{N}}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial y_{M}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{M}}{\partial x_{N}}}\end{array}\right) \tag{7}
$$
同理(3)和(5)公式在这里仍然成立，只不过加号右边为矩阵和列向量的乘积(matrix-vector multiplication)

#### 1.4 Generalized Jacobian: Tensor in, Tensor out (输入和输出都是张量)

> 这里的张量tensor是多维度的矩阵(D-dimensional grid of numbers)

- 函数：$f : \mathbb{R}^{N_{1} \times \cdots \times N_{D_{x}}} \rightarrow \mathbb{R}^{M_{1} \times \cdots \times M_{D_{y}}}$

