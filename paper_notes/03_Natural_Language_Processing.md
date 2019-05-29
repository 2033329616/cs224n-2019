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

# Natural Language Processing

[TOC]

---

## 1. Natural Language Processing (Almost) from Scratch

本文干什么：定义了**统一的神经网络架构**和学习算法用到part-of-speech tagging, chunking, named entity recognition, and semantic role labeling等NLP任务中。

使用单一的学习系统来发现足够的中间表示
将中间表示迁移到其他任务中可以实现好的效果，而中间表示是在大量的未标注的数据集上训练得到的
almost from scratch => 减少对NLP先验知识的依赖

### 1. Benchmark Tasks (基准任务)

1. Part-Of-Speech Tagging(POS)：词性标注
评价指标：per-word accuracy
2. Chunking：单词分块
评价指标：F1 score
3. Named Entity Recognition(NER)：命名实体识别
评价指标：F1 score
4. Semantic Role Labeling(SRL)：语义角色标注
评价指标：F1 score

越复杂的NLP任务需要**越复杂**的语义理解sophisticated semantic understanding

之前的任务中用到的特征都是**手动**选择的，首先根据linguistic知觉选一些，然后不断地试错和修改得到，该过程是基于经验的(empirical process)，现在是通过一个神经网络来**自动**地为每个任务学习相应的特征

神经网络可以当作复合函数组合：
$$
f_{\theta}(\cdot)=f_{\theta}^{L}\left(f_{\theta}^{L-1}\left(\ldots f_{\theta}^{1}(\cdot) \ldots\right)\right) \tag{1}
$$

使用两种方法每个时刻给一个单词标注(tag one word at the time)：

- window approach
- convolutional sentence approach

实验时除了SRL任务，基于窗口的方法都可以实现好的性能，因为SRL任务中tag依赖于动词(verb)，为了保证该词不落在window外，直接使用完整的句子

1989年Time Delay Neural Networks(TDNNs)已经提出了，这个是时延的卷积用在NLP中的方法

一个卷积层可以视为广义的window approach

在卷积层后为了输入affine层，需要对句子进行average或max操作，这里选择max操作，因为它可以将句子中最有用的局部信息捕捉到

在序列标注任务中，某个单词对应的tag和它的上下文的tag有关联(或依赖)，所以直接使用交叉熵作为损失函数不合适

训练大规模语料，可以不断增加字典的规模，用小字典训练的词向量初始化大字典的词向量，来加快训练速度

训练的标准criteria是pairwise ranking approach，该损失函数可以用于发现句法和语义信息

解析树(parse tree)中包含大量语法先验知识

多任务学习Multi-task learning(MTL)利用了迁移学习的观点，将多个任务联合训练来增加泛化能力

基准任务中引入其他方法提升性能
>1.在POS词性标注任务中，单词后缀在西方的语言中对句法预测有帮助
2.NER任务中引入大量的实体名称词典有帮助
3.在CHUNK和NER中还会引入POS的词性信息，同样在SRL中引入CHUNK信息，这是一种级联(cascading)或迁移的思想
4.模型集成(ensembles)集成多个分类器，voting比average效果好
5.句法解析树(syntastic parsing)在SRL语义角色标注中起作用