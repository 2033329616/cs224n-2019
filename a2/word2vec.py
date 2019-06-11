# encoding:utf-8 
"""
1. 注意词向量的维度，中心词和上下文都是(d,V)，d是词嵌入维度，V是词表的大小 (与written solution中的维度向对应)
2. 梯度和损失函数累加的地方，注意是'+='
3. 梯度检查时使用损失函数来求其对各参数的数值梯度，然后与解析梯度进行比较
"""

import numpy as np
import random

from utils.gradcheck import gradcheck_naive
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    (v_c in the pdf handout)                     维度：(d,1) 是列向量
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors (rows of matrix) for all words in vocab
                      (U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    (dJ / dU)
    """

    ### YOUR CODE HERE

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 
    # 注意这里\hat{y}是一个概率分布，所以V个位置都要求概率，但损失函数只使用正确标签对应位置的概率
    score_out = np.dot(centerWordVec.T, outsideVectors)                # 行向量的分数，维度为 (1,V) 
    prob_out = softmax(score_out)                                      # (1,V) \hat{y}
    loss = -np.log(prob_out[:,outsideWordIdx:outsideWordIdx+1])        # (1,V)

    diff_prob = prob_out.T                                             # (V,1)
    diff_prob[outsideWordIdx] -= 1                                     # \hat{y} -y
    gradCenterVec = outsideVectors.dot(diff_prob)                      # (d, 1) d是词嵌入的大小
    gradOutsideVecs = centerWordVec.dot(diff_prob.T)                   # (d,1)x(1,V)=> (d,v)
    ### END YOUR CODE
    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:                  # 采样到负样本结束
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx                 # 将负样本的索引保存起来
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE
    ### Please use your implementation of sigmoid in here.
    loss = 0
    gradCenterVec = np.zeros_like(centerWordVec)         # 创建与中心词同维度的零向量
    gradOutsideVecs = np.zeros_like(outsideVectors)      # (d, V)

    u_o = outsideVectors[:, outsideWordIdx:outsideWordIdx+1]                # 正样本outside word (d,1)
    positive_out = np.dot(u_o.T, centerWordVec)

    for k in indices[1:]:
        u_k = outsideVectors[:,k:k+1]                                              # (d,1)
        negative_out = np.dot(u_k.T, centerWordVec)

        loss += -np.log(sigmoid(-negative_out))                                    # 累加负样本的损失
        gradCenterVec += -(sigmoid(-negative_out) - 1) * u_k                       # 累加负样本对中心词的梯度
        # 这里的k:k+1为了保持维度，从而与后面的中心词维度的列向量对应
        gradOutsideVecs[:, k:k+1] += (1 - sigmoid(-negative_out)) * centerWordVec

    loss += -np.log(sigmoid(positive_out))                                         # 累加正样本的损失
    gradCenterVec += (sigmoid(positive_out)-1) * u_o                               # 累加正样本对中心词的梯度
    gradOutsideVecs[:, outsideWordIdx:outsideWordIdx+1] += (sigmoid(positive_out)-1) * centerWordVec  # 对正样本outside word的梯度
    
    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) for all words in vocab
                        (V in pdf handout)                                        # (V,d)
    outsideVectors -- outside word vectors (as rows) for all words in vocab
                    (U in pdf handout)                                            # (V,d)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vectors
            (dJ / dV in the pdf handout)
    gradOutsideVectors -- the gradient with respect to the outside word vectors
                        (dJ / dU in the pdf handout)
    """

    loss = 0.0
    centerWordVectors, outsideVectors = centerWordVectors.T, outsideVectors.T           # 取转置，为了与written的U和V对应 (d,V)
    gradCenterVecs = np.zeros(centerWordVectors.shape)                                  # (d, V)
    gradOutsideVectors = np.zeros(outsideVectors.shape)                                 # (d, V)
    # gradOutsideVectors = np.zeros_like(outsideVectors)                                # 也可以创建相同维度的零矩阵

    ### YOUR CODE HERE
    center_idx = word2Ind[currentCenterWord]
    for outsideword in outsideWords:                                                    # 按照outsidewords遍历
        outsideWordIdx = word2Ind[outsideword] 
        centerWordVec = centerWordVectors[:, center_idx:center_idx+1]                   # (d,1) 这里传入的是列向量
        
        # dataset        
        temp_loss, temp_centergrad, temp_outsidegrad = \
            word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
        loss += temp_loss
        # print(loss)
        gradCenterVecs[:, center_idx:center_idx+1] += temp_centergrad
        gradOutsideVectors += temp_outsidegrad
    ### END YOUR CODE

    return loss, gradCenterVecs, gradOutsideVectors

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset, 
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """
    word2vecModel: 词向量模型
    word2Ind: 单词：索引的字典
    wordVectors:   词向量(中心词+上下文词)
    dataset:       数据集
    windowSize:    窗口大小
    word2vecLossAndGradient: 损失函数
    """
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)                                   # (2*V,d)
    N = wordVectors.shape[0]                                             # 词向量
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        # 窗口大小随机的原因：提高模型的鲁棒性？？？
        windowSize1 = random.randint(1, windowSize)                      # 产生[1,windowSize]内随机数
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        gin, gout = map(lambda x:x.T, [gin, gout])   
        loss += c / batchsize                            # 当前批次的损失函数
        grad[:int(N/2), :] += gin / batchsize            # 对中心词的梯度
        grad[int(N/2):, :] += gout / batchsize           # 对outside单词的梯度

    return loss, grad


def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    # 元类的解释：https://stackoverflow.com/questions/39200612/python-type-function-for-dummy-type
    dataset = type('dummy', (), {})()                                    # 元类：创造类, dummy是名称，()无继承，{}无属性和方法
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]            # 随机采样生成2*C个窗口位置
    dataset.sampleTokenIdx = dummySampleTokenIdx                         # 方法赋值
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))                 # (10,3)高斯分布向量
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")

    print("\n=== Results ===")
    print ("Skip-Gram with naiveSoftmaxLossAndGradient")

    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\nGradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
            *skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
                dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset) 
        )
    )

    print ("Expected Result: Value should approximate these:")
    print("""Loss: 11.16610900153398
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-1.26947339 -1.36873189  2.45158957]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
Gradient wrt Outside Vectors (dJ/dU):
 [[-0.41045956  0.18834851  1.43272264]
 [ 0.38202831 -0.17530219 -1.33348241]
 [ 0.07009355 -0.03216399 -0.24466386]
 [ 0.09472154 -0.04346509 -0.33062865]
 [-0.13638384  0.06258276  0.47605228]]
    """)

    print ("Skip-Gram with negSamplingLossAndGradient")   
    print ("Your Result:")
    print("Loss: {}\nGradient wrt Center Vectors (dJ/dV):\n {}\n Gradient wrt Outside Vectors (dJ/dU):\n {}\n".format(
        *skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:],
            dummy_vectors[5:,:], dataset, negSamplingLossAndGradient)
        )
    )
    print ("Expected Result: Value should approximate these:")
    print("""Loss: 16.15119285363322
Gradient wrt Center Vectors (dJ/dV):
 [[ 0.          0.          0.        ]
 [ 0.          0.          0.        ]
 [-4.54650789 -1.85942252  0.76397441]
 [ 0.          0.          0.        ]
 [ 0.          0.          0.        ]]
 Gradient wrt Outside Vectors (dJ/dU):
 [[-0.69148188  0.31730185  2.41364029]
 [-0.22716495  0.10423969  0.79292674]
 [-0.45528438  0.20891737  1.58918512]
 [-0.31602611  0.14501561  1.10309954]
 [-0.80620296  0.36994417  2.81407799]]
    """)

if __name__ == "__main__":
    test_word2vec()
