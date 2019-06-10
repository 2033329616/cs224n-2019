#!/usr/bin/env python

import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x, gradientText):
    """ Gradient check for a function f.
    
    Arguments:
    f -- a function that takes a single argument and outputs the
         loss and its gradients
    x -- the point (numpy array) to check the gradient at
    gradientText -- a string detailing some context about the gradient computation ？？？
    """

    rndstate = random.getstate()
    random.setstate(rndstate)         # 恢复生成器的中间状态，方便复现前面采样的结果
    fx, grad = f(x)                   # Evaluate function value at original point
    h = 1e-4                          # Do not change this!

    # Iterate over all indexes ix in x to check the gradient.
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])   # readwrite表示可以读写迭代器的结果
    while not it.finished:
        ix = it.multi_index

        x[ix] += h                       # increment by h
        random.setstate(rndstate)
        fxh, _ = f(x)                    # evalute f(x + h)
        x[ix] -= 2 * h                   # restore to previous value (very important!)/ x+h -2*h => x-h
        random.setstate(rndstate)        # evalute f(x - h)
        fxnh, _ = f(x)
        x[ix] += h
        numgrad = (fxh - fxnh) / 2 / h   # 双向微分，实际中效果更好

        # Compare gradients 注意numgrad是数值梯度，而grad是解析梯度
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed for %s." % gradientText)
            print("First gradient error found at index %s in the vector of gradients" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (
                grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!")
