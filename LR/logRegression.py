# -*- coding:utf-8 -*-

#################################################
# logRegression: Logistic Regression
# Author : yishijie
# Date   : 2017-02-03
# HomePage : https://github.com/JayEworld
# Email  : eworldjay@163.com
# refer  : (1) http://blog.csdn.net/zouxy09/article/details/20319673
#          (2) http://blog.csdn.net/lookqlp/article/details/51161640
#################################################

import numpy
import random

# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + numpy.exp(-inX))


class LogRegres():

    def __init__(self, alpha=0.1, maxIter=100, batchSize=10, optimizeType='gradDescent'):
        self.alpha = alpha
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.optimizeType = optimizeType
        self.numFeatures = 0
        self.weights = None

    def train(self, train_x, train_y):
        # trans train_x and train_y
        train_x = [sample+[1.0] for sample in train_x]
        train_x = numpy.mat(train_x)
        train_y = numpy.mat(train_y).transpose()
        # numSamples, numFeatures, numFeatures
        numSamples, numFeatures = numpy.shape(train_x)
        self.numFeatures = numFeatures - 1
        weights = numpy.ones((numFeatures, 1))
        # iteration
        for k in range(self.maxIter):
            # batch gradient descent algorilthm
            if self.optimizeType == 'batchgradDescent': # wi = wi - alpha * xigema[(h(x)-y)xi]
                output = sigmoid(train_x * weights)
                error = train_y - output
                weights = weights + self.alpha * train_x.transpose() * error
            # mini-batch gradient descent
            elif self.optimizeType == 'miniBatchGradDescent': # wi = wi - alpha * xigema-mini[(h(x)-y)xi]
                dataIndex = 0
                while(dataIndex < numSamples):
                    output = sigmoid(train_x[dataIndex:(dataIndex + batchSize), :] * weights)
                    error = train_y[dataIndex:(dataIndex + batchSize), 0] - output
                    weights = weights + self.alpha * train_x[dataIndex:(dataIndex + batchSize), :].transpose() * error
                    dataIndex += batchSize
            # stochastic gradient descent
            elif self.optimizeType == 'stocGradDescent': # wi = wi - alpha * (h(x)-y)xi
                for i in range(numSamples):
                    output = sigmoid(train_x[i, :] * weights)
                    error = train_y[i, 0] - output
                    weights = weights + self.alpha * train_x[i, :].transpose() * error
            # smooth stochastic gradient descent
            elif self.optimizeType == 'smoothStocGradDescent': # 
                # randomly select samples to optimize for reducing cycle fluctuations
                dataIndex = range(numSamples)
                for i in range(numSamples):
                    alpha = 4.0 / (1.0 + k + i) + 0.01
                    randIndex = int(random.uniform(0, len(dataIndex)))
                    output = sigmoid(train_x[randIndex, :] * weights)
                    error = train_y[randIndex, 0] - output
                    weights = weights + alpha * train_x[randIndex, :].transpose() * error
                    del(dataIndex[randIndex]) # during one interation, delete the optimized sample
            else:
                raise NameError('Not support optimize method type!')
        print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
        self.weights = weights
        return weights

    def test(self, test_x, test_y):
        # trans test_x and test_y
        test_x = [sample+[1.0] for sample in test_x]
        test_x = numpy.mat(test_x)
        test_y = numpy.mat(test_y).transpose()
        numSamples, numFeatures = shape(test_x)
        if numFeatures != (self.numFeatures + 1):
            raise NameError('Test data have different numFeatures with train data!')
        matchCount = 0
        for i in xrange(numSamples):
            predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
            if predict == bool(test_y[i, 0]):
                matchCount += 1
        accuracy = float(matchCount) / numSamples
        return accuracy
