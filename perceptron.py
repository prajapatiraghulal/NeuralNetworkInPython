import numpy as np

class perceptron:
    '''a single neuron with the sigmoid activation fn
        attributes:
            inputs: the number of inputs in the perceptron not config the bias.
            bias: the bias term. By default it is 1.0. '''
    def __init__(self,inputs,bias=1.0):
        '''return a new perceptron object with the specified number of inputs (+1 for bias).'''
        self.weights= (np.random.rand(inputs+1)*2)-1
        self.bias = bias


    def run(self, x):
        '''run the perceptron. x is a python list with input values.'''
        sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(sum)