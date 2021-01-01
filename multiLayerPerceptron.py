import numpy as np
import perceptron

class multiLayerPerceptron:
    '''  a multilayer perceptron class that uses the perceptron class above
    attributes:
        layers:  a python list with no. of elements per layer.
        bias:   the bias term . the same bias is used for all neurons.
        eta:    the learning rate.  '''

    def __init__(self, layers, bias=1.0):
        ''' return a new MLP object with the specified parameters. '''

        self.layers = np.array(layers, dtype = object)
        self.bias = bias
        self.network = []   # the list of lists of neurons
        self.values = []    # the list of lists of output values

        for i in range(len(self.layers)):
            self.network.append([])
            self.values.append([])

            # initialize every value with zero in every perceptron
            for j in range(self.layers[i]):
                self.values[i].append(0.0)

            #input layer doesn't have neurons so i>0

            if i > 0:
                for j in range(self.layers[i]):
                    self.network.append(perceptron(input=self.layers[i-1], bias=self.bias))

            # convert list to np.array of dtype = object
            self.network = np.array([np.array(x) for x in self.network], dtype=object)
            self.values = np.array([np.array(x) for x in self.values], dtype=object)


    def set_weights(self, w_init):
        '''set the weights.
            w_init is a list of lists with the weights for all but the input layer left'''

        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])  # since network[0] is not present as input has no neurons
    def printWeights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("layer", i+1, "neuron", j, self.network[i][j].weights)

    def run(self, x):
        '''feed a simple x into the multiplayer perceptron.'''
        x = np.array(x, dtype=object)
        self.values[0]= x
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]



#testing codes

mlp = multiLayerPerceptron(layers=[2,2,1]) #mlp
mlp.set_weights([[[-10,-10,15],[15,15,-10]],[[10,10,-15]]])
mlp.printWeights()
print("mlp: ")
print("0 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))




