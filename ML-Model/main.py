import numpy as np
import base_layer as Layer

class Dense:
    def __init__ (self, input_size, output_size):   #initializing input and output neurons
        self.weights = np.random.randn(output_size, input_size) #initializing bias and weights as random numbers (will change in the future)
        self.bias = np.random.randn(output_size, 1)
        
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        '''
        update the paramenters and return the input gradient
        '''
        pass