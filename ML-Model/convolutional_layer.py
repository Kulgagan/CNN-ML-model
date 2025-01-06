import numpy as np
from base_layer import Layer
from scipy import signal

class Convolutional(Layer):
    def __init__(self, input_shape, kernal_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth

        self.output_shape = (depth, input_height - kernal_size + 1, input_width - kernal_size + 1)
        self.kernal_shape = (depth, input_depth, kernal_size, kernal_size)

        self.kernals = np.random.randn(*self.kernal_shape)
        self.bias = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernals[i, j], "valid")
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        '''
        implement this. update parameters, return input gradient 
        '''


#comment on this