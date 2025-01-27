import numpy as np
from base_layer import Layer

class Activation(Layer):
    """
    Explanation...
    """
    def __init__(self, activation, activation_prime):   #takes 2 parameters(both are functions)
        self.activation = activation    
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)  #applies activation to the input
    
    def backward(self, output_gradient, learing_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))  #implementing the element-wise multiplication function
