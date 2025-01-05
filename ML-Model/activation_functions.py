from activation_layer import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x) #the hyperbolic tan function
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2  #the derivative of the hyperbolic tan function
        super().__init__(tanh, tanh_prime)
