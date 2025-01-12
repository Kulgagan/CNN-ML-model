from activation_layer import Activation
from base_layer import Layer
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x) #the hyperbolic tan function
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2  #the derivative of the hyperbolic tan function
        super().__init__(tanh, tanh_prime)

class ReLU(Activation):
    '''
    implement
    '''
def relu(x):
    return np.where(0, x)
def relu_prime(x):
    return np.where(x > 0, 1, 0) 

relu_activation = Activation(activation=relu, activation_prime=relu_prime) 
