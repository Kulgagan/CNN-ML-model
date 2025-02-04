import numpy as np
from base_layer import Layer
"""
Fully connected dense layer (every neuron in the previous layer is connected to every neuron in current layer)
1. initialization: Weights and biases are initialized randomly so each neuron can learn different things
2. forward propagation: 
- computes Y = W (dot) X + b
- weights determine the stregth of each connection
- bias allows network to shigt the output space so it can model more complex functions
- activation func should be applied after this
3. Uses back propogation (method of updationg weights and biases based on gradient descent). (overall calculates how changing weights would affect the function) 
    1. compute the gradient of the loss with respect to weights
    2. compute the gradient of the loss with respect to the bias
    3. compute the gradient of the loss with respect to the input to pass it to the previous layer
4. update parameters: Adjust the wieghts and biases using gradient descent
(goal is to minimize loss function. as it gets smaller, the model gets better at training. sort of its metric of accuracy) 
"""

class Dense(Layer):
    def __init__(self, input_size, output_size):   #initializing input and output neurons
        self.weights = np.random.randn(output_size, input_size) #initializing bias and weights as random numbers (will change in the future)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate): #calculates the derivative of the error wrt the weights
        weights_gradient = np.dot(output_gradient, self.input.T)   #derivative of error wrt to the basis
        input_gradient = np.dot(self.weights.T, output_gradient)    #the derivative of the error wrt to the input(for other) (returned)
        self.weights -= learning_rate * weights_gradient    #updating the paramenters with gradient decent
        self.bias -= learning_rate * output_gradient
        return input_gradient   #returns the derivative of the error wrt the input
    