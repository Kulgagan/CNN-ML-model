from convolutional_layer import Convolutional
from dense_layer import Dense
from activation_layer import Activation
from activation_functions import ReLU as relu_activation
import numpy as np
class CNN:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the model layers
        - Convolutional layer followed by relu activation
        - flattening layer to reshape output for dense layers 
        - dense layer for classification
        """
        self.conv1 = Convolutional(input_shape, kernal_size = 3, depth = 8) #convolutional layer
        self.first_dense_layer_input_flattened = 8 * (input_shape[1] - 2) * (input_shape[2] -2) #flattened size for first dense layer 
        self.first_dense_layer = Dense(self.first_dense_layer_input_flattened, 64)  #first dense layer
        self.second_dense_layer = Dense(64, num_classes) #second dense layer / output layer
        self.relu = relu_activation #this could be wrong check logic
        self.softmax = lambda x:np.exp(x) / np.sum(np.exp(x)) #softmax for final outputs
