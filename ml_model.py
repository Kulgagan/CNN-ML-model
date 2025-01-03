class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        '''
        process the input and return the output.
        this method should be overridden by child classes.
        '''
        raise NotImplementedError("Implement forward method in this class")
    def backward(self, output_gradient, learning_rate):
        '''
        process the input and return the output.
        this method should be overridden by child classes.
        '''
        raise NotImplementedError("Implement backward method in this class")