import numpy as np

def mse(y_true, y_pred):    #implementing MSE function
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):  #implementing MSE prime function (how last layer gets the derivative of e wrt the input)
    return 2 * (y_pred - y_true) / np.size(y_true)
