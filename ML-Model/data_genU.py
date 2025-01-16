import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as training

def generate_shape(shape_type, image_size = 128):
    """
    1. takes in the shape type and auto sets the image size to be 128 to keep things consistent
    2. ...
    """
    image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
    color = (255, 255, 255) 
    thickness = -1 


