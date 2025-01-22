import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as training

def generate_shape(shape_type, image_size = 128):
    """
    1. takes in the shape type and auto sets the image size to be 128 to keep things consistent
    2. we create a black image(all zeros) with 3 color channels. 8 bit unsigned int so values range from 0-255. make image white. fill the shape with color(thickness=-1)
    3. define shape types using open cv
    """
    image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
    color = (255, 255, 255) 
    thickness = -1 

    if shape_type == "circle":
        center = (np.random.randint(image_size - 20, image_size - 20), np.random.randint(image_size - 20, image_size - 20))   #(?,?). 2 random values as (x,y) for a random center point between 20 and 20 minus image size
        radius = np.random.randint(10,30)
        cv2.circle(image, center, radius, color, thickness)

