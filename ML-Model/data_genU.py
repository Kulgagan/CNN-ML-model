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
    4. raise error if shape conditions not met 
    """
    image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
    color = (255, 255, 255) 
    thickness = -1 

    if shape_type == "circle":
        center = (np.random.randint(20, image_size - 20), np.random.randint(20, image_size - 20))   #(?,?). 2 random values as (x,y) for a random center point between 20 and 20 minus image size so its not too close to the edges
        radius = np.random.randint(10,30)   #random between 10-30
        cv2.circle(image, center, radius, color, thickness)
    elif shape_type == "square": 
        start_point = (np.random.randint(10, image_size - 40), np.random.randint(10, image_size - 40))  #-40 as we dont want it going off the edge
        end_point = (start_point[0] + np.random.randint(10, 40), start_point[1] + np.random.randint(10, 40))    #adds random(10-40) to start_points([0], [1])
        cv2.rectangle(image, start_point, end_point, color, thickness)
    elif shape_type == "triangle": 
        Tpoint = image_size - 10
        points = np.array([
            [np.random.randint(10, Tpoint), np.random.randint(10, Tpoint)]  #generates 3(x,y) points for cv2 traingle func to plot a triangle 
            [np.random.randint(10, Tpoint), np.random.randint(10, Tpoint)]
            [np.random.randint(10, Tpoint), np.random.randint(10, Tpoint)]
        ])
        cv2.drawContours(image, points, 0, color, thickness)    #points should be an array so it shouldnt throw an error
    else:
        raise ValueError("Invalid shape type gang")
    
    return image
 