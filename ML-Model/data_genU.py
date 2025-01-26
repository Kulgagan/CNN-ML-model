import numpy as np
import cv2
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as training    #takes in X, Y, test_size, random state 

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
        raise ValueError("Invalid shape type gang, jus give up alr LMAOOOOOOO")
    
    return image

def create_dataset(num_samples_per_class=500, image_size=128):
    """
    1. outer loop = goes through the shape type list (labels = 0,1,2)
    2. iterates 500 times (amount of photos for every shape) 
    3. image = shape type and image size to list
    4. append image info and label to list
    """
    shape_types = ["circle", "square", "triangle"] 
    images = []
    labels = []

    for label, shape_type in enumerate(shape_types):
        for _ in range(num_samples_per_class): 
            image = generate_shape(shape_type, image_size)
            images.append(image) 
            labels.append(label) 
    return np.array(images), np.array(labels)

def split_data(images, labels, test_size=0.2, val_size=0.1): 
    """
    1. split images and labels into two subsets
    2. (X_train, y_train) = training set, (X_temp, y_temp) = temp set that will be split into validation and test sets
    3. size of temp set = test_size + val_size, (this fraction of data is allocated to validation and testing)
    4. random_state = 42, (same random shuffle each time the function is called) 
    5. val fraction computes frac of the temporary set that should go to the validation set
    6. 
    """
    X_train, X_temp, y_train, y_temp = training(
        images, labels,
        test_size = test_size + val_size,
        random_state = 42
    )
    val_fraction = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = training(
        X_temp, y_temp,
        test_size=val_fraction, 
        random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


