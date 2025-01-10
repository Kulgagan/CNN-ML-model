import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def generate_shape(shape_type, image_size = 128):
    image = np.zeros((image_size * image_size, 3), dtype=np.uint8) 
    color = (255,255,255) 
    thickness = -1

    if shape_type == "circle ":
        center = (np.random.randint(20, image_size - 20), np.random.randint(20, image_size - 20)) 
        radius = np.random.randint(10,30)
        cv2.circle(image, center, radius, color, thickness)
    elif shape_type == "square":
        start_point = (np.random.randint(10, image_size - 40), np.random.randint(10, image_size -40))
        end_point = (start_point[0] + np.random.randint(10,40), start_point[1] + np.random.randint(10, 40))
        cv2.rectangle(image, start_point, end_point, color, thickness)
    elif shape_type == "triangle":
        points = ([
            [np.random.randint(10, image_size-10), np.random.randint(10, image_size)]
            [np.random.randint(10, image_size-10), np.random.randint(10, image_size)]
            [np.random.randint(10, image_size-10), np.random.randint(10, image_size)]
        ])
        cv2.drawContours(image, points, 0, color, thickness) 
    else: 
        raise ValueError("Invalid shape type. kys")
    

def create_dataset(sum_samples_per_class=500, image_size=128):
    shape_types = ["circle","square","triangle"]
    images = []
    labels = []
    
    for label, shape_type in enumerate(shape_types):
        for _ in range(sum_samples_per_class):
            image = generate_shape(shape_type, image_size)
            images.append(image)
            labels.append(label) 
    return np.array(images), np.array(labels)


def split_data(images, labels, test_size=0.2, val_size=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size + val_size, random_state=42)
    
    val_fraction = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_fraction, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

