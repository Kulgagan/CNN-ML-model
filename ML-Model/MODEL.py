import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def generate_shape(shape_type, image_size = 128):
    image = np.zeros((image_size * image_size, 3), dtype=np.uint8) 
    color = (255,255,255) 
    thickness = -1