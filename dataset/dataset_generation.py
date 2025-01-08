import cv2
import numpy as np

def generate_shape(shape, size=28):
    img = np.zeros((size, size), dtype=np.uint8)
    if shape == 'circle':
        cv2.circle(img, (size//2, size//2), size//4, 255, -1)
    elif shape == 'square':
        cv2.rectangle(img, (size//4, size//4), (3*size//4, 3*size//4), 255, -1)
    elif shape == 'triangle':
        pts = np.array([[size//2, size//4], [size//4, 3*size//4], [3*size//4, 3*size//4]], dtype=np.int32)
        cv2.fillPoly(img, [pts], 255)
    return img

# Example dataset
shapes = ['circle', 'square', 'triangle']
X = np.array([generate_shape(shape) for shape in shapes * 100])  # 100 samples each
y = np.array([0, 1, 2] * 100)  # Labels



import matplotlib.pyplot as plt

# Visualize the first circle
plt.imshow(X[2], cmap='gray')
plt.title("square")
plt.show()


# look on kaggle for better shape datasets