import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import numpy as np

def image_to_binary(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

    # Apply thresholding to convert to binary
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) 

    return binary_img

# Example usage
binary_image = image_to_binary("your_image.jpg")
print(binary_image) 

I = plt.imread("/content/drive/MyDrive/Data/Ocalot.jpg")
BW = np.mean(I, axis=2)
plt.imshow(BW, cmap='gray')