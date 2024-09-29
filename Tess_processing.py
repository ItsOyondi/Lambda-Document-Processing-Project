# pip install pytesseract tesseract
# pip install pillow opencv-python

import pytesseract
import PIL.Image
import cv2
from cv2 import dnn_superres
import pandas as pd
import textblob
from textblob import TextBlob
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from PIL import Image

# import required module
import os


def image_upscale_to_binary(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path) 

    # Get the shape of the image
    height, width, channels = img.shape

    # Define the new size
    new_width = 2*width
    new_height = 2*height

    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    # Create a super-resolution model (EDSR)
    #sr = cv2.dnn_superres.DnnSuperResImpl_create()
    #sr.readModel('EDSR_x4.pb')
    #sr.setModel('edsr', 4)

    # Upscale the image
    #upscaled_img = sr.upsample(img)
    
    # Convert to Grayscale
    #grey = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Convert to Black and White
    #thresh, BW = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

    # Apply thresholding to convert to binary
    #_, binary_img = cv2.threshold(BW, 127, 255, cv2.THRESH_BINARY) 

    return resized_image

def noise_removal(image):
   kernel = np.ones((1,1), np.uint8)
   image = cv2.dilate(image, kernel, iterations=1)
   kernel = np.ones((1, 1), np.uint8)
   image = cv2.erode(image, kernel, iterations=1)
   image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
   image = cv2.medianBlur(image, 3)
   return (image)

# assign directory
directory = r'C:\\Users\Spawtan\Documents\DSA_Stuffs\Python_Projects\tess_sample\cropped_images\lm_text'
 
# iterate over files in
# that directory

files_list = []
text_list = []
c = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
      print(f)

      # Read the image
      img = image_upscale_to_binary(f)
      #img = noise_removal(img)
      cv2.imwrite(f"temp/text_prod_{c}.jpg", img)
      c += 1
      # Create a super-resolution model (EDSR)
      #sr = cv2.dnn_superres.DnnSuperResImpl_create()
      #sr.readModel('EDSR_x4.pb')
      #sr.setModel('edsr', 30)

      # Upscale the image
      #upscaled_img = sr.upsample(img)

      myconfig = r"--psm 6 --oem 3"

      #text = pytesseract.image_to_string(PIL.Image.open(f), config=myconfig)
      ocr_text = pytesseract.image_to_string(img, config=myconfig)

      #reader = easyocr.Reader(['en'], gpu=False)
      #text_detections = reader.readtext(img)

      #text = TextBlob(ocr_text)

      files_list.append(filename)
      #text_list.append(text.correct())
      text_list.append(ocr_text)

      print(filename)
      #print(text.correct())
      print(ocr_text)

data_frame = pd.DataFrame({"filename": files_list, "text": text_list})

data_frame.to_csv('tess_text_test_Auto.csv', index=False)