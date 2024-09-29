# pip install pytesseract tesseract
# pip install pillow opencv-python

import pytesseract
import cv2

# import required module
import os


def image_upscale(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path) 

    # Get the shape of the image
    height, width, channels = img.shape

    # Define the new size
    new_width = 2*width
    new_height = 2*height

    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return resized_image
    

# assign directory
directory = r'C:\\Users\Spawtan\Documents\DSA_Stuffs\Python_Projects\tess_sample\cropped_images\lm_text'
 
# Give file name
filename = ""


f = os.path.join(directory, filename)
# checking if it is a file
if os.path.isfile(f):
  print(f)

  # Read the image
  img = image_upscale(f)

  myconfig = r"--psm 6 --oem 3"

  ocr_text = pytesseract.image_to_string(img, config=myconfig)

 

  print(filename)
  print(ocr_text)