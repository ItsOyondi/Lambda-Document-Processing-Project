# pip install pytesseract tesseract
# pip install pillow opencv-python

import pytesseract
import PIL.Image
import cv2
from cv2 import dnn_superres
import pandas as pd

# import required module
import os
# assign directory
directory = r'C:\\Users\Spawtan\Documents\DSA_Stuffs\Python_Projects\tess_sample\cropped_images\lm_text'
 
# iterate over files in
# that directory

files_list = []
text_list = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
      print(f)

      # Read the image
      img = cv2.imread(f)

      # Create a super-resolution model (EDSR)
      sr = cv2.dnn_superres.DnnSuperResImpl_create()
      sr.readModel('EDSR_x4.pb')
      sr.setModel('edsr', 4)

      # Upscale the image
      upscaled_img = sr.upsample(img)

      myconfig = r"--psm 6 --oem 3"

      #text = pytesseract.image_to_string(PIL.Image.open(f), config=myconfig)
      text = pytesseract.image_to_string(upscaled_img, config=myconfig)

      files_list.append(filename)
      text_list.append(text)

      print(filename)
      print(text)

data_frame = pd.DataFrame({"filename": files_list, "text": text_list})

data_frame.to_csv('tess_process_test.csv', index=False)