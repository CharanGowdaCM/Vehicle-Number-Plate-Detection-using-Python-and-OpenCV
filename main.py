from character_segmentation import segment_characters
from plate_detection import find_contours
from license_plate_extraction import extract_plate
from results import show_results
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import keras
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

original_image = cv2.imread(r'C:\Users\LENOVO\Desktop\license plate\License-Plate-Recognition\test1.jpeg')
plate_img, plate = extract_plate(original_image)
cv2.waitKey(1000)

dimensions, img_dilate = segment_characters(plate)

char_list = find_contours(dimensions, img_dilate)

#model = keras.models.load_model('model.h5')

image = Image.fromarray(img_dilate)

extracted_text = pytesseract.image_to_string(image)
special_characters = [
    "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=",
    "{", "}", "[", "]", "|", "\\", ":", ";", "\"", "'", "<", ">", ",", ".",
    "?", "/", "`", "~"
]
for char in special_characters:
    new_text = extracted_text.replace(char, "")
    extracted_text=new_text.strip()


print("The extracted number plate is:",extracted_text)
cv2.imshow('Detected Plate on image', plate_img)
cv2.imshow('Dilated Image', img_dilate)
cv2.waitKey(0)

'''
    test1= 1.52
    test2= 1.61
    test3=1.43
    test4= 1.502
    test5=1.55
'''