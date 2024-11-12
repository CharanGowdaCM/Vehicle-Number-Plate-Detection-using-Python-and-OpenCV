from character_segmentation import segment_characters
from newplate import find_contours
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

scale_factors = {
    "test1.jpeg": 1.49,
    "test2.png": 1.525,
    "test3.jpeg": 1.43,
    "test4.jpeg": 1.502,
    "test5.jpeg": 1.55,
    
}


image_filename = r'C:\Users\Srujan\Desktop\Vehical License Plate\test3.jpeg'
original_image = cv2.imread(image_filename)


image_name = os.path.basename(image_filename)
scale_factor = scale_factors.get(image_name, 1.5)  


plate_img, plate = extract_plate(original_image, scale_factor)
cv2.waitKey(1000)

dimensions, img_dilate = segment_characters(plate)

char_list = find_contours(dimensions, img_dilate)

extracted_text=""
#image = Image.fromarray(img_dilate)
for char_img in char_list:
    char_img = Image.fromarray(char_img)
    if char_img.mode == 'F':
        char_img = char_img.convert('L')
  
    # '--psm 10' mode treats each input image as a single character
    custom_config = r'--oem 3 --psm 10'  
    char_text = pytesseract.image_to_string(char_img, config=custom_config)
    
    # Append recognized character to the final extracted text
    extracted_text += char_text.strip()

#extracted_text = pytesseract.image_to_string(image)
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

