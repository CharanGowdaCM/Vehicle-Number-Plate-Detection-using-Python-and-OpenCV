import numpy as np
import cv2

def find_contours(dimensions, img):
    # Apply adaptive thresholding for better contrast
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

    # Find all contours
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and shape constraints for characters
    lower_width, upper_width, lower_height, upper_height = dimensions
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    x_positions = []
    characters = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if lower_width < w < upper_width and lower_height < h < upper_height:
            x_positions.append(x)

            # Extract character and resize with padding
            char_img = img[y:y+h, x:x+w]
            char_img = cv2.resize(char_img, (20, 40))
            padded_char = np.zeros((44, 24), dtype=np.uint8)
            padded_char[2:42, 2:22] = char_img

            characters.append(padded_char)

    # Sort characters by x position to maintain correct order
    sorted_chars = [characters[i] for i in np.argsort(x_positions)]
    
    return np.array(sorted_chars)
