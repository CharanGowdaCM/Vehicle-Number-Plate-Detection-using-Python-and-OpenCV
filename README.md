# Vehicle Number Plate Detection<br>
This project implements a vehicle number plate detection system using Python, OpenCV, and Tesseract OCR. The solution automates the identification of license plates, aiming to improve road safety by assisting in traffic rule enforcement.

**Introduction**


With the increase in vehicle numbers, monitoring traffic violations has become challenging. This system automates license plate detection, helping traffic departments easily identify and enforce rules against offenders.

**Objectives**


License Plate Extraction: Detect and process license plate regions from images.

Optical Character Recognition (OCR): Recognize and read characters from extracted license plate images.


**Methodology**


Objective 1: License Plate Extraction


Detection: Uses OpenCV and a pre-trained Haar Cascade classifier to identify potential license plate regions.


Segmentation: The identified plate region is processed with resizing, binary thresholding, erosion, and dilation to prepare for character recognition.


Objective 2: OCR for Character Recognition


Binary Image Conversion: Applies thresholding for enhanced character contrast.


Character Recognition: Tesseract OCR engine reads and converts characters on the binary image to a text string.


**Results**


Detection Accuracy: High precision in identifying license plates under varying conditions.


OCR Performance: The Tesseract engine successfully converts license plate characters to text with high accuracy.
