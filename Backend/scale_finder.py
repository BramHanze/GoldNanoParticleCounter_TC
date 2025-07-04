import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

def scale_finder(image_path, width_range=0.8):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    image_crop = image[int(height * 0.9725):int(height * 0.9825), int(width * width_range):]

    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    #Find the black bar
    bar_rect = None
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > 50 and (w / h) > 3:
            bar_rect = (x, y, w, h)
            break

    if bar_rect: #If bar found, look below it for text
        x, y, w, h = bar_rect

        text_roi = image[int(height * 0.98):,int(width * width_range):]

        reader = easyocr.Reader(['en'])
        results = reader.readtext(text_roi)

        if results:
            _, text, confidence = results[0] #get the first (and only) detection
        else:
            return None
        return int(text.split(' ')[0])/w #nm per pixel

    else:
        print("No bar detected.")
        return None
