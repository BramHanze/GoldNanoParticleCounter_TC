import cv2

file = 'data\\2024-08i compl OADChi E2_31.tif'
import numpy as np
import matplotlib.pyplot as plt

#Open image and apply filters
image = cv2.imread(file)

def cell_finder(image):
    blur = cv2.medianBlur(image, 81)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] #if len(cnts) == 2 else cnts[1]

    min_area = 10
    cells = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            #cv2.drawContours(image, [c], -1, (36, 255, 12), 2)  # Draw contours in green
            cells.append(c)

    if len(cells) != 1:
        print(f'{len(cells)} cells found, only 1 cell is required')
    else:
        print("Cell surface area is:", cv2.contourArea(cells[0]))

    return cells

def cell_extracter(image, cells):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  #Create 2d grid with just zero

    # Draw filled contours in the mask
    cv2.drawContours(mask, cells, -1, 255, thickness=cv2.FILLED)  #Fill the inside with 255 (white)

    kernel = np.ones((10, 10), np.uint8)

    mask = cv2.dilate(mask, kernel, iterations=16)  #Expand the filled contour

    # Create a white output image
    out = np.full_like(image, 255)

    # Extract the object with the expanded contour
    out[mask == 255] = image[mask == 255]
    return out

def dot_counter(image):
    height = image.shape[0]
    cutoff = int(height * 0.9)
    image = image[:cutoff, :]
    
    blur = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 47, 255, cv2.THRESH_BINARY_INV)[1]
    
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    min_area = 75
    circularity_threshold = 0.7
    black_dots = []
    
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue  # Avoid division by zero
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    
        if area > min_area and circularity > circularity_threshold:
            cv2.drawContours(image, [c], -1, (36, 255, 12), 2)  # Draw contours in green
            black_dots.append(c)
    return len(black_dots)
 

cells = cell_finder(image)
image = cell_extracter(image, cells)
dots = dot_counter(image)


print("Black Dots count is:", dots)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
