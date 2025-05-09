import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class BlackDotDetector:
    def __init__(self, image_path, min_area=120, circularity_threshold=0.8, cell_threshold=False, adapt_threshold=2, dot_blur=15, dot_thresh=40):
        self.image_path = image_path
        self.min_area = min_area
        self.circularity_threshold = circularity_threshold
        self.adapt_threshold = adapt_threshold
        self.dot_blur = dot_blur
        self.dot_thresh = dot_thresh
        self.cell_threshold = cell_threshold

        self.original_image = None
        self.cell_masked_image = None
        self.thresholded_image = None
        self.cell_contours = []
        self.black_dots = []
        self.cluster_dots = []
        self.extra_dots = 0
        self.dot_areas = []
        
        self.outputJSON = {
            'normal_dots': 0,
            'cluster_dots': 0,
            'found_dots': 0,
            }

        self.load_image()

    def show_image(self, image):
        plt.figure(figsize=(12, 8))
        plt.imshow(image, cmap='gray')
        plt.title("Thresholded for Black Dots")
        plt.axis('off')
        plt.colorbar()
        plt.show()

    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Image at '{self.image_path}' could not be loaded.")

    def detect_cell(self):
        image = self.original_image.copy()
        mean_pixel_color = np.median(image)
        blur = cv2.medianBlur(image, 81)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        if not self.cell_threshold:
            self.cell_threshold = mean_pixel_color - (255-mean_pixel_color)/2
        thresh = cv2.threshold(gray, self.cell_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cell_contours = []

        height, width = image.shape[:2]
        min_area = (height*width)/20
        
        #self.show_image(thresh)
        def touches_images_border(c, height, width):
            x, y, w, h = cv2.boundingRect(c)
            return x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height

        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area and not touches_images_border(c, height, width):
                self.cell_contours.append(c)
    
        if len(self.cell_contours) != 1:
            print(f"{len(self.cell_contours)} cells found — exactly 1 required. Skipping dot detection.")
            cv2.drawContours(image, self.cell_contours, -1, (36, 255, 12), 2)
            self.show_image(image)
            return False
        else:
            print("Cell surface area is:", cv2.contourArea(self.cell_contours[0]))
            return True
    
    def dots_inside_cell(self):
        """
        Check if the black dots are inside the detected cell contour.
        If they are, keep them; otherwise, discard them.

        This is done by checking if the centre of each dot is inside the cell contour.
        """
        cell_contour = self.cell_contours[0]

        def is_inside(dot_list, cell_contour, margin=150):
            inside_contours = []
            for dot in dot_list:
                M = cv2.moments(dot)
                if M["m00"] == 0:
                    continue  # Skip invalid contours

                # Calculate centroid of dot
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Check distance to cell contour
                distance = cv2.pointPolygonTest(cell_contour, (cx, cy), True)
                if distance >= -margin:  # inside or within 'margin' pixels outside
                    inside_contours.append(dot)
            return inside_contours

        self.black_dots = is_inside(self.black_dots, cell_contour)
        self.cluster_dots = is_inside(self.cluster_dots, cell_contour)

    def preprocess_image(self):
        blur = cv2.medianBlur(self.original_image, 9)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, self.adapt_threshold)
        
        thresh = cv2.medianBlur(thresh, self.dot_blur)
        self.thresholded_image = cv2.threshold(thresh, self.dot_thresh, 255, cv2.THRESH_BINARY_INV)[1]

    def find_black_dots(self):
        """
        Find black dots in the thresholded image using contour detection.
        Classify them based on area and circularity.
        """
        potential_clusters = []
        cnts, _ = cv2.findContours(self.thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:

            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if area > self.min_area:
                if circularity > self.circularity_threshold:
                    self.black_dots.append(c)
                    self.dot_areas.append(area)
                else:
                    if circularity > 0.30:
                        potential_clusters.append(c)

        average_dot_size = sum(self.dot_areas)/len(self.dot_areas) if self.dot_areas else 50

        for uncircular_dot in potential_clusters:
            dots_in_cluster = cv2.contourArea(uncircular_dot)/average_dot_size
            if dots_in_cluster > 1.5:
                self.extra_dots += round(dots_in_cluster)
                self.cluster_dots.append(uncircular_dot)
        
        self.outputJSON['normal_dots'] = len(self.black_dots)
        self.outputJSON['cluster_dots'] = self.extra_dots
        self.outputJSON['found_dots'] = len(self.black_dots)+self.extra_dots
        

    def create_output(self, output_path='output/'):
         img_copy = self.original_image.copy()
         cv2.drawContours(img_copy, self.black_dots, -1, (36, 255, 12), 2)
         cv2.drawContours(img_copy, self.cluster_dots, -1, (36, 12, 255), 2)

         file_name = os.path.basename(self.image_path).rsplit('.', 1)[0]
         file_path = os.path.join(output_path, file_name)

         os.makedirs(output_path, exist_ok=True)

         cv2.imwrite(f"{file_path}.jpg", img_copy)

         with open(f"{file_path}.json", 'w', encoding='utf-8') as f:
            json.dump(self.outputJSON, f, ensure_ascii=False)

        

    def analyze_dot_areas(self):
        self.dot_areas.sort()
        if not self.dot_areas:
            print("No dots detected.")
            return

        cutoff_index = int(len(self.dot_areas) * 0.5)
        filtered_areas = self.dot_areas[cutoff_index:]

        if filtered_areas:
            smallest_remaining_dot = filtered_areas[0]
            print(f"Size of the smallest dot after discarding 50%: {smallest_remaining_dot} pixels²")
        else:
            print("No dots remain after filtering.")

        print("All dot areas:", self.dot_areas)

    def run(self):
        print("Detecting cell...")
        if not self.detect_cell():
            return
        print("Cell detected.\n")

        print("Finding black dots...")
        self.preprocess_image()
        self.find_black_dots()
        self.dots_inside_cell()
        self.create_output()
        print("Total black dots found:", self.outputJSON['found_dots'])
        self.analyze_dot_areas()
        #self.show_image(self.thresholded_image)

# object = BlackDotDetector('data/Complemented/2024-08i compl OADChi E2_02.tif')
# object.run()
