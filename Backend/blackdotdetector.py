
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class BlackDotDetector:
    def __init__(self, image_path, min_area=120, circularity_threshold=0.8, dot_blur=13, prevent_false_positives=True):
        self.image_path = image_path
        self.min_area = min_area
        self.circularity_threshold = circularity_threshold
        self.dot_blur = dot_blur
        self.prevent_false_positives = prevent_false_positives

        self.original_image = None
        self.cell_masked_image = None
        self.thresholded_image = None
        self.cnts = []
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

        cell_threshold = mean_pixel_color - (255-mean_pixel_color)/2
        thresh = cv2.threshold(gray, cell_threshold, 255, cv2.THRESH_BINARY_INV)[1]
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
        Returns a list of contours inside the detected cell contour, or close to the cell contour.

        It first searches for all contours in image, then checks which are inside the required area.
        This is done by checking if the centre of a dot/contour is inside the cell contour, if not it checks how far away.
        If the distance between the centre of a contour and the edge of the cell contour is lower than a given value (standard is 150), allow this contour.
        """
        cell_contour = self.cell_contours[0]
        inside_contours = []

        self.cnts, _ = cv2.findContours(self.thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for dot in self.cnts:
            M = cv2.moments(dot)
            if M["m00"] == 0:
                continue  # Skip invalid contours

            #Calculate centroid of dot
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            #Check distance to cell contour
            distance = cv2.pointPolygonTest(cell_contour, (cx, cy), True)
            if distance >= -150:  #inside or within 'margin' pixels outside
                inside_contours.append(dot)
        self.cnts = inside_contours

    def preprocess_image(self):
        blur = cv2.medianBlur(self.original_image, max(self.dot_blur-4,9))
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
        
        thresh = cv2.medianBlur(thresh, self.dot_blur)
        self.thresholded_image = cv2.threshold(thresh, 45, 255, cv2.THRESH_BINARY_INV)[1]

    def find_black_dots(self):
        """
        Detects and tests singular and clusters of black dots from the preprocessed image.

        After detecting all contours in the preprocessed image, the contours are tested on:
        1. Size:
            - All contours have to be bigger than `self.min_area`.
        1. Circularity:
            - Contours with high circularity are considered potential single black dots.
            - Contours with lower circularity are considered potential clusters.
        2. False Positive Filtering (optional): 
            - If `self.prevent_false_positives` is enabled, all contours are filtered using 
            `filter_darker_than_surroundings()` which remove contours that are not significantly darker 
            than their surroundings.
        3. Cluster processing:
            - Low-circularity contours are further analyzed to estimate how many dots they likely contain 
            based on how many (normal non cluster) dots fit inside.
            - Potential clusters are filtered on circularity, required circularity goes down the bigger the cluster.
        """
        potential_contours = []
        potential_clusters = []
        round_clusters = []
        for c in self.cnts:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            if area > self.min_area:
                area = cv2.contourArea(c)
                perimeter = cv2.arcLength(c, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity > self.circularity_threshold:
                    potential_contours.append(c)
                else:
                    potential_clusters.append(c)
        
        if self.prevent_false_positives:
            potential_contours = self.darker_than_surroundings(potential_contours)
        for c in potential_contours:
            self.dot_areas.append(cv2.contourArea(c))
            self.black_dots = potential_contours

        average_dot_size = sum(self.dot_areas) / len(self.dot_areas) if self.dot_areas else 50
        for uncircular_dot in potential_clusters:
            area = cv2.contourArea(uncircular_dot)
            dots_in_cluster = area / average_dot_size
            if dots_in_cluster > 1.5:
                perimeter = cv2.arcLength(uncircular_dot, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity > 0.4 - (dots_in_cluster * 0.05):
                    round_clusters.append(uncircular_dot)

        if self.prevent_false_positives:
            round_clusters = self.darker_than_surroundings(round_clusters)
        for uncircular_dot in round_clusters:
            area = cv2.contourArea(uncircular_dot)
            dots_in_cluster = area / average_dot_size
            self.extra_dots += round(dots_in_cluster)
            self.cluster_dots.append(uncircular_dot)

    def create_output(self, output_path='output/'):
        img_copy = self.original_image.copy()
        self.outputJSON['normal_dots'] = len(self.black_dots)
        self.outputJSON['cluster_dots'] = self.extra_dots
        self.outputJSON['found_dots'] = len(self.black_dots)+self.extra_dots

        #cv2.drawContours(img_copy, self.cell_contours, -1, (255, 255, 255), 2)
        cv2.drawContours(img_copy, self.black_dots, -1, (36, 255, 12), 2)
        cv2.drawContours(img_copy, self.cluster_dots, -1, (36, 12, 255), 2)

        file_name = os.path.basename(self.image_path).rsplit('.', 1)[0]
        file_path = os.path.join(output_path, file_name)

        os.makedirs(output_path, exist_ok=True)

        cv2.imwrite(f"{file_path}.jpg", img_copy)
        self.show_image(img_copy)
        with open(f"{file_path}.json", 'w', encoding='utf-8') as f:
            json.dump(self.outputJSON, f, ensure_ascii=False)
            
    def darker_than_surroundings(self, contours, image=None, difference_threshold=10, dilate_size=15):
        """
        Filters contours to retain only those where the mean intensity(/grayness) is at least `difference_threshold`
        darker than that of the surrounding area. The `difference_threshold` adapts to the surrounding area,
        lighter areas need a bigger difference between dot and surroundings.
        """
        if image is None:
            image = self.original_image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        filtered_contours = []

        mask_dot = np.zeros(gray.shape, dtype=np.uint8)
        mask_dilated = np.zeros_like(mask_dot)

        for contour in contours:
            #Clear previous mask
            mask_dot[:] = 0
            cv2.drawContours(mask_dot, [contour], -1, 255, thickness=cv2.FILLED)

            mask_dilated[:] = 0
            cv2.dilate(mask_dot, kernel, dst=mask_dilated, iterations=1)

            mean_dot = cv2.mean(gray, mask=mask_dot)[0]

            if mean_dot > 80:
                continue #Skip bright dot

            #Get surrounding area by subtracting the filled contour from the dilated one
            mask_surrounding = cv2.subtract(mask_dilated, mask_dot)
            mean_surrounding = cv2.mean(gray, mask=mask_surrounding)[0]

            dynamic_threshold = difference_threshold + (mean_surrounding / 10.0)

            if (mean_surrounding - mean_dot) >= dynamic_threshold:
                filtered_contours.append(contour)

        return filtered_contours

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
        self.dots_inside_cell()
        self.find_black_dots()
        self.create_output()
        print("Total black dots found:", self.outputJSON['found_dots'])
        self.analyze_dot_areas()
        #self.show_image(self.thresholded_image)

#object = BlackDotDetector('data/Complemented/2024-08i compl OADChi E2_02.tif')
#object.run()
