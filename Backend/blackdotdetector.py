
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import yaml

class BlackDotDetector:
    def __init__(self, image_path, min_area=None, circularity_threshold=None, dot_blur=None, prevent_false_positives=None):
        self.config = yaml.safe_load(open("config.yml"))
        self.image_path = image_path
        self.min_area = min_area if min_area is not None else self.config['min_area']
        self.circularity_threshold = circularity_threshold if circularity_threshold is not None else self.config['circularity_threshold']
        self.dot_blur = dot_blur if dot_blur is not None else self.config['dot_blur']
        self.prevent_false_positives = prevent_false_positives if prevent_false_positives is not None else self.config['prevent_false_positives']
        
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
        blur = cv2.medianBlur(image, self.config['cell_blur'])
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        cell_threshold = mean_pixel_color - (255-mean_pixel_color)/2
        thresh = cv2.threshold(gray, cell_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cell_contours = []

        height, width = image.shape[:2]
        cell_min_area = (height*width)*self.config['min_area_of_entire_image_cell']
        
        #self.show_image(thresh)
        def touches_images_border(c, height, width):
            x, y, w, h = cv2.boundingRect(c)
            return x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height

        for c in cnts:
            area = cv2.contourArea(c)
            if area > cell_min_area:
                if self.config['prevent_cell_touching_border']:
                    if not touches_images_border(c, height, width):
                        self.cell_contours.append(c)
                else:
                    self.cell_contours.append(c)
    
        if len(self.cell_contours) != 1: #REMOVE. ADD BETTER SOLUTION IF MORE THAN 1 CELL FOUND.
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
            if distance >= -self.config['cell_perimeter']:  #inside or within 'margin' pixels outside
                inside_contours.append(dot)
        self.cnts = inside_contours

    def preprocess_image(self):
        """
        Preprocesses the original image to prepare for dot detection (find_black_dots()).

        This function does the following with the image:
        - Applies a median blur to reduce noise, using a blur size based on `dot_blur` and a config multiplier.
        - Converts the image to grayscale.
        - Applies adaptive thresholding to find spots that are darker than its surroundings.
        - Applies another median blur to smooth small blotches of noise.
        - Performs binary inverse thresholding using a threshold value from the config.

        The final preporcessed image is stored in `self.thresholded_image`.
        """
        blur = cv2.medianBlur(self.original_image, max(self.dot_blur*self.config['1st_blur_multiplier'],9))
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
        
        thresh = cv2.medianBlur(thresh, self.dot_blur)
        self.thresholded_image = cv2.threshold(thresh, self.config['dot_threshold'], 255, cv2.THRESH_BINARY_INV)[1]

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
                elif circularity > self.config['cluster_circularity_threshold']:
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
            if dots_in_cluster > self.config['dots_needed_for_cluster']:
                perimeter = cv2.arcLength(uncircular_dot, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if self.config['dynamic_circularity']:
                    circ_req_decrease_per_dot = self.config['required_circularity_lowering_per_dot']
                    if circularity > self.config['cluster_circularity_threshold'] + (2*circ_req_decrease_per_dot) - (dots_in_cluster*circ_req_decrease_per_dot):
                        round_clusters.append(uncircular_dot)
                else:
                    round_clusters.append(uncircular_dot)

        if self.prevent_false_positives:
            round_clusters = self.darker_than_surroundings(round_clusters)
        for uncircular_dot in round_clusters:
            area = cv2.contourArea(uncircular_dot)
            dots_in_cluster = area / average_dot_size
            self.extra_dots += round(dots_in_cluster)
            self.cluster_dots.append(uncircular_dot)

    def create_output(self):
        """
        Generates and saves the output image and a corresponding JSON file.

        This function performs the following actions:
        - Draws contours for detected dots and optionally cell contour on a copy of the original image.
        - Saves the image in the configured output folder in the specified format.
        - Creates and saves a JSON file with counts of normal dots, cluster dots, and total dots found.

        Output files are saved using the original image filename (without extension) as the base.
        """
        output_path = self.config['output_directory']
        img_copy = self.original_image.copy()
        self.outputJSON['normal_dots'] = len(self.black_dots)
        self.outputJSON['cluster_dots'] = self.extra_dots
        self.outputJSON['found_dots'] = len(self.black_dots)+self.extra_dots
        self.outputJSON['tags'] = []
        if self.config['show_cell_outline']:
            cv2.drawContours(img_copy, self.cell_contours, -1, self.config['cell_outline_colour'], 2)
        cv2.drawContours(img_copy, self.black_dots, -1, self.config['single_dot_colour'], 2)
        cv2.drawContours(img_copy, self.cluster_dots, -1, self.config['cluster_dot_colour'], 2)

        file_name = os.path.basename(self.image_path).rsplit('.', 1)[0]
        file_path = os.path.join(output_path, file_name)

        os.makedirs(output_path, exist_ok=True)

        cv2.imwrite(f"{file_path}.{self.config['output_image_type']}", img_copy)
      #   self.show_image(img_copy) #REMOVE
        with open(f"{file_path}.json", 'w', encoding='utf-8') as f:
            json.dump(self.outputJSON, f, ensure_ascii=False)
            
    def darker_than_surroundings(self, contours, image=None):
        """
        Filters contours to retain only those where the mean intensity(/grayness) is at least `difference_threshold`
        darker than that of the surrounding area. The `difference_threshold` adapts to the surrounding area,
        lighter areas need a bigger difference between dot and surroundings.
        """
        if image is None:
            image = self.original_image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dilate_size = self.config['dilate_size']
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

            if mean_dot > self.config['absolute_threshold']:
                continue #Skip bright dot

            #Get surrounding area by subtracting the filled contour from the dilated one
            mask_surrounding = cv2.subtract(mask_dilated, mask_dot)
            mean_surrounding = cv2.mean(gray, mask=mask_surrounding)[0]

            difference_threshold = self.config['difference_threshold']
            if self.config['dynamic_threshold']:
                difference_threshold = difference_threshold + (mean_surrounding / 10.0)

            if (mean_surrounding - mean_dot) >= difference_threshold:
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