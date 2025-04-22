import cv2
import numpy as np
import matplotlib.pyplot as plt

class BlackDotDetector:
    def __init__(self, image_path, min_area=120, circularity_threshold=0.85, cell_min_area=1_000_000):
        self.image_path = image_path
        self.min_area = min_area
        self.circularity_threshold = circularity_threshold
        self.cell_min_area = cell_min_area

        self.original_image = None
        self.cell_masked_image = None
        self.thresholded_image = None
        self.cell_contours = []
        self.black_dots = []
        self.cluster_dots = []
        self.extra_dots = 0
        self.dot_areas = []

        self.load_image()

    def load_image(self):
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Image at '{self.image_path}' could not be loaded.")

    def detect_cell(self):
        image = self.original_image.copy()
        mean_pixel_color = np.median(image)
        blur = cv2.medianBlur(image, 81)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, mean_pixel_color - 15, 255, cv2.THRESH_BINARY_INV)[1]

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cell_contours = []

        for c in cnts:
            area = cv2.contourArea(c)
            if area > self.cell_min_area:
                self.cell_contours.append(c)

        if len(self.cell_contours) != 1:
            print(f"{len(self.cell_contours)} cells found — only 1 required. Skipping dot detection.")
            return False
        else:
            print("Cell surface area is:", cv2.contourArea(self.cell_contours[0]))

            return True
    
    def dots_inside_cell(self):
        """
        Check if the black dots are inside the detected cell contour.
        If they are, keep them; otherwise, discard them.

        This is done by checking if the centroid of each dot is inside the cell contour.
        """
        cell_contour = self.cell_contours[0]
        dots = self.black_dots

        inside_contours = []

        for dot in dots:
            M = cv2.moments(dot)
            if M["m00"] == 0:
                continue  # Skip invalid contours

            #Calculate centroid of dot
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            #Check if centroid is inside the cell contour
            result = cv2.pointPolygonTest(cell_contour, (cx, cy), False)
            if result >= 0:
                inside_contours.append(dot)
                
        self.black_dots = inside_contours


    def preprocess_image(self):
        blur = cv2.medianBlur(self.original_image, 9)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2
        )
        thresh = cv2.medianBlur(thresh, 15)
        self.thresholded_image = cv2.threshold(thresh, 45, 255, cv2.THRESH_BINARY_INV)[1]

    def find_black_dots(self):
        """
        Find black dots in the thresholded image using contour detection.
        Classify them based on area and circularity.
        """
        cnts, _ = cv2.findContours(self.thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.black_dots = []
        self.dot_areas = []

        for c in cnts:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if area > self.min_area:
                if circularity > self.circularity_threshold:
                    self.black_dots.append(c)
                else:
                    self.cluster_dots.append(c)
                self.black_dots.append(c)
                self.dot_areas.append(area)

        total_area = 0.0
        for dot in self.black_dots:
            total_area += cv2.contourArea(dot)
        average_dot_size = total_area/len(self.black_dots)

        temp = []
        for cluster_dot in self.cluster_dots:
            dots_in_cluster = round(cv2.contourArea(cluster_dot)/average_dot_size)
            if dots_in_cluster > 1.95:
                self.extra_dots += dots_in_cluster
                temp.append(cluster_dot)
        self.cluster_dots = temp

    def draw_contours(self, output_path='result_adapt_thresh.jpg'):
        img_copy = self.original_image.copy()
        cv2.drawContours(img_copy, self.black_dots, -1, (36, 255, 12), 2)
        cv2.drawContours(img_copy, self.cluster_dots, -1, (36, 12, 255), 2)
        cv2.imwrite(output_path, img_copy)

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

    def show_threshold_image(self):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.thresholded_image, cmap='gray')
        plt.title("Thresholded for Black Dots")
        plt.axis('off')
        plt.colorbar()
        plt.show()

    def run(self):
        print("🔍 Detecting cell...")
        if not self.detect_cell():
            return
        print("✅ Cell detected.\n")

        print("🔎 Finding black dots...")
        self.preprocess_image()
        self.find_black_dots()
        self.dots_inside_cell()
        self.draw_contours()
        print("Total black dots found:", len(self.black_dots))
        self.analyze_dot_areas()
        self.show_threshold_image()

object = BlackDotDetector('data/Complemented/2024-08i compl OADChi E2_24.tif')
object.run()
