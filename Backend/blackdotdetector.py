import cv2
import numpy as np
import matplotlib.pyplot as plt

class BlackDotDetector:
    def __init__(self, image_path, min_area=120, circularity_threshold=0.8):
        self.image_path = image_path
        self.min_area = min_area
        self.circularity_threshold = circularity_threshold

        self.original_image = None
        self.cell_masked_image = None
        self.thresholded_image = None
        self.cell_contours = []
        self.black_dots = []
        self.cluster_dots = []
        self.extra_dots = 0
        self.dot_areas = []

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
        thresh = cv2.threshold(gray, mean_pixel_color - (255-mean_pixel_color)/2, 255, cv2.THRESH_BINARY_INV)[1]
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

        This is done by checking if the centroid of each dot is inside the cell contour.
        """
        cell_contour = self.cell_contours[0]

        def is_inside(dot_list, cell_contour, margin=75):
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
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2
        )
        thresh = cv2.medianBlur(thresh, 15)
        self.thresholded_image = cv2.threshold(thresh, 45, 255, cv2.THRESH_BINARY_INV)[1]

    def find_black_dots(self):
        """
        Find black dots in the thresholded image using contour detection.
        Classify them based on area and circularity.
        """
        uncircular_particles = []
        cnts, _ = cv2.findContours(self.thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area > self.min_area:
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    pass
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                if circularity > self.circularity_threshold:
                    self.black_dots.append(c)
                    self.dot_areas.append(area)
                else:
                    if circularity > 0.35:
                        uncircular_particles.append(c)
        if self.dot_areas:
            average_dot_size = sum(self.dot_areas)/len(self.dot_areas)
        else:
            average_dot_size = 45


        for uncircular_particle in uncircular_particles:
            dots_in_cluster = cv2.contourArea(uncircular_particle)/average_dot_size
            if dots_in_cluster > 1.5:
                circularity = (4 * np.pi * cv2.contourArea(c)) / (cv2.arcLength(c, True) ** 2)
                if circularity > (0.4 - dots_in_cluster/5): #allow larger clusters to be less round
                    print((0.5 - dots_in_cluster/10))
                    self.extra_dots += round(dots_in_cluster)
                    self.cluster_dots.append(uncircular_particle)
        

    def draw_contours(self, output_path='blackdotdetector_result.jpg'):
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
        print("Total black dots found:", len(self.black_dots)+self.extra_dots)
        self.analyze_dot_areas()
        self.show_image(self.thresholded_image)
