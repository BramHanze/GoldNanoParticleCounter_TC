import cv2
import numpy as np
import matplotlib.pyplot as plt

class BlackDotDetector:
    def __init__(self, image_path, min_area=120, circularity_threshold=0.82, cell_min_area=1_000_000):
        self.image_path = image_path
        self.min_area = min_area
        self.circularity_threshold = circularity_threshold
        self.cell_min_area = cell_min_area

        self.original_image = None
        self.cell_masked_image = None
        self.thresholded_image = None
        self.cell_contours = []
        self.black_dots = []
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

            # Create a mask for the cell
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, self.cell_contours, -1, 255, -1)  # Fill the cell

            # Apply mask to original image
            masked = cv2.bitwise_and(image, image, mask=mask)
            self.cell_masked_image = masked

            # Show the masked cell
            plt.figure(figsize=(12, 8))
            plt.imshow(masked[..., ::-1])
            plt.title("Masked Cell Region")
            plt.axis('off')
            plt.colorbar()
            plt.show()
            return True

    def preprocess_image(self):
        blur = cv2.medianBlur(self.cell_masked_image, 9)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2
        )
        thresh = cv2.medianBlur(thresh, 15)
        self.thresholded_image = cv2.threshold(thresh, 45, 255, cv2.THRESH_BINARY_INV)[1]

    def find_black_dots(self):
        cnts, _ = cv2.findContours(self.thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.black_dots = []
        self.dot_areas = []

        for c in cnts:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if area > self.min_area and circularity > self.circularity_threshold:
                self.black_dots.append(c)
                self.dot_areas.append(area)

    def draw_black_dot_contours(self, output_path='result_adapt_thresh.jpg'):
        img_copy = self.cell_masked_image.copy()
        cv2.drawContours(img_copy, self.black_dots, -1, (36, 255, 12), 2)
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
        self.draw_black_dot_contours()
        print("Total black dots found:", len(self.black_dots))
        self.analyze_dot_areas()
        self.show_threshold_image()
