dotsizes = []
for c in cnts:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        continue  # Avoid division by zero
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if area > 25 and circularity > 0.8:
        dot_sizes.append(area)

dot_sizes.sort()
min_area = dot_sizes[int(round(len(dot_sizes)/4,0))]
print(min_area)
