import cv2
import sys
import numpy as np

def add_label(image, text):
    img_copy = image.copy()
    cv2.putText(img_copy, text, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(img_copy, text, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4, cv2.LINE_AA)
    return img_copy

source = "Imagens/BallPit.jpg"

img = cv2.imread(source)
if img is None:
    print(f"Error: Could not read image {source}")
    sys.exit(1)

if img.shape[0] < 480 or img.shape[1] < 480:
    img = cv2.resize(img, (max(img.shape[1], 480), max(img.shape[0], 480)))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

canny1 = cv2.Canny(blurred, 50, 150)
canny2 = cv2.Canny(blurred, 100, 200)

contours, _ = cv2.findContours(canny1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

filtered_contours = []
areas = []

contour_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
        filtered_contours.append(cnt)
        areas.append(area)

        if area < 1000:
            color = (0, 255, 0)
        elif area < 5000:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.drawContours(contour_img, [cnt], -1, color, 3)

total_contours = len(filtered_contours)
max_area = max(areas) if areas else 0.0

print(f"Total valid contours (area > 200 px^2): {total_contours}")
print(f"Largest contour area: {max_area:.2f} px^2")

c1_bgr = cv2.cvtColor(canny1, cv2.COLOR_GRAY2BGR)
c2_bgr = cv2.cvtColor(canny2, cv2.COLOR_GRAY2BGR)

lbl_c1 = add_label(c1_bgr, "Canny (50, 150)")
lbl_c2 = add_label(c2_bgr, "Canny (100, 200)")
lbl_cnt = add_label(contour_img, "Contours by Area")

side_by_side = cv2.hconcat([lbl_c1, lbl_c2, lbl_cnt])

while True:
    cv2.imshow("Canny Pair 1 vs Canny Pair 2 vs Categorized Contours - TP1 Ex3 Item B", cv2.resize(side_by_side, (1440, 360)))
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
