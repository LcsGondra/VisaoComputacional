import cv2
import sys
import os
import random

source = "Vídeos/warden and the paunch.mp4"

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames > 1:
    random_frame_idx = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    print(f"random frame: {random_frame_idx} of {total_frames}")

ret, frame = cap.read()
cap.release()

saved_path = "tp1/single_frame.png"
cv2.imwrite(saved_path, frame)

color_img = cv2.imread(saved_path)
if color_img is None:
    print("Error: Could not reload saved image.")
    sys.exit(1)

gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

print(f"Color Image - Shape: {color_img.shape}, Dtype: {color_img.dtype}")
print(f"Grayscale Image - Shape: {gray_img.shape}, Dtype: {gray_img.dtype}")

gray_3ch = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
side_by_side = cv2.hconcat([color_img, gray_3ch])

while True:
    cv2.imshow("Original (Color) vs Grayscale - TP1 Ex1 Item B", side_by_side)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()

