import cv2
import sys
import os
import random
import numpy as np


def imshow_keep_aspect_ratio(winname, img, bg_color=(0, 0, 0)):
    try:
        rect = cv2.getWindowImageRect(winname)
        win_w, win_h = rect[2], rect[3]
    except cv2.error:
        win_w, win_h = 0, 0

    if win_w <= 0 or win_h <= 0:
        cv2.imshow(winname, img)
        return

    img_h, img_w = img.shape[:2]
    img_aspect = img_w / float(img_h)
    win_aspect = win_w / float(win_h)

    if win_aspect > img_aspect:
        new_h = win_h
        new_w = int(win_h * img_aspect)
    else:
        new_w = win_w
        new_h = int(win_w / img_aspect)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        canvas = np.full((win_h, win_w), bg_color[0], dtype=np.uint8)
    else:
        canvas = np.full((win_h, win_w, 3), bg_color, dtype=np.uint8)

    top = (win_h - new_h) // 2
    left = (win_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    cv2.imshow(winname, canvas)


source = "./Videos/warden and the paunch.mp4"

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

script_dir = os.path.dirname(os.path.abspath(__file__))
saved_path = os.path.join(script_dir, "single_frame.png")
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

window_name = "Original (Color) vs Grayscale - TP1 Ex1 Item B"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 360)

while True:
    imshow_keep_aspect_ratio(window_name, side_by_side)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
