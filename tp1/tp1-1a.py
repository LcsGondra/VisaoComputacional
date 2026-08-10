import cv2
import sys
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

video_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = video_fps if video_fps > 0 else 30.0
target_frame_time = 1.0 / target_fps

window_name = "Video Stream - TP1 Ex1 Item A"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

frame_count = 0
prev_tick = cv2.getTickCount()
freq = cv2.getTickFrequency()

while True:
    frame_start = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Video stream ended or frame failed to load.")
        break

    frame_count += 1
    current_tick = cv2.getTickCount()
    elapsed = (current_tick - prev_tick) / freq
    prev_tick = current_tick
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    height, width = frame.shape[:2]
    print(f"Frame: {frame_count} | Resolution: {width}x{height} | FPS: {fps:.2f}")

    imshow_keep_aspect_ratio(window_name, frame)

    processing_time = (cv2.getTickCount() - frame_start) / freq
    delay_ms = max(1, int((target_frame_time - processing_time) * 1000))

    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
