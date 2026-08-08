import cv2
import sys

source = "Vídeos/warden and the paunch.mp4"

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    sys.exit(1)

frame_count = 0
prev_tick = cv2.getTickCount()
freq = cv2.getTickFrequency()

while True:
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

    cv2.imshow("Video Stream - TP1 Ex1 Item A", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
