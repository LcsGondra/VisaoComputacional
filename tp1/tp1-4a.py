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

source = "Videos/warden and the paunch.mp4"

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    sys.exit(1)

video_fps = cap.get(cv2.CAP_PROP_FPS)
target_fps = video_fps if video_fps > 0 else 30.0
target_frame_time = 1.0 / target_fps
freq = cv2.getTickFrequency()

window_name = "Robot Perception Object Tracker - TP1 Ex4 Item A"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

lower_red1 = np.array([0, 160, 100])
upper_red1 = np.array([6, 255, 255])
lower_red2 = np.array([172, 160, 100])
upper_red2 = np.array([180, 255, 255])

while True:
    frame_start = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Video stream ended or frame failed to load.")
        break

    height, width = frame.shape[:2]
    center_x = width // 2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_hsv1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_hsv2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    b, g, r = cv2.split(frame.astype(np.float32))
    redness = r - np.maximum(g, b)
    redness = np.clip(redness, 0, 255).astype(np.uint8)

    _, mask_redness = cv2.threshold(redness, 50, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(mask_red_hsv, mask_redness)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame.copy()
    cv2.line(annotated, (center_x, 0), (center_x, height), (255, 255, 255), 2)

    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)

        if area > 300:
            x, y, w, h = cv2.boundingRect(largest_cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            M = cv2.moments(largest_cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(annotated, (cx, cy), 7, (0, 0, 255), -1)

                dev_x = (cx - center_x) / float(center_x)

                hud_y = height - 40
                arrow_base = (center_x, hud_y)

                if dev_x < -0.05:
                    direction_str = f"Objeto a esquerda, desvio X = {dev_x:.2f}"
                    action_str = f"CORRECAO: VIRAR A ESQUERDA ({dev_x:.2f})"
                    arrow_tip = (center_x + int(dev_x * (center_x * 0.8)), hud_y)
                    arrow_color = (0, 165, 255)
                elif dev_x > 0.05:
                    direction_str = f"Objeto a direita, desvio X = {dev_x:.2f}"
                    action_str = f"CORRECAO: VIRAR A DIREITA (+{dev_x:.2f})"
                    arrow_tip = (center_x + int(dev_x * (center_x * 0.8)), hud_y)
                    arrow_color = (0, 255, 255)
                else:
                    direction_str = f"Objeto centralizado, desvio X = {dev_x:.2f}"
                    action_str = "CORRECAO: EM FRENTE (ALINHADO)"
                    arrow_tip = (center_x, hud_y - 40)
                    arrow_color = (0, 255, 0)

                print(f"{direction_str} | {action_str}")

                if arrow_base != arrow_tip:
                    cv2.arrowedLine(annotated, arrow_base, arrow_tip, arrow_color, 4, tipLength=0.3)

                cv2.putText(annotated, direction_str, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated, direction_str, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(annotated, action_str, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(annotated, action_str, (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    imshow_keep_aspect_ratio(window_name, annotated)

    processing_time = (cv2.getTickCount() - frame_start) / freq
    delay_ms = max(1, int((target_frame_time - processing_time) * 1000))

    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
