import cv2
import sys
import numpy as np

source = "Vídeos/warden and the paunch.mp4"

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Error: Could not open video source {source}")
    sys.exit(1)

lower_red1 = np.array([0, 160, 100])
upper_red1 = np.array([6, 255, 255])
lower_red2 = np.array([172, 160, 100])
upper_red2 = np.array([180, 255, 255])

while True:
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

    cv2.imshow("Robot Perception Object Tracker - TP1 Ex4 Item A", annotated)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
