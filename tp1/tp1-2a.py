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

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

h, s, v = cv2.split(hsv)
l, a, b = cv2.split(lab)

print("Channel Representations:")
print("HSV - H (Hue): Matiz da cor (0-179 em OpenCV)")
print("HSV - S (Saturation): Pureza/saturacao da cor (0-255)")
print("HSV - V (Value): Brilho/luminosidade da cor (0-255)")
print("LAB - L (Lightness): Luminosidade (0-255 em uint8)")
print("LAB - a: Eixo Verde (valores baixos) ate Vermelho (valores altos)")
print("LAB - b: Eixo Azul (valores baixos) ate Amarelo (valores altos)")

h_bgr = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
s_bgr = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
v_bgr = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

l_bgr = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)
a_bgr = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
b_bgr = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)

lbl_img = add_label(img, "Original BGR")
lbl_h = add_label(h_bgr, "HSV - H (Hue)")
lbl_s = add_label(s_bgr, "HSV - S (Saturation)")
lbl_v = add_label(v_bgr, "HSV - V (Value)")

lbl_l = add_label(l_bgr, "LAB - L (Lightness)")
lbl_a = add_label(a_bgr, "LAB - a (Green-Red)")
lbl_b = add_label(b_bgr, "LAB - b (Blue-Yellow)")

row1 = cv2.hconcat([lbl_img, lbl_h, lbl_s, lbl_v])
row2 = cv2.hconcat([lbl_img, lbl_l, lbl_a, lbl_b])
channels_grid = cv2.vconcat([row1, row2])

hsv_0 = hsv.copy()
hsv_0[:, :, 1] = 0
bgr_sat_0 = cv2.cvtColor(hsv_0, cv2.COLOR_HSV2BGR)

hsv_50 = hsv.copy()
hsv_50[:, :, 1] = np.clip(hsv[:, :, 1] * 0.5, 0, 255).astype(np.uint8)
bgr_sat_50 = cv2.cvtColor(hsv_50, cv2.COLOR_HSV2BGR)

hsv_150 = hsv.copy()
hsv_150[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
bgr_sat_150 = cv2.cvtColor(hsv_150, cv2.COLOR_HSV2BGR)

lbl_sat_0 = add_label(bgr_sat_0, "Saturation 0%")
lbl_sat_50 = add_label(bgr_sat_50, "Saturation 50%")
lbl_sat_150 = add_label(bgr_sat_150, "Saturation 150%")

sat_grid = cv2.hconcat([lbl_sat_0, lbl_sat_50, lbl_sat_150])

while True:
    cv2.imshow("7 Channels (Row1: BGR, H, S, V | Row2: BGR, L, a, b)", cv2.resize(channels_grid, (1280, 480)))
    cv2.imshow("Saturation (0%, 50%, 150%)", cv2.resize(sat_grid, (1280, 320)))
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
