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


def add_label(image, text, max_w_ratio=0.88, max_h_ratio=0.12):
    img_copy = image.copy()
    h, w = img_copy.shape[:2]

    target_max_w = int(w * max_w_ratio)
    target_max_h = int(h * max_h_ratio)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    if text_w > 0 and text_h > 0:
        scale_w = target_max_w / float(text_w)
        scale_h = target_max_h / float(text_h)
        font_scale = min(scale_w, scale_h, 1.2)
        font_scale = max(0.4, font_scale)

    thickness = max(1, int(font_scale * 2.5))
    outline_thickness = max(2, thickness * 3)

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    margin_x = max(10, int((w - text_w) / 2))
    margin_y = max(text_h + 10, int(h * 0.08 + text_h * 0.5))

    cv2.putText(
        img_copy,
        text,
        (margin_x, margin_y),
        font,
        font_scale,
        (0, 0, 0),
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        img_copy,
        text,
        (margin_x, margin_y),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return img_copy


source = "./Imagens/BallPit.jpg"

img = cv2.imread(source)
if img is None:
    print(f"Error: Could not read image {source}")
    sys.exit(1)

if img.shape[0] < 480 or img.shape[1] < 480:
    img = cv2.resize(img, (max(img.shape[1], 480), max(img.shape[0], 480)))

# Conversao para os espacos de cor HSV e LAB
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# Separacao individual dos canais
h, s, v = cv2.split(hsv)
l, a, b = cv2.split(lab)

# ==============================================================================
# REPRESENTACAO DOS CANAIS NOS ESPACOS DE COR (REQUISITO EXERCICIO 2 ITEM A):
#
# 1. ESPACO HSV (Hue, Saturation, Value):
#    - H (Hue / Matiz): Representa a tonalidade pura da cor em um circulo cromatico.
#      No OpenCV, assume valores de 0 a 179 (onde 0/180=Vermelho, 60=Verde, 120=Azul).
#    - S (Saturation / Saturacao): Representa a pureza ou intensidade da cor.
#      Varia de 0 (cinza totalmente descolorido) a 255 (cor pura e viva).
#    - V (Value / Brilho): Representa a intensidade luminosa ou brilho da cor.
#      Varia de 0 (preto absoluto) a 255 (brilho maximo).
#
# 2. ESPACO LAB (Lightness, a, b):
#    - L (Lightness / Luminosidade): Representa a percepcao de brilho do olho humano.
#      Varia de 0 (preto) a 255 (branco), independente das componentes de cor.
#    - a (Eixo de cor Verde-Vermelho): Representa a posicao da cor no eixo oponente verde-vermelho.
#      Valores baixos/escuros tendem ao verde; valores altos/claros tendem ao vermelho.
#    - b (Eixo de cor Azul-Amarelo): Representa a posicao da cor no eixo oponente azul-amarelo.
#      Valores baixos/escuros tendem ao azul; valores altos/claros tendem ao amarelo.
# ==============================================================================

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

# Alteracao da saturacao para 0%, 50% e 150% do valor original
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

win1 = "7 Channels (Row1: BGR, H, S, V | Row2: BGR, L, a, b)"
win2 = "Saturation (0%, 50%, 150%)"
cv2.namedWindow(win1, cv2.WINDOW_NORMAL)
cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win1, 1280, 480)
cv2.resizeWindow(win2, 1280, 320)

while True:
    imshow_keep_aspect_ratio(win1, channels_grid)
    imshow_keep_aspect_ratio(win2, sat_grid)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
