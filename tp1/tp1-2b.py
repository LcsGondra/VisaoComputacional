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

    cv2.putText(img_copy, text, (margin_x, margin_y), font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA)
    cv2.putText(img_copy, text, (margin_x, margin_y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return img_copy

source = "Imagens/BallPit.jpg"

img = cv2.imread(source)
if img is None:
    print(f"Error: Could not read image {source}")
    sys.exit(1)

if img.shape[0] < 480 or img.shape[1] < 480:
    img = cv2.resize(img, (max(img.shape[1], 480), max(img.shape[0], 480)))

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpened_manual = cv2.filter2D(img, -1, kernel)

blurred = cv2.GaussianBlur(img, (9, 9), 10.0)
unsharp_mask = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

var_orig = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
var_manual = cv2.Laplacian(cv2.cvtColor(sharpened_manual, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
var_unsharp = cv2.Laplacian(cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

print(f"Laplacian Variance - Original: {var_orig:.2f}")
print(f"Laplacian Variance - Sharpening Manual (cv2.filter2D): {var_manual:.2f}")
print(f"Laplacian Variance - Unsharp Masking: {var_unsharp:.2f}")

if var_manual > var_unsharp and var_manual > var_orig:
    best_method = "Sharpening Manual (cv2.filter2D)"
elif var_unsharp > var_manual and var_unsharp > var_orig:
    best_method = "Unsharp Masking"
else:
    best_method = "Original"

print(f"Metodo com maior nitidez (maior variancia do Laplaciano): {best_method}")

lbl_orig = add_label(img, "Original")
lbl_manual = add_label(sharpened_manual, "Sharpening Manual")
lbl_unsharp = add_label(unsharp_mask, "Unsharp Masking")

side_by_side = cv2.hconcat([lbl_orig, lbl_manual, lbl_unsharp])

win_name = "Original vs Sharpening Manual vs Unsharp Masking - TP1 Ex2 Item B"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 1440, 360)

while True:
    imshow_keep_aspect_ratio(win_name, side_by_side)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
