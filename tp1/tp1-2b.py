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

while True:
    cv2.imshow(
        "Original vs Sharpening Manual vs Unsharp Masking - TP1 Ex2 Item B",
        cv2.resize(side_by_side, (1440, 360)),
    )
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

cv2.destroyAllWindows()
