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

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
otsu_val, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Calculated Otsu Threshold Value: {otsu_val:.2f}")
print("Technical Justification:")
print("O metodo de Otsu eh superior ao limiar global em iluminacao nao uniforme pois calcula automaticamente")
print("o limiar otimo minimizando a variancia intraclasse (ou maximizando a variancia interclasse) do histograma,")
print("adaptando-se a distribuicao bimodal de intensidade da imagem sem depender de um valor fixo arbitrario.")

g_bgr = cv2.cvtColor(thresh_global, cv2.COLOR_GRAY2BGR)
a_bgr = cv2.cvtColor(thresh_adapt, cv2.COLOR_GRAY2BGR)
o_bgr = cv2.cvtColor(thresh_otsu, cv2.COLOR_GRAY2BGR)

lbl_g = add_label(g_bgr, "Global (T=127)")
lbl_a = add_label(a_bgr, "Adaptive Gaussian")
lbl_o = add_label(o_bgr, f"Otsu (T={otsu_val:.0f})")

side_by_side = cv2.hconcat([lbl_g, lbl_a, lbl_o])

while True:
    cv2.imshow("Global vs Adaptive vs Otsu Thresholding - TP1 Ex3 Item A", cv2.resize(side_by_side, (1440, 360)))
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

cv2.destroyAllWindows()
