import cv2
import numpy as np
import time
from pathlib import Path


def gerar_quadro_sintetico(t, largura=640, altura=480):
    quadro = np.full((altura, largura, 3), (230, 230, 230), dtype=np.uint8)

    cx = int(largura / 2 + (largura / 3) * np.sin(t * 1.5))
    cy = int(altura / 2 + (altura / 4) * np.cos(t * 1.5))
    raio = int(45 + 15 * np.sin(t * 0.8))

    cv2.circle(quadro, (cx, cy), raio, (20, 40, 230), -1)
    cv2.circle(quadro, (cx - 15, cy - 15), raio // 3, (40, 80, 255), -1)

    ruido = np.random.normal(0, 8, quadro.shape).astype(np.int16)
    quadro = np.clip(quadro.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    return quadro


def main():
    saida_dir = Path("dados/saidas")
    saida_dir.mkdir(parents=True, exist_ok=True)

    fonte = 0
    cap = cv2.VideoCapture(fonte)
    usar_sintetico = False

    if not cap.isOpened():
        print("Camera fisica nao detectada. Utilizando gerador de stream de video sintetico...")
        usar_sintetico = True

    hsv_min = np.array([0, 120, 70], dtype=np.uint8)
    hsv_max = np.array([10, 255, 255], dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    janela_nome = "Segmentacao de ROI - TP2 Exercicio 1B"
    cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela_nome, 960, 540)

    frame_idx = 0
    start_time = time.perf_counter()
    max_frames_demo = 150

    while frame_idx < max_frames_demo:
        if usar_sintetico:
            t = (time.perf_counter() - start_time)
            frame = gerar_quadro_sintetico(t)
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        h, w = frame.shape[:2]
        area_total = float(h * w)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, hsv_min, hsv_max)

        mascara_limpa = cv2.erode(mascara, kernel, iterations=1)
        mascara_limpa = cv2.dilate(mascara_limpa, kernel, iterations=2)

        contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_saida = frame.copy()
        area_roi = 0.0
        proporcao_area = 0.0

        if len(contornos) > 0:
            maior_contorno = max(contornos, key=cv2.contourArea)
            area_roi = float(cv2.contourArea(maior_contorno))

            if area_roi > 100:
                proporcao_area = (area_roi / area_total) * 100.0
                x, y, bw, bh = cv2.boundingRect(maior_contorno)

                cv2.rectangle(frame_saida, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

                overlay = frame_saida.copy()
                cv2.drawContours(overlay, [maior_contorno], -1, (0, 200, 255), -1)
                cv2.addWeighted(overlay, 0.45, frame_saida, 0.55, 0, frame_saida)

                cv2.putText(
                    frame_saida,
                    f"ROI Area: {area_roi:.0f} px ({proporcao_area:.2f}%)",
                    (x, max(25, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        info_txt = f"Frame: {frame_idx:04d} | Area Frame: {area_total:.0f} | Proporcao ROI: {proporcao_area:.3f}%"
        if frame_idx % 10 == 0:
            print(info_txt)

        cv2.putText(
            frame_saida,
            f"Frame {frame_idx} | ROI: {proporcao_area:.2f}% do frame",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(janela_nome, frame_saida)
        cv2.imshow("Mascara Morfologica", mascara_limpa)

        if frame_idx == 30:
            cv2.imwrite(str(saida_dir / "tp2_1b_roi_destacada.png"), frame_saida)
            cv2.imwrite(str(saida_dir / "tp2_1b_mascara_morfologica.png"), mascara_limpa)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

        frame_idx += 1

    if not usar_sintetico:
        cap.release()
    cv2.destroyAllWindows()
    print("Processamento concluido.")


if __name__ == "__main__":
    main()
