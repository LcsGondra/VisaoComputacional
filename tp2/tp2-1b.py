import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio, win_transform, obter_diretorios


def main():
    dirs = obter_diretorios(__file__)
    saida_dir = dirs["saidas"]

    video_candidatos = [
        Path("../Videos/warden and the paunch.mp4"),
        Path("./Videos/warden and the paunch.mp4"),
    ]
    video_path = None
    for vc in video_candidatos:
        if vc.exists():
            video_path = str(vc)
            break

    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)

    if cap is None or not cap.isOpened():
        raise RuntimeError("Nenhuma camera ou video disponivel para execucao do Exercicio 1B.")

    print(f"Stream carregado com sucesso (Fonte: {video_path if video_path else 'Webcam 0'}).")

    presets_cores = {
        "1": (
            "Azul",
            [np.array([92, 70, 45], dtype=np.uint8)],
            [np.array([132, 255, 255], dtype=np.uint8)],
        ),
        "2": (
            "Verde",
            [np.array([35, 70, 45], dtype=np.uint8)],
            [np.array([85, 255, 255], dtype=np.uint8)],
        ),
        "3": (
            "Vermelho",
            [
                np.array([0, 90, 50], dtype=np.uint8),
                np.array([168, 90, 50], dtype=np.uint8),
            ],
            [
                np.array([10, 255, 255], dtype=np.uint8),
                np.array([179, 255, 255], dtype=np.uint8),
            ],
        ),
        "4": (
            "Amarelo / Laranja",
            [np.array([14, 85, 70], dtype=np.uint8)],
            [np.array([36, 255, 255], dtype=np.uint8)],
        ),
    }

    preset_inicial = "1"
    nome_cor_ativa, lista_min, lista_max = presets_cores[preset_inicial]

    tolerancia_extra = 0

    mouse_data = {
        "clique_solicitado": False,
        "x": 0,
        "y": 0,
        "hsv_frame": None,
    }

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and mouse_data["hsv_frame"] is not None:
            nw = float(win_transform["new_w"])
            nh = float(win_transform["new_h"])
            if nw > 0 and nh > 0:
                gx = int((x - win_transform["left"]) * (win_transform["img_w"] / nw))
                gy = int((y - win_transform["top"]) * (win_transform["img_h"] / nh))
                if 0 <= gx < win_transform["img_w"] and 0 <= gy < win_transform["img_h"]:
                    mouse_data["x"] = gx
                    mouse_data["y"] = gy
                    mouse_data["clique_solicitado"] = True

    kernel_abertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    janela_feed = "Segmentacao de ROI - TP2 Exercicio 1B"
    janela_mask = "Mascara Morfologica Limpa - TP2 Ex1B"
    cv2.namedWindow(janela_feed, cv2.WINDOW_NORMAL)
    cv2.namedWindow(janela_mask, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela_feed, 800, 480)
    cv2.resizeWindow(janela_mask, 400, 300)
    cv2.setMouseCallback(janela_feed, on_mouse)

    print("=== INICIANDO PIPELINE DE SEGMENTACAO DE ROI (TP2 - 1B) ===")
    print("Controles:")
    print("  - [Clique no Objeto]: Calibra a cor automaticamente")
    print("  - [1] Azul | [2] Verde | [3] Vermelho | [4] Amarelo/Laranja")
    print("  - [+] / [-]: Ajusta tolerancia")
    print("  - [Q]: Encerrar")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        h, w = frame.shape[:2]
        area_total = float(h * w)

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
        mouse_data["hsv_frame"] = hsv

        if mouse_data["clique_solicitado"]:
            cx, cy = mouse_data["x"], mouse_data["y"]
            x1, x2 = max(0, cx - 2), min(w, cx + 3)
            y1, y2 = max(0, cy - 2), min(h, cy + 3)
            regiao_hsv = hsv[y1:y2, x1:x2]
            mouse_data["clique_solicitado"] = False

            if regiao_hsv.size > 0:
                mh = float(np.mean(regiao_hsv[:, :, 0]))
                ms = float(np.mean(regiao_hsv[:, :, 1]))
                mv = float(np.mean(regiao_hsv[:, :, 2]))

                if not (np.isnan(mh) or np.isnan(ms) or np.isnan(mv)):
                    media_h, media_s, media_v = int(mh), int(ms), int(mv)
                    margem_h = 16 + max(0, tolerancia_extra)
                    h_min_val = max(0, media_h - margem_h)
                    h_max_val = min(179, media_h + margem_h)
                    s_min_val = max(65, media_s - 50)
                    s_max_val = 255
                    v_min_val = max(40, media_v - 70)
                    v_max_val = 255

                    if media_h < 15 or media_h > 165:
                        lista_min = [
                            np.array([0, s_min_val, v_min_val], dtype=np.uint8),
                            np.array([165, s_min_val, v_min_val], dtype=np.uint8),
                        ]
                        lista_max = [
                            np.array([15, s_max_val, v_max_val], dtype=np.uint8),
                            np.array([179, s_max_val, v_max_val], dtype=np.uint8),
                        ]
                    else:
                        lista_min = [np.array([h_min_val, s_min_val, v_min_val], dtype=np.uint8)]
                        lista_max = [np.array([h_max_val, s_max_val, v_max_val], dtype=np.uint8)]

                    nome_cor_ativa = f"Calibrado (H:{media_h}, S:{media_s}, V:{media_v})"
                    print(f"-> Cor calibrada: H=[{h_min_val}, {h_max_val}], S=[{s_min_val}, {s_max_val}], V=[{v_min_val}, {v_max_val}]")

        mascara = np.zeros((h, w), dtype=np.uint8)
        for h_min, h_max in zip(lista_min, lista_max):
            h_min_ajustado = np.array([
                max(0, int(h_min[0]) - tolerancia_extra),
                max(50, int(h_min[1]) - tolerancia_extra),
                max(30, int(h_min[2]) - tolerancia_extra),
            ], dtype=np.uint8)
            h_max_ajustado = np.array([
                min(179, int(h_max[0]) + tolerancia_extra),
                255,
                255,
            ], dtype=np.uint8)
            sub_mask = cv2.inRange(hsv, h_min_ajustado, h_max_ajustado)
            mascara = cv2.bitwise_or(mascara, sub_mask)

        mascara_limpa = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_abertura, iterations=1)
        mascara_limpa = cv2.morphologyEx(mascara_limpa, cv2.MORPH_CLOSE, kernel_fechamento, iterations=2)

        contornos, _ = cv2.findContours(
            mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_saida = frame.copy()
        area_roi = 0.0
        proporcao_area = 0.0

        area_minima_valida = max(350.0, area_total * 0.001)
        contornos_validos = [c for c in contornos if cv2.contourArea(c) >= area_minima_valida]

        if len(contornos_validos) > 0:
            maior_contorno = max(contornos_validos, key=cv2.contourArea)
            area_roi = float(cv2.contourArea(maior_contorno))
            proporcao_area = (area_roi / area_total) * 100.0
            x, y, bw, bh = cv2.boundingRect(maior_contorno)

            cv2.rectangle(frame_saida, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            overlay = frame_saida.copy()
            cv2.drawContours(overlay, [maior_contorno], -1, (0, 215, 255), -1)
            cv2.addWeighted(overlay, 0.45, frame_saida, 0.55, 0, frame_saida)

            cv2.putText(
                frame_saida,
                f"ROI: {area_roi:.0f} px ({proporcao_area:.2f}%)",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if frame_idx % 10 == 0:
            print(
                f"Frame: {frame_idx:04d} | Area Total: {area_total:.0f} px | Proporcao ROI: {proporcao_area:.3f}% | Cor: {nome_cor_ativa} | Tol: +{tolerancia_extra}"
            )

        cv2.rectangle(frame_saida, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.putText(
            frame_saida,
            f"Cor: {nome_cor_ativa} | ROI: {proporcao_area:.2f}% | Tol: +{tolerancia_extra}",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_saida,
            "Clique no objeto p/ calibrar | [1] Azul [2] Verde [3] Vermelho [4] Amarelo | [+] / [-] Tolerancia | [Q] Sair",
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        imshow_keep_aspect_ratio(janela_feed, frame_saida)
        imshow_keep_aspect_ratio(janela_mask, mascara_limpa)

        if frame_idx == 30:
            cv2.imwrite(str(saida_dir / "tp2_1b_roi_destacada.png"), frame_saida)
            cv2.imwrite(
                str(saida_dir / "tp2_1b_mascara_morfologica.png"), mascara_limpa
            )

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif chr(key) in presets_cores:
            nome_cor_ativa, lista_min, lista_max = presets_cores[chr(key)]
            print(f"-> Preset alterado para: {nome_cor_ativa}")
        elif key == ord("+") or key == ord("="):
            tolerancia_extra = min(25, tolerancia_extra + 3)
            print(f"-> Tolerancia aumentada para: +{tolerancia_extra}")
        elif key == ord("-") or key == ord("_"):
            tolerancia_extra = max(-15, tolerancia_extra - 3)
            print(f"-> Tolerancia reduzida para: +{tolerancia_extra}")

        frame_idx += 1

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Processamento do Exercicio 1B concluido.")


if __name__ == "__main__":
    main()
