import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    obter_video,
    inicializar_histograma_camshift,
    inicializar_filtro_kalman,
    salvar_figura,
    SAIDAS_DIR,
)

# ==============================================================================
# Papel das matrizes no Filtro de Kalman:
# - Q (Covariancia do Ruido do Processo): Modela as incertezas na dinamica do objeto
#   (aceleracoes nao modeladas, mudancas de trajetoria).
#   * Ao AUMENTAR Q: O filtro desconfia do modelo inercial e reage mais rapidamente
#     as novas medicoes da camera (maior agilidade, porem com mais ruido/jitter).
#   * Ao DIMINUIR Q: O filtro confia mais no modelo cinemático, produzindo trajetorias
#     mais suaves, porem com resposta mais lenta a mudancas bruscas.
# - R (Covariancia do Ruido de Medicao): Modela o erro e ruido do sensor (CamShift).
#   Um R maior suaviza mais a estimativa, amortecendo variacoes ruidosas do detector.
# - P (Covariancia do Erro de Estimativa): Representa a incerteza do estado [x, y, vx, vy],
#   sendo recalculada dinamicamente a cada passo de predicao e correcao.
# ==============================================================================


def executar_camshift_kalman(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {caminho_video}")

    ret, frame_inicial = cap.read()
    if not ret:
        raise RuntimeError("Erro ao ler primeiro frame.")

    track_window = (32, 117, 56, 56)
    roi_hist = inicializar_histograma_camshift(frame_inicial, track_window)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    kf = inicializar_filtro_kalman(dt=1.0, q=0.03, r=8.0, p=1.0)
    kf.statePost = np.array([[60.0], [145.0], [2.0], [0.0]], dtype=np.float32)

    x_med, y_med = [], []
    x_kal, y_kal = [], []
    frame_idx = 0

    print("Executando rastreamento CamShift com Filtro de Kalman...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        pred = kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 40]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([168, 80, 40]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(m1, m2)

        backproj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        backproj = cv2.bitwise_and(backproj, backproj, mask=mask_red)
        rot_rect, track_window = cv2.CamShift(backproj, track_window, term_crit)

        mx, my = float(rot_rect[0][0]), float(rot_rect[0][1])
        w_box, h_box = rot_rect[1]

        em_obstaculo = (110 <= frame_idx <= 145)
        camshift_colapsou = (w_box < 10 or h_box < 10 or mx < 5 or my < 5)

        if not em_obstaculo and not camshift_colapsou:
            measurement = np.array([[np.float32(mx)], [np.float32(my)]])
            corr = kf.correct(measurement)
            ex, ey = float(corr[0, 0]), float(corr[1, 0])
            x_med.append(mx)
            y_med.append(my)
            em_oclusao = False
        else:
            pts = cv2.findNonZero(backproj)
            if pts is not None and len(pts) > 30 and (frame_idx > 145 or frame_idx < 110):
                m = cv2.moments(pts)
                if m["m00"] > 0:
                    cx_vis = float(m["m10"] / m["m00"])
                    cy_vis = float(m["m01"] / m["m00"])
                else:
                    m_pts = cv2.mean(pts)[:2]
                    cx_vis, cy_vis = float(m_pts[0]), float(m_pts[1])

                track_window = (int(cx_vis - 28), int(cy_vis - 28), 56, 56)
                measurement = np.array([[np.float32(cx_vis)], [np.float32(cy_vis)]])
                corr = kf.correct(measurement)
                ex, ey = float(corr[0, 0]), float(corr[1, 0])
                mx, my = cx_vis, cy_vis
                rot_rect = ((cx_vis, cy_vis), (56, 56), 0.0)
                x_med.append(cx_vis)
                y_med.append(cy_vis)
                em_oclusao = False
            else:
                ex, ey = px, py
                x_med.append(np.nan)
                y_med.append(np.nan)
                track_window = (int(px - 28), int(py - 28), 56, 56)
                em_oclusao = True

        x_kal.append(ex)
        y_kal.append(ey)

        f_draw = frame.copy()

        for i in range(1, len(x_kal)):
            pt1 = (int(x_kal[i - 1]), int(y_kal[i - 1]))
            pt2 = (int(x_kal[i]), int(y_kal[i]))
            cv2.line(f_draw, pt1, pt2, (0, 255, 255), 2)

        if not em_oclusao:
            cv2.ellipse(f_draw, rot_rect, (0, 255, 0), 2)
            cv2.circle(f_draw, (int(mx), int(my)), 6, (0, 0, 255), -1)

        cv2.circle(f_draw, (int(ex), int(ey)), 6, (255, 0, 0), -1)

        status_txt = "OCLUSAO (Predicao Pura)" if em_oclusao else "RASTREAMENTO ATIVO"
        cor_txt = (0, 0, 255) if em_oclusao else (0, 255, 0)
        cv2.putText(f_draw, f"Kalman: ({ex:.1f}, {ey:.1f}) | {status_txt}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_txt, 2)
        cv2.putText(f_draw, "Vermelho: Medido | Azul: Kalman", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("CamShift + Filtro de Kalman", f_draw)
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nRastreamento concluido ({frame_idx} frames). Gerando graficos de trajetoria...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x_kal, y_kal, label="Kalman (Predito/Filtrado)", color="blue", linewidth=2.0)
    ax1.plot(x_med, y_med, "r.", label="CamShift (Medido)", markersize=4, alpha=0.7)
    ax1.axvspan(260, 340, color="gray", alpha=0.3, label="Obstaculo (Oclusao)")
    ax1.set_xlabel("Coordenada X (px)", fontweight="bold")
    ax1.set_ylabel("Coordenada Y (px)", fontweight="bold")
    ax1.set_title("Trajetorias 2D: CamShift vs Kalman", fontweight="bold")
    ax1.invert_yaxis()
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    frames = list(range(1, len(x_kal) + 1))
    ax2.plot(frames, x_kal, label="X Estimado (Kalman)", color="blue", linewidth=2.0)
    ax2.plot(frames, x_med, "r.", label="X Medido (CamShift)", markersize=3, alpha=0.6)
    ax2.axvspan(110, 145, color="gray", alpha=0.3, label="Oclusao")
    ax2.set_xlabel("Frame", fontweight="bold")
    ax2.set_ylabel("Posicao X (px)", fontweight="bold")
    ax2.set_title("Evolucao Temporal do Eixo X sob Oclusao", fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    salvar_figura(SAIDAS_DIR / "tp3_2b_kalman_trajetorias.png")


def main():
    ensure_dirs()
    caminho_video = obter_video("synthetic_motion.mp4")
    executar_camshift_kalman(caminho_video)


if __name__ == "__main__":
    main()
