import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import ensure_dirs, obter_video, inicializar_histograma_camshift, salvar_figura, SAIDAS_DIR


def executar_subtracao_e_camshift(caminho_video):
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {caminho_video}")

    ret, frame_inicial = cap.read()
    if not ret:
        raise RuntimeError("Erro ao ler primeiro frame.")

    # Janela inicial para CamShift (objeto vermelho ou pedestre)
    if "vtest" in str(caminho_video).lower():
        track_window = (380, 200, 60, 100)
    else:
        track_window = (32, 117, 56, 56)

    roi_hist = inicializar_histograma_camshift(frame_inicial, track_window)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    sub_mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    sub_knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

    tempos_mog2, tempos_knn, tempos_cam = [], [], []
    contagens_mog2, contagens_knn = [], []
    frame_idx = 0
    snapshots = []

    print("Executando subtracao de fundo (MOG2 / KNN) e CamShift...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1. Subtracao MOG2
        t0 = time.perf_counter()
        mask_mog2 = sub_mog2.apply(frame)
        tempos_mog2.append((time.perf_counter() - t0) * 1000)

        # 2. Subtracao KNN
        t1 = time.perf_counter()
        mask_knn = sub_knn.apply(frame)
        tempos_knn.append((time.perf_counter() - t1) * 1000)

        # Contornos e contagem de objetos
        cnts_mog2, _ = cv2.findContours((mask_mog2 == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objs_mog2 = [c for c in cnts_mog2 if cv2.contourArea(c) > 100]
        contagens_mog2.append(len(objs_mog2))

        cnts_knn, _ = cv2.findContours((mask_knn == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objs_knn = [c for c in cnts_knn if cv2.contourArea(c) > 100]
        contagens_knn.append(len(objs_knn))

        # 3. CamShift
        t2 = time.perf_counter()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, 80, 40]), np.array([12, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([168, 80, 40]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(m1, m2)
        backproj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        backproj = cv2.bitwise_and(backproj, backproj, mask=mask_red)
        rot_rect, track_window = cv2.CamShift(backproj, track_window, term_crit)
        tempos_cam.append((time.perf_counter() - t2) * 1000)

        # Anotacao visual
        frame_anotado = frame.copy()
        for c in objs_mog2:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame_anotado, (x, y), (x + w, y + h), (255, 255, 0), 1)

        cv2.ellipse(frame_anotado, rot_rect, (0, 255, 0), 2)
        cx, cy = map(int, rot_rect[0])
        cv2.circle(frame_anotado, (cx, cy), 4, (0, 0, 255), -1)

        cv2.imshow("MOG2 Foreground", mask_mog2)
        cv2.imshow("KNN Foreground", mask_knn)
        cv2.imshow("Rastreamento CamShift", frame_anotado)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

        if frame_idx in [30, 80, 130]:
            snapshots.append({
                "idx": frame_idx,
                "orig": frame_anotado.copy(),
                "mog2": mask_mog2.copy(),
                "knn": mask_knn.copy(),
            })

    cap.release()
    cv2.destroyAllWindows()

    m_mog2 = np.mean(tempos_mog2) if tempos_mog2 else 0
    m_knn = np.mean(tempos_knn) if tempos_knn else 0
    m_cam = np.mean(tempos_cam) if tempos_cam else 0

    print(f"\nResultados do Rastreamento ({frame_idx} frames):")
    print(f"MOG2    : Tempo medio = {m_mog2:.2f} ms ({1000/max(1e-3, m_mog2):.1f} FPS) | Media de objetos = {np.mean(contagens_mog2):.1f}")
    print(f"KNN     : Tempo medio = {m_knn:.2f} ms ({1000/max(1e-3, m_knn):.1f} FPS) | Media de objetos = {np.mean(contagens_knn):.1f}")
    print(f"CamShift: Tempo medio = {m_cam:.2f} ms ({1000/max(1e-3, m_cam):.1f} FPS)")

    if snapshots:
        fig, axs = plt.subplots(len(snapshots), 3, figsize=(14, 3 * len(snapshots)))
        for i, s in enumerate(snapshots):
            axs[i, 0].imshow(cv2.cvtColor(s["orig"], cv2.COLOR_BGR2RGB))
            axs[i, 0].set_ylabel(f"Frame {s['idx']}", fontweight="bold")
            axs[i, 1].imshow(s["mog2"], cmap="gray")
            axs[i, 2].imshow(s["knn"], cmap="gray")
            if i == 0:
                axs[i, 0].set_title("CamShift Tracking", fontweight="bold")
                axs[i, 1].set_title("Mascara MOG2", fontweight="bold")
                axs[i, 2].set_title("Mascara KNN", fontweight="bold")
            for col in range(3):
                axs[i, col].set_xticks([])
                axs[i, col].set_yticks([])

        salvar_figura(SAIDAS_DIR / "tp3_2a_subtracao_camshift.png")


def main():
    ensure_dirs()
    caminho_video = obter_video("synthetic_motion.mp4")
    executar_subtracao_e_camshift(caminho_video)


if __name__ == "__main__":
    main()
