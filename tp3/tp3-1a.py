import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import ensure_dirs, obter_video, salvar_figura, SAIDAS_DIR


def processar_video_hog(caminho_video, max_frames=60):
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {caminho_video}")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cenarios = [
        {"nome": "Cenario Preciso", "winStride": (4, 4), "padding": (8, 8), "scale": 1.03},
        {"nome": "Cenario Balanceado", "winStride": (8, 8), "padding": (8, 8), "scale": 1.05},
        {"nome": "Cenario Rapido", "winStride": (16, 16), "padding": (8, 8), "scale": 1.10},
    ]

    tabela = []
    snapshots = []

    for cenario in cenarios:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tempos = []
        total_det = 0
        frames_proc = 0
        frame_exemplo = None

        print(f"\nProcessando {cenario['nome']}...")
        while frames_proc < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames_proc += 1

            t0 = time.perf_counter()
            rects, weights = hog.detectMultiScale(
                frame,
                winStride=cenario["winStride"],
                padding=cenario["padding"],
                scale=cenario["scale"],
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            tempos.append(dt_ms)

            rects_list = [[x, y, w, h] for (x, y, w, h) in rects]
            weights_list = [float(w[0]) if isinstance(w, (list, np.ndarray)) else float(w) for w in weights] if len(weights) > 0 else []
            indices = cv2.dnn.NMSBoxes(rects_list, weights_list, score_threshold=0.15, nms_threshold=0.4) if rects_list else []
            n_det = len(indices) if len(indices) > 0 else 0
            total_det += n_det

            f_draw = frame.copy()
            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    x, y, w, h = rects_list[int(idx)]
                    cv2.rectangle(f_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(f_draw, f"Pedestre", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(f_draw, f"{cenario['nome']} | Frame {frames_proc} | {n_det} det | {dt_ms:.1f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            if frame_exemplo is None or n_det >= 2:
                frame_exemplo = f_draw

            cv2.imshow("Deteccao de Pedestres HOG", f_draw)
            if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
                break

            if frames_proc % 15 == 0 or frames_proc == 1:
                print(f"  Frame {frames_proc:02d}/{max_frames:02d} | Detecoes: {n_det} | Tempo: {dt_ms:.1f} ms")

        cv2.destroyAllWindows()
        media_ms = float(np.mean(tempos)) if tempos else 0.0
        fps = 1000.0 / max(1e-3, media_ms)
        det_media = total_det / max(1, frames_proc)

        tabela.append({
            "Cenario": cenario["nome"],
            "winStride": str(cenario["winStride"]),
            "scale": cenario["scale"],
            "Det/Frame": f"{det_media:.2f}",
            "Latencia": f"{media_ms:.2f} ms",
            "FPS": f"{fps:.1f}",
        })
        snapshots.append((cenario["nome"], frame_exemplo if frame_exemplo is not None else frame))

    cap.release()
    return tabela, snapshots


def main():
    ensure_dirs()
    caminho_video = obter_video("vtest.avi")
    print(f"Video utilizado: {Path(caminho_video).name}")

    tabela, snapshots = processar_video_hog(caminho_video, max_frames=45)

    print("\nTabela Comparativa — HOG Pedestres:")
    print(f"{'Cenario':<20} | {'winStride':<10} | {'scale':<6} | {'Det/Frame':<10} | {'Latencia':<12} | {'FPS'}")
    print("-" * 72)
    for r in tabela:
        print(f"{r['Cenario']:<20} | {r['winStride']:<10} | {r['scale']:<6} | {r['Det/Frame']:<10} | {r['Latencia']:<12} | {r['FPS']}")
    print("-" * 72)

    fig, axs = plt.subplots(1, len(snapshots), figsize=(16, 5))
    for i, (nome, img) in enumerate(snapshots):
        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"{nome}\n({tabela[i]['Latencia']} — {tabela[i]['FPS']} FPS)", fontweight="bold")
        axs[i].axis("off")

    salvar_figura(SAIDAS_DIR / "tp3_1a_hog_comparativo.png")


if __name__ == "__main__":
    main()
