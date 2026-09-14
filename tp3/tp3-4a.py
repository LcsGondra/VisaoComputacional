import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from utils import (
    ensure_dirs,
    obter_video,
    obter_modelos_caffe_idade_genero,
    prever_idade_genero_caffe,
    salvar_figura,
    SAIDAS_DIR,
)


def classificar_5_rostos(age_net, gender_net):
    """Testa e gera o relatorio de 5 rostos distintos conforme solicitado no enunciado."""
    olivetti = fetch_olivetti_faces()
    indices_faces = [0, 20, 50, 100, 150]
    relatorio = []
    snapshots = []

    print("Relatorio de Classificacao em 5 Rostos (Caffe DNN):")
    print(f"{'Rosto':<24} | {'Genero':<12} | {'Conf. Gen.':<10} | {'Idade':<12} | {'Conf. Idade'}")
    print("-" * 75)

    for i, idx in enumerate(indices_faces, start=1):
        face_gray = (olivetti.images[idx] * 255).astype(np.uint8)
        face_bgr = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
        face_bgr = cv2.resize(face_bgr, (227, 227))

        genero, conf_g, idade, conf_a = prever_idade_genero_caffe(face_bgr, age_net, gender_net)

        linha = {
            "Rosto": f"Individuo #{i:02d} (Face {idx})",
            "Genero": genero,
            "Conf_Gen": f"{conf_g * 100:.1f}%",
            "Idade": idade,
            "Conf_Idade": f"{conf_a * 100:.1f}%",
        }
        relatorio.append(linha)
        print(f"{linha['Rosto']:<24} | {linha['Genero']:<12} | {linha['Conf_Gen']:<10} | {linha['Idade']:<12} | {linha['Conf_Idade']}")

        # Snap para plot
        face_anotada = face_bgr.copy()
        cv2.putText(face_anotada, f"{genero} ({conf_g*100:.0f}%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(face_anotada, f"{idade} ({conf_a*100:.0f}%)", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        snapshots.append((f"Rosto #{i}", face_anotada))

    print("-" * 75)

    # Plota os 5 rostos
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.5))
    for i, (tit, img) in enumerate(snapshots):
        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].set_title(tit, fontweight="bold")
        axs[i].axis("off")
    salvar_figura(SAIDAS_DIR / "tp3_4a_5_rostos_caffe.png")


def executar_pipeline_video(age_net, gender_net, caminho_video):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        print(f"[!] Nao foi possivel abrir video para o feed: {caminho_video}")
        return

    tempos = []
    frame_count = 0
    print("\nIniciando feed de video com deteccao facial e classificacao...")

    while frame_count < 60:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t0 = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in faces:
            pad = 10
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)
            roi = frame[y1:y2, x1:x2]

            if roi.shape[0] > 10 and roi.shape[1] > 10:
                gen, conf_g, idade, conf_a = prever_idade_genero_caffe(roi, age_net, gender_net)
                rotulo = f"{gen} ({conf_g*100:.0f}%) | {idade}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, rotulo, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        dt = time.perf_counter() - t0
        tempos.append(dt)
        fps = 1.0 / max(1e-4, dt)

        cv2.putText(frame, f"OpenCV DNN Caffe | {fps:.1f} FPS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Deteccao de Idade e Genero Caffe", frame)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    fps_medio = len(tempos) / max(1e-4, sum(tempos))
    print(f"\nThroughput medio no video: {fps_medio:.1f} FPS")


def main():
    ensure_dirs()
    age_net, gender_net = obter_modelos_caffe_idade_genero()
    classificar_5_rostos(age_net, gender_net)

    caminho_video = obter_video("face_demographics_walking.mp4")
    executar_pipeline_video(age_net, gender_net, caminho_video)


if __name__ == "__main__":
    main()
