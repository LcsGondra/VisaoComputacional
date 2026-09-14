import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    obter_video,
    obter_modelos_caffe_idade_genero,
    prever_idade_genero_caffe,
    salvar_figura,
    SAIDAS_DIR,
)


def executar_pipeline_video(age_net, gender_net, caminho_video):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {caminho_video}")

    tempos = []
    amostras_5_rostos = []
    passagens = {
        "p1": (65, 115, None),
        "p2_esq": (270, 310, "esq"),
        "p2_dir": (270, 310, "dir"),
        "p3": (470, 510, None),
        "p4": (660, 695, None),
    }
    coletados = set()
    frame_count = 0

    print("Iniciando feed de video com deteccao facial e classificacao...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t0 = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60)
        )

        for x, y, w, h in faces:
            if y > frame.shape[0] * 0.45:
                continue

            pad = 12
            y1 = max(0, y - pad)
            y2 = min(frame.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(frame.shape[1], x + w + pad)
            roi = frame[y1:y2, x1:x2]

            if roi.shape[0] > 40 and roi.shape[1] > 40:
                gen, conf_g, idade, conf_a = prever_idade_genero_caffe(
                    roi, age_net, gender_net
                )
                rotulo = f"{gen} ({conf_g*100:.0f}%) | {idade}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    rotulo,
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

                for k, (f_ini, f_fim, lado) in passagens.items():
                    if k not in coletados and f_ini <= frame_count <= f_fim:
                        if lado == "esq" and x > frame.shape[1] * 0.5:
                            continue
                        if lado == "dir" and x <= frame.shape[1] * 0.5:
                            continue
                        roi_anotada = cv2.resize(roi.copy(), (200, 200))
                        cv2.putText(
                            roi_anotada,
                            f"{gen} ({conf_g*100:.0f}%)",
                            (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 0),
                            1,
                        )
                        cv2.putText(
                            roi_anotada,
                            f"{idade} ({conf_a*100:.0f}%)",
                            (8, 42),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            1,
                        )
                        amostras_5_rostos.append(
                            {
                                "id": len(amostras_5_rostos) + 1,
                                "frame": frame_count,
                                "genero": gen,
                                "conf_g": conf_g,
                                "idade": idade,
                                "conf_a": conf_a,
                                "img": roi_anotada,
                            }
                        )
                        coletados.add(k)
                        break

        dt = time.perf_counter() - t0
        tempos.append(dt)
        fps = 1.0 / max(1e-4, dt)

        cv2.putText(
            frame,
            f"OpenCV DNN Caffe | {fps:.1f} FPS",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Deteccao de Idade e Genero Caffe", frame)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()

    fps_medio = len(tempos) / max(1e-4, sum(tempos))
    print(f"\nThroughput medio no video: {fps_medio:.1f} FPS")

    if amostras_5_rostos:
        print("\nRelatorio de Classificacao em 5 Rostos do Video (Caffe DNN):")
        print(
            f"{'Rosto':<26} | {'Genero':<12} | {'Conf. Gen.':<10} | {'Idade':<12} | {'Conf. Idade'}"
        )
        print("-" * 77)
        for a in amostras_5_rostos:
            nome_r = f"Individuo #{a['id']:02d} (Frame {a['frame']})"
            conf_g_str = f"{a['conf_g']*100:.1f}%"
            conf_a_str = f"{a['conf_a']*100:.1f}%"
            print(
                f"{nome_r:<26} | {a['genero']:<12} | {conf_g_str:<10} | {a['idade']:<12} | {conf_a_str}"
            )
        print("-" * 77)

        fig, axs = plt.subplots(1, len(amostras_5_rostos), figsize=(15, 3.5))
        if len(amostras_5_rostos) == 1:
            axs = [axs]
        for i, a in enumerate(amostras_5_rostos):
            axs[i].imshow(cv2.cvtColor(a["img"], cv2.COLOR_BGR2RGB))
            axs[i].set_title(
                f"Individuo #{a['id']} (Frame {a['frame']})", fontweight="bold"
            )
            axs[i].axis("off")
        salvar_figura(SAIDAS_DIR / "tp3_4a_5_rostos_caffe.png")


def main():
    ensure_dirs()
    age_net, gender_net = obter_modelos_caffe_idade_genero()
    caminho_video = obter_video("face_demographics_walking.mp4")
    executar_pipeline_video(age_net, gender_net, caminho_video)


if __name__ == "__main__":
    main()

# ==============================================================================
# Relatório de Validação e Acertos (5 Rostos Distintos do Vídeo):
# 1. Indivíduo #01 (Frame 75):  Masculino (99.8%) | Idade: (8-12) (59.3%)
#    Acerto no gênero; subestimou faixa etária devido à iluminação/óculos.
# 2. Indivíduo #02 (Frame 279): Masculino (93.9%) | Idade: (38-43) (56.6%)
#    Acerto no gênero e faixa etária condizente com adulto.
# 3. Indivíduo #03 (Frame 280): Feminino (99.8%)  | Idade: (15-20) (79.1%)
#    Acerto no gênero e falha na faixa etária, indicando jovem quando parece mulher adulta.
# 4. Indivíduo #04 (Frame 478): Masculino (83.3%) | Idade: (25-32) (89.4%)
#    Falha no gênero (mulher com cabelo preso/óculos); acerto na idade.
# 5. Indivíduo #05 (Frame 668): Feminino (100.0%) | Idade: (25-32) (89.5%)
#    Acerto no gênero e faixa etária para uma mulher adulta (25-32 anos).
# ==============================================================================
