import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import time
from utils import imshow_keep_aspect_ratio, obter_diretorios


def main():
    dirs = obter_diretorios(__file__)
    saida_dir = dirs["faces"]
    saida_img_dir = dirs["saidas"]

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Nao foi possivel carregar haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nenhuma camera/webcam detectada para execucao do Exercicio 2A.")

    print("Webcam conectada com sucesso para deteccao facial em tempo real.")

    configuracoes = {
        "1": {
            "nome": "Configuracao 1 (Alta Especificidade / Padrao)",
            "scaleFactor": 1.15,
            "minNeighbors": 5,
            "minSize": (45, 45),
            "cor": (0, 255, 0),
        },
        "2": {
            "nome": "Configuracao 2 (Alta Sensibilidade / Menos Restritiva)",
            "scaleFactor": 1.05,
            "minNeighbors": 2,
            "minSize": (30, 30),
            "cor": (0, 200, 255),
        },
    }

    cfg_ativa_chave = "1"
    estatisticas = {
        "1": {"frames": 0, "faces_detectadas": 0, "falsos_positivos": 0},
        "2": {"frames": 0, "faces_detectadas": 0, "falsos_positivos": 0},
    }

    salvo_count = 0
    janela_feed = "Deteccao Facial Haar Cascade em Tempo Real (TP2 Ex2A)"
    cv2.namedWindow(janela_feed, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela_feed, 900, 560)

    print("=== INICIANDO DETECCAO FACIAL COM HAAR CASCADE (TP2 - 2A) ===")
    print("Controles:")
    print("  - [1]: Configuracao 1 (Alta Especificidade: scaleFactor=1.15, minNeighbors=5)")
    print("  - [2]: Configuracao 2 (Alta Sensibilidade: scaleFactor=1.05, minNeighbors=2)")
    print("  - [S]: Forcar salvamento da face detectada em 48x48 px")
    print("  - [Q]: Encerrar e exibir relatorio de metricas")

    frame_idx = 0
    ultima_face_48x48 = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cinza = cv2.equalizeHist(cinza)

        cfg = configuracoes[cfg_ativa_chave]
        t0 = time.perf_counter()
        faces = detector.detectMultiScale(
            cinza,
            scaleFactor=cfg["scaleFactor"],
            minNeighbors=cfg["minNeighbors"],
            minSize=cfg["minSize"],
        )
        tempo_ms = (time.perf_counter() - t0) * 1000.0

        estatisticas[cfg_ativa_chave]["frames"] += 1
        estatisticas[cfg_ativa_chave]["faces_detectadas"] += len(faces)
        estatisticas[cfg_ativa_chave]["falsos_positivos"] += max(0, len(faces) - 1)

        frame_saida = frame.copy()

        for idx_f, (x, y, fw, fh) in enumerate(faces):
            cv2.rectangle(frame_saida, (x, y), (x + fw, y + fh), cfg["cor"], 2)
            cv2.putText(
                frame_saida,
                f"Face {fw}x{fh}",
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cfg["cor"],
                2,
                cv2.LINE_AA,
            )

            roi_face = frame[max(0, y) : min(h, y + fh), max(0, x) : min(w, x + fw)]
            if roi_face.size > 0:
                face_48 = cv2.resize(roi_face, (48, 48), interpolation=cv2.INTER_AREA)
                ultima_face_48x48 = face_48
                if frame_idx % 15 == 0 or salvo_count < 10:
                    caminho_salvo = saida_dir / f"face_cfg{cfg_ativa_chave}_{salvo_count:04d}.png"
                    cv2.imwrite(str(caminho_salvo), face_48)
                    salvo_count += 1

        if ultima_face_48x48 is not None:
            thumb = cv2.resize(ultima_face_48x48, (80, 80), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(frame_saida, (w - 95, h - 95), (w - 10, h - 10), (0, 0, 0), -1)
            frame_saida[h - 90 : h - 10, w - 90 : w - 10] = thumb
            cv2.putText(
                frame_saida,
                "48x48",
                (w - 85, h - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame_saida, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.putText(
            frame_saida,
            f"{cfg['nome']} | Faces: {len(faces)} | Latencia: {tempo_ms:.1f} ms",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            cfg["cor"],
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_saida,
            "Teclas [1] Config 1 (Especificidade) | [2] Config 2 (Sensibilidade) | [S] Salvar | [Q] Sair",
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        if frame_idx % 20 == 0:
            print(
                f"Frame: {frame_idx:04d} | {cfg['nome']} | Faces Detectadas: {len(faces)} | Latencia: {tempo_ms:.2f} ms"
            )

        imshow_keep_aspect_ratio(janela_feed, frame_saida)

        if frame_idx == 30:
            cv2.imwrite(str(saida_img_dir / "tp2_2a_deteccao_tempo_real.png"), frame_saida)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("1"):
            cfg_ativa_chave = "1"
            print("-> Ativada: Configuracao 1 (scaleFactor=1.15, minNeighbors=5)")
        elif key == ord("2"):
            cfg_ativa_chave = "2"
            print("-> Ativada: Configuracao 2 (scaleFactor=1.05, minNeighbors=2)")
        elif key == ord("s") or key == ord("S"):
            if ultima_face_48x48 is not None:
                caminho_manual = saida_dir / f"face_manual_{salvo_count:04d}.png"
                cv2.imwrite(str(caminho_manual), ultima_face_48x48)
                salvo_count += 1
                print(f"-> Face 48x48 salva com sucesso em: {caminho_manual}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== AVALIACAO FINAL DE DETECCAO FACIAL COM HAAR CASCADE (TP2 - 2A) ===")
    for k, v in configuracoes.items():
        st = estatisticas[k]
        total_f = max(1, st["frames"])
        media_faces = st["faces_detectadas"] / float(total_f)
        taxa_deteccao = min(100.0, media_faces * 100.0)
        falsos_positivos = st["falsos_positivos"]
        print(f"\n[{v['nome']}]")
        print(f"  - Parametros: scaleFactor={v['scaleFactor']}, minNeighbors={v['minNeighbors']}, minSize={v['minSize']}")
        print(f"  - Total de frames avaliados: {st['frames']}")
        print(f"  - Numero de Falsos Positivos Visiveis: {falsos_positivos}")
        print(f"  - Taxa Estimada de Deteccao (Recall): {taxa_deteccao:.1f}%")

    # Justificativa do Trade-off entre Sensibilidade e Especificidade:
    # 1. Configuracao 2 (scaleFactor=1.05, minNeighbors=2 - Alta Sensibilidade):
    #    A reducao de scaleFactor para 1.05 realiza uma busca mais densa na piramide de escalas,
    #    e minNeighbors=2 exige menos retangulos coincidentes para aprovar a deteccao.
    #    Isso maximiza a taxa de deteccao (Recall), identificando rostos mesmo com oclusao parcial,
    #    iluminacao fraca ou angulacao, porem ao custo de elevar substancialmente os falsos positivos.
    # 2. Configuracao 1 (scaleFactor=1.15, minNeighbors=5 - Alta Especificidade):
    #    Ao exigir minNeighbors=5, o classificador demanda maior consenso de caracteristicas de Haar,
    #    filtrando quase 100% dos falsos positivos no fundo da cena. Em contrapartida, torna o detector
    #    mais restritivo, podendo perder rostos em condicoes subotimas de contraste ou distancia.

    print(f"\nTotal de faces 48x48 salvas em disco: {salvo_count} imagens no diretorio '{saida_dir}'")


if __name__ == "__main__":
    main()
