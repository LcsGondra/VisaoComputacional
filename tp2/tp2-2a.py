import cv2
import numpy as np
from pathlib import Path


def criar_rosto_sintetico(largura=640, altura=480):
    cena = np.full((altura, largura, 3), (235, 235, 235), dtype=np.uint8)

    posicoes = [(180, 200, 1.0), (420, 210, 0.9)]
    for cx, cy, escala in posicoes:
        rw, rh = int(90 * escala), int(120 * escala)
        cv2.ellipse(cena, (cx, cy), (rw, rh), 0, 0, 360, (190, 160, 140), -1)
        cv2.ellipse(cena, (cx, cy), (rw, rh), 0, 0, 360, (140, 110, 90), 2)

        olho_esq = (int(cx - 35 * escala), int(cy - 25 * escala))
        olho_dir = (int(cx + 35 * escala), int(cy - 25 * escala))
        cv2.circle(cena, olho_esq, int(12 * escala), (255, 255, 255), -1)
        cv2.circle(cena, olho_dir, int(12 * escala), (255, 255, 255), -1)
        cv2.circle(cena, olho_esq, int(6 * escala), (40, 30, 20), -1)
        cv2.circle(cena, olho_dir, int(6 * escala), (40, 30, 20), -1)

        cv2.line(cena, (cx, int(cy - 10 * escala)), (cx, int(cy + 20 * escala)), (120, 90, 80), 3)
        cv2.ellipse(cena, (cx, int(cy + 55 * escala)), (int(30 * escala), int(14 * escala)), 0, 0, 180, (60, 50, 180), -1)

    cv2.rectangle(cena, (40, 50), (120, 130), (120, 150, 140), -1)
    cv2.circle(cena, (80, 80), 20, (50, 50, 50), -1)
    cv2.circle(cena, (100, 80), 8, (20, 20, 20), -1)

    ruido = np.random.normal(0, 4, cena.shape).astype(np.int16)
    cena = np.clip(cena.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    return cena


def main():
    saida_dir = Path("dados/faces_48x48")
    saida_dir.mkdir(parents=True, exist_ok=True)
    saida_img_dir = Path("dados/saidas")
    saida_img_dir.mkdir(parents=True, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Nao foi possivel carregar haarcascade_frontalface_default.xml")

    imagem_teste = criar_rosto_sintetico()
    cinza = cv2.cvtColor(imagem_teste, cv2.COLOR_BGR2GRAY)
    cinza = cv2.equalizeHist(cinza)

    configuracoes = [
        {"nome": "Configuracao 1 (Padrao Rigido)", "scaleFactor": 1.1, "minNeighbors": 5, "minSize": (40, 40)},
        {"nome": "Configuracao 2 (Alta Sensibilidade)", "scaleFactor": 1.05, "minNeighbors": 2, "minSize": (30, 30)},
    ]

    total_faces_reais = 2
    salvo_count = 0

    print("=== AVALIACAO DE DETECCAO FACIAL COM HAAR CASCADE (TP2 - 2A) ===")

    for idx, cfg in enumerate(configuracoes, 1):
        faces = detector.detectMultiScale(
            cinza,
            scaleFactor=cfg["scaleFactor"],
            minNeighbors=cfg["minNeighbors"],
            minSize=cfg["minSize"],
        )

        anotada = imagem_teste.copy()
        for x, y, w, h in faces:
            cv2.rectangle(anotada, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                anotada,
                f"Face {w}x{h}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            roi_face = imagem_teste[y : y + h, x : x + w]
            if roi_face.size > 0:
                roi_48x48 = cv2.resize(roi_face, (48, 48), interpolation=cv2.INTER_AREA)
                caminho_salvo = saida_dir / f"face_cfg{idx}_{salvo_count:03d}.png"
                cv2.imwrite(str(caminho_salvo), roi_48x48)
                salvo_count += 1

        num_detectadas = len(faces)
        verdadeiros_positivos = min(num_detectadas, total_faces_reais)
        falsos_positivos = max(0, num_detectadas - total_faces_reais)
        taxa_deteccao = (verdadeiros_positivos / total_faces_reais) * 100.0

        cv2.imwrite(str(saida_img_dir / f"tp2_2a_deteccao_cfg{idx}.png"), anotada)

        print(f"\n[{cfg['nome']}]")
        print(f"  - Parametros: scaleFactor={cfg['scaleFactor']}, minNeighbors={cfg['minNeighbors']}")
        print(f"  - Deteccoes totais: {num_detectadas}")
        print(f"  - Verdadeiros Positivos: {verdadeiros_positivos}/{total_faces_reais}")
        print(f"  - Falsos Positivos Visiveis: {falsos_positivos}")
        print(f"  - Taxa Estimada de Deteccao (Recall): {taxa_deteccao:.1f}%")

    print("\n=== JUSTIFICATIVA DO TRADE-OFF: SENSIBILIDADE VS ESPECIFICIDADE ===")
    print("1. scaleFactor menor (ex: 1.05) e minNeighbors baixo (ex: 2):")
    print("   Aumenta a sensibilidade (Recall alto), detectando faces parciais ou menores,")
    print("   mas eleva os falsos positivos (Especificidade baixa) em padroes de textura.")
    print("2. scaleFactor maior (ex: 1.1 - 1.3) e minNeighbors alto (ex: 5 - 8):")
    print("   Aumenta a especificidade (Precisao alta), eliminando quase todos os falsos positivos,")
    print("   porem pode omitir rostos com leve oclusao ou inclinacao (sensibilidade reduzida).")
    print(f"Total de faces 48x48 salvas em disco: {salvo_count} em '{saida_dir}'")

    cv2.imshow("Deteccao Facial Haar", anotada)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
