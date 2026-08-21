import cv2
import numpy as np
import time
from pathlib import Path


def gerar_par_imagens(largura=800, altura=600):
    rng = np.random.default_rng(101)
    ref = np.full((altura, largura, 3), (240, 240, 240), dtype=np.uint8)

    cv2.rectangle(ref, (40, 40), (largura - 40, altura - 40), (30, 30, 30), 4)

    tam_celula = 40
    for r in range(6):
        for c in range(8):
            cor = (35, 35, 35) if (r + c) % 2 == 0 else (235, 235, 235)
            x1, y1 = 80 + c * tam_celula, 120 + r * tam_celula
            cv2.rectangle(ref, (x1, y1), (x1 + tam_celula, y1 + tam_celula), cor, -1)

    cv2.circle(ref, (550, 200), 70, (40, 180, 240), -1)
    cv2.circle(ref, (550, 200), 40, (20, 20, 20), 4)
    cv2.line(ref, (500, 200), (600, 200), (20, 20, 20), 4)
    cv2.line(ref, (550, 150), (550, 250), (20, 20, 20), 4)

    triangulo = np.array([[500, 400], [680, 430], [580, 520]], np.int32)
    cv2.fillPoly(ref, [triangulo], (50, 190, 80))
    cv2.polylines(ref, [triangulo], True, (20, 20, 20), 4)

    cv2.putText(ref, "DR2 ROBOTICS VISION", (120, 90), cv2.FONT_HERSHEY_DUPLEX, 1.1, (20, 20, 20), 2)
    cv2.putText(ref, "SIFT / ORB / AKAZE", (100, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

    for px, py in rng.integers([60, 60], [largura - 60, altura - 60], size=(60, 2)):
        if not (70 < px < 420 and 110 < py < 380):
            cv2.drawMarker(ref, (int(px), int(py)), (180, 40, 30), cv2.MARKER_TILTED_CROSS, 8, 2)

    centro = (largura / 2.0, altura / 2.0)
    matriz_rot = cv2.getRotationMatrix2D(centro, 15.0, 0.92)
    transformada = cv2.warpAffine(ref, matriz_rot, (largura, altura), borderValue=(220, 220, 220))

    gradiente = np.linspace(0.75, 1.15, largura, dtype=np.float32)[None, :, None]
    transformada = np.clip(transformada.astype(np.float32) * gradiente + 10, 0, 255).astype(np.uint8)

    ruido = rng.normal(0, 3.0, transformada.shape).astype(np.float32)
    transformada = np.clip(transformada.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    return ref, transformada


def criar_detector(metodo_nome):
    if metodo_nome == "SIFT":
        return cv2.SIFT_create(nfeatures=1500)
    if metodo_nome == "ORB":
        return cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
    if metodo_nome == "AKAZE":
        if hasattr(cv2, "AKAZE_create"):
            return cv2.AKAZE_create()
        elif hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "AKAZE_create"):
            return cv2.xfeatures2d.AKAZE_create()
    raise ValueError(f"Metodo desconhecido: {metodo_nome}")


def extrair_caracteristicas(metodo_nome, imagem, n_repeticoes=5):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    detector = criar_detector(metodo_nome)

    detector.detectAndCompute(cinza, None)

    tempos = []
    kps, desc = None, None
    for _ in range(n_repeticoes):
        t0 = time.perf_counter()
        kps, desc = detector.detectAndCompute(cinza, None)
        tempos.append((time.perf_counter() - t0) * 1000.0)

    tempo_medio = float(np.mean(tempos))
    return kps, desc, tempo_medio


def main():
    saida_dir = Path("dados/saidas")
    saida_dir.mkdir(parents=True, exist_ok=True)

    img_ref, img_transf = gerar_par_imagens()
    cv2.imwrite(str(saida_dir / "tp2_3_referencia.png"), img_ref)
    cv2.imwrite(str(saida_dir / "tp2_3_transformada.png"), img_transf)

    metodos = ["SIFT", "ORB", "AKAZE"]
    resultados = []

    print("=== EXTRACAO E COMPARACAO DE DESCRITORES (TP2 - 3A) ===")

    for metodo in metodos:
        kp_a, desc_a, t_a = extrair_caracteristicas(metodo, img_ref)
        kp_b, desc_b, t_b = extrair_caracteristicas(metodo, img_transf)

        vis_a = cv2.drawKeypoints(img_ref, kp_a, None, (0, 200, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_b = cv2.drawKeypoints(img_transf, kp_b, None, (0, 200, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        par_vis = np.hstack([vis_a, vis_b])
        caminho_vis = saida_dir / f"tp2_3a_keypoints_{metodo.lower()}.png"
        cv2.imwrite(str(caminho_vis), par_vis)

        dim_desc = desc_a.shape[1] if desc_a is not None else 0
        tipo_desc = str(desc_a.dtype) if desc_a is not None else "None"
        t_total_medio = (t_a + t_b) / 2.0

        resultados.append({
            "metodo": metodo,
            "kp_ref": len(kp_a),
            "kp_transf": len(kp_b),
            "tempo_ms": t_total_medio,
            "dimensao": dim_desc,
            "tipo": tipo_desc,
        })

    print("\n" + "=" * 82)
    print(f"{'Metodo':<10} | {'Keypoints (A)':<14} | {'Keypoints (B)':<14} | {'Tempo Medio (ms)':<18} | {'Vetor/Ponto':<16}")
    print("-" * 82)
    for r in resultados:
        print(f"{r['metodo']:<10} | {r['kp_ref']:<14} | {r['kp_transf']:<14} | {r['tempo_ms']:<18.2f} | {r['dimensao']} ({r['tipo']})")
    print("=" * 82)

    print("\n=== ANALISE COMPARATIVA DOS METODOS ===")
    print("1. SIFT (Scale-Invariant Feature Transform):")
    print("   - Descritor continuo em float32 de 128 dimensoes (512 bytes por ponto).")
    print("   - Alta invariancia a escala, rotacao e variacao de iluminacao; maior custo computacional.")
    print("2. ORB (Oriented FAST and Rotated BRIEF):")
    print("   - Descritor binario de 32 bytes (256 bits) em uint8.")
    print("   - Extremamente rapido e leve para sistemas embarcados em tempo real (drones e robos).")
    print("3. AKAZE (Accelerated-KAZE):")
    print("   - Espaco de escala nao linear (filtragem por difusao) com descritor binario MLDB (61 bytes).")
    print("   - Excelente preservacao de bordas e contornos, superior ao ORB em iluminacao nao linear.")

    cv2.imshow("Keypoints SIFT", cv2.imread(str(saida_dir / "tp2_3a_keypoints_sift.png")))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
