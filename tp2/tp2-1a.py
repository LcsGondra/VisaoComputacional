import cv2
import numpy as np
from pathlib import Path


def gerar_par_estereo_sintetico(largura=640, altura=360):
    rng = np.random.default_rng(42)
    fundo = rng.integers(60, 190, size=(altura, largura, 1), dtype=np.uint8)
    esquerda = np.repeat(fundo, 3, axis=2)
    direita = esquerda.copy()

    objetos = [
        ((30, 200, 240), (140, 100), "PERTO", (90, 180), 48),
        ((220, 110, 30), (120, 90), "MEIO", (280, 120), 28),
        ((60, 190, 70), (100, 80), "LONGE", (470, 60), 12),
    ]

    for cor, dim, txt, pos, disp in objetos:
        w, h = dim
        bloco = rng.integers(0, 40, size=(h, w, 3), dtype=np.uint8)
        bloco = cv2.add(bloco, np.full_like(bloco, cor))
        cv2.rectangle(bloco, (2, 2), (w - 3, h - 3), (255, 255, 255), 2)
        cv2.putText(bloco, txt, (10, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        x, y = pos

        esquerda[y : y + h, x : x + w] = bloco
        direita[y : y + h, (x - disp) : (x - disp + w)] = bloco

    return esquerda, direita


def main():
    saida_dir = Path("dados/saidas")
    saida_dir.mkdir(parents=True, exist_ok=True)

    img_esq, img_dir = gerar_par_estereo_sintetico()
    cv2.imwrite(str(saida_dir / "tp2_1a_esquerda.png"), img_esq)
    cv2.imwrite(str(saida_dir / "tp2_1a_direita.png"), img_dir)

    cinza_esq = cv2.cvtColor(img_esq, cv2.COLOR_BGR2GRAY)
    cinza_dir = cv2.cvtColor(img_dir, cv2.COLOR_BGR2GRAY)

    num_disp = 64
    block_size = 5

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 1 * block_size**2,
        P2=32 * 1 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparidade = sgbm.compute(cinza_esq, cinza_dir).astype(np.float32) / 16.0
    mascara_valida = disparidade > 0

    disp_norm = np.zeros_like(disparidade, dtype=np.float32)
    disp_vis = np.zeros_like(cinza_esq, dtype=np.uint8)

    if np.any(mascara_valida):
        min_val = float(np.min(disparidade[mascara_valida]))
        max_val = float(np.max(disparidade[mascara_valida]))
        disp_norm[mascara_valida] = (disparidade[mascara_valida] - min_val) / max(max_val - min_val, 1e-6)
        disp_vis[mascara_valida] = (disp_norm[mascara_valida] * 255.0).astype(np.uint8)

    disp_jet = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    disp_jet[~mascara_valida] = (0, 0, 0)

    cv2.imwrite(str(saida_dir / "tp2_1a_disparidade_jet.png"), disp_jet)

    valid_indices = np.argwhere(mascara_valida)
    if len(valid_indices) > 0:
        valid_values = disparidade[mascara_valida]
        idx_max = np.argmax(valid_values)
        idx_min = np.argmin(valid_values)

        coord_perto_y, coord_perto_x = valid_indices[idx_max]
        val_perto = valid_values[idx_max]
        val_norm_perto = disp_norm[coord_perto_y, coord_perto_x]

        coord_longe_y, coord_longe_x = valid_indices[idx_min]
        val_longe = valid_values[idx_min]
        val_norm_longe = disp_norm[coord_longe_y, coord_longe_x]

        print("=== RESULTADO ESTIMATIVA DE PROFUNDIDADE (TP2 - 1A) ===")
        print(f"Dimensao das imagens: {img_esq.shape[1]}x{img_esq.shape[0]} px")
        print(f"Pixels com disparidade valida: {100.0 * np.mean(mascara_valida):.2f}%")
        print(f"Pixel mais proximo (Maior disparidade):")
        print(f"  - Coordenadas (X, Y): ({coord_perto_x}, {coord_perto_y})")
        print(f"  - Disparidade bruta: {val_perto:.2f} px | Normalizada: {val_norm_perto:.4f}")
        print(f"Pixel mais distante (Menor disparidade valida):")
        print(f"  - Coordenadas (X, Y): ({coord_longe_x}, {coord_longe_y})")
        print(f"  - Disparidade bruta: {val_longe:.2f} px | Normalizada: {val_norm_longe:.4f}")

    print("\n=== ANALISE TECNICA: COMPORTAMENTO EM DRONE EMBARCADO ===")
    print("1. Baseline e Distancia Minima/Maxima:")
    print("   Em drones, a distancia entre cameras (baseline B) costuma ser curta (ex: 5 a 15 cm)")
    print("   devido ao peso e espaco. Isso limita o alcance util de profundidade (Z = f * B / d).")
    print("2. Vibracao e Calibracao Epipolar:")
    print("   As vibracoes dos motores desalinham os eixos opticos, quebrando a retificacao")
    print("   epipolar e introduzindo erros na busca estereo.")
    print("3. Superficies Homogeneas e Iluminacao Solar:")
    print("   Areas lisas (grama homogenea, asfalto, ceu) carecem de textura para matching de blocos,")
    print("   exigindo filtragem temporal ou fusao com sensores ultrassom / LiDAR.")
    print("4. Custo Computacional:")
    print("   O SGBM exige processamento expressivo, exigindo aceleracao em hardware dedicado")
    print("   ou uso de disparidade esparsa para controle de voo em tempo real (>30 FPS).")

    cv2.imshow("Par Estereo (Esquerda | Direita)", np.hstack([img_esq, img_dir]))
    cv2.imshow("Mapa de Disparidade (COLORMAP_JET)", disp_jet)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
