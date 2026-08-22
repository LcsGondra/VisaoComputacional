import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio, obter_diretorios


def colar(destino, objeto, x, y):
    h, w = objeto.shape[:2]
    destino[y : y + h, x : x + w] = objeto


def gerar_par_estereo(largura=640, altura=360):
    rng = np.random.default_rng(12)
    textura = rng.integers(70, 180, size=(altura, largura, 1), dtype=np.uint8)
    esquerda = np.repeat(textura, 3, axis=2)
    direita = esquerda.copy()

    objetos = []
    for cor, tamanho, texto in [
        ((30, 200, 240), (150, 110), "PERTO"),
        ((220, 110, 30), (130, 95), "MEIO"),
        ((60, 190, 70), (105, 80), "LONGE"),
    ]:
        ow, oh = tamanho
        obj = rng.integers(0, 45, size=(oh, ow, 3), dtype=np.uint8)
        obj = cv2.add(obj, np.full_like(obj, cor))
        cv2.rectangle(obj, (2, 2), (ow - 3, oh - 3), (250, 250, 250), 3)
        cv2.putText(
            obj, texto, (10, oh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        objetos.append(obj)

    posicoes = [(90, 190, 48), (300, 120, 28), (500, 55, 12)]
    for obj, (x, y, disparidade) in zip(objetos, posicoes):
        colar(esquerda, obj, x, y)
        colar(direita, obj, x - disparidade, y)

    return esquerda, direita


def main():
    dirs = obter_diretorios(__file__)
    saida_dir = dirs["saidas"]

    esquerda, direita = gerar_par_estereo()
    cv2.imwrite(str(saida_dir / "tp2_1a_esquerda.png"), esquerda)
    cv2.imwrite(str(saida_dir / "tp2_1a_direita.png"), direita)

    cinza_esq = cv2.cvtColor(esquerda, cv2.COLOR_BGR2GRAY)
    cinza_dir = cv2.cvtColor(direita, cv2.COLOR_BGR2GRAY)

    num_disparidades = 96
    bloco = 5
    canais = 1

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparidades,
        blockSize=bloco,
        P1=8 * canais * bloco**2,
        P2=32 * canais * bloco**2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparidade = sgbm.compute(cinza_esq, cinza_dir).astype(np.float32) / 16.0
    valida = disparidade > 0.5

    disp_norm = np.zeros_like(disparidade, dtype=np.float32)
    disp_vis = np.zeros_like(cinza_esq, dtype=np.uint8)

    if np.any(valida):
        minimo, maximo = np.percentile(disparidade[valida], [2, 98])
        disp_norm[valida] = (disparidade[valida] - minimo) / max(maximo - minimo, 1e-6)
        disp_norm = np.clip(disp_norm, 0.0, 1.0)
        disp_vis = (disp_norm * 255.0).astype(np.uint8)

    disp_jet = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    disp_jet[~valida] = 0
    cv2.imwrite(str(saida_dir / "tp2_1a_disparidade_jet.png"), disp_jet)

    focal_px = 500.0
    baseline_m = 0.12
    max_m = 10.0

    profundidade = np.full(disparidade.shape, np.nan, dtype=np.float32)
    profundidade[valida] = focal_px * baseline_m / disparidade[valida]

    escala_prof = np.zeros(disparidade.shape, dtype=np.uint8)
    escala_prof[valida] = np.clip(255 * (1.0 - profundidade[valida] / max_m), 0, 255).astype(np.uint8)
    prof_jet = cv2.applyColorMap(escala_prof, cv2.COLORMAP_JET)
    prof_jet[~valida] = 0
    cv2.imwrite(str(saida_dir / "tp2_1a_profundidade_jet.png"), prof_jet)

    indices_validos = np.argwhere(valida)
    if len(indices_validos) > 0:
        valores_disp = disparidade[valida]
        idx_mais_perto = np.argmax(valores_disp)
        idx_mais_longe = np.argmin(valores_disp)

        coord_perto_y, coord_perto_x = indices_validos[idx_mais_perto]
        disp_perto = valores_disp[idx_mais_perto]
        norm_perto = disp_norm[coord_perto_y, coord_perto_x]
        prof_perto = profundidade[coord_perto_y, coord_perto_x]

        coord_longe_y, coord_longe_x = indices_validos[idx_mais_longe]
        disp_longe = valores_disp[idx_mais_longe]
        norm_longe = disp_norm[coord_longe_y, coord_longe_x]
        prof_longe = profundidade[coord_longe_y, coord_longe_x]

        p = np.nanpercentile(profundidade, [5, 50, 95])

        print("=== RESULTADO ESTIMATIVA DE PROFUNDIDADE (TP2 - 1A) ===")
        print(f"Dimensao das imagens: {esquerda.shape[1]}x{esquerda.shape[0]} px")
        print(f"Pixels com disparidade valida: {100.0 * np.mean(valida):.2f}%")
        print(f"Faixa valida de disparidade (p5-p95): [{np.percentile(disparidade[valida], 5):.2f}, {np.percentile(disparidade[valida], 95):.2f}] px")
        print(f"Profundidade metrica (Z = f*B/d): p5={p[0]:.2f}m, p50={p[1]:.2f}m, p95={p[2]:.2f}m (f={focal_px:.0f}px, B={baseline_m*100:.0f}cm)")
        print(f"\nPixel mais proximo (Maior disparidade / Menor profundidade):")
        print(f"  - Coordenadas (X, Y): ({coord_perto_x}, {coord_perto_y})")
        print(f"  - Disparidade bruta: {disp_perto:.2f} px | Normalizada: {norm_perto:.4f} | Profundidade: {prof_perto:.2f} m")
        print(f"Pixel mais distante (Menor disparidade / Maior profundidade):")
        print(f"  - Coordenadas (X, Y): ({coord_longe_x}, {coord_longe_y})")
        print(f"  - Disparidade bruta: {disp_longe:.2f} px | Normalizada: {norm_longe:.4f} | Profundidade: {prof_longe:.2f} m")

    # Analise tecnica: Comportamento da estimativa em camera embarcada em drone
    # 1. Baseline e Alcance Util:
    #    Em drones, o baseline reduzido (ex: 12 cm) limita o alcance util de profundidade (Z = f * B / d).
    #    Para distancias superiores a 10m, a disparidade resultante e subpixel (< 1px), degradando a precisao.
    # 2. Vibracao e Quebra da Retificacao Epipolar:
    #    A vibracao dos motores causa microdesalinhamento entre os eixos opticos das cameras,
    #    invalidando a busca epipolar horizontal pura do SGBM e exigindo retificacao dinâmica ou gimbals.
    # 3. Textura e Iluminacao Solar Direta:
    #    Superficies homogeneas (ceu, asfalto, gramado) geram regioes sem correspondencia valida,
    #    tornando essencial a fusao sensorial com LiDAR, sensores ultrassonicos ou fluxo optico temporal.
    # 4. Custo Computacional e Latencia:
    #    O algoritmo SGBM exige alto poder de processamento, demandando aceleracao em hardware embarcado
    #    dedicado (FPGA/GPU Jetson) ou uso de disparidade esparsa para controle de voo em tempo real (>30 FPS).

    print("\n[Pressione 'q' ou 'Q' em qualquer janela para encerrar]")

    win_stereo = "Par Estereo (Esquerda | Direita) - TP2 Ex1A"
    win_disp = "Mapa de Disparidade (COLORMAP_JET) - TP2 Ex1A"

    cv2.namedWindow(win_stereo, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_disp, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_stereo, 960, 320)
    cv2.resizeWindow(win_disp, 640, 360)

    par_stereo = np.hstack([esquerda, direita])

    while True:
        imshow_keep_aspect_ratio(win_stereo, par_stereo)
        imshow_keep_aspect_ratio(win_disp, disp_jet)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
