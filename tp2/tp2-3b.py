import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio, obter_diretorios


def carregar_ou_gerar_par(saida_dir):
    caminho_a = saida_dir / "tp2_3_referencia.png"
    caminho_b = saida_dir / "tp2_3_transformada.png"

    if caminho_a.exists() and caminho_b.exists():
        img_a = cv2.imread(str(caminho_a))
        img_b = cv2.imread(str(caminho_b))
        if img_a is not None and img_b is not None:
            return img_a, img_b

    largura, altura = 800, 600
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

    centro = (largura / 2.0, altura / 2.0)
    matriz_rot = cv2.getRotationMatrix2D(centro, 15.0, 0.92)
    transf = cv2.warpAffine(ref, matriz_rot, (largura, altura), borderValue=(220, 220, 220))

    cv2.imwrite(str(caminho_a), ref)
    cv2.imwrite(str(caminho_b), transf)
    return ref, transf


def main():
    dirs = obter_diretorios(__file__)
    saida_dir = dirs["saidas"]

    img_a, img_b = carregar_ou_gerar_par(saida_dir)
    cinza_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    cinza_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=2000)
    kp_a, desc_a = sift.detectAndCompute(cinza_a, None)
    kp_b, desc_b = sift.detectAndCompute(cinza_b, None)

    bf_cross = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_bf = bf_cross.match(desc_a, desc_b)
    matches_bf = sorted(matches_bf, key=lambda m: m.distance)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(desc_a, desc_b, k=2)

    limiar_lowe = 0.75
    bons_matches = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < limiar_lowe * n.distance:
                bons_matches.append(m)

    img_matches = cv2.drawMatches(
        img_a,
        kp_a,
        img_b,
        kp_b,
        bons_matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(saida_dir / "tp2_3b_matches_filtrados.png"), img_matches)

    pontos_a = np.float32([kp_a[m.queryIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
    pontos_b = np.float32([kp_b[m.trainIdx].pt for m in bons_matches]).reshape(-1, 1, 2)

    h_mat, mascara_ransac = cv2.findHomography(pontos_a, pontos_b, cv2.RANSAC, 4.0)

    inliers_total = int(np.sum(mascara_ransac)) if mascara_ransac is not None else 0
    taxa_inliers = (inliers_total / max(len(bons_matches), 1)) * 100.0

    h, w = img_b.shape[:2]
    img_alinhada = cv2.warpPerspective(img_a, h_mat, (w, h), borderValue=(0, 0, 0))
    cv2.imwrite(str(saida_dir / "tp2_3b_imagem_alinhada_homografia.png"), img_alinhada)

    sobreposicao = cv2.addWeighted(img_b, 0.5, img_alinhada, 0.5, 0)
    cv2.putText(sobreposicao, f"Homografia RANSAC: {inliers_total} Inliers ({taxa_inliers:.1f}%)", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.imwrite(str(saida_dir / "tp2_3b_sobreposicao_alinhada.png"), sobreposicao)

    print("=== MATCHING E ESTIMATIVA DE HOMOGRAFIA COM RANSAC (TP2 - 3B) ===")
    print(f"Keypoints Imagem A: {len(kp_a)} | Imagem B: {len(kp_b)}")
    print(f"Total de Matches BFMatcher (cross-check): {len(matches_bf)}")
    print(f"Total de Matches FLANN (Lowe ratio < {limiar_lowe}): {len(bons_matches)}")
    print(f"Inliers RANSAC confirmados: {inliers_total}/{len(bons_matches)} ({taxa_inliers:.2f}%)")
    print(f"Matriz de Homografia estimada (3x3):")
    print(np.array2string(h_mat, precision=4, suppress_small=True))

    # Relevancia de Homografia e RANSAC para Localizacao Visual e SLAM em Robotica:
    # 1. Estimativa de Pose e Odometria Visual (VO):
    #    O alinhamento geometrico projetivo (Homografia H ou Matriz Essencial E) entre pares de quadros
    #    permite decompor a rotacao (R) e translacao (t) da camera do robo/drone em relacao ao ambiente.
    # 2. Rejeicao Robusta de Outliers com RANSAC:
    #    Em cenas reais, alteracoes de iluminacao, repeticao de texturas ou objetos em movimento geram
    #    falsas correspondencias. O RANSAC amostra hipoteses minimas e seleciona o modelo com maior consenso,
    #    garantindo que apenas inliers geometricamente consistentes guiem a navegacao.
    # 3. Deteccao de Fechamento de Ciclo (Loop Closure) e Mapeamento Global:
    #    Ao retornar a uma regiao mapeada anteriormente, o casamento de pontos-chave com verificacao por
    #    homografia valida o reencontro do local e corrige a deriva acumulada (drift) do robo.

    print("\n[Pressione 'q' ou 'Q' para fechar as janelas]")

    win_matches = "Matches Filtrados (FLANN + Lowe Ratio < 0.75) - TP2 Ex3B"
    win_homografia = "Sobreposicao Alinhada por Homografia RANSAC - TP2 Ex3B"

    cv2.namedWindow(win_matches, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_homografia, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_matches, 1200, 500)
    cv2.resizeWindow(win_homografia, 800, 600)

    while True:
        imshow_keep_aspect_ratio(win_matches, img_matches)
        imshow_keep_aspect_ratio(win_homografia, sobreposicao)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
