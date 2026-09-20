"""
Exercício 1 — Item A: Calibração Completa de Câmera e Correção de Distorção
Competências: 1.2, 1.3 e 4.1

Este script realiza a calibração geométrica de uma câmera utilizando imagens de um
tabuleiro de xadrez com cantos internos 7x6 em múltiplos ângulos (pitch, yaw, roll) e distâncias.

Significado Físico dos Parâmetros Estimados:
--------------------------------------------
1. Matriz Intrínseca K:
   K = [[fx,  s, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]]
   - fx, fy: Distâncias focais expressas em pixels (focal length). Representam a distância
     entre o centro óptico da lente e o plano do sensor multiplicada pela densidade de pixels por mm.
   - s: Fator de cisalhamento (skew factor). Em sensores digitais modernos com pixels estritamente
     ortogonais, s = 0.
   - cx, cy: Ponto principal (coordenadas do centro óptico projetado no plano da imagem em pixels).
     Tipicamente próximo ao centro geométrico da resolução do sensor (w/2, h/2).

2. Coeficientes de Distorção D (5 parâmetros de Brown-Conrady):
   D = [k1, k2, p1, p2, k3]
   - k1, k2, k3 (Distorção Radial): Decorrente da curvatura esférica da lente, fazendo com que
     raios de luz nas bordas sofram refração diferente dos raios centrais.
     Se k > 0: distorção tipo barril (barrel distortion).
     Se k < 0: distorção tipo almofada (pincushion distortion).
   - p1, p2 (Distorção Tangencial): Ocorre quando o conjunto de lentes ópticas não está perfeitamente
     paralelo ao plano do sensor semicondutor (CCD/CMOS).

Tolerância Aceitável para Aplicações Robóticas e Veículos Autônomos:
-------------------------------------------------------------------
- Aplicações industriais de alta precisão e metrologia: erro < 0.20 pixels.
- Robôs móveis e veículos autônomos (odometria visual e fusão sensorial): erro < 0.50 pixels.
- Limite máximo tolerável em robótica de serviço: erro < 1.00 pixel.
Erros superiores a 1.0 pixel causam deriva severa (drift) no cálculo de pose 3D e odometria visual.
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    gerar_dataset_calibracao,
    salvar_figura,
    CALIB_FILE,
    SAIDAS_DIR,
)


def calibrar_camera(imagens_caminhos, pattern_size=(7, 6), square_size_mm=30.0):
    """
    Detecta cantos do tabuleiro e calcula a calibração intrínseca da câmera.
    """
    cols, rows = pattern_size

    # Coordenadas 3D dos cantos internos no referencial do mundo (Z = 0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    objpoints = []  # Pontos 3D no espaço do mundo
    imgpoints = []  # Pontos 2D no plano da imagem
    valid_images = []
    img_shape = None

    criterio_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"[+] Processando {len(imagens_caminhos)} imagens para calibração de câmera...")
    for idx, caminho in enumerate(imagens_caminhos, start=1):
        img = cv2.imread(str(caminho))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # Detecção dos cantos internos
        ret, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            # Refinamento subpixel para máxima precisão
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criterio_subpix)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            valid_images.append(caminho)
            print(f"    Imagem {idx:02d}/{len(imagens_caminhos):02d}: Cantos detectados com sucesso!")
        else:
            print(f"    Imagem {idx:02d}/{len(imagens_caminhos):02d}: Falha na detecção dos cantos.")

    if len(objpoints) < 10:
        raise RuntimeError(f"Apenas {len(objpoints)} imagens válidas. Mínimo de 10 necessário.")

    # Execução da calibração fotogramétrica
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # Cálculo do erro de reprojeção por imagem e global
    erros_por_imagem = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        erro = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        erros_por_imagem.append(erro)

    erro_medio = float(np.mean(erros_por_imagem))
    return K, dist, rvecs, tvecs, erros_por_imagem, erro_medio, valid_images, img_shape


def main():
    ensure_dirs()
    print("=" * 75)
    print("EXERCÍCIO 1A — CALIBRAÇÃO DE CÂMERA E CORREÇÃO DE DISTORÇÃO")
    print("=" * 75)

    # 1. Obter ou gerar dataset de calibração com 18 poses distintas
    caminhos_calib = gerar_dataset_calibracao(num_imagens=18, pattern_size=(7, 6), square_size=30)

    # 2. Realizar calibração completa
    K, dist, rvecs, tvecs, erros_img, erro_medio, valid_imgs, img_shape = calibrar_camera(
        caminhos_calib, pattern_size=(7, 6), square_size_mm=30.0
    )

    # 3. Salvar parâmetros intrínsecos e extrínsecos para uso nos próximos exercícios
    np.savez(
        CALIB_FILE,
        K=K,
        dist=dist,
        erro_medio=erro_medio,
        img_shape=img_shape,
    )
    print(f"\n[+] Matriz K e coeficientes de distorção salvos em: {CALIB_FILE.name}")

    # 4. Impressão dos parâmetros e métricas no terminal
    print("\n" + "-" * 75)
    print("RESULTADOS DA CALIBRAÇÃO DE CÂMERA:")
    print("-" * 75)
    print("Matriz Intrínseca K [3x3]:")
    for row in K:
        print(f"  [{row[0]:10.3f}, {row[1]:10.3f}, {row[2]:10.3f}]")

    print(f"\nParâmetros Intrínsecos Individuais:")
    print(f"  fx (Focal X): {K[0, 0]:.3f} px")
    print(f"  fy (Focal Y): {K[1, 1]:.3f} px")
    print(f"  cx (Optical Center X): {K[0, 2]:.3f} px (Res: {img_shape[0]})")
    print(f"  cy (Optical Center Y): {K[1, 2]:.3f} px (Res: {img_shape[1]})")

    print(f"\nCoeficientes de Distorção (5 coeficientes de Brown-Conrady):")
    d_flat = dist.flatten()
    print(f"  k1 (Radial 1)   : {d_flat[0]:+9.6f}")
    print(f"  k2 (Radial 2)   : {d_flat[1]:+9.6f}")
    print(f"  p1 (Tangencial 1): {d_flat[2]:+9.6f}")
    print(f"  p2 (Tangencial 2): {d_flat[3]:+9.6f}")
    print(f"  k3 (Radial 3)   : {d_flat[4]:+9.6f}")

    print(f"\nQualidade da Calibração (Erro de Reprojeção):")
    print(f"  Erro Médio Global: {erro_medio:.4f} pixels")
    print(f"  Menor Erro Imagem: {min(erros_img):.4f} pixels")
    print(f"  Maior Erro Imagem: {max(erros_img):.4f} pixels")

    if erro_medio < 0.5:
        classificacao = "EXCELENTE (Apta para Odometria Visual e Condução Autônoma)"
    elif erro_medio < 1.0:
        classificacao = "ACEITÁVEL (Apta para Navegação Robótica Básica)"
    else:
        classificacao = "INSUFICIENTE (Necessita recalibração para evitar drift métrico)"
    print(f"  Avaliação Robótica: {classificacao}")
    print("-" * 75)

    # 5. Aplicar undistort em uma das imagens e montar painel comparativo lado a lado
    exemplo_caminho = valid_imgs[0]
    img_distorcida = cv2.imread(str(exemplo_caminho))
    img_corrigida = cv2.undistort(img_distorcida, K, dist)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(cv2.cvtColor(img_distorcida, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Imagem Original (Com Distorção de Lente)\nCurvaturas nas bordas", fontsize=12, fontweight="bold")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img_corrigida, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Imagem Corrigida (cv2.undistort)\nLinhas retificadas | Erro: {erro_medio:.3f} px", fontsize=12, fontweight="bold", color="darkgreen")
    axs[1].axis("off")

    plt.suptitle("Exercício 1A: Calibração de Câmera e Retificação de Distorção", fontsize=14, fontweight="bold")
    salvar_figura(SAIDAS_DIR / "at1a_undistort_comparativo.png", dpi=200)


if __name__ == "__main__":
    main()
