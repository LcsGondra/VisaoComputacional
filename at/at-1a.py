# Significado Físico dos Parâmetros Estimados:
# --------------------------------------------
# 1. Matriz Intrínseca K:
#    K = [[fx,  s, cx],
#         [ 0, fy, cy],
#         [ 0,  0,  1]]
#    - fx, fy: Distâncias focais expressas em pixels (focal length). Representam a distância
#      entre o centro óptico da lente e o plano do sensor multiplicada pela densidade de pixels por mm.
#    - s: Fator de cisalhamento (skew factor = 0 para sensores CCD/CMOS com pixels ortogonais).
#    - cx, cy: Ponto principal (centro óptico projetado no plano da imagem em pixels).
#
# 2. Coeficientes de Distorção D (5 parâmetros de Brown-Conrady):
#    D = [k1, k2, p1, p2, k3]
#    - k1, k2, k3 (Distorção Radial): Decorrente da curvatura esférica da lente (barril se k > 0, almofada se k < 0).
#    - p1, p2 (Distorção Tangencial): Desalinhamento físico entre as lentes ópticas e o plano do sensor semicondutor.
#
# Tolerância Aceitável para Aplicações Robóticas e Veículos Autônomos:
# -------------------------------------------------------------------
# - Aplicações industriais e metrologia: erro < 0.20 pixels.
# - Robôs móveis e veículos autônomos (odometria visual e fusão sensorial): erro < 0.50 pixels.
# - Limite máximo tolerável em robótica de serviço: erro < 1.00 pixel.

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    obter_dataset_calibracao_opencv,
    salvar_figura,
    exibir_janela_interativa,
    criar_mosaico_imagens,
    CALIB_FILE,
    SAIDAS_DIR,
)


def calibrar_camera(imagens_caminhos, pattern_size=(9, 6), square_size_mm=25.0):
    # Detecta cantos do tabuleiro no dataset oficial do OpenCV e calcula a calibração intrínseca.
    cols, rows = pattern_size

    # Coordenadas 3D dos cantos internos no mundo (plano Z = 0)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    objpoints = []
    imgpoints = []
    valid_images = []
    img_shape = None

    criterio_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    imgs_com_cantos = []
    titulos_com_cantos = []

    print(f"Processando {len(imagens_caminhos)} fotos reais do OpenCV para calibração...")
    for idx, caminho in enumerate(imagens_caminhos, start=1):
        img = cv2.imread(str(caminho))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # Detecção dos cantos internos no padrão (9, 6)
        ret, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        vis = img.copy()
        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criterio_subpix)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            valid_images.append(caminho)
            print(f"    Imagem {idx:02d}/{len(imagens_caminhos):02d} ({caminho.name}): Cantos (9x6) detectados!")

            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, ret)
            cv2.putText(vis, f"{caminho.name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            imgs_com_cantos.append(vis)
            titulos_com_cantos.append(f"{caminho.name} (OK)")
        else:
            print(f"    Imagem {idx:02d}/{len(imagens_caminhos):02d} ({caminho.name}): Cantos não detectados.")
            cv2.putText(vis, f"{caminho.name} (Nao detectado)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            imgs_com_cantos.append(vis)
            titulos_com_cantos.append(f"{caminho.name} (FALHA)")

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
        erro = cv2.norm(imgpoints[i].reshape(-1, 2).astype(np.float32), imgpoints2.reshape(-1, 2).astype(np.float32), cv2.NORM_L2) / len(imgpoints2)
        erros_por_imagem.append(erro)

    erro_medio = float(np.mean(erros_por_imagem))
    return K, dist, rvecs, tvecs, erros_por_imagem, erro_medio, valid_images, img_shape, imgs_com_cantos, titulos_com_cantos


def main():
    ensure_dirs()
    print("=" * 80)
    print("EXERCÍCIO 1A — CALIBRAÇÃO DE CÂMERA COM DATASET OFICIAL DO OPENCV")
    print("=" * 80)

    # 1. Carrega as 18 fotos reais do dataset oficial de calibração do OpenCV
    caminhos_calib = obter_dataset_calibracao_opencv(num_imagens=18)

    # 2. Realiza a calibração com cantos (9, 6) e aresta de 25mm
    K, dist, rvecs, tvecs, erros_img, erro_medio, valid_imgs, img_shape, imgs_cantos, titulos_cantos = calibrar_camera(
        caminhos_calib, pattern_size=(9, 6), square_size_mm=25.0
    )

    # 2.1 Exibir todas as 18 fotos do dataset em uma única janela mosaico interativa
    mosaico_dataset = criar_mosaico_imagens(
        imgs_cantos,
        titulos=titulos_cantos,
        cols=6,
        thumb_size=(240, 180),
        titulo_geral="Dataset Oficial OpenCV: 18 Fotos Reais com Cantos 9x6 Detectados"
    )
    exibir_janela_interativa(
        "Exercicio 1A - Dataset de Calibracao (18 Fotos Reais)",
        mosaico_dataset,
        "Pressione 'q', ESC ou feche no [X] para prosseguir"
    )

    # 3. Salvar calibração em arquivo binário para os demais exercícios
    np.savez(
        CALIB_FILE,
        K=K,
        dist=dist,
        erro_medio=erro_medio,
        img_shape=img_shape,
    )
    print(f"\nMatriz K e distorção salvas em: {CALIB_FILE.name}")

    # 4. Impressão dos parâmetros no terminal
    print("\n" + "-" * 80)
    print("PARÂMETROS ESTIMADOS DA CÂMERA REAL (DATASET OPENCV):")
    print("-" * 80)
    print("Matriz Intrínseca K [3x3]:")
    for row in K:
        print(f"  [{row[0]:10.3f}, {row[1]:10.3f}, {row[2]:10.3f}]")

    print(f"\nDistâncias Focais e Centro Óptico:")
    print(f"  fx = {K[0, 0]:.3f} px | fy = {K[1, 1]:.3f} px")
    print(f"  cx = {K[0, 2]:.3f} px | cy = {K[1, 2]:.3f} px (Resolução: {img_shape[0]}x{img_shape[1]})")

    print(f"\nCoeficientes de Distorção (Brown-Conrady):")
    d = dist.flatten()
    print(f"  k1 (Radial 1)    : {d[0]:+9.6f}")
    print(f"  k2 (Radial 2)    : {d[1]:+9.6f}")
    print(f"  p1 (Tangencial 1): {d[2]:+9.6f}")
    print(f"  p2 (Tangencial 2): {d[3]:+9.6f}")
    print(f"  k3 (Radial 3)    : {d[4]:+9.6f}")

    print(f"\nErro de Reprojeção:")
    print(f"  Erro Médio Global: {erro_medio:.4f} pixels")
    print(f"  Menor Erro Imagem: {min(erros_img):.4f} pixels")
    print(f"  Maior Erro Imagem: {max(erros_img):.4f} pixels")

    if erro_medio < 0.50:
        classificacao = "EXCELENTE (Apta para Odometria Visual e Condução Autônoma)"
    elif erro_medio < 1.00:
        classificacao = "ACEITÁVEL (Apta para Navegação Robótica Básica)"
    else:
        classificacao = "INSUFICIENTE (Necessita recalibração)"
    print(f"  Avaliação Robótica: {classificacao}")
    print("-" * 80)

    # 5. Aplicar undistort em foto real do dataset e salvar painel comparativo
    exemplo_p = valid_imgs[0]
    img_real = cv2.imread(str(exemplo_p))
    img_undist = cv2.undistort(img_real, K, dist)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"Foto Original Real ({exemplo_p.name})\nDistorção natural da lente", fontsize=11, fontweight="bold")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Foto Corrigida com cv2.undistort\nRetificação ótica | Erro: {erro_medio:.3f} px", fontsize=11, fontweight="bold", color="darkgreen")
    axs[1].axis("off")

    plt.suptitle("Exercício 1A: Calibração de Câmera Real — Dataset Oficial OpenCV Samples", fontsize=13, fontweight="bold")
    salvar_figura(SAIDAS_DIR / "at1a_undistort_comparativo.png", dpi=200)

    # 6. Exibir janela OpenCV interativa com o comparativo lado a lado
    comp = np.hstack((img_real, img_undist))
    cv2.putText(comp, "ORIGINAL (COM DISTORCAO)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(comp, f"UNDISTORT RETIFICADA (ERRO: {erro_medio:.3f} px)", (img_real.shape[1] + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    exibir_janela_interativa(
        "Exercicio 1A - Comparativo de Calibracao (Original vs Retificada)",
        comp,
        "Pressione 'q', ESC ou feche no [X] para prosseguir"
    )

    # 7. Gerar e exibir gráfico de perda/erro de reprojeção da calibração
    fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5))
    axs2[0].bar(range(1, len(erros_img) + 1), erros_img, color="steelblue", edgecolor="black")
    axs2[0].axhline(y=erro_medio, color="red", linestyle="--", label=f"Erro Médio Global: {erro_medio:.4f} px")
    axs2[0].axhline(y=0.50, color="orange", linestyle=":", label="Limite Robótica Móvel (0.50 px)")
    axs2[0].set_title("Curva de Perda de Calibração: Erro de Reprojeção por Imagem", fontsize=11, fontweight="bold")
    axs2[0].set_xlabel("Índice da Foto de Calibração (OpenCV Samples)", fontsize=10)
    axs2[0].set_ylabel("Erro de Reprojeção (pixels)", fontsize=10)
    axs2[0].set_xticks(range(1, len(erros_img) + 1))
    axs2[0].legend(fontsize=9)
    axs2[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # Histograma / Distribuição de resíduos
    axs2[1].hist(erros_img, bins=8, color="seagreen", edgecolor="black", alpha=0.8)
    axs2[1].axvline(x=erro_medio, color="red", linestyle="--", label=f"Média: {erro_medio:.4f} px")
    axs2[1].set_title("Distribuição Estatística dos Resíduos de Calibração", fontsize=11, fontweight="bold")
    axs2[1].set_xlabel("Erro de Reprojeção (pixels)", fontsize=10)
    axs2[1].set_ylabel("Frequência", fontsize=10)
    axs2[1].legend(fontsize=9)
    axs2[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Exercício 1A: Curva de Perda e Resíduos da Calibração Fotogramétrica", fontsize=13, fontweight="bold")
    caminho_calib_loss = SAIDAS_DIR / "at1a_metricas_calibracao_loss.png"
    salvar_figura(caminho_calib_loss, dpi=200)

    fig_loss_img = cv2.imread(str(caminho_calib_loss))
    if fig_loss_img is not None:
        exibir_janela_interativa(
            "Exercicio 1A - Curva de Perda e Residuos de Calibracao",
            fig_loss_img,
            "Pressione 'q', ESC ou feche no [X] para finalizar"
        )


if __name__ == "__main__":
    main()
