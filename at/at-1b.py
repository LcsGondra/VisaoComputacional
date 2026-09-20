# Exercício 1 — Item B: Sobreposição de Realidade Aumentada com Estimação de Pose 3D
# Competências: 1.2, 1.3 e 4.1
#
# Este script utiliza a calibração intrínseca obtida no Item A sobre o dataset oficial do OpenCV
# para realizar a estimação de pose 3D (solvePnP) e projetar um cubo virtual de realidade
# aumentada sobre o tabuleiro real de xadrez nas fotografias da câmera.
#
# Pipeline de Realidade Aumentada (RA):
# ------------------------------------
# 1. Leitura das fotos reais de calibração do OpenCV e detecção dos cantos internos (9x6).
# 2. Refinamento subpixel dos cantos detectados (cv2.cornerSubPix).
# 3. Estimação de pose 3D (cv2.solvePnP) calculando vetor de rotação (rvec) e translação (tvec).
# 4. Projeção dos 8 vértices do cubo 3D virtual com aresta de 1 quadrado (25 mm) via cv2.projectPoints.
# 5. Renderização das faces translúcidas com cores distintas por face.
# 6. Telemetria contínua exibindo rvec (graus) e tvec (distância em mm).

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    salvar_figura,
    exibir_janela_interativa,
    criar_mosaico_imagens,
    CALIB_FILE,
    SAIDAS_DIR,
    CALIB_DIR,
)


def carregar_calibracao():
    # Carrega matriz intrínseca K e coeficientes de distorção gerados no Ex 1A.
    if not CALIB_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de calibração '{CALIB_FILE.name}' não encontrado! "
            "Execute primeiro o script 'at-1a.py'."
        )
    dados = np.load(CALIB_FILE)
    K = dados["K"]
    dist = dados["dist"]
    erro_medio = float(dados["erro_medio"])
    print(f"[+] Calibração carregada com sucesso ({CALIB_FILE.name}) | Erro base: {erro_medio:.4f} px")
    return K, dist


def definir_geometria_cubo(square_size=25.0, tamanho_aresta=37.5, pos_centro=True):
    # Define os 8 vértices 3D de um cubo virtual de Realidade Aumentada:
    # - Se pos_centro=True: posicionado exatamente no CENTRO do tabuleiro 9x6
    #   (ponto médio X=4.0*s, Y=2.5*s), perfeitamente assentado no miolo do padrão
    #   sem risco de oclusão por bordas ou textos.
    # - Se pos_centro=False: posicionado no canto de origem (0, 0, 0).
    # Base no plano Z = 0, topo em Z = -tamanho_aresta (Z apontando para a câmera).
    s = square_size
    edge = tamanho_aresta
    if pos_centro:
        cx = 4.0 * s
        cy = 2.5 * s
        x0, x1 = cx - edge / 2.0, cx + edge / 2.0
        y0, y1 = cy - edge / 2.0, cy + edge / 2.0
    else:
        x0, x1 = 0.0, edge
        y0, y1 = 0.0, edge

    vertices_3d = np.float32([
        [x0, y0, 0],       # 0: base inferior-esq
        [x1, y0, 0],       # 1: base inferior-dir
        [x1, y1, 0],       # 2: base superior-dir
        [x0, y1, 0],       # 3: base superior-esq
        [x0, y0, -edge],   # 4: topo inferior-esq
        [x1, y0, -edge],   # 5: topo inferior-dir
        [x1, y1, -edge],   # 6: topo superior-dir
        [x0, y1, -edge],   # 7: topo superior-esq
    ])
    return vertices_3d


def desenhar_cubo_faces_coloridas(img, pts_2d, alpha=0.45):
    # Desenha as faces do cubo com cores sólidas translúcidas e arestas coloridas.
    overlay = img.copy()
    pts = np.int32(pts_2d).reshape(-1, 2)

    faces = [
        ([pts[0], pts[1], pts[2], pts[3]], (0, 200, 0), "Base Verde"),
        ([pts[4], pts[5], pts[6], pts[7]], (220, 50, 0), "Topo Azul"),
        ([pts[0], pts[1], pts[5], pts[4]], (0, 0, 220), "Frontal Vermelha"),
        ([pts[1], pts[2], pts[6], pts[5]], (0, 220, 220), "Direita Amarela"),
        ([pts[3], pts[0], pts[4], pts[7]], (220, 220, 0), "Esquerda Ciano"),
    ]

    for poly, cor, _ in faces:
        cv2.fillConvexPoly(overlay, np.array(poly), cor)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Arestas da base e topo
    cv2.polylines(img, [np.array([pts[0], pts[1], pts[2], pts[3]])], True, (0, 255, 0), 2)
    cv2.polylines(img, [np.array([pts[4], pts[5], pts[6], pts[7]])], True, (255, 100, 0), 2)

    # Pilares verticais
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), (0, 0, 255), 2)

    # Origem do referencial 3D
    cv2.circle(img, tuple(pts[0]), 5, (0, 215, 255), -1)
    return img


def processar_sequencia_ra_real(caminhos_imagens, K, dist, pattern_size=(9, 6), square_size=25.0):
    # Processa a sequência de fotos reais de calibração do OpenCV estimando a pose 3D.
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    cubo_3d = definir_geometria_cubo(square_size=square_size, tamanho_aresta=37.5, pos_centro=True)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n" + "-" * 75)
    print("ESTIMAÇÃO DE POSE E REALIDADE AUMENTADA (FOTOS REAIS OPENCV):")
    print(f"{'Foto':<12} | {'Rot X (deg)':<12} | {'Rot Y (deg)':<12} | {'Rot Z (deg)':<12} | {'Distância (mm)':<16} | {'Status'}")
    print("-" * 75)

    dados_processados = []

    for caminho in caminhos_imagens:
        frame = cv2.imread(str(caminho))
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criterio)
            ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)

            if ret_pnp:
                imgpts, _ = cv2.projectPoints(cubo_3d, rvec, tvec, K, dist)
                # Mantém imagem 100% limpa, sem caixas de texto sobrepostas que possam ocluir a cena
                vis = desenhar_cubo_faces_coloridas(frame.copy(), imgpts)

                R, _ = cv2.Rodrigues(rvec)
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                ry = np.degrees(np.arctan2(-R[2, 0], sy))
                rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                dist_euclidiana = float(np.linalg.norm(tvec))

                dados_processados.append({
                    "nome": caminho.name,
                    "img": vis,
                    "rvec": rvec,
                    "tvec": tvec,
                    "euler": (rx, ry, rz),
                    "dist": dist_euclidiana
                })
                print(f"{caminho.name:<12} | {rx:+10.2f}° | {ry:+10.2f}° | {rz:+10.2f}° | {dist_euclidiana:14.2f} mm | RASTREADO")
        else:
            print(f"{caminho.name:<12} | {'--':<12} | {'--':<12} | {'--':<12} | {'--':<16} | NÃO DETECTADO")

    print("-" * 75)
    return dados_processados


def main():
    ensure_dirs()
    print("=" * 80)
    print("EXERCÍCIO 1B — REALIDADE AUMENTADA SOBRE FOTOS REAIS DO OPENCV")
    print("=" * 80)

    # 1. Carrega calibração real
    K, dist = carregar_calibracao()

    # 2. Carrega fotos de calibração reais em CALIB_DIR
    caminhos = sorted(list(CALIB_DIR.glob("*.jpg")) + list(CALIB_DIR.glob("*.png")))
    if not caminhos:
        raise FileNotFoundError("Nenhuma imagem encontrada em calibracao/. Execute at-1a.py.")

    # 3. Executa sobreposição de realidade aumentada
    dados_ra = processar_sequencia_ra_real(caminhos, K, dist, pattern_size=(9, 6), square_size=25.0)

    # 4. Salvar painel estático comparando 3 poses reais da câmera com dados ABAIXO de cada imagem
    # Seleciona fotos representativas com ângulos distintos (frontal, inclinação extrema e ângulo oblíquo)
    nomes_alvo = ["left01.jpg", "left05.jpg", "right01.jpg"]
    dados_amostra = [item for item in dados_ra if item["nome"] in nomes_alvo]
    if len(dados_amostra) < 3:
        # Fallback para 3 índices espaçados caso algum nome varie
        indices = [0, len(dados_ra) // 2, len(dados_ra) - 1]
        dados_amostra = [dados_ra[i] for i in indices]

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    for i, item in enumerate(dados_amostra):
        nome = item["nome"]
        img = item["img"]
        rx, ry, rz = item["euler"]
        tvec = item["tvec"]
        dist_mm = item["dist"]

        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Pose Real: {nome}", fontsize=12, fontweight="bold", pad=8)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        # Legenda informativa posicionada LIMPA ABAIXO da imagem, sem cobrir o tabuleiro ou o cubo
        texto_telemetria = (
            f"Rot (Euler): X={rx:+.1f}° | Y={ry:+.1f}° | Z={rz:+.1f}°\n"
            f"Trans: [{tvec[0,0]:.0f}, {tvec[1,0]:.0f}, {tvec[2,0]:.0f}] mm | Distância: {dist_mm:.1f} mm"
        )
        axs[i].set_xlabel(
            texto_telemetria,
            fontsize=9.5,
            fontweight="medium",
            labelpad=8,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#0066cc", alpha=0.9)
        )

    plt.suptitle("Exercício 1B: Realidade Aumentada (Cubo 3D no Centro do Tabuleiro)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    salvar_figura(SAIDAS_DIR / "at1b_ra_cubo.png", dpi=200)

    # 5. Salvar vídeo com a sequência de fotos com RA
    caminho_video_out = SAIDAS_DIR / "at1b_ra_cubo.mp4"
    h, w = dados_ra[0]["img"].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(caminho_video_out), fourcc, 2, (w, h))
    for item in dados_ra:
        # Repete 3 frames por imagem para visualização suave da pose
        for _ in range(3):
            vw.write(item["img"])
    vw.release()
    print(f"[+] Demonstração em vídeo salva em: {caminho_video_out.name}")

    # 6. Exibir todas as 18 fotos com o cubo 3D sobreposto em janela mosaico única
    imgs_cubos = [item["img"] for item in dados_ra]
    titulos_cubos = [item["nome"] for item in dados_ra]
    mosaico_ra = criar_mosaico_imagens(
        imgs_cubos,
        titulos=titulos_cubos,
        cols=6,
        thumb_size=(240, 180),
        titulo_geral="Realidade Aumentada (Cubo 3D Centralizado): 18 Fotos Reais com Pose solvePnP"
    )
    exibir_janela_interativa(
        "Exercicio 1B - Realidade Aumentada (18 Fotos Reais com Cubo 3D)",
        mosaico_ra,
        "Pressione 'q', ESC ou feche no [X] para prosseguir"
    )

    # 7. Exibir painel comparativo de 3 poses em alta resolução
    painel_3p = cv2.imread(str(SAIDAS_DIR / "at1b_ra_cubo.png"))
    if painel_3p is not None:
        exibir_janela_interativa(
            "Exercicio 1B - Comparativo 3 Poses de Realidade Aumentada",
            painel_3p,
            "Pressione 'q', ESC ou feche no [X] para finalizar"
        )


if __name__ == "__main__":
    main()
