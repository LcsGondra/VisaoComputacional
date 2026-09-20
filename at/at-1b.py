"""
Exercício 1 — Item B: Sobreposição de Realidade Aumentada com Estimação de Pose 3D
Competências: 1.2, 1.3 e 4.1

Este script utiliza a calibração intrínseca obtida no Item A (matriz K e coeficientes de distorção)
para realizar rastreamento de pose em tempo real via Perspective-n-Point (solvePnP) e
projetar um cubo virtual 3D sobre o canto de origem do tabuleiro de xadrez.

Pipeline de Realidade Aumentada (RA):
------------------------------------
1. Leitura do frame e detecção dos cantos internos do tabuleiro (cv2.findChessboardCorners).
2. Refinamento subpixel dos cantos detectados (cv2.cornerSubPix).
3. Estimação de pose 3D (cv2.solvePnP):
   Calcula o vetor de rotação (rvec) e translação (tvec) que mapeiam pontos do mundo para a câmera:
   [x_c, y_c, z_c]^T = R * [X_w, Y_w, Z_w]^T + t
4. Projeção geométrica dos 8 vértices do cubo 3D virtual (cv2.projectPoints):
   Aresta = 1 quadrado (30 mm) no canto de origem (0, 0, 0).
5. Renderização das faces com cores distintas e semitransparência (cv2.addWeighted).
6. Telemetria contínua exibindo rvec (graus/rad) e tvec (distância euclidiana em mm).
"""

from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    salvar_figura,
    CALIB_DIR,
    CALIB_FILE,
    SAIDAS_DIR,
)


def carregar_calibracao():
    """Carrega matriz intrínseca K e coeficientes de distorção gerados no Ex 1A."""
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


def definir_geometria_cubo(square_size=30.0):
    """
    Define os 8 vértices 3D de um cubo com aresta igual a 1 quadrado (30mm)
    posicionado no canto de origem (canto superior esquerdo do padrão):
    Base no plano Z=0, topo em Z = -square_size (sistema de coordenadas OpenCV: Z aponta para frente).
    """
    s = square_size
    vertices_3d = np.float32([
        [0, 0, 0],       # 0: base origem
        [s, 0, 0],       # 1: base x
        [s, s, 0],       # 2: base diagonal
        [0, s, 0],       # 3: base y
        [0, 0, -s],      # 4: topo origem
        [s, 0, -s],      # 5: topo x
        [s, s, -s],      # 6: topo diagonal
        [0, s, -s],      # 7: topo y
    ])
    return vertices_3d


def desenhar_cubo_faces_coloridas(img, pts_2d, alpha=0.45):
    """
    Desenha as faces do cubo com cores sólidas translúcidas e arestas destacadas.
    Cores por face:
    - Base: Verde
    - Topo: Azul
    - Lateral Frontal: Vermelho
    - Lateral Direita: Amarelo
    - Lateral Esquerda: Ciano
    """
    overlay = img.copy()
    pts = np.int32(pts_2d).reshape(-1, 2)

    # Faces do cubo definidas por índices dos vértices
    # Base: 0, 1, 2, 3
    # Topo: 4, 5, 6, 7
    # Lateral frontal: 0, 1, 5, 4
    # Lateral direita: 1, 2, 6, 5
    # Lateral traseira: 2, 3, 7, 6
    # Lateral esquerda: 3, 0, 4, 7
    faces = [
        ([pts[0], pts[1], pts[2], pts[3]], (0, 200, 0), "Base Verde"),
        ([pts[4], pts[5], pts[6], pts[7]], (220, 50, 0), "Topo Azul"),
        ([pts[0], pts[1], pts[5], pts[4]], (0, 0, 220), "Frontal Vermelha"),
        ([pts[1], pts[2], pts[6], pts[5]], (0, 220, 220), "Direita Amarela"),
        ([pts[3], pts[0], pts[4], pts[7]], (220, 220, 0), "Esquerda Ciano"),
    ]

    for poly, cor, _ in faces:
        cv2.fillConvexPoly(overlay, np.array(poly), cor)

    # Combina com transparência
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Desenha arestas externas com espessura 2
    # Base
    cv2.polylines(img, [np.array([pts[0], pts[1], pts[2], pts[3]])], True, (0, 255, 0), 2)
    # Topo
    cv2.polylines(img, [np.array([pts[4], pts[5], pts[6], pts[7]])], True, (255, 100, 0), 2)
    # Pilares verticais conectando base e topo
    for i in range(4):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 4]), (0, 0, 255), 2)

    # Marca o vértice de origem com esfera dourada
    cv2.circle(img, tuple(pts[0]), 5, (0, 215, 255), -1)
    return img


def processar_sequencia_ra(caminhos_imagens, K, dist, pattern_size=(7, 6), square_size=30.0):
    """
    Processa a sequência de imagens simulando feed em tempo real de câmera.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    cubo_3d = definir_geometria_cubo(square_size)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n" + "-" * 75)
    print("ESTIMAÇÃO DE POSE E REALIDADE AUMENTADA (solvePnP + projectPoints):")
    print(f"{'Frame':<8} | {'Rot X (deg)':<12} | {'Rot Y (deg)':<12} | {'Rot Z (deg)':<12} | {'Distância (mm)':<16} | {'Status'}")
    print("-" * 75)

    frames_anotados = []
    poses_estimadas = []

    for idx, caminho in enumerate(caminhos_imagens, start=1):
        frame = cv2.imread(str(caminho))
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criterio)

            # Estimação de pose PnP (rotação e translação)
            ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)

            if ret_pnp:
                # Projeção dos vértices do cubo 3D no plano 2D
                imgpts, _ = cv2.projectPoints(cubo_3d, rvec, tvec, K, dist)

                # Renderização do cubo virtual com faces coloridas
                vis = desenhar_cubo_faces_coloridas(frame.copy(), imgpts)

                # Conversão de rvec em graus de Euler
                R, _ = cv2.Rodrigues(rvec)
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                ry = np.degrees(np.arctan2(-R[2, 0], sy))
                rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

                dist_euclidiana = float(np.linalg.norm(tvec))
                poses_estimadas.append((rvec, tvec))

                # HUD sobre o frame
                cv2.rectangle(vis, (10, 10), (360, 95), (0, 0, 0), -1)
                cv2.rectangle(vis, (10, 10), (360, 95), (0, 255, 255), 1)
                cv2.putText(vis, "Realidade Aumentada (Pose solvePnP)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(vis, f"Rot (deg): X={rx:+.1f} Y={ry:+.1f} Z={rz:+.1f}", (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.putText(vis, f"Trans (mm): [{tvec[0,0]:.0f}, {tvec[1,0]:.0f}, {tvec[2,0]:.0f}] | d={dist_euclidiana:.1f}mm", (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

                frames_anotados.append(vis)
                print(f"Frame {idx:02d} | {rx:+10.2f}° | {ry:+10.2f}° | {rz:+10.2f}° | {dist_euclidiana:14.2f} mm | RASTREADO")
        else:
            print(f"Frame {idx:02d} | {'--':<12} | {'--':<12} | {'--':<12} | {'--':<16} | NÃO DETECTADO")

    print("-" * 75)
    return frames_anotados, poses_estimadas


def main():
    ensure_dirs()
    print("=" * 75)
    print("EXERCÍCIO 1B — REALIDADE AUMENTADA EM TEMPO REAL COM CUBO 3D")
    print("=" * 75)

    # 1. Carregar calibração prévia
    K, dist = carregar_calibracao()

    # 2. Carregar imagens do tabuleiro
    caminhos = sorted(list(CALIB_DIR.glob("calib_*.png")))
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {CALIB_DIR}. Execute 'at-1a.py'.")

    # 3. Executar pipeline de rastreamento de pose e projeção de RA
    frames_ra, poses = processar_sequencia_ra(caminhos, K, dist, pattern_size=(7, 6), square_size=30.0)

    if not frames_ra:
        raise RuntimeError("Nenhum frame com cubo sobreposto foi gerado.")

    # 4. Salvar demonstração em figura estática de alta qualidade
    # Mostra 3 poses em ângulos distintos evidenciando estabilidade 3D do cubo
    indices_amostra = [0, len(frames_ra) // 2, len(frames_ra) - 1]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    for i, idx in enumerate(indices_amostra):
        axs[i].imshow(cv2.cvtColor(frames_ra[idx], cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Pose {idx + 1} — Rotação e Translação 3D", fontsize=11, fontweight="bold")
        axs[i].axis("off")

    plt.suptitle("Exercício 1B: Realidade Aumentada com Cubo 3D Estável sobre Tabuleiro Calibrado", fontsize=13, fontweight="bold")
    salvar_figura(SAIDAS_DIR / "at1b_ra_cubo.png", dpi=200)

    # 5. Salvar também vídeo de demonstração do cubo 3D
    caminho_video_out = SAIDAS_DIR / "at1b_ra_cubo.mp4"
    h, w = frames_ra[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(caminho_video_out), fourcc, 5, (w, h))
    for f in frames_ra:
        vw.write(f)
    vw.release()
    print(f"[+] Demonstração em vídeo salva em: {caminho_video_out.name}")


if __name__ == "__main__":
    main()
