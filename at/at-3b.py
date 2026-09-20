"""
Exercício 3 — Item B: Rastreamento com ID Persistente, Trilhas Temporais e Contagem
Competências: 3.1, 3.2 e 4.3

Este script integra o detector de objetos em tempo real (YOLOv4-tiny) com um algoritmo
de rastreamento por associação espacial e temporal (IoUTracker), garantindo:
1. Atribuição de ID persistente e consistente para cada objeto detectado.
2. Histórico e desenho das trilhas dos últimos 30 frames como linha contínua colorida.
3. Linha virtual de contagem com detecção de fluxo bidirecional (entradas e saídas)
   e acumulador de tráfego.
4. Contabilização e cálculo da taxa de trocas de ID (ID switches) por minuto de vídeo.
5. Discussão técnica e ética detalhada sobre sistemas de vigilância e contagem aérea por drones.
"""

from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    detectar_yolo_tiny,
    gerar_video_transito,
    obter_modelo_yolo_tiny,
    salvar_figura,
    DADOS_DIR,
    IoUTracker,
    PALETA_CORES,
    SAIDAS_DIR,
)


def desenhar_rastreamento(frame, objetos_rastreados, tracker, line_x=400):
    """
    Desenha o estado completo do rastreador sobre o frame:
    - Bounding box colorido por classe com ID persistente.
    - Trilha dos últimos 30 frames.
    - Linha virtual de contagem com HUD de telemetria.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Linha virtual de contagem (Tracejada e iluminada em amarelo)
    for y in range(0, h, 20):
        cv2.line(vis, (line_x, y), (line_x, y + 10), (0, 255, 255), 2)
    cv2.putText(vis, "LINHA DE CONTAGEM VIRTUAL", (line_x - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    # Cores fixas para identificação visual dos IDs rastreados
    cores_ids = [
        (255, 120, 0), (0, 220, 255), (0, 255, 120), (255, 0, 200),
        (255, 255, 0), (0, 160, 255), (180, 0, 255), (0, 255, 0)
    ]

    for obj in objetos_rastreados:
        tid = obj["id"]
        x1, y1, x2, y2 = obj["bbox"]
        cls = obj["cls"]
        conf = obj["conf"]
        cor_track = cores_ids[tid % len(cores_ids)]

        # Bounding box do objeto
        cv2.rectangle(vis, (x1, y1), (x2, y2), cor_track, 2)

        # Rótulo com ID persistente
        tag = f"ID #{tid:02d} | {cls.upper()} {conf*100:.0f}%"
        cv2.rectangle(vis, (x1, max(0, y1 - 22)), (x1 + len(tag) * 8, y1), cor_track, -1)
        cv2.putText(vis, tag, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        # Desenho da trilha dos últimos frames
        trail = obj["trail"]
        if len(trail) > 1:
            for p1, p2 in zip(trail[:-1], trail[1:]):
                cv2.line(vis, p1, p2, cor_track, 2)
            cv2.circle(vis, trail[-1], 4, (0, 255, 255), -1)

    # Painel de Telemetria Superior (HUD)
    hud_w = 360
    overlay = vis.copy()
    cv2.rectangle(overlay, (10, 10), (hud_w, 115), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)
    cv2.rectangle(vis, (10, 10), (hud_w, 115), (0, 255, 0), 1)

    cv2.putText(vis, "TELEMETRIA DE RASTREAMENTO E CONTAGEM", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
    cv2.putText(vis, f"Objetos Ativos Rastreados: {len(objetos_rastreados)}", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
    cv2.putText(vis, f"Entradas (L -> R): {tracker.crossed_in}  |  Saídas (R -> L): {tracker.crossed_out}", (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
    cv2.putText(vis, f"Total Cumulativo Cruzado: {tracker.crossed_in + tracker.crossed_out}", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1)
    cv2.putText(vis, f"ID Switches Detectados: {tracker.id_switches}", (18, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 255), 1)

    return vis


def main():
    ensure_dirs()
    print("=" * 80)
    print("EXERCÍCIO 3B — RASTREAMENTO IOU, PERSISTÊNCIA DE ID E LINHA DE CONTAGEM")
    print("=" * 80)

    # 1. Carrega modelo YOLOv4-tiny
    net_yolo, classes_yolo, _, _ = obter_modelo_yolo_tiny()

    # 2. Carrega vídeo de trânsito
    caminho_video = DADOS_DIR / "transito_urbano.mp4"
    gerar_video_transito(caminho_video, n_frames=90, fps=20, size=(800, 450))

    cap = cv2.VideoCapture(str(caminho_video))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao_segundos = total_frames / fps_video
    duracao_minutos = duracao_segundos / 60.0

    print(f"[+] Processando vídeo: {total_frames} frames ({duracao_segundos:.1f} segundos / {duracao_minutos:.3f} min)")

    # 3. Inicializa o rastreador por IoU com linha virtual no centro da pista (x = 400)
    tracker = IoUTracker(iou_thresh=0.30, max_missing=10, trail_len=30, line_x=400)

    snapshots = []
    frames_anotados = []
    frame_count = 0

    print("\n" + "-" * 75)
    print("EXECUÇÃO DO RASTREADOR (IoU Association + Persistent Tracks):")
    print(f"{'Frame':<8} | {'Dets':<6} | {'Ativos':<8} | {'Entradas':<10} | {'Saídas':<10} | {'ID Switches'}")
    print("-" * 75)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Detecção YOLO
        dets, _ = detectar_yolo_tiny(net_yolo, frame, classes_yolo, score_thresh=0.20, nms_thresh=0.40)

        # Atualização do rastreador
        rastreados = tracker.update(dets)

        # Renderização das anotações e trilhas
        vis = desenhar_rastreamento(frame, rastreados, tracker, line_x=400)
        frames_anotados.append(vis)

        if frame_count % 20 == 0 or frame_count == 1:
            snapshots.append((frame_count, vis))
            print(f"{frame_count:02d}/{total_frames:02d}    | {len(dets):<6} | {len(rastreados):<8} | {tracker.crossed_in:<10} | {tracker.crossed_out:<10} | {tracker.id_switches}")

    cap.release()

    # Cálculo da taxa de ID Switches por minuto
    taxa_id_switches_min = tracker.id_switches / max(1e-3, duracao_minutos)

    print("-" * 75)
    print("\nRELATÓRIO DE TELEMETRIA DO RASTREAMENTO:")
    print("=" * 75)
    print(f"Total de Objetos Criados (Novos IDs) : {tracker.total_tracks_criadas}")
    print(f"Contagem Cumulativa de Entradas     : {tracker.crossed_in}")
    print(f"Contagem Cumulativa de Saídas       : {tracker.crossed_out}")
    print(f"Fluxo Total de Tráfego Detectado    : {tracker.crossed_in + tracker.crossed_out}")
    print(f"Trocas de ID (ID Switches) Totais   : {tracker.id_switches}")
    print(f"Taxa de ID Switches por Minuto      : {taxa_id_switches_min:.2f} switches/min")
    print("=" * 75)

    # 4. Discussão Técnica e Considerações Éticas para Drones de Vigilância Urbana
    discussao_etica = (
        "\nDISCUSSÃO TÉCNICA E CONSIDERAÇÕES ÉTICAS (VIGILÂNCIA COM DRONES):\n"
        "-----------------------------------------------------------------\n"
        "1. Aplicação em Drones de Monitoramento Urbano:\n"
        "   Sistemas de visão computacional embarcados em drones permitem mapear densidade\n"
        "   de pedestres em eventos públicos, monitorar fluxo de tráfego em tempo real e\n"
        "   planejar rotas de evacuação em situações de emergência sem necessidade de sensores fixos.\n"
        "\n"
        "2. Impactos na Privacidade e Legislação (LGPD / GDPR):\n"
        "   O rastreamento aéreo contínuo gera preocupações críticas de privacidade. Mesmo sem\n"
        "   reconhecimento facial direto, a combinação de trilhas de deslocamento (trajetórias)\n"
        "   com localização espaço-temporal permite re-identificar cidadãos e rastrear rotas privadas.\n"
        "   A conformidade exige anonimização na borda (Edge Computing) transmitindo apenas contagens\n"
        "   e vetores de fluxo agregados, sem gravar ou transmitir imagens individuais de rostos.\n"
        "\n"
        "3. Riscos de Viés Algorítmico e Policiamento Preditivo:\n"
        "   Erros de detecção (falsos positivos/negativos decorrentes de oclusão, variações de luz\n"
        "   ou vestimenta) podem levar a alarmes falsos em operações policiais autônomas.\n"
        "   Decisões críticas de segurança pública nunca devem ser automatizadas sem supervisão\n"
        "   humana rigorosa (Human-in-the-Loop)."
    )
    print(discussao_etica)

    # 5. Salvar painel de demonstração
    fig, axs = plt.subplots(1, len(snapshots), figsize=(18, 5))
    for i, (fc, img_s) in enumerate(snapshots):
        axs[i].imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"Frame {fc} — Trilhas e Contagem", fontsize=11, fontweight="bold")
        axs[i].axis("off")

    plt.suptitle("Exercício 3B: Rastreamento Multi-Objetos por IoU com IDs Persistentes e Linha Virtual", fontsize=13, fontweight="bold")
    salvar_figura(SAIDAS_DIR / "at3b_rastreamento_ids.png", dpi=200)

    # 6. Salvar também o vídeo do rastreamento
    caminho_video_track = SAIDAS_DIR / "at3b_rastreamento_completo.mp4"
    h_v, w_v = frames_anotados[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(caminho_video_track), fourcc, 15, (w_v, h_v))
    for f in frames_anotados:
        vw.write(f)
    vw.release()
    print(f"[+] Vídeo de rastreamento com trilhas salvo em: {caminho_video_track.name}")


if __name__ == "__main__":
    main()
