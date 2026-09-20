"""
Exercício 4 — Item A: Segmentação Semântica com FCN-ResNet50 e Comparativo com HSV
Competências: 4.4 e todas

Este script implementa segmentação semântica pixel a pixel utilizando o modelo profundo
FCN-ResNet50 (Fully Convolutional Network com backbone ResNet-50 pré-treinado no Pascal VOC)
via OpenCV DNN, processa 5 imagens de cenas externas urbanas (rua, calçada, parque, cruzamento, rodovia),
e realiza uma análise comparativa rigorosa com a segmentação por cor HSV estudada no TP1.

Etapas Executadas:
------------------
1. Carregamento do modelo FCN-ResNet50 ONNX via OpenCV DNN.
2. Processamento de 5 cenas externas:
   - Geração do mapa semântico colorido por categoria (21 classes VOC).
   - Sobreposição de máscara semitransparente (cv2.addWeighted, alpha=0.50).
   - Cálculo e impressão da porcentagem de área pixel a pixel ocupada por cada classe detectada.
3. Segmentação cromática por cor no espaço HSV (técnica do TP1) para as mesmas imagens.
4. Painel comparativo visual lado a lado (Original | Segmentação Semântica FCN | Segmentação HSV TP1).
5. Discussão técnica fundamentada sobre o papel de cada abordagem em veículos autônomos.
"""

from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    obter_modelo_fcn_segmentacao,
    salvar_figura,
    DADOS_DIR,
    SAIDAS_DIR,
    TESTE_DIR,
)

# 21 Classes do Pascal VOC
VOC_CLASSES = [
    "fundo/estrada", "aviao", "bicicleta", "passaro", "barco",
    "garrafa", "onibus", "carro", "gato", "cadeira",
    "vaca", "mesa", "cachorro", "cavalo", "moto",
    "pedestre", "arvore/planta", "ovelha", "sofa", "trem", "monitor"
]

# Paleta de 21 cores distintas (RGB) para visualização do mapa semântico
VOC_COLOR_MAP = np.array([
    [0, 0, 0],         # 0: fundo/estrada (preto/cinza)
    [128, 0, 0],       # 1: avião
    [0, 128, 0],       # 2: bicicleta
    [128, 128, 0],     # 3: pássaro
    [0, 0, 128],       # 4: barco
    [128, 0, 128],     # 5: garrafa
    [0, 128, 128],     # 6: ônibus
    [128, 128, 128],   # 7: carro (cinza metálico/azul)
    [64, 0, 0],        # 8: gato
    [192, 0, 0],       # 9: cadeira
    [64, 128, 0],      # 10: vaca
    [192, 128, 0],     # 11: mesa
    [64, 0, 128],      # 12: cachorro
    [192, 0, 128],     # 13: cavalo
    [64, 128, 128],    # 14: moto
    [192, 128, 128],   # 15: pedestre (rosa/vermelho claro)
    [0, 64, 0],        # 16: planta/árvore (verde escuro)
    [128, 64, 0],      # 17: ovelha
    [0, 192, 0],       # 18: sofá
    [128, 192, 0],     # 19: trem
    [0, 64, 128],      # 20: monitor
], dtype=np.uint8)


def gerar_cenas_externas(num_cenas=5):
    """
    Gera/retorna 5 imagens de cenas externas urbanas com objetos típicos de trânsito:
    pista, calçadas, veículos, pedestres e vegetação.
    """
    ensure_dirs()
    caminhos = []
    w, h = 480, 360

    descricoes = [
        ("cena_01_rua.png", "Rua Urbana com Carro e Pista", "carro"),
        ("cena_02_calcada.png", "Calçada Urbana com Pedestre", "pedestre"),
        ("cena_03_parque.png", "Parque com Ciclista e Árvores", "bicicleta"),
        ("cena_04_cruzamento.png", "Cruzamento com Ônibus Urbano", "onibus"),
        ("cena_05_rodovia.png", "Rodovia com Veículo e Faixas", "carro"),
    ]

    for nome_arq, titulo, elemento in descricoes[:num_cenas]:
        p = TESTE_DIR / nome_arq
        if p.exists():
            caminhos.append(p)
            continue

        img = np.full((h, w, 3), (220, 225, 230), dtype=np.uint8)

        # Céu
        cv2.rectangle(img, (0, 0), (w, 130), (210, 180, 140), -1)

        # Vegetação no horizonte
        for x in range(0, w, 40):
            cv2.circle(img, (x, 130), 30, (40, 120, 40), -1)

        # Calçada
        cv2.rectangle(img, (0, 130), (w, 180), (160, 160, 165), -1)

        # Pista asfáltica
        cv2.rectangle(img, (0, 180), (w, h), (65, 65, 70), -1)

        # Faixas centrais brancas
        for x in range(20, w, 90):
            cv2.line(img, (x, 270), (x + 50, 270), (240, 240, 240), 3)

        # Elemento específico da cena
        if elemento == "carro":
            cx, cy = 240, 250
            # Sombra
            cv2.ellipse(img, (cx, cy + 30), (80, 18), 0, 0, 360, (30, 30, 30), -1)
            # Chassi
            cv2.rectangle(img, (cx - 75, cy - 10), (cx + 75, cy + 30), (30, 50, 200), -1)
            # Cabine
            cv2.rectangle(img, (cx - 45, cy - 35), (cx + 35, cy - 10), (210, 210, 230), -1)
            # Rodas
            cv2.circle(img, (cx - 50, cy + 30), 16, (20, 20, 20), -1)
            cv2.circle(img, (cx + 50, cy + 30), 16, (20, 20, 20), -1)
        elif elemento == "pedestre":
            px, py = 200, 170
            cv2.circle(img, (px, py - 35), 12, (180, 160, 140), -1)
            cv2.rectangle(img, (px - 10, py - 20), (px + 10, py + 15), (200, 50, 30), -1)
            cv2.line(img, (px - 5, py + 15), (px - 6, py + 50), (40, 40, 40), 5)
            cv2.line(img, (px + 5, py + 15), (px + 6, py + 50), (40, 40, 40), 5)
        elif elemento == "bicicleta":
            bx, by = 240, 220
            cv2.circle(img, (bx - 30, by + 20), 18, (15, 15, 15), 3)
            cv2.circle(img, (bx + 30, by + 20), 18, (15, 15, 15), 3)
            cv2.line(img, (bx - 30, by + 20), (bx, by), (0, 180, 220), 4)
            cv2.line(img, (bx, by), (bx + 30, by + 20), (0, 180, 220), 4)
            cv2.circle(img, (bx, by - 25), 10, (180, 150, 130), -1)
        elif elemento == "onibus":
            ox, oy = 240, 230
            cv2.rectangle(img, (ox - 110, oy - 50), (ox + 110, oy + 35), (0, 140, 255), -1)
            for wx in range(ox - 90, ox + 90, 35):
                cv2.rectangle(img, (wx, oy - 40), (wx + 25, oy - 15), (220, 240, 255), -1)
            cv2.circle(img, (ox - 70, oy + 35), 18, (20, 20, 20), -1)
            cv2.circle(img, (ox + 70, oy + 35), 18, (20, 20, 20), -1)

        cv2.putText(img, titulo, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
        cv2.imwrite(str(p), img)
        caminhos.append(p)

    return caminhos


def segmentar_semantica_fcn(net, imagem_bgr, input_size=(256, 256)):
    """
    Executa segmentação semântica com FCN-ResNet50:
    Retorna mapa de classes por pixel [H, W], máscara RGB colorida e estatísticas de área.
    """
    h_orig, w_orig = imagem_bgr.shape[:2]

    # Prepara o blob normalizado
    blob = cv2.dnn.blobFromImage(
        imagem_bgr,
        scalefactor=1.0 / 255.0,
        size=input_size,
        mean=(0.485, 0.456, 0.406),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)

    t0 = time.perf_counter()
    out = net.forward()
    latencia_ms = (time.perf_counter() - t0) * 1000.0

    # out tem shape [1, 21, H_in, W_in]
    class_map_small = np.argmax(out[0], axis=0).astype(np.uint8)
    class_map = cv2.resize(class_map_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Gera mapa de cores RGB
    color_mask = VOC_COLOR_MAP[class_map]

    # Calcula estatísticas percentuais de área
    total_pixels = float(h_orig * w_orig)
    estatisticas = []
    for cls_id in np.unique(class_map):
        count = np.count_nonzero(class_map == cls_id)
        pct = (count / total_pixels) * 100.0
        nome_classe = VOC_CLASSES[cls_id] if cls_id < len(VOC_CLASSES) else f"classe_{cls_id}"
        estatisticas.append((cls_id, nome_classe, pct, count))

    estatisticas.sort(key=lambda x: x[2], reverse=True)
    return class_map, color_mask, estatisticas, latencia_ms


def segmentar_cor_hsv(imagem_bgr):
    """
    Segmentação cromática clássica por espaço HSV (técnica do TP1):
    Segmenta regiões correspondentes a pista/veículo/obstáculo com base em limiares de matiz e saturação.
    """
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    # Segmenta elementos quentes/vermelhos e elementos escuros de asfalto
    m1 = cv2.inRange(hsv, np.array([0, 60, 40]), np.array([15, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 60, 40]), np.array([180, 255, 255]))
    # Segmenta elementos azuis/frios
    m3 = cv2.inRange(hsv, np.array([95, 60, 40]), np.array([130, 255, 255]))
    mask_hsv = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return mask_hsv, lat_ms


def main():
    ensure_dirs()
    print("=" * 80)
    print("EXERCÍCIO 4A — SEGMENTAÇÃO SEMÂNTICA (FCN-RESNET50) vs. SEGMENTAÇÃO HSV (TP1)")
    print("=" * 80)

    # 1. Carrega modelo FCN-ResNet50
    net_fcn, onnx_p = obter_modelo_fcn_segmentacao()
    print(f"[+] Modelo FCN-ResNet50 carregado via OpenCV DNN ({onnx_p.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Obter as 5 imagens de cenas externas
    cenas_caminhos = gerar_cenas_externas(num_cenas=5)
    print(f"[+] {len(cenas_caminhos)} cenas externas carregadas para teste.")

    # 3. Processamento das 5 imagens
    resultados_painel = []

    print("\n" + "-" * 80)
    print("ESTATÍSTICAS DE SEGMENTAÇÃO SEMÂNTICA (% DE ÁREA POR CLASSE):")
    print("-" * 80)

    for idx, p in enumerate(cenas_caminhos, start=1):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue

        # Segmentação Semântica FCN
        class_map, color_mask, stats, lat_fcn = segmentar_semantica_fcn(net_fcn, img_bgr)

        # Sobreposição da máscara semitransparente (alpha = 0.45)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        sobreposicao = cv2.addWeighted(img_bgr, 0.55, color_mask_bgr, 0.45, 0)

        # Segmentação por Cor HSV (TP1)
        mask_hsv, lat_hsv = segmentar_cor_hsv(img_bgr)
        mask_hsv_bgr = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)

        # Imprime no terminal a porcentagem de área por classe
        print(f"\nCena {idx:02d} — {p.name} (Latência FCN: {lat_fcn:.1f} ms | HSV: {lat_hsv:.1f} ms):")
        for cid, nome_c, pct, pixels in stats:
            print(f"  Classe #{cid:02d} ({nome_c:<16}): {pct:6.2f}% da área ({pixels:6d} pixels)")

        resultados_painel.append({
            "nome": p.stem,
            "orig": img_bgr,
            "fcn": sobreposicao,
            "hsv": mask_hsv_bgr,
            "lat_fcn": lat_fcn,
            "lat_hsv": lat_hsv,
            "stats": stats,
        })

    print("-" * 80)

    # 4. Discussão Técnica sobre Vantagens e Limitações para Veículos Autônomos
    discussao_veiculos_autonomos = (
        "\nDISCUSSÃO TÉCNICA: SEGMENTAÇÃO SEMÂNTICA vs. SEGMENTAÇÃO HSV EM VEÍCULOS AUTÔNOMOS:\n"
        "===================================================================================\n"
        "1. Vantagens e Limitações da Segmentação Semântica Profunda (FCN / DeepLabV3):\n"
        "   - Vantagens: Compreensão conceitual de alto nível da cena. O modelo rotula 'o que é cada\n"
        "     pixel' independentemente de variações severas de iluminação, sombras projetadas, oclusões\n"
        "     parciais ou pinturas no solo. Permite distinguir categoricamente pista dirigível (drivable area),\n"
        "     obstáculos verticais (pedestres/veículos) e limites da via (calçadas e vegetação).\n"
        "   - Limitações: Elevadíssimo custo computacional e latência de processamento (~50 a 150 ms em CPU),\n"
        "     inviável para loops de controle de alta frequência (> 30 Hz) em hardwares embarcados sem GPU/NPU.\n"
        "     Vulnerável a domínios fora da distribuição de treinamento (chuva, neve, névoa densa).\n"
        "\n"
        "2. Vantagens e Limitações da Segmentação por Cor HSV (Técnica Clássica do TP1):\n"
        "   - Vantagens: Latência desprezível (< 2 ms) e consumo mínimo de energia (< 1W), rodando em tempo\n"
        "     real (> 100 FPS) até nos processadores mais simples (ARM Cortex-M / Raspberry Pi Zero).\n"
        "     Ideal para tarefas específicas com assinaturas de cor controladas (marcações viárias amarelas,\n"
        "     cones de sinalização laranjas, placas reflexivas e luzes de freio/semáforos).\n"
        "   - Limitações: Altamente suscetível a variações de luminância, sombras, reflexos e asfalto molhado.\n"
        "     Incapaz de atribuir significado semântico abstrato (uma lona azul no asfalto é confundida com carro).\n"
        "\n"
        "3. Arquitetura Híbrida Recomendada para Condução Autônoma:\n"
        "   Um veículo autônomo de produção adota uma abordagem hierárquica:\n"
        "   - Camada Reativa Rápida (100 Hz, HSV/Canny): detecção de faixas de rodagem e cones de perigo imediato.\n"
        "   - Camada Deliberativa (15-20 Hz, FCN/DeepLab/YOLO): segmentação semântica global de espaço livre e pedestres."
    )
    print(discussao_veiculos_autonomos)

    # 5. Salvar Painel Comparativo Lado a Lado (Original | FCN Semântica | HSV TP1)
    num_linhas = len(resultados_painel)
    fig, axs = plt.subplots(num_linhas, 3, figsize=(16, 3.2 * num_linhas))

    for i, res in enumerate(resultados_painel):
        # Coluna 1: Original
        axs[i, 0].imshow(cv2.cvtColor(res["orig"], cv2.COLOR_BGR2RGB))
        axs[i, 0].set_title(f"Cena {i+1}: {res['nome']} (Original)", fontsize=10, fontweight="bold")
        axs[i, 0].axis("off")

        # Coluna 2: FCN-ResNet50
        top_cls = res["stats"][0][1] if res["stats"] else "fundo"
        axs[i, 1].imshow(cv2.cvtColor(res["fcn"], cv2.COLOR_BGR2RGB))
        axs[i, 1].set_title(f"FCN-ResNet50 Semântica ({res['lat_fcn']:.1f} ms | Top: {top_cls})", fontsize=10, fontweight="bold", color="darkblue")
        axs[i, 1].axis("off")

        # Coluna 3: HSV TP1
        axs[i, 2].imshow(cv2.cvtColor(res["hsv"], cv2.COLOR_BGR2RGB))
        axs[i, 2].set_title(f"Segmentação HSV TP1 ({res['lat_hsv']:.1f} ms)", fontsize=10, fontweight="bold", color="darkgreen")
        axs[i, 2].axis("off")

    plt.suptitle("Exercício 4A: Comparativo Visual — Imagem Original vs. FCN-ResNet50 Semântica vs. HSV (TP1)", fontsize=13, fontweight="bold")
    salvar_figura(SAIDAS_DIR / "at4a_segmentacao_comparativo.png", dpi=200)


if __name__ == "__main__":
    main()
