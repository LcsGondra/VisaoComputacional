"""
Módulo de Utilitários Compartilhados — Assessment Test (AT) de Visão Computacional

Fornece infraestrutura unificada para:
- Gerenciamento de diretórios e salvamento padronizado de figuras (dpi=200).
- Download e cache local seguro de modelos de Deep Learning (SqueezeNet, YOLO, SSD, FCN).
- Geração geométrica de imagens calibradas de xadrez (7x6 cantos internos) em ângulos 3D variados.
- Geração e disponibilização de datasets de classificação, cenas urbanas e frames do pipeline integrativo.
- Algoritmo de rastreamento de múltiplos objetos por IoU com IDs persistentes, trilhas e contagem.
"""

from collections import deque
from pathlib import Path
import csv
import math
import os
import time
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Definição de diretórios estruturados
BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
SAIDAS_DIR = DADOS_DIR / "saidas"
MODELOS_DIR = BASE_DIR / "modelos"
CALIB_DIR = DADOS_DIR / "calibracao"
CLASS_DIR = DADOS_DIR / "classificacao"
TESTE_DIR = DADOS_DIR / "teste"
CALIB_FILE = DADOS_DIR / "calibracao_camera.npz"

# Paleta de cores e rótulos
COCO_CLASSES_DETECCAO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
PALETA_CORES = {
    "person": (255, 128, 0),      # Laranja em BGR
    "car": (0, 200, 255),         # Amarelo/Dourado
    "bicycle": (255, 0, 180),     # Magenta
    "bus": (0, 90, 255),          # Laranja escuro
    "truck": (0, 255, 120),       # Verde claro
    "padrao": (0, 255, 255),      # Amarelo
}


def ensure_dirs():
    """Garante a existência de todos os diretórios do projeto."""
    for p in (DADOS_DIR, SAIDAS_DIR, MODELOS_DIR, CALIB_DIR, CLASS_DIR, TESTE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def salvar_figura(caminho, dpi=200, mostrar=False, fechar=True):
    """Salva a figura atual do matplotlib com layout ajustado e alta resolução."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(caminho, dpi=dpi, bbox_inches="tight")
    print(f"[+] Gráfico salvo com sucesso em: {caminho.name}")
    if mostrar:
        try:
            plt.show()
        except Exception:
            pass
    if fechar:
        plt.close()


def baixar_arquivo_se_necessario(caminho_local, url, descricao="arquivo"):
    """Faz download de arquivo da web com headers apropriados se ainda não existir localmente."""
    caminho_local = Path(caminho_local)
    caminho_local.parent.mkdir(parents=True, exist_ok=True)

    if caminho_local.exists() and caminho_local.stat().st_size > 500:
        return caminho_local

    print(f"[+] Baixando {descricao}: {caminho_local.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(caminho_local, "wb") as f:
            tamanho = resp.length or 0
            baixado = 0
            bloco = 1024 * 128
            while True:
                buffer = resp.read(bloco)
                if not buffer:
                    break
                baixado += len(buffer)
                f.write(buffer)
        print(f"    [OK] Download concluído ({caminho_local.stat().st_size / (1024*1024):.2f} MB)")
        return caminho_local
    except Exception as e:
        if caminho_local.exists():
            caminho_local.unlink()
        raise RuntimeError(f"Falha ao baixar {descricao} de {url}: {e}")


def obter_labels_imagenet():
    """Retorna lista de rótulos do ImageNet (1000 classes)."""
    ensure_dirs()
    caminho = MODELOS_DIR / "imagenet_labels.txt"
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    baixar_arquivo_se_necessario(caminho, url, "labels ImageNet")
    with open(caminho, "r", encoding="utf-8") as f:
        labels = [linha.strip() for linha in f if linha.strip()]
    return labels


def obter_modelo_squeezenet():
    """Baixa e carrega o modelo SqueezeNet v1.1 Caffe pré-treinado no ImageNet."""
    ensure_dirs()
    proto = MODELOS_DIR / "squeezenet_v1.1.prototxt"
    caffemodel = MODELOS_DIR / "squeezenet_v1.1.caffemodel"

    url_proto = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/squeezenet_v1.1.prototxt"
    url_caffemodel = "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"

    baixar_arquivo_se_necessario(proto, url_proto, "prototxt SqueezeNet")
    baixar_arquivo_se_necessario(caffemodel, url_caffemodel, "caffemodel SqueezeNet (~4.9MB)")

    net = cv2.dnn.readNetFromCaffe(str(proto), str(caffemodel))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, proto, caffemodel


def obter_modelo_yolo_tiny():
    """Baixa e carrega o YOLOv4-tiny via OpenCV DNN (Darknet)."""
    ensure_dirs()
    cfg = MODELOS_DIR / "yolov4-tiny.cfg"
    weights = MODELOS_DIR / "yolov4-tiny.weights"
    names = MODELOS_DIR / "coco.names"

    url_cfg = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
    url_weights = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
    url_names = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

    baixar_arquivo_se_necessario(cfg, url_cfg, "configuração YOLOv4-tiny")
    baixar_arquivo_se_necessario(names, url_names, "classes COCO")
    baixar_arquivo_se_necessario(weights, url_weights, "pesos YOLOv4-tiny (~24MB)")

    with open(names, "r", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]

    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, classes, cfg, weights


def obter_modelo_ssd_mobilenet():
    """Baixa e carrega o SSD MobileNet v2 via OpenCV DNN (TensorFlow)."""
    ensure_dirs()
    pbtxt = MODELOS_DIR / "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    pb = MODELOS_DIR / "ssd_mobilenet_v2_coco_2018_03_29.pb"

    url_pbtxt = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    baixar_arquivo_se_necessario(pbtxt, url_pbtxt, "pbtxt SSD MobileNet v2")

    if not pb.exists() or pb.stat().st_size < 1000:
        tar_path = MODELOS_DIR / "ssd_mobilenet_v2.tar.gz"
        url_tar = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
        baixar_arquivo_se_necessario(tar_path, url_tar, "tar.gz SSD MobileNet (~69MB)")

        import tarfile
        print("[+] Extraindo frozen_inference_graph.pb do arquivo tar...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("frozen_inference_graph.pb"):
                    member.name = pb.name
                    tar.extract(member, path=MODELOS_DIR)
                    break
        if tar_path.exists():
            tar_path.unlink()

    net = cv2.dnn.readNetFromTensorflow(str(pb), str(pbtxt))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, pb, pbtxt


def obter_modelo_fcn_segmentacao():
    """Baixa e carrega o modelo FCN-ResNet50 ONNX para segmentação semântica."""
    ensure_dirs()
    onnx_path = MODELOS_DIR / "fcn-resnet50-12.onnx"
    url_onnx = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx"

    baixar_arquivo_se_necessario(onnx_path, url_onnx, "FCN-ResNet50 ONNX (~130MB)")

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, onnx_path


# ==============================================================================
# GERAÇÃO DE DADOS SINTÉTICOS / CIENTÍFICOS PARA CALIBRAÇÃO E PIPELINE
# ==============================================================================

def gerar_dataset_calibracao(num_imagens=18, pattern_size=(7, 6), square_size=30, img_size=(640, 480)):
    """
    Gera conjunto de imagens calibradas de um tabuleiro de xadrez em 3D.
    Simula variações realistas de rotação (pitch, yaw, roll), translação (distância e deslocamento)
    e distorção de lente (k1, k2) para calibração com ground truth geométrico exato.
    """
    ensure_dirs()
    existentes = sorted(list(CALIB_DIR.glob("calib_*.png")))
    if len(existentes) >= num_imagens:
        return existentes

    cols, rows = pattern_size
    w_px = (cols + 1) * square_size
    h_px = (rows + 1) * square_size

    # Cria textura básica do tabuleiro
    tabuleiro = np.zeros((h_px, w_px), dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                tabuleiro[r * square_size:(r + 1) * square_size, c * square_size:(c + 1) * square_size] = 255

    # Matriz intrínseca sintética de câmera de referência
    fx = fy = 600.0
    cx, cy = img_size[0] / 2.0, img_size[1] / 2.0
    K_ref = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_ref = np.array([-0.18, 0.06, 0.001, -0.001, 0.0], dtype=np.float64)

    # Vértices 3D do plano do tabuleiro
    corners_3d = np.array([
        [-w_px / 2, -h_px / 2, 0],
        [w_px / 2, -h_px / 2, 0],
        [w_px / 2, h_px / 2, 0],
        [-w_px / 2, h_px / 2, 0],
    ], dtype=np.float64)

    caminhos = []
    np.random.seed(42)

    for i in range(num_imagens):
        # Variações controladas de ângulo e profundidade
        pitch = np.radians(np.random.uniform(-25, 25))
        yaw = np.radians(np.random.uniform(-28, 28))
        roll = np.radians(np.random.uniform(-18, 18))
        tz = np.random.uniform(550, 780)
        tx = np.random.uniform(-110, 110)
        ty = np.random.uniform(-80, 80)

        # Matrizes de rotação 3D
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        rvec, _ = cv2.Rodrigues(R)
        tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

        # Projeta os 4 cantos do plano usando a câmera com distorção
        pts_2d, _ = cv2.projectPoints(corners_3d, rvec, tvec, K_ref, dist_ref)
        pts_2d = pts_2d.reshape(-1, 2).astype(np.float32)

        src_pts = np.array([[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, pts_2d)

        # Fundo de bancada com textura suave
        fundo = np.full((img_size[1], img_size[0]), int(np.random.uniform(180, 220)), dtype=np.uint8)
        # Adiciona leve ruído gaussiano para realismo sensorial
        ruido = np.random.normal(0, 4, fundo.shape).astype(np.int16)
        fundo = np.clip(fundo.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

        warped = cv2.warpPerspective(tabuleiro, H, img_size, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpPerspective(np.ones_like(tabuleiro) * 255, H, img_size, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        frame = np.where(mask > 128, warped, fundo)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        caminho = CALIB_DIR / f"calib_{i+1:02d}.png"
        cv2.imwrite(str(caminho), frame_bgr)
        caminhos.append(caminho)

    print(f"[+] {len(caminhos)} imagens de calibração geradas em: {CALIB_DIR.name}/")
    return caminhos


def obter_dataset_classificacao(num_imagens=10):
    """
    Gera/retorna 10 imagens de categorias distintas para teste de classificação OpenCV DNN.
    """
    ensure_dirs()
    existentes = sorted(list(CLASS_DIR.glob("img_*.png")))
    if len(existentes) >= num_imagens:
        return existentes

    categorias = [
        ("caneca", (255, 100, 100), "Caneca de cafe ceramica"),
        ("bola", (60, 220, 60), "Bola esportiva de tenis"),
        ("carro", (50, 100, 240), "Automovel sedan urbano"),
        ("livro", (200, 60, 180), "Livro de engenharia capa dura"),
        ("garrafa", (240, 200, 50), "Garrafa de agua mineral"),
        ("teclado", (120, 120, 120), "Teclado de computador mecanico"),
        ("robo", (230, 140, 40), "Robo movel autonomo rover"),
        ("caixa", (180, 130, 90), "Caixa de papelao embalagem"),
        ("capacete", (40, 200, 200), "Capacete de seguranca EPI"),
        ("planta", (40, 160, 40), "Vaso de planta ornamental"),
    ]

    caminhos = []
    w, h = 480, 360
    for i, (nome, cor, desc) in enumerate(categorias[:num_imagens]):
        img = np.full((h, w, 3), (240, 240, 245), dtype=np.uint8)
        # Fundo e sombras
        cv2.ellipse(img, (240, 280), (140, 30), 0, 0, 360, (200, 200, 205), -1)

        # Geometria simbólica detalhada com texturas
        if nome == "caneca":
            cv2.rectangle(img, (180, 130), (280, 270), cor, -1)
            cv2.ellipse(img, (230, 130), (50, 18), 0, 0, 360, (255, 160, 160), -1)
            cv2.ellipse(img, (290, 200), (28, 45), 0, 0, 360, cor, 12)
        elif nome == "bola":
            cv2.circle(img, (240, 200), 75, cor, -1)
            cv2.circle(img, (215, 175), 18, (255, 255, 255), -1)
            cv2.ellipse(img, (240, 200), (74, 30), 45, 0, 360, (40, 160, 40), 4)
        elif nome == "carro":
            cv2.rectangle(img, (120, 180), (360, 260), cor, -1)
            cv2.rectangle(img, (170, 125), (310, 180), cor, -1)
            cv2.circle(img, (170, 260), 28, (30, 30, 30), -1)
            cv2.circle(img, (310, 260), 28, (30, 30, 30), -1)
            cv2.circle(img, (170, 260), 12, (180, 180, 180), -1)
            cv2.circle(img, (310, 260), 12, (180, 180, 180), -1)
        else:
            cv2.rectangle(img, (150, 120), (330, 260), cor, -1)
            cv2.rectangle(img, (170, 140), (310, 240), (255, 255, 255), 2)
            cv2.circle(img, (240, 190), 30, (cor[0]//2, cor[1]//2, cor[2]//2), -1)

        # Anotação descritiva
        cv2.putText(img, f"Objeto: {nome.upper()}", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
        cv2.putText(img, desc, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        caminho = CLASS_DIR / f"img_{i+1:02d}_{nome}.png"
        cv2.imwrite(str(caminho), img)
        caminhos.append(caminho)

    return caminhos


def obter_frame_pipeline():
    """
    Cria frame completo para validação do pipeline integrativo sequencial (Ex 2B):
    Contém região vermelha para HSV, textura complexa para ORB, pedestre para HOG e rótulo.
    """
    ensure_dirs()
    caminho = TESTE_DIR / "frame_pipeline_integrado.png"
    if caminho.exists():
        return cv2.imread(str(caminho)), caminho

    w, h = 640, 480
    frame = np.full((h, w, 3), (230, 235, 240), dtype=np.uint8)

    # 1. Pista e calçada
    cv2.rectangle(frame, (0, 300), (w, h), (90, 90, 95), -1)
    cv2.rectangle(frame, (0, 240), (w, 300), (140, 140, 145), -1)

    # 2. ROI Vermelha marcante para segmentação HSV (TP1)
    cv2.rectangle(frame, (80, 110), (230, 250), (30, 30, 220), -1)
    cv2.rectangle(frame, (95, 125), (215, 235), (255, 255, 255), 2)

    # 3. Textura com detalhes ricos para extração de keypoints ORB (TP2)
    for x in range(105, 210, 20):
        for y in range(135, 225, 20):
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)
            cv2.line(frame, (x - 6, y), (x + 6, y), (255, 255, 0), 1)

    # 4. Silhueta de pedestre para detecção HOG+SVM / Haar (TP3)
    px, py = 450, 260
    # Cabeça
    cv2.circle(frame, (px, py - 60), 18, (40, 40, 40), -1)
    # Tronco
    cv2.rectangle(frame, (px - 16, py - 40), (px + 16, py + 10), (40, 40, 40), -1)
    # Pernas
    cv2.line(frame, (px - 10, py + 10), (px - 12, py + 60), (40, 40, 40), 6)
    cv2.line(frame, (px + 10, py + 10), (px + 12, py + 60), (40, 40, 40), 6)

    cv2.putText(frame, "Cena de Teste Integrada: HSV + ORB + HOG + DNN", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.imwrite(str(caminho), frame)
    return frame, caminho


def gerar_video_transito(caminho_video, n_frames=90, fps=20, size=(800, 450)):
    """
    Gera um vídeo urbano com pedestres, ciclistas e veículos em movimento contínuo
    para benchmark e rastreamento em tempo real (Ex 3A e 3B).
    """
    caminho_video = Path(caminho_video)
    caminho_video.parent.mkdir(parents=True, exist_ok=True)
    if caminho_video.exists() and caminho_video.stat().st_size > 5000:
        return str(caminho_video)

    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(caminho_video), fourcc, fps, (w, h))

    for f in range(n_frames):
        frame = np.full((h, w, 3), (220, 225, 230), dtype=np.uint8)

        # Cenário urbano (rua, calçada, prédios)
        cv2.rectangle(frame, (0, 0), (w, 140), (190, 170, 150), -1)  # Prédios
        cv2.rectangle(frame, (0, 140), (w, 200), (130, 130, 135), -1)  # Calçada superior
        cv2.rectangle(frame, (0, 200), (w, 400), (60, 60, 65), -1)    # Pista de rolamento
        cv2.rectangle(frame, (0, 400), (w, h), (130, 130, 135), -1)   # Calçada inferior

        # Faixas de trânsito
        for x in range(0, w, 80):
            cv2.line(frame, (x, 300), (x + 40, 300), (255, 255, 255), 3)

        # Pedestre 1 (caminhando para a direita na calçada superior)
        x_ped1 = int(50 + f * 5.0) % (w + 60) - 30
        y_ped1 = 175
        cv2.circle(frame, (x_ped1, y_ped1 - 25), 10, PALETA_CORES["person"], -1)
        cv2.rectangle(frame, (x_ped1 - 10, y_ped1 - 15), (x_ped1 + 10, y_ped1 + 15), PALETA_CORES["person"], -1)
        cv2.line(frame, (x_ped1 - 5, y_ped1 + 15), (x_ped1 - 8, y_ped1 + 35), PALETA_CORES["person"], 4)
        cv2.line(frame, (x_ped1 + 5, y_ped1 + 15), (x_ped1 + 8, y_ped1 + 35), PALETA_CORES["person"], 4)

        # Carro 1 (trafegando para a esquerda na faixa 1)
        x_car1 = int(w + 100 - f * 8.5) % (w + 200) - 100
        y_car1 = 250
        cv2.rectangle(frame, (x_car1 - 60, y_car1 - 22), (x_car1 + 60, y_car1 + 22), PALETA_CORES["car"], -1)
        cv2.rectangle(frame, (x_car1 - 35, y_car1 - 40), (x_car1 + 25, y_car1 - 22), (240, 220, 180), -1)
        cv2.circle(frame, (x_car1 - 40, y_car1 + 22), 12, (20, 20, 20), -1)
        cv2.circle(frame, (x_car1 + 40, y_car1 + 22), 12, (20, 20, 20), -1)

        # Carro 2 (trafegando para a direita na faixa 2)
        x_car2 = int(-50 + f * 9.0) % (w + 220) - 100
        y_car2 = 350
        cv2.rectangle(frame, (x_car2 - 65, y_car2 - 25), (x_car2 + 65, y_car2 + 25), (40, 60, 220), -1)
        cv2.rectangle(frame, (x_car2 - 40, y_car2 - 45), (x_car2 + 30, y_car2 - 25), (200, 220, 255), -1)
        cv2.circle(frame, (x_car2 - 45, y_car2 + 25), 14, (20, 20, 20), -1)
        cv2.circle(frame, (x_car2 + 45, y_car2 + 25), 14, (20, 20, 20), -1)

        # Ciclista (caminhando na diagonal ou calçada)
        x_bike = int(w + 50 - f * 4.5) % (w + 120) - 40
        y_bike = 415
        cv2.circle(frame, (x_bike - 15, y_bike + 10), 12, (10, 10, 10), 2)
        cv2.circle(frame, (x_bike + 15, y_bike + 10), 12, (10, 10, 10), 2)
        cv2.circle(frame, (x_bike, y_bike - 20), 8, PALETA_CORES["bicycle"], -1)

        writer.write(frame)

    writer.release()
    print(f"[+] Vídeo de trânsito urbano sintetizado em: {caminho_video.name}")
    return str(caminho_video)


# ==============================================================================
# RASTREADOR DE OBJETOS COM PERSISTÊNCIA POR IOU (EXERCÍCIO 3B)
# ==============================================================================

def calcular_iou(box_a, box_b):
    """Calcula Intersection over Union (IoU) entre duas caixas no formato [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return 0.0 if union <= 0 else inter / union


class IoUTracker:
    """
    Rastreador de objetos com associação por IoU (Intersection over Union).
    Gerencia identificadores únicos (IDs persistentes), mantém trilha dos últimos 30 frames,
    detecta cruzamentos de linha virtual (entradas e saídas) e contabiliza trocas de ID (ID switches).
    """
    def __init__(self, iou_thresh=0.30, max_missing=10, trail_len=30, line_x=400):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.trail_len = trail_len
        self.line_x = line_x

        self.next_id = 1
        self.tracks = {}      # id -> {'bbox', 'cls', 'conf', 'missing', 'last_x'}
        self.trails = {}      # id -> deque de (cx, cy)
        self.crossed_in = 0   # Da esquerda para a direita
        self.crossed_out = 0  # Da direita para a esquerda
        self.id_switches = 0
        self.total_tracks_criadas = 0

    def update(self, detections):
        """
        Recebe lista de detecções: [{'bbox': [x1, y1, x2, y2], 'cls': 'car', 'conf': 0.85}, ...]
        Retorna lista de objetos rastreados anotados com ID persistente.
        """
        assigned_tracks = set()
        assigned_dets = set()
        active_ids = list(self.tracks.keys())

        # Matriz de afinidade por IoU
        pares = []
        for tid in active_ids:
            for di, det in enumerate(detections):
                score = calcular_iou(self.tracks[tid]["bbox"], det["bbox"])
                pares.append((score, tid, di))
        pares.sort(reverse=True, key=lambda x: x[0])

        for score, tid, di in pares:
            if score < self.iou_thresh:
                continue
            if tid in assigned_tracks or di in assigned_dets:
                continue

            # Se a classe mudar abruptamente entre associações com bom IoU, detecta troca de ID
            if self.tracks[tid]["cls"] != detections[di]["cls"]:
                self.id_switches += 1

            old_cx = (self.tracks[tid]["bbox"][0] + self.tracks[tid]["bbox"][2]) / 2.0
            new_cx = (detections[di]["bbox"][0] + detections[di]["bbox"][2]) / 2.0

            # Atualiza estado do objeto rastreado
            self.tracks[tid].update({
                "bbox": detections[di]["bbox"],
                "cls": detections[di]["cls"],
                "conf": detections[di]["conf"],
                "missing": 0,
            })

            # Verifica cruzamento da linha virtual
            if old_cx < self.line_x <= new_cx:
                self.crossed_in += 1
            elif old_cx > self.line_x >= new_cx:
                self.crossed_out += 1

            # Atualiza histórico de coordenadas da trilha
            cy = (detections[di]["bbox"][1] + detections[di]["bbox"][3]) / 2.0
            self.trails[tid].append((int(new_cx), int(cy)))

            assigned_tracks.add(tid)
            assigned_dets.add(di)

        # Registra novas detecções como novos tracks com IDs únicos
        for di, det in enumerate(detections):
            if di not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.total_tracks_criadas += 1

                cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                cy = (det["bbox"][1] + det["bbox"][3]) / 2.0

                self.tracks[tid] = {
                    "bbox": det["bbox"],
                    "cls": det["cls"],
                    "conf": det["conf"],
                    "missing": 0,
                }
                self.trails[tid] = deque(maxlen=self.trail_len)
                self.trails[tid].append((int(cx), int(cy)))
                assigned_tracks.add(tid)

        # Incrementa contador de frames perdidos e remove tracks inativos
        to_delete = []
        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    to_delete.append(tid)

        for tid in to_delete:
            self.tracks.pop(tid, None)

        # Monta resultado final
        resultados = []
        for tid, tr in self.tracks.items():
            resultados.append({
                "id": tid,
                "bbox": tr["bbox"],
                "cls": tr["cls"],
                "conf": tr["conf"],
                "trail": list(self.trails.get(tid, [])),
            })
        return resultados


class Timer:
    """Cronômetro utilitário para medição de latência em milissegundos."""
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self):
        return (time.perf_counter() - self.t0) * 1000.0

    def reset(self):
        self.t0 = time.perf_counter()


COCO_CLASSES_SSD = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard",
    77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
    88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def detectar_yolo_tiny(net, frame, classes, score_thresh=0.25, nms_thresh=0.40):
    """Executa detecção YOLOv4-tiny com decodificação multiescala e NMS."""
    h_orig, w_orig = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.perf_counter()
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    latencia_ms = (time.perf_counter() - t0) * 1000.0

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > score_thresh:
                cx = int(detection[0] * w_orig)
                cy = int(detection[1] * h_orig)
                bw = int(detection[2] * w_orig)
                bh = int(detection[3] * h_orig)
                bx = int(cx - bw / 2)
                by = int(cy - bh / 2)

                boxes.append([bx, by, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, nms_thresh)
    deteccoes_finais = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            bx, by, bw, bh = boxes[i]
            cid = class_ids[i]
            lbl = classes[cid] if cid < len(classes) else f"obj_{cid}"
            deteccoes_finais.append({
                "bbox": [bx, by, bx + bw, by + bh],
                "cls": lbl,
                "conf": confidences[i],
            })

    return deteccoes_finais, latencia_ms


def detectar_ssd_mobilenet(net, frame, score_thresh=0.25, nms_thresh=0.40):
    """Executa detecção SSD MobileNet v2 com decodificação e NMS."""
    h_orig, w_orig = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)

    t0 = time.perf_counter()
    out = net.forward()
    latencia_ms = (time.perf_counter() - t0) * 1000.0

    boxes, confidences, labels_list = [], [], []
    detections = out[0, 0]

    for det in detections:
        confidence = float(det[2])
        if confidence > score_thresh:
            cid = int(det[1])
            lbl = COCO_CLASSES_SSD.get(cid, f"obj_{cid}")

            x1 = int(det[3] * w_orig)
            y1 = int(det[4] * h_orig)
            x2 = int(det[5] * w_orig)
            y2 = int(det[6] * h_orig)
            bw = max(0, x2 - x1)
            bh = max(0, y2 - y1)

            boxes.append([x1, y1, bw, bh])
            confidences.append(confidence)
            labels_list.append(lbl)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_thresh, nms_thresh)
    deteccoes_finais = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            x, y, w, h = boxes[i]
            deteccoes_finais.append({
                "bbox": [x, y, x + w, y + h],
                "cls": labels_list[i],
                "conf": confidences[i],
            })

    return deteccoes_finais, latencia_ms
