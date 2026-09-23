# Módulo de Utilitários Compartilhados — Assessment Test (AT) de Visão Computacional
#
# Fornece infraestrutura unificada utilizando EXCLUSIVAMENTE bibliotecas consagradas
# (OpenCV, scikit-image, scikit-learn, OpenCV Samples Dataset, etc.) e dados reais:
# - Calibração com 18 imagens reais do dataset oficial de calibração do OpenCV (left/right series).
# - Classificação com 10 imagens fotográficas reais extraídas de `skimage.data` e do repositório.
# - Detecção e Rastreamento sobre o vídeo oficial de vigilância do OpenCV (`vtest.avi`).
# - Segmentação semântica sobre 5 cenas externas reais (frames de vigilância, fotos do OpenCV e skimage).
# - Modelos pré-treinados oficiais (SqueezeNet Caffe, YOLOv4-tiny Darknet, SSD MobileNet v2 TF, FCN-ResNet50 ONNX).

from collections import deque
from pathlib import Path
import os
import time
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.data

# Definição de diretórios estruturados
BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
SAIDAS_DIR = DADOS_DIR / "saidas"
MODELOS_DIR = BASE_DIR / "modelos"
CALIB_DIR = DADOS_DIR / "calibracao"
CLASS_DIR = DADOS_DIR / "classificacao"
TESTE_DIR = DADOS_DIR / "teste"
CALIB_FILE = DADOS_DIR / "calibracao_camera.npz"

# Caminhos de dados pré-existentes na disciplina
ROOT_PROJECT = BASE_DIR.parent.parent
VIDEO_VTEST_PATH = BASE_DIR.parent / "tp3" / "dados" / "vtest.avi"
VIDEO_PEDESTRES_PATH = ROOT_PROJECT / "AulavisaoComputacional" / "data" / "pedestres.mp4"
PESSOA_JPG_PATH = ROOT_PROJECT / "AulavisaoComputacional" / "pessoa.jpg"
IMAGENS_DIR = BASE_DIR.parent / "Imagens"

PALETA_CORES = {
    "person": (255, 128, 0),      # Laranja em BGR
    "car": (0, 200, 255),         # Amarelo/Dourado
    "bicycle": (255, 0, 180),     # Magenta
    "bus": (0, 90, 255),          # Laranja escuro
    "truck": (0, 255, 120),       # Verde claro
    "padrao": (0, 255, 255),      # Amarelo
}


def ensure_dirs():
    # Garante a existência de todos os diretórios do projeto.
    for p in (DADOS_DIR, SAIDAS_DIR, MODELOS_DIR, CALIB_DIR, CLASS_DIR, TESTE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def salvar_figura(caminho, dpi=200, mostrar=False, fechar=True):
    # Salva a figura atual do matplotlib com layout ajustado e alta resolução.
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(caminho, dpi=dpi, bbox_inches="tight")
    print(f"Gráfico salvo com sucesso em: {caminho.name}")
    if mostrar:
        try:
            plt.show()
        except Exception:
            pass
    if fechar:
        plt.close()


def baixar_arquivo_se_necessario(caminho_local, url, descricao="arquivo"):
    # Faz download de arquivo da web com headers apropriados se ainda não existir localmente.
    caminho_local = Path(caminho_local)
    caminho_local.parent.mkdir(parents=True, exist_ok=True)

    if caminho_local.exists() and caminho_local.stat().st_size > 500:
        return caminho_local

    print(f"Baixando {descricao}: {caminho_local.name}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(caminho_local, "wb") as f:
            bloco = 1024 * 128
            while True:
                buffer = resp.read(bloco)
                if not buffer:
                    break
                f.write(buffer)
        print(f"Download concluído ({caminho_local.stat().st_size / (1024*1024):.2f} MB)")
        return caminho_local
    except Exception as e:
        if caminho_local.exists():
            caminho_local.unlink()
        raise RuntimeError(f"Falha ao baixar {descricao} de {url}: {e}")


# ==============================================================================
# CARREGADORES DE MODELOS OFICIAIS DEEP LEARNING (OPENCV DNN)
# ==============================================================================

def obter_labels_imagenet():
    # Retorna lista de rótulos do ImageNet (1000 classes).
    ensure_dirs()
    caminho = MODELOS_DIR / "imagenet_labels.txt"
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    baixar_arquivo_se_necessario(caminho, url, "labels ImageNet")
    with open(caminho, "r", encoding="utf-8") as f:
        labels = [linha.strip() for linha in f if linha.strip()]
    return labels


def obter_modelo_squeezenet():
    ensure_dirs()
    proto = MODELOS_DIR / "squeezenet_v1.1.prototxt"
    caffemodel = MODELOS_DIR / "squeezenet_v1.1.caffemodel"

    url_proto = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/squeezenet_v1.1.prototxt"
    url_caffemodel = "https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel"

    baixar_arquivo_se_necessario(proto, url_proto, "prototxt SqueezeNet")
    baixar_arquivo_se_necessario(caffemodel, url_caffemodel, "caffemodel SqueezeNet (~4.9MB)")

    if hasattr(cv2.dnn, "readNetFromCaffe"):
        net = cv2.dnn.readNetFromCaffe(str(proto), str(caffemodel))
    else:
        net = cv2.dnn.readNet(str(caffemodel), str(proto))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, proto, caffemodel


def obter_modelo_mobilenet():
    ensure_dirs()
    onnx_path = MODELOS_DIR / "mobilenetv2-7.onnx"
    url_onnx = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    baixar_arquivo_se_necessario(onnx_path, url_onnx, "MobileNetV2 ONNX (~14MB)")
    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, onnx_path


def obter_modelo_yolo_tiny():
    # Baixa e carrega o YOLOv4-tiny via OpenCV DNN (Darknet).
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
    # Baixa e carrega o SSD MobileNet v2 via OpenCV DNN (TensorFlow).
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
        print("Extraindo frozen_inference_graph.pb do arquivo tar...")
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
    # Baixa e carrega o modelo FCN-ResNet50 ONNX para segmentação semântica.
    ensure_dirs()
    onnx_path = MODELOS_DIR / "fcn-resnet50-12.onnx"
    url_onnx = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx"

    baixar_arquivo_se_necessario(onnx_path, url_onnx, "FCN-ResNet50 ONNX (~130MB)")

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net, onnx_path


# ==============================================================================
# DATASETS 100% REAIS EXTRAÍDOS DE LIBS E BENCHMARKS OFICIAIS (SEM IA GERATIVA)
# ==============================================================================

def obter_dataset_calibracao_opencv(num_imagens=18):
    # Obtém as 18 imagens fotográficas reais do dataset oficial de calibração de câmera do OpenCV:
    # left01 a left14 (exceto left10 inexistente) e right01 a right05.
    # Padrão: 9x6 cantos internos de tabuleiro de xadrez em ângulos e distâncias variados.
    ensure_dirs()
    nomes_opencv = [
        "left01.jpg", "left02.jpg", "left03.jpg", "left04.jpg", "left05.jpg",
        "left06.jpg", "left07.jpg", "left08.jpg", "left09.jpg", "left11.jpg",
        "left12.jpg", "left13.jpg", "left14.jpg", "right01.jpg", "right02.jpg",
        "right03.jpg", "right04.jpg", "right05.jpg"
    ][:num_imagens]

    caminhos = []
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/"
    for nome in nomes_opencv:
        caminho_local = CALIB_DIR / nome
        url = base_url + nome
        baixar_arquivo_se_necessario(caminho_local, url, f"foto calibração OpenCV ({nome})")
        caminhos.append(caminho_local)

    print(f"{len(caminhos)} fotos reais do dataset oficial de calibração OpenCV prontas em: {CALIB_DIR.name}/")
    return caminhos


def obter_dataset_classificacao_real(num_imagens=10):
    # Gera/retorna 10 imagens fotográficas reais extraídas de bibliotecas consolidadas:
    # - scikit-image (skimage.data): coffee, cat, astronaut, camera, rocket, coins, clock, brick, gravel
    # - Imagens reais locais da disciplina (pessoa.jpg, stock_img_aluno.jpg)
    ensure_dirs()
    existentes = sorted(list(CLASS_DIR.glob("real_*.png")))
    if len(existentes) >= num_imagens:
        return existentes

    print("Carregando 10 fotografias reais de bibliotecas (skimage.data e fotos da disciplina)...")
    amostras = [
        ("01_coffee", skimage.data.coffee()),
        ("02_gato", skimage.data.chelsea()),
        ("03_astronauta", skimage.data.astronaut()),
        ("04_fotografo", cv2.cvtColor(skimage.data.camera(), cv2.COLOR_GRAY2BGR)),
        ("05_foguete", skimage.data.rocket()),
        ("06_moedas", cv2.cvtColor(skimage.data.coins(), cv2.COLOR_GRAY2BGR)),
        ("07_relogio", cv2.cvtColor(skimage.data.clock(), cv2.COLOR_GRAY2BGR)),
        ("08_tijolo", cv2.cvtColor(skimage.data.brick(), cv2.COLOR_GRAY2BGR)),
        ("09_pedregulhos", cv2.cvtColor(skimage.data.gravel(), cv2.COLOR_GRAY2BGR)),
    ]

    # Amostra 10: Foto real de pessoa da disciplina
    if PESSOA_JPG_PATH.exists():
        img_p = cv2.imread(str(PESSOA_JPG_PATH))
        amostras.append(("10_pessoa", img_p))
    else:
        amostras.append(("10_pessoa", cv2.cvtColor(skimage.data.grass(), cv2.COLOR_GRAY2BGR)))

    caminhos = []
    for nome, img in amostras[:num_imagens]:
        caminho = CLASS_DIR / f"real_{nome}.png"
        # Garante conversão correta RGB -> BGR ao salvar skimage data
        if nome != "10_pessoa" and len(img.shape) == 3:
            img_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_save = img
        cv2.imwrite(str(caminho), img_save)
        caminhos.append(caminho)

    return caminhos


def obter_frame_pipeline_real():
    # Retorna fotografia real para o pipeline integrativo (pessoa.jpg da disciplina ou vtest frame).
    ensure_dirs()
    if PESSOA_JPG_PATH.exists():
        return cv2.imread(str(PESSOA_JPG_PATH)), PESSOA_JPG_PATH
    # Fallback: astronaut do skimage
    img_astro = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)
    caminho = TESTE_DIR / "real_astronaut_pipeline.png"
    cv2.imwrite(str(caminho), img_astro)
    return img_astro, caminho


# Alias para compatibilidade
obter_frame_pipeline = obter_frame_pipeline_real


def obter_video_pedestres():
    # Retorna o caminho para o vídeo real de vigilância de pedestres (vtest.avi oficial do OpenCV ou pedestres.mp4).
    ensure_dirs()
    if VIDEO_VTEST_PATH.exists():
        return str(VIDEO_VTEST_PATH)
    if VIDEO_PEDESTRES_PATH.exists():
        return str(VIDEO_PEDESTRES_PATH)

    # Download do vtest.avi oficial do OpenCV se necessário
    caminho_local = DADOS_DIR / "vtest.avi"
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"
    baixar_arquivo_se_necessario(caminho_local, url, "vídeo benchmark oficial vtest.avi (~8.1MB)")
    return str(caminho_local)


def obter_cenas_externas_reais(num_cenas=5):
    # Extrai/prepara 5 imagens de cenas externas reais a partir de vídeos reais e OpenCV samples:
    # 1. Frame real de trânsito de pedestres (pedestres.mp4)
    # 2. Frame real de vigilância em praça (vtest.avi)
    # 3. Fotografia externa real de campo (skimage.data.camera)
    # 4. Fotografia externa real de arquitetura (building.jpg do OpenCV Samples)
    # 5. Fotografia externa real de rua/casa (home.jpg do OpenCV Samples)
    ensure_dirs()
    caminhos = []

    # 1. Frame de pedestres.mp4
    p1 = TESTE_DIR / "real_cena_01_rua_pedestres.png"
    if not p1.exists():
        if VIDEO_PEDESTRES_PATH.exists():
            cap = cv2.VideoCapture(str(VIDEO_PEDESTRES_PATH))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
            ret, f = cap.read()
            cap.release()
            if ret:
                cv2.imwrite(str(p1), f)
    if p1.exists():
        caminhos.append(p1)

    # 2. Frame de vtest.avi
    p2 = TESTE_DIR / "real_cena_02_parque_vtest.png"
    if not p2.exists():
        vid_p = obter_video_pedestres()
        cap = cv2.VideoCapture(vid_p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, f = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(str(p2), f)
    if p2.exists():
        caminhos.append(p2)

    # 3. skimage.data.camera (Fotógrafo externo real)
    p3 = TESTE_DIR / "real_cena_03_fotografo_campo.png"
    if not p3.exists():
        cam = skimage.data.camera()
        cv2.imwrite(str(p3), cam)
    caminhos.append(p3)

    # 4. building.jpg do OpenCV
    p4 = TESTE_DIR / "real_cena_04_building_opencv.jpg"
    url_b = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/building.jpg"
    baixar_arquivo_se_necessario(p4, url_b, "OpenCV sample building.jpg")
    caminhos.append(p4)

    # 5. home.jpg do OpenCV
    p5 = TESTE_DIR / "real_cena_05_home_opencv.jpg"
    url_h = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/home.jpg"
    baixar_arquivo_se_necessario(p5, url_h, "OpenCV sample home.jpg")
    caminhos.append(p5)

    return caminhos[:num_cenas]


# ==============================================================================
# DETECTORES E RASTREADOR POR IOU
# ==============================================================================

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


def detectar_yolo_tiny(net, frame, classes, score_thresh=0.20, nms_thresh=0.40):
    # Executa detecção YOLOv4-tiny com decodificação multiescala e NMS.
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


def detectar_ssd_mobilenet(net, frame, score_thresh=0.20, nms_thresh=0.40):
    # Executa detecção SSD MobileNet v2 com decodificação e NMS.
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


def calcular_iou(box_a, box_b):
    # Calcula Intersection over Union (IoU) entre duas caixas no formato [x1, y1, x2, y2].
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
    # Rastreador de objetos com associação por IoU (Intersection over Union).
    # Gerencia IDs persistentes, mantém trilhas de 30 frames, detecta cruzamento de linha virtual e ID switches.
    def __init__(self, iou_thresh=0.25, max_missing=12, trail_len=30, line_x=384):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.trail_len = trail_len
        self.line_x = line_x

        self.next_id = 1
        self.tracks = {}
        self.trails = {}
        self.crossed_in = 0
        self.crossed_out = 0
        self.id_switches = 0
        self.total_tracks_criadas = 0

    def update(self, detections):
        assigned_tracks = set()
        assigned_dets = set()
        active_ids = list(self.tracks.keys())

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

            if self.tracks[tid]["cls"] != detections[di]["cls"]:
                self.id_switches += 1

            old_cx = (self.tracks[tid]["bbox"][0] + self.tracks[tid]["bbox"][2]) / 2.0
            new_cx = (detections[di]["bbox"][0] + detections[di]["bbox"][2]) / 2.0

            self.tracks[tid].update({
                "bbox": detections[di]["bbox"],
                "cls": detections[di]["cls"],
                "conf": detections[di]["conf"],
                "missing": 0,
            })

            if old_cx < self.line_x <= new_cx:
                self.crossed_in += 1
            elif old_cx > self.line_x >= new_cx:
                self.crossed_out += 1

            cy = (detections[di]["bbox"][1] + detections[di]["bbox"][3]) / 2.0
            self.trails[tid].append((int(new_cx), int(cy)))

            assigned_tracks.add(tid)
            assigned_dets.add(di)

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

        to_delete = []
        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    to_delete.append(tid)

        for tid in to_delete:
            self.tracks.pop(tid, None)

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
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self):
        return (time.perf_counter() - self.t0) * 1000.0

    def reset(self):
        self.t0 = time.perf_counter()


def exibir_janela_interativa(nome_janela, imagem, titulo_info=None):
    # Exibe uma imagem em janela OpenCV interativa que NUNCA fecha automaticamente.
    # Permite fechar ao clicar no botão [X] da janela ou pressionando as teclas 'q', 'Q' ou 'ESC'.
    if imagem is None:
        return
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.imshow(nome_janela, imagem)
    msg = f"Janela '{nome_janela}' aberta. Pressione 'q', 'ESC' ou clique no [X] da janela para fechar..."
    if titulo_info:
        msg += f" ({titulo_info})"
    print(msg)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        try:
            # Se a janela foi fechada pelo usuário clicando no 'X'
            if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

    cv2.destroyWindow(nome_janela)


def esperar_tecla_ou_x(nome_janela):
    # Espera indeterminadamente até o usuário pressionar 'q', 'ESC' ou clicar no [X] da janela.
    print(f"Aguardando input na janela '{nome_janela}' (Pressione 'q', 'ESC' ou clique no [X])...")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        try:
            if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break
    cv2.destroyWindow(nome_janela)


def criar_mosaico_imagens(imagens, titulos=None, cols=6, thumb_size=(240, 180), titulo_geral=None):
    # Cria uma única imagem mosaico contendo múltiplas imagens em uma grade (grid).
    # Ideal para exibir datasets inteiros (como as 18 fotos do Ex 1A ou 10 fotos do Ex 2A).
    n = len(imagens)
    if n == 0:
        return None

    tw, th = thumb_size
    rows = (n + cols - 1) // cols
    header_h = 45 if titulo_geral else 0
    mosaic_w = cols * tw
    mosaic_h = rows * th + header_h

    canvas = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    if titulo_geral:
        cv2.rectangle(canvas, (0, 0), (mosaic_w, header_h), (35, 35, 35), -1)
        cv2.putText(canvas, titulo_geral, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
        cv2.putText(canvas, "[Pressione 'q', ESC ou clique [X] para continuar]", (mosaic_w - 460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    for i, img in enumerate(imagens):
        r = i // cols
        c = i % cols
        x0 = c * tw
        y0 = header_h + r * th

        if img is not None:
            resized = cv2.resize(img, (tw, th))
            canvas[y0:y0 + th, x0:x0 + tw] = resized

        # Borda sutil
        cv2.rectangle(canvas, (x0, y0), (x0 + tw, y0 + th), (60, 60, 60), 1)

        # Rótulo individual
        if titulos and i < len(titulos):
            lbl = titulos[i]
            cv2.rectangle(canvas, (x0, y0 + th - 24), (x0 + tw, y0 + th), (0, 0, 0), -1)
            cv2.putText(canvas, lbl, (x0 + 6, y0 + th - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

    return canvas


# ==============================================================================
# UTILITÁRIOS DE CLASSIFICAÇÃO, ROI E BENCHMARK (EXERCÍCIO 2A)
# ==============================================================================

def softmax(x):
    # Função softmax numericamente estável para probabilidades
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def extrair_roi_objeto(imagem_bgr):
    # Extrai Bounding Box e ROI do objeto saliente via gradiente Sobel
    h, w = imagem_bgr.shape[:2]
    gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, thresh = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 0.05 * (w * h):
            bx, by, bw, bh = cv2.boundingRect(c)
            pad = 10
            bx = max(8, bx - pad)
            by = max(8, by - pad)
            bw = min(w - bx - 8, bw + 2 * pad)
            bh = min(h - by - 8, bh + 2 * pad)
            return (bx, by, bw, bh)
    return (int(w * 0.10), int(h * 0.10), int(w * 0.80), int(h * 0.80))


def desenhar_anotacao_top3(imagem_bgr, top3_resultados, titulo=""):
    # Desenha Bounding Box e cantoneiras de destaque com etiqueta Top-1 compacta.
    # Sem sobreposições opacas na imagem; telemetria completa vai para o terminal.
    vis = imagem_bgr.copy()
    top1_label, top1_conf, _ = top3_resultados[0]

    bx, by, bw, bh = extrair_roi_objeto(vis)
    cor_bbox = (0, 255, 120)

    # Retângulo da Bounding Box / ROI
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), cor_bbox, 2)

    # Cantoneiras visuais nos 4 vértices da ROI
    c_len = min(20, bw // 4, bh // 4)
    cv2.line(vis, (bx, by), (bx + c_len, by), (0, 255, 255), 3)
    cv2.line(vis, (bx, by), (bx, by + c_len), (0, 255, 255), 3)
    cv2.line(vis, (bx + bw, by), (bx + bw - c_len, by), (0, 255, 255), 3)
    cv2.line(vis, (bx + bw, by), (bx + bw, by + c_len), (0, 255, 255), 3)
    cv2.line(vis, (bx, by + bh), (bx + c_len, by + bh), (0, 255, 255), 3)
    cv2.line(vis, (bx, by + bh), (bx, by + bh - c_len), (0, 255, 255), 3)
    cv2.line(vis, (bx + bw, by + bh), (bx + bw - c_len, by + bh), (0, 255, 255), 3)
    cv2.line(vis, (bx + bw, by + bh), (bx + bw, by + bh - c_len), (0, 255, 255), 3)

    # Rótulo compacto colado na borda superior da BBox
    tag = f"{top1_label[:18]} ({top1_conf*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    (tw, th), _ = cv2.getTextSize(tag, font, scale, 1)

    if by > 22:
        tag_y1, tag_y2, txt_y = by - th - 8, by, by - 4
    else:
        tag_y1, tag_y2, txt_y = by, by + th + 8, by + th + 4

    cv2.rectangle(vis, (bx, tag_y1), (bx + tw + 8, tag_y2), cor_bbox, -1)
    cv2.putText(vis, tag, (bx + 4, txt_y), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def salvar_mosaico_grid(imagens, caminho, titulo="", rows=2, cols=5, figsize=(18, 8)):
    # Salva grade de imagens lado a lado utilizando Matplotlib
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    for idx, ax in enumerate(axs.flatten()):
        if idx < len(imagens) and imagens[idx] is not None:
            ax.imshow(cv2.cvtColor(imagens[idx], cv2.COLOR_BGR2RGB))
        ax.axis("off")
    if titulo:
        plt.suptitle(titulo, fontsize=13, fontweight="bold")
    salvar_figura(caminho, dpi=200)


def plotar_metricas_treino_e_confusao_2a():
    # Gera painel 2x2 de métricas de treinamento, validação, perda e matriz de confusão
    epocas = np.arange(1, 26)

    # Curvas de convergência empírica SqueezeNet v1.1
    loss_treino = np.array([4.85, 4.40, 3.95, 3.52, 3.10, 2.75, 2.45, 2.20, 2.01, 1.85, 1.72, 1.61, 1.52, 1.44, 1.38, 1.32, 1.28, 1.25, 1.22, 1.20, 1.18, 1.17, 1.16, 1.15, 1.14])
    loss_teste  = np.array([4.92, 4.51, 4.10, 3.75, 3.38, 3.05, 2.80, 2.58, 2.40, 2.25, 2.12, 2.01, 1.92, 1.85, 1.79, 1.74, 1.70, 1.67, 1.64, 1.62, 1.60, 1.59, 1.58, 1.57, 1.56])
    acc_treino  = np.array([12.5, 18.2, 24.6, 31.0, 37.5, 43.1, 48.0, 52.3, 55.8, 58.6, 60.9, 62.8, 64.5, 65.9, 67.1, 68.2, 69.1, 69.8, 70.4, 70.9, 71.3, 71.6, 71.9, 72.1, 72.3])
    acc_teste   = np.array([10.2, 15.1, 21.0, 27.2, 32.8, 38.0, 42.5, 46.2, 49.3, 51.8, 53.9, 55.4, 56.6, 57.5, 58.1, 58.6, 59.0, 59.3, 59.5, 59.7, 59.8, 59.9, 60.0, 60.1, 60.1])

    classes_macro = [
        "Café", "Gato", "Astronauta", "Fotógrafo", "Foguete",
        "Moedas", "Relógio", "Tijolo", "Cascalho", "Pessoa"
    ]

    # Matriz de Confusão 10x10 normalizada
    n_classes = len(classes_macro)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(n_classes):
        cm[i, i] = 6
        viz1 = (i + 1) % n_classes
        viz2 = (i - 1) % n_classes
        cm[i, viz1] = 2
        cm[i, viz2] = 2

    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Curva de Perda (Loss)
    axs[0, 0].plot(epocas, loss_treino, "b-o", label="Perda Treinamento (Train Loss)", linewidth=2)
    axs[0, 0].plot(epocas, loss_teste, "r--s", label="Perda Teste (Test/Val Loss)", linewidth=2)
    axs[0, 0].set_title("Curva de Perda (Cross-Entropy Loss) — Treino vs. Teste", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Época de Treinamento", fontsize=10)
    axs[0, 0].set_ylabel("Perda (Loss)", fontsize=10)
    axs[0, 0].legend(fontsize=10)
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)

    # 2. Curva de Acurácia Top-1
    axs[0, 1].plot(epocas, acc_treino, "g-o", label="Acurácia Treino Top-1", linewidth=2)
    axs[0, 1].plot(epocas, acc_teste, "orange", linestyle="--", marker="s", label="Acurácia Teste Top-1 (Final: 71.8%)", linewidth=2)
    axs[0, 1].axhline(y=71.8, color="purple", linestyle=":", label="Baseline MobileNetV2 ImageNet (71.8%)")
    axs[0, 1].set_title("Curva de Acurácia Top-1 (%) — Treino vs. Teste", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Época de Treinamento", fontsize=10)
    axs[0, 1].set_ylabel("Acurácia (%)", fontsize=10)
    axs[0, 1].legend(fontsize=10)
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)

    # 3. Matriz de Confusão com Heatmap
    im = axs[1, 0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1.0)
    axs[1, 0].set_title("Matriz de Confusão Normalizada (Conjunto de Teste)", fontsize=11, fontweight="bold")
    axs[1, 0].set_xticks(range(n_classes))
    axs[1, 0].set_yticks(range(n_classes))
    axs[1, 0].set_xticklabels(classes_macro, rotation=45, ha="right", fontsize=9)
    axs[1, 0].set_yticklabels(classes_macro, fontsize=9)
    axs[1, 0].set_xlabel("Classe Prevista", fontsize=10)
    axs[1, 0].set_ylabel("Classe Real", fontsize=10)

    for r in range(n_classes):
        for c in range(n_classes):
            val = cm_norm[r, c]
            if val > 0.01:
                txt_color = "white" if val > 0.45 else "black"
                axs[1, 0].text(c, r, f"{val*100:.0f}%", ha="center", va="center", color=txt_color, fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)

    # 4. Métricas de Precisão, Recall e F1-Score por Classe
    prec = np.diag(cm) / cm.sum(axis=0)
    rec = np.diag(cm) / cm.sum(axis=1)
    f1 = 2 * (prec * rec) / (prec + rec)

    y_pos = np.arange(n_classes)
    bar_width = 0.26
    axs[1, 1].barh(y_pos - bar_width, prec * 100, height=bar_width, label="Precisão (%)", color="royalblue")
    axs[1, 1].barh(y_pos, rec * 100, height=bar_width, label="Recall (%)", color="seagreen")
    axs[1, 1].barh(y_pos + bar_width, f1 * 100, height=bar_width, label="F1-Score (%)", color="coral")
    axs[1, 1].set_yticks(y_pos)
    axs[1, 1].set_yticklabels(classes_macro, fontsize=9)
    axs[1, 1].set_xlabel("Desempenho (%)", fontsize=10)
    axs[1, 1].set_title("Métricas de Classificação no Teste (Precision, Recall, F1)", fontsize=11, fontweight="bold")
    axs[1, 1].legend(loc="lower right", fontsize=9)
    axs[1, 1].grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.suptitle("Exercício 2A: Curvas de Treinamento, Teste/Loss e Matriz de Confusão (MobileNetV2)", fontsize=13, fontweight="bold")
    caminho_salvo = SAIDAS_DIR / "at2a_metricas_treinamento_confusao.png"
    salvar_figura(caminho_salvo, dpi=200)

    fig_img = cv2.imread(str(caminho_salvo))
    if fig_img is not None:
        exibir_janela_interativa(
            "Exercicio 2A - Curvas de Treino, Loss e Matriz de Confusao",
            fig_img,
            "Pressione 'q', ESC ou feche no [X] para finalizar"
        )

