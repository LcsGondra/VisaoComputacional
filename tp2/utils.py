import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

win_transform = {"left": 0, "top": 0, "new_w": 640, "new_h": 480, "img_w": 640, "img_h": 480}


def obter_diretorios(arquivo_origem=None):
    if arquivo_origem is not None:
        base_dir = Path(arquivo_origem).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    dados_dir = base_dir / "dados"
    saidas_dir = dados_dir / "saidas"
    modelos_dir = base_dir / "modelos"
    faces_dir = dados_dir / "faces_48x48"
    ref_dir = modelos_dir / "referencias"

    for d in (dados_dir, saidas_dir, modelos_dir, faces_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "dados": dados_dir,
        "saidas": saidas_dir,
        "modelos": modelos_dir,
        "faces": faces_dir,
        "referencias": ref_dir,
    }


def imshow_keep_aspect_ratio(winname, img, bg_color=(0, 0, 0)):
    if img is None:
        return
    try:
        rect = cv2.getWindowImageRect(winname)
        win_w, win_h = rect[2], rect[3]
    except Exception:
        win_w, win_h = 0, 0

    if win_w <= 10 or win_h <= 10:
        cv2.imshow(winname, img)
        return

    img_h, img_w = img.shape[:2]
    img_aspect = img_w / float(img_h)
    win_aspect = win_w / float(win_h)

    if win_aspect > img_aspect:
        new_h = win_h
        new_w = int(win_h * img_aspect)
    else:
        new_w = win_w
        new_h = int(win_w / img_aspect)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        canvas = np.full((win_h, win_w), bg_color[0], dtype=np.uint8)
    else:
        canvas = np.full((win_h, win_w, 3), bg_color, dtype=np.uint8)

    top = (win_h - new_h) // 2
    left = (win_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    win_transform["left"] = left
    win_transform["top"] = top
    win_transform["new_w"] = new_w
    win_transform["new_h"] = new_h
    win_transform["img_w"] = img_w
    win_transform["img_h"] = img_h

    cv2.imshow(winname, canvas)


def selecionar_dispositivo():
    if torch.cuda.is_available():
        nome_gpu = torch.cuda.get_device_name(0)
        propriedades = torch.cuda.get_device_properties(0)
        vram_gb = propriedades.total_memory / (1024**3)
        print(f"-> Acelerador detectado: [NVIDIA/ROCm CUDA] {nome_gpu} ({vram_gb:.1f} GB VRAM)")
        return torch.device("cuda")

    try:
        import torch_directml
        if torch_directml.is_available():
            dml_device = torch_directml.device()
            nome_dml = torch_directml.device_name(0) if hasattr(torch_directml, "device_name") else "DirectML Device"
            print(f"-> Acelerador detectado: [AMD/Intel DirectML] {nome_dml}")
            return dml_device
    except ImportError:
        pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("-> Acelerador detectado: [Apple Silicon MPS]")
        return torch.device("mps")

    print("-> Acelerador GPU compativel nao detectado. Utilizando processamento em [CPU].")
    return torch.device("cpu")


def criar_cnn_keras(input_shape=(28, 28, 1), num_classes=3):
    try:
        from tensorflow.keras import layers, models
    except ImportError:
        import tf_keras.layers as layers
        import tf_keras.models as models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.MaxPooling2D((2, 2), name="maxpool_1"),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.MaxPooling2D((2, 2), name="maxpool_2"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu", name="feature_dense"),
        layers.Dropout(0.25, name="dropout"),
        layers.Dense(num_classes, activation="softmax", name="saida_softmax"),
    ])
    return model


class CNNRobotics(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNRobotics, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.penultima_dense = nn.Sequential(
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.camada_saida = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        feat = self.penultima_dense(x)
        out = self.camada_saida(feat)
        return out

    def extrair_features(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.flatten(x)
        for layer in self.penultima_dense:
            if isinstance(layer, nn.Dropout):
                continue
            x = layer(x)
        return x
