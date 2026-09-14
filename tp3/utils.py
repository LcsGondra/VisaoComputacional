import os
import sys
import time
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import skimage.data
from sklearn.datasets import fetch_olivetti_faces

# Diretórios principais
BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"
SAIDAS_DIR = DADOS_DIR / "saidas"
MODELOS_DIR = BASE_DIR / "modelos"
POSITIVAS_DIR = DADOS_DIR / "processadas" / "positivas"
NEGATIVAS_DIR = DADOS_DIR / "processadas" / "negativas"
TESTE_DIR = DADOS_DIR / "teste"

# Rótulos Caffe
GENDER_LABELS = ["Masculino", "Feminino"]
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
MEAN_VALUES_CAFFE = (78.4263377603, 87.7689143744, 114.895847746)


def ensure_dirs():
    for p in (DADOS_DIR, SAIDAS_DIR, MODELOS_DIR, POSITIVAS_DIR, NEGATIVAS_DIR, TESTE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def salvar_figura(caminho, dpi=200, fechar=True):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(caminho, dpi=dpi, bbox_inches="tight")
    print(f"[+] Gráfico salvo em: {caminho.name}")
    if fechar:
        plt.close()


def extrair_features_hog(imagens, win_size=(64, 128)):
    hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
    features = []
    for img in imagens:
        if img.shape[:2] != (win_size[1], win_size[0]):
            img = cv2.resize(img, win_size)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        desc = hog.compute(img)
        features.append(desc.flatten())
    return np.array(features, dtype=np.float32)


def sliding_window_multiescala(imagem, modelo_svm, win_size=(64, 128), step_size=8, scale_factor=1.2, min_confidence=0.70):
    hog = cv2.HOGDescriptor(win_size, (16, 16), (8, 8), (8, 8), 9)
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if len(imagem.shape) == 3 else imagem

    boxes = []
    scores = []
    current_scale = 1.0
    scaled_img = gray.copy()

    while scaled_img.shape[0] >= win_size[1] and scaled_img.shape[1] >= win_size[0]:
        h_scaled, w_scaled = scaled_img.shape[:2]
        for y in range(0, h_scaled - win_size[1] + 1, step_size):
            for x in range(0, w_scaled - win_size[0] + 1, step_size):
                patch = scaled_img[y : y + win_size[1], x : x + win_size[0]]
                desc = hog.compute(patch).reshape(1, -1)

                if hasattr(modelo_svm, "predict_proba"):
                    prob = float(modelo_svm.predict_proba(desc)[0, 1])
                else:
                    prob = float(modelo_svm.decision_function(desc)[0])

                if prob > min_confidence:
                    boxes.append((int(x * current_scale), int(y * current_scale), int(win_size[0] * current_scale), int(win_size[1] * current_scale)))
                    scores.append(prob)

        current_scale *= scale_factor
        new_w = int(w_scaled / scale_factor)
        new_h = int(h_scaled / scale_factor)
        if new_w < win_size[0] or new_h < win_size[1]:
            break
        scaled_img = cv2.resize(scaled_img, (new_w, new_h))

    return boxes, scores


def carregar_ou_extrair_dataset_hog(num_positivos=100, num_negativos=100, win_size=(64, 128)):
    ensure_dirs()
    w, h = win_size
    pos_arqs = sorted([p for p in POSITIVAS_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg"}])
    neg_arqs = sorted([p for p in NEGATIVAS_DIR.glob("*") if p.suffix.lower() in {".png", ".jpg"}])

    if len(pos_arqs) >= num_positivos and len(neg_arqs) >= num_negativos:
        imagens, rotulos = [], []
        for p in pos_arqs[:num_positivos]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imagens.append(cv2.resize(img, (w, h)))
                rotulos.append(1)
        for p in neg_arqs[:num_negativos]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imagens.append(cv2.resize(img, (w, h)))
                rotulos.append(0)
        return imagens, rotulos, "Dataset local (processadas/positivas e processadas/negativas)"

    faces = fetch_olivetti_faces()
    imagens, rotulos = [], []

    for i in range(min(num_positivos, len(faces.images))):
        face = (faces.images[i] * 255).astype(np.uint8)
        face_res = cv2.resize(face, (w, h))
        cv2.imwrite(str(POSITIVAS_DIR / f"pos_{i:03d}.png"), face_res)
        imagens.append(face_res)
        rotulos.append(1)

    bg_img = skimage.data.camera()
    np.random.seed(42)
    for i in range(num_negativos):
        yr = np.random.randint(0, bg_img.shape[0] - 80)
        xr = np.random.randint(0, bg_img.shape[1] - 50)
        patch = cv2.resize(bg_img[yr : yr + 80, xr : xr + 50], (w, h))
        cv2.imwrite(str(NEGATIVAS_DIR / f"neg_{i:03d}.png"), patch)
        imagens.append(patch)
        rotulos.append(0)

    return imagens, rotulos, "Gerado a partir de Olivetti Faces e Skimage Camera"


def obter_ou_gerar_cena_teste():
    ensure_dirs()
    caminho = TESTE_DIR / "cena_teste.png"
    if caminho.exists():
        img = cv2.imread(str(caminho))
        if img is not None:
            return img

    faces = fetch_olivetti_faces()
    cena = cv2.resize(skimage.data.camera(), (480, 360))
    face1 = cv2.resize((faces.images[250] * 255).astype(np.uint8), (64, 128))
    face2 = cv2.resize((faces.images[350] * 255).astype(np.uint8), (64, 128))

    cena[80:208, 96:160] = face1
    cena[144:272, 304:368] = face2
    cena_bgr = cv2.cvtColor(cena, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(caminho), cena_bgr)
    return cena_bgr


def obter_video(nome_padrao="vtest.avi"):
    ensure_dirs()
    caminho = DADOS_DIR / nome_padrao
    if caminho.exists():
        return str(caminho)
    arquivos = list(DADOS_DIR.glob("*.avi")) + list(DADOS_DIR.glob("*.mp4"))
    if arquivos:
        return str(arquivos[0])
    raise FileNotFoundError(f"Arquivo de vídeo não encontrado em: {caminho}")


def inicializar_histograma_camshift(frame, window):
    x, y, w, h = window
    roi = frame[y : y + h, x : x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv_roi, np.array([0, 80, 40]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv_roi, np.array([168, 80, 40]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def inicializar_filtro_kalman(dt=1.0, q=0.03, r=8.0, p=1.0):
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * np.float32(q)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * np.float32(r)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * np.float32(p)
    return kf


def carregar_dataset_digitos(limite_amostras=3000):
    from sklearn.datasets import load_digits
    data = load_digits()
    raw_images = data.images
    raw_targets = data.target

    X, y = [], []
    np.random.seed(42)
    for i in range(limite_amostras):
        idx = np.random.randint(0, len(raw_images))
        img = cv2.resize(raw_images[idx], (28, 28), interpolation=cv2.INTER_CUBIC)
        img = img / (img.max() + 1e-6)
        X.append(img)
        y.append(raw_targets[idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:], [f"Digito {i}" for i in range(10)]


def aplicar_aumento_dados_digito(img):
    angle = float(np.random.uniform(-15.0, 15.0))
    scale = float(np.random.uniform(0.9, 1.1))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def extrair_features_convolucionais(imagens):
    kernels = [
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32),
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
    ]
    feats = []
    for img in imagens:
        img_f = (img * 255.0).astype(np.float32) if img.max() <= 1.0 else img.astype(np.float32)
        f_vec = []
        for k in kernels:
            conv = cv2.filter2D(img_f, -1, k)
            conv_relu = np.maximum(0, conv)
            pool = cv2.resize(conv_relu, (4, 4), interpolation=cv2.INTER_AREA)
            f_vec.extend(pool.flatten())
        f_vec.extend(cv2.resize(img_f, (4, 4), interpolation=cv2.INTER_AREA).flatten())
        feats.append(f_vec)
    return np.array(feats, dtype=np.float32)


def obter_modelos_caffe_idade_genero():
    ensure_dirs()
    pasta = MODELOS_DIR / "opencv_age_gender"
    pasta.mkdir(parents=True, exist_ok=True)

    age_proto = pasta / "age_deploy.prototxt"
    age_model = pasta / "age_net.caffemodel"
    gender_proto = pasta / "gender_deploy.prototxt"
    gender_model = pasta / "gender_net.caffemodel"

    urls = {
        age_proto: "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
        gender_proto: "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
        age_model: "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_net.caffemodel",
        gender_model: "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel",
    }

    import urllib.request
    for path_dest, url in urls.items():
        if not path_dest.exists() or path_dest.stat().st_size < 100:
            print(f"[+] Baixando modelo Caffe: {path_dest.name}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as resp, open(path_dest, "wb") as f:
                f.write(resp.read())

    age_net = cv2.dnn.readNet(str(age_model), str(age_proto))
    gender_net = cv2.dnn.readNet(str(gender_model), str(gender_proto))
    return age_net, gender_net


def prever_idade_genero_caffe(face_bgr, age_net, gender_net):
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MEAN_VALUES_CAFFE, swapRB=False)
    gender_net.setInput(blob)
    g_preds = gender_net.forward()[0]
    g_idx = int(np.argmax(g_preds))

    age_net.setInput(blob)
    a_preds = age_net.forward()[0]
    a_idx = int(np.argmax(a_preds))

    return GENDER_LABELS[g_idx], float(g_preds[g_idx]), AGE_BUCKETS[a_idx], float(a_preds[a_idx])


def quantizar_pesos_int8(pesos):
    max_val = float(np.max(np.abs(pesos)))
    scale = max_val / 127.0 if max_val > 0 else 1.0
    pesos_int8 = np.clip(np.round(pesos / scale), -128, 127).astype(np.int8)
    pesos_dequant = pesos_int8.astype(np.float32) * scale
    return pesos_int8, scale, pesos_dequant


def obter_pesos_rede_neural():
    caminho = MODELOS_DIR / "modelo_cnn_digitos.joblib"
    if not caminho.exists():
        caminho = MODELOS_DIR / "mobilenetv2_finetuned.joblib"
    if caminho.exists():
        clf = joblib.load(caminho)
        if isinstance(clf, dict) and "modelo" in clf:
            clf = clf["modelo"]
        return np.concatenate([c.flatten() for c in clf.coefs_])

    np.random.seed(42)
    w1 = np.random.randn(202, 128).astype(np.float32) * 0.05
    w2 = np.random.randn(128, 64).astype(np.float32) * 0.05
    w3 = np.random.randn(64, 10).astype(np.float32) * 0.05
    return np.concatenate([w1.flatten(), w2.flatten(), w3.flatten()])
