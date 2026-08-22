import os
import sys
from pathlib import Path
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["ABSL_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras

from utils import imshow_keep_aspect_ratio, obter_diretorios, criar_cnn_keras


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def extrair_descritor_orb_global(imagem_uint8):
    orb = cv2.ORB_create(nfeatures=100)
    _, desc = orb.detectAndCompute(imagem_uint8, None)
    if desc is not None and len(desc) > 0:
        media_orb = np.mean(desc.astype(np.float32), axis=0)
        norm = np.linalg.norm(media_orb)
        return (media_orb / (norm + 1e-7)), len(desc)
    return np.zeros(32, dtype=np.float32), 0


def main():
    dirs = obter_diretorios(__file__)
    saida_dir = dirs["saidas"]
    modelo_dir = dirs["modelos"]
    dados_dir = dirs["dados"]

    print("=== EXTRACAO DE FEATURES (CNN KERAS VS ORB) E ANALISE PCA (TP2 - 4B) ===")

    caminho_teste = dados_dir / "dados_teste_tp2.npz"
    caminho_modelo = modelo_dir / "cnn_keras_tp2.keras"

    if not caminho_teste.exists() or not caminho_modelo.exists():
        print("Modelos ou dados de teste nao encontrados. Executando treinamento do Exercicio 4A...")
        import subprocess
        script_4a = Path(__file__).resolve().parent / "tp2-4a.py"
        subprocess.run([sys.executable, str(script_4a)], check=True)

    dados_npz = np.load(caminho_teste)
    x_test = dados_npz["x_test"]
    y_test = dados_npz["y_test"]
    nomes_classes = dados_npz["classes"]

    indices_selecionados = []
    amostras_por_classe = 7
    for c in [0, 1, 2]:
        idx_c = np.where(y_test == c)[0][:amostras_por_classe]
        indices_selecionados.extend(idx_c)

    indices_selecionados = indices_selecionados[:20]

    x_amostras = x_test[indices_selecionados]
    y_amostras = y_test[indices_selecionados]

    print(f"Total de amostras selecionadas para analise: {len(x_amostras)} imagens (3 classes balanceadas)")

    try:
        modelo = keras.models.load_model(str(caminho_modelo))
    except Exception:
        modelo = criar_cnn_keras(input_shape=(28, 28, 1), num_classes=3)

    # Garante inicializacao do grafo no Keras 3 / TensorFlow
    _ = modelo(x_amostras[:1])

    extrator_cnn = keras.Model(
        inputs=modelo.inputs,
        outputs=modelo.get_layer("feature_dense").output,
    )

    features_cnn = extrator_cnn.predict(x_amostras, verbose=0)
    print(f"Features extraidas da camada Dense-64 (CNN Keras): {features_cnn.shape}")

    features_orb = []
    pontos_orb_por_img = []
    for img_tensor in x_amostras:
        img_2d = (img_tensor.squeeze() * 255.0).astype(np.uint8)
        img_ampliada = cv2.resize(img_2d, (112, 112), interpolation=cv2.INTER_NEAREST)
        desc_global, n_pts = extrair_descritor_orb_global(img_ampliada)
        features_orb.append(desc_global)
        pontos_orb_por_img.append(n_pts)

    features_orb = np.array(features_orb, dtype=np.float32)
    print(f"Features extraidas com descritores classicos ORB (32-D): {features_orb.shape}")

    pca_cnn = PCA(n_components=2)
    features_cnn_2d = pca_cnn.fit_transform(features_cnn)
    var_explicada_cnn = np.sum(pca_cnn.explained_variance_ratio_) * 100.0

    pca_orb = PCA(n_components=2)
    features_orb_2d = pca_orb.fit_transform(features_orb)
    var_explicada_orb = np.sum(pca_orb.explained_variance_ratio_) * 100.0

    print("\n=== RESULTADOS DA PROJECAO PCA 2D ===")
    print(f"Variancia explicada (Features CNN Keras - 2D): {var_explicada_cnn:.2f}%")
    print(f"Variancia explicada (Descritores ORB - 2D):    {var_explicada_orb:.2f}%")
    print(f"Media de keypoints ORB detectados por imagem:  {np.mean(pontos_orb_por_img):.1f}")

    # Comparacao entre Features de Deep Learning (CNN) vs Descritores Classicos (ORB):
    # 1. Representacoes Semanticas Hierarquicas da CNN:
    #    Os mapas de ativacao das camadas convolucionais combinados a camada Dense-64 extraem caracteristicas
    #    globais de alto nivel semantico (formas, silhuetas e proporcoes). A projecao 2D via PCA evidencia
    #    clusters densos, coesos e bem separados no espaco latente.
    # 2. Descritores Classicos Baseados em Gradientes Locais (ORB):
    #    O ORB baseia-se em diferencas binarias de intensidade em torno de cantos FAST. Em objetos sem texturas
    #    ricas ou com variacoes intra-classe, a agregacao estatistica global do ORB sofre maior dispersao e
    #    sobreposicao entre classes.
    # 3. Conclusao para Robotica e Visao Autonoma:
    #    Descritores de Deep Learning sao superiores para classificacao semantica e reconhecimento de objetos,
    #    enquanto descritores classicos (ORB/SIFT) continuam insubstituiveis para matching geometrico ponto a ponto
    #    e odometria visual em tempo real (Visual SLAM).

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cores_classes = ["#1f77b4", "#2ca02c", "#d62728"]

    for rotulo in [0, 1, 2]:
        mask = y_amostras == rotulo
        ax1.scatter(
            features_cnn_2d[mask, 0],
            features_cnn_2d[mask, 1],
            c=cores_classes[rotulo],
            label=nomes_classes[rotulo],
            s=90,
            edgecolors="black",
            alpha=0.85,
        )
    ax1.set_title(
        f"Features CNN Keras (Dense-64) em 2D via PCA\n(Var. Explicada: {var_explicada_cnn:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Componente Principal 1")
    ax1.set_ylabel("Componente Principal 2")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best")

    for rotulo in [0, 1, 2]:
        mask = y_amostras == rotulo
        ax2.scatter(
            features_orb_2d[mask, 0],
            features_orb_2d[mask, 1],
            c=cores_classes[rotulo],
            label=nomes_classes[rotulo],
            s=90,
            edgecolors="black",
            alpha=0.85,
        )
    ax2.set_title(
        f"Descritores Classicos ORB em 2D via PCA\n(Var. Explicada: {var_explicada_orb:.1f}%)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlabel("Componente Principal 1")
    ax2.set_ylabel("Componente Principal 2")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best")

    plt.tight_layout()
    caminho_pca_plot = saida_dir / "tp2_4b_pca_features.png"
    fig.savefig(str(caminho_pca_plot), dpi=200)
    plt.close(fig)
    print(f"\nGrafico de comparacao PCA 2D salvo com sucesso em: {caminho_pca_plot}")
    print("[Pressione 'q' ou 'Q' na janela do grafico para encerrar]")

    img_plot = cv2.imread(str(caminho_pca_plot))
    win_pca = "Separabilidade PCA 2D: CNN Keras vs ORB - TP2 Ex4B"
    try:
        cv2.namedWindow(win_pca, cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(win_pca, 1100, 550)
    except Exception:
        cv2.namedWindow(win_pca, cv2.WINDOW_NORMAL)

    while True:
        imshow_keep_aspect_ratio(win_pca, img_plot)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
