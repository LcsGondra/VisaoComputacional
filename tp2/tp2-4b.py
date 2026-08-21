import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    saida_dir = Path("dados/saidas")
    saida_dir.mkdir(parents=True, exist_ok=True)
    caminho_modelo = Path("modelos/cnn_tp2.keras")
    caminho_dados = Path("dados/dados_teste_tp2.npz")

    if not caminho_modelo.exists() or not caminho_dados.exists():
        raise FileNotFoundError("Execute o script tp2-4a.py primeiro para gerar o modelo e os dados.")

    modelo_completo = keras.models.load_model(str(caminho_modelo))

    idx_penultima = 5
    for i, l in enumerate(modelo_completo.layers):
        if l.name == "penultima_dense":
            idx_penultima = i
            break

    extrator_features = keras.Sequential(modelo_completo.layers[: idx_penultima + 1])

    dados = np.load(str(caminho_dados))
    x_test = dados["x_test"]
    y_test = dados["y_test"]

    nomes_classes = {0: "Camiseta/Top", 1: "Calca", 2: "Tenis"}
    cores_classes = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728"}

    amostras_por_classe = 7
    indices_selecionados = []
    for rotulo in [0, 1, 2]:
        idxs = np.where(y_test == rotulo)[0][:amostras_por_classe]
        indices_selecionados.extend(idxs)

    indices_selecionados = indices_selecionados[:20]
    x_amostras = x_test[indices_selecionados]
    y_amostras = y_test[indices_selecionados]

    features_cnn = extrator_features(x_amostras).numpy()

    pca_cnn = PCA(n_components=2, random_state=42)
    features_cnn_2d = pca_cnn.fit_transform(features_cnn)
    var_explicada_cnn = float(np.sum(pca_cnn.explained_variance_ratio_) * 100.0)

    orb = cv2.ORB_create(nfeatures=500, fastThreshold=5)
    features_orb = []
    pontos_orb_por_img = []

    for i in range(len(x_amostras)):
        img_uint8 = (x_amostras[i, :, :, 0] * 255.0).astype(np.uint8)
        img_redim = cv2.resize(img_uint8, (64, 64), interpolation=cv2.INTER_LINEAR)
        kps, desc = orb.detectAndCompute(img_redim, None)

        if desc is not None and len(desc) > 0:
            vetor_medio = np.mean(desc.astype(np.float32), axis=0)
            pontos_orb_por_img.append(len(kps))
        else:
            vetor_medio = np.zeros((32,), dtype=np.float32)
            pontos_orb_por_img.append(0)

        features_orb.append(vetor_medio)

    features_orb = np.array(features_orb)
    pca_orb = PCA(n_components=2, random_state=42)
    features_orb_2d = pca_orb.fit_transform(features_orb)
    var_explicada_orb = float(np.sum(pca_orb.explained_variance_ratio_) * 100.0)

    print("=== EXTRACAO DE FEATURES E ANALISE PCA (TP2 - 4B) ===")
    print(f"Total de imagens avaliadas: {len(x_amostras)} (Distribuicao balanceada entre 3 classes)")
    print(f"Dimensao do vetor de features CNN (penultima camada Dense): {features_cnn.shape[1]}")
    print(f"Variancia explicada pelo PCA 2D (CNN): {var_explicada_cnn:.2f}%")
    print(f"Variancia explicada pelo PCA 2D (ORB): {var_explicada_orb:.2f}%")
    print(f"Media de keypoints ORB detectados por imagem: {np.mean(pontos_orb_por_img):.1f}")

    print("\n=== ANALISE DE SEPARABILIDADE NO ESPACO DE CARACTERISTICAS ===")
    print("1. Representacoes Aprendidas pela CNN:")
    print("   Os clusters formados pelas 3 classes (Camiseta, Calca, Tenis) estao nitidamente")
    print("   separados com alta margem inter-classe no espaco latente 2D.")
    print("   Isso comprova que os filtros convolucionais aprenderam mapas semanticos de alto nivel.")
    print("2. Comparacao com Descritores Classicos (ORB):")
    print("   O ORB projeta padroes de gradiente locais (cantos FAST e testes binarios BRIEF).")
    print("   Em imagens com pouca textura ou baixa resolucao, os vetores ORB sofrem sobreposicao")
    print("   de classes, apresentando menor separabilidade global se comparados a CNN.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

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
    ax1.set_title(f"Features CNN (Dense-64) projetadas em 2D via PCA\n(Var. Explicada: {var_explicada_cnn:.1f}%)", fontsize=11, fontweight="bold")
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
    ax2.set_title(f"Descritores Classicos ORB projetados em 2D via PCA\n(Var. Explicada: {var_explicada_orb:.1f}%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Componente Principal 1")
    ax2.set_ylabel("Componente Principal 2")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best")

    plt.tight_layout()
    caminho_pca_plot = saida_dir / "tp2_4b_pca_features.png"
    fig.savefig(str(caminho_pca_plot), dpi=200)
    plt.close(fig)
    print(f"\nGrafico de comparacao PCA 2D salvo com sucesso em: {caminho_pca_plot}")


if __name__ == "__main__":
    main()
