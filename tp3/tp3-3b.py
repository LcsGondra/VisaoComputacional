import time
import warnings
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
from utils import (
    ensure_dirs,
    carregar_dataset_digitos,
    aplicar_aumento_dados_digito,
    extrair_features_convolucionais,
    salvar_figura,
    MODELOS_DIR,
    SAIDAS_DIR,
)

# ==============================================================================
# Discussao sobre Domain Gap (Treino Limpo vs Captura Real):
# Em datasets padronizados de treino (como o MNIST limpo), as imagens sao perfeitamente
# centralizadas, sem ruido de fundo e com tracos uniformes.
# Ao transferir o modelo para capturas reais de camera em papel fotografado, ocorrem
# sombras, reflexos, inclinacao de angulo e artefatos de binarizacao por Otsu (Domain Shift).
# Essa diferenca na distribuicao estatistica dos dados reduz a acuracia do modelo original.
# A aplicacao de Data Augmentation (rotacoes, zoom, translacoes) expande a variabilidade
# do treinamento, tornando a rede neural robusta e diminuindo o impacto do Domain Gap.
# ==============================================================================


def gerar_ou_carregar_digitos_reais():
    """Gera 10 imagens simulando fotografias reais de digitos 0-9 em papel."""
    X_tr, y_tr, _, _, _ = carregar_dataset_digitos(limite_amostras=500)
    amostras_reais = []
    labels_reais = list(range(10))

    for d in range(10):
        # Seleciona uma imagem do digito
        idx = np.where(y_tr == d)[0][0]
        base_img = X_tr[idx]

        # Simula papel fotografado: inverte fundo para branco, adiciona ruido e iluminacao nao uniforme
        papel = np.ones((80, 80), dtype=np.float32) * 0.95
        dig_patch = cv2.resize(base_img, (45, 45))

        # Insere digito preto sobre fundo branco com leve inclinacao
        rot_mat = cv2.getRotationMatrix2D((22, 22), np.random.uniform(-10, 10), 1.0)
        dig_patch = cv2.warpAffine(dig_patch, rot_mat, (45, 45))
        papel[18:63, 18:63] -= dig_patch * 0.85
        papel += np.random.normal(0, 0.03, papel.shape).astype(np.float32)
        papel = np.clip(papel, 0.0, 1.0)

        amostras_reais.append((papel * 255).astype(np.uint8))

    return amostras_reais, labels_reais


def preprocessar_imagem_real(img_papel_gray):
    # 1. Escala de cinza (ja em uint8) -> 2. Binarizacao Otsu (invertida para digito branco em fundo preto)
    blur = cv2.GaussianBlur(img_papel_gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Resize 28x28 -> 4. Normalizacao para [0, 1]
    res = cv2.resize(otsu, (28, 28), interpolation=cv2.INTER_AREA)
    norm = res.astype(np.float32) / 255.0
    return norm


def executar_experimento_domain_gap():
    ensure_dirs()
    print("1. Carregando dataset limpo de treino e gerando 10 amostras reais fotografadas (0-9)...")
    X_train_limpo, y_train_limpo, X_test_limpo, y_test_limpo, _ = carregar_dataset_digitos(limite_amostras=2000)
    amostras_reais_img, y_reais = gerar_ou_carregar_digitos_reais()

    # Pre-processa os 10 digitos reais com Otsu + resize + normalizacao
    X_reais_proc = np.array([preprocessar_imagem_real(img) for img in amostras_reais_img], dtype=np.float32)

    # 2. Treina modelo baseline SEM Data Augmentation
    print("2. Treinando CNN baseline (SEM Data Augmentation)...")
    F_train_limpo = extrair_features_convolucionais(X_train_limpo)
    F_reais = extrair_features_convolucionais(X_reais_proc)

    clf_base = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=15, random_state=42)
    clf_base.fit(F_train_limpo, y_train_limpo)

    preds_base = clf_base.predict(F_reais)
    acc_base_real = accuracy_score(y_reais, preds_base) * 100.0

    # 3. Treina modelo COM Data Augmentation (rotacao +-15 graus, zoom +-10%)
    print("3. Treinando CNN COM Data Augmentation...")
    X_aug, y_aug = [], []
    for img, lbl in zip(X_train_limpo, y_train_limpo):
        X_aug.append(img)
        y_aug.append(lbl)
        X_aug.append(aplicar_aumento_dados_digito(img))
        y_aug.append(lbl)

    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int32)
    F_train_aug = extrair_features_convolucionais(X_aug)

    clf_aug = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=15, random_state=42)
    clf_aug.fit(F_train_aug, y_aug)

    preds_aug = clf_aug.predict(F_reais)
    acc_aug_real = accuracy_score(y_reais, preds_aug) * 100.0

    print("\nResultados nas 10 Amostras Reais Fotografadas:")
    print(f"Acuracia SEM Data Augmentation : {acc_base_real:.1f}%")
    print(f"Acuracia COM Data Augmentation : {acc_aug_real:.1f}% (Ganho de +{acc_aug_real - acc_base_real:.1f}%)")

    # Salva modelo treinado
    caminho_modelo = MODELOS_DIR / "modelo_cnn_digitos.joblib"
    joblib.dump(clf_aug, caminho_modelo)

    # 4. Plota painel com os 10 digitos reais, binarizacao e predicao sobreposta
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(10):
        r, c = i // 5, i % 5
        vis = cv2.cvtColor(amostras_reais_img[i], cv2.COLOR_GRAY2BGR)
        cor = (0, 200, 0) if preds_aug[i] == y_reais[i] else (0, 0, 220)
        txt = f"Real: {y_reais[i]} | Pred: {preds_aug[i]}"
        axs[r, c].imshow(vis)
        axs[r, c].set_title(txt, color="darkgreen" if preds_aug[i] == y_reais[i] else "red", fontweight="bold", fontsize=11)
        axs[r, c].axis("off")

    plt.suptitle(f"Deteccao em Amostras Reais (Acuracia: {acc_aug_real:.1f}%)", fontweight="bold", fontsize=13)
    salvar_figura(SAIDAS_DIR / "tp3_3b_digitos_reais_predicoes.png")


def main():
    executar_experimento_domain_gap()


if __name__ == "__main__":
    main()
