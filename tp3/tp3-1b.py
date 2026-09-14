import time
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils import (
    ensure_dirs,
    carregar_ou_extrair_dataset_hog,
    extrair_features_hog,
    sliding_window_multiescala,
    obter_ou_gerar_cena_teste,
    salvar_figura,
    MODELOS_DIR,
    SAIDAS_DIR,
)

# ==============================================================================
# Custo computacional: Janela Deslizante vs. YOLO
# Na abordagem de janela deslizante (Sliding Window), o algoritmo precisa recortar
# e extrair descritores HOG de milhares de sub-regioes em diversas escalas da imagem.
# Isso torna o processo computacionalmente caro e lento (tempo na faixa de centenas
# de milissegundos a segundos por frame), inviabilizando tempo real em muitas aplicacoes.
# Em contrapartida, detectores como o YOLO (You Only Look Once) operam em uma unica
# passada direta pela rede neural convolucional (single-shot), processando a imagem
# de forma global e realizando a deteccao com alta taxa de quadros (>30 a 60 FPS).
# ==============================================================================


def treinar_e_avaliar_svm():
    ensure_dirs()
    print("1. Carregando dataset (100 positivas e 100 negativas em 64x128)...")
    imagens, rotulos, origem = carregar_ou_extrair_dataset_hog(num_positivos=100, num_negativos=100)
    print(f"   Origem dos dados: {origem} (Total: {len(imagens)} amostras)")

    print("2. Extraindo features HOG (3780 dimensoes)...")
    X = extrair_features_hog(imagens, win_size=(64, 128))
    y = np.array(rotulos, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print("3. Treinando SVM com kernel RBF...")
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    svm.fit(X_train, y_train)

    # Avaliacao no conjunto de teste
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nResultados no Conjunto de Teste:")
    print(f"Acuracia : {acc * 100:.2f}%")
    print(f"Precisao : {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print("\nMatriz de Confusao:")
    print(cm)
    print(f"(TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]})")

    # Salva o modelo treinado
    caminho_modelo = MODELOS_DIR / "modelo_hog_svm.joblib"
    joblib.dump(svm, caminho_modelo)
    print(f"\n[+] Modelo salvo em: {caminho_modelo.name}")

    # 4. Janela Deslizante na imagem de teste
    print("\n4. Aplicando janela deslizante na imagem de teste...")
    cena_teste = obter_ou_gerar_cena_teste()

    t0 = time.perf_counter()
    boxes, scores = sliding_window_multiescala(
        cena_teste,
        svm,
        win_size=(64, 128),
        step_size=8,
        scale_factor=1.2,
        min_confidence=0.70,
    )
    tempo_janela = (time.perf_counter() - t0) * 1000
    print(f"   {len(boxes)} caixas encontradas em {tempo_janela:.1f} ms")

    # NMS para filtrar retangulos redundantes
    indices_nms = cv2.dnn.NMSBoxes(
        [[x, y, w, h] for (x, y, w, h) in boxes],
        scores,
        score_threshold=0.70,
        nms_threshold=0.30,
    ) if boxes else []

    img_resultado = cena_teste.copy()
    if len(indices_nms) > 0:
        for idx in np.array(indices_nms).flatten():
            x, y, w, h = boxes[int(idx)]
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_resultado, f"Alvo: {scores[int(idx)]:.2f}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Plota resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.imshow(cm, cmap="Blues")
    ax1.set_title(f"Matriz de Confusao (Acc: {acc*100:.1f}%)", fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Fundo", "Alvo"])
    ax1.set_yticklabels(["Fundo", "Alvo"])
    for r in range(2):
        for c in range(2):
            ax1.text(c, r, str(cm[r, c]), ha="center", va="center", color="black" if cm[r, c] < len(y_test)//2 else "white", fontweight="bold")

    ax2.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"Deteccoes apos Janela Deslizante + NMS ({len(indices_nms)} objetos)", fontweight="bold")
    ax2.axis("off")

    salvar_figura(SAIDAS_DIR / "tp3_1b_svm_deteccoes.png")


def main():
    treinar_e_avaliar_svm()


if __name__ == "__main__":
    main()
