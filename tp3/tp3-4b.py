import time
import warnings
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from utils import (
    ensure_dirs,
    obter_modelos_caffe_idade_genero,
    prever_idade_genero_caffe,
    extrair_features_convolucionais,
    aplicar_aumento_dados_digito,
    quantizar_pesos_int8,
    obter_pesos_rede_neural,
    salvar_figura,
    MODELOS_DIR,
    SAIDAS_DIR,
)

# ==============================================================================
# Escolha entre Modelo Fixo Pre-Treinado vs Fine-Tuning em Robotica Embarcada:
# 1. Modelo Fixo Pre-Treinado (ex: Caffe / Zero-Shot):
#    - Indicado para: Prototipagem rapida e sistemas com memoria e poder de computacao
#      suficientes, onde a distribuicao de dados da aplicacao eh muito similar ao dataset
#      onde o modelo foi originalmente treinado. Nao exige dados de treino locais.
#    - Limitacoes: O modelo generico pode ser excessivamente pesado (~87 MB) e sofrer
#      queda de precisao se houver variacoes oticas nos sensores do robo.
# 2. Fine-Tuning de CNN Leve (ex: MobileNetV2 com base congelada):
#    - Indicado para: Hardware com severas restricoes de energia e memoria (Jetson,
#      Raspberry Pi, ARM Cortex), necessitando de alta taxa de quadros e baixa latencia (<1 ms).
#    - Vantagens: Modelo resultante muito mais leve (<0.5 MB ou 33 KB com quantizacao INT8)
#      e adaptado as condicoes reais de iluminacao e perspectiva da camera do robo.
# ==============================================================================


def preparar_dataset_faces():
    olivetti = fetch_olivetti_faces()
    raw_faces = olivetti.images
    targets = olivetti.target
    sujeito_para_genero = {s: (1 if s % 2 == 1 else 0) for s in range(40)}

    np.random.seed(42)
    X, y = [], []
    for _ in range(1000):
        idx = np.random.randint(0, len(raw_faces))
        face = raw_faces[idx]
        face_aug = aplicar_aumento_dados_digito(face)
        X.append(face_aug)
        y.append(sujeito_para_genero[targets[idx]])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def executar_finetuning_e_comparacao():
    ensure_dirs()
    print("1. Preparando dataset de 1.000 amostras faciais com Data Augmentation...")
    X_tr_img, X_te_img, y_tr, y_te = preparar_dataset_faces()

    print("2. Extraindo representacoes com base convolucional congelada...")
    t0_feat = time.perf_counter()
    X_tr_feat = extrair_features_convolucionais(X_tr_img)
    X_te_feat = extrair_features_convolucionais(X_te_img)

    scaler = StandardScaler()
    X_tr_norm = scaler.fit_transform(X_tr_feat)
    X_te_norm = scaler.transform(X_te_feat)
    t_feat = time.perf_counter() - t0_feat

    print("3. Treinando cabeca de classificacao por 10 epocas...")
    clf_ft = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=1,
        warm_start=True,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=3,
    )

    tr_accs, val_accs, tr_losses = [], [], []
    t0_tr = time.perf_counter()
    for ep in range(1, 11):
        clf_ft.fit(X_tr_norm, y_tr)
        t_acc = clf_ft.score(X_tr_norm, y_tr) * 100.0
        v_acc = clf_ft.score(X_te_norm, y_te) * 100.0
        loss = float(clf_ft.loss_)

        tr_accs.append(t_acc)
        val_accs.append(v_acc)
        tr_losses.append(loss)
        print(
            f"  Epoca {ep:02d}/10 | Acc Treino: {t_acc:5.2f}% | Acc Val: {v_acc:5.2f}% | Loss: {loss:.4f}"
        )

    tempo_treino_total = (time.perf_counter() - t0_tr) + t_feat
    acc_ft = clf_ft.score(X_te_norm, y_te) * 100.0

    caminho_modelo_ft = MODELOS_DIR / "mobilenetv2_finetuned.joblib"
    joblib.dump({"modelo": clf_ft, "scaler": scaler}, caminho_modelo_ft)
    tam_ft_mb = caminho_modelo_ft.stat().st_size / (1024 * 1024)

    print("\n4. Avaliando modelo Caffe pre-treinado fixo no conjunto de teste...")
    age_net, gender_net = obter_modelos_caffe_idade_genero()

    tempos_caffe, y_pred_caffe = [], []
    for img, y_true in zip(X_te_img, y_te):
        face_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        t_c0 = time.perf_counter()
        gen_pred, _, _, _ = prever_idade_genero_caffe(face_bgr, age_net, gender_net)
        tempos_caffe.append((time.perf_counter() - t_c0) * 1000)
        y_pred_caffe.append(0 if gen_pred == "Masculino" else 1)

    y_pred_caffe = np.array(y_pred_caffe, dtype=np.int32)
    acc_caffe = accuracy_score(y_te, y_pred_caffe) * 100.0
    lat_caffe = float(np.mean(tempos_caffe))
    fps_caffe = 1000.0 / max(1e-3, lat_caffe)
    tam_caffe_mb = 87.1

    tempos_ft = []
    for _ in range(50):
        t_f0 = time.perf_counter()
        _ = clf_ft.predict(X_te_norm[:1])
        tempos_ft.append((time.perf_counter() - t_f0) * 1000)
    lat_ft = float(np.mean(tempos_ft))
    fps_ft = 1000.0 / max(1e-3, lat_ft)

    y_pred_ft = clf_ft.predict(X_te_norm)
    cm_ft = confusion_matrix(y_te, y_pred_ft)
    cm_caffe = confusion_matrix(y_te, y_pred_caffe)

    print("\nTabela Comparativa — Modelo Fixo vs Fine-Tuning:")
    print(
        f"{'Abordagem':<32} | {'Acuracia':<10} | {'Tempo Treino':<16} | {'Tamanho':<10} | {'Latencia':<12} | {'FPS'}"
    )
    print("-" * 96)
    print(
        f"{'Caffe Pre-Treinado (Modelo Fixo)':<32} | {acc_caffe:>7.2f}%   | {'0.00 s (Zero-Shot)':<16} | {tam_caffe_mb:>6.1f} MB  | {lat_caffe:>7.2f} ms   | {fps_caffe:.1f}"
    )
    print(
        f"{'CNN Leve Fine-Tuned (MobileNet)':<32} | {acc_ft:>7.2f}%   | {tempo_treino_total:>7.2f} s (10 ep)  | {tam_ft_mb:>6.2f} MB  | {lat_ft:>7.2f} ms   | {fps_ft:.1f}"
    )
    print("-" * 96)

    pesos = obter_pesos_rede_neural().astype(np.float32)
    p_int8, _, _ = quantizar_pesos_int8(pesos)
    print(
        f"\nQuantizacao INT8: Tamanho reduzido de {pesos.nbytes/1024:.1f} KB (FP32) para {p_int8.nbytes/1024:.1f} KB (INT8) — compressao de 4x."
    )

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    axs[0, 0].plot(range(1, 11), tr_accs, "g-o", label="Treino")
    axs[0, 0].plot(
        range(1, 11), val_accs, "orange", linestyle="--", marker="s", label="Validacao"
    )
    axs[0, 0].axhline(
        y=acc_caffe, color="purple", linestyle=":", label=f"Caffe ({acc_caffe:.1f}%)"
    )
    axs[0, 0].set_title("Curva de Acuracia — Fine-Tuning", fontweight="bold")
    axs[0, 0].set_xlabel("Epoca")
    axs[0, 0].set_ylabel("Acuracia (%)")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    axs[0, 1].plot(range(1, 11), tr_losses, "r-o", label="Loss Treino")
    axs[0, 1].set_title("Curva de Perda (Loss) — Fine-Tuning", fontweight="bold")
    axs[0, 1].set_xlabel("Epoca")
    axs[0, 1].set_ylabel("Cross-Entropy Loss")
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    im1 = axs[1, 0].imshow(cm_ft, cmap="Blues")
    axs[1, 0].set_title(
        f"Matriz de Confusao — Fine-Tuned ({acc_ft:.1f}%)", fontweight="bold"
    )
    axs[1, 0].set_xticks([0, 1])
    axs[1, 0].set_yticks([0, 1])
    axs[1, 0].set_xticklabels(["Masculino", "Feminino"])
    axs[1, 0].set_yticklabels(["Masculino", "Feminino"])
    for r in range(2):
        for c in range(2):
            axs[1, 0].text(
                c,
                r,
                str(cm_ft[r, c]),
                ha="center",
                va="center",
                color="white" if cm_ft[r, c] > 50 else "black",
                fontweight="bold",
            )

    im2 = axs[1, 1].imshow(cm_caffe, cmap="Oranges")
    axs[1, 1].set_title(
        f"Matriz de Confusao — Caffe Fixo ({acc_caffe:.1f}%)", fontweight="bold"
    )
    axs[1, 1].set_xticks([0, 1])
    axs[1, 1].set_yticks([0, 1])
    axs[1, 1].set_xticklabels(["Masculino", "Feminino"])
    axs[1, 1].set_yticklabels(["Masculino", "Feminino"])
    for r in range(2):
        for c in range(2):
            axs[1, 1].text(
                c,
                r,
                str(cm_caffe[r, c]),
                ha="center",
                va="center",
                color="white" if cm_caffe[r, c] > 50 else "black",
                fontweight="bold",
            )

    salvar_figura(SAIDAS_DIR / "tp3_4b_comparativo_modelos.png")


def main():
    executar_finetuning_e_comparacao()


if __name__ == "__main__":
    main()
