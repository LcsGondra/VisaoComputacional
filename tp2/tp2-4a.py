import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def carregar_dados_3_classes():
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.fashion_mnist.load_data()

    classes_alvo = [0, 1, 7]
    nomes_classes = {0: "Camiseta/Top", 1: "Calca", 7: "Tenis"}

    mascara_treino = np.isin(y_train_raw, classes_alvo)
    mascara_teste = np.isin(y_test_raw, classes_alvo)

    x_train = x_train_raw[mascara_treino].astype("float32") / 255.0
    y_train_orig = y_train_raw[mascara_treino]

    x_test = x_test_raw[mascara_teste].astype("float32") / 255.0
    y_test_orig = y_test_raw[mascara_teste]

    mapa_rotulos = {0: 0, 1: 1, 7: 2}
    y_train = np.array([mapa_rotulos[val] for val in y_train_orig], dtype=np.int32)
    y_test = np.array([mapa_rotulos[val] for val in y_test_orig], dtype=np.int32)

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test), nomes_classes


def construir_cnn(input_shape=(28, 28, 1), num_classes=3):
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu", name="penultima_dense"),
            layers.Dropout(0.25),
            layers.Dense(num_classes, activation="softmax", name="camada_saida"),
        ]
    )
    return modelo


def main():
    saida_dir = Path("dados/saidas")
    saida_dir.mkdir(parents=True, exist_ok=True)
    modelo_dir = Path("modelos")
    modelo_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test), nomes_classes = carregar_dados_3_classes()

    print("=== TREINAMENTO DE CNN COM TENSORFLOW/KERAS (TP2 - 4A) ===")
    print(f"Classes selecionadas (Fashion-MNIST reduzido): {list(nomes_classes.values())}")
    print(f"Tamanho do conjunto de treino: {x_train.shape[0]} imagens de {x_train.shape[1]}x{x_train.shape[2]} px")
    print(f"Tamanho do conjunto de teste:  {x_test.shape[0]} imagens")

    np.savez_compressed(
        "dados/dados_teste_tp2.npz",
        x_test=x_test,
        y_test=y_test,
    )

    modelo = construir_cnn()
    modelo.summary()

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    epocas = 10
    batch_size = 64

    historico = modelo.fit(
        x_train,
        y_train,
        epochs=epocas,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1,
    )

    caminho_modelo = modelo_dir / "cnn_tp2.keras"
    modelo.save(str(caminho_modelo))
    print(f"\nModelo salvo com sucesso em: {caminho_modelo}")

    test_loss, test_acc = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"\n=== AVALIACAO NO CONJUNTO DE TESTE ===")
    print(f"Loss final no teste:     {test_loss:.4f}")
    print(f"Acuracia final no teste: {test_acc * 100.0:.2f}%")

    treino_loss = historico.history["loss"]
    val_loss = historico.history["val_loss"]
    treino_acc = historico.history["accuracy"]
    val_acc = historico.history["val_accuracy"]

    gap_acc = float(treino_acc[-1] - val_acc[-1])
    gap_loss = float(val_loss[-1] - treino_loss[-1])

    print("\n=== DIAGNOSTICO DE OVERFITTING NAS CURVAS DE APRENDIZADO ===")
    if gap_loss > 0.15 or gap_acc > 0.08:
        print(f"- Indicio de leve overfitting detectado:")
        print(f"  Gap de acuracia (Treino - Val): {gap_acc * 100.0:.2f}%")
        print(f"  A loss de validacao ({val_loss[-1]:.4f}) comecou a divergir da loss de treino ({treino_loss[-1]:.4f}).")
    else:
        print(f"- Modelo bem regularizado e convergente:")
        print(f"  Gap de acuracia (Treino - Val): {gap_acc * 100.0:.2f}%")
        print(f"  Loss de treino ({treino_loss[-1]:.4f}) e validacao ({val_loss[-1]:.4f}) evoluiram de forma compativel.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(1, epocas + 1), treino_acc, "o-", label="Treino", color="#1f77b4", linewidth=2)
    ax1.plot(range(1, epocas + 1), val_acc, "s-", label="Validacao", color="#ff7f0e", linewidth=2)
    ax1.set_title("Evolucao da Acuracia por Epoca", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Acuracia")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(range(1, epocas + 1), treino_loss, "o-", label="Treino", color="#1f77b4", linewidth=2)
    ax2.plot(range(1, epocas + 1), val_loss, "s-", label="Validacao", color="#d62728", linewidth=2)
    ax2.set_title("Evolucao da Loss por Epoca", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Loss (Cross-Entropy)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    caminho_plot = saida_dir / "tp2_4a_curvas_treino.png"
    fig.savefig(str(caminho_plot), dpi=200)
    plt.close(fig)
    print(f"Grafico de curvas de treino salvo em: {caminho_plot}")


if __name__ == "__main__":
    main()
