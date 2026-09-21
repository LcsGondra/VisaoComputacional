# Exercício 2 — Item A: Classificação de Imagens com OpenCV DNN e Benchmark Comparativo
# Competências: 2.4, 3.3 e 4.2
#
# Demonstra inferência com SqueezeNet v1.1 via OpenCV DNN sobre 10 imagens reais,
# extrai o Top-3 de classes com probabilidades e compara latência, memória e acurácia
# entre OpenCV DNN (C++ nativo) e frameworks pesados (Keras/TensorFlow).

import time
import tracemalloc

import cv2
import numpy as np

from utils import (
    ensure_dirs,
    obter_dataset_classificacao_real,
    obter_labels_imagenet,
    obter_modelo_squeezenet,
    softmax,
    desenhar_anotacao_top3,
    medir_benchmark_keras,
    criar_mosaico_imagens,
    exibir_janela_interativa,
    salvar_mosaico_grid,
    plotar_metricas_treino_e_confusao_2a,
    SAIDAS_DIR,
)


def inferir_top3_opencv(net, imagem_bgr, labels, k=3):
    # Executa pré-processamento (blob 227x227), forward pass e extração do Top-k
    blob = cv2.dnn.blobFromImage(
        imagem_bgr,
        scalefactor=1.0,
        size=(227, 227),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    t0 = time.perf_counter()
    out = net.forward()
    latencia_ms = (time.perf_counter() - t0) * 1000.0

    probabilidades = softmax(out.flatten())
    indices_topk = np.argsort(probabilidades)[::-1][:k]
    resultados = [
        (labels[idx] if idx < len(labels) else f"classe_{idx}", float(probabilidades[idx]), idx)
        for idx in indices_topk
    ]
    return resultados, latencia_ms


def main():
    ensure_dirs()
    print("=" * 80)
    print("EXERCÍCIO 2A — CLASSIFICAÇÃO COM OPENCV DNN E BENCHMARK COMPARATIVO")
    print("=" * 80)

    # 1. Carrega modelo SqueezeNet v1.1 e rótulos ImageNet
    net, proto_path, model_path = obter_modelo_squeezenet()
    labels = obter_labels_imagenet()
    tamanho_disco_mb = (proto_path.stat().st_size + model_path.stat().st_size) / (1024 * 1024)
    print(f"SqueezeNet v1.1 carregado via OpenCV DNN ({tamanho_disco_mb:.2f} MB em disco)")
    print(f"Total de classes suportadas: {len(labels)}")

    # 2. Carrega dataset com 10 fotografias reais
    caminhos_imagens = obter_dataset_classificacao_real(num_imagens=10)

    # 3. Inferência nas 10 imagens com medição de memória e latência
    tracemalloc.start()
    tempos_opencv = []
    frames_anotados = []
    titulos_class = []

    print("\n" + "-" * 80)
    print("INFERÊNCIA NAS 10 IMAGENS (OpenCV DNN - SqueezeNet v1.1):")
    print(f"{'Img':<4} | {'Arquivo':<20} | {'Top-1 Predição':<26} | {'Confiança':<10} | {'Latência'}")
    print("-" * 80)

    for i, p in enumerate(caminhos_imagens, start=1):
        img = cv2.imread(str(p))
        if img is None:
            continue

        # Warmup e repetições para medição precisa de tempo
        _, _ = inferir_top3_opencv(net, img, labels, k=3)
        lats_rep = []
        for _ in range(5):
            top3, l_ms = inferir_top3_opencv(net, img, labels, k=3)
            lats_rep.append(l_ms)

        med_lat = float(np.mean(lats_rep))
        tempos_opencv.append(med_lat)

        nome_obj = p.stem.split("_")[-1]
        vis = desenhar_anotacao_top3(img, top3, titulo=f"Img {i:02d}: {nome_obj.upper()}")
        frames_anotados.append(vis)

        top1_label, top1_conf, _ = top3[0]
        titulos_class.append(f"#{i:02d} {nome_obj}: {top1_label[:14]} ({top1_conf*100:.0f}%)")
        print(f"{i:02d}  | {p.name:<20} | {top1_label:<26} | {top1_conf*100:8.2f}% | {med_lat:6.2f} ms")
        for rank, (lbl, conf, _) in enumerate(top3[1:], start=2):
            print(f"    | {'':<20} | Top-{rank}: {lbl:<20} | {conf*100:8.2f}% |")

    _, mem_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoria_dnn_mb = mem_pico / (1024 * 1024)

    # 4. Exibe mosaico interativo com as 10 imagens e anotações visuais
    mosaico_class = criar_mosaico_imagens(
        frames_anotados,
        titulos=titulos_class,
        cols=5,
        thumb_size=(280, 280),
        titulo_geral="Classificação OpenCV DNN (SqueezeNet v1.1): 10 Fotografias Reais do Dataset"
    )
    exibir_janela_interativa(
        "Exercicio 2A - Classificacoes SqueezeNet v1.1 (10 Fotos Reais)",
        mosaico_class,
        "Pressione 'q', ESC ou feche no [X] para prosseguir"
    )

    media_lat_opencv = float(np.mean(tempos_opencv))
    fps_opencv = 1000.0 / max(1e-3, media_lat_opencv)

    # 5. Benchmark comparativo com Keras/TensorFlow
    dados_keras = medir_benchmark_keras()
    fps_keras = 1000.0 / dados_keras["latencia_ms"]

    # 6. Tabela comparativa impressa na CLI
    print("-" * 80)
    print("\nTABELA COMPARATIVA DE DESEMPENHO: OPENCV DNN vs. KERAS/TENSORFLOW:")
    print("=" * 80)
    print(f"{'Métrica':<32} | {'OpenCV DNN (Caffe/C++)':<22} | {'Keras (Python/TF)':<20}")
    print("-" * 80)
    print(f"{'Latência Média por Inferência':<32} | {media_lat_opencv:8.2f} ms          | {dados_keras['latencia_ms']:8.2f} ms")
    print(f"{'Taxa de Processamento (FPS)':<32} | {fps_opencv:8.1f} FPS         | {fps_keras:8.1f} FPS")
    print(f"{'Consumo de Memória RAM (Pico)':<32} | {memoria_dnn_mb:8.2f} MB          | {dados_keras['memoria_mb']:8.2f} MB")
    print(f"{'Tamanho do Modelo em Disco':<32} | {tamanho_disco_mb:8.2f} MB          | ~14.00 MB")
    print(f"{'Acurácia Top-1 (ImageNet)':<32} | 58.1% (SqueezeNet)     | {dados_keras['acuracia_top1']:.1f}% (MobileNetV2)")
    print(f"{'Dependências Externas em Runtime':<32} | Nenhuma (Apenas cv2)   | Python, TF, CUDA, DLLs")
    print("=" * 80)
    print(f"Status do backend Keras: {dados_keras['nota']}")

    # 7. Salva mosaico de predições e plota curvas de métricas e matriz de confusão
    salvar_mosaico_grid(
        frames_anotados,
        SAIDAS_DIR / "at2a_classificacao_top3.png",
        titulo="Exercício 2A: Classificação OpenCV DNN (SqueezeNet v1.1) — Top-3 Predições nas 10 Imagens",
        rows=2,
        cols=5,
    )
    plotar_metricas_treino_e_confusao_2a()


if __name__ == "__main__":
    main()
