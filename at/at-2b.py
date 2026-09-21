# Exercício 2 — Item B: Pipeline Integrativo Completo da Disciplina de Visão Computacional
# Competências: 2.4, 3.3 e 4.2
#
# Este script unifica e encadeia sequencialmente todas as principais técnicas estudadas
# ao longo da disciplina em um único fluxo de percepção robótica contínuo:
#
# Fluxo do Pipeline Integrativo:
# -----------------------------
# Etapa 1 (Calibração Geométrica — Ex 1):
#   Aplica retificação de distorção de lente (cv2.undistort) utilizando os parâmetros intrínsecos K
#   e coeficientes de distorção calculados no Exercício 1A.
# Etapa 2 (Segmentação por Cor — TP1):
#   Converte para espaço de cor HSV e segmenta ROI de interesse com limiares cromáticos e
#   morfologia matemática (eliminação de ruídos e detecção de maior contorno).
# Etapa 3 (Descritores Locais e Pontos-Chave — TP2):
#   Extrai pontos-chave e descritores invariantes ORB (cv2.ORB_create) sobre a ROI segmentada,
#   desenhando os keypoints para rastreamento visual e correspondência esparsa.
# Etapa 4 (Detecção de Objetos Clássica — TP3):
#   Executa detector de pessoas HOG+SVM multiescala (cv2.HOGDescriptor) ou Haar Cascade,
#   identificando a presença de pedestres ou faces no cenário.
# Etapa 5 (Classificação Profunda — Ex 2A):
#   Executa inferência com rede neural profunda via OpenCV DNN (SqueezeNet v1.1) sobre a ROI recortada,
#   extraindo as classes mais prováveis e probabilidades.
#
# Saídas:
# - Painel de telemetria com latência de cada etapa em milissegundos.
# - Frame final com sobreposição integrada de todas as anotações gráficas.
# - Arquivo de saída salvo em 'dados/saidas/at2b_pipeline_integrado.png'.

from pathlib import Path
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    ensure_dirs,
    obter_frame_pipeline,
    obter_labels_imagenet,
    obter_modelo_squeezenet,
    salvar_figura,
    exibir_janela_interativa,
    CALIB_FILE,
    SAIDAS_DIR,
)


def carregar_calibracao():
    # Carrega parâmetros intrínsecos de calibração ou usa valores padrão se ausente.
    if CALIB_FILE.exists():
        dados = np.load(CALIB_FILE)
        return dados["K"], dados["dist"]
    # Fallback calibrado
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.array([[-0.15, 0.05, 0.0, 0.0, 0.0]], dtype=np.float32)
    return K, dist


def etapa1_undistort(frame, K, dist):
    # Etapa 1: Correção de distorção óptica da lente.
    t0 = time.perf_counter()
    corrigido = cv2.undistort(frame, K, dist)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return corrigido, dt_ms


def etapa2_segmentacao_hsv(frame):
    # Etapa 2 (TP1): Segmentação cromática no espaço HSV de região de interesse (pele/vestimenta).
    t0 = time.perf_counter()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Faixa de pele/tons quentes e tons avermelhados no espaço HSV
    m1 = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    mascara = cv2.bitwise_or(m1, m2)

    # Morfologia matemática para limpeza de ruídos (abertura e fechamento)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_roi = None
    if cnts:
        c_max = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c_max) > 500:
            x, y, w, h = cv2.boundingRect(c_max)
            bbox_roi = (x, y, w, h)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return mascara, bbox_roi, dt_ms


def etapa3_features_orb(roi_bgr, max_features=150):
    # Etapa 3 (TP2): Extração de keypoints e descritores invariantes ORB.
    t0 = time.perf_counter()
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, descs = orb.detectAndCompute(gray, None)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return keypoints, descs, dt_ms


def etapa4_deteccao_pessoas_e_faces(frame):
    # Etapa 4 (TP3): Detecção clássica de pessoas/faces (Haar Cascade e HOG+SVM).
    t0 = time.perf_counter()
    boxes = []

    # 1. Detector de faces Haar Cascade (TP3 - rápido e ideal para retratos)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    for (fx, fy, fw, fh) in faces:
        boxes.append(("Haar: Face (TP3)", [int(fx), int(fy), int(fw), int(fh)]))

    # 2. Detector de corpo inteiro HOG + Linear SVM (TP3 - pedestres)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if len(rects) > 0:
        rects_list = [[x, y, w, h] for (x, y, w, h) in rects]
        weights_list = [float(w[0]) if isinstance(w, (list, np.ndarray)) else float(w) for w in weights]
        idxs = cv2.dnn.NMSBoxes(rects_list, weights_list, score_threshold=0.10, nms_threshold=0.40)
        if len(idxs) > 0:
            for idx in np.array(idxs).flatten():
                boxes.append(("HOG: Pedestre (TP3)", rects_list[int(idx)]))

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return boxes, dt_ms


def etapa5_classificacao_dnn(roi_bgr, net, labels):
    # Etapa 5 (Ex 2A): Classificação profunda da ROI com SqueezeNet v1.1 via OpenCV DNN.
    t0 = time.perf_counter()
    blob = cv2.dnn.blobFromImage(
        roi_bgr,
        scalefactor=1.0,
        size=(227, 227),
        mean=(104.0, 117.0, 123.0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    out = net.forward()
    logits = out.flatten()
    e_x = np.exp(logits - np.max(logits))
    probs = e_x / e_x.sum(axis=0)

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(labels[i] if i < len(labels) else f"classe_{i}", float(probs[i])) for i in top3_idx]
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return top3, dt_ms


def main():
    ensure_dirs()
    print("=" * 80)
    print("EXERCÍCIO 2B — PIPELINE INTEGRATIVO COMPLETO (TP1 + TP2 + TP3 + AT)")
    print("=" * 80)

    # Carrega calibração e modelos
    K, dist = carregar_calibracao()
    net, _, _ = obter_modelo_squeezenet()
    labels = obter_labels_imagenet()

    # Obtém frame para teste do pipeline integrado
    frame_raw, caminho_frame = obter_frame_pipeline()
    print(f"Frame de entrada carregado: {caminho_frame.name} ({frame_raw.shape[1]}x{frame_raw.shape[0]})")

    tempos = {}

    # 1. Undistort (Ex 1)
    frame_corrigido, t1 = etapa1_undistort(frame_raw, K, dist)
    tempos["1. Undistort (Ex 1)"] = t1

    # 2. Segmentação HSV (TP1)
    mask_hsv, roi_bbox, t2 = etapa2_segmentacao_hsv(frame_corrigido)
    tempos["2. Segmentação HSV (TP1)"] = t2

    # Recorte da ROI
    if roi_bbox is not None:
        rx, ry, rw, rh = roi_bbox
        roi_img = frame_corrigido[ry:ry + rh, rx:rx + rw]
    else:
        rx, ry, rw, rh = 80, 110, 150, 140
        roi_img = frame_corrigido[ry:ry + rh, rx:rx + rw]

    # 3. Features ORB (TP2)
    kps, descs, t3 = etapa3_features_orb(roi_img, max_features=120)
    tempos["3. Features ORB (TP2)"] = t3

    # 4. Detecção HOG+SVM ou Haar Cascade (TP3)
    det_boxes, t4 = etapa4_deteccao_pessoas_e_faces(frame_corrigido)
    tempos["4. Detector Pessoas/Faces (TP3)"] = t4

    # 5. Classificação OpenCV DNN (Ex 2A)
    top3, t5 = etapa5_classificacao_dnn(roi_img, net, labels)
    tempos["5. Classificação DNN (Ex 2A)"] = t5

    # ==============================================================================
    # RENDERIZAÇÃO DO FRAME FINAL COM TODAS AS ANOTAÇÕES INTEGRADAS
    # ==============================================================================
    vis = frame_corrigido.copy()

    # Desenho da ROI HSV (Verde Limão)
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 120), 2)
    cv2.putText(vis, "ROI HSV (TP1)", (rx, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

    # Desenho dos Keypoints ORB (Pontos e círculos verdes)
    for kp in kps:
        px = int(kp.pt[0] + rx)
        py = int(kp.pt[1] + ry)
        cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)

    # Desenho das Detecções Haar / HOG (Azul ciano / Laranja)
    for rotulo, (bx, by, bw, bh) in det_boxes:
        cor = (255, 140, 0) if "Haar" in rotulo else (255, 60, 0)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), cor, 2)
        cv2.putText(vis, rotulo, (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2)

    # Rótulo conciso da classificação profunda associada à ROI
    top1_str = f"DNN: {top3[0][0][:16]} ({top3[0][1]*100:.1f}%)"
    cv2.putText(vis, top1_str, (rx, ry + rh + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 2)

    # Salva o resultado final com a cena limpa (apenas bounding boxes, ROI e keypoints)
    saida_img = SAIDAS_DIR / "at2b_pipeline_integrado.png"
    cv2.imwrite(str(saida_img), vis)

    tempo_total = sum(tempos.values())

    # Exibição rica de todas as informações técnicas diretamente na CLI
    print("\n" + "-" * 70)
    print("CLASSIFICAÇÃO SQUEEZENET V1.1 NA ROI (OPENCV DNN):")
    print("-" * 70)
    for rank, (lbl, conf) in enumerate(top3, start=1):
        print(f"  #{rank}: {lbl:<32} ({conf*100:5.1f}%)")

    print("\n" + "-" * 70)
    print("MÉTRICAS DE EXECUÇÃO DO PIPELINE INTEGRADO:")
    print("-" * 70)
    for nome, ms in tempos.items():
        pct = (ms / tempo_total) * 100.0
        print(f"  {nome:<32} : {ms:7.2f} ms  ({pct:5.1f}%)")
    print("-" * 70)
    print(f"  {'TEMPO TOTAL DE PROCESSAMENTO':<32} : {tempo_total:7.2f} ms")
    print(f"  {'TAXA DE ATUALIZAÇÃO ESTIMADA':<32} : {1000/tempo_total:7.1f} FPS")
    print(f"Frame com anotações integradas salvo em: {saida_img.name}")

    # 6. Exibe o resultado final integrado na janela do OpenCV
    exibir_janela_interativa(
        "Exercicio 2B - Pipeline Integrativo Completo (Undistort + HSV + ORB + HOG/Haar + DNN)",
        vis,
        "Pressione 'q', ESC ou feche no [X] para finalizar"
    )


if __name__ == "__main__":
    main()
