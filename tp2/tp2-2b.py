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
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms

try:
    from deepface import DeepFace  # type: ignore
except Exception:
    DeepFace = None

from utils import imshow_keep_aspect_ratio, obter_diretorios, selecionar_dispositivo


class ExtratorFacialGPU:
    def __init__(self):
        self.device = selecionar_dispositivo()
        self.nome_hardware = (
            torch.cuda.get_device_name(0)
            if self.device.type == "cuda"
            else ("DirectML / MPS" if self.device.type != "cpu" else "CPU")
        )

        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.model.classifier[3] = torch.nn.Identity()
        self.model = self.model.to(self.device).eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dummy = torch.zeros((1, 3, 224, 224), device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)

    def extrair(self, imagem_bgr):
        if imagem_bgr is None or imagem_bgr.size == 0:
            return np.zeros(1024, dtype=np.float32)
        try:
            rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(tensor).cpu().numpy()[0]
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                return feat / norm
            return feat
        except Exception:
            cinza = (
                cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
                if imagem_bgr.ndim == 3
                else imagem_bgr
            )
            redim = cv2.resize(cinza, (96, 96), interpolation=cv2.INTER_AREA)
            norm = cv2.equalizeHist(redim)
            hist, _ = np.histogram(norm, bins=128, range=(0, 256), density=True)
            vetor = hist.astype(np.float32)
            return vetor / (np.linalg.norm(vetor) + 1e-7)


def main():
    dirs = obter_diretorios(__file__)
    base_dir = dirs["base"]
    saida_dir = dirs["saidas"]
    ref_dir = dirs["referencias"]

    extrator = ExtratorFacialGPU()

    caminho_imagens_projeto = base_dir.parent / "Imagens"

    referencias_estaticas = [
        {
            "nome": "Carlos (Stock Man)",
            "caminhos": [caminho_imagens_projeto / "stock_img_man.jpg"],
        },
        {
            "nome": "Mariana (Stock Woman)",
            "caminhos": [caminho_imagens_projeto / "stock_img_woman.jpg"],
        },
        {
            "nome": "Alex (Stock Aluno)",
            "caminhos": [caminho_imagens_projeto / "stock_img_aluno.jpg"],
        },
    ]

    banco_identidades = {}

    for ref in referencias_estaticas:
        vetores = []
        caminhos_validos = []
        for p in ref["caminhos"]:
            if p.exists():
                img_estatica = cv2.imread(str(p))
                if img_estatica is not None:
                    vetores.append(extrator.extrair(img_estatica))
                    caminhos_validos.append(p)
        if vetores:
            banco_identidades[ref["nome"]] = {
                "caminhos": caminhos_validos,
                "vetores": vetores,
            }

    pasta_lucas = ref_dir / "lucas"
    pasta_lucas.mkdir(parents=True, exist_ok=True)

    fotos_lucas = list(pasta_lucas.glob("*.png")) + list(pasta_lucas.glob("*.jpg"))
    if len(fotos_lucas) == 0:
        candidatos_foto = [
            dirs["faces"] / "face_cfg1_0000.png",
            dirs["faces"] / "face_manual_0052.png",
            dirs["dados"] / "referencias" / "usuario" / "face_001.png",
        ]
        for cf in candidatos_foto:
            if cf.exists():
                img_cf = cv2.imread(str(cf))
                if img_cf is not None:
                    dest = pasta_lucas / "face_001.png"
                    cv2.imwrite(str(dest), img_cf)
                    fotos_lucas.append(dest)
                    break

    vetores_lucas = []
    caminhos_lucas = []
    for f in sorted(fotos_lucas):
        im = cv2.imread(str(f))
        if im is not None:
            vetores_lucas.append(extrator.extrair(im))
            caminhos_lucas.append(f)

    if vetores_lucas:
        banco_identidades["Lucas"] = {
            "caminhos": caminhos_lucas,
            "vetores": vetores_lucas,
        }

    print("=== SISTEMA DE RECONHECIMENTO FACIAL ACELERADO POR GPU (TP2 - 2B) ===")
    print(f"Hardware Ativo: {extrator.nome_hardware} ({extrator.device})")
    print(
        f"Total de identidades carregadas a partir de fotos estaticas: {len(banco_identidades)}"
    )
    for nome, dados in banco_identidades.items():
        print(f"  - {nome}: {len(dados['vetores'])} face(s) registrada(s)")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nenhuma camera/webcam detectada para o Exercicio 2B.")

    print("\nWebcam conectada com sucesso para reconhecimento em tempo real.")

    limiar_similaridade = 0.52
    latencias_ms = []

    janela_nome = "Reconhecimento Facial em Tempo Real (GPU) - TP2 Ex2B"
    cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela_nome, 920, 580)

    print("\nControles:")
    print(
        "  - [C] ou [ESPACO]: Cadastrar nova amostra de face para 'Lucas' em 'modelos/'"
    )
    print("  - [Q]: Encerrar e exibir relatorio de latencia")

    frame_idx = 0
    ultima_face_capturada = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        h, w = frame.shape[:2]
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cinza = cv2.equalizeHist(cinza)

        faces = detector.detectMultiScale(
            cinza, scaleFactor=1.12, minNeighbors=5, minSize=(60, 60)
        )
        frame_saida = frame.copy()

        for x, y, fw, fh in faces:
            roi_face = frame[max(0, y) : min(h, y + fh), max(0, x) : min(w, x + fw)]
            if roi_face.size == 0:
                continue

            ultima_face_capturada = roi_face.copy()

            t_inicio = time.perf_counter()
            vetor_atual = extrator.extrair(roi_face)

            melhor_nome = "Desconhecido"
            maior_similaridade = -1.0

            for nome_cad, dados_cad in banco_identidades.items():
                if len(dados_cad["vetores"]) == 0:
                    continue
                sims = [float(np.dot(vetor_atual, v)) for v in dados_cad["vetores"]]
                max_sim = max(sims)
                if max_sim > maior_similaridade:
                    maior_similaridade = max_sim
                    if max_sim >= limiar_similaridade:
                        melhor_nome = nome_cad

            t_fim = time.perf_counter()
            latencia_frame = (t_fim - t_inicio) * 1000.0
            latencias_ms.append(latencia_frame)

            reconhecido = melhor_nome != "Desconhecido"
            cor_box = (0, 220, 0) if reconhecido else (0, 0, 230)

            cv2.rectangle(frame_saida, (x, y), (x + fw, y + fh), cor_box, 2)
            rotulo = (
                f"{melhor_nome} ({maior_similaridade*100:.1f}%)"
                if reconhecido
                else f"Desconhecido ({maior_similaridade*100:.1f}%)"
            )

            (rw, rh), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(
                frame_saida, (x, max(0, y - rh - 8)), (x + rw + 8, y), cor_box, -1
            )
            cv2.putText(
                frame_saida,
                rotulo,
                (x + 4, max(18, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(frame_saida, (0, 0), (w, 55), (15, 15, 15), -1)
        media_atual_lat = np.mean(latencias_ms[-30:]) if len(latencias_ms) > 0 else 0.0
        n_faces_lucas = (
            len(banco_identidades["Lucas"]["vetores"])
            if "Lucas" in banco_identidades
            else 0
        )
        cv2.putText(
            frame_saida,
            f"GPU: {extrator.nome_hardware} | Faces 'Lucas': {n_faces_lucas} | Latencia: {media_atual_lat:.2f} ms",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_saida,
            f"Identidades: {list(banco_identidades.keys())} | [C] Add Face | [Q] Sair",
            (12, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        if frame_idx % 25 == 0:
            print(
                f"Frame: {frame_idx:04d} | Faces: {len(faces)} | Latencia Inferencia ({extrator.nome_hardware}): {media_atual_lat:.2f} ms"
            )

        imshow_keep_aspect_ratio(janela_nome, frame_saida)

        if frame_idx == 30:
            cv2.imwrite(str(saida_dir / "tp2_2b_reconhecimento_gpu.png"), frame_saida)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("c") or key == ord("C") or key == 32:
            if ultima_face_capturada is not None and ultima_face_capturada.size > 0:
                if "Lucas" not in banco_identidades:
                    banco_identidades["Lucas"] = {"caminhos": [], "vetores": []}
                qtd_atual = len(banco_identidades["Lucas"]["vetores"]) + 1
                caminho_amostra = pasta_lucas / f"face_{qtd_atual:03d}.png"
                cv2.imwrite(str(caminho_amostra), ultima_face_capturada)
                novo_vetor = extrator.extrair(ultima_face_capturada)
                banco_identidades["Lucas"]["vetores"].append(novo_vetor)
                print(f"-> Nova face #{qtd_atual} salva em '{caminho_amostra}'")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    latencia_media_final = (
        float(np.mean(latencias_ms)) if len(latencias_ms) > 0 else 0.0
    )
    fps_estimado = 1000.0 / max(latencia_media_final, 1e-4)

    print("\n=== RESULTADOS FINAIS DE RECONHECIMENTO FACIAL (TP2 - 2B) ===")
    print(f"Hardware de Inferencia: {extrator.nome_hardware} ({extrator.device})")
    print(f"Total de identidades avaliadas: {len(banco_identidades)}")
    for nome, dados in banco_identidades.items():
        print(f"  - {nome}: {len(dados['vetores'])} face(s) de referencia registradas")
    print(f"Tempo medio de inferencia por frame: {latencia_media_final:.2f} ms")
    print(f"Taxa estimada de processamento: {fps_estimado:.1f} FPS")

    # Discussao etica: Cuidados em aplicacoes roboticas e drones (vigilancia)
    # 1. Privacidade e Vigilancia Massiva Indiscriminada:
    #    Drones equipados com reconhecimento facial podem rastrear individuos em espacos publicos
    #    sem consentimento previo, violando principios de privacidade e leis de protecao de dados (LGPD / GDPR).
    # 2. Vies Algoritmico e Disparidade de Erro:
    #    Muitos classificadores apresentam taxas de erro superiores para minorias etnicas e de genero,
    #    criando risco de identificacoes incorretas e abordagens injustas em sistemas de seguranca automatizados.
    # 3. Seguranca e Decisao Autonoma:
    #    O emprego de reconhecimento facial para acionamento de atuadores, trancas ou drones autonomos
    #    exige garantias estritas de auditabilidade e intervencao humana (human-in-the-loop) para evitar acidentes.


if __name__ == "__main__":
    main()
