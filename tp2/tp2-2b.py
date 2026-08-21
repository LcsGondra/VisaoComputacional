import cv2
import numpy as np
import time
from pathlib import Path


def gerar_rosto_identidade(nome, cor_pele, cor_olhos, cor_cabelo):
    img = np.full((200, 200, 3), (240, 240, 240), dtype=np.uint8)
    cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, cor_pele, -1)
    cv2.ellipse(img, (100, 50), (65, 35), 0, 180, 360, cor_cabelo, -1)
    cv2.circle(img, (75, 90), 8, (255, 255, 255), -1)
    cv2.circle(img, (125, 90), 8, (255, 255, 255), -1)
    cv2.circle(img, (75, 90), 4, cor_olhos, -1)
    cv2.circle(img, (125, 90), 4, cor_olhos, -1)
    cv2.line(img, (100, 95), (100, 115), (100, 80, 70), 2)
    cv2.ellipse(img, (100, 140), (22, 10), 0, 0, 180, (50, 50, 180), -1)
    cv2.putText(img, nome, (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    return img


def extrair_embedding_lbph_ou_template(imagem_face, tamanho=(96, 96)):
    cinza = cv2.cvtColor(imagem_face, cv2.COLOR_BGR2GRAY) if imagem_face.ndim == 3 else imagem_face
    redim = cv2.resize(cinza, tamanho, interpolation=cv2.INTER_AREA)
    norm = cv2.equalizeHist(redim).astype(np.float32) / 255.0
    return norm.flatten()


def main():
    saida_dir = Path("dados/saidas")
    saida_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = Path("dados/referencias")
    ref_dir.mkdir(parents=True, exist_ok=True)

    identidades = [
        {"nome": "Alice (Eng. Robotica)", "pele": (195, 175, 160), "olhos": (40, 150, 40), "cabelo": (40, 40, 160)},
        {"nome": "Bruno (Piloto Drone)", "pele": (170, 140, 120), "olhos": (140, 60, 20), "cabelo": (20, 20, 20)},
        {"nome": "Carla (Pesquisadora)", "pele": (210, 195, 180), "olhos": (30, 30, 30), "cabelo": (30, 90, 140)},
    ]

    banco_vetores = {}
    for id_info in identidades:
        nome = id_info["nome"]
        img_ref = gerar_rosto_identidade(nome.split()[0], id_info["pele"], id_info["olhos"], id_info["cabelo"])
        caminho_ref = ref_dir / f"{nome.split()[0].lower()}.png"
        cv2.imwrite(str(caminho_ref), img_ref)
        banco_vetores[nome] = extrair_embedding_lbph_ou_template(img_ref)

    print("=== SISTEMA DE RECONHECIMENTO FACIAL (TP2 - 2B) ===")
    print(f"Total de identidades cadastradas no banco: {len(banco_vetores)}")
    for nome in banco_vetores.keys():
        print(f"  - Registrado: {nome}")

    limiar_distancia = 0.28
    latencias = []

    quadros_teste = []
    for id_info in identidades:
        img_face = gerar_rosto_identidade(id_info["nome"].split()[0], id_info["pele"], id_info["olhos"], id_info["cabelo"])
        quadros_teste.append((id_info["nome"].split()[0], img_face))

    img_desconhecido = gerar_rosto_identidade("Estranho", (140, 110, 90), (200, 200, 200), (200, 200, 200))
    quadros_teste.append(("Desconhecido", img_desconhecido))

    janela_nome = "Reconhecimento Facial - TP2 Exercicio 2B"
    cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)

    ultimo_resultado = None

    for i in range(40):
        nome_real, face_entrada = quadros_teste[i % len(quadros_teste)]
        cena = np.full((360, 480, 3), (230, 230, 230), dtype=np.uint8)

        offset_x = 140 + int(20 * np.sin(i * 0.4))
        offset_y = 80 + int(15 * np.cos(i * 0.4))
        fh, fw = face_entrada.shape[:2]
        cena[offset_y : offset_y + fh, offset_x : offset_x + fw] = face_entrada

        t_inicio = time.perf_counter()

        roi = cena[offset_y : offset_y + fh, offset_x : offset_x + fw]
        vetor_consulta = extrair_embedding_lbph_ou_template(roi)

        menor_dist = float("inf")
        melhor_id = "desconhecido"

        for nome_cadastrado, vetor_cadastrado in banco_vetores.items():
            dist = float(np.mean(np.abs(vetor_consulta - vetor_cadastrado)))
            if dist < menor_dist:
                menor_dist = dist
                if dist < limiar_distancia:
                    melhor_id = nome_cadastrado

        t_fim = time.perf_counter()
        latencia_ms = (t_fim - t_inicio) * 1000.0
        latencias.append(latencia_ms)

        cor_box = (0, 220, 0) if melhor_id != "desconhecido" else (0, 0, 220)
        cv2.rectangle(cena, (offset_x, offset_y), (offset_x + fw, offset_y + fh), cor_box, 2)
        cv2.putText(
            cena,
            f"{melhor_id} (d={menor_dist:.3f})",
            (offset_x - 10, max(25, offset_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cor_box,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            cena,
            f"Latencia: {latencia_ms:.2f} ms | Frame {i}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(janela_nome, cena)
        ultimo_resultado = cena
        cv2.waitKey(30)

    if ultimo_resultado is not None:
        cv2.imwrite(str(saida_dir / "tp2_2b_reconhecimento_saida.png"), ultimo_resultado)

    media_latencia = float(np.mean(latencias))
    print(f"\nTempo medio de inferencia por frame: {media_latencia:.2f} ms")
    print(f"Taxa estimada de processamento: {1000.0 / max(media_latencia, 1e-4):.1f} FPS")

    print("\n=== DISCUSSAO ETICA: SISTEMAS FACIAIS EM ROBOTICA E DRONES ===")
    print("1. Privacidade e Vigilancia Massiva Indiscriminada:")
    print("   Drones equipados com reconhecimento facial podem rastrear individuos em espacos publicos")
    print("   sem consentimento previo, ferindo leis de protecao de dados (ex: LGPD e GDPR).")
    print("2. Vies Algoritmico e Taxa de Erro Desproporcional:")
    print("   Muitos modelos apresentam menor precisao para minorias etnicas e de genero,")
    print("   podendo gerar acusacoes e intervencoes policiais erroneas em sistemas de seguranca.")
    print("3. Autonomia Letal e Seguranca Operacional:")
    print("   A tomada de decisao autonoma baseada em reconhecimento facial em sistemas belicos")
    print("   ou de restricao fisica impoe graves riscos a direitos humanos e integridade fisica.")

    cv2.waitKey(1000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
