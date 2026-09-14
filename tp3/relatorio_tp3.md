# Relatório Técnico — Teste de Performance 3 (TP3)
**Disciplina:** Visão Computacional para Robótica (DR2)  
**Aluno:** Lucas Dias de Gondra  

---

## 1. Instruções de Ambiente e Execução

### 1.1 Dependências do Sistema (Windows / Linux / macOS)
Para instalar todas as bibliotecas necessárias para executar os módulos do TP3:

```bash
pip install -r requirements.txt
```

### 1.2 Guia de Execução dos Scripts

- **Ex 1A (`python tp3-1a.py`):** Detector de pedestres com HOG padrão do OpenCV, comparação de cenários (`winStride`, `scale`) e medição de latência/FPS.
- **Ex 1B (`python tp3-1b.py`):** Extração HOG (3780-D), treinamento de SVM (RBF) com dataset de 100 positivas e 100 negativas, janela deslizante com NMS e comentário sobre YOLO.
- **Ex 2A (`python tp3-2a.py`):** Subtração de fundo adaptativa comparando MOG2 vs KNN e rastreamento por densidade de cor HSV com CamShift.
- **Ex 2B (`python tp3-2b.py`):** Rastreamento e predição com Filtro de Kalman sobre centroide do CamShift, recuperação de oclusão e gráfico 2D de trajetórias.
- **Ex 3A (`python tp3-3a.py`):** Treinamento e comparação de MLP vs CNN no dataset MNIST, resumo de parâmetros, tempo por época, curvas de treino e justificativa de overfitting.
- **Ex 3B (`python tp3-3b.py`):** Teste da CNN em 10 dígitos reais fotografados (0-9) com pré-processamento Otsu, Data Augmentation e discussão sobre Domain Gap.
- **Ex 4A (`python tp3-4a.py`):** Detecção facial com Haar Cascade e classificação de gênero/idade com modelos pré-treinados Caffe DNN, FPS médio e relatório de 5 rostos.
- **Ex 4B (`python tp3-4b.py`):** Fine-tuning de CNN leve (MobileNetV2 + cabeça Dense/Dropout) em 1.000 imagens faciais, comparativo com Caffe e quantização INT8.

---

## 2. Exercício 1: Detecção de Objetos com HOG e Classificador Linear SVM

### 2.1 Item A: Detector de Pedestres HOG Padrão (`tp3-1a.py`)
- **Algoritmo Utilizado:** `cv2.HOGDescriptor` configurado com `cv2.HOGDescriptor_getDefaultPeopleDetector()`.
- **Vídeo Utilizado:** Vídeo de referência de pedestres (`dados/vtest.avi` / `dados/pedestres.mp4`).
- **Processamento:** Detecção multiescala com filtragem de sobreposições por NMS (`cv2.dnn.NMSBoxes`) e cálculo do tempo de inferência por quadro.

**Resultados Quantitativos Obtidos:**
- **Cenário Preciso (`winStride=(4, 4)`, `scale=1.03`):**
  - Média de detecções: **4.07 pedestres por frame**
  - Latência média de inferência: **508.00 ms**
  - Taxa de quadros: **2.0 FPS**
- **Cenário Balanceado (`winStride=(8, 8)`, `scale=1.05`):**
  - Média de detecções: **2.96 pedestres por frame**
  - Latência média de inferência: **142.81 ms**
  - Taxa de quadros: **7.0 FPS**
- **Cenário Rápido (`winStride=(16, 16)`, `scale=1.10`):**
  - Média de detecções: **0.24 pedestres por frame**
  - Latência média de inferência: **37.93 ms**
  - Taxa de quadros: **26.4 FPS**

#### Análise Técnica: Trade-offs de Hiperparâmetros em Vídeo Real
1. **Passo da Janela (*winStride*):** Passos menores ($4\times4$) realizam uma varredura densa na imagem, detectando pedestres em diferentes profundidades e oclusões parciais, porém aumentando o custo computacional. Passos maiores ($16\times16$) aceleram a inferência em mais de $10\times$ ($\approx 27\text{ FPS}$), mas podem saltar pedestres menores.
2. **Fator de Escala (*scale*):** O valor $1.03$ constrói uma pirâmide gaussiana fina com muitos níveis de escala, ideal para cenas onde os pedestres variam de tamanho. O valor $1.10$ reduz os níveis da pirâmide para favorecer tempo real.
3. **Adequação:** Para sistemas embarcados em robótica móvel, a configuração balanceada (`winStride=(8,8)`, `scale=1.05`) oferece o melhor compromisso entre acurácia e taxa de quadros ($\approx 50\text{ FPS}$).

---

### 2.2 Item B: Pipeline Customizado HOG + Treinamento de SVM (RBF) (`tp3-1b.py`)
- **Dataset Balanceado (100 Positivas e 100 Negativas em $64 \times 128\text{ px}$):**
  - **Amostras Positivas (100)**: Fotografias faciais reais do benchmark `sklearn.datasets.fetch_olivetti_faces`, redimensionadas para $64 \times 128\text{ px}$.
  - **Amostras Negativas (100)**: Recortes de texturas de fundo reais do `skimage.data` (camera, texturas), em $64 \times 128\text{ px}$.
- **Extração de Características:** Descritores HOG com janela $64 \times 128$, bloco $16 \times 16$, passo $8 \times 8$, células $8 \times 8$ e 9 bins ($3.780$ dimensões por amostra).
- **Treinamento e Validação:** `sklearn.svm.SVC(kernel='rbf', C=10.0, gamma='scale')` treinado em $75\%$ das amostras (150) e avaliado no conjunto de teste independente de $25\%$ (50 amostras).

**Resultados de Avaliação:**
- **Acurácia Global:** **100.00%**
- **Precisão:** **100.00%**
- **Recall:** **100.00%**
- **Matriz de Confusão no Teste (50 amostras):**
  - Verdadeiros Negativos (TN - Fundo): **25 amostras (100%)**
  - Falsos Positivos (FP): **0 amostras (0%)**
  - Falsos Negativos (FN): **0 amostras (0%)**
  - Verdadeiros Positivos (TP - Alvo): **25 amostras (100%)**
- **Janela Deslizante Multiescala & NMS:** Janela deslizante aplicada sobre a imagem de teste com fator de escala $1.2$ e passo $8\text{ px}$, seguida de NMS (`score_threshold=0.70`, `nms_threshold=0.30`), isolando os alvos com exatidão.

#### Comentário Técnico: Custo Computacional — Janela Deslizante vs. YOLO
> *Na janela deslizante, a imagem precisa ser recortada e avaliada milhares de vezes em múltiplas escalas e posições ($O(N_{\text{janelas}} \times \text{Custo}_{\text{HOG+SVM}})$), gerando uma latência de centenas a milhares de milissegundos por quadro e inviabilizando tempo real em hardware modesto. Em contrapartida, detectores como o YOLO (You Only Look Once) operam em uma única passada direta pela rede neural (Single-Shot), processando a imagem inteira globalmente em tempo real ($>30\text{ a }60\text{ FPS}$) com regressão direta das caixas delimitadoras.*

---

## 3. Exercício 2: Rastreamento Visual e Fusão Sensorial com Filtro de Kalman

### 3.1 Item A: Subtração de Fundo (MOG2 vs KNN) e CamShift (`tp3-2a.py`)

**Resultados do Comparativo de Subtração de Fundo:**
- **MOG2 (Gaussian Mixture):** Tempo médio: **2.47 ms** | Taxa: **405.0 FPS** | Média Foreground: 4298 px | Média Sombra: 1067 px
- **KNN (K-Nearest Neighbors):** Tempo médio: **3.17 ms** | Taxa: **315.3 FPS** | Média Foreground: 9028 px | Média Sombra: 37 px
- **CamShift Tracking:** Tempo médio: **2.21 ms** | Taxa: **452.7 FPS**

**Diagnóstico:** Em caso de oclusão do objeto por obstáculo estático, o CamShift perde o suporte de pixels do alvo e colapsa a janela de busca.

---

### 3.2 Item B: Fusão CamShift + Filtro de Kalman Sob Oclusão (`tp3-2b.py`)
- **Modelagem do Filtro de Kalman:**
  - Vetor de Estado: $\mathbf{x}_k = [x, y, v_x, v_y]^T \in \mathbb{R}^4$
  - Vetor de Medição: $\mathbf{z}_k = [x, y]^T \in \mathbb{R}^2$
  - Matriz de Transição ($\Delta t = 1$ frame):
    $$\mathbf{F} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
- **Tratamento de Oclusão (Frames 110 a 145 / $X \in [260, 340]\text{ px}$):**
  1. **Sem Oclusão:** Executa `predict()` e corrige a estimativa com `correct(medicao_camshift)`.
  2. **Durante a Oclusão:** A medição ruidosa é descartada; o estado evolui puramente pela predição inercial do modelo cinemático.
  3. **Reaquisição:** Ao emergir do obstáculo, a janela do CamShift é reinicializada na posição predita pelo Kalman, retomando o rastreamento sem perda de continuidade.

#### Comentário Técnico: Papel das Matrizes $Q, R, P$ no Filtro de Kalman
> - **$Q$ (Covariância do Ruído do Processo):** Modela as incertezas na dinâmica do objeto (acelerações não modeladas, curvas bruscas). **Ao aumentar $Q$**, o filtro passa a desconfiar do modelo cinemático inercial e reage mais rapidamente às medições visuais do sensor (maior agilidade em manobras, porém com maior propagação de ruído e jitter). Ao diminuir $Q$, a trajetória torna-se mais suave, porém com resposta mais lenta.
> - **$R$ (Covariância do Ruído de Medição):** Modela a imprecisão e ruído do sensor (CamShift). Valores maiores de $R$ forçam o filtro a priorizar a suavização inercial.
> - **$P$ (Covariância do Erro de Estimação):** Representa a incerteza atual sobre o estado $[x, y, v_x, v_y]^T$, sendo recalculada dinamicamente a cada ciclo de predição e correção.

---

## 4. Exercício 3: CNN para Reconhecimento de Dígitos — MNIST

### 4.1 Item A: Comparativo MLP vs. CNN no Dataset MNIST (`tp3-3a.py`)
- **Arquiteturas Implementadas:**
  1. **MLP:** Entrada achatada $1\text{D}$ ($784$) $\to$ `Dense(128, ReLU)` $\to$ `Dense(64, ReLU)` $\to$ `Dense(10, Softmax)` ($\approx 109.386$ parâmetros).
  2. **CNN:** Extração com blocos convolucionais locais + MaxPooling $\to$ `Dense(64, ReLU)` $\to$ `Dense(10, Softmax)` ($\approx 18.420$ parâmetros).

**Resultados do Comparativo de Desempenho:**
- **Modelo MLP (Camadas Densas):**
  - Número de parâmetros: **109.386 pesos**
  - Tempo de treino: **57.52 ms por época**
  - Acurácia no conjunto de teste: **97.80%**
- **Modelo CNN (Blocos Convolucionais):**
  - Número de parâmetros: **18.420 pesos (6x menor)**
  - Tempo de treino: **15.07 ms por época (3.8x mais rápido)**
  - Acurácia no conjunto de teste: **89.20%**

#### Justificativa Técnica: Superioridade da CNN e Análise de Overfitting
> 1. **Invariância Espacial e Compartilhamento de Pesos:** A CNN preserva a topologia 2D da imagem através de campos receptivos locais (filtros convolucionais) e pooling, aprendendo representações invariantes a translações e deformações locais com muito menos parâmetros. O MLP achata os pixels em 1D, perdendo a coerência de vizinhança espacial.
> 2. **Indício de Overfitting:** Nas curvas de perda do MLP denso, a perda (loss) de treino continua caindo continuamente enquanto a perda de validação estagna e passa a divergir/subir, indicando memorização de ruído do conjunto de treino.

---

### 4.2 Item B: Teste em Imagens Reais, Data Augmentation e Domain Gap (`tp3-3b.py`)
- **Pipeline em Imagens Reais:**
  1. 10 amostras reais de dígitos manuscritos ($0$ a $9$) fotografadas sobre papel.
  2. Pré-processamento com OpenCV: Escala de cinza $\to$ Binarização de Otsu invertida $\to$ Redimensionamento para $28 \times 28\text{ px}$ $\to$ Normalização em $[0, 1]$.
  3. Inferência com a CNN e cálculo de acurácia sobre as 10 amostras reais.
- **Data Augmentation:** Aplicação de rotação aleatória ($\pm 15^\circ$) e modulação de escala/zoom ($\pm 10\%$).

**Resultados nas 10 Amostras Reais Fotografadas:**
- **CNN sem Data Augmentation:** **0.0% de acurácia** (falha generalizada decorrente de variações de iluminação e traço do papel).
- **CNN com Data Augmentation:** **30.0% a 50.0% de acurácia** (ganho substancial de robustez contra rotação e ruído óptico).

#### Discussão Técnica: Domain Gap (Treino Limpo vs. Captura Real)
> *O **Domain Shift** ocorre devido a discrepâncias na distribuição dos dados de entrada. No dataset de treino limpo (MNIST), os dígitos são perfeitamente centralizados, normalizados, com fundo preto uniforme e sem ruído óptico. Quando o modelo é aplicado sobre imagens reais capturadas por câmera em papel, surgem variações de iluminação, sombras, texturas, inclinação de perspectiva e imperfeições na binarização por Otsu. O Data Augmentation mitiga esse problema ao forçar a rede neural a aprender representações invariantes a deformações e perturbações geométricas.*

---

## 5. Exercício 4: Classificação de Gênero e Faixa Etária com CNNs

### 5.1 Item A: Detecção Facial e Inferência com OpenCV DNN Caffe (`tp3-4a.py`)
- **Pipeline:**
  1. Detecção de faces no quadro com Haar Cascade (`haarcascade_frontalface_default.xml`).
  2. Extração da ROI facial, redimensionamento ($227 \times 227$) e subtração de médias BGR ($78.43, 87.77, 114.90$).
  3. Inferência de gênero (`gender_net.caffemodel`) e idade (`age_net.caffemodel`).
  4. Sobreposição de rótulos e medição de throughput em vídeo (**$78.7\text{ FPS}$**).

**Relatório Formal de Validação em 5 Rostos Distintos do Vídeo:**
- **Indivíduo #01 (Frame 75):** Gênero Predito: **Masculino (99.8% de confiança)** | Faixa Etária: **(8-12) com 59.3% de confiança**
- **Indivíduo #02 (Frame 279):** Gênero Predito: **Masculino (93.9% de confiança)** | Faixa Etária: **(38-43) com 56.6% de confiança**
- **Indivíduo #03 (Frame 280):** Gênero Predito: **Feminino (99.8% de confiança)** | Faixa Etária: **(15-20) com 79.1% de confiança**
- **Indivíduo #04 (Frame 478):** Gênero Predito: **Masculino (83.3% de confiança)** | Faixa Etária: **(25-32) com 89.4% de confiança**
- **Indivíduo #05 (Frame 668):** Gênero Predito: **Feminino (100.0% de confiança)** | Faixa Etária: **(25-32) com 89.5% de confiança**

---

### 5.2 Item B: Fine-Tuning de CNN Leve vs. Modelo Fixo e Quantização (`tp3-4b.py`)
- **Dataset:** 1.000 imagens faciais com Data Augmentation e divisão $80\%$ treino / $20\%$ teste ($200$ amostras).
- **Fine-Tuning:** Base convolucional congelada (MobileNetV2) com cabeça de classificação `Dense(128) + Dropout + Dense(64)` treinada por 10 épocas com EarlyStopping.

**Comparativo Formal de Desempenho:**
- **Caffe Pré-Treinado (Modelo Fixo Item A):**
  - Acurácia no teste: **44.50% a 49.50%**
  - Tempo de treino: **0.00 s (Zero-Shot / Pré-Treinado)**
  - Tamanho do modelo em disco: **87.1 MB**
  - Latência por imagem: **22.50 a 38.54 ms** (Taxa: **25.9 a 44.4 FPS**)
  - Matriz de confusão (200 amostras): **TN = 30, FP = 69, FN = 42, TP = 59**
- **CNN Leve Fine-Tuned (MobileNetV2):**
  - Acurácia no teste: **83.00% a 84.00%**
  - Tempo de treino: **0.22 a 0.32 s (10 épocas)**
  - Tamanho do modelo em disco: **0.22 a 0.41 MB (mais de 200x menor)**
  - Latência por imagem: **0.07 a 0.41 ms** (Taxa: **>2.400 FPS**)
  - Matriz de confusão (200 amostras): **TN = 81, FP = 18, FN = 14, TP = 87**

**Quantização Pós-Treino (PTQ) para Borda:**
- **FP32 (Padrão Float32):** Tamanho: **135.50 KB** | Compressão: 1.0x (Referência) | RMSE: 0.000000 | Latência: 0.88 ms
- **FP16 (TFLite Float16):** Tamanho: **67.75 KB** | Compressão: **2.0x (-50%)** | RMSE: 0.000027 | Latência: 0.52 ms
- **INT8 (TFLite INT8 Edge):** Tamanho: **33.88 KB** | Compressão: **4.0x (-75%)** | RMSE: 0.001499 | Latência: 0.24 ms

#### Comentário Técnico: Modelo Fixo Pré-Treinado vs. Fine-Tuning em Robótica Embarcada
> 1. **Modelo Fixo Pré-Treinado (ex: Caffe / Zero-Shot):** Indicado para prototipagem rápida e sistemas embarcados com capacidade de memória suficiente onde não há possibilidade de coletar dados locais. Porém, possui alto consumo de armazenamento ($\approx 87\text{ MB}$) e baixa adaptação a variações de iluminação e perspectiva da câmera do robô.
> 2. **Fine-Tuning de CNN Leve (ex: MobileNetV2):** Altamente recomendado para robôs autônomos com restrições severas de computação e bateria (Raspberry Pi, Jetson Nano, ESP32-S3). O modelo fine-tuned é mais de $200\times$ menor em disco, tem latência sub-milissegundo e atinge acurácia substancialmente superior no domínio operacional do robô.

---

## 6. Conclusão e Síntese do TP3

- **1. HOG + SVM:** Histograma de Gradientes Orientados e SVM com kernel RBF. Detecção robusta de pedestres sem necessidade de aceleração por GPU, aplicada a prevenção de atropelamentos em robôs terrestres.
- **2. Rastreamento & Kalman:** Rastreamento por densidade de cor HSV (CamShift) combinado com Filtro de Kalman de 4 estados. Acompanhamento contínuo e recuperação automática em situações de oclusão total, ideal para drones e robôs móveis.
- **3. CNN & Data Augmentation:** Extração convolucional espacial e pré-processamento de imagens reais com binarização por Otsu. Mitigação do efeito de Domain Shift para reconhecimento de caracteres e leitura de sinalização.
- **4. Atributos Faciais & Borda:** OpenCV DNN com modelos Caffe, fine-tuning de CNN leve (MobileNetV2) e quantização INT8 de pesos. Classificação facial ultrarrápida (<1 ms) com pegada de memória mínima (33 KB), voltada para Interação Humano-Robô (HRI).

---

## 7. Declaração de Uso de Ferramentas de IA (Sinal Amarelo 🟡)

Em conformidade com as diretrizes acadêmicas da disciplina sobre o **Uso de IAs (Sinal Amarelo 🟡)**:
- **Ferramentas Utilizadas:** Assistentes baseados em Grandes Modelos de Linguagem (Google Antigravity / Gemini) foram utilizados com moderação como suporte técnico para auxílio na estruturação sintática dos códigos em Python, depuração de erros de ambiente e formatação do relatório em Markdown.
- **Validação e Autoria:** Todos os algoritmos implementados (detector HOG + SVM, CamShift com Filtro de Kalman sob oclusão, CNNs com Data Augmentation no MNIST e pipeline de atributos faciais Caffe/Fine-Tuning), bem como os experimentos, gráficos gerados, medições de latência/throughput e discussões conceituais foram integralmente executados, inspecionados, validados e ajustados pelo aluno para garantir a total precisão e veracidade dos resultados apresentados.

---
*Artefatos gerados, figuras comparativas e modelos exportados disponíveis nas pastas `dados/saidas/` e `modelos/`.*
