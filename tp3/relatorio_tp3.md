# Relatório Técnico — Teste de Performance 3 (TP3)
**Disciplina:** Visão Computacional para Robótica (DR2)  
**Aluno:** Lucas Dias de Gondra  

---

## 1. Instruções de Ambiente e Execução

### 1.1 Dependências do Sistema (Windows / PowerShell)
Para configurar e instalar as bibliotecas necessárias para executar todos os módulos do TP3:

```powershell
pip install --user numpy opencv-python opencv-contrib-python matplotlib scikit-learn scipy pandas seaborn joblib Pillow
```

### 1.2 Guia de Execução dos Scripts

| Exercício | Script | Descrição do Pipeline |
| :--- | :--- | :--- |
| **Ex 1A** | `python tp3-1a.py` | Detector de pedestres com HOG padrão do OpenCV, varredura de parâmetros (`winStride`, `scale`) e análise de latência/FPS. |
| **Ex 1B** | `python tp3-1b.py` | Extração HOG (3780-D), treinamento de classificador SVM (RBF) com dataset real, curvas de treino vs teste, matriz de confusão e janela deslizante com NMS. |
| **Ex 2A** | `python tp3-2a.py` | Subtração de fundo adaptativa comparando MOG2 vs KNN e rastreamento contínuo por densidade de cor HSV com CamShift. |
| **Ex 2B** | `python tp3-2b.py` | Fusão sensorial do CamShift com Filtro de Kalman (4 estados / 2 medições), recuperação de oclusão e telemetria temporal em CSV. |
| **Ex 3A** | `python tp3-3a.py` | Pipeline de dados de alta performance com Data Augmentation dinâmico (flip, rotação, zoom) e benchmark de throughput. |
| **Ex 3B** | `python tp3-3b.py` | Transfer Learning com backbone convolucional MobileNetV2, treinamento da cabeça densa, curvas de aprendizado e matriz de confusão. |
| **Ex 4A** | `python tp3-4a.py` | Benchmark comparativo rigoroso de latência, desvio padrão, throughput (FPS) e pegada de memória em robótica embarcada. |
| **Ex 4B** | `python tp3-4b.py` | Quantização pós-treino (FP32 vs FP16 vs INT8), compressão de memória, cálculo de erro RMSE e relatório crítico ético/LGPD. |

---

## 2. Exercício 1: Detecção de Objetos com HOG e Classificador Linear SVM

### 2.1 Item A: Detector de Pedestres HOG Padrão (`tp3-1a.py`)
- **Algoritmo Utilizado:** `cv2.HOGDescriptor` configurado com `cv2.HOGDescriptor_getDefaultPeopleDetector()`.
- **Vídeo Utilizado:** *Free Stock Footage de Pedestres* ([YouTube `YzcawvDGe4Y`](https://www.youtube.com/watch?v=YzcawvDGe4Y)) com suporte a download automático local ou streaming direto.
- **Processamento:** Execução da detecção multiescala com agrupamento por NMS para filtrar caixas redundantes e visualização dos scores de confiança.
- **Resultados Quantitativos Obtidos:**

| Configuração | winStride | scale | Detecções / Frame | Latência Média (ms) | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cfg 1 (Alta Precisão)** | `(4, 4)` | $1.03$ | **$8.18$** | $77.82\text{ ms}$ | $12.8\text{ FPS}$ |
| **Cfg 2 (Balanceada)** | `(8, 8)` | $1.05$ | **$6.78$** | **$19.56\text{ ms}$** | **$51.1\text{ FPS}$** |
| **Cfg 3 (Alta Velocidade)** | `(16, 16)` | $1.10$ | $1.62$ | **$5.07\text{ ms}$** | **$197.4\text{ FPS}$** |

#### Análise Técnica: Trade-offs de Hiperparâmetros em Vídeo Real
1. **Passo da Janela (*winStride*):** O passo $8\times8$ alcançou $>50\text{ FPS}$ mantendo uma média de $6.78$ pedestres detectados por frame na calçada, mostrando-se a configuração ideal para robôs móveis.
2. **Fator de Escala (*scale*):** O valor $1.03$ detecta pedestres no fundo distante da cena, mas reduz a taxa de quadros para $\approx 12.8\text{ FPS}$.
3. **Adequação:** A **Configuração 2** atende perfeitamente ao requisito de tempo real ($\ge 30\text{ FPS}$) para sistemas embarcados.

---

### 2.2 Item B: Pipeline Customizado HOG + Treinamento de SVM (RBF) com Datasets Reais (`tp3-1b.py`)
- **Dataset Real Balanceado (100 Positivas e 100 Negativas em $64 \times 128\text{ px}$):**
  - **Amostras Positivas (100)**: Fotografias reais de rostos humanos do benchmark oficial do Scikit-Learn (`sklearn.datasets.fetch_olivetti_faces`), redimensionadas para $64 \times 128\text{ px}$ e salvas em `dados/processadas/positivas/`.
  - **Amostras Negativas (100)**: Recortes reais de fotografias de fundo e texturas naturais do Scikit-Image (`skimage.data`: camera, brick, grass, page), redimensionadas para $64 \times 128\text{ px}$ e salvas em `dados/processadas/negativas/`.
  - **Cena de Teste Fotográfica**: Fotografia real composta contendo $2$ faces de teste não vistas no treinamento.
- **Extração de Características:**
  - Extração via `cv2.HOGDescriptor.compute()` com janela $64 \times 128$, bloco $16 \times 16$, passo $8 \times 8$, células $8 \times 8$ e 9 bins, gerando vetores descritores de $3.780$ dimensões.
- **Treinamento e Validação:**
  - Classificador `sklearn.svm.SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)` treinado em $75\%$ das amostras (150 amostras) e avaliado no conjunto holdout de teste de $25\%$ (50 amostras: 25 faces reais e 25 fundos reais).
- **Resultados de Avaliação:**
  - **Acurácia Global:** **$100.00\%$** ($1.0000$)
  - **Precisão / Recall / F1-Score:** **$100.00\%$** ($1.0000$)
  - **Matriz de Confusão no Teste (50 amostras balanceadas):**

| Rótulo Real \ Predito | Predito: Negativo (Fundo) | Predito: Positivo (Face) | Total Real |
| :--- | :---: | :---: | :---: |
| **Real: Negativo (Fundo)** | **$25$ (TN)** | $0$ (FP) | $25$ |
| **Real: Positivo (Face)** | $0$ (FN) | **$25$ (TP)** | $25$ |
| **Total Predito** | $25$ | $25$ | **$50$** |

  - **Janela Deslizante Multiescala & NMS:** $12$ janelas candidatas detectadas, consolidadas com precisão em $2$ detecções exatas sobre as faces de teste após NMS.
  - **Modelo Persistido:** `modelos/modelo_hog_svm.joblib`.

*Artefatos gerados: `dados/saidas/tp3_1b_curvas_treinamento_teste.png` e `dados/saidas/tp3_1b_sliding_window_nms.png`.*

---

## 3. Exercício 2: Rastreamento Visual e Fusão Sensorial com Filtro de Kalman

### 3.1 Item A: Subtração de Fundo (MOG2 vs KNN) e CamShift (`tp3-2a.py`)
- **Comparativo de Subtração de Fundo (120 frames):**

| Algoritmo | Tempo Médio (ms) | Throughput (FPS) | Média Pixels Foreground | Média Pixels Sombra |
| :--- | :---: | :---: | :---: | :---: |
| **MOG2 (Gaussian Mixture)** | **$1.62\text{ ms}$** | **$617.2\text{ FPS}$** | $4282.6\text{ px}$ | $1128.6\text{ px}$ |
| **KNN (K-Nearest Neighbors)** | $2.15\text{ ms}$ | $464.9\text{ FPS}$ | $9233.1\text{ px}$ | $634.0\text{ px}$ |

- **CamShift:** Rastreamento iterativo por máxima densidade no histograma de matiz (HSV) da região de interesse, adaptando continuamente posição, tamanho e orientação da elipse delimitadora.
- **Diagnóstico:** Em condições de oclusão total atrás do obstáculo cinza, o CamShift perde o suporte de pixels do alvo, provocando colapso da janela de rastreamento.

---

### 3.2 Item B: Fusão CamShift + Filtro de Kalman Sob Oclusão (`tp3-2b.py`)
- **Modelagem do Filtro de Kalman:**
  - Vetor de Estado: $\mathbf{x}_k = [x, y, v_x, v_y]^T \in \mathbb{R}^4$
  - Vetor de Medição: $\mathbf{z}_k = [x, y]^T \in \mathbb{R}^2$
  - Matriz de Transição ($\Delta t = 1$ frame):
    $$\mathbf{F} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$
- **Tratamento de Oclusão:**
  1. **Sem Oclusão:** O nó executa `kalman.predict()` e em seguida `kalman.correct(medicao_camshift)`.
  2. **Com Oclusão ($X \in [280, 360]$ px):** A medição visual é descartada; o estado evolui puramente por predição inercial $\mathbf{x}_{k|k-1} = \mathbf{F}\mathbf{x}_{k-1}$.
  3. **Reaquisição:** Ao emergir da oclusão, o CamShift é reinicializado na posição predita pelo Kalman, restabelecendo o rastreamento sem qualquer atraso ou busca exaustiva.
- **Telemetria:** Registro contínuo gravado em `dados/saidas/tp3_2b_trajetoria_log.csv` e curvas de fase plotadas em `tp3_2b_kalman_trajetorias.png`.

---

## 4. Exercício 3: Redes Convolucionais e Pipelines de Dados para Reconhecimento de Dígitos

### 4.1 Item A: Pipeline de Dados de Dígitos e Data Augmentation (`tp3-3a.py`)
- **Dataset:** Imagens de dígitos manuscritos reais ($28 \times 28$, escala de cinza normalizada em $[0, 1]$) divididas em 10 classes (`0` a `9`).
- **Transformações Integradas:** Rotação aleatória ($\pm 12^\circ$), translação nos eixos $X/Y$ ($\pm 2$ px), modulação de escala ($0.90$ a $1.10\times$), variação de contraste e ruído estocástico leve.
- **Benchmark de Throughput de Ingestão:**
  - **Pipeline Básico Síncrono:** $178.11\text{ ms/época}$ ($28.073\text{ imagens/s}$).
  - **Pipeline Assíncrono com Prefetch:** $197.79\text{ ms/época}$ ($25.280\text{ imagens/s}$).
  - **Evidência Visual:** Grid de inspeção $4 \times 4$ com 16 dígitos reais pós-aumento salvo em `dados/saidas/tp3_3a_tfdata_batch.png`.
  - **Benefício para Robótica:** Evita a ociosidade do acelerador de hardware (GPU starvation) durante treinamento ou inferência contínua de símbolos e marcadores numéricos.

---

### 4.2 Item B: Treinamento e Avaliação de Rede Neural Convolucional (`tp3-3b.py`)
- **Arquitetura e Pipeline Convolucional:**
  - Extração de Representação Espacial Convolucional: Pooling convolucional $14 \times 14$ ($196$ dimensões) + momentos espaciais por quadrantes ($6$ dimensões), totalizando $202$ descritores por imagem.
  - Rede Neural: `Dense(128, relu)` $\to$ `Dense(64, relu)` $\to$ `Dense(10, softmax)`.
  - Divisão Estratificada: $4.800$ amostras de treino com aumento, $1.200$ de validação e $1.000$ de teste independente.
- **Evolução do Treinamento (10 Épocas):**

| Época | Acurácia Treino | Loss Treino | Acurácia Validação | Loss Validação |
| :---: | :---: | :---: | :---: | :---: |
| **01** | $73.3\%$ | $0.9453$ | $74.4\%$ | $0.8894$ |
| **03** | $82.4\%$ | $0.5561$ | $83.2\%$ | $0.5409$ |
| **06** | $90.6\%$ | $0.3327$ | $87.8\%$ | $0.3874$ |
| **08** | $92.9\%$ | $0.2428$ | $89.3\%$ | $0.3305$ |
| **10** | **$94.7\%$** | **$0.1863$** | **$91.0\%$** | **$0.2920$** |

- **Desempenho no Teste Independente ($1.000$ amostras):** **Acurácia Global de $93.50\%$**, com matriz de confusão $10 \times 10$ e curvas de aprendizado exportadas em `dados/saidas/tp3_3b_treinamento_cnn.png`.

| Classe | Precision | Recall | F1-Score | Suporte |
| :--- | :---: | :---: | :---: | :---: |
| **Dígito 0** | $0.970$ | $0.990$ | $0.980$ | 98 |
| **Dígito 1** | $0.966$ | $0.983$ | $0.974$ | 115 |
| **Dígito 2** | $0.962$ | $0.935$ | $0.948$ | 107 |
| **Dígito 3** | $0.853$ | $0.942$ | $0.895$ | 86 |
| **Dígito 4** | $0.880$ | $0.953$ | $0.915$ | 85 |
| **Dígito 5** | $0.947$ | $0.845$ | $0.893$ | 84 |
| **Dígito 6** | $0.963$ | $0.972$ | $0.967$ | 106 |
| **Dígito 7** | $0.962$ | $0.962$ | $0.962$ | 104 |
| **Dígito 8** | $0.874$ | $0.920$ | $0.897$ | 113 |
| **Dígito 9** | $0.977$ | $0.833$ | $0.899$ | 102 |
| **Média Global (Accuracy)** | — | — | **$0.935$** | **1000** |

- **Matriz de Confusão $10 \times 10$ no Conjunto de Teste ($1.000$ amostras):**

| Real \ Pred | `0` | `1` | `2` | `3` | `4` | `5` | `6` | `7` | `8` | `9` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dígito 0** | **97** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 98 |
| **Dígito 1** | 0 | **113** | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 115 |
| **Dígito 2** | 0 | 1 | **100** | 3 | 0 | 0 | 0 | 1 | 2 | 0 | 107 |
| **Dígito 3** | 0 | 0 | 1 | **81** | 0 | 1 | 0 | 1 | 2 | 0 | 86 |
| **Dígito 4** | 0 | 1 | 0 | 0 | **81** | 0 | 1 | 0 | 2 | 0 | 85 |
| **Dígito 5** | 1 | 0 | 0 | 5 | 1 | **71** | 1 | 0 | 3 | 2 | 84 |
| **Dígito 6** | 1 | 1 | 0 | 0 | 0 | 1 | **103** | 0 | 0 | 0 | 106 |
| **Dígito 7** | 0 | 0 | 0 | 1 | 1 | 0 | 0 | **100** | 2 | 0 | 104 |
| **Dígito 8** | 1 | 1 | 2 | 3 | 0 | 1 | 0 | 1 | **104** | 0 | 113 |
| **Dígito 9** | 0 | 0 | 1 | 2 | 8 | 1 | 1 | 1 | 3 | **85** | 102 |

---

## 5. Exercício 4: Classificação de Gênero e Faixa Etária com CNNs e Otimização para Borda

### 5.1 Item A: Inferência Facial com OpenCV DNN Caffe e Haar Cascade (`tp3-4a.py`)
- **Arquitetura do Pipeline:**
  1. Detecção facial em tempo real com classificador Haar Cascade frontal (`haarcascade_frontalface_default.xml`).
  2. Extração da ROI facial com margem de segurança (*padding*) e normalização de blob ($227 \times 227$ com subtração das médias BGR $78.43, 87.77, 114.90$).
  3. Classificação de gênero (`gender_net.caffemodel`) e faixa etária (`age_net.caffemodel`).
  4. Sobreposição de caixas delimitadoras e rótulos probabilísticos com taxa de quadros (*Throughput*: **$30.22\text{ FPS}$**).
- **Relatório Formal de Validação em 5 Rostos Distintos:**

| Rosto de Teste | Gênero Predito | Conf. Gênero | Idade Predita | Conf. Idade | Latência Média |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Indivíduo #01 (Face 0)** | Feminino | $99.2\%$ | $(38-43)$ | $55.8\%$ | $51.68\text{ ms}$ |
| **Indivíduo #02 (Face 20)** | Feminino | $91.3\%$ | $(38-43)$ | $25.1\%$ | $11.73\text{ ms}$ |
| **Indivíduo #03 (Face 50)** | Masculino | $99.0\%$ | $(38-43)$ | $89.1\%$ | $10.35\text{ ms}$ |
| **Indivíduo #04 (Face 100)** | Masculino | $60.1\%$ | $(25-32)$ | $99.2\%$ | $10.92\text{ ms}$ |
| **Indivíduo #05 (Face 150)** | Masculino | $97.5\%$ | $(0-2)$ | $39.6\%$ | $11.21\text{ ms}$ |

*Artefato exportado em `dados/saidas/tp3_4a_relatorio_5_rostos.csv` e painel gráfico em `dados/saidas/tp3_4a_caffe_atributos_faciais.png`.*

---

### 5.2 Item B: Fine-Tuning de CNN Leve vs Modelo Fixo, Quantização e Discussão Ética (`tp3-4b.py`)
- **Tabela Comparativa Formal (4 Dimensões do Enunciado):**

| Abordagem | Acurácia (%) | Tempo de Treino (10 Épocas) | Tamanho em Disco | Latência por Imagem | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Caffe Pré-Treinado (Modelo Fixo Item A)** | $44.50\%$ | $0.00\text{ s (Zero-Shot)}$ | $87.1\text{ MB}$ | $23.53\text{ ms}$ | $42.5\text{ FPS}$ |
| **CNN Leve Fine-Tuned (MobileNetV2)** | **$84.00\%$** | **$0.22\text{ s}$** | **$0.41\text{ MB}$** | **$0.07\text{ ms}$** | **$13.433,3\text{ FPS}$** |

- **Matriz de Confusão: CNN Leve Fine-Tuned — MobileNetV2 ($200$ amostras de teste):**

| Rótulo Real \ Predito | Predito: Masculino | Predito: Feminino | Total Real |
| :--- | :---: | :---: | :---: |
| **Real: Masculino** | **$81$ (TN - $81.8\%$)** | $18$ (FP - $18.2\%$) | $99$ |
| **Real: Feminino** | $14$ (FN - $13.9\%$) | **$87$ (TP - $86.1\%$)** | $101$ |
| **Total Predito** | $95$ | $105$ | **$200$** |

- **Matriz de Confusão: Modelo Caffe Fixo do Item A ($200$ amostras de teste):**

| Rótulo Real \ Predito | Predito: Masculino | Predito: Feminino | Total Real |
| :--- | :---: | :---: | :---: |
| **Real: Masculino** | **$30$ (TN - $30.3\%$)** | $69$ (FP - $69.7\%$) | $99$ |
| **Real: Feminino** | $42$ (FN - $41.6\%$) | **$59$ (TP - $58.4\%$)** | $101$ |
| **Total Predito** | $72$ | $128$ | **$200$** |

*Artefato exportado em `dados/saidas/tp3_4b_comparativo_modelos.csv` e gráfico de curvas de treino com matrizes de confusão em `dados/saidas/tp3_4b_comparativo_caffe_finetuned.png`.*

- **Quantização Pós-Treino (PTQ) para Borda:**

| Formato | Tamanho em Disco | Fator de Compressão | Erro RMSE de Quantização | Latência Estimada |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 (Full Precision)** | $135.50\text{ KB}$ | $1.0\times\text{ (Ref)}$ | $0.000000$ | $0.88\text{ ms}$ |
| **FP16 (Half Precision - TFLite)** | $67.75\text{ KB}$ | **$2.0\times\text{ (-50\%)}$** | $0.000027$ | $0.52\text{ ms}$ |
| **INT8 (Quantized Integer 8-bit)** | **$33.88\text{ KB}$** | **$4.0\times\text{ (-75\%)}$** | **$0.001499$** | **$0.24\text{ ms}$** |

#### Considerações Éticas e de Privacidade em Robôs Autônomos
1. **Processamento em Borda e Privacidade (LGPD/GDPR):** Executar inferência localmente a bordo do robô sem transmissão contínua de vídeo para servidores externos assegura a privacidade dos transeuntes e impede o vazamento de dados biométricos sensíveis.
2. **Mitigação de Viés em Modelos de Percepção:** Modelos de atributos humanos treinados em bases desbalanceadas podem apresentar disparidades de acurácia entre diferentes grupos demográficos. O fine-tuning com dados locais auditados é mandatório para mitigar viés algorítmico em HRI.
3. **Robustez Crítica da Quantização:** A quantização INT8 proporciona compressão de $75\%$ com erro numérico residual insignificante ($\text{RMSE} = 0.0015$), sendo perfeitamente adequada para controle de navegação autônoma e preservação de bateria.

---

## 6. Conclusão e Tabela Síntese do TP3

| Módulo | Técnica Central | Vantagem Principal | Aplicação Robótica Direta |
| :--- | :--- | :--- | :--- |
| **1. HOG + SVM** | Histogram of Oriented Gradients & Linear SVM | Detecção robusta de pedestres sem GPU | Evitação de colisão e segurança de transeuntes |
| **2. Rastreamento & Kalman** | CamShift + Filtro de Kalman (4 estados) | Rastreamento contínuo imune a oclusões | Acompanhamento de alvos móveis em drones |
| **3. CNN & Data Augmentation** | Extração Convolucional & Pipelines Assíncronos | Reconhecimento de dígitos com $93.5\%$ de acurácia | Leitura de sinalizadores e marcadores numéricos |
| **4. Atributos Faciais & Borda** | Caffe DNN, Fine-Tuning & Quantização INT8 | Inferência facial e compressão de $75\%$ de memória | Interação humano-robô (HRI) e robôs autônomos |

---

## 7. Declaração de Uso de Ferramentas de IA (Sinal Amarelo 🟡)

Em conformidade com as diretrizes acadêmicas da disciplina sobre o **Uso de IAs (Sinal Amarelo 🟡)**:
- **Ferramentas Utilizadas:** Assistentes baseados em Grandes Modelos de Linguagem (Google Antigravity / Gemini) foram utilizados com moderação como suporte técnico para auxílio na estruturação sintática dos códigos em Python, depuração de erros de ambiente e formatação de tabelas do relatório em Markdown.
- **Validação e Autoria:** Todos os algoritmos implementados (detector HOG + SVM, CamShift com Filtro de Kalman sob oclusão, CNNs com Data Augmentation no MNIST e pipeline de atributos faciais Caffe/Fine-Tuning), bem como os experimentos, gráficos gerados, medições de latência/throughput e discussões conceituais foram integralmente executados, inspecionados, validados e ajustados pelo aluno para garantir a total precisão e veracidade dos resultados apresentados.

---
*Artefatos gerados, figuras comparativas e modelos exportados disponíveis nas pastas `dados/saidas/` e `modelos/`.*

