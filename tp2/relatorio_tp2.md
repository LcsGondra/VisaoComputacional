# Relatório Técnico — Teste de Performance 2 (TP2)
**Disciplina:** Visão Computacional para Robótica  
**Aluno:** Lucas Dias de Gondra  

---

## 1. Instruções de Ambiente e Execução

### 1.1 Dependências do Sistema (Windows / PowerShell)
Para configurar e instalar as bibliotecas necessárias para executar todos os módulos do TP2:

```powershell
pip install --user opencv-python opencv-contrib-python tensorflow matplotlib scikit-learn deepface numpy
```

### 1.2 Guia de Execução dos Scripts

| Exercício | Script | Descrição do Pipeline |
| :--- | :--- | :--- |
| **Ex 1A** | `python tp2-1a.py` | Estimativa de disparidade estéreo (StereoSGBM), colormap JET e identificação de pixel mais próximo/distante. |
| **Ex 1B** | `python tp2-1b.py` | Segmentação de ROI em tempo real por HSV, morfologia (erode/dilate) e cálculo contínuo de proporção de área. |
| **Ex 2A** | `python tp2-2a.py` | Detector facial Haar Cascade, recorte/redimensionamento 48×48 px e análise de trade-off sensibilidade vs especificidade. |
| **Ex 2B** | `python tp2-2b.py` | Sistema de reconhecimento facial com identidades cadastradas, medição de latência em ms e discussão ética. |
| **Ex 3A** | `python tp2-3a.py` | Extração e comparação quantitativa de descritores SIFT, ORB e AKAZE sob transformação e iluminação. |
| **Ex 3B** | `python tp2-3b.py` | Correspondência de features (BFMatcher + FLANN + Lowe ratio), homografia RANSAC e alinhamento por perspectiva. |
| **Ex 4A** | `python tp2-4a.py` | Projeto e treinamento de CNN convolucional do zero (Keras), curvas de treino/validação e diagnóstico de overfitting. |
| **Ex 4B** | `python tp2-4b.py` | Extração de representações latentes (Dense-64), projeção PCA 2D e comparação de separabilidade com ORB. |

---

## 2. Exercício 1: Estimativa de Profundidade e Segmentação de ROI

### 2.1 Item A: Mapa de Disparidade e Profundidade Relativa (`tp2-1a.py`)
- **Algoritmo Utilizado:** `cv2.StereoSGBM_create` com 64 níveis de disparidade, tamanho de bloco $5\times5$, ponderadores de suavidade $P_1 = 8 \cdot C \cdot \text{block}^2$ e $P_2 = 32 \cdot C \cdot \text{block}^2$.
- **Processamento:** Normalização min-max da disparidade válida e mapeamento para escala pseudocor `cv2.COLORMAP_JET`.
- **Resultados Quantitativos Obtidos:**
  - Resolução das imagens estéreo: $640 \times 360$ pixels.
  - Cobertura de pixels com disparidade válida: **14.30%** (focada nas superfícies com textura discriminativa).
  - **Pixel mais próximo (maior disparidade):** Coordenadas $(66, 193)$, disparidade bruta $= 53.00\text{ px}$, valor normalizado $= 1.0000$.
  - **Pixel mais distante (menor disparidade válida):** Coordenadas $(468, 130)$, disparidade bruta $= 10.94\text{ px}$, valor normalizado $= 0.0000$.

#### Análise Técnica: Comportamento em Câmera Embarcada em Drone
1. **Linha de Base (*Baseline* $B$):** Drones leves impõem restrições de envergadura, limitando $B$ a $5\text{--}15\text{ cm}$. Pela relação $Z = \frac{f \cdot B}{d}$, pequenos erros na disparidade subpixel $d$ causam incerteza quadrática na profundidade para distâncias maiores que $10\text{ m}$.
2. **Vibrações e Perda de Retificação:** As vibrações de alta frequência dos motores provocam descalibração contínua dos eixos ópticos, exigindo retificação epipolar dinâmica ou algoritmos robustos a pequenos desalinhamentos verticais.
3. **Superfícies Homogêneas:** Voo sobre copas de árvores uniformes, corpos d'água ou asfalto gera ausência de textura, resultando em regiões de disparidade nula que demandam fusão sensorial com LiDAR ou radar de ondas milimétricas.

### 2.2 Item B: Pipeline de Segmentação de ROI em Vídeo (`tp2-1b.py`)
- **Etapas da Pipeline:**
  1. Conversão para espaço HSV e filtragem por faixa de matiz/saturação/brilho (`cv2.inRange`).
  2. Limpeza morfológica com elemento estruturante elíptico $5\times5$: erosão (1 iteração) para eliminar ruídos espúrios seguida de dilatação (2 iterações) para fechar lacunas internas.
  3. Extração do contorno de maior área e sobreposição de bounding box retangular com máscara colorida semitransparente via mistura linear ponderada (`cv2.addWeighted`).
  4. Cálculo em tempo real da fração da área: $\text{Proporção} = \frac{\text{Área}_{\text{ROI}}}{\text{Área}_{\text{Total}}} \times 100\%$.
- **Resultados:** Monitoramento contínuo com variação dinâmica da proporção de área entre $1.00\%$ e $3.72\%$ conforme o objeto se movimenta e varia de escala.

---

## 3. Exercício 2: Detecção e Reconhecimento Facial

### 3.1 Item A: Detector Facial com Haar Cascades (`tp2-2a.py`)
- **Classificador:** `cv2.CascadeClassifier` baseado em *Haar Feature-based Cascade Classifiers* (`haarcascade_frontalface_default.xml`).
- **Recorte e Normalização:** Para cada face detectada, a ROI facial é extraída, redimensionada para $48 \times 48$ pixels e persistida na pasta `dados/faces_48x48/`.
- **Avaliação Comparativa de Parâmetros:**

| Configuração | scaleFactor | minNeighbors | Detecções | Falsos Positivos | Recall Estimado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Configuração 1 (Alta Especificidade)** | 1.10 | 5 | 0 | 0 | 0.0% |
| **Configuração 2 (Alta Sensibilidade)** | 1.05 | 2 | 1 | 0 | 50.0% |

#### Justificativa do *Trade-off* Sensibilidade vs. Especificidade
- Um `scaleFactor` baixo ($1.05$) avalia uma densidade maior de escalas da janela deslizante, e um `minNeighbors` reduzido ($2$) requer menos janelas candidatas coincidentes para confirmar uma face. Isso eleva a **sensibilidade** (maior capacidade de detectar faces parciais ou em baixa resolução), mas diminui a especificidade em fundos ruidosos.
- Valores mais altos de `minNeighbors` ($\ge 5$) maximizam a **especificidade**, eliminando falsos positivos, porém correm o risco de omitir faces com oclusões parciais ou rotações fora do plano.

### 3.2 Item B: Sistema de Reconhecimento Facial e Considerações Éticas (`tp2-2b.py`)
- **Arquitetura de Reconhecimento:** Cadastro de banco de vetores de features para 3 identidades (*Alice*, *Bruno*, *Carla*), processamento de stream com matching de distância euclidiana/absoluta e classificação com limiar de rejeição para rótulo `"desconhecido"`.
- **Latência de Inferência:** Tempo médio de **$0.18\text{ ms}$ por frame** ($\approx 5400\text{ FPS}$ em modo vetorizado local), atendendo com folga à restrição de tempo real em sistemas embarcados robóticos.

#### Discussão Ética em Aplicações Robóticas e Drones
1. **Privacidade e Vigilância Massiva:** Câmeras móveis autônomas operando em espaço público sem aviso criam rastreamento permanente de cidadãos, violando diretrizes de proteção de dados (LGPD / GDPR).
2. **Viés Algorítmico e Assimetria de Erro:** Modelos treinados em bases desbalanceadas apresentam taxas de erro substancialmente maiores para minorias raciais e de gênero, podendo induzir falsas abordagens de segurança.
3. **Armamento Autônomo e Decisão Crítica:** A delegação de intervenções físicas ou letais baseadas unicamente em inferência de visão computacional apresenta risco inaceitável de falha operacional.

---

## 4. Exercício 3: Descritores de Características e Homografia

### 4.1 Item A: Extração e Comparação de SIFT, ORB e AKAZE (`tp2-3a.py`)
Sob transformações combinadas de rotação ($15^\circ$), escala ($0.92$), gradiente de iluminação e ruído aditivo:

| Descritor | Keypoints (Cena A) | Keypoints (Cena B) | Tempo Médio (ms) | Dimensão do Vetor | Natureza do Vetor |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SIFT** | 1201 | 1148 | 35.95 ms | 128 | Contínuo (`float32`, 512 bytes) |
| **ORB** | 1461 | 1500 | **4.94 ms** | 32 | Binário (`uint8`, 256 bits) |
| **AKAZE** | 1395 | 1499 | 34.73 ms | 61 | Binário MLDB (`uint8`, 488 bits) |

- **Análise Técnica:** O ORB demonstrou ser mais de **7 vezes mais rápido** que SIFT e AKAZE devido ao uso de testes binários baseados em intensidades de pares de pixels (BRIEF orientado), sendo a escolha ideal para robôs com restrição de bateria e processamento (ex: Raspberry Pi / Jetson Nano). O SIFT, por sua vez, preserva maior repetibilidade sob distorções fotométricas severas.

### 4.2 Item B: Casamento de Pontos, Filtro de Lowe e Homografia RANSAC (`tp2-3b.py`)
- **Estratégias de Casamento:**
  - `BFMatcher` com verificação cruzada (*cross-check*): 480 correspondências biunívocas.
  - `FlannBasedMatcher` com KD-Tree e teste de razão de Lowe ($\text{ratio} < 0.75$): **508 correspondências filtradas**.
- **Estimativa de Homografia com RANSAC:**
  - Total de inliers geometricamente consistentes: **358 de 508 (taxa de inliers de 70.47%)**.
  - Matriz de Homografia estimada:
    $$\mathbf{H} = \begin{bmatrix} 0.8883 & 0.2381 & -26.8186 \\ -0.2385 & 0.8888 & 128.7911 \\ 0.0000 & 0.0000 & 1.0000 \end{bmatrix}$$
- **Alinhamento por Projeção Perspectiva:** A imagem de entrada foi reprojetada com `cv2.warpPerspective` e sobreposta à imagem de referência, comprovando alinhamento pixel a pixel das formas geométricas.

#### Relevância para Localização Visual e SLAM Robótico
O cálculo de homografia com RANSAC é o alicerce da **Odometria Visual** e do **Fechamento de Laço (*Loop Closure*)** em SLAM: permite deduzir a pose relativa da câmera (rotação $\mathbf{R}$ e translação $\mathbf{t}$) entre quadros sucessivos, rejeitando *outliers* causados por reflexos ou oclusões.

---

## 5. Exercício 4: Redes Neurais Convolucionais com TensorFlow/Keras

### 5.1 Item A: Treinamento da Arquitetura CNN (`tp2-4a.py`)
- **Arquitetura Projetada:**
  - Camada de Entrada: $28 \times 28 \times 1$
  - Bloco Conv 1: `Conv2D(32, 3x3, relu, padding='same')` $\to$ `MaxPooling2D(2x2)`
  - Bloco Conv 2: `Conv2D(64, 3x3, relu, padding='same')` $\to$ `MaxPooling2D(2x2)`
  - Camada Densa: `Flatten()` $\to$ `Dense(64, relu, name='penultima_dense')` $\to$ `Dropout(0.25)`
  - Camada de Saída: `Dense(3, softmax)`
- **Treinamento:** 10 épocas com otimizador Adam ($\alpha = 0.001$), perda `sparse_categorical_crossentropy` e divisão de validação de $20\%$.
- **Resultados Finais:**
  - **Acurácia no Conjunto de Teste:** **$99.60\%$**
  - **Loss no Conjunto de Teste:** **$0.0148$**

#### Diagnóstico de Overfitting nas Curvas de Treino
- A acurácia de treino atingiu $99.93\%$ enquanto a de validação estabilizou em $99.50\%$ ($\Delta_{\text{acc}} = 0.43\%$).
- A loss de validação decaiu suavemente junto à loss de treino, demonstrando que a inclusão de camadas de pooling e regularização por Dropout evitou sobreajuste aos dados de treino.

### 5.2 Item B: Extração de Features Densas e Análise PCA 2D (`tp2-4b.py`)
- **Extração Latente:** Obtenção do vetor de $64$ dimensões da camada `penultima_dense` para 20 amostras de teste (balanceadas entre Camisetas, Calças e Tênis).
- **Projeção PCA 2D:**
  - Variância explicada pelas duas primeiras componentes (CNN): **$95.64\%$**.
  - Variância explicada pelo PCA 2D dos descritores ORB: **$100.00\%$** (devido ao colapso de variância em baixa dimensionalidade).

#### Comparação de Separabilidade: CNN vs Descritores Clássicos (ORB)
1. **Representações Aprendidas pela CNN:** A projeção 2D exibe três agrupamentos (*clusters*) perfeitamente isolados no espaço latente com margem inter-classe elevada. A rede aprendeu invariâncias semânticas globais (silhueta de calçado vs vestimenta superior vs inferior).
2. **Descritores Clássicos (ORB):** Como o ORB é sensível a cantos locais de alta frequência, em imagens $28\times28$ com pouca textura ocorre escassez de keypoints (média de $0.1$ keypoint/imagem), resultando em vetores médios colapsados que não conseguem discriminar as classes semanticamente.

---

## 6. Conclusão e Tabela Síntese do TP2

| Módulo | Técnica Central | Vantagem Principal | Aplicação Robótica Direta |
| :--- | :--- | :--- | :--- |
| **1. Profundidade & ROI** | StereoSGBM & HSV Morfológico | Baixa latência, não requer treinamento | Detecção de obstáculos e aproximação |
| **2. Detecção Facial** | Haar Cascades & Embeddings | Eficiente para triagem rápida | Interação humano-robô e controle de acesso |
| **3. Características Locais** | SIFT / ORB & Homografia RANSAC | Invariância geométrica rigorosa | SLAM visual e reconstrução de mapas 3D |
| **4. Deep Learning** | CNN (Keras) & Extração Latente PCA | Alta abstração semântica e acurácia ($>99\%$) | Reconhecimento e classificação de alvos |

---
*Artefatos, figuras geradas e modelos compilados disponíveis no diretório `dados/saidas/` e `modelos/`.*
