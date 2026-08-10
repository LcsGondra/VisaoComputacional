# Relatório Técnico: Pipeline de Percepção Visual para Robótica Autônoma

---

## 1. Visão Geral e Diagrama em ASCII do Pipeline Completo

Em sistemas robóticos autônomos — como veículos terrestres guiados (AGVs/AMRs) ou veículos aéreos não tripulados (drones) —, a percepção visual em tempo real fornece a realimentação sensorial necessária para navegação, desvio de obstáculos e rastreamento de alvos. 

As técnicas implementadas no módulo TP1 (captura de streaming, manipulação de espaços de cor, limiarização, detecção de bordas e extração de contornos) conectam-se em um pipeline sequencial determinístico, transformando quadros de vídeo brutos em sinais acionáveis de controle de trajetória:

```text
+-----------------------------------------------------------------------------------+
|                  PIPELINE DE PERCEPÇÃO VISUAL PARA VEÍCULOS AUTÔNOMOS             |
+-----------------------------------------------------------------------------------+
| 1. CAPTURA E PROCESSAMENTO DE FRAMES (Exercício 1)                                |
|    VideoCapture -> Frame BGR -> Verificação de FPS & Aspect Ratio (tp1-1a.py)     |
|    Amostragem & Conversão BGR para Escala de Cinza (tp1-1b.py)                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. ESPAÇOS DE COR E REALCE DE NITIDEZ (Exercício 2)                              |
|    - Análise de 7 Canais BGR / HSV (H, S, V) / LAB (L, a, b) (tp1-2a.py)          |
|    - Ajuste de Saturação (0%, 50%, 150%)                                         |
|    - Realce de Nitidez: Convolução Sharpening 3x3 vs Unsharp Masking (tp1-2b.py)  |
|    - Métrica de Avaliação Objetiva: Variância do Laplaciano (Var(Laplacian))      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. LIMIARIZAÇÃO E ANÁLISE GEOMÉTRICA DE CONTORNOS (Exercício 3)                   |
|    - Binarização: Global (T=127) vs Adaptativa Gaussiana vs Otsu (tp1-3a.py)       |
|    - Detecção de Bordas Canny com 2 Pares de Limiares (30/90 e 120/240) (tp1-3b.py)|
|    - Extração de Contornos (findContours) & Filtragem por Área > 200 px²         |
|    - Categorização de Tamanho (Verde: Pequeno, Amarelo: Médio, Vermelho: Grande)  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. RASTREAMENTO E GERAÇÃO DE SINAL DE CONTROLE NO HUD (Exercício 4)               |
|    - Segmentação Híbrida de Cor (HSV Dupla Faixa + Redness R - max(G,B))         |
|    - Cálculo do Centroide Espacial via Momentos de Imagem (M00, M10, M01)        |
|    - Cálculo do Erro Lateral Normalizado: e_x = (cx - center_x) / center_x        |
|    - Interface HUD com Seta Dinâmica (arrowedLine) e Comandos de Guinada (tp1-4a) |
+-----------------------------------------------------------------------------------+
```

---

## 2. Detalhamento Técnico das Etapas e Algoritmos

### 2.1 Captura e Pré-processamento de Frames
O pipeline inicia com o recebimento do fluxo contínuo de vídeo do sensor de câmera. Para evitar distorções de proporção durante a renderização em diferentes resoluções de tela, implementou-se a função `imshow_keep_aspect_ratio`, adicionando preenchimento neutro (*letterboxing/pillarboxing*). O tempo entre frames é monitorado via `cv2.getTickCount()` para garantir estabilidade da taxa de quadros por segundo (FPS).

### 2.2 Manipulação em Espaços de Cor e Nitidez
A conversão do espaço RGB/BGR original para **HSV (Hue, Saturation, Value)** e **LAB (Lightness, a, b)** isola a informação cromática da luminância. Enquanto o espaço HSV facilita a segmentação por cor independentemente da iluminação, o espaço LAB espelha a percepção humana de brilho. A aplicação de filtros de nitidez (**Sharpening 3x3** e **Unsharp Masking**) restaura bordas de alta frequência, e a **Variância do Operador Laplaciano** ($\text{Var}(\nabla^2 I)$) fornece uma medição objetiva do nível de nitidez da cena.

### 2.3 Binarização e Extração Geométricas de Contornos
A imagem em escala de cinza passa por processos de binarização. O algoritmo de **Otsu** calcula automaticamente o limiar ótimo ($T$) minimizando a variância intraclasse dos pixels. Em seguida, o detector **Canny** extrai o mapa de bordas utilizando dois pares de limiares distintos ($30/90$ e $120/240$). Os contornos resultantes são extraídos via `cv2.findContours`, filtrados por área mínimo ($\text{Área} > 200\text{ px}^2$) e classificados em faixas cromáticas (Verde para pequeno, Amarelo para médio e Vermelho para grande).

### 2.4 Rastreamento de Alvos e Sinal de Controle Lateral
Para o rastreamento em tempo real, combina-se uma **máscara HSV de dupla faixa** (tratando a descontinuidade do vermelho em $0^\circ / 180^\circ$) com a **Redness Relativa** ($R - \max(G, B)$). O centroide $(c_x, c_y)$ do maior contorno é calculado por momentos de imagem ($M_{00}, M_{10}, M_{01}$):

$$c_x = \frac{M_{10}}{M_{00}}, \quad c_y = \frac{M_{01}}{M_{00}}$$

O desvio lateral normalizado $e_x \in [-1.0, +1.0]$ em relação ao centro da imagem ($c_{\text{centro}}$) gera a resposta do controlador de guinada:

$$e_x = \frac{c_x - c_{\text{centro}}}{c_{\text{centro}}}$$

- $e_x < -0.05 \implies$ `CORRECAO: VIRAR A ESQUERDA`
- $e_x > +0.05 \implies$ `CORRECAO: VIRAR A DIREITA`
- $|e_x| \le 0.05 \implies$ `CORRECAO: EM FRENTE (ALINHADO)`

---

## 3. Limitações de Cada Etapa em Condições Reais de Operação

Apesar da eficiência computacional do pipeline clássico, a operação de robôs em ambientes reais não controlados expõe fragilidades técnicas inerentes a cada etapa:

### 3.1 Limitações da Etapa de Captura e FPS
- **Condições Reais (Velocidade e Motion Blur):** Movimentos de alta velocidade do robô ou rajadas de vento em drones provocam desfoque de movimento (*motion blur*) e vibração da câmera.
- **Impacto no Sistema:** Borra transições de borda nítidas, reduzindo o valor da variância do Laplaciano e provocando oscilações bruscas ou perda temporária na detecção do centroide.

### 3.2 Limitações dos Espaços de Cor e Filtragem
- **Condições Reais (Iluminação Variável e Sombras Dinâmicas):** Alterações de luz solar, entrada em túneis ou transições de iluminação artificial alteram severamente os valores de saturação ($S$) e brilho ($V$) no espaço HSV.
- **Impacto no Sistema:** Faixas fixas de cor deixam de segmentar o objeto alvo ou passam a capturar ruídos no fundo que possuem a mesma tonalidade sob nova iluminação.

### 3.3 Limitações da Limiarização e Contornos
- **Condições Reais (Oclusão Parcial e Fundos Complexos):** Se o objeto de interesse for parcialmente bloqueado por um obstáculo físico (ex: uma árvore ou outro veículo), o contorno contínuo é fragmentado em múltiplos pedaços menores.
- **Impacto no Sistema:** A área filtrada de cada fragmento cai abaixo do limiar operacional ($\text{Área} > 200\text{ px}^2$), levando o robô a assumir que o alvo desapareceu completamente.

---

## 4. Competências do TP2 em Diante para Endereçar as Limitações

Para superar essas fragilidades e evoluir para sistemas robóticos autônomos de padrão industrial e automotivo, os próximos módulos introduzirão técnicas avançadas de Visão Computacional:

1. **Rastreamento Temporal e Filtros de Estado (Filtro de Kalman / Particle Filters):**
   - *Solução para Oclusão e Ruído:* Manterão a estimativa da posição e velocidade do objeto mesmo durante oclusões temporárias ou lapsos de detecção de frames individuais.

2. **Descritores de Características Invariantes (SIFT, SURF, ORB):**
   - *Solução para Iluminação e Rotação:* Permitirão a identificação do objeto por pontos de interesse característicos locais, invariantes a mudanças severas de iluminação, escala e ângulo de visão.

3. **Redes Neurais Convolucionais para Detecção de Objetos (YOLO / MobileNet):**
   - *Solução para Fundos Complexos:* Substituirão limiares manuais de cor e cor por aprendizado profundo, identificando objetos por classes semânticas em qualquer tipo de fundo.

4. **Estimativa de Pose 3D e SLAM Visual (Simultaneous Localization and Mapping):**
   - *Solução para Navegação e Geometria:* Permitirão ao robô estimar sua própria posição tridimensional no espaço e mapear o ambiente em 3D a partir das câmeras embarcadas.
