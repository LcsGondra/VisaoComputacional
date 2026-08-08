# Relatório Técnico: Pipeline de Percepção Visual para Robótica Autônoma

## 1. Introdução e Encadeamento do Pipeline de Visão

Em sistemas robóticos autônomos, como veículos terrestres guiados (AGVs) ou veículos aéreos não tripulados (drones), a percepção visual fornece a realimentação sensorial necessária para navegação, desvio de obstáculos e rastreamento de alvos em tempo real. As técnicas fundamentais exploradas no TP1 conectam-se em uma cadeia sequencial determinística, transformando frames de vídeo brutos em sinais acionáveis de controle de trajetória.

```
+-----------------------------------------------------------------------------------+
|                            PIPELINE DE PERCEPÇÃO VISUAL                          |
+-----------------------------------------------------------------------------------+
| 1. CAPTURA E STREAMING                                                           |
|    VideoCapture -> Frame BGR -> Verificação de Resolução e Taxa de FPS            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. PRÉ-PROCESSAMENTO E SEGMENTAÇÃO HÍBRIDA                                        |
|    - Conversão BGR para HSV (Faixas duplas de vermelho [0..6] e [172..180])       |
|    - Cálculo da Redness Relativa: R - max(G, B) com Limiarização de Otsu          |
|    - Fusão de Máscaras (bitwise_and) + Filtro Gaussiano                           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. EXTRAÇÃO GEOMÉTRICA E ANÁLISE DE CONTORNOS                                    |
|    cv2.findContours -> Filtros por Área -> Cálculo do Centroide via Momentos      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. GERAÇÃO DE SINAL DE CONTROLE E NAVEGAÇÃO (HUD SIMULADO)                        |
|    - Cálculo do Erro Lateral Normalizado: e_x = (cx - center_x) / center_x        |
|    - Seta Dinâmica de Correção de Trajetória e Comando de Guinada no HUD          |
+-----------------------------------------------------------------------------------+
```

## 2. Detalhamento Técnico das Etapas e Algoritmos

1. **Captura e Streaming:**
   O pipeline captura o stream de vídeo em tempo real, monitorando continuamente o tempo entre frames via `cv2.getTickCount()` para garantir que a taxa de FPS permaneça compatível com a malha de controle do robô.

2. **Segmentação Híbrida de Cor e Contraste:**
   Para contornar o problema de ambiguidade de fundo avermelhado e presença de cores próximas (amarelo e laranja), implementou-se uma fusão lógica:
   - **Máscara HSV de Dupla Faixa:** Filtra o matiz (Hue) especificamente nas extremidades do espectro vermelho (`[0..6]` e `[172..180]`), descartando matizes amarelos/laranjas (`H > 6`).
   - **Redness Relativa:** Avalia a dominância da cor vermelha através de $R - \max(G, B)$, eliminando superfícies com alta intensidade mas baixa pureza cromática.
   - **Combinação Lógica (`cv2.bitwise_and`):** Garante a segmentação precisa do objeto de interesse.

3. **Extração do Centroide e Sinal de Erro:**
   A partir dos contornos extraídos (`cv2.findContours`), o maior blob é isolado e seus momentos de imagem ($M_{00}, M_{10}, M_{01}$) são calculados para obter o centroide $(c_x, c_y)$. O desvio lateral normalizado $e_x \in [-1.0, +1.0]$ é enviado ao controlador:
   $$e_x = \frac{c_x - c_{\text{centro}}}{c_{\text{centro}}}$$

4. **Visualização HUD de Controle:**
   O sistema sobrepõe no frame uma linha guia vertical central, o retângulo delimitador (*bounding box*), o ponto do centroide e um **vetor dinâmico de correção de trajetória** (`cv2.arrowedLine`), indicando a direção e magnitude da guinada necessária (`VIRAR A ESQUERDA` ou `VIRAR A DIREITA`).

## 3. Limitações Técnicas em Condições Reais de Operação

Apesar de funcional e robusto para cenários de teste, o pipeline clássico apresenta limitações inerentes sob condições operacionais adversas:

### a) Iluminação Variável e Sombras Dinâmicas
- **Limitação:** Variações bruscas de luminosidade solar, sombras dinâmicas ou iluminação artificial alteram os valores de saturação e brilho no espaço HSV.
- **Impacto:** Podem provocar a perda de segmentação ou fragmentação do blob principal.

### b) Oclusão Parcial e Fundos Complexos
- **Limitação:** Se o objeto de interesse for parcialmente bloqueado por um obstáculo, o contorno é dividido em múltiplos fragmentos menores.
- **Impacto:** A área filtrada pode cair abaixo do limiar operacional (`area > 300 px²`), resultando na perda temporária do alvo.

### c) Movimento de Alta Velocidade e Blur
- **Limitação:** Movimentos rápidos do robô ou da câmera geram manchas de movimento (*motion blur*), borrando as bordas do objeto.
- **Impacto:** Causa oscilações de alta frequência no cálculo do centroide, gerando ruído no sinal de controle.

## 4. Perspectivas e Soluções Futuras (TP2 em Diante)

Para superar essas fragilidades em sistemas robóticos de nível industrial ou automotivo, os próximos módulos introduzirão abordagens avançadas:

1. **Rastreamento Temporal e Filtros de Estado (Filtro de Kalman / Particle Filters):**
   Manterão a estimativa da posição e velocidade do objeto mesmo durante oclusões temporárias ou lapsos de detecção.
2. **Descritores de Características Invariantes (SIFT, SURF, ORB):**
   Permitirão a identificação do objeto por pontos de interesse característicos invariantes à iluminação, rotação e escala.
3. **Redes Neurais Convolucionais para Detecção (YOLO / MobileNet):**
   Substituirão os limiares manuais de cor por redes treinadas capazes de detectar objetos complexos em fundos heterogêneos.
4. **Estimativa de Pose e SLAM Visual:**
   Possibilitarão ao robô construir o mapa 3D do ambiente e estimar sua própria trajetória no espaço (odometria visual).
