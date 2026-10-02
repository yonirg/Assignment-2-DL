# PA2 — roteiro da apresentação

> **Como ler.** As Partes 0–5 abaixo contam a história no **ambiente sintético**, onde cada fator
> (velocidade, oclusão, densidade) é controlado. A seção **[MOT17 — resultados reais](#mot17--resultados-reais)**,
> no fim, repete as mesmas Partes 1–5 no MOT17 real e mostra onde a história muda.
> Os slides estão em `PA2_apresentacao.pptx`, com notas do apresentador em cada slide.

Reprodução: `bash run_all.sh synth` (figuras em `results/synth/partN/`) e
`bash run_all.sh mot17 data/MOT17` (figuras em `results/mot17/partN/`; ~5 min em CPU, a galeria da
Parte 4 precisa das imagens `img1/`).

---

## Parte 0 — testes sintéticos

**Gerador** (`pa2/synth.py`). Vídeos 128×128 de 30–60 quadros, 5–15 elipses com tamanhos, ruído e
contraste variáveis. Botões: `n_objects`, `speed`, `occlusion_len` (+ ruído e contraste). Cada
elipse tem uma profundidade e o render é feito do fundo para a frente (algoritmo do pintor). Um
"poste" opaco fica sempre na frente, e a largura dele é `occlusion_len · speed + 2·eixo_max`. A
visibilidade é a fração de pixels da elipse que ainda pertencem a ela no mapa de donos, o mesmo
campo `visibility` do MOT17.
→ `part0/fig_occlusion.png`: o objeto 3 some atrás do poste por **13 quadros** (vis = 0,00) e volta.

**Simulador de detector** (`pa2/detsim.py`). Descarta p% das caixas, adiciona ruído ∝ tamanho,
injeta FPs com Poisson por quadro e duplicatas (removidas pelo NMS próprio). Objetos com
visibilidade < 0,35 não são detectados. → `part0/fig_detector_sim.png`.

**Métrica** (`metrics.py`, `tests/test_metrics.py`, saída em `part0/metric_tests.txt`):

| caso | IDSW | Frag | IDF1 | erro de contagem |
|---|---|---|---|---|
| (a) predição = gt | 0 | 0 | **1,0** | 0 |
| (b) 2 objetos trocam ids no quadro k=11 (N=20) | **2** | 0 | max(k−1, N−k+1)/N = **0,5** | 0 |
| (c) mesma cena, objeto 1 partido em 2 com buraco de 2 quadros | **1** | **1** | **60/78 = 0,769** | **+1** |

Por que (b) ≠ (c): a troca estraga **duas** identidades sem mudar a contagem. A partição estraga
**uma**, e o pedaço que sobra vira IDFP *e* uma identidade a mais na contagem. O IDSW conta
eventos e o IDF1 conta quadros sob a melhor atribuição global, por isso medem coisas diferentes.

**Piso fácil.** IoU ingênuo com 3–5 elipses lentas (0,5 px/q), sem oclusão e detector quase
perfeito: **IDF1 = 0,990, 0 switches, #ids previstas/#ids verdadeiras = 1,00**.

**Onde o baseline quebra** (`part0/fig_sweep.png`, 10 vídeos por ponto, detector simulado padrão):

| botão | valores → IDF1 do IoU ingênuo |
|---|---|
| nº de objetos | 3: 0,92 · 9: 0,88 · 15: 0,85 · 20: **0,79** |
| velocidade (px/q) | 0,5: 0,91 · 2: 0,85 · 3: 0,73 · 6: **0,45** |
| oclusão (quadros) | 0: 0,90 · 4: **0,82** · 24: 0,85 (sobrevivência à oclusão ≈ 0) |

A velocidade é o que mais derruba o baseline: com caixas de ~10 px, a 3 px/q o IoU entre quadros
consecutivos já fica perto do limiar. Na oclusão o IDF1 cai no primeiro degrau e depois satura,
porque com k=3 toda oclusão maior que 3 quadros já termina em ID novo (sobrevivência ~0 em
qualquer duração). Oclusões mais longas não pioram o que já está perdido, e o poste mais largo
também reduz o número de quadros visíveis avaliados.

## Parte 1 — baseline por quadro

**Detecções.** No sintético a fonte é o simulador (mAP@0,5 ≈ 0,86 no teste). **[MOT17]** Fonte
padrão: **FRCNN**. É da mesma família do detector do torchvision (Faster R-CNN), o que permite
comparar as duas fontes sem trocar de arquitetura. Os scores ficam em [0,1], então um único
`det_thr` vale para tudo (os do DPM não são calibrados). E evita o SDP, que é o melhor dos três
e esconderia parte do problema de associação. mAP@0,5 médio nas 7 sequências de treino do MOT17
(`results/mot17/part1/detector_map.json`): **DPM 0,415 · FRCNN 0,542 · torchvision 0,647 · SDP 0,652**. O torchvision
(Faster R-CNN v2 COCO, classe person, NMS próprio; `python -m pa2.detect`) fica quase empatado com o
SDP e bem acima do FRCNN público. Comparação de rastreamento com ele na seção do MOT17.

**Associação ingênua.** IoU entre a última caixa observada da track e a detecção, limiar fixo,
ID novo quando nada casa, morte após k quadros. Varredura na **validação** (sementes 2000+,
`part1/fig_assoc_sweep.png`): matcher {guloso, Hungarian} × IoU {0,1; 0,2; 0,3; 0,5} ×
k {1, 3, 5, 10, 30}. Melhor: **guloso, IoU ≥ 0,1, k = 3** (IDF1 0,745). O guloso empata ou vence o
Hungarian em 14 das 20 combinações, inclusive na melhor (0,745 vs 0,722 com os mesmos limiares). O Hungarian maximiza o número de pares e
aceita casamentos de IoU baixo que o guloso deixaria livres. Um k grande piora o IoU ingênuo,
porque a última caixa fica velha e passa a casar com o objeto errado. Regras completas de
nascimento e morte no README.

**Descolamento** (`part1/fig_decoupling.png`, 10 vídeos de dificuldade crescente: +objetos,
+velocidade, +oclusão). O **mAP fica praticamente plano (0,92 → 0,82)**, enquanto o **IDF1 cai de
0,96 para 0,48**. A razão #ids previstas/#ids verdadeiras sobe de 1,00 para 2,05 e os ID
switches por identidade de 0 para 1,8. O detector continua bom e o que quebra é a identidade.

## Parte 2 — Trilha A: RNN como modelo de movimento

**O que a recorrência carrega.** Um estado GRU (h=61, ~13k parâmetros na célula) por track.
Entrada por quadro: `[Δcx/h, Δcy/h, Δlog w, Δlog h]` em relação à entrada anterior (invariante a
translação e à escala do objeto), flag *observado*, score, `cy/H`, `log(h/H)`. Saída: μ e
log σ² do deslocamento até o próximo quadro. Sob oclusão a track é alimentada com a **própria
previsão** e observado=0, e o estado roda para frente sem observação.

**Perda.** Smooth-L1 sobre μ + NLL gaussiana para log σ² (com μ destacado), treinada em janelas
de trajetórias da **gt** com ruído de detector, buracos simulados de até 12 quadros e *scheduled
sampling* (prob. de alimentar a própria previsão 0 → 0,3). BPTT truncado T=16.

**Decodificação.** IoU(caixa prevista, detecção) com limiar 0,2 + **portão adaptativo**:
Mahalanobis² ≤ χ²₄(0,99) com a variância prevista, que cresce durante a oclusão. Matching guloso
em dois estágios, morte após 30 quadros. A mesma gestão é usada no Kalman, que só serve de
baseline.

**Teste (30 vídeos nunca vistos, sementes 0–29)** — `part2/table.md`, `part2/fig_bars.png`:

| rastreador | IDF1 | IDSW | IDSW/id verd. | #ids prev./verd. | erro de contagem | oclusões com id preservada | mAP |
|---|---|---|---|---|---|---|---|
| IoU ingênuo | 0,749 | 222 | 0,71 | 1,51 | 50% | 3,5% | 0,859 |
| Kalman v. const. | 0,813 | 131 | 0,42 | 1,29 | 30% | 70,8% | 0,859 |
| **MotionRNN** | **0,886** | **41** | **0,13** | **1,11** | **12%** | **91,0%** | 0,859 |

**Em que aspecto melhora o fracasso da Parte 1.** O baseline quebra por dois motivos: (1) a
caixa de referência é a do quadro anterior, então com movimento rápido o IoU não fecha;
(2) sob oclusão a referência fica parada e a track morre ou casa errado. A RNN ataca os dois:
prevê *onde* o objeto vai estar, e sob oclusão continua extrapolando, com incerteza crescente
que abre o portão de associação. Veja `part2/fig_occlusion_example.png` (previsão atravessando o
poste) e `part2/fig_sweep_all.png`. A 4 px/q o IDF1 é IoU 0,72 / Kalman 0,64 / **RNN 0,89**.
A 6 px/q, 0,45 / 0,45 / **0,71**. O Kalman de velocidade constante sofre com as reflexões nas
bordas e com a aceleração aleatória. A RNN aprende a dinâmica real, inclusive as quicadas.

**Pergunta: inferência em janelas de T quadros.** No PA1 a fronteira entre tiles cortava
*objetos*. Aqui a fronteira entre janelas corta *identidades*. Uma track viva no fim da janela j
renasce com outro id na janela j+1. Isso infla a contagem de identidades únicas e gera um ID
switch por track ativa em cada fronteira. Se o estado for zerado, perde-se também a velocidade
(o modelo precisa de alguns quadros de aquecimento) e a memória das tracks ocluídas, e uma
oclusão que atravessa a fronteira vira, na prática, uma track nova. Na nossa representação o
estado de uma track é pequeno e explícito (vetor GRU de 61 floats + última caixa), então dá para:
(a) **passar o estado adiante**, como se faz com BPTT truncado: a inferência é causal, só a
memória da *janela de dados* é limitada, e o custo é O(#tracks vivas), não O(T); (b) com janelas
**sobrepostas**, costurar as identidades por IoU / Mahalanobis entre as caixas que as duas
janelas produzem nos quadros em comum (Hungarian entre tracks); (c) para tracks ocluídas na
fronteira, usar a previsão da RNN com a variância dela como portão para casar com tracks que
nascem no início da próxima janela. É o mesmo mecanismo da reidentificação após oclusão.

## Parte 3 — Ablação, Eixo 1: a célula recorrente

RNN simples (h=109) vs. LSTM (h=52) vs. GRU (h=61), todas com ~13k parâmetros na célula,
T ∈ {4, 8, 16, 32}, 3 sementes cada, avaliadas na validação. `part3/table.md`, `part3/fig_ablation.png`:

| célula | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|
| RNN  | 0,803 ± 0,003 | 0,859 ± 0,010 | 0,876 ± 0,002 | 0,883 ± 0,004 |
| LSTM | 0,807 ± 0,012 | 0,849 ± 0,011 | 0,879 ± 0,004 | 0,884 ± 0,002 |
| GRU  | 0,807 ± 0,005 | **0,867 ± 0,010** | **0,884 ± 0,003** | **0,885 ± 0,002** |

(IDF1, média ± desvio. A sobrevivência à oclusão vai de ~0,2–0,3 em T=4 para ~0,85 em T=32
nas três células.)

**Onde a RNN simples quebra, e isso bate com a aula?** A curva analítica (Parte 4) mostra o
gradiente que some exatamente como na aula. Na RNN simples, ‖∂L_t/∂h_{t−k}‖ cai **44× em 30
passos**. Na GRU cai só **5×**. Mas no IDF1 a diferença é pequena (RNN ≤ GRU em todo T, por
0,5–1 ponto, perto do desvio), e quem domina é **T**. A história é coerente quando se olha *o
que* a tarefa precisa lembrar: estimar velocidade exige poucos quadros, e a RNN simples tem
memória de sobra para isso. O que ela não teria é memória para atravessar oclusões longas, mas
com T=4 nem a GRU recebe supervisão que atravesse um buraco de 8 quadros. O gargalo é a janela
de BPTT (de onde vem o sinal), não a célula. A perda de treino cai com T porque, em janelas
curtas, a fração de passos de "aquecimento" (sem velocidade estimada) é maior, então ela não é
comparável entre colunas, só dentro delas.

## Parte 4 — galeria de falhas e horizonte de memória

**Galeria** (`part4/fig_failure_{1,2,3}.png`, suíte com oclusões de 4 a 32 quadros). Tira de
quadros com gt (sólido) e predição (tracejado) coloridos por identidade, caixa prevista pela
recorrência para a track perdida (vermelho pontilhado) e elipse de 2σ da incerteza prevista.
Dos 48 ID switches dessa suíte, **39 vêm de oclusão longa** e 9 de fragmentação curta em
aglomerados.

1. *Oclusão longa (gt 5, id 3 → 16).* O objeto fica ocluído por **49 quadros** (53 sem
   casamento). A minha janela de BPTT é 16, os buracos de treino têm ≤ 12 quadros e
   `max_age` = 30. A track morre no quadro 30 do buraco, então o objeto renasce com outro id.
   A elipse de incerteza cresce de forma visível até lá, porque o modelo "sabe" que está
   extrapolando, mas nunca recebeu sinal de supervisão que atravessasse um buraco desse tamanho.
2. *Fragmentação curta em aglomerado (gt 3, id 1 → 11).* Elipses vizinhas se sobrepõem, a
   detecção some por 3 quadros (visibilidade entre 0,2 e 0,35: o objeto é avaliado mas o
   detector não o vê). A previsão da RNN desliza para o vizinho e, quando a detecção volta, uma
   tentativa nova vence. A falha é de associação/aparência, não de memória: geometria sozinha
   não separa dois objetos colados.
3. *Oclusão longa (gt 7, id 6 → 3).* **44 quadros** ocluído. Mesmo mecanismo de (1), e aqui o
   objeto reaparece perto de uma track existente e é capturado por ela: um switch "roubado".

**Horizonte analítico** (`part4/fig_grad_horizon.png`, 256 janelas de 48 passos, perda só no
último passo). Queda de ‖∂L_t/∂h_{t−k}‖:

| modelo | k=8 | k=16 | k=30 |
|---|---|---|---|
| RNN simples (T=32) | 2,6× | 7,8× | **44×** |
| LSTM — h (T=32) | 8,0× | 17,5× | 62× |
| LSTM — célula c | (ver figura: decai bem mais devagar que h) | | |
| GRU (T=32) | 1,2× | 2,2× | **5×** |
| GRU final (T=16) | 1,1× | 1,7× | 2,8× |

Na LSTM, o gradiente em **h** cai rápido, mas a memória de longo prazo dela está em **c**
(curva pontilhada), que decai bem mais devagar. A comparação "simples vs. com portas" tem que
olhar o caminho certo. A GRU, que só tem h, mostra a porta de atualização fazendo o papel de
atalho do gradiente.

**Horizonte empírico** (`part4/fig_empirical_horizon.png`). P(identidade sobrevive | duração da
oclusão), comparada com o histograma de durações do dataset: ≈ 0,85–0,94 até 23 quadros, **0,54
em 24–31 e 0 a partir de 32**. Cerca de 18% das oclusões do dataset estão na faixa de ≥ 32
quadros, em que o modelo final perde a identidade sempre.

**Correção.** Diagnóstico 1: "oclusão > `max_age` e > buracos vistos no treino". Mudança:
retreinar com **T=48 e buracos de até 40 quadros** e usar `max_age` = 60. Para separar os dois
efeitos, um controle muda só `max_age`:

| | IDF1 (suíte de oclusões longas) | sobrev. 24–31 q | sobrev. 32–47 q | IDF1 teste padrão | IDSW teste |
|---|---|---|---|---|---|
| final (T=16, max_age 30) | 0,872 | 0,54 | 0,00 | 0,886 | 41 |
| só max_age=60 | 0,888 | 0,79 | 0,72 | 0,887 | 39 |
| **corrigido** (T=48, buracos ≤ 40, max_age 60) | **0,894** | **0,83** | **0,83** | **0,892** | **32** |

O diagnóstico estava certo nas duas metades. A maior parte do ganho vem de não matar a track,
e o retreino com supervisão que atravessa buracos longos acrescenta mais +11 pontos de
sobrevivência em 32–47 quadros e −7 switches no teste padrão, sem piorar o caso comum.

## Parte 5 — estresse: qualidade do detector (sem retreinar)

`part5/table.md`, `part5/fig_stress.png`:

| degradação | drop | ruído | FP/q | mAP | IDF1 IoU | IDF1 Kalman | IDF1 RNN | IDSW IoU / Kalman / RNN |
|---|---|---|---|---|---|---|---|---|
| original | 0 | 0 | 0 | 0,859 | 0,749 | 0,813 | **0,886** | 222 / 131 / 41 |
| leve | 0,10 | 0,03 | 0,3 | 0,754 | 0,687 | 0,728 | **0,820** | 280 / 197 / 75 |
| média | 0,25 | 0,06 | 1,0 | 0,571 | 0,559 | 0,582 | **0,666** | 417 / 347 / 199 |
| forte | 0,40 | 0,10 | 2,0 | 0,365 | 0,375 | **0,425** | 0,419 | 571 / 502 / 543 |

**Absorve ou amplifica?** Em degradação leve e média a RNN **absorve**. O IDF1 relativo cai
menos que o mAP relativo (média: mAP cai 34%, IDF1 da RNN 25%): ela atravessa detecções
perdidas como se fossem oclusões curtas, e o portão de incerteza rejeita FPs isolados. Na
degradação forte ela passa a **amplificar** (IDSW 41 → 543, ×13, contra ×2,6 no IoU) e perde
para o Kalman. O ruído de caixa de 10% está fora da distribuição de treino (4%), então as
entradas Δ ficam ruidosas, a velocidade estimada fica errada e a extrapolação longa que ajudava
passa a puxar a track para FPs. O Kalman, com ruído de processo explícito, degrada com mais
suavidade. Treinar com ruído de entrada variável (ou passar o score como confiança da
observação, que já entra na entrada) é a correção óbvia.

## MOT17 — resultados reais

Split por sequência: treino `02, 04, 05, 11` · validação `09` · teste `10, 13`. Detecções públicas
**FRCNN**. A MotionRNN treina só com as trajetórias da gt (~15 s em CPU). Tudo gerado por
`bash run_all.sh mot17 data/MOT17`; figuras em `results/mot17/partN/`.

**Parte 1 — baseline e descolamento** (`part1/fig_decoupling.png`, `part1/fig_assoc_sweep.png`).
Varredura na validação (MOT17-09): o guloso empata ou vence o Hungarian em **16 de 20**
combinações. Melhor: guloso, IoU ≥ 0,5, k = 3 (IDF1 0,551). Com IoU ≥ 0,5 os dois matchers
coincidem; com limiares baixos o Hungarian piora mais, de novo por forçar pares ruins.
O descolamento existe, mas é mais fraco que no sintético, porque a densidade não é o único fator
(câmera parada ou em movimento, altura da câmera):

| sequência | caixas gt/quadro | mAP | IDF1 (IoU ingênuo) | #ids prev./verd. |
|---|---|---|---|---|
| MOT17-05 | 8,3 | 0,533 | 0,531 | 1,01 |
| MOT17-09 | 10,1 | 0,568 | 0,551 | 1,92 |
| MOT17-11 | 10,5 | 0,606 | 0,545 | 1,37 |
| MOT17-13 | 15,5 | 0,582 | 0,398 | 2,85 |
| MOT17-10 | 19,6 | 0,598 | 0,421 | 4,28 |
| MOT17-02 | 31,0 | 0,352 | 0,363 | 1,79 |
| MOT17-04 | 45,3 | 0,557 | 0,568 | 1,66 |

De MOT17-05 a MOT17-10 o mAP *sobe* (0,53 → 0,60) enquanto o IDF1 cai (0,53 → 0,42) e a razão de
identidades vai de 1,01 a 4,28. MOT17-04 é a mais densa, mas a câmera é fixa e alta e as pessoas
andam devagar, então a associação ingênua funciona bem.

**Parte 2 — teste (MOT17-10 + MOT17-13)** (`part2/table.md`):

| rastreador | IDF1 | IDSW | Frag | #ids prev./verd. | oclusões com id preservada | mAP |
|---|---|---|---|---|---|---|
| IoU ingênuo | 0,410 | 591 | 563 | 3,34 | 17% | 0,590 |
| Kalman v. const. | 0,493 | 397 | 640 | 0,86 | 51% | 0,590 |
| **MotionRNN** | **0,500** | **383** | 598 | 1,63 | 40% | 0,590 |

A RNN fica praticamente empatada com o Kalman (+0,7 ponto de IDF1, −14 switches), longe da folga do
sintético (+7 pontos, 131 → 41 switches). **Por quê:** no sintético a vantagem vinha de movimento não
linear (quicadas nas bordas, aceleração aleatória). Pedestres andam quase em linha reta e com
velocidade quase constante, que é exatamente o modelo do Kalman. Sobra pouca dinâmica para a RNN
aprender. O Kalman também preserva mais oclusões e acerta melhor a contagem. A RNN fragmenta mais
(1,63 ids por pessoa), mas a amostra de oclusões é pequena (~45 eventos em 2 sequências).
**Trocando o detector pelo torchvision** (`python -m pa2.evaluate --data mot17 --split test
--ckpt checkpoints/mot17_gru.pt --det-file det_tv.txt`, saída em `part2/torchvision.json`):

| rastreador | IDF1 FRCNN → torchvision | IDSW FRCNN → torchvision |
|---|---|---|
| IoU ingênuo | 0,410 → **0,372** | 591 → 880 |
| Kalman | 0,493 → 0,494 | 397 → 513 |
| MotionRNN | 0,500 → **0,517** | 383 → **350** |

O mAP sobe de 0,590 para 0,665, e mesmo assim o IoU ingênuo *piora*: o torchvision gera quase o
dobro de caixas (17,7 mil contra 9,7 mil em MOT17-10), então há mais candidatas perto de cada track
e a associação por IoU com a última caixa se confunde mais (5,9 ids previstos por pessoa). É o descolamento da Parte 1 na direção
oposta: um detector melhor não conserta a identidade. A RNN é a única que aproveita o detector
melhor, e a vantagem dela sobre o Kalman cresce (350 vs 513 switches), embora ela também
fragmente mais (2,37 ids por pessoa). Ressalva: os limiares de
score (`det_thr` 0,5, `new_thr` 0,6) foram escolhidos com o FRCNN e não foram reajustados.

Inferência de demonstração em MOT17-10 (`inferencia.ipynb` → `results/inferencia_mot17.mp4`):
131 identidades previstas para 57 verdadeiras, IDF1 0,464.

**Parte 3 — ablação** (`part3/table.md`, validação = MOT17-09):

| célula | T=4 | T=8 | T=16 | T=32 |
|---|---|---|---|---|
| RNN  | 0,557 ± 0,011 | 0,545 ± 0,041 | 0,574 ± 0,028 | 0,511 ± 0,028 |
| LSTM | 0,567 ± 0,001 | 0,552 ± 0,008 | 0,543 ± 0,010 | 0,561 ± 0,034 |
| GRU  | 0,537 ± 0,024 | **0,594 ± 0,010** | 0,570 ± 0,023 | 0,590 ± 0,008 |

Com uma única sequência de validação (26 identidades) o IDF1 é ruidoso e não ordena células nem
janelas. O sinal que se repete é a **sobrevivência à oclusão, que sobe com T** nas três células
(GRU: 0,17 em T=4 → 0,65 em T≥8; RNN: 0,38 → 0,69; LSTM: 0,40 → 0,71). Como no sintético, a janela
de BPTT importa mais que a célula. A norma do gradiente do modelo final cai 1,8× em 4 passos, 3,7×
em 16 e 8,1× em 32 (`part4/fig_grad_horizon.png`).

**Parte 4 — galeria e correção** (`part4/fig_failure_{1,2,3}.png`). Os 383 switches do teste se
dividem em **150 trocas entre vizinhos, 118 fragmentações curtas e 115 oclusões longas**. No
sintético, 81% eram oclusão longa.

1. *Oclusão longa (MOT17-10, gt 25, id 79 → 247).* Buraco de 404 quadros sem casamento (211
   ocluído): nenhum `max_age` razoável atravessa isso.
2. *Troca entre vizinhos (MOT17-10, gt 39, id 45 → 47).* Pessoas lado a lado; a detecção some por
   8 quadros **sem oclusão**, a previsão desliza e outra track captura a detecção que volta.
3. *Fragmentação curta (MOT17-10, gt 44, id 123 → 148).* Buraco de 9 quadros (6 ocluído); a
   tentativa nova vence a track antiga.

| | IDF1 teste | IDSW | #ids prev./verd. | oclusões com id |
|---|---|---|---|---|
| final (T=16, max_age 30) | 0,500 | 383 | 1,63 | 40% |
| só max_age = 60 | 0,509 | 394 | 1,57 | 41% |
| corrigido (T=48, buracos ≤ 40, max_age 60) | 0,503 | 444 | **1,17** | **51%** |

A correção ataca a falha certa *do sintético*: a sobrevivência à oclusão sobe 11 pontos e a
contagem melhora muito (1,63 → 1,17 ids por pessoa). Mas o IDF1 não se mexe e os switches sobem,
porque tracks que vivem mais tempo em multidão capturam o vizinho. No MOT17 o gargalo é
**aparência**: geometria sozinha não separa duas pessoas coladas.

**Parte 5 — estresse** (`part5/table.md`):

| degradação | mAP | IDF1 IoU | IDF1 Kalman | IDF1 RNN | IDSW IoU / Kalman / RNN |
|---|---|---|---|---|---|
| original | 0,590 | 0,410 | 0,493 | **0,500** | 591 / 397 / 383 |
| leve | 0,522 | 0,335 | 0,438 | **0,456** | 674 / 405 / 391 |
| média | 0,417 | 0,202 | 0,353 | **0,372** | 725 / 437 / 412 |
| forte | 0,289 | 0,090 | **0,255** | 0,241 | 551 / 482 / 446 |

Mesmo padrão do sintético: a RNN absorve a degradação leve e a média e perde para o Kalman em IDF1
na forte. O IoU ingênuo desaba (0,41 → 0,09), porque detecções perdidas viram ids novos.

## Pergunta extra (Parte 5, alternativa não escolhida): queda de taxa de quadros

Um modelo de movimento aprendido com Δt fixo aprende "deslocamento por passo". Subamostrando a
1/2 ou 1/5, o deslocamento real por passo é 2× ou 5× maior. A previsão fica para trás, o IoU com
a detecção cai e o portão (calibrado na variância de 1 passo) fica estreito demais. Alimentar Δt
na recorrência ajuda só se o treino vir vários Δt: com Δt sempre igual a 1 a rede aprende a
ignorar a entrada. Treinando com subamostragem aleatória, Δt vira um condicionador legítimo e o
modelo pode escalar o deslocamento. A alternativa é integrar velocidade × Δt fora da rede.
