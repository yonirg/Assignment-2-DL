# PA2 — roteiro da apresentação

> **Importante.** Todos os números abaixo são do **ambiente sintético** (Parte 0), gerados neste
> repositório. O MOT17 não pôde ser baixado no ambiente em que o código foi escrito (o proxy bloqueava
> `motchallenge.net`). Todo o pipeline do MOT17 está implementado e foi testado de ponta a ponta com
> uma árvore falsa no formato MOT17. Para gerar as mesmas figuras e tabelas no MOT17 real:
> `bash run_all.sh mot17 data/MOT17` (só o pacote de anotações de ~10 MB é obrigatório). As seções
> marcadas com **[MOT17]** precisam desses números.

Reprodução: `bash run_all.sh synth`. Cada figura citada está em `results/synth/partN/`.

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
e esconderia parte do problema de associação. `part1_baseline.py --data mot17` grava a tabela
de mAP das 4 fontes (`detector_map.json`). Para o torchvision: `python -m pa2.detect --seq ...`.

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

## Pergunta extra (Parte 5, alternativa não escolhida): queda de taxa de quadros

Um modelo de movimento aprendido com Δt fixo aprende "deslocamento por passo". Subamostrando a
1/2 ou 1/5, o deslocamento real por passo é 2× ou 5× maior. A previsão fica para trás, o IoU com
a detecção cai e o portão (calibrado na variância de 1 passo) fica estreito demais. Alimentar Δt
na recorrência ajuda só se o treino vir vários Δt: com Δt sempre igual a 1 a rede aprende a
ignorar a entrada. Treinando com subamostragem aleatória, Δt vira um condicionador legítimo e o
modelo pode escalar o deslocamento. A alternativa é integrar velocidade × Δt fora da rede.
