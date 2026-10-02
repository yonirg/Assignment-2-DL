# AI_LOG.md — como usei IA neste assignment

**Resumo honesto:** as primeiras versões do código deste repositório (pacote `pa2/`, `metrics.py`,
scripts de cada parte, testes, notebook, README) foram geradas com o Claude Code (Anthropic) a partir
do PDF do PA, e depois rodadas e depuradas em conversa. A primeira rodada completa de experimentos foi
feita numa sessão em nuvem sem acesso ao MOT17, então ficou só no sintético. Depois, com o Claude Code
rodando no meu Mac (Apple M4 Pro), rodei todas as Partes 1–5 no MOT17 real, o detector do torchvision
na GPU (MPS) e a galeria de falhas com as imagens reais, e conferi onde os resultados reais
contradiziam o sintético (episódios 8–12).

## Episódios

1. **Escolha da trilha (Trilha A).** O ambiente em que o código foi escrito bloqueava o
   `motchallenge.net` e os pesos COCO do torchvision. A Trilha A (RNN como modelo de movimento) treina
   só com as trajetórias da gt e roda só com o `det.txt`, então dava para escrever e testar o pipeline
   inteiro do MOT17 com o pacote de anotações de ~10 MB. Ele foi testado primeiro numa árvore falsa no
   formato MOT17 (sequências sintéticas com os nomes `MOT17-XX-{DPM,FRCNN,SDP}`).

2. **Teste (c) da métrica (Parte 0).** A primeira versão comparava "troca no quadro k" com "track
   partida no quadro k" em cenas diferentes, e as duas davam IDF1 = 0,5, ou seja, não mostrava a
   diferença que o enunciado pede. O teste foi reescrito para usar a **mesma cena** de dois objetos: a
   troca estraga duas identidades (2 switches, IDF1 0,5, contagem correta); a partição estraga uma
   (1 switch, 1 fragmentação, IDF1 0,769, uma identidade a mais na contagem).

3. **Bug no gerador sintético.** O ruído da imagem consumia o mesmo gerador aleatório da dinâmica,
   então a mesma semente produzia ground truths diferentes com e sem renderização. Corrigido com um
   gerador separado para o ruído; a gt agora independe do render.

4. **Bug na sobrevivência à oclusão.** A primeira versão exigia casamento exatamente no último quadro
   visível. Como o simulador só detecta objetos com visibilidade ≥ 0,35 e a oclusão era definida com
   0,2, quase todo evento ficava "não informativo" e a taxa dava 0 em tudo. Corrigido com uma janela de
   ±10 quadros antes e depois do buraco.

5. **Duplicatas roubando detecções.** O Kalman criava mais identidades que o IoU ingênuo: uma
   duplicata de detecção nascia como track e, no quadro seguinte, competia com a track verdadeira.
   Solução: associação em dois estágios (confirmadas primeiro, tentativas depois) e morte imediata das
   tentativas que falham.

6. **Hiperparâmetros só na validação.** Varredura de matcher × limiar de IoU × `max_age` em sementes
   de validação (2000+), nunca no teste. Surpresa: **guloso ≥ Hungarian** (14 de 20 combinações no
   sintético, 16 de 20 no MOT17). O Hungarian maximiza o número de pares e acaba aceitando casamentos
   de IoU baixo que o guloso deixaria livres.

7. **Diagnóstico e correção da Parte 4, com controle.** A galeria mostrou que, no sintético, a pior
   falha é oclusão mais longa que `max_age` e que os buracos vistos no treino (≤ 12 quadros). A
   correção (T=48, buracos até 40, `max_age` 60) foi comparada com um controle que só aumenta o
   `max_age`, para separar o efeito do retreino do efeito da regra de morte: sobrevivência em oclusões
   de 32–47 quadros 0% → 72% (controle) → 83% (correção).

8. **MOT17 real: a Parte 4 quebrou sem as imagens.** `bash run_all.sh mot17 data/MOT17` rodou as
   Partes 1–3 em ~5 min de CPU, mas a galeria de falhas lê `img1/`, que não vem no pacote de
   anotações. Baixei as imagens (5,9 GB) e rodei de novo a Parte 4 e, como o script aborta nesse
   ponto (`set -e`), também a Parte 5.

9. **O MOT17 contradiz parte do sintético.** A RNN quase empata com o Kalman (IDF1 0,500 vs 0,493;
   no sintético era 0,886 vs 0,813), e a correção da Parte 4 não move o IDF1 (0,500 → 0,503), embora
   melhore a sobrevivência à oclusão (40% → 51%) e a contagem (1,63 → 1,17 ids por pessoa). A galeria
   real explica: só 30% dos 383 switches vêm de oclusão longa; 39% são trocas entre pessoas lado a
   lado, que pedem aparência, não memória de movimento. Esses números entraram na apresentação como
   estão, sem reajustar nada para a RNN ganhar.

10. **Parâmetros da Parte 1 lidos de um arquivo que ainda não existia.** A regra ingênua lê os
    melhores parâmetros que a própria Parte 1 grava (`_part1_best`). Na primeira execução no MOT17 o
    arquivo não existia, então a figura de descolamento usou os padrões (IoU 0,3, k=5) em vez dos
    escolhidos na validação (IoU 0,5, k=3). Rodando a Parte 1 de novo, os números ficaram consistentes
    com as Partes 2–5. Quem rodar `run_all.sh` do zero deve rodar a Parte 1 duas vezes.

11. **Detector do torchvision na GPU do Mac.** O Python do python.org não achava os certificados SSL
    para baixar os pesos COCO; baixei com `curl` para `~/.cache/torch/hub/checkpoints/`. O
    `pa2/detect.py` passou a usar MPS quando não há CUDA (~0,5 s por quadro, ~35 min nas 7 sequências
    de treino), e o `pa2.evaluate` ganhou `--det-file` para rastrear com essas detecções. Resultado:
    o mAP sobe (0,590 → 0,665), mas o IoU ingênuo **piora** (IDF1 0,410 → 0,372), porque o detector
    gera quase o dobro de caixas. A RNN é a única que aproveita (0,517, 350 switches contra 513 do
    Kalman).

12. **Notebook e vídeo.** O `inferencia.ipynb` foi executado em MOT17-10. O vídeo em resolução cheia
    não caberia no repositório, então é reduzido para 960 px de largura (24 MB).

## O que é meu e o que é da IA

| | |
|---|---|
| Da IA (revisado por mim) | estrutura do pacote, versões iniciais de todos os módulos, métricas, scripts das partes, testes, notebook, README, primeira versão do roteiro e dos slides |
| Meu | levar o pipeline para o MOT17 real antes de entregar, em vez de ficar só no sintético; rodar o detector do torchvision; manter na apresentação os resultados do MOT17 que contradizem o sintético; interpretação dos resultados e apresentação |
| Extensões não feitas | pista de aparência (a falha dominante no MOT17 são trocas entre vizinhos); validação com mais de uma sequência (a ablação no MOT17 usa só a MOT17-09 e fica ruidosa); reajustar `det_thr`/`new_thr` para os scores do torchvision |

Ferramentas: Claude Code (terminal, numa sessão em nuvem e depois no meu Mac); nenhum rastreador
pronto, métrica de biblioteca ou código de repositório de terceiros foi usado.
