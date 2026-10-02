# AI_LOG — uso de IA neste assignment

> **Dupla: revisem e completem este arquivo com a experiência de vocês.** A política do curso
> pede que vocês entendam tudo o que entregam. O registro abaixo descreve, com honestidade, o que
> foi gerado com IA; a seção final é para vocês contarem como revisaram, testaram e o que mudaram.

## Ferramenta

Claude Code (agente de programação da Anthropic), numa sessão em nuvem com acesso ao repositório.
O enunciado (PA2.pdf) foi entregue ao agente com o pedido "resolva para mim".

## O que foi gerado com IA

Praticamente toda a primeira versão do código, dos experimentos e da documentação:
`metrics.py`, `pa2/*`, `experiments/*`, `tests/test_metrics.py`, `inferencia.ipynb`, README,
APRESENTACAO.md e todas as figuras/tabelas em `results/synth/`.

## Episódios

1. **Ambiente sem acesso ao MOT17.** O proxy da sessão bloqueava `motchallenge.net` (e também
   `download.pytorch.org`, então os pesos COCO do torchvision não puderam ser baixados). Decisão:
   escolher a **Trilha A**, que treina só em trajetórias da gt e roda só com `det.txt`, e montar
   o código para que o MOT17 rode com o pacote de anotações (~10 MB). Nessa sessão só
   existiam números do **sintético**; os scripts do MOT17 foram testados com uma árvore falsa no
   formato MOT17 (sequências sintéticas exportadas com os nomes `MOT17-XX-{DPM,FRCNN,SDP}`). Os
   dados reais foram rodados depois, na sessão local (episódio 8).

2. **Teste (c) da métrica.** A primeira versão dos testes comparava "troca no quadro k" com
   "track partida no quadro k" em cenas diferentes, e as duas davam IDF1 = 0,5: não mostrava a
   diferença pedida pelo enunciado. Reescrito para usar a **mesma cena** de dois objetos: a
   troca estraga duas identidades (2 switches, IDF1 0,5, contagem correta); a partição estraga uma
   (1 switch, IDF1 0,75, uma identidade a mais na contagem).

3. **Bug no gerador.** O ruído da imagem consumia o mesmo gerador aleatório da dinâmica, então a
   mesma semente produzia ground truths diferentes com e sem renderização. Corrigido com um gerador
   separado para o ruído (a gt agora independe do render).

4. **Bug na sobrevivência à oclusão.** A primeira versão exigia casamento exatamente no último
   quadro visível; como o simulador só detecta objetos com visibilidade >= 0,35 e a oclusão era
   definida com 0,2, quase todo evento ficava "não informativo" (taxa 0 em tudo). Corrigido com
   uma janela de ±10 quadros antes/depois do buraco.

5. **Duplicatas roubando detecções.** O Kalman criava mais identidades que o IoU ingênuo: uma
   duplicata de detecção nascia como track e, no quadro seguinte, competia com a track verdadeira.
   Solução: associação em dois estágios (confirmadas primeiro, tentativas depois) e morte imediata
   de tentativas.

6. **Escolha de hiperparâmetros na validação, não no teste.** Varredura de
   matcher/limiar/max_age em sementes de validação (2000+); surpresa: **guloso > Hungarian**
   (o Hungarian maximiza o número de pares e força casamentos de IoU baixo), e `max_age` é o
   fator dominante para Kalman e RNN.

7. **Diagnóstico da Parte 4.** A galeria mostrou que a pior falha é oclusão mais longa que
   `max_age` e que os buracos vistos no treino (≤ 12). A correção (T=48, buracos até 40,
   `max_age`=60) foi comparada com o controle "só aumentar max_age", para separar o efeito do
   retreino do efeito da regra de morte.

8. **Segunda sessão, local, com o MOT17 real (02/10).** Uma nova sessão do Claude Code, agora na
   máquina da dupla (Mac M4 Pro, CPU), clonou o repositório, baixou o MOT17 e rodou
   `bash run_all.sh mot17 data/MOT17` (~5 min). Ocorrências:
   - A Parte 4 quebrou na primeira execução: a galeria de falhas lê `img1/`, que não vem no pacote
     de anotações. As imagens foram baixadas (5,9 GB) e só a Parte 4 foi rodada de novo. O
     `run_all.sh` aborta nesse ponto (`set -e`), então a Parte 5 também foi rodada à parte.
   - **Os resultados no MOT17 contrariam parte da história do sintético.** A RNN quase empata com o
     Kalman (IDF1 0,500 vs 0,493), e a correção da Parte 4 não melhora o IDF1, porque no MOT17 só 30%
     dos switches vêm de oclusão longa (39% são trocas entre vizinhos). Isso foi registrado como
     resultado em APRESENTACAO.md, sem ajustar nada para "fazer a RNN ganhar".
   - O Python do python.org não achava os certificados SSL para baixar os pesos COCO; os pesos
     foram baixados com `curl` para `~/.cache/torch/hub/checkpoints/`. `pa2/detect.py` passou a
     usar a GPU do Mac (MPS) por padrão quando não há CUDA.
   - `inferencia.ipynb` foi executado em MOT17-10; o vídeo é reduzido para 960 px de largura para
     caber no repositório (24 MB).
   - **Pegadinha da Parte 1:** a regra ingênua lê os melhores parâmetros que a própria Parte 1
     grava (`_part1_best`). Na primeira execução do MOT17 o arquivo ainda não existia, então a
     figura de descolamento usou os padrões (IoU 0,3, k=5). A Parte 1 foi rodada de novo e agora
     usa os parâmetros da validação (IoU 0,5, k=3), como as Partes 2–5. Quem rodar `run_all.sh`
     do zero deve rodar a Parte 1 duas vezes.
   - O detector do torchvision rodou na GPU do Mac (MPS) nas 7 sequências de treino (~35 min).
     Foi adicionada a opção `--det-file` em `pa2.evaluate` para rastrear com essas detecções.
   - Os slides `PA2_apresentacao.pptx` foram gerados pela IA a partir de APRESENTACAO.md e das
     figuras em `results/`, com notas do apresentador.

## O que a dupla precisa fazer / verificar

- [x] Rodar `bash run_all.sh mot17 data/MOT17` com os dados reais e atualizar APRESENTACAO.md.
- [x] (Opcional) gerar `det_tv.txt` com `python -m pa2.detect` para comparar o torchvision.
- [ ] Ler `metrics.py` e `pa2/trackers.py` até conseguirem explicar cada linha na apresentação.
- [ ] Revisar os slides e as notas do apresentador; conferir os números com `results/`.
- [ ] Completar o relato abaixo com a revisão de vocês.

## Relato da dupla

**Como usamos a IA.** Entregamos o enunciado ao Claude Code numa sessão em nuvem, que escreveu o
código, os experimentos sintéticos e a documentação (episódios 1–7). Como aquela sessão não
conseguia baixar o MOT17, abrimos uma segunda sessão local, que rodou o pipeline no MOT17 real,
gerou o detector do torchvision, o notebook e os slides (episódio 8).

**O que aprendemos com os resultados.** O sintético sozinho teria nos levado a concluir que a RNN é
claramente melhor que o Kalman e que o problema principal é a oclusão longa. No MOT17 as duas
conclusões caem: pedestres andam quase em linha reta, então velocidade constante já basta, e a
maior parte dos erros vem de pessoas lado a lado, o que pede aparência e não memória de movimento.

**Nossa revisão** _(completar com o que vocês fizeram de fato)_:
- trechos de código que lemos linha a linha e conseguimos explicar:
- testes/verificações que rodamos por conta própria:
- o que mudamos, discordamos ou corrigimos no que a IA produziu:

