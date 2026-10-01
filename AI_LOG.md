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
   o código para que o MOT17 rode com o pacote de anotações (~10 MB). Todos os números deste
   repositório são do **sintético**; os scripts do MOT17 foram testados com uma árvore falsa no
   formato MOT17 (sequências sintéticas exportadas com os nomes `MOT17-XX-{DPM,FRCNN,SDP}`) e
   precisam ser rodados com os dados reais (`bash run_all.sh mot17 data/MOT17`).

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

## O que a dupla precisa fazer / verificar

- [ ] Rodar `bash run_all.sh mot17 data/MOT17` com os dados reais e atualizar APRESENTACAO.md.
- [ ] (Opcional) gerar `det_tv.txt` com `python -m pa2.detect` para comparar o torchvision.
- [ ] Ler `metrics.py` e `pa2/trackers.py` até conseguirem explicar cada linha na apresentação.
- [ ] Escrever aqui como vocês revisaram o código e o que mudaram.

## Relato da dupla

_(preencher)_
