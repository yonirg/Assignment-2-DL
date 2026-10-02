# PA2 — Identidade ao longo do tempo: detecção, recorrência e rastreamento

Rastreamento multiobjeto *identity-aware* sem rastreador pronto: detector congelado + **Trilha A
(RNN como modelo de movimento)**, um estado recorrente por track. As métricas (IDF1, ID switches,
fragmentações, AP), o NMS, a associação e a gestão de tracks foram todos escritos do zero.

* Roteiro da apresentação, com os resultados e as respostas às perguntas: **[APRESENTACAO.md](APRESENTACAO.md)**
* Slides (com notas do apresentador): **`PA2_apresentacao.pptx`** · versão curta (~15 min): **`PA2_apresentacao_curta.pptx`**
* Vídeo com as previsões da RNN e a incerteza: `python tools/video_previsao.py --seq <sequência> --out <saída.mp4>`
* Uso de IA: **[AI_LOG.md](AI_LOG.md)**

## Ambiente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python >= 3.10; CPU é suficiente
python -m pytest tests -q                # testes da métrica (Parte 0.3)
```

## Dados

**Sintético** — gerado na hora (`pa2/synth.py`), nada a baixar. Uma sequência de demonstração em
formato MOT já está em `data/synth_demo/SYNTH-DEMO`.

**MOT17** — <https://motchallenge.net/data/MOT17/> (sem conta):

```bash
mkdir -p data && cd data
wget https://motchallenge.net/data/MOT17Labels.zip && unzip MOT17Labels.zip -d MOT17   # ~10 MB: gt + det
wget https://motchallenge.net/data/MOT17.zip       && unzip MOT17.zip                  # ~5,5 GB: imagens (opcional)
```

Estrutura esperada: `data/MOT17/train/MOT17-02-FRCNN/{seqinfo.ini, gt/gt.txt, det/det.txt, img1/}`.
Treino, Partes 1–3 e 5 rodam **só com o pacote de anotações**: a Trilha A treina em trajetórias da
gt e o rastreador consome `det.txt`. As imagens só são necessárias para o detector do torchvision,
a galeria de falhas (Parte 4) e o vídeo do notebook.

Split (por sequência, `pa2/data.py`): treino `02, 04, 05, 11` · validação `09` · teste `10, 13`.
Fonte de detecção padrão: **FRCNN** (justificativa em APRESENTACAO.md).

## Um comando que treina / um comando que avalia

```bash
# treina a MotionRNN (GRU, T=16, scheduled sampling + oclusões simuladas)
python -m pa2.train --data synth --out checkpoints/synth_gru.pt
python -m pa2.train --data mot17 --mot-root data/MOT17 --out checkpoints/mot17_gru.pt

# avalia IoU ingênuo vs Kalman vs MotionRNN no teste
python -m pa2.evaluate --data synth --ckpt checkpoints/synth_gru.pt
python -m pa2.evaluate --data mot17 --mot-root data/MOT17 --split test --ckpt checkpoints/mot17_gru.pt
```

Tudo de uma vez (todas as figuras/tabelas em `results/<data>/partN/`; ~1 h no sintético, ~5 min no
MOT17, ambos em CPU):

```bash
bash run_all.sh synth
bash run_all.sh mot17 data/MOT17
```

Detecções do torchvision (Faster R-CNN v2 COCO, classe person, NMS próprio; usa CUDA ou a GPU do
Mac via MPS se houver, senão CPU):
`python -m pa2.detect --seq data/MOT17/train/MOT17-02-FRCNN` → `det/det_tv.txt` (a Parte 1 compara
automaticamente se o arquivo existir).
Para rastrear com essas detecções: `python -m pa2.evaluate --data mot17 --split test --ckpt
checkpoints/mot17_gru.pt --det-file det_tv.txt`.

Inferência em qualquer sequência: abra **`inferencia.ipynb`**, troque `SEQ_PATH` e rode. Sai um
`.mp4` com as identidades coloridas de forma consistente e a contagem de objetos únicos. O notebook
está executado em MOT17-10 (`results/inferencia_mot17.mp4`); a versão sintética, que não precisa de
download, está em `results/inferencia.mp4`.

## Checkpoints

| arquivo | o que é |
|---|---|
| `checkpoints/synth_gru.pt` | modelo final da Trilha A no sintético (GRU h=61, T=16) |
| `checkpoints/synth_gru_fix.pt` | modelo após a correção da Parte 4 (T=48, buracos até 40) |
| `checkpoints/ablation/synth/*.pt` | 36 modelos da ablação (3 células × 4 janelas × 3 sementes) |
| `checkpoints/mot17_gru.pt` | modelo final da Trilha A no MOT17 (GRU h=61, T=16; ~15 s em CPU) |
| `checkpoints/mot17_gru_fix.pt` | correção da Parte 4 no MOT17 (T=48, buracos até 40) |
| `checkpoints/ablation/mot17/*.pt` | 36 modelos da ablação no MOT17 |

## Estrutura

```
metrics.py              IDF1, IDSW, fragmentações, MOTA, AP, NMS, sobrevivência à oclusão (próprios)
pa2/synth.py            gerador de elipses com z-buffer (oclusão real) — Parte 0.1
pa2/detsim.py           simulador / degradador de detector — Partes 0.2 e 5
pa2/data.py             MOT17 + sintético num formato único, split por sequência
pa2/models.py           MotionRNN (RNN/LSTM/GRU) — Trilha A
pa2/trackers.py         associação + nascimento/morte; IoU, Kalman (baseline), RNN
pa2/train.py            treino (BPTT truncado, teacher forcing / scheduled / free-running)
pa2/evaluate.py         roda rastreadores e agrega métricas
pa2/detect.py           detector do torchvision com NMS próprio
experiments/partN_*.py  um script por parte do enunciado
tests/test_metrics.py   casos (a), (b), (c) da Parte 0.3
inferencia.ipynb        notebook de inferência
```

## Regra de associação e gestão de tracks (`pa2/trackers.py`)

1. Descarta detecções com score < `det_thr` (0,5).
2. Cada track tem uma caixa prevista para o quadro: IoU ingênuo = última caixa observada;
   Kalman = predição de velocidade constante; RNN = saída da recorrência.
3. Custo `1 − IoU`, pares com IoU < `iou_thr` proibidos — exceto (Kalman/RNN) se a distância de
   Mahalanobis² da detecção à previsão, com a variância prevista, for ≤ χ²₄(0,99) = 13,28
   (portão adaptativo: a incerteza cresce sob oclusão).
4. Atribuição **gulosa** por menor custo (venceu o Hungarian na validação, Parte 1) em dois
   estágios: tracks confirmadas primeiro, depois tentativas com as detecções restantes.
5. **Nascimento**: detecção não casada com score ≥ `new_thr` (0,6) abre track tentativa;
   confirmada após `min_hits` observações (os quadros tentativos são emitidos retroativamente).
6. **Morte**: tentativa morre na 1ª falha; confirmada após `max_age` quadros sem observação.
   Enquanto viva sem observação, a RNN roda para frente alimentada pela própria previsão.
7. Saída = caixa da detecção casada (detector congelado).

| parâmetro | IoU ingênuo | Kalman / RNN | escolhido por |
|---|---|---|---|
| matcher | guloso | guloso | validação |
| `iou_thr` | 0,1 (sint.) | 0,2 (sint.) / 0,3 (MOT17) | validação |
| `max_age` | 3 (sint.) | 30 | validação |
| `min_hits` | 2 (sint.) / 3 (MOT17) | idem | — |

## Regras do enunciado

* Nenhum rastreador pronto, nenhuma métrica pronta (`motmetrics`/`TrackEval` não são usados),
  NMS próprio (`metrics.nms`; no detector do torchvision o NMS do ROI head é desligado e o nosso
  aplicado por cima).
* Kalman só aparece como baseline de comparação; quem carrega o estado na Parte 2 é a GRU.
