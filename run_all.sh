#!/usr/bin/env bash
# Reproduz todas as tabelas e figuras de uma fonte de dados.
#   bash run_all.sh synth                 # tudo no sintético (~1 h em 4 CPUs)
#   bash run_all.sh mot17 data/MOT17      # tudo no MOT17 (só precisa do pacote de anotações,
#                                         #  exceto vídeos/galeria/torchvision, que leem img1/)
set -euo pipefail
DATA=${1:-synth}
ROOT_MOT=${2:-data/MOT17}
EXTRA=()
if [ "$DATA" = "mot17" ]; then EXTRA=(--mot-root "$ROOT_MOT"); fi

python -m pytest tests -q
if [ "$DATA" = "synth" ]; then
  python -m pa2.train --data synth --out checkpoints/synth_gru.pt
  python experiments/part0_synthetic.py --trackers iou kalman rnn
else
  python -m pa2.train --data mot17 "${EXTRA[@]}" --out checkpoints/mot17_gru.pt
fi
python experiments/part1_baseline.py --data "$DATA" "${EXTRA[@]}"
python experiments/part2_compare.py  --data "$DATA" "${EXTRA[@]}"
python experiments/part3_ablation.py --data "$DATA" "${EXTRA[@]}" --jobs 4
python -m pa2.train --data "$DATA" "${EXTRA[@]}" --T 48 --max-occ 40 --out "checkpoints/${DATA}_gru_fix.pt"
python experiments/part4_memory.py   --data "$DATA" "${EXTRA[@]}" --fix-ckpt "checkpoints/${DATA}_gru_fix.pt"
python experiments/part5_stress.py   --data "$DATA" "${EXTRA[@]}"
