"""Parte 5 — teste de estresse: qualidade do detector (sem retreinar).

Degrada as detecções da fonte padrão em 3 intensidades (descarte p%, ruído nas
caixas proporcional ao tamanho, falsos positivos com score na faixa real) e
reporta mAP e IDF1 juntos, para IoU, Kalman e MotionRNN.

  python experiments/part5_stress.py --data synth
  python experiments/part5_stress.py --data mot17 --mot-root data/MOT17
"""
from __future__ import annotations

import os

import numpy as np

from common import COLORS, LABELS, base_parser, default_ckpt, out_dir, plt, save_json, test_sequences
from pa2.detsim import degrade_detections
from pa2.evaluate import aggregate, evaluate_seq, run_tracker, tracker_params
from pa2.models import MotionRNN

LEVELS = {
    "original": dict(drop_p=0.0, noise=0.0, fp_rate=0.0),
    "leve": dict(drop_p=0.10, noise=0.03, fp_rate=0.3),
    "média": dict(drop_p=0.25, noise=0.06, fp_rate=1.0),
    "forte": dict(drop_p=0.40, noise=0.10, fp_rate=2.0),
}
TRACKERS = ["iou", "kalman", "rnn"]


def main():
    args = base_parser(__doc__).parse_args()
    od = out_dir(args.data, "part5")
    model, _ = MotionRNN.from_checkpoint(default_ckpt(args))
    seqs = test_sequences(args)
    res = {}
    for lvl, kw in LEVELS.items():
        res[lvl] = {}
        for kind in TRACKERS:
            rows = []
            for i, s in enumerate(seqs):
                d = degrade_detections(s.dets, img_wh=(s.width, s.height), seed=100 + i, **kw)
                m = evaluate_seq(s, run_tracker(s, kind, model, dets=d, **tracker_params(args.data, kind)), dets=d)
                rows.append(m)
            res[lvl][kind] = aggregate(rows)
        print(lvl, "mAP=%.3f" % res[lvl]["iou"]["mAP"],
              {k: round(res[lvl][k]["IDF1"], 3) for k in TRACKERS}, flush=True)

    lines = ["| degradação | drop | ruído | FP/quadro | mAP | IDF1 IoU | IDF1 Kalman | IDF1 RNN | IDSW IoU | IDSW Kalman | IDSW RNN |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lvl, kw in LEVELS.items():
        r = res[lvl]
        lines.append(f"| {lvl} | {kw['drop_p']:.2f} | {kw['noise']:.2f} | {kw['fp_rate']:.1f} | {r['iou']['mAP']:.3f} | "
                     + " | ".join(f"{r[k]['IDF1']:.3f}" for k in TRACKERS) + " | "
                     + " | ".join(str(r[k]["IDSW"]) for k in TRACKERS) + " |")
    print("\n".join(lines))
    with open(os.path.join(od, "table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    maps = [res[l]["iou"]["mAP"] for l in LEVELS]
    for kind in TRACKERS:
        idf = [res[l][kind]["IDF1"] for l in LEVELS]
        axs[0].plot(maps, idf, "o-", color=COLORS[kind], label=LABELS[kind])
        for x, y, l in zip(maps, idf, LEVELS):
            if kind == "rnn":
                axs[0].annotate(l, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        rel = [res[l][kind]["IDF1"] / res["original"][kind]["IDF1"] for l in LEVELS]
        axs[1].plot(range(len(LEVELS)), rel, "o-", color=COLORS[kind], label=LABELS[kind])
    rel_map = [m / maps[0] for m in maps]
    axs[1].plot(range(len(LEVELS)), rel_map, "k--", label="mAP (relativo)")
    axs[0].set_xlabel("mAP@0.5 das detecções degradadas")
    axs[0].set_ylabel("IDF1")
    axs[0].invert_xaxis()
    axs[1].set_xticks(range(len(LEVELS)))
    axs[1].set_xticklabels(list(LEVELS))
    axs[1].set_ylabel("relativo ao original")
    axs[1].set_title("abaixo da linha do mAP = o rastreador amplifica a falha")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Parte 5 — qualidade do detector ({args.data}, sem retreinar)")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_stress.png"), dpi=110)
    plt.close(fig)
    save_json(dict(levels=LEVELS, results=res), os.path.join(od, "part5.json"))


if __name__ == "__main__":
    main()
