"""Parte 1 — baseline por quadro (associação ingênua por IoU).

1. Fontes de detecção: mAP por sequência de cada fonte (MOT17: DPM, FRCNN, SDP
   públicos + torchvision, se ``det/det_tv.txt`` existir — gere com
   ``python -m pa2.detect``). No sintético a fonte é o simulador de detector.
2. Varredura da regra de associação (guloso vs. Hungarian, limiar de IoU,
   k = quadros até a morte) na VALIDAÇÃO -> escolha documentada.
3. Gráfico do descolamento: em cima mAP e IDF1, embaixo #ids previstas/#ids
   verdadeiras e ID switches por identidade verdadeira, com as sequências
   ordenadas por dificuldade (densidade).

  python experiments/part1_baseline.py --data synth
  python experiments/part1_baseline.py --data mot17 --mot-root data/MOT17
"""
from __future__ import annotations

import itertools
import os

import numpy as np

from common import base_parser, out_dir, plt, save_json
from metrics import average_precision
from pa2.data import MOT17_SPLIT, load_mot_sequence, synth_sequence
from pa2.evaluate import aggregate, evaluate_seq, run_tracker, tracker_params
from pa2.synth import SynthConfig


def difficulty_suite(n_levels=10, seeds_per_level=1):
    """Sequências sintéticas de dificuldade crescente (densidade + velocidade + oclusão)."""
    out = []
    for d in range(n_levels):
        for k in range(seeds_per_level):
            cfg = SynthConfig(seed=3000 + 10 * d + k, n_objects=4 + 2 * d, speed=1.0 + 0.3 * d,
                              occlusion_len=2 * d, n_frames=60)
            s = synth_sequence(cfg, name=f"synth-d{d}")
            s.meta["difficulty"] = d
            out.append(s)
    return out


def all_mot17(args, detector):
    names = sum(MOT17_SPLIT.values(), [])
    seqs = []
    for n in names:
        d = os.path.join(args.mot_root, "train", f"{n}-{detector}")
        seqs.append(load_mot_sequence(d))
    return seqs


def detector_table(args, od):
    """mAP de cada fonte de detecção por sequência."""
    rows = []
    for det in ["DPM", "FRCNN", "SDP"]:
        for s in all_mot17(args, det):
            r = dict(detector=det, seq=s.name, mAP=average_precision(s.gt, s.dets, s.ignore_all))
            rows.append(r)
            tv = os.path.join(os.path.dirname(s.img_dir), "det", "det_tv.txt")
            if det == "FRCNN" and os.path.exists(tv):
                s2 = load_mot_sequence(os.path.dirname(s.img_dir), det_file=tv)
                rows.append(dict(detector="torchvision", seq=s.name,
                                 mAP=average_precision(s2.gt, s2.dets, s2.ignore_all)))
    by = {}
    for r in rows:
        by.setdefault(r["detector"], []).append(r["mAP"])
    print("mAP@0.5 médio por fonte:", {k: round(float(np.mean(v)), 3) for k, v in by.items()})
    save_json(rows, os.path.join(od, "detector_map.json"))
    return rows


def association_sweep(val_seqs, data):
    grid = dict(matcher=["greedy", "hungarian"], iou_thr=[0.1, 0.2, 0.3, 0.5],
                max_age=[1, 3, 5, 10, 30])
    res = []
    for matcher, thr, age in itertools.product(*grid.values()):
        kw = dict(tracker_params(data, "iou"), matcher=matcher, iou_thr=thr, max_age=age)
        agg = aggregate([evaluate_seq(s, run_tracker(s, "iou", **kw)) for s in val_seqs])
        res.append(dict(matcher=matcher, iou_thr=thr, max_age=age, **agg))
    best = max(res, key=lambda r: r["IDF1"])
    print("melhor regra ingênua na validação:", {k: best[k] for k in ["matcher", "iou_thr", "max_age", "IDF1"]})
    return res, best


def plot_sweep(res, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, matcher in zip(axs, ["greedy", "hungarian"]):
        for thr in sorted({r["iou_thr"] for r in res}):
            rr = [r for r in res if r["matcher"] == matcher and r["iou_thr"] == thr]
            ax.plot([r["max_age"] for r in rr], [r["IDF1"] for r in rr], "o-", label=f"IoU >= {thr}")
        ax.set_xscale("log")
        ax.set_xlabel("k: quadros sem observação até a morte da track")
        ax.set_title(f"matching {matcher}")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("IDF1 (validação)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def decoupling(seqs, kw, path, title, extra=None):
    rows = []
    for s in seqs:
        m = evaluate_seq(s, run_tracker(s, "iou", **kw))
        m["seq"] = s.name
        m["difficulty"] = s.meta.get("difficulty", s.density)
        rows.append(m)
    rows.sort(key=lambda r: r["difficulty"])
    x = np.arange(len(rows))
    names = [f"{r['seq']}\n{r['density']:.1f}/q" for r in rows]
    fig, axs = plt.subplots(2, 1, figsize=(max(8, 1.0 * len(rows)), 7), sharex=True)
    axs[0].plot(x, [r["mAP"] for r in rows], "s-", color="#2ca02c", label="mAP@0.5 por quadro (detector)")
    axs[0].plot(x, [r["IDF1"] for r in rows], "o-", color="#7f7f7f", label="IDF1 (IoU ingênuo)")
    if extra:
        for lab, vals, st in extra:
            axs[0].plot(x, [vals[r["seq"]] for r in rows], st, label=lab)
    axs[0].set_ylim(0, 1.02)
    axs[0].legend(loc="lower left")
    axs[0].grid(alpha=0.3)
    ax2 = axs[1]
    ax2.plot(x, [r["id_ratio"] for r in rows], "o-", color="#9467bd", label="#ids previstas / #ids verdadeiras")
    ax2.axhline(1, color="#9467bd", ls=":", lw=1)
    ax2.set_ylabel("razão de identidades", color="#9467bd")
    ax3 = ax2.twinx()
    ax3.plot(x, [r["IDSW_per_gt"] for r in rows], "^-", color="#d62728", label="ID switches / id verdadeira")
    ax3.set_ylabel("IDSW por identidade verdadeira", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax2.set_xlabel("sequências ordenadas por dificuldade (sintético: nível = +objetos, +velocidade, "
                   "+oclusão; MOT17: densidade = caixas gt por quadro)")
    ax2.grid(alpha=0.3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return rows


def main():
    args = base_parser(__doc__).parse_args()
    od = out_dir(args.data, "part1")
    out = {}
    if args.data == "synth":
        from pa2.data import synth_suite
        val = synth_suite(6 if args.quick else 15, seed0=2000, n_frames=(40, 60), occlusion_len=8, speed=2.0)
        seqs = difficulty_suite()
    else:
        out["detectors"] = detector_table(args, od)
        val = [load_mot_sequence(os.path.join(args.mot_root, "train", f"{n}-{args.detector}"))
               for n in MOT17_SPLIT["val"]]
        seqs = all_mot17(args, args.detector)
    res, best = association_sweep(val, args.data)
    out["assoc_sweep"], out["assoc_best"] = res, best
    plot_sweep(res, os.path.join(od, "fig_assoc_sweep.png"))
    kw = tracker_params(args.data, "iou")
    out["params_used"] = kw
    rows = decoupling(seqs, kw, os.path.join(od, "fig_decoupling.png"),
                      f"Parte 1 — descolamento entre detecção e identidade ({args.data})")
    out["decoupling"] = rows
    for r in rows:
        print(f"{r['seq']:18s} dens={r['density']:5.1f} mAP={r['mAP']:.3f} IDF1={r['IDF1']:.3f} "
              f"ids={r['id_ratio']:.2f} IDSW/gt={r['IDSW_per_gt']:.2f}")
    save_json(out, os.path.join(od, "part1.json"))


if __name__ == "__main__":
    main()
