"""Roda rastreadores sobre sequências e calcula as métricas próprias.

  python -m pa2.evaluate --data synth --trackers iou kalman rnn --ckpt checkpoints/synth_gru.pt
  python -m pa2.evaluate --data mot17 --mot-root data/MOT17 --split test \
      --trackers iou kalman rnn --ckpt checkpoints/mot17_gru.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from metrics import average_precision, evaluate_tracking, occlusion_survival
from .data import load_mot_sequence, mot17_sequences, synth_suite
from .trackers import CHI2_4_99, build_tracker

# Parâmetros de associação por fonte. No sintético foram escolhidos por
# varredura na VALIDAÇÃO (sementes 2000+, ver experiments/part1_baseline.py e
# experiments/part2_compare.py). No MOT17, a Parte 1 grava a melhor regra
# ingênua da validação (MOT17-09) em results/mot17/part1/part1.json e ela é
# usada automaticamente.
TRACKER_DEFAULTS = {
    "synth": dict(iou_thr=0.2, max_age=30, min_hits=2, det_thr=0.5, new_thr=0.6, matcher="greedy"),
    "mot17": dict(iou_thr=0.3, max_age=30, min_hits=3, det_thr=0.5, new_thr=0.6, matcher="greedy"),
}
# a associação ingênua da Parte 1: IoU com a última caixa, morte rápida
IOU_OVERRIDES = {"synth": dict(iou_thr=0.1, max_age=3, matcher="greedy"),
                 "mot17": dict(iou_thr=0.3, max_age=5, matcher="greedy")}
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _part1_best(data):
    path = os.path.join(_ROOT, "results", data, "part1", "part1.json")
    if data == "synth" or not os.path.exists(path):
        return {}
    with open(path) as f:
        b = json.load(f).get("assoc_best", {})
    return {k: b[k] for k in ("matcher", "iou_thr", "max_age") if k in b}


def tracker_params(data: str, kind: str, **over):
    kw = dict(TRACKER_DEFAULTS[data])
    if kind == "iou":
        kw.update(IOU_OVERRIDES[data])
        kw.update(_part1_best(data))
    else:
        kw["maha_gate"] = CHI2_4_99
    kw.update({k: v for k, v in over.items() if v is not None})
    return kw


def run_tracker(seq, kind, model=None, dets=None, **kw):
    """Roda um rastreador numa sequência. ``dets`` substitui ``seq.dets``."""
    tr = build_tracker(kind, model=model, img_h=seq.height, **kw)
    d = seq.dets if dets is None else dets
    by_f = {f: np.zeros((0, 5)) for f in range(1, seq.n_frames + 1)}
    for f in np.unique(d[:, 0]).astype(int):
        by_f[f] = d[d[:, 0] == f][:, 1:6]
    return tr.run(by_f, seq.n_frames)


OCC_VIS = {"synth": 0.2, "mot17": 0.25}   # abaixo disso o objeto conta como ocluído


def evaluate_seq(seq, pred, dets=None, occ_vis=None):
    m = evaluate_tracking(seq.gt, pred, ignore=seq.ignore_all)
    m["mAP"] = average_precision(seq.gt, seq.dets if dets is None else dets, ignore=seq.ignore_all)
    m["density"] = seq.density
    # horizonte de memória empírico: a identidade sobrevive às oclusões?
    vis = occ_vis if occ_vis is not None else max(seq.min_vis, 0.2)
    surv = [s for s in occlusion_survival(seq.gt_full, pred, vis_thr=vis, min_len=2) if s[1] is not None]
    m["occ_events"] = len(surv)
    m["occ_survived"] = int(sum(s[1] for s in surv))
    m["occ_durations"] = [int(s[0]) for s in surv]
    m["occ_ok"] = [bool(s[1]) for s in surv]
    return m


def aggregate(rows: list) -> dict:
    """Combina sequências somando contagens (IDF1 global, não média de IDF1s)."""
    s = {k: sum(r[k] for r in rows) for k in ["IDTP", "IDFP", "IDFN", "IDSW", "Frag", "FP", "FN",
                                             "n_gt_ids", "n_pred_ids"]}
    idf1 = 2 * s["IDTP"] / max(2 * s["IDTP"] + s["IDFP"] + s["IDFN"], 1)
    return dict(IDF1=idf1, IDSW=s["IDSW"], Frag=s["Frag"], FP=s["FP"], FN=s["FN"],
                n_gt_ids=s["n_gt_ids"], n_pred_ids=s["n_pred_ids"],
                id_ratio=s["n_pred_ids"] / max(s["n_gt_ids"], 1),
                IDSW_per_gt=s["IDSW"] / max(s["n_gt_ids"], 1),
                count_err_rel=float(np.mean([r["count_err_rel"] for r in rows])),
                mAP=float(np.nanmean([r["mAP"] for r in rows])),
                occ_events=sum(r.get("occ_events", 0) for r in rows),
                occ_survival=(sum(r.get("occ_survived", 0) for r in rows)
                              / sum(r.get("occ_events", 0) for r in rows))
                if sum(r.get("occ_events", 0) for r in rows) else float("nan"))


def synth_test_suite(n=30, seed0=0, **kw):
    """Suíte de teste sintética (sementes nunca usadas no treino, que começa em 1000)."""
    base = dict(n_frames=(40, 60), occlusion_len=8, speed=2.0)
    base.update(kw)
    return synth_suite(n, seed0=seed0, **base)


def load_sequences(args):
    if args.data == "synth":
        return synth_test_suite(args.n_synth, args.synth_seed)
    return mot17_sequences(args.mot_root, args.split, args.detector)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", choices=["synth", "mot17"], default="synth")
    p.add_argument("--mot-root", default="data/MOT17")
    p.add_argument("--split", default="test")
    p.add_argument("--detector", default="FRCNN")
    p.add_argument("--n-synth", type=int, default=30)
    p.add_argument("--synth-seed", type=int, default=0)
    p.add_argument("--trackers", nargs="+", default=["iou", "kalman", "rnn"])
    p.add_argument("--ckpt", default="checkpoints/synth_gru.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--det-file", default=None,
                   help="MOT17: arquivo em det/ no lugar de det.txt (ex.: det_tv.txt, do pa2.detect)")
    args = p.parse_args()

    model = None
    if "rnn" in args.trackers:
        from .models import MotionRNN
        model, _ = MotionRNN.from_checkpoint(args.ckpt)
    seqs = load_sequences(args)
    if args.det_file:
        for s in seqs:
            d = os.path.dirname(s.img_dir)
            s.dets = load_mot_sequence(d, det_file=os.path.join(d, "det", args.det_file)).dets
    results = {}
    for kind in args.trackers:
        kw = tracker_params(args.data, kind)
        rows = []
        for s in seqs:
            m = evaluate_seq(s, run_tracker(s, kind, model, **kw))
            m["seq"] = s.name
            rows.append(m)
        results[kind] = dict(per_seq=rows, total=aggregate(rows), params=kw)
    print(f"{'tracker':8s} {'IDF1':>6s} {'IDSW':>6s} {'Frag':>6s} {'#pred/#gt':>10s} {'IDSW/gt':>8s} {'mAP':>6s}")
    for kind, r in results.items():
        t = r["total"]
        print(f"{kind:8s} {t['IDF1']:6.3f} {t['IDSW']:6d} {t['Frag']:6d} {t['id_ratio']:10.2f} "
              f"{t['IDSW_per_gt']:8.3f} {t['mAP']:6.3f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=float)


if __name__ == "__main__":
    main()
