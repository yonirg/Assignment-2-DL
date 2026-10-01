"""Utilidades compartilhadas pelos scripts de experimento."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pa2.data import mot17_sequences  # noqa: E402
from pa2.evaluate import synth_test_suite  # noqa: E402

COLORS = {"iou": "#7f7f7f", "kalman": "#1f77b4", "rnn": "#d62728",
          "rnn_fix": "#2ca02c", "gru": "#d62728", "lstm": "#9467bd", "rnn_cell": "#ff7f0e"}
LABELS = {"iou": "IoU ingênuo (Parte 1)", "kalman": "Kalman v. const. (baseline)",
          "rnn": "MotionRNN (Trilha A)"}


def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--data", choices=["synth", "mot17"], default="synth")
    p.add_argument("--mot-root", default=os.path.join(ROOT, "data", "MOT17"))
    p.add_argument("--detector", default="FRCNN")
    p.add_argument("--ckpt", default=None, help="checkpoint da MotionRNN (padrão: checkpoints/<data>_gru.pt)")
    p.add_argument("--quick", action="store_true", help="menos sequências/iterações (depuração)")
    return p


def out_dir(data, part):
    d = os.path.join(ROOT, "results", data, part)
    os.makedirs(d, exist_ok=True)
    return d


def default_ckpt(args):
    return args.ckpt or os.path.join(ROOT, "checkpoints", f"{args.data}_gru.pt")


def test_sequences(args, split="test", n=30):
    if args.data == "synth":
        return synth_test_suite(n if not args.quick else 8, seed0=0)
    return mot17_sequences(args.mot_root, split, args.detector)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))


def fmt(m, keys=("IDF1", "IDSW", "Frag", "id_ratio", "IDSW_per_gt", "count_err_rel", "mAP")):
    return " ".join(f"{k}={m[k]:.3f}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in keys)


def id_color(i):
    rng = np.random.default_rng(int(i) * 7919 + 13)
    c = rng.uniform(0.15, 1.0, 3)
    return tuple(c)
