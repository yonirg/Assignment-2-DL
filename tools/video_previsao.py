"""Vídeo do rastreador mostrando o que a recorrência prevê.

Para cada quadro:
  * caixa sólida   = saída do rastreador (detecção casada), cor fixa por identidade;
  * caixa tracejada = caixa que a MotionRNN previu para este quadro antes de ver a detecção;
  * sem detecção (oclusão ou falha do detector): só a previsão, em vermelho-tracejado, com a
    elipse de 2σ da incerteza prevista — ela cresce enquanto a track anda sozinha.

  python tools/video_previsao.py --seq data/MOT17/train/MOT17-10-FRCNN --out results/previsao_mot17.mp4 --fps 12
  python tools/video_previsao.py --seq data/synth_demo/SYNTH-DEMO --out results/previsao_synth.mp4 --fps 5
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from pa2.data import load_mot_sequence  # noqa: E402
from pa2.evaluate import tracker_params  # noqa: E402
from pa2.models import MotionRNN  # noqa: E402
from pa2.trackers import build_tracker  # noqa: E402

RED = (60, 60, 255)


def color(i):
    rng = np.random.default_rng(int(i) * 7919 + 13)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def dashed_rect(img, p1, p2, col, th, dash=6):
    (x1, y1), (x2, y2) = p1, p2
    for a, b in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for k in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            e = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, s, e, col, th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=float, default=0, help="0 = fps da sequência; menor = mais lento")
    a = ap.parse_args()

    seq = load_mot_sequence(a.seq)
    data = "synth" if seq.width <= 256 else "mot17"
    if data == "synth":
        seq.min_vis = 0.2
    model, _ = MotionRNN.from_checkpoint(a.ckpt or f"checkpoints/{data}_gru.pt")
    n = min(seq.n_frames, a.max_frames) if a.max_frames else seq.n_frames

    tr = build_tracker("rnn", model, img_h=seq.height, record=True, **tracker_params(data, "rnn"))
    by_f = {f: np.zeros((0, 5)) for f in range(1, n + 1)}
    for f in np.unique(seq.dets[:, 0]).astype(int):
        if f <= n:
            by_f[f] = seq.dets[seq.dets[:, 0] == f][:, 1:6]
    out = tr.run(by_f, n)
    confirmed = set(out[:, 1].astype(int))
    log = np.asarray(tr.log, dtype=np.float64).reshape(-1, 9)  # f, id, x1,y1,x2,y2, miss, sx, sy

    scale = max(1, int(round(640 / seq.width))) if seq.width <= 640 else 960 / seq.width
    W, H = int(seq.width * scale), int(seq.height * scale)
    th = 2 if scale >= 1 else 2
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps or seq.fps, (W, H))
    seen = set()
    for f in range(1, n + 1):
        img = cv2.resize(seq.image(f)[..., ::-1].copy(), (W, H),
                         interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA)
        n_occ = 0
        for r in log[log[:, 0] == f]:
            tid, miss = int(r[1]), int(r[6])
            if tid not in confirmed:
                continue
            x1, y1, x2, y2 = (r[2:6] * scale).astype(int)
            if miss > 0:   # sem detecção: a recorrência anda sozinha
                n_occ += 1
                dashed_rect(img, (x1, y1), (x2, y2), RED, th + 1)
                if np.isfinite(r[7]) and np.isfinite(r[8]):
                    c = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    ax = (max(2, int(2 * r[7] * scale)), max(2, int(2 * r[8] * scale)))
                    cv2.ellipse(img, c, ax, 0, 0, 360, RED, 1)
                cv2.putText(img, f"{tid} prev. +{miss}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
            else:          # previsão feita antes de ver a detecção
                dashed_rect(img, (x1, y1), (x2, y2), color(tid), 1)
        for r in out[out[:, 0] == f]:
            tid = int(r[1])
            seen.add(tid)
            x1, y1, x2, y2 = (r[2:6] * scale).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color(tid), th)
            cv2.putText(img, str(tid), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color(tid), 2)
        hud = f"frame {f}  ids unicos: {len(seen)}  so previsao: {n_occ}"
        cv2.rectangle(img, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(img, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "solida: saida  tracejada: previsao RNN  vermelho: sem deteccao (2 sigma)",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        vw.write(img)
    vw.release()
    print(f"vídeo salvo em {a.out} | {n} quadros | objetos únicos: {len(seen)}")


if __name__ == "__main__":
    main()
