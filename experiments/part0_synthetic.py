"""Parte 0 — testes sintéticos.

Gera:
  results/synth/part0/fig_occlusion.png     trajetória que some N quadros atrás do poste e volta
  results/synth/part0/fig_detector_sim.png  simulador de detector (TP, FP, perdidas)
  results/synth/part0/metric_tests.txt      saída dos testes (a), (b), (c) da métrica
  results/synth/part0/floor.json            baseline ingênuo no piso fácil
  results/synth/part0/fig_sweep.png         onde o baseline quebra (objetos, velocidade, oclusão)

  python experiments/part0_synthetic.py [--trackers iou kalman rnn]
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from common import COLORS, LABELS, ROOT, base_parser, out_dir, plt, save_json
from metrics import occlusion_events
from pa2 import synth
from pa2.data import synth_sequence, synth_suite
from pa2.evaluate import aggregate, evaluate_seq, run_tracker, tracker_params

MILD_DET = dict(drop_p=0.02, noise=0.02, fp_rate=0.05, dup_p=0.0)


def fig_occlusion(od):
    cfg = synth.SynthConfig(seed=3, n_objects=6, n_frames=60, occlusion_len=12, speed=1.5)
    v = synth.generate(cfg)
    ev = [e for e in occlusion_events(v["gt"], vis_thr=0.01) if e[3] >= 4]
    g, f0, f1, dur = min(ev, key=lambda e: abs(e[3] - cfg.occlusion_len))
    tr = v["gt"][v["gt"][:, 1] == g]
    show = np.linspace(f0 - 3, f1 + 3, 8).round().astype(int)
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(2, 8, height_ratios=[2.2, 1])
    for k, f in enumerate(show):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(v["frames"][f - 1], cmap="gray", vmin=0, vmax=1)
        r = tr[tr[:, 0] == f][0]
        ls = "-" if r[6] > 0.01 else "--"
        ax.add_patch(plt.Rectangle((r[2], r[3]), r[4] - r[2], r[5] - r[3], fill=False,
                                   ec="lime" if r[6] > 0.01 else "red", ls=ls, lw=1.5))
        ax.set_title(f"t={f}  vis={r[6]:.2f}", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(tr[:, 0], tr[:, 6], "k.-")
    ax.axvspan(f0 + 0.5, f1 - 0.5, color="red", alpha=0.15, label=f"invisível por {dur} quadros")
    ax.set_xlabel("quadro")
    ax.set_ylabel("visibilidade")
    ax.legend(loc="lower left")
    fig.suptitle(f"Objeto {g}: passa atrás do poste (sempre na frente no z-buffer) — some por {dur} "
                 f"quadros (occlusion_len={cfg.occlusion_len}) e volta; tracejado vermelho = gt sob oclusão total")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_occlusion.png"), dpi=110)
    plt.close(fig)
    return dict(object=int(g), last_visible=f0, reappears=f1, hidden_frames=dur)


def fig_detector(od):
    s = synth_sequence(synth.SynthConfig(seed=5, n_objects=10, occlusion_len=8),
                       det_kw=dict(drop_p=0.15, noise=0.06, fp_rate=1.0, dup_p=0.3), render=True)
    f = 20
    from metrics import iou_matrix
    g = s.gt_full[s.gt_full[:, 0] == f]
    d = s.dets[s.dets[:, 0] == f]
    iou = iou_matrix(d[:, 1:5], g[:, 2:6])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(s.frames[f - 1], cmap="gray", vmin=0, vmax=1)
    for r in g:
        ax.add_patch(plt.Rectangle((r[2], r[3]), r[4] - r[2], r[5] - r[3], fill=False, ec="lime", lw=2,
                                   ls="-" if r[6] >= 0.35 else ":"))
    for k, r in enumerate(d):
        tp = iou.shape[1] and iou[k].max() >= 0.5
        ax.add_patch(plt.Rectangle((r[1], r[2]), r[3] - r[1], r[4] - r[2], fill=False,
                                   ec="cyan" if tp else "red", lw=1.2))
        ax.text(r[1], r[2] - 1, f"{r[5]:.2f}", color="cyan" if tp else "red", fontsize=6)
    ax.set_title("Simulador de detector — verde: gt (pontilhado = ocluído),\n"
                 "ciano: detecção casada, vermelho: falso positivo", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_detector_sim.png"), dpi=120)
    plt.close(fig)


def sweep(trackers, model, n_seq):
    axes = {
        "n_objects": [3, 6, 9, 12, 15, 20],
        "speed": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
        "occlusion_len": [0, 4, 8, 12, 16, 24],
    }
    base = dict(n_objects=8, speed=1.5, occlusion_len=0, n_frames=(40, 60))
    res = {}
    for ax_name, values in axes.items():
        res[ax_name] = {}
        for val in values:
            cfg = dict(base)
            cfg[ax_name] = val
            seqs = synth_suite(n_seq, seed0=500, **cfg)
            for kind in trackers:
                kw = tracker_params("synth", kind)
                rows = [evaluate_seq(s, run_tracker(s, kind, model, **kw)) for s in seqs]
                res[ax_name].setdefault(kind, []).append(dict(value=val, **aggregate(rows)))
            print(ax_name, val, {k: round(res[ax_name][k][-1]["IDF1"], 3) for k in trackers}, flush=True)
    return res


def plot_sweep(res, path, trackers, title):
    fig, axs = plt.subplots(3, 3, figsize=(14, 10), sharey="row")
    names = {"n_objects": "número de objetos", "speed": "velocidade típica (px/quadro)",
             "occlusion_len": "duração da oclusão (quadros)"}
    for c, (ax_name, by_tr) in enumerate(res.items()):
        for kind in trackers:
            rows = by_tr[kind]
            x = [r["value"] for r in rows]
            axs[0, c].plot(x, [r["IDF1"] for r in rows], "o-", color=COLORS[kind], label=LABELS[kind])
            axs[1, c].plot(x, [r["IDSW_per_gt"] for r in rows], "o-", color=COLORS[kind])
            axs[2, c].plot(x, [r["occ_survival"] for r in rows], "o-", color=COLORS[kind])
        axs[2, c].set_xlabel(names[ax_name])
        for rr in range(3):
            axs[rr, c].grid(alpha=0.3)
    axs[0, 0].set_ylabel("IDF1")
    axs[1, 0].set_ylabel("ID switches / identidade verdadeira")
    axs[2, 0].set_ylabel("fração de oclusões (>=2 quadros)\ncom a identidade preservada")
    axs[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = base_parser(__doc__)
    p.add_argument("--trackers", nargs="+", default=["iou"])
    args = p.parse_args()
    od = out_dir("synth", "part0")
    info = {"occlusion_figure": fig_occlusion(od)}
    fig_detector(od)
    r = subprocess.run([sys.executable, "-m", "pytest", "-v", os.path.join(ROOT, "tests")],
                       capture_output=True, text=True, cwd=ROOT)
    with open(os.path.join(od, "metric_tests.txt"), "w") as f:
        f.write(r.stdout)
    print(r.stdout.splitlines()[-1])

    # piso fácil: poucas elipses, lentas, sem oclusão, detector quase perfeito
    easy = synth_suite(10 if not args.quick else 3, seed0=900, n_objects=(3, 5), speed=0.5,
                       occlusion_len=0, det_kw=MILD_DET)
    kw = tracker_params("synth", "iou")
    floor = aggregate([evaluate_seq(s, run_tracker(s, "iou", **kw)) for s in easy])
    info["floor"] = floor
    print("piso fácil (IoU ingênuo):", {k: round(floor[k], 4) for k in ["IDF1", "IDSW", "id_ratio"]})

    model = None
    if "rnn" in args.trackers:
        from common import default_ckpt
        from pa2.models import MotionRNN
        model, _ = MotionRNN.from_checkpoint(default_ckpt(args))
    res = sweep(args.trackers, model, 10 if not args.quick else 3)
    info["sweep"] = res
    save_json(info, os.path.join(od, "part0.json"))
    plot_sweep({k: {"iou": v["iou"]} for k, v in res.items()}, os.path.join(od, "fig_sweep.png"), ["iou"],
               "Parte 0 — onde a associação ingênua quebra (detector simulado padrão, 10 vídeos/ponto)")
    if len(args.trackers) > 1:
        od2 = out_dir("synth", "part2")
        plot_sweep(res, os.path.join(od2, "fig_sweep_all.png"), args.trackers,
                   "Parte 2 — mesmos eixos de dificuldade da Parte 0, três rastreadores")


if __name__ == "__main__":
    main()
