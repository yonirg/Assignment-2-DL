"""Parte 2 — Trilha A (RNN como modelo de movimento) vs. baselines.

Mesmas sequências de teste, mesmas métricas, lado a lado:
  IoU ingênuo (Parte 1) | Kalman velocidade constante | MotionRNN

Gera tabela (part2.json), barras, sobrevivência à oclusão por duração e uma
figura mostrando *onde* a recorrência ajuda: a caixa prevista atravessando a
oclusão enquanto a caixa do baseline fica parada.

  python experiments/part2_compare.py --data synth
  python experiments/part2_compare.py --data mot17 --mot-root data/MOT17
"""
from __future__ import annotations

import os

import numpy as np

from common import COLORS, LABELS, base_parser, default_ckpt, id_color, out_dir, plt, save_json, test_sequences
from pa2.evaluate import aggregate, evaluate_seq, run_tracker, tracker_params
from pa2.models import MotionRNN
from pa2.trackers import build_tracker

TRACKERS = ["iou", "kalman", "rnn"]


def survival_curve(rows, bins):
    d = np.concatenate([r["occ_durations"] for r in rows]) if rows else np.zeros(0)
    ok = np.concatenate([r["occ_ok"] for r in rows]).astype(float) if rows else np.zeros(0)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        out.append((float(ok[m].mean()) if m.any() else np.nan, int(m.sum())))
    return out


def fig_occlusion_example(seq, model, data, path):
    """Uma oclusão do teste: caixa prevista por Kalman e RNN enquanto o objeto some."""
    from metrics import occlusion_events
    ev = [e for e in occlusion_events(seq.gt_full, 0.2, 6)]
    if not ev or seq.frames is None:
        return
    g, f0, f1, dur = max(ev, key=lambda e: e[3])
    logs = {}
    for kind in ["iou", "kalman", "rnn"]:
        tr = build_tracker(kind, model=model, img_h=seq.height, record=True, **tracker_params(data, kind))
        tr.run(seq.dets_by_frame(), seq.n_frames)
        logs[kind] = np.array(tr.log)
    gtr = seq.gt_full[seq.gt_full[:, 1] == g]
    frames = np.linspace(f0, f1 + 1, 6).round().astype(int)
    fig, axs = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3.4))
    for ax, f in zip(axs, frames):
        ax.imshow(seq.image(f))
        r = gtr[gtr[:, 0] == f]
        if len(r):
            r = r[0]
            ax.add_patch(plt.Rectangle((r[2], r[3]), r[4] - r[2], r[5] - r[3], fill=False, ec="lime", lw=2))
        for kind, st in [("iou", ":"), ("kalman", "--"), ("rnn", "-")]:
            L = logs[kind]
            if not len(L):
                continue
            L = L[L[:, 0] == f]
            if not len(L) or not len(r):
                continue
            # previsão da track mais próxima do objeto
            c = np.array([(r[2] + r[4]) / 2, (r[3] + r[5]) / 2])
            pc = np.column_stack([(L[:, 2] + L[:, 4]) / 2, (L[:, 3] + L[:, 5]) / 2])
            k = np.argmin(np.linalg.norm(pc - c, axis=1))
            if np.linalg.norm(pc[k] - c) > 40:
                continue
            b = L[k]
            ax.add_patch(plt.Rectangle((b[2], b[3]), b[4] - b[2], b[5] - b[3], fill=False,
                                       ec=COLORS[kind], lw=1.5, ls=st))
        ax.set_title(f"t={f} (oclusão {f0 + 1}–{f1 - 1})", fontsize=8)
        ax.axis("off")
    fig.suptitle("verde: gt  |  previsão para o quadro — cinza pontilhado: IoU (última caixa), "
                 "azul tracejado: Kalman, vermelho: MotionRNN", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    args = base_parser(__doc__).parse_args()
    od = out_dir(args.data, "part2")
    model, ck = MotionRNN.from_checkpoint(default_ckpt(args))
    seqs = test_sequences(args)
    res = {}
    for kind in TRACKERS:
        kw = tracker_params(args.data, kind)
        rows = []
        for s in seqs:
            m = evaluate_seq(s, run_tracker(s, kind, model, **kw))
            m["seq"] = s.name
            rows.append(m)
        res[kind] = dict(total=aggregate(rows), per_seq=rows, params=kw)
    keys = ["IDF1", "IDSW", "Frag", "id_ratio", "IDSW_per_gt", "count_err_rel", "occ_survival", "mAP"]
    lines = ["| rastreador | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for kind in TRACKERS:
        t = res[kind]["total"]
        lines.append(f"| {kind} | " + " | ".join(f"{t[k]:.3f}" if isinstance(t[k], float) else str(t[k])
                                                for k in keys) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(od, "table.md"), "w") as f:
        f.write(table + "\n")

    # barras
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k, lab in zip(axs, ["IDF1", "IDSW_per_gt", "id_ratio", "occ_survival"],
                          ["IDF1", "ID switches / id verdadeira", "#ids prev. / #ids verd.",
                           "oclusões com id preservada"]):
        ax.bar(range(3), [res[t]["total"][k] for t in TRACKERS], color=[COLORS[t] for t in TRACKERS])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["IoU", "Kalman", "RNN"])
        ax.set_title(lab)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Parte 2 — teste ({args.data}, {len(seqs)} sequências)")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_bars.png"), dpi=110)
    plt.close(fig)

    # sobrevivência por duração de oclusão
    bins = [2, 4, 8, 12, 16, 24, 32, 64, 1000]
    fig, ax = plt.subplots(figsize=(7, 4))
    surv = {}
    for kind in TRACKERS:
        c = survival_curve(res[kind]["per_seq"], bins)
        surv[kind] = c
        ax.plot(range(len(c)), [v for v, _ in c], "o-", color=COLORS[kind], label=LABELS[kind])
    counts = [n for _, n in surv["rnn"]]
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{lo}-{hi - 1}\n(n={n})" for lo, hi, n in zip(bins[:-1], bins[1:], counts)], fontsize=8)
    ax.set_xlabel("duração da oclusão (quadros)")
    ax.set_ylabel("fração com identidade preservada")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_survival.png"), dpi=110)
    plt.close(fig)
    res["survival_bins"] = bins
    res["survival"] = surv
    if args.data == "synth":
        from pa2.data import synth_sequence
        from pa2.synth import SynthConfig
        ex = synth_sequence(SynthConfig(seed=7, n_objects=6, occlusion_len=12, speed=1.5, n_frames=60),
                            render=True)
        fig_occlusion_example(ex, model, args.data, os.path.join(od, "fig_occlusion_example.png"))
    save_json(res, os.path.join(od, "part2.json"))


if __name__ == "__main__":
    main()
