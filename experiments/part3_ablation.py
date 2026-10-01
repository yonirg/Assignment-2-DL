"""Parte 3 — Ablação, Eixo 1: a célula recorrente.

RNN simples vs. LSTM vs. GRU no mesmo orçamento aproximado de parâmetros da
célula (``--budget``), variando a janela de BPTT truncado T ∈ {4, 8, 16, 32},
3 sementes cada (média ± desvio). Avaliação na VALIDAÇÃO (sintético: sementes
2000+; MOT17: MOT17-09) com o mesmo rastreador e regras de associação.

Os checkpoints ficam em checkpoints/ablation/<data>/ e são reaproveitados na
Parte 4 (curva analítica de gradiente).

  python experiments/part3_ablation.py --data synth --jobs 4
  python experiments/part3_ablation.py --data mot17 --mot-root data/MOT17 --jobs 4
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import ROOT, base_parser, out_dir, plt, save_json
from pa2.data import MOT17_SPLIT, load_mot_sequence, synth_suite
from pa2.evaluate import aggregate, evaluate_seq, run_tracker, tracker_params
from pa2.models import MotionRNN

CELLS = ["rnn", "lstm", "gru"]
TS = [4, 8, 16, 32]
SEEDS = [0, 1, 2]
CELL_COLORS = {"rnn": "#ff7f0e", "lstm": "#9467bd", "gru": "#d62728"}


def ckpt_path(data, cell, T, seed):
    return os.path.join(ROOT, "checkpoints", "ablation", data, f"{cell}_T{T}_s{seed}.pt")


def train_one(job):
    data, cell, T, seed, iters, extra = job
    out = ckpt_path(data, cell, T, seed)
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, "-m", "pa2.train", "--data", data, "--cell", cell, "--T", str(T),
           "--seed", str(seed), "--iters", str(iters), "--log-every", "100000", "--out", out] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
    print("ok", os.path.basename(out), flush=True)
    return out


def main():
    p = base_parser(__doc__)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--iters", type=int, default=3000)
    args = p.parse_args()
    od = out_dir(args.data, "part3")
    extra = ["--mot-root", args.mot_root, "--detector", args.detector] if args.data == "mot17" else []
    jobs = [(args.data, c, T, s, args.iters, extra) for c, T, s in itertools.product(CELLS, TS, SEEDS)]
    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(train_one, jobs))

    if args.data == "synth":
        val = synth_suite(8 if args.quick else 20, seed0=2000, n_frames=(40, 60), occlusion_len=8, speed=2.0)
    else:
        val = [load_mot_sequence(os.path.join(args.mot_root, "train", f"{n}-{args.detector}"))
               for n in MOT17_SPLIT["val"]]
    kw = tracker_params(args.data, "rnn")
    rows = []
    for cell, T, seed in itertools.product(CELLS, TS, SEEDS):
        model, ck = MotionRNN.from_checkpoint(ckpt_path(args.data, cell, T, seed))
        agg = aggregate([evaluate_seq(s, run_tracker(s, "rnn", model, **kw)) for s in val])
        h = ck["history"]
        final_loss = float(np.mean([x["loss"] for x in h[-100:]])) if h else float("nan")
        rows.append(dict(cell=cell, T=T, seed=seed, hidden=ck["model_config"]["hidden"],
                         n_params_cell=ck["n_params_cell"], final_loss=final_loss, **agg))
        print(cell, T, seed, f"IDF1={agg['IDF1']:.4f} surv={agg['occ_survival']:.3f} loss={final_loss:.4f}",
              flush=True)

    summary = []
    for cell, T in itertools.product(CELLS, TS):
        rr = [r for r in rows if r["cell"] == cell and r["T"] == T]
        s = dict(cell=cell, T=T, hidden=rr[0]["hidden"], n_params_cell=rr[0]["n_params_cell"])
        for k in ["IDF1", "IDSW", "occ_survival", "final_loss"]:
            v = np.array([r[k] for r in rr], dtype=float)
            s[k + "_mean"], s[k + "_std"] = float(v.mean()), float(v.std(ddof=1))
        summary.append(s)
    lines = ["| célula | h | parâm. célula | T | IDF1 | IDSW | sobrevivência à oclusão | perda final |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['cell']} | {s['hidden']} | {s['n_params_cell']} | {s['T']} | "
                     f"{s['IDF1_mean']:.3f} ± {s['IDF1_std']:.3f} | {s['IDSW_mean']:.1f} ± {s['IDSW_std']:.1f} | "
                     f"{s['occ_survival_mean']:.3f} ± {s['occ_survival_std']:.3f} | "
                     f"{s['final_loss_mean']:.3f} ± {s['final_loss_std']:.3f} |")
    print("\n".join(lines))
    with open(os.path.join(od, "table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for cell in CELLS:
        ss = [s for s in summary if s["cell"] == cell]
        for ax, k in zip(axs, ["IDF1", "occ_survival", "final_loss"]):
            ax.errorbar(TS, [s[k + "_mean"] for s in ss], yerr=[s[k + "_std"] for s in ss], marker="o",
                        capsize=4, color=CELL_COLORS[cell], label=f"{cell.upper()} (h={ss[0]['hidden']})")
    for ax, t in zip(axs, ["IDF1 (validação)", "fração de oclusões com id preservada", "perda de treino final"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(TS)
        ax.set_xticklabels(TS)
        ax.set_xlabel("janela de BPTT truncado T")
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axs[0].legend()
    fig.suptitle(f"Parte 3 — Eixo 1: célula × janela de BPTT, ~{summary[0]['n_params_cell']} parâm. "
                 f"na célula, 3 sementes ({args.data})")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_ablation.png"), dpi=110)
    plt.close(fig)
    save_json(dict(rows=rows, summary=summary), os.path.join(od, "part3.json"))


if __name__ == "__main__":
    main()
