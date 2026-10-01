"""Parte 4 — galeria de falhas e horizonte de memória.

1. Galeria: os 3 piores trechos do modelo final no teste (ID switches de tipos
   diferentes), com tira de quadros (gt e predição coloridos por identidade)
   + o mapa intermediário: caixa prevista pela recorrência e sua incerteza.
2. Horizonte analítico: ||∂L_t / ∂h_{t-k}|| em função de k, para RNN simples,
   LSTM e GRU (checkpoints da ablação, mesma janela T) e para o modelo final.
3. Horizonte empírico: fração de oclusões em que a identidade sobrevive, por
   duração, comparada com a distribuição de duração de oclusão do dataset.
4. Correção: re-treino guiado pelo diagnóstico (``--fix-ckpt``), antes/depois.

  python experiments/part4_memory.py --data synth [--fix-ckpt checkpoints/synth_gru_fix.pt]
"""
from __future__ import annotations

import os

import numpy as np
import torch

from common import (COLORS, ROOT, base_parser, default_ckpt, id_color, out_dir, plt, save_json,
                    test_sequences)
from metrics import clear_mot, occlusion_events
from pa2.data import gt_tracklets, mot17_sequences, synth_sequence, synth_suite
from pa2.evaluate import OCC_VIS, aggregate, evaluate_seq, run_tracker, tracker_params
from pa2.models import MotionRNN
from pa2.synth import SynthConfig
from pa2.trackers import build_tracker
from pa2.train import WindowSampler, loss_fn, observation_mask, unroll


# ----------------------------------------------------------------------------
# 2. horizonte analítico
# ----------------------------------------------------------------------------
class _MaskArgs:
    min_vis, drop_p, p_occ, max_occ = 0.2, 0.0, 0.0, 1


def gradient_horizon(model, tracklets, img_hs, K=48, B=256, seed=0, part="h"):
    """Média sobre janelas de ||∂L_K / ∂h_{K-k}||, k = 0..K-1 (perda só no último passo).

    ``part='c'`` mede o gradiente na célula de memória da LSTM (onde mora a memória longa).
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    sampler = WindowSampler(tracklets, img_hs, K, rng)
    boxes, img_h = sampler.sample(B)
    mask = observation_mask(boxes[:, :-1, 4], "scheduled", 0.0, _MaskArgs, gen)
    model.train()
    mu, lv, tgt, states = unroll(model, boxes, img_h, mask, 0.0, gen, keep_states=True)
    for s in states:
        s.retain_grad()
    loss = loss_fn(mu[:, -1:], lv[:, -1:], tgt[:, -1:], "l1")
    loss.backward()
    model.eval()
    H = model.hidden
    sl = slice(0, H) if part == "h" else slice(H, 2 * H)
    norms = [states[K - 1 - k].grad[:, sl].norm(dim=1).mean().item() for k in range(K)]
    return np.array(norms)


def long_tracklets(args):
    if args.data == "synth":
        seqs = synth_suite(60, seed0=7000, n_frames=(70, 90), occlusion_len=8, speed=2.0)
    else:
        seqs = mot17_sequences(args.mot_root, "val", args.detector)
    trs, hs = [], []
    for s in seqs:
        for tr in gt_tracklets(s):
            trs.append(tr)
            hs.append(s.height)
    return trs, hs


def plot_horizon(curves, path, K, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for lab, (c, color, ls, *ref) in curves.items():
        ax.semilogy(np.arange(K), c / (ref[0] if ref else c[0]), ls, color=color, label=lab)
    ax.set_xlabel("k (passos para trás)")
    ax.set_ylabel(r"$\|\partial L_t / \partial h_{t-k}\|$  (normalizado por k=0)")
    ax.set_ylim(bottom=1e-3)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. horizonte empírico
# ----------------------------------------------------------------------------
def long_occlusion_suite(n_per=4):
    out = []
    for L in [4, 8, 12, 16, 20, 24, 28, 32]:
        out += synth_suite(n_per, seed0=8000 + 100 * L, occlusion_len=L, speed=1.5, n_frames=(60, 80),
                           n_objects=(5, 10))
    return out


def empirical(seqs, model, data, bins, **over):
    kw = tracker_params(data, "rnn", **over)
    rows = [evaluate_seq(s, run_tracker(s, "rnn", model, **kw), occ_vis=OCC_VIS[data]) for s in seqs]
    d = np.concatenate([r["occ_durations"] for r in rows])
    ok = np.concatenate([r["occ_ok"] for r in rows]).astype(float)
    curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d < hi)
        curve.append(float(ok[m].mean()) if m.any() else np.nan)
    # distribuição de duração de oclusão do dataset (todas as oclusões da gt)
    all_d = np.array([e[3] for s in seqs for e in occlusion_events(s.gt_full, OCC_VIS[data], 2)])
    return dict(curve=curve, durations=d, ok=ok, dataset_durations=all_d, total=aggregate(rows))


# ----------------------------------------------------------------------------
# 1. galeria de falhas
# ----------------------------------------------------------------------------
def find_failures(seqs, model, data, n=3):
    kw = tracker_params(data, "rnn")
    events = []
    for si, s in enumerate(seqs):
        pred = run_tracker(s, "rnn", model, **kw)
        cm = clear_mot(s.gt, pred, return_matches=True)
        matched = {}
        for f, g, p in cm["matches"]:
            matched.setdefault(g, []).append(f)
        for f, g, p_old, p_new in cm["switches"]:
            prev = [x for x in matched[g] if x < f]
            gap = f - max(prev) - 1 if prev else 0
            tr = s.gt_full[(s.gt_full[:, 1] == g) & (s.gt_full[:, 0] < f) & (s.gt_full[:, 0] > f - gap - 1)]
            occl = int((tr[:, 6] < OCC_VIS[data]).sum()) if len(tr) else 0
            # o novo id já pertencia a outro objeto? -> troca entre vizinhos
            stolen = any(pp == p_new and gg != g and ff < f for ff, gg, pp in cm["matches"])
            kind = "oclusão longa" if gap >= 10 else ("troca entre vizinhos" if stolen else "fragmentação curta")
            events.append(dict(seq=si, frame=int(f), gt=int(g), old=int(p_old), new=int(p_new), gap=int(gap),
                               occluded=occl, kind=kind, density=float(s.density)))
    chosen = []
    for kind in ["oclusão longa", "troca entre vizinhos", "fragmentação curta"]:
        ev = sorted([e for e in events if e["kind"] == kind], key=lambda e: -e["gap"] - e["density"])
        if ev:
            chosen.append(ev[0])
    rest = sorted([e for e in events if e not in chosen], key=lambda e: -e["gap"])
    chosen += rest[: max(0, n - len(chosen))]
    return chosen[:n], events


def draw_failure(seq, model, data, ev, path, diag):
    kw = tracker_params(data, "rnn")
    tr = build_tracker("rnn", model=model, img_h=seq.height, record=True, **kw)
    pred = tr.run(seq.dets_by_frame(), seq.n_frames)
    log = np.array(tr.log)
    f_sw = ev["frame"]
    start = max(1, f_sw - ev["gap"] - 3)
    frames = np.unique(np.linspace(start, min(seq.n_frames, f_sw + 2), 6).round().astype(int))
    cm = clear_mot(seq.gt, pred, return_matches=True)
    fig, axs = plt.subplots(1, len(frames), figsize=(3.1 * len(frames), 3.6))
    g = seq.gt_full[seq.gt_full[:, 1] == ev["gt"]]
    cx = (g[:, 2] + g[:, 4]) / 2
    cy = (g[:, 3] + g[:, 5]) / 2
    pad = max(40, 2.5 * float(np.median(g[:, 5] - g[:, 3])))
    for ax, f in zip(np.atleast_1d(axs), frames):
        ax.imshow(seq.image(int(f)))
        for r in seq.gt_full[seq.gt_full[:, 0] == f]:
            lw = 2.2 if r[1] == ev["gt"] else 0.8
            ax.add_patch(plt.Rectangle((r[2], r[3]), r[4] - r[2], r[5] - r[3], fill=False,
                                       ec=id_color(r[1] + 1000), lw=lw, ls="-" if r[6] >= 0.2 else ":"))
        for r in pred[pred[:, 0] == f]:
            hl = r[1] in (ev["old"], ev["new"])
            ax.add_patch(plt.Rectangle((r[2], r[3]), r[4] - r[2], r[5] - r[3], fill=False,
                                       ec=id_color(r[1]), lw=2 if hl else 0.8, ls="--"))
            if hl:
                ax.text(r[2], r[3] - 1, f"id {int(r[1])}", color=id_color(r[1]), fontsize=7)
        if len(log):
            L = log[(log[:, 0] == f) & (log[:, 1] == ev["old"])]
            for b in L:
                ax.add_patch(plt.Rectangle((b[2], b[3]), b[4] - b[2], b[5] - b[3], fill=False, ec="red",
                                           lw=1.2, ls=":"))
                if len(b) > 8 and np.isfinite(b[7]):
                    from matplotlib.patches import Ellipse
                    ax.add_patch(Ellipse(((b[2] + b[4]) / 2, (b[3] + b[5]) / 2), 4 * b[7], 4 * b[8],
                                         fill=False, ec="red", lw=0.8))
                ax.text(b[2], b[5] + 4, f"prev. id {ev['old']} (miss {int(b[6])})", color="red", fontsize=6)
        j = int(np.argmin(np.abs(g[:, 0] - f)))
        ax.set_xlim(cx[j] - pad, cx[j] + pad)
        ax.set_ylim(cy[j] + pad, cy[j] - pad)
        ax.set_title(f"t={f}" + ("  <- switch" if f == f_sw else ""), fontsize=8)
        ax.axis("off")
    fig.suptitle(f"[{ev['kind']}] gt {ev['gt']}: id {ev['old']} -> {ev['new']} no quadro {f_sw} — {diag}",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    p = base_parser(__doc__)
    p.add_argument("--fix-ckpt", default=None)
    p.add_argument("--fix-max-age", type=int, default=60)
    p.add_argument("--K", type=int, default=48)
    args = p.parse_args()
    od = out_dir(args.data, "part4")
    model, ck = MotionRNN.from_checkpoint(default_ckpt(args))
    T_train = ck["train_args"]["T"]
    max_occ = ck["train_args"]["max_occ"]
    out = dict(final_T=T_train, final_max_occ=max_occ)

    # ---- horizonte analítico
    trs, hs = long_tracklets(args)
    K = args.K
    curves = {}
    abl = os.path.join(ROOT, "checkpoints", "ablation", args.data)
    for cell, color in [("rnn", "#ff7f0e"), ("lstm", "#9467bd"), ("gru", "#d62728")]:
        path = os.path.join(abl, f"{cell}_T32_s0.pt")
        if os.path.exists(path):
            m, _ = MotionRNN.from_checkpoint(path)
            curves[f"{cell.upper()} (ablação, T=32)"] = (gradient_horizon(m, trs, hs, K), color, "-")
            if cell == "lstm":
                # ∂L/∂c no último passo é 0 (a saída lê só h): normaliza pelo h da própria LSTM em k=0
                href = curves[f"{cell.upper()} (ablação, T=32)"][0][0]
                curves["LSTM, célula c (ablação, T=32)"] = (gradient_horizon(m, trs, hs, K, part="c"), color, ":",
                                                            href)
    curves[f"GRU final (T={T_train})"] = (gradient_horizon(model, trs, hs, K), "k", "--")
    if args.fix_ckpt and os.path.exists(args.fix_ckpt):
        mf, ckf = MotionRNN.from_checkpoint(args.fix_ckpt)
        curves[f"GRU corrigido (T={ckf['train_args']['T']})"] = (gradient_horizon(mf, trs, hs, K), COLORS["rnn_fix"], "--")
    plot_horizon(curves, os.path.join(od, "fig_grad_horizon.png"), K,
                 "Horizonte analítico: norma do gradiente da perda do último passo\n"
                 "em relação ao estado k passos antes (média sobre 256 janelas)")
    out["grad_horizon"] = {k: v[0] for k, v in curves.items()}
    gfinal = curves[f"GRU final (T={T_train})"][0]
    ratio = lambda k: float(gfinal[0] / max(gfinal[min(k, K - 1)], 1e-30))
    for k in [4, 8, 16, 32]:
        print(f"queda da norma do gradiente do modelo final em {k} passos: {ratio(k):.1f}x")

    # ---- horizonte empírico
    bins = [2, 4, 8, 12, 16, 20, 24, 32, 48, 1000]
    if args.data == "synth":
        emp_seqs = long_occlusion_suite(2 if args.quick else 4)
    else:
        emp_seqs = test_sequences(args)
    emp = empirical(emp_seqs, model, args.data, bins)
    out["empirical"] = dict(curve=emp["curve"], bins=bins, total=emp["total"])
    emp_fix = emp_age = None
    if args.fix_ckpt and os.path.exists(args.fix_ckpt):
        # controle: só aumentar max_age, sem retreinar (isola o efeito do treino)
        emp_age = empirical(emp_seqs, model, args.data, bins, max_age=args.fix_max_age)
        emp_fix = empirical(emp_seqs, mf, args.data, bins, max_age=args.fix_max_age)
        out["empirical_age_only"] = dict(curve=emp_age["curve"], total=emp_age["total"])
        out["empirical_fix"] = dict(curve=emp_fix["curve"], total=emp_fix["total"])
        # e no teste padrão (garante que a correção não piora o caso comum)
        std = test_sequences(args)
        out["test_before"] = aggregate([evaluate_seq(s, run_tracker(s, "rnn", model, **tracker_params(args.data, "rnn")))
                                        for s in std])
        out["test_age_only"] = aggregate([evaluate_seq(s, run_tracker(
            s, "rnn", model, **tracker_params(args.data, "rnn", max_age=args.fix_max_age))) for s in std])
        out["test_after"] = aggregate([evaluate_seq(s, run_tracker(
            s, "rnn", mf, **tracker_params(args.data, "rnn", max_age=args.fix_max_age))) for s in std])
        for k in ["test_before", "test_age_only", "test_after"]:
            print(k, {x: round(out[k][x], 4) for x in ["IDF1", "IDSW", "occ_survival", "id_ratio"]})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(bins) - 1)
    hist = [((emp["dataset_durations"] >= lo) & (emp["dataset_durations"] < hi)).mean()
            for lo, hi in zip(bins[:-1], bins[1:])]
    ax.bar(xs, hist, color="#cccccc", label="distribuição de duração de oclusão (dataset)")
    ax.set_ylabel("fração das oclusões do dataset")
    ax2 = ax.twinx()
    ax2.plot(xs, emp["curve"], "o-", color=COLORS["rnn"], label=f"P(id sobrevive) — final (T={T_train})")
    if emp_fix:
        ax2.plot(xs, emp_age["curve"], "s--", color="#1f77b4",
                 label=f"só max_age={args.fix_max_age} (sem retreino)")
        ax2.plot(xs, emp_fix["curve"], "o-", color=COLORS["rnn_fix"],
                 label=f"corrigido: T={ckf['train_args']['T']}, buracos<={ckf['train_args']['max_occ']}, "
                       f"max_age={args.fix_max_age}")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("fração com identidade preservada")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lo}-{hi - 1}" if hi < 1000 else f"{lo}+" for lo, hi in zip(bins[:-1], bins[1:])])
    ax.set_xlabel("duração da oclusão (quadros)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    ax.set_title("Horizonte empírico: quantos quadros o estado sobrevive a uma oclusão", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fig_empirical_horizon.png"), dpi=110)
    plt.close(fig)
    print("sobrevivência por duração:", [round(c, 2) if c == c else None for c in emp["curve"]])
    if emp_fix:
        print("só max_age:", [round(c, 2) if c == c else None for c in emp_age["curve"]])
        print("corrigido:", [round(c, 2) if c == c else None for c in emp_fix["curve"]])
        print("IDF1 (suíte de oclusões longas) antes / só max_age / corrigido:", round(emp["total"]["IDF1"], 4),
              round(emp_age["total"]["IDF1"], 4), round(emp_fix["total"]["IDF1"], 4))

    # ---- galeria
    seqs = emp_seqs if args.data == "synth" else test_sequences(args)
    chosen, events = find_failures(seqs, model, args.data)
    out["failures"] = chosen
    out["n_switch_events"] = len(events)
    out["switch_kinds"] = {k: sum(e["kind"] == k for e in events) for k in {e["kind"] for e in events}}
    for i, ev in enumerate(chosen):
        s = seqs[ev["seq"]]
        if args.data == "synth":
            s = synth_sequence(SynthConfig(**s.meta["config"]), render=True, name=s.name)
        diag = (f"buraco de {ev['gap']} quadros sem casamento ({ev['occluded']} ocluído); "
                f"janela de BPTT {T_train}, buracos de treino <= {max_occ}; "
                f"norma do gradiente cai {ratio(ev['gap']):.0f}x em {min(ev['gap'], K - 1)} passos")
        ev["diagnosis"] = diag
        draw_failure(s, model, args.data, ev, os.path.join(od, f"fig_failure_{i + 1}.png"), diag)
        print(f"falha {i + 1}: {ev}")
    save_json(out, os.path.join(od, "part4.json"))


if __name__ == "__main__":
    main()
