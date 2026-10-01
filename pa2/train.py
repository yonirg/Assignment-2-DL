"""Treino da MotionRNN (Trilha A) em trajetórias da ground truth.

Cada exemplo é uma janela de T+1 quadros de uma trajetória verdadeira. A
recorrência é desenrolada por T passos (BPTT truncado em T). Em cada passo a
entrada é:
  * a caixa observada (gt + ruído de detector), com flag observado=1; ou
  * a própria previsão do passo anterior (destacada do grafo), com observado=0,
    quando o passo cai em uma oclusão (visibilidade baixa na gt, buraco
    sintético) ou é sorteado pelo *scheduled sampling*.
Perda: smooth-L1 sobre a caixa do próximo quadro (μ) + NLL gaussiana para a
log-variância (com μ destacado) — ``--loss nll`` usa a NLL completa.

Regimes (Parte 3, Eixo 2):
  teacher   — sempre a observação (sem oclusões simuladas);
  scheduled — prob. de alimentar a própria previsão cresce de 0 a ``eps_max``;
  free      — prob. ``eps_max`` desde o início + oclusões simuladas.

Exemplos:
  python -m pa2.train --data synth --out checkpoints/synth_gru.pt
  python -m pa2.train --data mot17 --mot-root data/MOT17 --out checkpoints/mot17_gru.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import gt_tracklets, mot17_sequences, synth_suite
from .models import MotionRNN, decode, encode_inputs, encode_target, hidden_for_budget, n_params


class WindowSampler:
    def __init__(self, tracklets, img_hs, T, rng):
        self.T = T
        self.rng = rng
        ok = [k for k, tr in enumerate(tracklets) if len(tr) >= T + 1]
        if not ok:
            raise ValueError(f"nenhuma trajetória com >= {T + 1} quadros")
        self.tracks = [tracklets[k] for k in ok]
        self.img_h = np.array([img_hs[k] for k in ok], dtype=np.float32)
        lens = np.array([len(t) - T for t in self.tracks], dtype=np.float64)
        self.p = lens / lens.sum()   # cada janela possível tem a mesma probabilidade

    def sample(self, B):
        idx = self.rng.choice(len(self.tracks), size=B, p=self.p)
        out = np.zeros((B, self.T + 1, 5), dtype=np.float32)
        for b, k in enumerate(idx):
            tr = self.tracks[k]
            s = self.rng.integers(0, len(tr) - self.T)
            out[b] = tr[s:s + self.T + 1, 1:6]
        return torch.from_numpy(out), torch.from_numpy(self.img_h[idx])


def jitter(boxes, noise, gen):
    if noise <= 0:
        return boxes
    w, h = boxes[..., 2], boxes[..., 3]
    n = torch.randn(boxes.shape, generator=gen) * noise
    return torch.stack([boxes[..., 0] + n[..., 0] * w, boxes[..., 1] + n[..., 1] * h,
                        boxes[..., 2] * torch.exp(n[..., 2]), boxes[..., 3] * torch.exp(n[..., 3])], -1)


def observation_mask(vis, regime, eps, args, gen):
    """(B,T) bool — True se o passo recebe observação."""
    B, T = vis.shape
    if regime == "teacher":
        return torch.ones(B, T, dtype=torch.bool)
    m = vis >= args.min_vis
    m &= torch.rand(B, T, generator=gen) >= args.drop_p
    # oclusões sintéticas (buracos contíguos)
    occ = torch.rand(B, generator=gen) < args.p_occ
    if occ.any() and T > 2:
        L = torch.randint(1, max(2, min(args.max_occ, T - 1)) + 1, (B,), generator=gen)
        s = (torch.rand(B, generator=gen) * (T - 1)).long() + 1
        ar = torch.arange(T)[None]
        hole = (ar >= s[:, None]) & (ar < (s + L)[:, None]) & occ[:, None]
        m &= ~hole
    if eps > 0:
        m &= torch.rand(B, T, generator=gen) >= eps
    m[:, 0] = True   # a track nasce de uma observação
    return m


def unroll(model, boxes, img_h, mask, noise, gen, keep_states=False):
    """Desenrola a recorrência numa janela. Retorna μ, logσ², alvos (B,T,4)."""
    B, T1, _ = boxes.shape
    T = T1 - 1
    gtb = boxes[..., :4]
    obs = jitter(gtb, noise, gen)
    score = (0.6 + 0.4 * boxes[..., 4].clamp(0, 1))
    state = model.init_state(B)
    prev_in = obs[:, 0]
    pred = obs[:, 0]
    mus, lvs, tgts, states = [], [], [], []
    for t in range(T):
        m = mask[:, t]
        inp = torch.where(m[:, None], obs[:, t], pred.detach())
        x = encode_inputs(inp, prev_in, m, torch.where(m, score[:, t], torch.zeros_like(score[:, t])), img_h)
        mu, lv, state = model.step(x, state)
        if keep_states:
            states.append(state)
        tgts.append(encode_target(gtb[:, t + 1], inp))
        mus.append(mu)
        lvs.append(lv)
        pred = decode(mu, inp)
        prev_in = inp
    out = (torch.stack(mus, 1), torch.stack(lvs, 1), torch.stack(tgts, 1))
    return (*out, states) if keep_states else out


def loss_fn(mu, lv, tgt, kind="l1"):
    if kind == "nll":
        return (0.5 * (lv + (tgt - mu) ** 2 / lv.exp())).mean()
    l1 = F.smooth_l1_loss(mu, tgt, beta=0.5)
    nll = (0.5 * (lv + (tgt - mu.detach()) ** 2 / lv.exp())).mean()
    return l1 + 0.1 * nll


def load_tracklets(args):
    if args.data == "synth":
        seqs = synth_suite(args.n_synth, seed0=args.synth_seed, render=False,
                           n_frames=(40, 70), occlusion_len=0)
        seqs += synth_suite(args.n_synth // 2, seed0=args.synth_seed + 50_000, render=False,
                            n_frames=(40, 70), occlusion_len=8, speed=2.0)
    else:
        seqs = mot17_sequences(args.mot_root, "train", args.detector)
    trs, hs = [], []
    for s in seqs:
        for tr in gt_tracklets(s, min_len=2):
            trs.append(tr)
            hs.append(s.height)
    return trs, hs


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    trs, hs = load_tracklets(args)
    sampler = WindowSampler(trs, hs, args.T, rng)
    hidden = args.hidden or hidden_for_budget(args.cell, args.budget)
    model = MotionRNN(args.cell, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        frac = it / max(args.iters - 1, 1)
        if args.regime == "scheduled":
            eps = args.eps_max * min(1.0, frac / 0.6)
        elif args.regime == "free":
            eps = args.eps_max
        else:
            eps = 0.0
        boxes, img_h = sampler.sample(args.batch)
        mask = observation_mask(boxes[:, :-1, 4], args.regime, eps, args, gen)
        mu, lv, tgt = unroll(model, boxes, img_h, mask, args.noise, gen)
        loss = loss_fn(mu, lv, tgt, args.loss)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip if args.clip > 0 else float("inf"))
        opt.step()
        sched.step()
        if not torch.isfinite(loss):
            print(f"[it {it}] perda não finita — abortando")
            break
        hist.append(dict(it=it, loss=loss.item(), grad_norm=gnorm.item(), eps=eps))
        if it % args.log_every == 0 or it == args.iters - 1:
            print(f"it {it:5d} loss {loss.item():.4f} |g| {gnorm.item():.3f} eps {eps:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), model_config=model.config(),
                    train_args=vars(args), history=hist,
                    n_params=n_params(model), n_params_cell=n_params(model, True)), args.out)
    print(f"salvo em {args.out} ({n_params(model)} parâmetros, célula {n_params(model, True)})")
    return model, hist


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["synth", "mot17"], default="synth")
    p.add_argument("--mot-root", default="data/MOT17")
    p.add_argument("--detector", default="FRCNN")
    p.add_argument("--n-synth", type=int, default=120)
    p.add_argument("--synth-seed", type=int, default=1000)
    p.add_argument("--cell", choices=["rnn", "gru", "lstm"], default="gru")
    p.add_argument("--hidden", type=int, default=0, help="0 = derivar de --budget")
    p.add_argument("--budget", type=int, default=13000, help="parâmetros da célula recorrente")
    p.add_argument("--T", type=int, default=16, help="janela de BPTT truncado")
    p.add_argument("--regime", choices=["teacher", "scheduled", "free"], default="scheduled")
    p.add_argument("--eps-max", type=float, default=0.3)
    p.add_argument("--p-occ", type=float, default=0.5)
    p.add_argument("--max-occ", type=int, default=12)
    p.add_argument("--drop-p", type=float, default=0.05)
    p.add_argument("--min-vis", type=float, default=0.2)
    p.add_argument("--noise", type=float, default=0.04)
    p.add_argument("--loss", choices=["l1", "nll"], default="l1")
    p.add_argument("--clip", type=float, default=1.0, help="<=0 desliga o gradient clipping")
    p.add_argument("--iters", type=int, default=2500)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--out", default="checkpoints/synth_gru.pt")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
