"""Parte 0 — gerador sintético de vídeos com elipses e oclusão real.

Cada elipse tem profundidade (z) própria; o render é feito do fundo para a
frente (z-buffer por pintor), então uma elipse que passa atrás de outra — ou
atrás de um "poste" (oclusor estático, sempre na frente) — realmente some da
imagem. A visibilidade de cada objeto é medida no mapa de dono de cada pixel,
no mesmo formato do campo ``visibility`` do MOT17.

Botões expostos (``SynthConfig``): número de objetos, velocidade típica,
duração da oclusão (largura dos postes), ruído e contraste.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class SynthConfig:
    size: int = 128
    n_frames: tuple = (30, 60)
    n_objects: tuple = (5, 15)
    speed: float = 1.5            # px/quadro (média do módulo da velocidade)
    speed_jitter: float = 0.3     # variação relativa da velocidade entre objetos
    accel_std: float = 0.04       # ruído de aceleração (px/quadro^2)
    axes: tuple = (4.0, 9.0)      # semi-eixos das elipses (px)
    occlusion_len: int = 0        # quadros totalmente escondido atrás de um poste
    n_occluders: int = 1          # postes (só se occlusion_len > 0)
    noise: tuple = (0.02, 0.10)   # desvio do ruído gaussiano da imagem
    contrast: tuple = (0.25, 0.9) # |intensidade objeto - fundo|
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def _rng_range(rng, r, integer=False):
    if np.isscalar(r):
        return r
    lo, hi = r
    return int(rng.integers(lo, hi + 1)) if integer else float(rng.uniform(lo, hi))


def _ellipse_mask(xx, yy, cx, cy, a, b, theta):
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = xx - cx, yy - cy
    u = (c * dx + s * dy) / a
    v = (-s * dx + c * dy) / b
    return u * u + v * v <= 1.0


def _ellipse_bbox(cx, cy, a, b, theta):
    c, s = np.cos(theta), np.sin(theta)
    hw = np.sqrt((a * c) ** 2 + (b * s) ** 2)
    hh = np.sqrt((a * s) ** 2 + (b * c) ** 2)
    return cx - hw, cy - hh, cx + hw, cy + hh


def generate(cfg: SynthConfig, render: bool = True) -> dict:
    """Gera um vídeo sintético.

    Retorna dict com
      * ``frames``: (T, H, W) float32 em [0, 1] (se ``render``)
      * ``gt``: (N, 7) ``[frame, id, x1, y1, x2, y2, vis]`` (frames a partir de 1;
        caixa = extensão total da elipse, como o MOT17 anota pedestres ocluídos)
      * ``occluders``: lista de (x1, x2) dos postes
      * ``config``
    """
    rng = np.random.default_rng(cfg.seed)
    S = cfg.size
    T = _rng_range(rng, cfg.n_frames, integer=True)
    N = _rng_range(rng, cfg.n_objects, integer=True)

    # postes: largura tal que um objeto horizontal fica ~occlusion_len quadros escondido
    occluders = []
    if cfg.occlusion_len > 0:
        width = cfg.occlusion_len * cfg.speed + 2 * cfg.axes[1]
        width = min(width, S * 0.6)
        n_occ = max(1, cfg.n_occluders)
        for k in range(n_occ):
            cx = S * (k + 1) / (n_occ + 1) + rng.uniform(-4, 4)
            occluders.append((cx - width / 2, cx + width / 2))

    # estado inicial dos objetos
    a = rng.uniform(*cfg.axes, size=N)
    b = rng.uniform(*cfg.axes, size=N)
    theta = rng.uniform(0, np.pi, size=N)
    pos = np.stack([rng.uniform(10, S - 10, N), rng.uniform(10, S - 10, N)], 1)
    spd = cfg.speed * np.clip(1 + cfg.speed_jitter * rng.standard_normal(N), 0.2, None)
    if occluders:  # movimento predominantemente horizontal para cruzar os postes
        ang = rng.uniform(-np.pi / 5, np.pi / 5, N) + np.pi * rng.integers(0, 2, N)
    else:
        ang = rng.uniform(0, 2 * np.pi, N)
    vel = np.stack([np.cos(ang), np.sin(ang)], 1) * spd[:, None]
    depth = rng.permutation(N)  # maior = mais perto
    bg = rng.uniform(0.3, 0.7)
    contrast = rng.uniform(*cfg.contrast, size=N) * rng.choice([-1, 1], size=N)
    intensity = np.clip(bg + contrast, 0.0, 1.0)
    noise_std = _rng_range(rng, cfg.noise)
    occ_int = np.clip(bg + rng.choice([-1, 1]) * 0.45, 0, 1)

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) + 0.5
    occ_mask = np.zeros((S, S), dtype=bool)
    for x1, x2 in occluders:
        occ_mask |= (xx >= x1) & (xx <= x2)

    frames = np.zeros((T, S, S), dtype=np.float32) if render else None
    img_rng = np.random.default_rng(cfg.seed + 7919)  # ruído de imagem não altera a gt
    gt_rows = []
    draw_order = np.argsort(depth)  # do fundo para a frente
    for t in range(T):
        owner = -np.ones((S, S), dtype=np.int32)
        masks = []
        for i in range(N):
            masks.append(_ellipse_mask(xx, yy, pos[i, 0], pos[i, 1], a[i], b[i], theta[i]))
        for i in draw_order:
            owner[masks[i]] = i
        owner[occ_mask] = -2  # poste sempre na frente
        if render:
            img = np.full((S, S), bg, dtype=np.float32)
            for i in range(N):
                img[owner == i] = intensity[i]
            img[occ_mask] = occ_int + 0.05 * np.sin(yy[occ_mask] / 3.0)
            img += img_rng.normal(0, noise_std, img.shape).astype(np.float32)
            frames[t] = np.clip(img, 0, 1)
        for i in range(N):
            area = masks[i].sum()
            vis = float((owner == i).sum() / area) if area > 0 else 0.0
            x1, y1, x2, y2 = _ellipse_bbox(pos[i, 0], pos[i, 1], a[i], b[i], theta[i])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, S), min(y2, S)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            gt_rows.append([t + 1, i + 1, x1, y1, x2, y2, vis])
        # dinâmica: velocidade quase constante + reflexão nas bordas
        vel += rng.normal(0, cfg.accel_std, vel.shape)
        sp = np.linalg.norm(vel, axis=1, keepdims=True)
        vel = vel / np.maximum(sp, 1e-6) * np.clip(sp, 0.5 * spd[:, None], 1.5 * spd[:, None])
        pos += vel
        for d in range(2):
            lo = pos[:, d] < 6
            hi = pos[:, d] > S - 6
            vel[lo | hi, d] *= -1
            pos[:, d] = np.clip(pos[:, d], 6, S - 6)
    return dict(frames=frames, gt=np.asarray(gt_rows, dtype=np.float64),
                occluders=occluders, config=cfg.to_dict(), size=(S, S))


def split_gt(gt_full: np.ndarray, min_vis: float = 0.2):
    """Separa a gt em (avaliada, ignorada) pelo limiar de visibilidade."""
    vis = gt_full[:, 6] >= min_vis
    gt_eval = gt_full[vis]
    ignore = gt_full[~vis][:, [0, 2, 3, 4, 5]]
    return gt_eval, ignore
