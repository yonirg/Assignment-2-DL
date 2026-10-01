"""Simulador de detector (Parte 0.2) e degradação de detecções (Parte 5).

``simulate_detections`` estraga a gt de propósito: descarta uma fração das
caixas, adiciona ruído nas coordenadas, injeta falsos positivos e duplicatas
(que o NMS próprio remove). ``degrade_detections`` aplica a mesma família de
estragos sobre detecções já existentes (ex.: det.txt público do MOT17).
"""
from __future__ import annotations

import numpy as np

from metrics import nms


def _jitter(boxes: np.ndarray, noise: float, rng) -> np.ndarray:
    """Ruído gaussiano proporcional ao tamanho da caixa (centro e escala)."""
    if noise <= 0 or len(boxes) == 0:
        return boxes.copy()
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 + rng.normal(0, noise, len(boxes)) * w
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 + rng.normal(0, noise, len(boxes)) * h
    w = w * np.exp(rng.normal(0, noise, len(boxes)))
    h = h * np.exp(rng.normal(0, noise, len(boxes)))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def _false_positives(frame_ids, rate, size_ref, img_wh, rng):
    W, H = img_wh
    rows = []
    for f in frame_ids:
        for _ in range(rng.poisson(rate)):
            w, h = size_ref[rng.integers(len(size_ref))]
            w, h = w * rng.uniform(0.7, 1.3), h * rng.uniform(0.7, 1.3)
            x = rng.uniform(0, max(W - w, 1))
            y = rng.uniform(0, max(H - h, 1))
            rows.append([f, x, y, x + w, y + h, rng.uniform(0.05, 0.7)])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 6)


def simulate_detections(gt_full: np.ndarray, drop_p: float = 0.1, noise: float = 0.05,
                        fp_rate: float = 0.3, dup_p: float = 0.0, min_vis: float = 0.3,
                        img_wh=(128, 128), n_frames: int | None = None, nms_thr: float = 0.6,
                        seed: int = 0) -> np.ndarray:
    """gt (N,7) [frame,id,x1,y1,x2,y2,vis] -> dets (M,6) [frame,x1,y1,x2,y2,score]."""
    rng = np.random.default_rng(seed)
    gt = gt_full[gt_full[:, 6] >= min_vis]
    keep = rng.random(len(gt)) >= drop_p
    gt = gt[keep]
    boxes = _jitter(gt[:, 2:6], noise, rng)
    # score maior quando o objeto está mais visível
    score = np.clip(0.55 + 0.4 * gt[:, 6] + rng.normal(0, 0.08, len(gt)), 0.05, 1.0)
    dets = np.column_stack([gt[:, 0], boxes, score])
    if dup_p > 0 and len(gt):
        d = rng.random(len(gt)) < dup_p
        dup = np.column_stack([gt[d, 0], _jitter(gt[d, 2:6], noise * 2 + 0.05, rng), score[d] * 0.8])
        dets = np.vstack([dets, dup])
    if n_frames is None:
        n_frames = int(gt_full[:, 0].max())
    sizes = np.column_stack([gt_full[:, 4] - gt_full[:, 2], gt_full[:, 5] - gt_full[:, 3]])
    dets = np.vstack([dets, _false_positives(range(1, n_frames + 1), fp_rate, sizes, img_wh, rng)])
    dets = _clip(dets, img_wh)
    return apply_nms(dets, nms_thr)


def degrade_detections(dets: np.ndarray, drop_p: float = 0.0, noise: float = 0.0,
                       fp_rate: float = 0.0, img_wh=(1920, 1080), seed: int = 0) -> np.ndarray:
    """Degrada detecções existentes (Parte 5 — qualidade do detector)."""
    rng = np.random.default_rng(seed)
    dets = dets[rng.random(len(dets)) >= drop_p]
    boxes = _jitter(dets[:, 1:5], noise, rng)
    out = np.column_stack([dets[:, 0], boxes, dets[:, 5]])
    if fp_rate > 0 and len(dets):
        sizes = np.column_stack([dets[:, 3] - dets[:, 1], dets[:, 4] - dets[:, 2]])
        fr = np.unique(dets[:, 0]).astype(int)
        fps = _false_positives(range(fr.min(), fr.max() + 1), fp_rate, sizes, img_wh, rng)
        # FP com score na mesma faixa das detecções reais (difícil de filtrar por limiar)
        if len(fps):
            fps[:, 5] = rng.choice(dets[:, 5], len(fps))
        out = np.vstack([out, fps])
    return _clip(out, img_wh)


def _clip(dets, img_wh):
    W, H = img_wh
    dets = dets.copy()
    dets[:, [1, 3]] = dets[:, [1, 3]].clip(0, W)
    dets[:, [2, 4]] = dets[:, [2, 4]].clip(0, H)
    ok = (dets[:, 3] - dets[:, 1] > 1) & (dets[:, 4] - dets[:, 2] > 1)
    return dets[ok]


def apply_nms(dets: np.ndarray, thr: float) -> np.ndarray:
    """NMS próprio aplicado quadro a quadro."""
    if len(dets) == 0:
        return dets
    out = []
    for f in np.unique(dets[:, 0]):
        d = dets[dets[:, 0] == f]
        out.append(d[nms(d[:, 1:5], d[:, 5], thr)])
    return np.vstack(out)
