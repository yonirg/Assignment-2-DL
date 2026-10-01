"""Sequências (MOT17 ou sintéticas) em um formato único + split por sequência.

``Sequence`` carrega apenas anotações e detecções; as imagens só são lidas
quando necessário (detector do torchvision, vídeos, figuras). Isso permite
rodar quase o PA inteiro só com o pacote de anotações do MOT17 (~10 MB).
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field

import numpy as np

from . import synth
from .detsim import simulate_detections, apply_nms

# ---------------------------------------------------------------------------
# Split do MOT17 (por sequência, nunca por quadro)
#   treino : 02 (estática, média dens.), 04 (estática, alta dens., vista do alto),
#            05 (móvel, 14 fps, baixa resolução), 11 (móvel, shopping)
#   val    : 09 (estática, baixa densidade, ambiente interno/porta)
#   teste  : 10 (móvel, noite, densa)  13 (câmera em veículo, ego-motion forte)
# O teste nunca é visto no treino e cobre os dois regimes mais difíceis para um
# modelo de movimento (câmera móvel); a validação escolhe hiperparâmetros.
# ---------------------------------------------------------------------------
MOT17_SPLIT = {
    "train": ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-11"],
    "val": ["MOT17-09"],
    "test": ["MOT17-10", "MOT17-13"],
}
MOT17_CAMERA = {"MOT17-02": "static", "MOT17-04": "static", "MOT17-05": "moving",
                "MOT17-09": "static", "MOT17-10": "moving", "MOT17-11": "moving",
                "MOT17-13": "moving"}
# classes do gt.txt: 1 pedestre (avaliado); 2 pessoa em veículo, 7 pessoa
# parada, 8 distrator, 12 reflexo -> regiões ignoradas
MOT17_IGNORE_CLASSES = (2, 7, 8, 12)


@dataclass
class Sequence:
    name: str
    gt_full: np.ndarray          # (N,7) [frame,id,x1,y1,x2,y2,vis] — objetos avaliáveis
    ignore: np.ndarray           # (M,5) [frame,x1,y1,x2,y2]
    dets: np.ndarray             # (K,6) [frame,x1,y1,x2,y2,score]
    width: int
    height: int
    n_frames: int
    fps: float = 30.0
    img_dir: str | None = None
    frames: np.ndarray | None = None   # vídeo em memória (sintético)
    min_vis: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def gt(self) -> np.ndarray:
        """gt avaliada (visibilidade >= min_vis)."""
        return self.gt_full[self.gt_full[:, 6] >= self.min_vis]

    @property
    def ignore_all(self) -> np.ndarray:
        low = self.gt_full[self.gt_full[:, 6] < self.min_vis][:, [0, 2, 3, 4, 5]]
        return np.vstack([self.ignore.reshape(-1, 5), low])

    @property
    def density(self) -> float:
        return len(self.gt) / max(self.n_frames, 1)

    def dets_by_frame(self, score_thr: float = -np.inf) -> dict:
        d = self.dets[self.dets[:, 5] >= score_thr]
        out = {f: np.zeros((0, 5)) for f in range(1, self.n_frames + 1)}
        for f in np.unique(d[:, 0]).astype(int):
            out[f] = d[d[:, 0] == f][:, 1:6]
        return out

    def image(self, frame: int) -> np.ndarray:
        """Quadro como uint8 RGB (H,W,3)."""
        if self.frames is not None:
            g = (self.frames[frame - 1] * 255).astype(np.uint8)
            return np.repeat(g[..., None], 3, axis=2)
        import cv2
        path = os.path.join(self.img_dir, f"{frame:06d}.jpg")
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img[..., ::-1].copy()


# ---------------------------------------------------------------------------
# MOT17
# ---------------------------------------------------------------------------
def _ltwh_to_xyxy(a):
    a = a.copy()
    a[:, 2] += a[:, 0]
    a[:, 3] += a[:, 1]
    return a


def read_seqinfo(seq_dir: str) -> dict:
    cp = configparser.ConfigParser()
    cp.read(os.path.join(seq_dir, "seqinfo.ini"))
    s = cp["Sequence"]
    return dict(name=s.get("name"), width=int(s.get("imWidth")), height=int(s.get("imHeight")),
                n_frames=int(s.get("seqLength")), fps=float(s.get("frameRate")),
                img_dir=os.path.join(seq_dir, s.get("imDir", "img1")))


def load_mot_txt(path: str) -> np.ndarray:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 10))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def load_mot_sequence(seq_dir: str, det_file: str | None = None, min_vis: float = 0.0,
                      det_nms: float | None = None) -> Sequence:
    """Carrega uma sequência no formato MOTChallenge.

    ``det_file``: caminho alternativo para as detecções (ex.: as do torchvision
    geradas por ``python -m pa2.detect``); padrão ``det/det.txt``.
    """
    info = read_seqinfo(seq_dir)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    gt_full = np.zeros((0, 7))
    ignore = np.zeros((0, 5))
    if os.path.exists(gt_path):
        raw = load_mot_txt(gt_path)
        boxes = _ltwh_to_xyxy(raw[:, 2:6])
        cls = raw[:, 7].astype(int)
        conf = raw[:, 6]
        valid = (cls == 1) & (conf > 0)
        ign = np.isin(cls, MOT17_IGNORE_CLASSES) | ((cls == 1) & (conf == 0))
        gt_full = np.column_stack([raw[valid, 0], raw[valid, 1], boxes[valid], raw[valid, 8]])
        ignore = np.column_stack([raw[ign, 0], boxes[ign]])
    det_path = det_file or os.path.join(seq_dir, "det", "det.txt")
    raw = load_mot_txt(det_path)
    dets = np.zeros((0, 6))
    if len(raw):
        dets = np.column_stack([raw[:, 0], _ltwh_to_xyxy(raw[:, 2:6]), raw[:, 6]])
        if det_nms is not None:
            dets = apply_nms(dets, det_nms)
    name = os.path.basename(os.path.normpath(seq_dir))
    base = "-".join(name.split("-")[:2])
    return Sequence(name=name, gt_full=gt_full, ignore=ignore, dets=dets,
                    width=info["width"], height=info["height"], n_frames=info["n_frames"],
                    fps=info["fps"], img_dir=info["img_dir"], min_vis=min_vis,
                    meta=dict(camera=MOT17_CAMERA.get(base, "?"), base=base,
                              detector=name.split("-")[-1] if name.count("-") >= 2 else "?"))


def mot17_sequences(root: str, split: str = "train", detector: str = "FRCNN", **kw) -> list:
    """Sequências do MOT17/train de um split (ver ``MOT17_SPLIT``)."""
    names = MOT17_SPLIT[split] if split in MOT17_SPLIT else [split]
    out = []
    for n in names:
        d = os.path.join(root, "train", f"{n}-{detector}")
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{d} não encontrado — veja README (download do MOT17).")
        out.append(load_mot_sequence(d, **kw))
    return out


# ---------------------------------------------------------------------------
# Sintético
# ---------------------------------------------------------------------------
DET_DEFAULT = dict(drop_p=0.1, noise=0.04, fp_rate=0.3, dup_p=0.1, min_vis=0.35)


def synth_sequence(cfg: synth.SynthConfig, det_kw: dict | None = None, min_vis: float = 0.2,
                   render: bool = False, name: str | None = None) -> Sequence:
    v = synth.generate(cfg, render=render)
    gt = v["gt"]
    T = int(gt[:, 0].max())
    kw = dict(DET_DEFAULT)
    kw.update(det_kw or {})
    dets = simulate_detections(gt, img_wh=v["size"], n_frames=T, seed=cfg.seed + 10_000, **kw)
    return Sequence(name=name or f"synth-{cfg.seed}", gt_full=gt, ignore=np.zeros((0, 5)),
                    dets=dets, width=cfg.size, height=cfg.size, n_frames=T, fps=30.0,
                    frames=v["frames"], min_vis=min_vis,
                    meta=dict(config=cfg.to_dict(), occluders=v["occluders"], det_kw=kw))


def synth_suite(n: int, seed0: int, render: bool = False, det_kw=None, **cfg_kw) -> list:
    return [synth_sequence(synth.SynthConfig(seed=seed0 + i, **cfg_kw), det_kw=det_kw, render=render)
            for i in range(n)]


def export_mot(seq: Sequence, out_dir: str, write_images: bool = True):
    """Grava uma sequência (ex.: sintética) no formato MOTChallenge."""
    import cv2
    os.makedirs(os.path.join(out_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "det"), exist_ok=True)
    with open(os.path.join(out_dir, "seqinfo.ini"), "w") as f:
        f.write(f"[Sequence]\nname={seq.name}\nimDir=img1\nframeRate={seq.fps:g}\n"
                f"seqLength={seq.n_frames}\nimWidth={seq.width}\nimHeight={seq.height}\nimExt=.jpg\n")
    g = seq.gt_full
    with open(os.path.join(out_dir, "gt", "gt.txt"), "w") as f:
        for r in g:
            f.write(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]-r[2]:.2f},{r[5]-r[3]:.2f},1,1,{r[6]:.3f}\n")
    with open(os.path.join(out_dir, "det", "det.txt"), "w") as f:
        for r in seq.dets:
            f.write(f"{int(r[0])},-1,{r[1]:.2f},{r[2]:.2f},{r[3]-r[1]:.2f},{r[4]-r[2]:.2f},{r[5]:.4f},-1,-1,-1\n")
    if write_images:
        os.makedirs(os.path.join(out_dir, "img1"), exist_ok=True)
        for t in range(1, seq.n_frames + 1):
            cv2.imwrite(os.path.join(out_dir, "img1", f"{t:06d}.jpg"), seq.image(t)[..., ::-1])


# ---------------------------------------------------------------------------
# Trajetórias da gt para treinar o modelo de movimento
# ---------------------------------------------------------------------------
def gt_tracklets(seq: Sequence, min_len: int = 2) -> list:
    """Lista de arrays (L, 6) [frame, cx, cy, w, h, vis] com quadros consecutivos."""
    out = []
    g = seq.gt_full
    for tid in np.unique(g[:, 1]):
        tr = g[g[:, 1] == tid]
        tr = tr[np.argsort(tr[:, 0])]
        cx = (tr[:, 2] + tr[:, 4]) / 2
        cy = (tr[:, 3] + tr[:, 5]) / 2
        w = tr[:, 4] - tr[:, 2]
        h = tr[:, 5] - tr[:, 3]
        arr = np.column_stack([tr[:, 0], cx, cy, w, h, tr[:, 6]])
        cuts = np.flatnonzero(np.diff(tr[:, 0]) != 1) + 1
        for seg in np.split(arr, cuts):
            if len(seg) >= min_len and (seg[:, 3:5] > 1).all():
                out.append(seg)
    return out
