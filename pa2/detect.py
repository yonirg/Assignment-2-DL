"""Detector pré-treinado do torchvision (COCO, classe person) em modo inferência.

O NMS final é o nosso (``metrics.nms``): o NMS interno do ROI head é
desligado (``box_nms_thresh=1.0``, que não suprime nada) e aplicamos o próprio
por cima. (O RPN do torchvision ainda usa NMS internamente para gerar as
propostas — isso é parte do detector congelado, não da nossa pós-filtragem.)

Grava no formato det.txt do MOTChallenge:
  python -m pa2.detect --seq data/MOT17/train/MOT17-02-FRCNN --out det/det_tv.txt
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from metrics import nms

PERSON = 1  # rótulo COCO no torchvision


def load_detector(device="cpu"):
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights,
                                              fasterrcnn_resnet50_fpn_v2)
    w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.05, box_nms_thresh=1.0,
                                       box_detections_per_img=300)
    return model.eval().to(device), w.transforms()


@torch.no_grad()
def detect_images(paths, model, tf, device="cpu", nms_thr=0.5, score_thr=0.1, batch=4):
    from torchvision.io import read_image
    rows = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        imgs = [tf(read_image(p)).to(device) for p in chunk]
        outs = model(imgs)
        for p, o in zip(chunk, outs):
            frame = int(os.path.splitext(os.path.basename(p))[0])
            keep = (o["labels"] == PERSON) & (o["scores"] >= score_thr)
            b = o["boxes"][keep].cpu().numpy()
            s = o["scores"][keep].cpu().numpy()
            k = nms(b, s, nms_thr)
            for bb, ss in zip(b[k], s[k]):
                rows.append([frame, *bb, ss])
    return np.asarray(rows).reshape(-1, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="pasta da sequência (com img1/)")
    ap.add_argument("--out", default="det/det_tv.txt", help="relativo à pasta da sequência")
    ap.add_argument("--nms", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.seq, "img1", "*.jpg")))
    if a.max_frames:
        paths = paths[: a.max_frames]
    model, tf = load_detector(a.device)
    d = detect_images(paths, model, tf, a.device, a.nms)
    out = os.path.join(a.seq, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for r in d:
            f.write(f"{int(r[0])},-1,{r[1]:.2f},{r[2]:.2f},{r[3]-r[1]:.2f},{r[4]-r[2]:.2f},{r[5]:.4f},-1,-1,-1\n")
    print(f"{len(d)} detecções em {len(paths)} quadros -> {out}")


if __name__ == "__main__":
    main()
