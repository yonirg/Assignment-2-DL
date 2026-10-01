"""Métricas de rastreamento implementadas do zero (sem motmetrics / TrackEval).

Convenções
----------
* Trajetórias (gt ou predição): array (N, >=6) com colunas
  ``[frame, id, x1, y1, x2, y2, ...]`` (caixas em xyxy, pixels).
* Detecções: array (N, 6) ``[frame, x1, y1, x2, y2, score]``.
* Regiões ignoradas: array (N, 5) ``[frame, x1, y1, x2, y2]`` — predições que
  casam com elas (IoU >= limiar) são removidas antes da avaliação
  (equivalente ao pré-processamento oficial do MOTChallenge para distratores
  e, no sintético, para objetos quase totalmente ocluídos).

Funções principais
------------------
* :func:`idf1` — atribuição global um-para-um (Hungarian) entre identidades
  previstas e verdadeiras ao longo da sequência inteira.
* :func:`clear_mot` — casamento quadro a quadro estilo CLEAR-MOT (com
  preferência para manter a correspondência anterior), contando ID switches,
  fragmentações, FP, FN e MOTA.
* :func:`evaluate_tracking` — junta tudo + erro de contagem de identidades.
* :func:`average_precision` — AP@IoU (VOC, interpolação em todos os pontos)
  para medir a qualidade da fonte de detecções.
* :func:`nms` — NMS próprio (torchvision.ops.nms é proibido).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "iou_matrix", "nms", "filter_ignored", "idf1", "clear_mot",
    "evaluate_tracking", "average_precision", "occlusion_events",
]


# ----------------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre todas as caixas de ``a`` (N,4) e ``b`` (M,4), formato xyxy."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Non-maximum suppression guloso. Retorna os índices mantidos (ordem de score)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def _as2d(arr, ncol: int = 6) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr.reshape(0, ncol) if arr.size == 0 else arr


def _group(arr: np.ndarray) -> dict:
    """Agrupa as linhas de um array por quadro (coluna 0)."""
    out = defaultdict(list)
    if arr is None or len(arr) == 0:
        return {}
    arr = np.asarray(arr)
    frames = arr[:, 0].astype(int)
    order = np.argsort(frames, kind="stable")
    arr, frames = arr[order], frames[order]
    cuts = np.flatnonzero(np.diff(frames)) + 1
    for chunk in np.split(arr, cuts):
        out[int(chunk[0, 0])] = chunk
    return dict(out)


def filter_ignored(pred: np.ndarray, ignore: np.ndarray | None, thr: float = 0.5) -> np.ndarray:
    """Remove predições casadas (Hungarian, IoU>=thr) com regiões ignoradas."""
    if ignore is None or len(ignore) == 0 or len(pred) == 0:
        return pred
    ign = _group(ignore)
    keep = np.ones(len(pred), dtype=bool)
    pframes = pred[:, 0].astype(int)
    for f, ig in ign.items():
        idx = np.flatnonzero(pframes == f)
        if len(idx) == 0:
            continue
        iou = iou_matrix(pred[idx, 2:6], ig[:, 1:5])
        r, c = linear_sum_assignment(-iou)
        ok = iou[r, c] >= thr
        keep[idx[r[ok]]] = False
    return pred[keep]


# ----------------------------------------------------------------------------
# IDF1
# ----------------------------------------------------------------------------
def idf1(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> dict:
    """IDF1 / IDP / IDR (Ristani et al., 2016).

    Para cada par (id verdadeiro g, id previsto p) conta-se em quantos quadros
    as duas caixas se sobrepõem com IoU >= ``iou_thr``. Uma atribuição global
    um-para-um entre identidades (Hungarian, maximizando a soma) define IDTP;
    IDFN = #caixas gt - IDTP e IDFP = #caixas previstas - IDTP.
    """
    gt, pred = _as2d(gt), _as2d(pred)
    n_gt, n_pr = len(gt), len(pred)
    if n_gt == 0 or n_pr == 0:
        return dict(IDF1=float(n_gt == n_pr), IDP=float(n_pr == 0), IDR=float(n_gt == 0),
                    IDTP=0, IDFP=n_pr, IDFN=n_gt, id_map={})
    gids, ginv = np.unique(gt[:, 1].astype(int), return_inverse=True)
    pids, pinv = np.unique(pred[:, 1].astype(int), return_inverse=True)
    overlap = np.zeros((len(gids), len(pids)), dtype=np.int64)

    g_by_f, p_by_f = defaultdict(list), defaultdict(list)
    for i, f in enumerate(gt[:, 0].astype(int)):
        g_by_f[f].append(i)
    for i, f in enumerate(pred[:, 0].astype(int)):
        p_by_f[f].append(i)
    for f, gi in g_by_f.items():
        pi = p_by_f.get(f)
        if not pi:
            continue
        iou = iou_matrix(gt[gi, 2:6], pred[pi, 2:6])
        rr, cc = np.nonzero(iou >= iou_thr)
        np.add.at(overlap, (ginv[np.asarray(gi)[rr]], pinv[np.asarray(pi)[cc]]), 1)

    r, c = linear_sum_assignment(-overlap)
    idtp = int(overlap[r, c].sum())
    idfn, idfp = n_gt - idtp, n_pr - idtp
    id_map = {int(gids[a]): int(pids[b]) for a, b in zip(r, c) if overlap[a, b] > 0}
    return dict(
        IDF1=2 * idtp / (2 * idtp + idfp + idfn),
        IDP=idtp / (idtp + idfp), IDR=idtp / (idtp + idfn),
        IDTP=idtp, IDFP=idfp, IDFN=idfn, id_map=id_map,
    )


# ----------------------------------------------------------------------------
# CLEAR-MOT: ID switches, fragmentações, MOTA
# ----------------------------------------------------------------------------
def clear_mot(gt: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5, return_matches: bool = False) -> dict:
    """Casamento por quadro com continuidade (Bernardin & Stiefelhagen, 2008).

    * Em cada quadro, pares (g, p) que já estavam casados no quadro anterior e
      continuam com IoU >= limiar são mantidos; o resto é resolvido com
      Hungarian sobre IoU.
    * **ID switch**: o gt g é casado com um id previsto diferente do último id
      com que ele foi casado (em qualquer quadro anterior).
    * **Fragmentação**: a trajetória verdadeira estava sendo rastreada, deixou
      de ser (em um quadro em que o gt existe) e volta a ser rastreada depois.
      Conta-se no momento da retomada.
    """
    g_by_f, p_by_f = _group(_as2d(gt)), _group(_as2d(pred))
    frames = sorted(set(g_by_f) | set(p_by_f))
    prev_pair: dict[int, int] = {}      # g -> p no quadro anterior
    last_match: dict[int, int] = {}     # g -> último p casado (para IDSW)
    was_tracked: dict[int, bool] = {}   # status no último quadro em que g existiu
    interrupted: dict[int, bool] = {}   # g já foi rastreado e depois perdido
    idsw = frag = fp = fn = tp = 0
    n_gt_total = 0
    switches, matches = [], []
    for f in frames:
        G = g_by_f.get(f, np.zeros((0, 6)))
        P = p_by_f.get(f, np.zeros((0, 6)))
        gid = G[:, 1].astype(int)
        pid = P[:, 1].astype(int)
        n_gt_total += len(G)
        iou = iou_matrix(G[:, 2:6], P[:, 2:6]) if len(G) and len(P) else np.zeros((len(G), len(P)))
        pairs = []
        used_g, used_p = set(), set()
        # 1) continuidade
        pcol = {p: j for j, p in enumerate(pid)}
        for i, g in enumerate(gid):
            p = prev_pair.get(g)
            if p is not None and p in pcol and iou[i, pcol[p]] >= iou_thr and pcol[p] not in used_p:
                pairs.append((i, pcol[p]))
                used_g.add(i)
                used_p.add(pcol[p])
        # 2) Hungarian para o restante
        rg = [i for i in range(len(G)) if i not in used_g]
        rp = [j for j in range(len(P)) if j not in used_p]
        if rg and rp:
            sub = iou[np.ix_(rg, rp)]
            cost = np.where(sub >= iou_thr, 1.0 - sub, 1e6)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if sub[a, b] >= iou_thr:
                    pairs.append((rg[a], rp[b]))
        new_prev = {}
        matched_g = set()
        for i, j in pairs:
            g, p = int(gid[i]), int(pid[j])
            if g in last_match and last_match[g] != p:
                idsw += 1
                switches.append((f, g, last_match[g], p))
            if interrupted.get(g, False):
                frag += 1
                interrupted[g] = False
            last_match[g] = p
            new_prev[g] = p
            matched_g.add(g)
            if return_matches:
                matches.append((f, g, p))
        for g in gid:
            g = int(g)
            if g not in matched_g and was_tracked.get(g, False):
                interrupted[g] = True
            was_tracked[g] = g in matched_g
        prev_pair = new_prev
        tp += len(pairs)
        fp += len(P) - len(pairs)
        fn += len(G) - len(pairs)
    mota = 1.0 - (fp + fn + idsw) / max(n_gt_total, 1)
    out = dict(IDSW=idsw, Frag=frag, FP=fp, FN=fn, TP=tp, MOTA=mota,
               Recall=tp / max(n_gt_total, 1), switches=switches)
    if return_matches:
        out["matches"] = matches
    return out


# ----------------------------------------------------------------------------
# Agregado
# ----------------------------------------------------------------------------
def evaluate_tracking(gt: np.ndarray, pred: np.ndarray, ignore: np.ndarray | None = None,
                      iou_thr: float = 0.5) -> dict:
    """IDF1, IDSW, Frag, MOTA e erro de contagem de identidades únicas."""
    gt, pred = _as2d(gt), _as2d(pred)
    pred = filter_ignored(pred, ignore, iou_thr)
    res = idf1(gt, pred, iou_thr)
    cm = clear_mot(gt, pred, iou_thr)
    n_gt_ids = len(np.unique(gt[:, 1])) if len(gt) else 0
    n_pr_ids = len(np.unique(pred[:, 1])) if len(pred) else 0
    return dict(
        IDF1=res["IDF1"], IDP=res["IDP"], IDR=res["IDR"],
        IDTP=res["IDTP"], IDFP=res["IDFP"], IDFN=res["IDFN"],
        IDSW=cm["IDSW"], Frag=cm["Frag"], MOTA=cm["MOTA"], FP=cm["FP"], FN=cm["FN"],
        Recall=cm["Recall"],
        n_gt_ids=n_gt_ids, n_pred_ids=n_pr_ids,
        id_ratio=n_pr_ids / max(n_gt_ids, 1),
        count_err=n_pr_ids - n_gt_ids,
        count_err_rel=abs(n_pr_ids - n_gt_ids) / max(n_gt_ids, 1),
        IDSW_per_gt=cm["IDSW"] / max(n_gt_ids, 1),
    )


# ----------------------------------------------------------------------------
# Qualidade da fonte de detecção
# ----------------------------------------------------------------------------
def average_precision(gt: np.ndarray, dets: np.ndarray, ignore: np.ndarray | None = None,
                      iou_thr: float = 0.5) -> float:
    """AP (classe única) sobre todos os quadros de uma sequência.

    ``gt``: (N, >=6) [frame, id, x1, y1, x2, y2, ...]; ``dets``: (M, 6)
    [frame, x1, y1, x2, y2, score]. Detecções que casam com regiões ignoradas
    não contam nem como TP nem como FP.
    """
    gt = np.asarray(gt, dtype=np.float64)
    dets = np.asarray(dets, dtype=np.float64).reshape(-1, 6)
    if len(gt) == 0:
        return float("nan")
    if len(dets) == 0:
        return 0.0
    g_by_f = _group(gt)
    i_by_f = _group(ignore) if ignore is not None and len(ignore) else {}
    order = np.argsort(-dets[:, 5], kind="stable")
    dets = dets[order]
    used = {f: np.zeros(len(g), dtype=bool) for f, g in g_by_f.items()}
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for k, d in enumerate(dets):
        f = int(d[0])
        G = g_by_f.get(f)
        best, bj = 0.0, -1
        if G is not None:
            ious = iou_matrix(d[None, 1:5], G[:, 2:6])[0]
            ious[used[f]] = -1
            bj = int(np.argmax(ious))
            best = ious[bj]
        if best >= iou_thr:
            tp[k] = 1
            used[f][bj] = True
            continue
        I = i_by_f.get(f)
        if I is not None and iou_matrix(d[None, 1:5], I[:, 1:5]).max() >= iou_thr:
            continue  # cai numa região ignorada: não é FP
        fp[k] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / len(gt)
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ----------------------------------------------------------------------------
# Horizonte de memória empírico
# ----------------------------------------------------------------------------
def occlusion_events(gt_full: np.ndarray, vis_thr: float = 0.1, min_len: int = 1) -> list:
    """Encontra oclusões na gt completa (com coluna 6 = visibilidade).

    Retorna lista de (id, frame_ultimo_visivel, frame_reaparece, duracao).
    """
    events = []
    for g in np.unique(gt_full[:, 1]).astype(int):
        tr = gt_full[gt_full[:, 1] == g]
        tr = tr[np.argsort(tr[:, 0])]
        vis = tr[:, 6] >= vis_thr
        frames = tr[:, 0].astype(int)
        i = 0
        while i < len(tr):
            if vis[i]:
                i += 1
                continue
            j = i
            while j < len(tr) and not vis[j]:
                j += 1
            if i > 0 and j < len(tr) and (j - i) >= min_len:
                events.append((g, int(frames[i - 1]), int(frames[j]), int(frames[j] - frames[i - 1] - 1)))
            i = j
    return events


def occlusion_survival(gt_full: np.ndarray, pred: np.ndarray, vis_thr: float = 0.1,
                       iou_thr: float = 0.5, min_len: int = 1, window: int = 10) -> list:
    """Para cada oclusão da gt: a identidade sobreviveu?

    Sobreviveu = o último id previsto casado com o objeto antes do buraco
    (até ``window`` quadros antes) é o mesmo do primeiro casamento depois do
    reaparecimento (até ``window`` quadros depois).
    Retorna lista de (duracao, sobreviveu: bool | None); ``None`` quando o
    objeto não estava rastreado antes ou não foi reencontrado depois (evento
    não informativo sobre a memória).
    """
    events = occlusion_events(gt_full, vis_thr, min_len)
    cm = clear_mot(gt_full, pred, iou_thr, return_matches=True)
    m = {(f, g): p for f, g, p in cm["matches"]}
    out = []
    for g, f0, f1, dur in events:
        before = next((m[(f, g)] for f in range(f0, f0 - window - 1, -1) if (f, g) in m), None)
        after = next((m[(f, g)] for f in range(f1, f1 + window + 1) if (f, g) in m), None)
        if before is None or after is None:
            out.append((dur, None))
        else:
            out.append((dur, after == before))
    return out
