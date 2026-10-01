"""Associação e gestão de tracks (autoria própria — nenhum rastreador pronto).

Regras comuns a todos os rastreadores (documentadas no README):
  * detecções com score < ``det_thr`` são descartadas;
  * custo = 1 − IoU(caixa prevista da track, detecção); pares com
    IoU < ``iou_thr`` são proibidos, a menos que passem no portão de
    Mahalanobis (``maha_gate``, só para modelos com incerteza);
  * atribuição Hungarian (ou gulosa, ``matcher='greedy'``);
  * **nascimento**: detecção não casada com score >= ``new_thr`` abre uma
    track tentativa; ela só é emitida após ``min_hits`` observações (os
    quadros tentativos são emitidos retroativamente na confirmação);
  * associação em dois estágios: primeiro as tracks confirmadas, depois as
    tentativas com as detecções que sobraram;
  * **morte**: track confirmada removida após ``max_age`` quadros seguidos sem
    observação; track tentativa morre na primeira falha;
  * saída: a caixa da detecção casada (o detector é congelado). Com
    ``fill_gaps=True`` os quadros perdidos de uma track que é reencontrada são
    preenchidos com as caixas previstas pelo modelo durante o buraco.

Rastreadores:
  * ``IoUTracker``     — Parte 1: IoU entre a última caixa observada e a detecção.
  * ``KalmanTracker``  — baseline de comparação (velocidade constante).
  * ``RNNTracker``     — Parte 2 Trilha A: o estado de cada track é o estado
                          recorrente da ``MotionRNN``.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from metrics import iou_matrix
from .models import encode_inputs, encode_target, decode

CHI2_4_99 = 13.28


def xyxy_to_cxcywh(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                            b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])


def cxcywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                            b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2])


def associate(pred_xyxy, det_xyxy, iou_thr=0.3, d2=None, gate=None, matcher="hungarian"):
    """Retorna (matches [(i_track, j_det)], tracks_livres, dets_livres)."""
    nt, nd = len(pred_xyxy), len(det_xyxy)
    if nt == 0 or nd == 0:
        return [], list(range(nt)), list(range(nd))
    iou = iou_matrix(pred_xyxy, det_xyxy)
    valid = iou >= iou_thr
    cost = 1.0 - iou
    if d2 is not None and gate is not None:
        extra = (~valid) & (d2 <= gate)
        cost = np.where(extra, 1.0 + d2 / gate, cost)
        valid |= extra
    cost = np.where(valid, cost, 1e6)
    pairs = []
    if matcher == "hungarian":
        r, c = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(r, c) if valid[i, j]]
    else:  # guloso: menor custo primeiro
        order = np.argsort(cost, axis=None)
        ut, ud = set(), set()
        for k in order:
            i, j = divmod(int(k), nd)
            if not valid[i, j]:
                break
            if i in ut or j in ud:
                continue
            pairs.append((i, j))
            ut.add(i)
            ud.add(j)
    mt = {i for i, _ in pairs}
    md = {j for _, j in pairs}
    return pairs, [i for i in range(nt) if i not in mt], [j for j in range(nd) if j not in md]


class Track:
    def __init__(self, tid, box_xyxy, score):
        self.id = tid
        self.hits = 1
        self.miss = 0
        self.age = 1
        self.box = np.asarray(box_xyxy, dtype=np.float64)   # última caixa observada (xyxy)
        self.score = score
        self.pending = []      # saídas antes da confirmação
        self.gap = []          # (frame, caixa prevista) durante o buraco
        self.pred = self.box.copy()   # caixa prevista para o quadro corrente (xyxy)
        self.d = {}            # estado específico do rastreador


class BaseTracker:
    name = "base"

    def __init__(self, iou_thr=0.3, max_age=10, min_hits=2, det_thr=0.0, new_thr=None,
                 matcher="hungarian", maha_gate=None, fill_gaps=False, record=False):
        self.record = record
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
        self.det_thr = det_thr
        self.new_thr = det_thr if new_thr is None else new_thr
        self.matcher, self.maha_gate, self.fill_gaps = matcher, maha_gate, fill_gaps
        self.reset()

    def reset(self):
        self.tracks: list[Track] = []
        self.next_id = 1
        self.out = []
        self.log = []   # diagnósticos opcionais por quadro

    # ganchos -------------------------------------------------------------
    def predict(self, frame):            # preenche t.pred (xyxy) para o quadro
        for t in self.tracks:
            t.pred = t.box.copy()

    def maha_d2(self, dets_xyxy):        # (n_tracks, n_dets) ou None
        return None

    def sigma(self, t):                  # desvio previsto do centro (px), se houver
        return (np.nan, np.nan)

    def on_match(self, t: Track, det):
        t.box = det[:4].copy()

    def on_miss(self, t: Track):
        pass

    def on_new(self, t: Track, det):
        pass

    def post_step(self, frame):
        pass

    # laço principal -------------------------------------------------------
    def step(self, frame: int, dets: np.ndarray):
        dets = np.asarray(dets, dtype=np.float64).reshape(-1, 5)
        dets = dets[dets[:, 4] >= self.det_thr]
        self.predict(frame)
        P = np.array([t.pred for t in self.tracks]).reshape(-1, 4)
        d2 = self.maha_d2(dets[:, :4]) if self.maha_gate else None
        # estágio 1: tracks confirmadas; estágio 2: tentativas com o que sobrou
        # (evita que uma track recém-nascida de uma duplicata roube a detecção)
        conf = [i for i, t in enumerate(self.tracks) if t.hits >= self.min_hits]
        tent = [i for i, t in enumerate(self.tracks) if t.hits < self.min_hits]
        pairs, ud = [], list(range(len(dets)))
        ut = []
        for group in (conf, tent):
            if not group:
                continue
            sub_d2 = None if d2 is None else d2[np.ix_(group, ud)]
            pr, u_t, u_d = associate(P[group], dets[ud, :4], self.iou_thr, sub_d2,
                                     self.maha_gate, self.matcher)
            pairs += [(group[a], ud[b]) for a, b in pr]
            ut += [group[a] for a in u_t]
            ud = [ud[b] for b in u_d]
        emitted = []
        for i, j in pairs:
            t = self.tracks[i]
            if self.fill_gaps and t.gap and t.hits >= self.min_hits:
                emitted.extend((f, t.id, b) for f, b in t.gap)
            t.gap = []
            self.on_match(t, dets[j])
            t.hits += 1
            t.miss = 0
            t.score = dets[j, 4]
            row = (frame, t.id, dets[j, :4].copy())
            if t.hits >= self.min_hits:
                emitted.extend(t.pending)
                t.pending = []
                emitted.append(row)
            else:
                t.pending.append(row)
        for i in ut:
            t = self.tracks[i]
            t.miss += 1
            t.gap.append((frame, t.pred.copy()))
            self.on_miss(t)
        for j in ud:
            if dets[j, 4] < self.new_thr:
                continue
            t = Track(self.next_id, dets[j, :4], dets[j, 4])
            self.next_id += 1
            self.on_new(t, dets[j])
            if self.min_hits <= 1:
                emitted.append((frame, t.id, dets[j, :4].copy()))
            else:
                t.pending.append((frame, t.id, dets[j, :4].copy()))
            self.tracks.append(t)
        if self.record:
            # o que a recorrência (ou o Kalman) previa para este quadro, por track
            for t in self.tracks:
                if t.age > 1 or t.miss > 0:
                    self.log.append((frame, t.id, *t.pred, t.miss, *self.sigma(t)))
        self.post_step(frame)
        for t in self.tracks:
            t.age += 1
        # tentativas morrem na primeira falha; confirmadas após max_age falhas
        self.tracks = [t for t in self.tracks
                       if t.miss <= (self.max_age if t.hits >= self.min_hits else 0)]
        for f, tid, b in emitted:
            self.out.append([f, tid, *b])
        return emitted

    def run(self, dets_by_frame: dict, n_frames: int) -> np.ndarray:
        self.reset()
        for f in range(1, n_frames + 1):
            self.step(f, dets_by_frame.get(f, np.zeros((0, 5))))
        out = np.asarray(self.out, dtype=np.float64).reshape(-1, 6)
        if len(out):
            out = out[np.lexsort((out[:, 1], out[:, 0]))]
        return out


class IoUTracker(BaseTracker):
    """Parte 1 — associação ingênua: IoU com a última caixa observada."""
    name = "iou"


class KalmanTracker(BaseTracker):
    """Baseline de comparação: Kalman de velocidade constante em (cx, cy, w, h)."""
    name = "kalman"

    def __init__(self, std_pos=1 / 20, std_vel=1 / 160, **kw):
        self.std_pos, self.std_vel = std_pos, std_vel
        super().__init__(**kw)
        F = np.eye(8)
        F[:4, 4:] = np.eye(4)
        self.F, self.H = F, np.eye(4, 8)

    def on_new(self, t, det):
        z = xyxy_to_cxcywh(det[:4])[0]
        h = z[3]
        t.d["x"] = np.r_[z, np.zeros(4)]
        sp, sv = self.std_pos * h, self.std_vel * h
        t.d["P"] = np.diag(np.r_[[2 * sp] * 4, [10 * sv] * 4] ** 2)

    def predict(self, frame):
        for t in self.tracks:
            x, P = t.d["x"], t.d["P"]
            h = max(x[3], 1.0)
            Q = np.diag(np.r_[[self.std_pos * h] * 4, [self.std_vel * h] * 4] ** 2)
            x = self.F @ x
            x[2:4] = np.maximum(x[2:4], 1.0)
            t.d["x"], t.d["P"] = x, self.F @ P @ self.F.T + Q
            t.pred = cxcywh_to_xyxy(x[:4])[0]

    def _S(self, t):
        R = np.diag([self.std_pos * max(t.d["x"][3], 1.0)] * 4) ** 2
        return self.H @ t.d["P"] @ self.H.T + R

    def sigma(self, t):
        S = self._S(t)
        return (float(np.sqrt(S[0, 0])), float(np.sqrt(S[1, 1])))

    def maha_d2(self, dets_xyxy):
        Z = xyxy_to_cxcywh(dets_xyxy)
        D = np.zeros((len(self.tracks), len(Z)))
        for i, t in enumerate(self.tracks):
            inv = np.linalg.inv(self._S(t))
            r = Z - t.d["x"][:4]
            D[i] = np.einsum("ni,ij,nj->n", r, inv, r)
        return D

    def on_match(self, t, det):
        super().on_match(t, det)
        z = xyxy_to_cxcywh(det[:4])[0]
        S = self._S(t)
        K = t.d["P"] @ self.H.T @ np.linalg.inv(S)
        t.d["x"] = t.d["x"] + K @ (z - self.H @ t.d["x"])
        t.d["P"] = (np.eye(8) - K @ self.H) @ t.d["P"]


class RNNTracker(BaseTracker):
    """Parte 2, Trilha A: estado recorrente por track carrega o movimento."""
    name = "rnn"

    def __init__(self, model, img_h: float, maha_gate=CHI2_4_99, **kw):
        self.model = model.eval()
        self.img_h = float(img_h)
        super().__init__(maha_gate=maha_gate, **kw)

    def reset(self):
        super().reset()
        self._feed = []   # (track, caixa de entrada cxcywh, observado, score)

    def on_new(self, t, det):
        b = xyxy_to_cxcywh(det[:4])[0]
        t.d["state"] = self.model.init_state(1)
        t.d["prev_in"] = b.copy()
        self._feed.append((t, b, 1.0, det[4]))

    def on_match(self, t, det):
        super().on_match(t, det)
        self._feed.append((t, xyxy_to_cxcywh(det[:4])[0], 1.0, det[4]))

    def on_miss(self, t):
        # sob oclusão a recorrência roda para frente alimentada pela própria previsão
        self._feed.append((t, t.d["pred_c"].copy(), 0.0, 0.0))

    def predict(self, frame):
        for t in self.tracks:
            t.pred = cxcywh_to_xyxy(t.d["pred_c"])[0]

    def sigma(self, t):
        from .models import S
        sd = np.exp(0.5 * t.d["logvar"][:2]) * max(t.d["in_c"][3], 1.0) / S
        return (float(sd[0]), float(sd[1]))

    def maha_d2(self, dets_xyxy):
        if not self.tracks or len(dets_xyxy) == 0:
            return np.zeros((len(self.tracks), len(dets_xyxy)))
        Z = torch.tensor(xyxy_to_cxcywh(dets_xyxy), dtype=torch.float32)
        D = np.zeros((len(self.tracks), len(Z)))
        for i, t in enumerate(self.tracks):
            base = torch.tensor(t.d["in_c"], dtype=torch.float32)[None].expand(len(Z), 4)
            y = encode_target(Z, base)
            mu = torch.tensor(t.d["mu"], dtype=torch.float32)
            var = torch.exp(torch.tensor(t.d["logvar"], dtype=torch.float32))
            D[i] = (((y - mu) ** 2) / var).sum(1).numpy()
        return D

    @torch.no_grad()
    def post_step(self, frame):
        if not self._feed:
            return
        tracks = [f[0] for f in self._feed]
        box = torch.tensor(np.array([f[1] for f in self._feed]), dtype=torch.float32)
        prev = torch.tensor(np.array([t.d["prev_in"] for t in tracks]), dtype=torch.float32)
        obs = torch.tensor([f[2] for f in self._feed])
        score = torch.tensor([f[3] for f in self._feed], dtype=torch.float32)
        state = torch.cat([t.d["state"] for t in tracks], 0)
        x = encode_inputs(box, prev, obs, score, self.img_h)
        mu, logvar, state = self.model.step(x, state)
        nxt = decode(mu, box).numpy()
        for k, t in enumerate(tracks):
            t.d["state"] = state[k:k + 1]
            t.d["prev_in"] = box[k].numpy().astype(np.float64)
            t.d["in_c"] = t.d["prev_in"]
            t.d["mu"] = mu[k].numpy()
            t.d["logvar"] = logvar[k].numpy()
            t.d["pred_c"] = nxt[k].astype(np.float64)
        self._feed = []


def build_tracker(kind: str, model=None, img_h=None, **kw):
    if kind == "iou":
        kw.pop("maha_gate", None)
        return IoUTracker(**kw)
    if kind == "kalman":
        return KalmanTracker(**kw)
    if kind == "rnn":
        return RNNTracker(model, img_h=img_h, **kw)
    raise ValueError(kind)
