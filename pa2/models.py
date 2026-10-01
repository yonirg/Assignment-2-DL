"""Parte 2, Trilha A — RNN como modelo de movimento.

Um estado recorrente por track. A cada quadro a rede recebe a última "entrada"
da track — a detecção casada (observada) ou, sob oclusão, a própria previsão
anterior (não observada) — e prevê a caixa do quadro seguinte junto com uma
incerteza (log-variância), usada como portão de associação adaptativo.

Representação (invariante a translação e a escala do objeto):
  entrada x_t = [S·Δcx/h, S·Δcy/h, S·Δlog w, S·Δlog h,   (vs. entrada anterior)
                 observado, score, cy/H − 0.5, log(h/H)]
  saída       = μ, log σ² de [S·(cx'−cx)/h, S·(cy'−cy)/h, S·log(w'/w), S·log(h'/h)]
com (cx, cy, w, h) a caixa de entrada atual e (cx', cy', w', h') a do próximo quadro.
"""
from __future__ import annotations

import math

import torch
from torch import nn

S = 10.0
IN_DIM = 8
OUT_DIM = 4


def encode_inputs(box, prev_box, observed, score, img_h):
    """box, prev_box: (B,4) cx,cy,w,h (tensores); observed, score: (B,)."""
    h0 = prev_box[:, 3].clamp(min=1.0)
    dx = S * (box[:, 0] - prev_box[:, 0]) / h0
    dy = S * (box[:, 1] - prev_box[:, 1]) / h0
    dw = S * torch.log(box[:, 2].clamp(min=1.0) / prev_box[:, 2].clamp(min=1.0))
    dh = S * torch.log(box[:, 3].clamp(min=1.0) / h0)
    if not torch.is_tensor(img_h):
        img_h = torch.full_like(dx, float(img_h))
    cy = box[:, 1] / img_h - 0.5
    lh = torch.log(box[:, 3].clamp(min=1.0) / img_h)
    return torch.stack([dx, dy, dw, dh, observed.float(), score.float(), cy, lh], 1)


def encode_target(next_box, box):
    h = box[:, 3].clamp(min=1.0)
    return torch.stack([
        S * (next_box[:, 0] - box[:, 0]) / h,
        S * (next_box[:, 1] - box[:, 1]) / h,
        S * torch.log(next_box[:, 2].clamp(min=1.0) / box[:, 2].clamp(min=1.0)),
        S * torch.log(next_box[:, 3].clamp(min=1.0) / h),
    ], 1)


def decode(mu, box):
    h = box[:, 3].clamp(min=1.0)
    cx = box[:, 0] + mu[:, 0] * h / S
    cy = box[:, 1] + mu[:, 1] * h / S
    w = box[:, 2] * torch.exp((mu[:, 2] / S).clamp(-1, 1))
    hh = box[:, 3] * torch.exp((mu[:, 3] / S).clamp(-1, 1))
    return torch.stack([cx, cy, w, hh], 1)


def hidden_for_budget(cell: str, budget: int, in_dim: int = IN_DIM) -> int:
    """Tamanho do estado para ~``budget`` parâmetros na célula recorrente."""
    g = {"rnn": 1, "gru": 3, "lstm": 4}[cell]
    # g * (h*(in+h) + 2h) = budget  ->  g h^2 + g(in+2) h - budget = 0
    a, b, c = g, g * (in_dim + 2), -budget
    return max(4, int(round((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a))))


class MotionRNN(nn.Module):
    def __init__(self, cell: str = "gru", hidden: int = 64, head: int = 64):
        super().__init__()
        self.cell_type, self.hidden = cell, hidden
        Cell = {"rnn": nn.RNNCell, "gru": nn.GRUCell, "lstm": nn.LSTMCell}[cell]
        kw = dict(nonlinearity="tanh") if cell == "rnn" else {}
        self.cell = Cell(IN_DIM, hidden, **kw)
        self.head = nn.Sequential(nn.Linear(hidden, head), nn.ReLU(), nn.Linear(head, 2 * OUT_DIM))
        with torch.no_grad():
            self.head[-1].bias[OUT_DIM:].fill_(0.0)

    @property
    def state_dim(self):
        return 2 * self.hidden if self.cell_type == "lstm" else self.hidden

    def init_state(self, batch: int, device=None):
        return torch.zeros(batch, self.state_dim, device=device)

    def h_of(self, state):
        return state[:, : self.hidden]

    def step(self, x, state):
        """Um passo. ``state`` é (B, state_dim) (h, ou [h|c] na LSTM)."""
        if self.cell_type == "lstm":
            h, c = self.cell(x, (state[:, : self.hidden], state[:, self.hidden:]))
            new = torch.cat([h, c], 1)
        else:
            h = self.cell(x, state)
            new = h
        out = self.head(h)
        mu, logvar = out[:, :OUT_DIM], out[:, OUT_DIM:].clamp(-6, 6)
        return mu, logvar, new

    def config(self):
        return dict(cell=self.cell_type, hidden=self.hidden, head=self.head[0].out_features)

    @staticmethod
    def from_checkpoint(path, map_location="cpu"):
        ck = torch.load(path, map_location=map_location, weights_only=False)
        m = MotionRNN(**ck["model_config"])
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return m, ck


def n_params(m: nn.Module, only_cell: bool = False):
    mod = m.cell if only_cell else m
    return sum(p.numel() for p in mod.parameters())
