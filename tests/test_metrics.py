"""Parte 0.3 — testes da métrica em casos construídos à mão.

  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import average_precision, clear_mot, evaluate_tracking, idf1, iou_matrix, nms  # noqa: E402

N = 20  # quadros


def straight(tid, y, frames, x0=0.0, v=5.0, w=10.0, h=20.0):
    return np.array([[f, tid, x0 + v * f, y, x0 + v * f + w, y + h] for f in frames], dtype=float)


def two_objects():
    fr = range(1, N + 1)
    return np.vstack([straight(1, 0, fr), straight(2, 100, fr)])


def test_a_identity():
    gt = two_objects()
    r = evaluate_tracking(gt, gt.copy())
    assert r["IDF1"] == pytest.approx(1.0)
    assert r["IDSW"] == 0 and r["Frag"] == 0
    assert r["count_err"] == 0


@pytest.mark.parametrize("k", [5, 11, 16])
def test_b_swap_from_frame_k(k):
    """As identidades previstas 10 e 20 se trocam a partir do quadro k."""
    gt = two_objects()
    pr = gt.copy()
    pr[:, 1] = np.where(gt[:, 1] == 1, 10, 20)
    after = pr[:, 0] >= k
    pr[after, 1] = np.where(gt[after, 1] == 1, 20, 10)
    r = evaluate_tracking(gt, pr)
    # cada objeto verdadeiro muda de id uma vez -> 2 switches
    assert r["IDSW"] == 2
    assert r["Frag"] == 0
    # IDF1: a atribuição global fica com o trecho mais longo de cada objeto
    n_before = k - 1
    expected = max(n_before, N - n_before) / N
    assert r["IDF1"] == pytest.approx(expected)
    assert r["count_err"] == 0   # a contagem de identidades não percebe a troca


def test_c_track_split_in_two():
    """Mesma cena de (b), mas o objeto 1 tem a track partida em duas no meio
    (com buraco de 2 quadros); o objeto 2 é rastreado perfeitamente."""
    fr = list(range(1, N + 1))
    pr = np.vstack([straight(7, 0, fr[:10]),      # obj 1, quadros 1..10
                    straight(8, 0, fr[12:]),      # obj 1, quadros 13..20 (11 e 12 perdidos)
                    straight(9, 100, fr)])        # obj 2, intacto
    r = evaluate_tracking(two_objects(), pr)
    assert r["IDSW"] == 1        # um único switch (7 -> 8)
    assert r["Frag"] == 1        # interrompida e retomada
    # IDF1: só um dos pedaços conta para o objeto 1: IDTP = 10 + 20,
    # IDFN = 40 - 30, IDFP = 38 - 30  ->  60 / (60 + 8 + 10)
    assert r["IDF1"] == pytest.approx(60 / 78)
    assert r["count_err"] == 1   # uma identidade a mais no vídeo


def test_b_vs_c_differ():
    """Corte no mesmo quadro k: a troca (b) estraga DUAS identidades (2 switches,
    IDF1 = 0.5) sem mudar a contagem; a partição (c) estraga uma (1 switch,
    IDF1 = 0.75) e infla a contagem de identidades."""
    k = 11
    gt = two_objects()
    pb = gt.copy()
    pb[:, 1] = np.where(gt[:, 1] == 1, 10, 20)
    after = pb[:, 0] >= k
    pb[after, 1] = np.where(gt[after, 1] == 1, 20, 10)
    rb = evaluate_tracking(gt, pb)
    fr = list(range(1, N + 1))
    pc = np.vstack([straight(7, 0, fr[:10]), straight(8, 0, fr[10:]), straight(9, 100, fr)])
    rc = evaluate_tracking(gt, pc)
    assert (rb["IDSW"], rc["IDSW"]) == (2, 1)
    assert rb["IDF1"] == pytest.approx(0.5) and rc["IDF1"] == pytest.approx(0.75)
    assert (rb["count_err"], rc["count_err"]) == (0, 1)


def test_idf1_is_global_not_framewise():
    """Uma id prevista que cobre dois objetos em tempos diferentes só conta para um."""
    gt = np.vstack([straight(1, 0, range(1, 11)), straight(2, 100, range(11, 21))])
    pr = gt.copy()
    pr[:, 1] = 5
    r = idf1(gt, pr)
    assert r["IDTP"] == 10
    assert clear_mot(gt, pr)["IDSW"] == 0   # objetos diferentes, nenhum switch


def test_iou_and_nms():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30]])
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]])
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert list(keep) == [0, 2]


def test_average_precision():
    gt = np.array([[1, 1, 0, 0, 10, 10], [1, 2, 20, 20, 30, 30]], dtype=float)
    perfect = np.array([[1, 0, 0, 10, 10, 0.9], [1, 20, 20, 30, 30, 0.8]], dtype=float)
    assert average_precision(gt, perfect) == pytest.approx(1.0)
    with_fp = np.vstack([perfect, [1, 50, 50, 60, 60, 0.95]])
    assert average_precision(gt, with_fp) < 1.0
