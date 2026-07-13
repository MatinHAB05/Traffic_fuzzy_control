"""Mamdani fuzzy controller that decides the green-time of road 1
from the two queue-length inputs (road 1, road 2)."""
import numpy as np

from . import config as cfg

IN_LABELS = ("low", "medium", "high")
OUT_LABELS = ("short", "medium", "long")

# fixed rule base: (queue1_label, queue2_label) -> output_label
RULE_BASE = (
    ("low", "low", "medium"),
    ("low", "medium", "short"),
    ("low", "high", "short"),
    ("medium", "low", "long"),
    ("medium", "medium", "medium"),
    ("medium", "high", "short"),
    ("high", "low", "long"),
    ("high", "medium", "long"),
    ("high", "high", "medium"),
)


def trimf(x, a, b, c):
    """Triangular membership function, scalar input. Handles flat
    shoulders correctly when a==b (left shoulder) or b==c (right shoulder)."""
    left = 1.0 if b <= a else (x - a) / (b - a)
    right = 1.0 if c <= b else (c - x) / (c - b)
    y = min(left, right)
    return min(max(y, 0.0), 1.0)


def trimf_vec(x, a, b, c):
    """Vectorized triangular membership function."""
    left = np.ones_like(x) if b <= a else (x - a) / (b - a)
    right = np.ones_like(x) if c <= b else (c - x) / (c - b)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


class FuzzyVariable:
    """Strong fuzzy partition: 3 triangular sets sharing vertices,
    built from 3 sorted centers over [lo, hi]."""

    def __init__(self, lo, hi, centers):
        self.lo, self.hi = float(lo), float(hi)
        c1, c2, c3 = sorted(float(c) for c in centers)
        self.c1, self.c2, self.c3 = c1, c2, c3

    def memberships(self, x):
        x = min(max(x, self.lo), self.hi)
        return {
            "low": trimf(x, self.lo, self.c1, self.c2),
            "medium": trimf(x, self.c1, self.c2, self.c3),
            "high": trimf(x, self.c2, self.c3, self.hi),
        }

    def memberships_vec(self, x_arr, label):
        if label == "low":
            return trimf_vec(x_arr, self.lo, self.c1, self.c2)
        if label == "medium":
            return trimf_vec(x_arr, self.c1, self.c2, self.c3)
        return trimf_vec(x_arr, self.c2, self.c3, self.hi)


class FuzzyController:
    """
    Parameter vector layout (18 values):
      [0:3]  centers of input fuzzy sets for queue length road 1
      [3:6]  centers of input fuzzy sets for queue length road 2
      [6:9]  centers of output fuzzy sets for green time road 1
      [9:18] weight (importance) of each of the 9 rules, RULE_BASE order
    """

    N_PARAMS = 18
    N_OUT_POINTS = 150

    def __init__(self, params=None):
        if params is None:
            params = self.default_params()
        params = np.asarray(params, dtype=float)

        self.q1_var = FuzzyVariable(0, cfg.QUEUE_MAX, params[0:3])
        self.q2_var = FuzzyVariable(0, cfg.QUEUE_MAX, params[3:6])
        self.out_var = FuzzyVariable(cfg.GREEN_MIN, cfg.GREEN_MAX, params[6:9])
        self.rule_weights = np.clip(params[9:18], 0.0, 1.0)

        self._out_x = np.linspace(
            cfg.GREEN_MIN, cfg.GREEN_MAX, self.N_OUT_POINTS)
        # output labels (short/medium/long) map to the generic set
        # positions (low/medium/high) used by FuzzyVariable
        out_to_generic = {"short": "low", "medium": "medium", "long": "high"}
        self._out_shapes = {
            lbl: self.out_var.memberships_vec(self._out_x, out_to_generic[lbl])
            for lbl in OUT_LABELS
        }

    @staticmethod
    def default_params():
        p = np.zeros(FuzzyController.N_PARAMS)
        p[0:3] = [0, cfg.QUEUE_MAX / 2, cfg.QUEUE_MAX]
        p[3:6] = [0, cfg.QUEUE_MAX / 2, cfg.QUEUE_MAX]
        p[6:9] = [cfg.GREEN_MIN,
                  (cfg.GREEN_MIN + cfg.GREEN_MAX) / 2, cfg.GREEN_MAX]
        p[9:18] = 1.0
        return p

    def compute_green_time(self, q1, q2):
        deg1 = self.q1_var.memberships(q1)
        deg2 = self.q2_var.memberships(q2)

        out_strength = {lbl: 0.0 for lbl in OUT_LABELS}
        for idx, (l1, l2, out_lbl) in enumerate(RULE_BASE):
            strength = min(deg1[l1], deg2[l2]) * self.rule_weights[idx]
            if strength > out_strength[out_lbl]:
                out_strength[out_lbl] = strength

        # Mamdani implication (min) + aggregation (max) over the output grid
        agg = np.zeros(self.N_OUT_POINTS)
        for lbl in OUT_LABELS:
            s = out_strength[lbl]
            if s <= 0:
                continue
            agg = np.maximum(agg, np.minimum(s, self._out_shapes[lbl]))

        total = agg.sum()
        if total <= 1e-9:
            return (cfg.GREEN_MIN + cfg.GREEN_MAX) / 2.0

        centroid = float((agg * self._out_x).sum() / total)
        return min(max(centroid, cfg.GREEN_MIN), cfg.GREEN_MAX)
