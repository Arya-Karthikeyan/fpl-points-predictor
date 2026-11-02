"""Pick the optimal 15-man FPL squad, starting XI and captain with integer programming."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_TEAM = 3
BENCH_WEIGHT = 0.1


def pick_squad(players: pd.DataFrame, score_col: str = "pred", budget: float = 1000,
               bench_weight: float = BENCH_WEIGHT) -> pd.DataFrame:
    """Choose a squad maximising XI score + captain bonus (+ a small weight on the bench).

    `players` needs columns: position, team, value (price in £0.1m), and `score_col`.
    Returns the selected 15 rows with boolean `starter` and `captain` columns.
    """
    p = players.reset_index(drop=True)
    n = len(p)
    s = p[score_col].to_numpy(dtype=float)
    # Decision vector: [squad (n) | starter (n) | captain (n)]
    c = -np.concatenate([bench_weight * s, (1 - bench_weight) * s, s])

    rows, lo, hi = [], [], []

    def add(coefs, lb, ub):
        rows.append(coefs); lo.append(lb); hi.append(ub)

    zero = np.zeros(n)
    pos = p["position"].astype(str).to_numpy()
    for position, k in SQUAD_SIZE.items():
        m = (pos == position).astype(float)
        add(np.concatenate([m, zero, zero]), k, k)
        a, b = XI_LIMITS[position]
        add(np.concatenate([zero, m, zero]), a, b)
    add(np.concatenate([zero, np.ones(n), zero]), 11, 11)
    add(np.concatenate([zero, zero, np.ones(n)]), 1, 1)
    add(np.concatenate([p["value"].to_numpy(dtype=float), zero, zero]), 0, budget)
    for team in p["team"].unique():
        m = (p["team"] == team).to_numpy(dtype=float)
        add(np.concatenate([m, zero, zero]), 0, MAX_PER_TEAM)
    A = np.vstack(rows)
    # Linking: starter <= squad, captain <= starter
    eye = np.eye(n)
    link = np.vstack([np.hstack([-eye, eye, 0 * eye]), np.hstack([0 * eye, -eye, eye])])
    A = np.vstack([A, link])
    lo = np.concatenate([lo, np.full(2 * n, -np.inf)])
    hi = np.concatenate([hi, np.zeros(2 * n)])

    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(3 * n),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"Squad optimisation failed: {res.message}")
    x = np.round(res.x).astype(bool)
    out = p.assign(starter=x[n:2 * n], captain=x[2 * n:])[x[:n]]
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    return out.sort_values(["starter", "position", score_col], ascending=[False, True, False],
                           key=lambda col: col.map(order) if col.name == "position" else col)


def squad_points(squad: pd.DataFrame, points_col: str) -> float:
    """Points scored by a squad: starting XI, captain counted twice (no auto-subs)."""
    xi = squad[squad["starter"]]
    return float(xi[points_col].sum() + xi.loc[xi["captain"], points_col].sum())
