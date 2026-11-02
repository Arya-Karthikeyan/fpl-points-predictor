"""Walk-forward backtest: train on past seasons, evaluate on an unseen season."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .features import TARGET
from .model import PointsModel
from .optimize import pick_squad, squad_points

BASELINES = {
    "form_last8": "total_points_r8",   # mean points over last 8 matches
    "form_ewm": "points_ewm",          # exponentially weighted form
    "long_run_avg": "points_r38",      # mean over last 38 matches
}


def split(df: pd.DataFrame, test_season: str):
    seasons = sorted(df["season"].unique())
    i = seasons.index(test_season)
    if i < 2:
        raise ValueError("Need at least two seasons before the test season")
    train = df[df["season"].isin(seasons[: i - 1])]
    valid = df[df["season"] == seasons[i - 1]]
    test = df[df["season"] == test_season]
    return train, valid, test


def fit_for_test(df: pd.DataFrame, features: list[str], test_season: str) -> tuple[PointsModel, pd.DataFrame]:
    """Early-stop on the season before `test_season`, then refit on everything before it."""
    train, valid, test = split(df, test_season)
    probe = PointsModel(features).fit(train, valid)
    n_rounds = int(probe.best_iteration * 1.1)
    model = PointsModel(features).fit(pd.concat([train, valid]), num_boost_round=n_rounds)
    return model, test.assign(pred=model.predict(test))


def to_gameweeks(test: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fixture rows to one row per player per gameweek (sums double gameweeks)."""
    score_cols = ["pred", TARGET, *BASELINES.values()]
    agg = {c: "sum" for c in score_cols}
    agg.update(position="first", team="first", value="first", name="first")
    tmp = test.assign(**{c: test[c].fillna(0) for c in BASELINES.values()})
    return tmp.groupby(["season", "gw", "player_key"], observed=True).agg(agg).reset_index()


def _metrics(gw: pd.DataFrame, col: str) -> dict:
    err = gw[col] - gw[TARGET]
    per_gw = gw.groupby("gw")
    rho = per_gw.apply(lambda g: spearmanr(g[col], g[TARGET]).statistic, include_groups=False).mean()

    def top_hit(g, k=20):
        return len(set(g.nlargest(k, col)["player_key"]) & set(g.nlargest(k, TARGET)["player_key"])) / k

    def top_pts(g, k=20):
        return g.nlargest(k, col)[TARGET].mean()

    return {
        "MAE": float(err.abs().mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1 - (err ** 2).sum() / ((gw[TARGET] - gw[TARGET].mean()) ** 2).sum()),
        "spearman_per_gw": float(rho),
        "top20_overlap": float(per_gw.apply(top_hit, include_groups=False).mean()),
        "top20_avg_actual_pts": float(per_gw.apply(top_pts, include_groups=False).mean()),
    }


def squad_backtest(gw: pd.DataFrame, cols: dict[str, str]) -> pd.DataFrame:
    """Each GW, pick the best squad by each score column and record the actual points it scored.

    This is a 'free hit every week' simulation: no transfer limits, isolating forecast quality.
    """
    out = []
    for week, g in gw.groupby("gw"):
        row = {"gw": week}
        for label, col in cols.items():
            row[label] = squad_points(pick_squad(g, score_col=col), TARGET)
        out.append(row)
    return pd.DataFrame(out)


def run_backtest(df: pd.DataFrame, features: list[str], test_season: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, test = fit_for_test(df, features, test_season)
    gw = to_gameweeks(test)

    results = {"test_season": test_season, "n_player_gameweeks": int(len(gw)),
               "model_rounds": model.best_iteration, "metrics": {}}
    results["metrics"]["lightgbm_model"] = _metrics(gw, "pred")
    for name, col in BASELINES.items():
        results["metrics"][name] = _metrics(gw, col)

    # Regulars only (played 60+ in most recent matches) - the players managers actually choose between.
    reg_keys = test.loc[test["played60_rate_r5"] >= 0.6, ["gw", "player_key"]].drop_duplicates()
    gw_reg = gw.merge(reg_keys, on=["gw", "player_key"])
    results["metrics_regulars"] = {"lightgbm_model": _metrics(gw_reg, "pred"),
                                   **{n: _metrics(gw_reg, c) for n, c in BASELINES.items()}}

    sq = squad_backtest(gw, {"lightgbm_model": "pred", "form_ewm": "points_ewm",
                             "form_last8": "total_points_r8", "oracle_hindsight": TARGET})
    results["squad_backtest"] = {c: {"total": float(sq[c].sum()), "per_gw": float(sq[c].mean())}
                                 for c in sq.columns if c != "gw"}
    sq.to_csv(out_dir / "squad_backtest.csv", index=False)

    imp = model.feature_importance()
    results["top_features"] = (imp / imp.sum()).head(15).round(4).to_dict()
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    _plots(gw, sq, imp, out_dir)
    return results


def _plots(gw: pd.DataFrame, sq: pd.DataFrame, imp: pd.Series, out_dir: Path) -> None:
    # Calibration: predicted vs actual by prediction decile
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bins = pd.qcut(gw["pred"].rank(method="first"), 20, labels=False)
    cal = gw.groupby(bins)[["pred", TARGET]].mean()
    ax.plot(cal["pred"], cal[TARGET], "o-", label="LightGBM model")
    lim = [0, max(cal.max()) * 1.05]
    ax.plot(lim, lim, "--", color="grey", lw=1, label="perfect calibration")
    ax.set(xlabel="Predicted points (bin mean)", ylabel="Actual points (bin mean)",
           title="Calibration on held-out season")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "calibration.png", dpi=130); plt.close(fig)

    # Cumulative squad points
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"oracle_hindsight": "Hindsight oracle (upper bound)", "lightgbm_model": "LightGBM model",
              "form_ewm": "Form (EWM) baseline", "form_last8": "Form (last 8) baseline"}
    for col, label in labels.items():
        ax.plot(sq["gw"], sq[col].cumsum(), label=label, ls="--" if "oracle" in col else "-")
    ax.set(xlabel="Gameweek", ylabel="Cumulative points", title="Optimised squad each GW (free-hit simulation)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "squad_backtest.png", dpi=130); plt.close(fig)

    # Feature importance
    fig, ax = plt.subplots(figsize=(6.5, 5))
    top = (imp / imp.sum()).head(15)[::-1]
    ax.barh(top.index, top.values)
    ax.set(xlabel="Share of total gain", title="Top 15 features")
    fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=130); plt.close(fig)
