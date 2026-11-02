"""Command-line interface.

    python -m fpl_predictor backtest --test-season 2025-26
    python -m fpl_predictor train
    python -m fpl_predictor predict --horizon 3 --squad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import DEFAULT_SEASONS, load_all
from .evaluate import run_backtest
from .features import TARGET, build_features, feature_columns
from .model import PointsModel
from .optimize import pick_squad

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "points_model.joblib"


def _features(refresh: bool):
    df = build_features(load_all(DEFAULT_SEASONS, refresh=refresh))
    return df, feature_columns(df)


def cmd_backtest(args):
    df, feats = _features(args.refresh)
    summary = []
    for season in args.test_season:
        res = run_backtest(df[df["season"] <= season], feats, season, ROOT / "reports" / season)
        for name, m in res["metrics"].items():
            summary.append({"season": season, "predictor": name, **m,
                            "squad_pts_per_gw": res["squad_backtest"].get(name, {}).get("per_gw")})
    table = pd.DataFrame(summary).round(3)
    table.to_csv(ROOT / "reports" / "backtest_summary.csv", index=False)
    print(table.to_string(index=False))


def cmd_train(args):
    df, feats = _features(args.refresh)
    df = df[df[TARGET].notna()]
    current = df["season"].max()
    train, valid = df[df["season"] < current], df[df["season"] == current]
    probe = PointsModel(feats).fit(train, valid if len(valid) > 5000 else None)
    model = PointsModel(feats).fit(df, num_boost_round=int(probe.best_iteration * 1.1))
    model.save(MODEL_PATH)
    print(f"Trained on {len(df):,} player-fixtures ({df['season'].nunique()} seasons); saved to {MODEL_PATH}")


def cmd_predict(args):
    from .predict import predict_upcoming

    if not MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run `python -m fpl_predictor train` first.")
    model = PointsModel.load(MODEL_PATH)
    history = load_all(DEFAULT_SEASONS, refresh=args.refresh)
    season = history["season"].max()
    preds = predict_upcoming(history, model, season, args.horizon)
    out = ROOT / "reports" / "predictions.csv"
    out.parent.mkdir(exist_ok=True)
    preds.to_csv(out, index=False)

    pd.set_option("display.width", 140)
    for gw, g in preds.groupby("gw"):
        print(f"\n=== GW{gw}: top {args.top} predicted players ===")
        print(g.head(args.top)[["name", "team", "position", "value", "opponents", "xpts"]]
              .assign(value=lambda d: d["value"] / 10).round(2).to_string(index=False))

    if args.squad:
        total = preds.groupby("player_key").agg(
            name=("name", "first"), team=("team", "first"), position=("position", "first"),
            value=("value", "first"), xpts=("xpts", "sum")).reset_index()
        squad = pick_squad(total, score_col="xpts", budget=args.budget * 10)
        label = f"next {args.horizon} GWs" if args.horizon > 1 else "next GW"
        print(f"\n=== Optimal squad for {label} (budget £{args.budget}m) ===")
        show = squad.assign(value=squad["value"] / 10,
                            role=squad["captain"].map({True: "C"}).fillna(squad["starter"].map({True: "XI", False: "bench"})))
        print(show[["role", "name", "team", "position", "value", "xpts"]].round(2).to_string(index=False))
        xi = squad[squad["starter"]]
        print(f"Squad cost £{squad['value'].sum() / 10:.1f}m | XI expected points incl. captain: "
              f"{xi['xpts'].sum() + xi.loc[xi['captain'], 'xpts'].sum():.1f}")
    print(f"\nFull predictions written to {out}")


def main():
    ap = argparse.ArgumentParser(prog="fpl_predictor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download data instead of using the cache")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backtest", help="train on past seasons, evaluate on a held-out season")
    b.add_argument("--test-season", nargs="+", default=["2023-24", "2024-25", "2025-26"])
    b.set_defaults(func=cmd_backtest)

    t = sub.add_parser("train", help="train the production model on all available data")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="forecast upcoming gameweeks")
    p.add_argument("--horizon", type=int, default=1, help="number of upcoming gameweeks")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--squad", action="store_true", help="also pick the optimal 15-man squad")
    p.add_argument("--budget", type=float, default=100.0)
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
