"""Leakage-free feature engineering.

Every feature for a fixture is computed only from information available
*before* that fixture kicks off: all rolling statistics are shifted by one
match, so a row never sees its own outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "total_points"

# Per-match stats that feed rolling "form" features.
ROLL_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "bonus", "bps", "ict_index", "influence", "creativity",
    "threat", "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "saves", "defensive_contribution", "starts",
]
SHORT_WINDOW, LONG_WINDOW, TEAM_WINDOW = 3, 8, 6


def _player_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_key", "kickoff_time", "fixture"]).copy()
    g = df.groupby("player_key", sort=False)

    new = {}
    for col in ROLL_STATS:
        prev = g[col].shift(1)
        pg = prev.groupby(df["player_key"], sort=False)
        new[f"{col}_r{SHORT_WINDOW}"] = pg.rolling(SHORT_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        new[f"{col}_r{LONG_WINDOW}"] = pg.rolling(LONG_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)

    prev_pts = g["total_points"].shift(1)
    prev_min = g["minutes"].shift(1)
    new["points_ewm"] = prev_pts.groupby(df["player_key"], sort=False).transform(lambda s: s.ewm(alpha=0.25).mean())
    new["minutes_ewm"] = prev_min.groupby(df["player_key"], sort=False).transform(lambda s: s.ewm(alpha=0.35).mean())
    new["points_r38"] = prev_pts.groupby(df["player_key"], sort=False).rolling(38, min_periods=1).mean().reset_index(level=0, drop=True)

    # Availability: how often the player has recently started / played 60+.
    played60 = (prev_min >= 60).astype(float).where(prev_min.notna())
    new["played60_rate_r5"] = played60.groupby(df["player_key"], sort=False).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    new["played_any_last"] = (prev_min > 0).astype(float).where(prev_min.notna())

    # Per-90 rates over the long window (only counts minutes actually played).
    for col in ["total_points", "expected_goal_involvements", "bps", "defensive_contribution"]:
        num = g[col].shift(1).groupby(df["player_key"], sort=False).rolling(LONG_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
        den = prev_min.groupby(df["player_key"], sort=False).rolling(LONG_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
        new[f"{col}_p90"] = np.where(den >= 90, num / den.clip(lower=1) * 90, np.nan)

    new["matches_seen"] = g.cumcount()
    new["days_rest"] = (df["kickoff_time"] - g["kickoff_time"].shift(1)).dt.total_seconds() / 86400
    new["season_match_idx"] = df.groupby(["player_key", "season"], sort=False).cumcount()
    new["log_selected"] = np.log1p(df["selected"].clip(lower=0))

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def _team_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture with pre-match rolling attack/defence strength."""
    rows = df.assign(
        goals_for=np.where(df["was_home"] == 1, df["team_h_score"], df["team_a_score"]),
        goals_against=np.where(df["was_home"] == 1, df["team_a_score"], df["team_h_score"]),
    )
    team = (rows.groupby(["season", "fixture", "team"], sort=False)
            .agg(kickoff_time=("kickoff_time", "first"), opponent=("opponent", "first"),
                 goals_for=("goals_for", "first"), goals_against=("goals_against", "first"),
                 xg_for=("expected_goals", lambda s: s.sum(min_count=1)))
            .reset_index())
    opp_xg = team[["season", "fixture", "team", "xg_for"]].rename(columns={"team": "opponent", "xg_for": "xg_against"})
    team = team.merge(opp_xg, on=["season", "fixture", "opponent"], how="left")
    # Seasons before xG existed report zeros; treat them as missing rather than "no chances".
    season_has_xg = team.groupby("season")["xg_for"].transform(lambda s: s.fillna(0).sum() > 0)
    team.loc[~season_has_xg, ["xg_for", "xg_against"]] = np.nan

    team = team.sort_values(["team", "kickoff_time"])
    tg = team.groupby("team", sort=False)
    for col in ["goals_for", "goals_against", "xg_for", "xg_against"]:
        team[f"team_{col}_r{TEAM_WINDOW}"] = (
            tg[col].shift(1).groupby(team["team"], sort=False)
            .rolling(TEAM_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
        )
    return team


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the input rows with all model features attached."""
    df = _player_features(df)
    team = _team_table(df)
    tcols = [c for c in team.columns if c.startswith("team_") and c != "team"]

    own = team[["season", "fixture", "team", *tcols]]
    opp = team[["season", "fixture", "team", *tcols]].rename(
        columns={"team": "opponent", **{c: c.replace("team_", "opp_", 1) for c in tcols}}
    )
    df = df.merge(own, on=["season", "fixture", "team"], how="left")
    df = df.merge(opp, on=["season", "fixture", "opponent"], how="left")

    # Interaction: an attacker's xGI against a leaky defence, a defender's CS odds vs a blunt attack.
    df["attack_vs_opp_def"] = df[f"expected_goal_involvements_r{LONG_WINDOW}"] * df[f"opp_goals_against_r{TEAM_WINDOW}"]
    df["cs_proxy"] = 1.0 / (1.0 + df[f"opp_goals_for_r{TEAM_WINDOW}"].fillna(1.3) + df[f"team_goals_against_r{TEAM_WINDOW}"].fillna(1.3))
    df["position"] = pd.Categorical(df["position"], categories=["GK", "DEF", "MID", "FWD"])
    return df.sort_values(["kickoff_time", "fixture", "player_key"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = ["position", "was_home", "value", "log_selected", "matches_seen", "days_rest",
            "season_match_idx", "points_ewm", "minutes_ewm", "points_r38", "played60_rate_r5",
            "played_any_last", "attack_vs_opp_def", "cs_proxy"]
    rolled = [c for c in df.columns if c.endswith((f"_r{SHORT_WINDOW}", f"_r{LONG_WINDOW}", "_p90"))]
    team = [c for c in df.columns if c.startswith(("team_", "opp_")) and c.endswith(f"_r{TEAM_WINDOW}")]
    return list(dict.fromkeys(base + rolled + team))
