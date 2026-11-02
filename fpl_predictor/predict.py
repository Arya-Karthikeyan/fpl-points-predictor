"""Forecast upcoming gameweeks for the current season."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_fixtures, load_players_raw, load_teams
from .features import build_features
from .model import PointsModel

ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def upcoming_rows(history: pd.DataFrame, season: str, horizon: int = 1) -> pd.DataFrame:
    """Build placeholder fixture rows (no outcomes) for the next `horizon` unplayed gameweeks."""
    fx = load_fixtures(season)
    teams = load_teams(season).set_index("id")["name"]
    players = load_players_raw(season)
    players = players[players["status"] != "u"]  # 'u' = left the club / unavailable for the season

    played_gws = history.loc[history["season"] == season, "gw"]
    next_gw = int(played_gws.max()) + 1 if len(played_gws) else 1
    fx = fx[fx["event"].between(next_gw, next_gw + horizon - 1) & ~fx["finished"].astype(bool)]

    sides = pd.concat([
        fx.assign(team_id=fx["team_h"], opp_id=fx["team_a"], was_home=1),
        fx.assign(team_id=fx["team_a"], opp_id=fx["team_h"], was_home=0),
    ])
    last_selected = history.groupby("player_key")["selected"].last()
    p = players.assign(
        name=(players["first_name"].str.strip() + " " + players["second_name"].str.strip()),
        position=players["element_type"].map(ELEMENT_TYPE),
        value=players["now_cost"],
        availability=players["chance_of_playing_next_round"].replace("None", np.nan).astype(float).fillna(100) / 100,
        element=players["id"],
    )[["element", "name", "web_name", "team", "position", "value", "availability"]]
    rows = p.merge(sides, left_on="team", right_on="team_id")
    rows = rows.assign(
        season=season, gw=rows["event"].astype(int), fixture=rows["id"],
        kickoff_time=pd.to_datetime(rows["kickoff_time"], utc=True),
        team=rows["team_id"].map(teams), opponent=rows["opp_id"].map(teams),
        player_key=rows["name"].str.strip(),
    )
    rows["selected"] = rows["player_key"].map(last_selected).fillna(0)
    keep = ["season", "gw", "fixture", "kickoff_time", "element", "name", "web_name", "player_key",
            "team", "opponent", "was_home", "position", "value", "selected", "availability"]
    return rows[keep].assign(_future=True)


def predict_upcoming(history: pd.DataFrame, model: PointsModel, season: str, horizon: int = 1) -> pd.DataFrame:
    """Expected points per player per upcoming gameweek (double gameweeks summed).

    Model predictions are scaled by FPL's 'chance of playing' flag for injured/doubtful players.
    """
    future = upcoming_rows(history, season, horizon)
    combined = pd.concat([history.assign(_future=False), future], ignore_index=True, sort=False)
    feats = build_features(combined)
    fut = feats[feats["_future"].astype(bool)].sort_values("kickoff_time").copy()

    # Rolling player features for the 2nd+ future fixture would otherwise "see" the earlier
    # placeholder rows as missing matches. Use the player's form as of today for every fixture.
    fixture_specific = {"was_home", "days_rest", "attack_vs_opp_def", "cs_proxy"}
    player_cols = [f for f in model.features
                   if f not in fixture_specific and not f.startswith(("team_", "opp_")) and f != "position"]
    fut[player_cols] = fut.groupby("player_key", sort=False)[player_cols].transform("first")
    fut["attack_vs_opp_def"] = fut["expected_goal_involvements_r8"] * fut["opp_goals_against_r6"]

    fut["xpts"] = model.predict(fut).clip(lower=0) * fut["availability"]
    fut["fx_label"] = fut["opponent"] + np.where(fut["was_home"] == 1, " (H)", " (A)")
    out = (fut.groupby(["gw", "player_key"], observed=True)
           .agg(name=("web_name", "first"), team=("team", "first"), position=("position", "first"),
                value=("value", "first"), opponents=("fx_label", ", ".join), availability=("availability", "first"),
                xpts=("xpts", "sum"))
           .reset_index())
    return out.sort_values(["gw", "xpts"], ascending=[True, False])
