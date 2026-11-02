"""Unit tests on small synthetic data (no network access needed)."""
import numpy as np
import pandas as pd
import pytest

from fpl_predictor.features import build_features, feature_columns
from fpl_predictor.model import PointsModel
from fpl_predictor.optimize import SQUAD_SIZE, pick_squad, squad_points

TEAMS = ["A", "B", "C", "D"]


def synthetic_history(n_gws=12, seasons=("2023-24", "2024-25"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, fixture = [], 0
    start = pd.Timestamp("2023-08-01", tz="UTC")
    for si, season in enumerate(seasons):
        for gw in range(1, n_gws + 1):
            ko = start + pd.Timedelta(days=365 * si + 7 * gw)
            for home, away in [("A", "B"), ("C", "D")] if gw % 2 else [("A", "C"), ("B", "D")]:
                fixture += 1
                hs, as_ = rng.poisson(1.5), rng.poisson(1.2)
                for team, opp, is_home in [(home, away, 1), (away, home, 0)]:
                    for pos, k in SQUAD_SIZE.items():
                        for j in range(k + 1):
                            skill = (j + 1) / (k + 1)
                            mins = int(rng.choice([0, 90], p=[0.2, 0.8]))
                            rows.append(dict(
                                season=season, gw=gw, fixture=fixture, kickoff_time=ko,
                                name=f"{team}-{pos}-{j}", player_key=f"{team}-{pos}-{j}",
                                team=team, opponent=opp, was_home=is_home, position=pos,
                                team_h_score=hs, team_a_score=as_, minutes=mins,
                                total_points=int(mins > 0) * (2 + rng.poisson(3 * skill)),
                                value=40 + 10 * j, selected=1000 * (j + 1),
                                **{c: rng.random() * (mins > 0) for c in [
                                    "goals_scored", "assists", "clean_sheets", "goals_conceded", "bonus",
                                    "bps", "ict_index", "influence", "creativity", "threat",
                                    "expected_goals", "expected_assists", "expected_goal_involvements",
                                    "expected_goals_conceded", "saves", "defensive_contribution", "starts"]},
                            ))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def feats():
    return build_features(synthetic_history())


def test_features_do_not_leak_current_match(feats):
    """Changing a match's outcome must not change that match's features."""
    hist = synthetic_history()
    target_idx = hist.index[(hist["player_key"] == "A-FWD-1") & (hist["gw"] == 5)][0]
    hist.loc[target_idx, ["total_points", "minutes", "expected_goals"]] = [99, 90, 9.9]
    altered = build_features(hist)
    cols = feature_columns(feats)
    key = (feats["player_key"] == "A-FWD-1") & (feats["gw"] == 5) & (feats["season"] == "2023-24")
    pd.testing.assert_frame_equal(
        feats.loc[key, cols].reset_index(drop=True), altered.loc[key, cols].reset_index(drop=True))
    # ...but the *next* match does see it
    nxt = (feats["player_key"] == "A-FWD-1") & (feats["gw"] == 6) & (feats["season"] == "2023-24")
    assert altered.loc[nxt, "total_points_r3"].iloc[0] > feats.loc[nxt, "total_points_r3"].iloc[0]


def test_first_appearance_has_no_history(feats):
    first = feats.sort_values("kickoff_time").groupby("player_key").head(1)
    assert first["total_points_r3"].isna().all()
    assert (first["matches_seen"] == 0).all()


def test_team_features_use_opponent(feats):
    row = feats[(feats["gw"] == 3) & (feats["team"] == "A")].iloc[0]
    opp = feats[(feats["gw"] == 3) & (feats["team"] == row["opponent"])].iloc[0]
    assert row["opp_goals_for_r6"] == pytest.approx(opp["team_goals_for_r6"])


def test_model_learns_signal(feats):
    cols = feature_columns(feats)
    train, test = feats[feats["season"] == "2023-24"], feats[feats["season"] == "2024-25"]
    model = PointsModel(cols, params={**PointsModel(cols).params, "min_child_samples": 20}).fit(train, num_boost_round=100)
    pred = model.predict(test)
    assert np.corrcoef(pred, test["total_points"])[0, 1] > 0.2


def test_squad_respects_fpl_rules():
    rng = np.random.default_rng(1)
    n = 120
    players = pd.DataFrame({
        "player_key": [f"p{i}" for i in range(n)],
        "position": rng.choice(list(SQUAD_SIZE), n),
        "team": rng.choice([f"T{i}" for i in range(10)], n),
        "value": rng.integers(40, 130, n),
    })
    players["pred"] = players["value"] / 20 + rng.random(n)
    squad = pick_squad(players, budget=1000)
    assert len(squad) == 15
    assert squad["value"].sum() <= 1000
    assert squad["position"].value_counts().to_dict() == SQUAD_SIZE
    assert squad["team"].value_counts().max() <= 3
    xi = squad[squad["starter"]]
    assert len(xi) == 11 and squad["captain"].sum() == 1
    assert (xi["position"] == "GK").sum() == 1 and (xi["position"] == "DEF").sum() >= 3
    assert xi.loc[xi["captain"], "pred"].iloc[0] == xi["pred"].max()
    assert squad_points(squad, "pred") == pytest.approx(xi["pred"].sum() + xi["pred"].max())
