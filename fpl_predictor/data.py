"""Download and cache historical FPL gameweek data.

Source: the community-maintained archive at
https://github.com/vaastav/Fantasy-Premier-League, which snapshots the
official FPL API after every gameweek.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DEFAULT_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
POSITIONS = ["GK", "DEF", "MID", "FWD"]

# Columns that exist in some seasons but not others; filled with 0 where absent.
OPTIONAL_STATS = [
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "starts", "defensive_contribution",
    "clearances_blocks_interceptions", "recoveries", "tackles",
]


def _fetch(url: str, dest: Path, refresh: bool = False) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_season(season: str, refresh: bool = False) -> pd.DataFrame:
    """Load one season of per-fixture player rows (one row per player per fixture)."""
    path = _fetch(f"{BASE_URL}/{season}/gws/merged_gw.csv", CACHE_DIR / season / "merged_gw.csv", refresh)
    raw = path.read_bytes()
    try:
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip")
    except UnicodeDecodeError:  # older seasons are latin-1 encoded
        df = pd.read_csv(io.BytesIO(raw), on_bad_lines="skip", encoding="latin-1")
    df["season"] = season
    for col in OPTIONAL_STATS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def load_fixtures(season: str, refresh: bool = False) -> pd.DataFrame:
    path = _fetch(f"{BASE_URL}/{season}/fixtures.csv", CACHE_DIR / season / "fixtures.csv", refresh)
    fx = pd.read_csv(path, usecols=["id", "event", "finished", "kickoff_time", "team_h", "team_a"])
    fx["season"] = season
    return fx


def load_teams(season: str, refresh: bool = False) -> pd.DataFrame:
    path = _fetch(f"{BASE_URL}/{season}/teams.csv", CACHE_DIR / season / "teams.csv", refresh)
    return pd.read_csv(path, usecols=["id", "name"])


def load_players_raw(season: str, refresh: bool = False) -> pd.DataFrame:
    path = _fetch(f"{BASE_URL}/{season}/players_raw.csv", CACHE_DIR / season / "players_raw.csv", refresh)
    return pd.read_csv(path)


def load_all(seasons: list[str] | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load and clean all seasons into one tidy frame sorted chronologically."""
    seasons = seasons or DEFAULT_SEASONS
    frames = []
    for s in seasons:
        df = load_season(s, refresh)
        teams = load_teams(s, refresh).set_index("id")["name"]
        df["opponent"] = df["opponent_team"].map(teams)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df[df["position"].isin(POSITIONS)].copy()  # drops 2024-25 "AM" (assistant manager) rows
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    df["was_home"] = df["was_home"].astype(str).str.lower().eq("true").astype(int)
    df = df.rename(columns={"GW": "gw"})
    # Players are re-numbered every season; the name is the stable cross-season key.
    df["player_key"] = df["name"].str.strip()
    return df.sort_values(["kickoff_time", "fixture", "player_key"]).reset_index(drop=True)
