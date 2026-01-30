#!/usr/bin/env python3
"""
Standalone NBA data update script for GitHub Actions.

Fetches the current season's game data from the NBA API and updates the CSV files.
Designed to run in GitHub Actions with no external dependencies beyond pandas and nba_api.

Based on the GLA admin CLI (backend/admin/cli.py).
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, TypeVar

import pandas as pd

from nba_api.stats.endpoints import leaguegamelog, boxscoresummaryv3, boxscoreadvancedv3


# ============================================================================
# Configuration
# ============================================================================

# Current working directory (repo root when run in GitHub Actions)
REPO_DIR = Path(__file__).parent.parent.resolve()

# Hard timeout for individual API calls (seconds)
API_HARD_TIMEOUT = 90

# Maximum games to fetch per run (to avoid rate limiting)
MAX_GAMES_PER_RUN = 100

# NBA Cup knockout dates (date -> game_type)
NBA_CUP_KNOCKOUT_DATES = {
    "2023-12-07": "nba_cup_semi",
    "2023-12-09": "nba_cup_final",
    "2024-12-14": "nba_cup_semi",
    "2024-12-17": "nba_cup_final",
    "2025-12-13": "nba_cup_semi",
    "2025-12-16": "nba_cup_final",
}

# Cancelled/invalid games to exclude (game was scheduled but never played)
# IND @ BOS on 2013-04-16 was cancelled due to Boston Marathon bombing, never rescheduled
CANCELLED_GAME_IDS = {
    "0021201214",  # 2012-13 IND @ BOS cancelled 4/16/2013
}


# ============================================================================
# Schema definitions (matches NBA_Data canonical format)
# ============================================================================

EXPECTED_COLUMNS = [
    "game_id", "game_date", "season", "game_type", "neutral_site",
    "team_id_home", "team_abbreviation_home", "team_name_home",
    "team_id_road", "team_abbreviation_road", "team_name_road",
    "pts_home", "pts_road", "wl_home",
    "fgm_home", "fga_home", "fg_pct_home", "fg3m_home", "fg3a_home", "fg3_pct_home",
    "ftm_home", "fta_home", "ft_pct_home",
    "oreb_home", "dreb_home", "reb_home", "ast_home", "stl_home", "blk_home", "tov_home", "pf_home", "plus_minus_home",
    "fgm_road", "fga_road", "fg_pct_road", "fg3m_road", "fg3a_road", "fg3_pct_road",
    "ftm_road", "fta_road", "ft_pct_road",
    "oreb_road", "dreb_road", "reb_road", "ast_road", "stl_road", "blk_road", "tov_road", "pf_road", "plus_minus_road",
]

INT_COLS = [
    "team_id_home", "team_id_road", "pts_home", "pts_road",
    "fgm_home", "fga_home", "fg3m_home", "fg3a_home", "ftm_home", "fta_home",
    "oreb_home", "dreb_home", "reb_home", "ast_home", "stl_home", "blk_home", "tov_home", "pf_home", "plus_minus_home",
    "fgm_road", "fga_road", "fg3m_road", "fg3a_road", "ftm_road", "fta_road",
    "oreb_road", "dreb_road", "reb_road", "ast_road", "stl_road", "blk_road", "tov_road", "pf_road", "plus_minus_road",
]

FLOAT_COLS = ["fg_pct_home", "fg3_pct_home", "ft_pct_home", "fg_pct_road", "fg3_pct_road", "ft_pct_road"]

STAT_MAP = {
    "pts": "PTS", "fgm": "FGM", "fga": "FGA", "fg_pct": "FG_PCT",
    "fg3m": "FG3M", "fg3a": "FG3A", "fg3_pct": "FG3_PCT",
    "ftm": "FTM", "fta": "FTA", "ft_pct": "FT_PCT",
    "oreb": "OREB", "dreb": "DREB", "reb": "REB", "ast": "AST",
    "stl": "STL", "blk": "BLK", "tov": "TOV", "pf": "PF", "plus_minus": "PLUS_MINUS",
}

LINESCORE_COLUMNS = [
    "game_id", "game_date", "season",
    "team_id_home", "team_abbreviation_home", "team_name_home",
    "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home", "pts_ot_total_home", "pts_home",
    "team_id_road", "team_abbreviation_road", "team_name_road",
    "pts_qtr1_road", "pts_qtr2_road", "pts_qtr3_road", "pts_qtr4_road", "pts_ot_total_road", "pts_road",
]

LINESCORE_INT_COLS = [
    "team_id_home", "team_id_road",
    "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home", "pts_ot_total_home", "pts_home",
    "pts_qtr1_road", "pts_qtr2_road", "pts_qtr3_road", "pts_qtr4_road", "pts_ot_total_road", "pts_road",
]

ADVANCED_COLUMNS = [
    "game_id", "game_date", "season",
    "team_id_home", "team_abbreviation_home", "minutes_home", "possessions_home",
    "team_id_road", "team_abbreviation_road", "minutes_road", "possessions_road",
]

ADVANCED_INT_COLS = ["team_id_home", "team_id_road", "minutes_home", "minutes_road"]
ADVANCED_FLOAT_COLS = ["possessions_home", "possessions_road"]


# ============================================================================
# Utility functions
# ============================================================================

T = TypeVar("T")


def _call_with_timeout(func: Callable[[], T], timeout: int = API_HARD_TIMEOUT) -> Optional[T]:
    """Execute a function with a hard timeout using a thread pool."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            print(f"[TIMEOUT after {timeout}s]", end=" ", flush=True)
            return None
        except Exception as e:
            raise e


def get_current_season() -> str:
    """Determine the current NBA season based on today's date."""
    today = datetime.now()
    year = today.year
    month = today.month
    # NBA season starts in October; if before October, use previous year
    if month < 10:
        start_year = year - 1
    else:
        start_year = year
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _snake_case(x: object) -> str:
    """Convert string to snake_case."""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if s == "regularseason":
        return "regular_season"
    return s


# ============================================================================
# Data normalization
# ============================================================================

def _normalize_game_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """Force exact NBA_Data game-log schema + dtypes."""
    d = df.copy()

    # Add missing columns and drop extras
    for c in EXPECTED_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[EXPECTED_COLUMNS]

    # game_type normalization
    d["game_type"] = d["game_type"].map(lambda v: _snake_case(v) if pd.notna(v) else v)

    # Date normalization to YYYY-MM-DD string
    gd = pd.to_datetime(d["game_date"], errors="coerce")
    d["game_date"] = gd.dt.date.astype("string")
    d.loc[gd.isna(), "game_date"] = pd.NA

    # neutral_site -> bool
    def _to_bool(v: object) -> bool:
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        s = str(v).strip().lower()
        return s in ("true", "1", "t", "yes", "y")

    d["neutral_site"] = d["neutral_site"].map(_to_bool).astype(bool)

    # Normalize game_id
    d["game_id"] = d["game_id"].astype("string")
    d["game_id"] = d["game_id"].str.strip()
    d["game_id"] = d["game_id"].str.replace(r"\.0$", "", regex=True)
    d.loc[d["game_id"].isin(["", "<NA>", "nan", "NaN", "None"]), "game_id"] = pd.NA
    d["game_id"] = d["game_id"].map(lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v)

    # Numeric coercions
    for c in INT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in FLOAT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Drop invalid rows and dedupe
    d = d.dropna(subset=["game_id"])
    d = d.drop_duplicates(subset=["game_id"], keep="first")

    # Remove known cancelled games
    cancelled_mask = d["game_id"].isin(CANCELLED_GAME_IDS)
    if cancelled_mask.any():
        print(f"[data] Filtering out {cancelled_mask.sum()} cancelled game(s)")
        d = d[~cancelled_mask].copy()

    # Sort by game_date
    if "game_date" in d.columns:
        d = d.sort_values("game_date")

    # Cast ints to true int64
    for c in INT_COLS:
        if d[c].isna().any():
            d[c] = d[c].astype("Int64")
        else:
            d[c] = d[c].astype("int64")

    d["game_id"] = d["game_id"].astype("string")

    for c in FLOAT_COLS:
        d[c] = d[c].astype("float64")

    return d


def _normalize_linescore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize linescore DataFrame to canonical schema."""
    d = df.copy()

    for c in LINESCORE_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[LINESCORE_COLUMNS]

    d["game_id"] = d["game_id"].astype("string").str.strip()
    d["game_id"] = d["game_id"].str.replace(r"\.0$", "", regex=True)
    d["game_id"] = d["game_id"].map(lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v)

    if "game_date" in d.columns:
        d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for c in LINESCORE_INT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")

    d = d.dropna(subset=["game_id"])
    d = d.drop_duplicates(subset=["game_id"], keep="first")
    if "game_date" in d.columns:
        d = d.sort_values("game_date")

    return d


def _normalize_advanced_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize advanced stats DataFrame to canonical schema."""
    d = df.copy()

    for c in ADVANCED_COLUMNS:
        if c not in d.columns:
            d[c] = pd.NA
    d = d[ADVANCED_COLUMNS]

    d["game_id"] = d["game_id"].astype("string").str.strip()
    d["game_id"] = d["game_id"].str.replace(r"\.0$", "", regex=True)
    d["game_id"] = d["game_id"].map(lambda v: v.zfill(10) if isinstance(v, str) and v.isdigit() else v)

    if "game_date" in d.columns:
        d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    for c in ADVANCED_INT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype("int64")
    for c in ADVANCED_FLOAT_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")

    d = d.dropna(subset=["game_id"])
    d = d.drop_duplicates(subset=["game_id"], keep="first")
    if "game_date" in d.columns:
        d = d.sort_values("game_date")

    return d


# ============================================================================
# NBA API fetching
# ============================================================================

def _fetch_season_team_game_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch team game logs for a given season using nba_api."""
    resp = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
    )
    df = resp.get_data_frames()[0]
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    return df


def _teamlogs_to_gamelogs(team_df: pd.DataFrame, season: str, game_type: str = "regular_season") -> pd.DataFrame:
    """Convert nba_api team-level logs -> NBA_Data game-level rows."""
    df = team_df.copy()

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    matchup = df.get("MATCHUP", pd.Series("", index=df.index)).astype(str)
    df["_IS_HOME"] = matchup.str.contains("vs.", na=False)
    df["_IS_AWAY"] = matchup.str.contains("@", na=False)

    out_rows: list[dict] = []
    df["GAME_ID"] = df["GAME_ID"].astype(str)

    for gid, g in df.groupby("GAME_ID"):
        home = g[g["_IS_HOME"]]
        away = g[g["_IS_AWAY"]]

        is_neutral_site = False

        if len(home) == 1 and len(away) == 1:
            h = home.iloc[0]
            a = away.iloc[0]
        elif len(home) == 0 and len(away) == 2:
            is_neutral_site = True
            row1, row2 = away.iloc[0], away.iloc[1]
            m1 = str(row1.get("MATCHUP", ""))
            if " @ " in m1:
                designated_home_abbr = m1.split(" @ ")[1].strip()
                if row1.get("TEAM_ABBREVIATION") == designated_home_abbr:
                    h, a = row1, row2
                else:
                    h, a = row2, row1
            else:
                if row1.get("TEAM_ABBREVIATION", "") < row2.get("TEAM_ABBREVIATION", ""):
                    h, a = row1, row2
                else:
                    h, a = row2, row1
        else:
            continue

        row: dict = {
            "game_id": str(gid),
            "game_date": (pd.to_datetime(h.get("GAME_DATE"), errors="coerce").date().isoformat()
                          if pd.notna(h.get("GAME_DATE")) else pd.NA),
            "season": season,
            "game_type": game_type,
            "neutral_site": is_neutral_site,
            "team_id_home": int(h.get("TEAM_ID")) if pd.notna(h.get("TEAM_ID")) else pd.NA,
            "team_abbreviation_home": h.get("TEAM_ABBREVIATION"),
            "team_name_home": h.get("TEAM_NAME"),
            "team_id_road": int(a.get("TEAM_ID")) if pd.notna(a.get("TEAM_ID")) else pd.NA,
            "team_abbreviation_road": a.get("TEAM_ABBREVIATION"),
            "team_name_road": a.get("TEAM_NAME"),
            "pts_home": h.get("PTS"),
            "pts_road": a.get("PTS"),
            "wl_home": h.get("WL"),
        }

        for stat_prefix, src_col in STAT_MAP.items():
            row[f"{stat_prefix}_home"] = h.get(src_col)
            row[f"{stat_prefix}_road"] = a.get(src_col)

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _fetch_linescore(game_id: str, game_date: str, season: str, home_team_id: int) -> Optional[dict]:
    """Fetch linescore data for a single game from BoxScoreSummaryV3."""
    def _do_fetch() -> Optional[dict]:
        resp = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=60)
        ls_df = resp.line_score.get_data_frame()

        if ls_df.empty or len(ls_df) < 2:
            return None

        home_row = ls_df[ls_df["teamId"] == home_team_id]
        road_row = ls_df[ls_df["teamId"] != home_team_id]

        if home_row.empty or road_row.empty:
            return None

        h = home_row.iloc[0]
        r = road_row.iloc[0]

        h_q1 = h.get("period1Score", 0) or 0
        h_q2 = h.get("period2Score", 0) or 0
        h_q3 = h.get("period3Score", 0) or 0
        h_q4 = h.get("period4Score", 0) or 0
        h_total = h.get("score", 0) or 0
        h_ot_total = h_total - (h_q1 + h_q2 + h_q3 + h_q4)

        r_q1 = r.get("period1Score", 0) or 0
        r_q2 = r.get("period2Score", 0) or 0
        r_q3 = r.get("period3Score", 0) or 0
        r_q4 = r.get("period4Score", 0) or 0
        r_total = r.get("score", 0) or 0
        r_ot_total = r_total - (r_q1 + r_q2 + r_q3 + r_q4)

        return {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "team_id_home": int(h.get("teamId", 0)),
            "team_abbreviation_home": h.get("teamTricode", ""),
            "team_name_home": h.get("teamName", ""),
            "pts_qtr1_home": h_q1, "pts_qtr2_home": h_q2, "pts_qtr3_home": h_q3, "pts_qtr4_home": h_q4,
            "pts_ot_total_home": h_ot_total, "pts_home": h_total,
            "team_id_road": int(r.get("teamId", 0)),
            "team_abbreviation_road": r.get("teamTricode", ""),
            "team_name_road": r.get("teamName", ""),
            "pts_qtr1_road": r_q1, "pts_qtr2_road": r_q2, "pts_qtr3_road": r_q3, "pts_qtr4_road": r_q4,
            "pts_ot_total_road": r_ot_total, "pts_road": r_total,
        }

    for attempt in range(3):
        try:
            result = _call_with_timeout(_do_fetch)
            if result is not None:
                return result
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"[err:{e}]", end=" ", flush=True)
    return None


def _fetch_advanced_stats(game_id: str, game_date: str, season: str, home_team_id: int) -> Optional[dict]:
    """Fetch possessions for a single game from BoxScoreAdvancedV3."""
    def _do_fetch() -> Optional[dict]:
        resp = boxscoreadvancedv3.BoxScoreAdvancedV3(
            game_id=game_id, start_period=0, end_period=0,
            start_range=0, end_range=28800, range_type=0, timeout=60,
        )
        team_df = resp.team_stats.get_data_frame()

        if team_df.empty or len(team_df) < 2:
            return None

        home_row = team_df[team_df["teamId"] == home_team_id]
        road_row = team_df[team_df["teamId"] != home_team_id]

        if home_row.empty or road_row.empty:
            return None

        h = home_row.iloc[0]
        r = road_row.iloc[0]

        def parse_minutes(mins_str: str) -> int:
            if not mins_str:
                return 0
            parts = str(mins_str).split(":")
            try:
                return int(parts[0])
            except (ValueError, IndexError):
                return 0

        return {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "team_id_home": int(h.get("teamId", 0)),
            "team_abbreviation_home": h.get("teamTricode", ""),
            "minutes_home": parse_minutes(h.get("minutes", "")),
            "possessions_home": h.get("possessions", 0.0),
            "team_id_road": int(r.get("teamId", 0)),
            "team_abbreviation_road": r.get("teamTricode", ""),
            "minutes_road": parse_minutes(r.get("minutes", "")),
            "possessions_road": r.get("possessions", 0.0),
        }

    for attempt in range(3):
        try:
            result = _call_with_timeout(_do_fetch)
            if result is not None:
                return result
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"[err:{e}]", end=" ", flush=True)
    return None


# ============================================================================
# Main update function
# ============================================================================

def update_data(season: str) -> int:
    """Update data for a single season."""
    start = time.time()

    gamelog_path = REPO_DIR / f"team_game_logs_{season}.csv"
    linescore_path = REPO_DIR / f"linescores_{season}.csv"
    advanced_path = REPO_DIR / f"box_score_advanced_{season}.csv"

    # Load existing data
    existing: Optional[pd.DataFrame] = None
    if gamelog_path.exists():
        existing = pd.read_csv(gamelog_path, dtype={"game_id": "string"})
        existing = _normalize_game_level_df(existing)

    # Fetch all season types
    season_types = [
        ("Regular Season", "regular_season"),
        ("IST", "regular_season"),  # NBA Cup games
        ("Playoffs", "playoffs"),
        ("PlayIn", "play_in"),
    ]

    all_gamelogs: list[pd.DataFrame] = []
    for api_type, game_type_label in season_types:
        print(f"[data] Fetching {season} {api_type} from NBA API...")
        try:
            team_logs = _fetch_season_team_game_logs(season, season_type=api_type)
            if not team_logs.empty:
                gamelogs = _teamlogs_to_gamelogs(team_logs, season=season, game_type=game_type_label)
                if not gamelogs.empty:
                    print(f"[data]   Found {len(gamelogs)} {api_type} games")
                    all_gamelogs.append(gamelogs)
                else:
                    print(f"[data]   No {api_type} games found")
            else:
                print(f"[data]   No {api_type} games found")
        except Exception as e:
            print(f"[data]   Error fetching {api_type}: {e}")

    if not all_gamelogs:
        print("[data] No games found for any season type")
        return 1

    print("[data] Combining all game types...")
    fresh_raw = pd.concat(all_gamelogs, ignore_index=True)

    # Apply NBA Cup knockout date overrides
    for idx, row in fresh_raw.iterrows():
        if row.get("game_type") == "regular_season":
            game_date = str(row.get("game_date", "")).strip()
            if game_date in NBA_CUP_KNOCKOUT_DATES:
                fresh_raw.at[idx, "game_type"] = NBA_CUP_KNOCKOUT_DATES[game_date]

    fresh = _normalize_game_level_df(fresh_raw)

    # Skip today's games
    today_str = datetime.now().strftime("%Y-%m-%d")
    games_today = fresh["game_date"] == today_str
    if games_today.any():
        skipped = games_today.sum()
        print(f"[data] Skipping {skipped} game(s) from today ({today_str}) - may be in progress")
        fresh = fresh[~games_today].copy()

    # Merge with existing
    if existing is None or existing.empty:
        merged = fresh
        added = len(fresh)
    else:
        existing_ids = set(existing["game_id"].astype(str).tolist())
        fresh_new = fresh[~fresh["game_id"].astype(str).isin(existing_ids)].copy()
        added = len(fresh_new)
        merged = pd.concat([existing, fresh_new], ignore_index=True)
        merged = _normalize_game_level_df(merged)

    merged.to_csv(gamelog_path, index=False)
    print(f"[data] Saved {gamelog_path.name} ({len(merged)} rows, +{added} new)")

    # Fetch boxscore data for new games
    existing_ls: Optional[pd.DataFrame] = None
    existing_adv: Optional[pd.DataFrame] = None
    if linescore_path.exists():
        existing_ls = pd.read_csv(linescore_path, dtype={"game_id": "string"})
    if advanced_path.exists():
        existing_adv = pd.read_csv(advanced_path, dtype={"game_id": "string"})

    all_game_ids = set(merged[merged["game_date"] != today_str]["game_id"].astype(str).tolist())
    existing_ls_ids = set(existing_ls["game_id"].astype(str).tolist()) if existing_ls is not None else set()
    existing_adv_ids = set(existing_adv["game_id"].astype(str).tolist()) if existing_adv is not None else set()

    already_fetched = existing_ls_ids & existing_adv_ids
    new_game_ids = list(all_game_ids - already_fetched)[:MAX_GAMES_PER_RUN]

    if new_game_ids:
        print(f"\n[data] Fetching boxscore data for {len(new_game_ids)} games...")

        game_info: Dict[str, tuple] = {}
        for _, row in merged.iterrows():
            gid = str(row["game_id"])
            if gid in new_game_ids:
                game_info[gid] = (row["game_date"], int(row["team_id_home"]))

        linescore_rows: list[dict] = []
        advanced_rows: list[dict] = []

        for i, gid in enumerate(new_game_ids, 1):
            game_date, home_team_id = game_info.get(gid, ("", 0))
            print(f"  [{i}/{len(new_game_ids)}] Fetching {gid}...", end=" ", flush=True)

            ls_row = _fetch_linescore(gid, game_date, season, home_team_id)
            if ls_row:
                linescore_rows.append(ls_row)
                print("LS:OK", end=" ", flush=True)
            else:
                print("LS:FAIL", end=" ", flush=True)
            time.sleep(1.0)

            adv_row = _fetch_advanced_stats(gid, game_date, season, home_team_id)
            if adv_row:
                advanced_rows.append(adv_row)
                print("ADV:OK")
            else:
                print("ADV:FAIL")
            time.sleep(1.0)

        # Save linescore
        if linescore_rows:
            new_ls = pd.DataFrame(linescore_rows)
            if existing_ls is not None and not existing_ls.empty:
                combined_ls = pd.concat([existing_ls, new_ls], ignore_index=True)
            else:
                combined_ls = new_ls
            combined_ls = _normalize_linescore_df(combined_ls)
            combined_ls.to_csv(linescore_path, index=False)
            print(f"[data] Saved {linescore_path.name} ({len(combined_ls)} rows, +{len(linescore_rows)} new)")

        # Save advanced
        if advanced_rows:
            new_adv = pd.DataFrame(advanced_rows)
            if existing_adv is not None and not existing_adv.empty:
                combined_adv = pd.concat([existing_adv, new_adv], ignore_index=True)
            else:
                combined_adv = new_adv
            combined_adv = _normalize_advanced_df(combined_adv)
            combined_adv.to_csv(advanced_path, index=False)
            print(f"[data] Saved {advanced_path.name} ({len(combined_adv)} rows, +{len(advanced_rows)} new)")
    else:
        print("[data] No new games need boxscore data")

    elapsed = time.time() - start
    print(f"\n[data] Update complete ({elapsed:.1f}s)")
    return 0


def main():
    """Main entry point."""
    season = get_current_season()
    print(f"[data] Updating season: {season}")
    print(f"[data] Repository: {REPO_DIR}")
    return update_data(season)


if __name__ == "__main__":
    exit(main())
