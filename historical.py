"""
Historical data layer for FPL Fixture Value Index (Phase 2).

Source: https://github.com/vaastav/Fantasy-Premier-League

Phase 1 metrics:
  - hist_ppg, last_season_ppg, hist_minutes, hist_minutes_pct, seasons_played

Phase 2 additions:
  - hist_ppg_home / hist_ppg_away  (minutes-weighted)
  - consistency_score              (0–100, higher = more consistent)
  - n_starts                       (games with 60+ minutes)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json
import time

import numpy as np
import pandas as pd
import requests


RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Completed seasons used for stable baselines
DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25"]


class HistoricalStore:
    """
    Loads and caches historical player metrics keyed by FPL `code`.
    """

    def __init__(
        self,
        seasons: Optional[List[str]] = None,
        cache_dir: str = "cache",
        cache_hours: int = 24 * 7,
    ):
        self.seasons = seasons or DEFAULT_SEASONS
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hours = cache_hours
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "FPL-Fixture-Value-Index/1.3", "Accept": "text/csv"}
        )
        self._lookup: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> "HistoricalStore":
        cache_file = self.cache_dir / "historical_lookup_v2.parquet"
        meta_file = self.cache_dir / "historical_meta_v2.json"

        if cache_file.exists() and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                age = time.time() - meta.get("built_at", 0)
                if age < self.cache_hours * 3600 and meta.get("seasons") == self.seasons:
                    self._lookup = pd.read_parquet(cache_file)
                    return self
            except Exception:
                pass

        season_frames = []
        gw_frames = []

        for season in self.seasons:
            raw = self._load_players_raw(season)
            if raw is not None and not raw.empty:
                season_frames.append(raw)

            gw = self._load_merged_gw(season, raw)
            if gw is not None and not gw.empty:
                gw_frames.append(gw)

        if not season_frames:
            self._lookup = pd.DataFrame()
            return self

        season_combined = pd.concat(season_frames, ignore_index=True)
        gw_combined = (
            pd.concat(gw_frames, ignore_index=True) if gw_frames else pd.DataFrame()
        )

        self._lookup = self._aggregate(season_combined, gw_combined)

        self._lookup.to_parquet(cache_file, index=False)
        meta_file.write_text(
            json.dumps({"built_at": time.time(), "seasons": self.seasons, "version": 2}),
            encoding="utf-8",
        )
        return self

    def get_lookup(self) -> pd.DataFrame:
        if self._lookup is None:
            self.load()
        return self._lookup if self._lookup is not None else pd.DataFrame()

    def metrics_for_code(self, code: int) -> Dict:
        df = self.get_lookup()
        if df.empty or "code" not in df.columns:
            return {}
        row = df.loc[df["code"] == int(code)]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {k: r.get(k) for k in r.index if k != "web_name"}

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    def _load_players_raw(self, season: str) -> Optional[pd.DataFrame]:
        url = f"{RAW_BASE}/{season}/players_raw.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            return None

        required = ["code", "total_points", "minutes"]
        if not all(c in df.columns for c in required):
            return None

        out = df[
            [c for c in [
                "code", "id", "total_points", "minutes", "points_per_game",
                "web_name", "first_name", "second_name", "element_type", "team",
            ] if c in df.columns]
        ].copy()

        out["season"] = season
        out["code"] = pd.to_numeric(out["code"], errors="coerce")
        out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(0)
        out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0)
        out["points_per_game"] = pd.to_numeric(
            out.get("points_per_game", 0), errors="coerce"
        ).fillna(0)

        out = out.dropna(subset=["code"])
        out = out[out["minutes"] > 0]
        return out

    def _load_merged_gw(
        self, season: str, players_raw: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Load gameweek-level data and attach stable `code`."""
        url = f"{RAW_BASE}/{season}/gws/merged_gw.csv"
        try:
            gw = pd.read_csv(url)
        except Exception:
            return None

        if "element" not in gw.columns or "total_points" not in gw.columns:
            return None

        gw = gw.copy()
        gw["element"] = pd.to_numeric(gw["element"], errors="coerce")
        gw["total_points"] = pd.to_numeric(gw["total_points"], errors="coerce").fillna(0)
        gw["minutes"] = pd.to_numeric(gw.get("minutes", 0), errors="coerce").fillna(0)

        if "was_home" in gw.columns:
            # was_home can be bool or string
            gw["was_home"] = gw["was_home"].map(
                lambda x: True if str(x).lower() in {"true", "1", "yes"} else False
            )
        else:
            gw["was_home"] = False

        # Map element (season id) → code
        if players_raw is not None and "id" in players_raw.columns:
            id_to_code = (
                players_raw.dropna(subset=["id", "code"])
                .drop_duplicates("id")
                .set_index("id")["code"]
                .to_dict()
            )
            gw["code"] = gw["element"].map(id_to_code)
        else:
            gw["code"] = np.nan

        gw = gw.dropna(subset=["code"])
        gw["code"] = gw["code"].astype(int)
        gw["season"] = season
        gw = gw[gw["minutes"] > 0]
        return gw[["code", "season", "total_points", "minutes", "was_home"]]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(
        self, season_combined: pd.DataFrame, gw_combined: pd.DataFrame
    ) -> pd.DataFrame:
        rows = []

        for code, g in season_combined.groupby("code"):
            g = g.sort_values("season")
            last = g.iloc[-1]

            total_mins = float(g["minutes"].sum())
            total_pts = float(g["total_points"].sum())
            seasons_played = int(g["season"].nunique())

            hist_ppg = total_pts / (total_mins / 90.0) if total_mins > 0 else 0.0

            last_ppg = float(last["points_per_game"]) if last["points_per_game"] > 0 else (
                float(last["total_points"]) / max(float(last["minutes"]) / 90.0, 0.1)
            )

            theoretical = seasons_played * 38 * 90
            minutes_pct = min(1.0, total_mins / theoretical) if theoretical > 0 else 0.0

            # --- Phase 2: home / away / consistency from gameweek data ---
            hist_ppg_home = hist_ppg
            hist_ppg_away = hist_ppg
            consistency_score = 50.0
            n_starts = 0

            if not gw_combined.empty:
                pg = gw_combined[gw_combined["code"] == int(code)]
                if not pg.empty:
                    home = pg[pg["was_home"] == True]
                    away = pg[pg["was_home"] == False]

                    def ppg_from(subset: pd.DataFrame) -> float:
                        m = float(subset["minutes"].sum())
                        if m <= 0:
                            return hist_ppg
                        return float(subset["total_points"].sum()) / (m / 90.0)

                    if not home.empty:
                        hist_ppg_home = ppg_from(home)
                    if not away.empty:
                        hist_ppg_away = ppg_from(away)

                    # Consistency from starts (60+ minutes)
                    starts = pg[pg["minutes"] >= 60]["total_points"]
                    n_starts = int(len(starts))
                    if n_starts >= 5:
                        mean_pts = float(starts.mean())
                        std_pts = float(starts.std(ddof=0))
                        if mean_pts > 0.1:
                            cv = std_pts / mean_pts
                            # Higher consistency_score = more reliable
                            # cv=0 → 100, cv=1 → ~50, cv=2 → ~33
                            consistency_score = 100.0 / (1.0 + cv)
                        else:
                            consistency_score = 40.0
                    elif n_starts > 0:
                        consistency_score = 45.0

            rows.append(
                {
                    "code": int(code),
                    "web_name": last.get("web_name", ""),
                    "hist_ppg": round(hist_ppg, 3),
                    "last_season_ppg": round(last_ppg, 3),
                    "hist_ppg_home": round(hist_ppg_home, 3),
                    "hist_ppg_away": round(hist_ppg_away, 3),
                    "hist_minutes": total_mins,
                    "hist_minutes_pct": round(minutes_pct, 3),
                    "seasons_played": seasons_played,
                    "consistency_score": round(consistency_score, 1),
                    "n_starts": n_starts,
                }
            )

        return pd.DataFrame(rows)
