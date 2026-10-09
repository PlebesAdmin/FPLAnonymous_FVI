"""
Historical data layer for FPL Fixture Value Index.

Uses the excellent open dataset from:
https://github.com/vaastav/Fantasy-Premier-League

We load the last few completed seasons, aggregate key metrics per player
(using the stable FPL `code`), and expose a simple lookup that the live
engine can merge against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json
import time

import pandas as pd
import requests


RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Completed seasons we trust for stable baselines.
# (Weekly updates stopped after 2024-25; these are full seasons.)
DEFAULT_SEASONS = ["2022-23", "2023-24", "2024-25"]


class HistoricalStore:
    """
    Loads and caches historical player metrics keyed by FPL `code`.

    Metrics produced:
      - hist_ppg          : minutes-weighted average points per game across seasons
      - last_season_ppg   : PPG from the most recent completed season available
      - hist_minutes      : total minutes across the seasons we loaded
      - hist_minutes_pct  : rough reliability proxy (minutes / (38*90 * seasons))
      - seasons_played    : how many of the loaded seasons the player appeared in
    """

    def __init__(
        self,
        seasons: Optional[List[str]] = None,
        cache_dir: str = "cache",
        cache_hours: int = 24 * 7,  # historical data changes rarely
    ):
        self.seasons = seasons or DEFAULT_SEASONS
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hours = cache_hours
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "FPL-Fixture-Value-Index/1.2", "Accept": "text/csv"}
        )
        self._lookup: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self) -> "HistoricalStore":
        cache_file = self.cache_dir / "historical_lookup.parquet"
        meta_file = self.cache_dir / "historical_meta.json"

        if cache_file.exists() and meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                age = time.time() - meta.get("built_at", 0)
                if age < self.cache_hours * 3600 and meta.get("seasons") == self.seasons:
                    self._lookup = pd.read_parquet(cache_file)
                    return self
            except Exception:
                pass  # fall through and rebuild

        frames = []
        for season in self.seasons:
            df = self._load_season(season)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            # Graceful degradation – engine will simply ignore historical
            self._lookup = pd.DataFrame()
            return self

        combined = pd.concat(frames, ignore_index=True)
        self._lookup = self._aggregate(combined)

        # Persist
        self._lookup.to_parquet(cache_file, index=False)
        meta_file.write_text(
            json.dumps({"built_at": time.time(), "seasons": self.seasons}),
            encoding="utf-8",
        )
        return self

    def get_lookup(self) -> pd.DataFrame:
        if self._lookup is None:
            self.load()
        return self._lookup if self._lookup is not None else pd.DataFrame()

    def metrics_for_code(self, code: int) -> Dict:
        """Return a plain dict of historical metrics for one player code."""
        df = self.get_lookup()
        if df.empty or "code" not in df.columns:
            return {}
        row = df.loc[df["code"] == int(code)]
        if row.empty:
            return {}
        r = row.iloc[0]
        return {
            "hist_ppg": float(r.get("hist_ppg", 0)),
            "last_season_ppg": float(r.get("last_season_ppg", 0)),
            "hist_minutes": float(r.get("hist_minutes", 0)),
            "hist_minutes_pct": float(r.get("hist_minutes_pct", 0)),
            "seasons_played": int(r.get("seasons_played", 0)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_season(self, season: str) -> Optional[pd.DataFrame]:
        """Load players_raw for a season and keep the columns we need."""
        url = f"{RAW_BASE}/{season}/players_raw.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            return None

        # Standardise
        needed = {
            "code": "code",
            "total_points": "total_points",
            "minutes": "minutes",
            "points_per_game": "points_per_game",
            "web_name": "web_name",
            "first_name": "first_name",
            "second_name": "second_name",
            "element_type": "element_type",
            "team": "team",
        }
        present = [c for c in needed if c in df.columns]
        if "code" not in present or "total_points" not in present:
            return None

        out = df[present].copy()
        out["season"] = season
        out["code"] = pd.to_numeric(out["code"], errors="coerce")
        out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(0)
        out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0)
        out["points_per_game"] = pd.to_numeric(
            out.get("points_per_game", 0), errors="coerce"
        ).fillna(0)

        # Drop rows with no code or zero minutes (never played)
        out = out.dropna(subset=["code"])
        out = out[out["minutes"] > 0]
        return out

    def _aggregate(self, combined: pd.DataFrame) -> pd.DataFrame:
        """
        Build one row per player code with the historical metrics.
        """
        # Minutes-weighted PPG across all loaded seasons
        def weighted_ppg(g: pd.DataFrame) -> float:
            mins = g["minutes"].sum()
            if mins <= 0:
                return 0.0
            # total points / (minutes / 90) ≈ PPG
            return float(g["total_points"].sum() / (mins / 90.0))

        rows = []
        for code, g in combined.groupby("code"):
            g = g.sort_values("season")
            last = g.iloc[-1]

            hist_ppg = weighted_ppg(g)
            last_ppg = float(last["points_per_game"]) if last["points_per_game"] > 0 else (
                float(last["total_points"]) / max(last["minutes"] / 90.0, 0.1)
            )
            total_mins = float(g["minutes"].sum())
            seasons_played = int(g["season"].nunique())

            # Rough reliability: actual minutes vs theoretical max (38*90 per season)
            theoretical = seasons_played * 38 * 90
            minutes_pct = min(1.0, total_mins / theoretical) if theoretical > 0 else 0.0

            rows.append(
                {
                    "code": int(code),
                    "web_name": last.get("web_name", ""),
                    "hist_ppg": round(hist_ppg, 3),
                    "last_season_ppg": round(last_ppg, 3),
                    "hist_minutes": total_mins,
                    "hist_minutes_pct": round(minutes_pct, 3),
                    "seasons_played": seasons_played,
                }
            )

        return pd.DataFrame(rows)
