from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json
import math
import time

import numpy as np
import pandas as pd
import requests


FPL_BASE = "https://fantasy.premierleague.com/api"


@dataclass
class FVIConfig:
    fixture_weight: float = 0.35
    points_weight: float = 0.30
    value_weight: float = 0.20
    differential_weight: float = 0.15

    # FDR 3 = neutral. FDR 1 ≈ +20%, FDR 5 ≈ -20% at sensitivity 0.10
    fixture_sensitivity: float = 0.10

    # Baseline points blend (used for single / short horizons)
    ep_next_weight: float = 0.45
    form_weight: float = 0.30
    ppg_weight: float = 0.25


class FPLClient:
    """Defensive client for the public FPL API with simple file caching."""

    def __init__(self, cache_dir: str = "cache", cache_hours: int = 1):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hours = cache_hours
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "FPL Fixture Value Index/1.1",
                "Accept": "application/json",
            }
        )

    def _get(self, path: str, cache_name: str):
        cache_file = self.cache_dir / cache_name

        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < self.cache_hours * 3600:
                return json.loads(cache_file.read_text(encoding="utf-8"))

        try:
            response = self.session.get(f"{FPL_BASE}/{path}", timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            # Fall back to stale cache if available
            if cache_file.exists():
                return json.loads(cache_file.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"Could not reach the FPL API ({path}). "
                "Check your internet connection or try again later."
            ) from exc

        cache_file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def bootstrap(self):
        return self._get("bootstrap-static/", "bootstrap-static.json")

    def fixtures(self):
        return self._get("fixtures/", "fixtures.json")


class FPLFixtureValueAgent:
    """
    Calculates a transparent Fixture Value Index (FVI) for FPL players
    over a selected gameweek range.

    Components (default weights):
      1. Fixture quality   35%
      2. Points potential  30%
      3. Price efficiency  20%
      4. Differential      15%

    Final FVI is normalised to 0–100.
    """

    def __init__(
        self,
        client: Optional[FPLClient] = None,
        config: Optional[FVIConfig] = None,
    ):
        self.client = client or FPLClient()
        self.config = config or FVIConfig()
        self.bootstrap_data = None
        self.fixtures_data = None
        # Pre-built lookup tables (filled on load)
        self._players: Optional[pd.DataFrame] = None
        self._fixtures_by_team: Dict[int, List[dict]] = {}

    def load(self):
        self.bootstrap_data = self.client.bootstrap()
        self.fixtures_data = self.client.fixtures()
        self._build_lookup_tables()
        return self

    def _build_lookup_tables(self):
        """Build player table and a per-team fixture list once."""
        b = self.bootstrap_data
        teams = pd.DataFrame(b["teams"])
        players = pd.DataFrame(b["elements"])

        team_map = teams.set_index("id")["name"].to_dict()
        short_map = teams.set_index("id")["short_name"].to_dict()
        position_map = {
            x["id"]: x["singular_name_short"]
            for x in b.get("element_types", [])
        }

        players["team_name"] = players["team"].map(team_map)
        players["team_short"] = players["team"].map(short_map)
        players["position"] = players["element_type"].map(position_map)
        players["price"] = players["now_cost"] / 10.0
        players["ownership"] = pd.to_numeric(
            players["selected_by_percent"], errors="coerce"
        ).fillna(0.0)

        self._players = players

        # Pre-group fixtures by team for fast lookup
        fixtures = pd.DataFrame(self.fixtures_data)
        self._fixtures_by_team = {}

        for _, f in fixtures.iterrows():
            if pd.isna(f.get("event")):
                continue
            gw = int(f["event"])
            home_id = int(f["team_h"])
            away_id = int(f["team_a"])

            home_diff = self._safe_float(f.get("team_h_difficulty"), 3.0)
            away_diff = self._safe_float(f.get("team_a_difficulty"), 3.0)

            self._fixtures_by_team.setdefault(home_id, []).append(
                {
                    "gw": gw,
                    "opponent_id": away_id,
                    "home": True,
                    "difficulty": home_diff,
                }
            )
            self._fixtures_by_team.setdefault(away_id, []).append(
                {
                    "gw": gw,
                    "opponent_id": home_id,
                    "home": False,
                    "difficulty": away_diff,
                }
            )

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _minmax(series: pd.Series, reverse: bool = False) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0.0)
        lo, hi = s.min(), s.max()
        if math.isclose(lo, hi):
            out = pd.Series(50.0, index=s.index)
        else:
            out = 100.0 * (s - lo) / (hi - lo)
        return 100.0 - out if reverse else out

    def _fixture_rows(self, team_id: int, gw_start: int, gw_end: int) -> List[dict]:
        """Return fixtures for a team inside the GW window (already pre-built)."""
        all_fx = self._fixtures_by_team.get(int(team_id), [])
        return [f for f in all_fx if gw_start <= f["gw"] <= gw_end]

    def _base_points_rate(self, p: pd.Series, num_gws: int) -> float:
        """
        Transparent baseline.
        For short horizons lean more on ep_next; for longer horizons lean on form + PPG.
        """
        ep_next = self._safe_float(p.get("ep_next"), 0.0)
        form = self._safe_float(p.get("form"), 0.0)
        ppg = self._safe_float(p.get("points_per_game"), 0.0)

        if num_gws <= 2:
            # Near-term: trust FPL's own expected points more
            w_ep, w_form, w_ppg = 0.55, 0.25, 0.20
        elif num_gws <= 5:
            w_ep, w_form, w_ppg = (
                self.config.ep_next_weight,
                self.config.form_weight,
                self.config.ppg_weight,
            )
        else:
            # Longer run: form + season average matter more
            w_ep, w_form, w_ppg = 0.25, 0.35, 0.40

        base = w_ep * ep_next + w_form * form + w_ppg * ppg
        return max(base, 0.1)

    def _availability_factor(self, p: pd.Series) -> float:
        chance = p.get("chance_of_playing_next_round")
        if chance is None:
            chance = 100
        chance = self._safe_float(chance, 100)
        status = str(p.get("status", ""))

        # Unavailable / suspended
        if status in {"u", "s"}:
            return 0.0
        return max(0.0, min(1.0, chance / 100.0))

    def calculate(
        self,
        gw_start: int,
        gw_end: int,
        min_minutes: int = 90,
        config: Optional[FVIConfig] = None,
    ) -> pd.DataFrame:
        if self._players is None:
            self.load()

        cfg = config or self.config
        num_gws = max(1, gw_end - gw_start + 1)
        players = self._players

        rows = []

        for _, p in players.iterrows():
            minutes = self._safe_float(p.get("minutes"), 0)
            if minutes < min_minutes:
                continue

            price = self._safe_float(p.get("price"), 0)
            if price <= 0:
                continue

            fixtures = self._fixture_rows(int(p["team"]), gw_start, gw_end)

            if fixtures:
                difficulties = [f["difficulty"] for f in fixtures]
                avg_fdr = float(np.mean(difficulties))
            else:
                avg_fdr = 5.0  # blank window treated as hard (no points)

            base = self._base_points_rate(p, num_gws)
            availability = self._availability_factor(p)

            projected_points = 0.0
            fixture_details = []

            for f in fixtures:
                difficulty = f["difficulty"]
                # FDR 3 neutral → FDR 1 ≈ +20 %, FDR 5 ≈ -20 %
                fixture_multiplier = 1.0 + (
                    (3.0 - difficulty) * cfg.fixture_sensitivity
                )
                home_multiplier = 1.03 if f["home"] else 0.98

                pts = base * fixture_multiplier * home_multiplier * availability
                projected_points += max(0.0, pts)

                fixture_details.append(
                    {
                        "gw": f["gw"],
                        "difficulty": difficulty,
                        "home": f["home"],
                    }
                )

            # Blank window → zero projected points
            if not fixtures:
                projected_points = 0.0

            rows.append(
                {
                    "id": int(p["id"]),
                    "player": p["web_name"],
                    "first_name": p.get("first_name", ""),
                    "second_name": p.get("second_name", ""),
                    "team": p["team_name"],
                    "team_short": p["team_short"],
                    "position": p["position"],
                    "price": price,
                    "ownership": self._safe_float(p.get("selected_by_percent")),
                    "total_points": self._safe_float(p.get("total_points")),
                    "ppg": self._safe_float(p.get("points_per_game")),
                    "form": self._safe_float(p.get("form")),
                    "ep_next": self._safe_float(p.get("ep_next")),
                    "xgi": (
                        self._safe_float(p.get("expected_goals"))
                        + self._safe_float(p.get("expected_assists"))
                    ),
                    "minutes": minutes,
                    "avg_fdr": avg_fdr,
                    "fixture_count": len(fixtures),
                    "fixture_details": fixture_details,
                    "projected_points": projected_points,
                    "availability": availability,
                    "news": p.get("news", "") or "",
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Component scores (0–100)
        df["fixture_score"] = self._minmax(df["avg_fdr"], reverse=True)
        df["points_score"] = self._minmax(df["projected_points"])

        df["points_per_million"] = (
            df["projected_points"] / df["price"].replace(0, np.nan)
        )
        df["points_per_million"] = (
            df["points_per_million"]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        df["value_score"] = self._minmax(df["points_per_million"])
        df["differential_score"] = self._minmax(df["ownership"], reverse=True)

        df["fvi"] = (
            df["fixture_score"] * cfg.fixture_weight
            + df["points_score"] * cfg.points_weight
            + df["value_score"] * cfg.value_weight
            + df["differential_score"] * cfg.differential_weight
        )

        # Soft penalty for availability issues
        df["fvi"] *= df["availability"].clip(lower=0.0, upper=1.0)

        df["recommendation"] = np.select(
            [
                df["fvi"] >= 80,
                df["fvi"] >= 70,
                df["fvi"] >= 60,
            ],
            [
                "🔥 Strong buy",
                "🟢 Buy / consider",
                "🟡 Watchlist",
            ],
            default="⚪ Avoid / monitor",
        )

        df = df.sort_values(
            ["fvi", "projected_points"], ascending=[False, False]
        ).reset_index(drop=True)

        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df

    @staticmethod
    def fixtures_string(row) -> str:
        parts = []
        for f in row["fixture_details"]:
            venue = "H" if f["home"] else "A"
            parts.append(f"GW{f['gw']} {venue} FDR{int(f['difficulty'])}")
        return " | ".join(parts) if parts else "No fixtures"

    def explain_player(self, row: pd.Series) -> str:
        return (
            f"{row['player']} ({row['team']}) scores {row['fvi']:.1f}/100. "
            f"Average FDR {row['avg_fdr']:.2f}, projected {row['projected_points']:.1f} "
            f"points over the selected window, £{row['price']:.1f}m, "
            f"{row['ownership']:.1f}% owned. "
            f"Fixture run: {self.fixtures_string(row)}."
        )
