import streamlit as st
import pandas as pd

from fvi_engine import FPLFixtureValueAgent, FVIConfig


st.set_page_config(
    page_title="FPL Fixture Value Index",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Header + how-to
# ---------------------------------------------------------------------------
st.title("⚽ FPL Fixture Value Index")
st.caption(
    "A transparent player-selection model combining fixture difficulty, "
    "points potential (current + historical), price efficiency and differential upside."
)

with st.expander("📖 How to use this tool (click to open)", expanded=False):
    st.markdown(
        """
**Quick start**
1. In the left sidebar choose a **Start** and **End** gameweek (e.g. next 5 weeks).
2. Optionally filter by **position**, **max price**, **ownership** and **minutes**.
3. The table updates automatically and shows the highest-scoring players.
4. Click any player expander below the table to see why the model likes them.
5. Use the **Download CSV** button at the bottom to save the results.

**What powers the score**
- Live FPL data (price, ownership, form, fixtures, availability)
- Historical baselines from previous seasons (PPG + minutes reliability)
- Transparent weights you can adjust in the sidebar

**Tips**
- Lower the **Maximum ownership** slider to find differentials.
- Longer gameweek ranges give more weight to historical performance.
- A blank gameweek contributes zero projected points.
        """
    )

# ---------------------------------------------------------------------------
# Load agent (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_agent():
    try:
        return FPLFixtureValueAgent().load()
    except Exception as e:
        st.error(f"Could not load FPL data: {e}")
        st.stop()


agent = get_agent()

events = agent.bootstrap_data["events"]
available_gws = [int(e["id"]) for e in events]

next_gw = next(
    (int(e["id"]) for e in events if e.get("is_next")),
    min(available_gws) if available_gws else 1,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Filters")

    start_gw = st.selectbox(
        "Start gameweek",
        available_gws,
        index=available_gws.index(next_gw) if next_gw in available_gws else 0,
        help="First gameweek in the window you care about.",
    )

    end_options = [gw for gw in available_gws if gw >= start_gw]
    default_end_idx = min(4, len(end_options) - 1)
    end_gw = st.selectbox(
        "End gameweek",
        end_options,
        index=default_end_idx,
        help="Last gameweek in the window. Double gameweeks are counted automatically.",
    )

    st.divider()

    positions = st.multiselect(
        "Positions",
        ["GKP", "DEF", "MID", "FWD"],
        default=["GKP", "DEF", "MID", "FWD"],
        help="Leave all selected to see every position.",
    )

    max_price = st.slider(
        "Maximum price (£m)",
        min_value=4.0,
        max_value=15.0,
        value=15.0,
        step=0.1,
        help="Only show players at or below this price.",
    )

    min_minutes = st.slider(
        "Minimum minutes played this season",
        min_value=0,
        max_value=900,
        value=90,
        step=30,
        help="Filters out players who have barely played this season.",
    )

    max_ownership = st.slider(
        "Maximum ownership (%)",
        min_value=1.0,
        max_value=100.0,
        value=100.0,
        step=1.0,
        help="Lower this to deliberately search for differentials.",
    )

    top_n = st.slider(
        "Players to display",
        5,
        40,
        15,
        help="How many rows appear in the main table.",
    )

    st.divider()
    st.markdown("### FVI weights")
    st.caption("These must add up to roughly 100%. Adjust to suit your style.")

    w_fixture = st.slider("Fixture ease %", 0, 60, 35, 5)
    w_points = st.slider("Points potential %", 0, 60, 30, 5)
    w_value = st.slider("Value (pts/£m) %", 0, 50, 20, 5)
    w_diff = st.slider("Differential %", 0, 40, 15, 5)

    total_w = w_fixture + w_points + w_value + w_diff
    if total_w == 0:
        st.warning("Weights cannot all be zero.")
        st.stop()

    cfg = FVIConfig(
        fixture_weight=w_fixture / total_w,
        points_weight=w_points / total_w,
        value_weight=w_value / total_w,
        differential_weight=w_diff / total_w,
    )

    if st.button("↺ Reset weights to default"):
        st.rerun()

    st.caption(
        f"Current mix → Fixture {cfg.fixture_weight*100:.0f}% · "
        f"Points {cfg.points_weight*100:.0f}% · "
        f"Value {cfg.value_weight*100:.0f}% · "
        f"Diff {cfg.differential_weight*100:.0f}%"
    )

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if end_gw < start_gw:
    st.error("End gameweek must be greater than or equal to Start gameweek.")
    st.stop()

if not positions:
    st.warning("Please select at least one position.")
    st.stop()

# ---------------------------------------------------------------------------
# Calculate
# ---------------------------------------------------------------------------
with st.spinner("Calculating Fixture Value Index (live + historical)…"):
    try:
        df = agent.calculate(
            start_gw,
            end_gw,
            min_minutes=min_minutes,
            config=cfg,
        )
    except Exception as e:
        st.error(f"Something went wrong while calculating: {e}")
        st.stop()

if df.empty:
    st.warning("No players matched the selected filters. Try lowering the minimum minutes.")
    st.stop()

filtered = df[
    df["position"].isin(positions)
    & (df["price"] <= max_price)
    & (df["ownership"] <= max_ownership)
].copy()

if filtered.empty:
    st.warning(
        "No players match these filters. "
        "Try increasing the price or ownership limit, or reducing minimum minutes."
    )
    st.stop()

display = filtered.head(top_n).copy()
display["fixtures"] = display.apply(agent.fixtures_string, axis=1)

# ---------------------------------------------------------------------------
# Summary metrics + headline
# ---------------------------------------------------------------------------
st.subheader(f"Top {len(display)} players · GW{start_gw} → GW{end_gw}")

best = display.iloc[0]
st.info(
    f"**Best pick right now:** {best['player']} ({best['team_short']}) — "
    f"FVI **{best['fvi']:.1f}** · projected **{best['projected_points']:.1f} pts** · "
    f"£{best['price']:.1f}m · {best['ownership']:.1f}% owned"
)

cols = st.columns(4)
cols[0].metric("Players analysed", f"{len(filtered):,}")
cols[1].metric("Best FVI", f"{best['fvi']:.1f}")
cols[2].metric("Best projected pts", f"{display['projected_points'].max():.1f}")
cols[3].metric("Avg FDR of #1", f"{best['avg_fdr']:.2f}")

chart_data = display.set_index("player")[["fvi"]].head(12)
st.bar_chart(chart_data, height=220)

# ---------------------------------------------------------------------------
# Main table
# ---------------------------------------------------------------------------
table = display[
    [
        "rank",
        "player",
        "team_short",
        "position",
        "price",
        "ownership",
        "avg_fdr",
        "fixture_count",
        "projected_points",
        "points_per_million",
        "fvi",
        "recommendation",
    ]
].copy()

table.columns = [
    "Rank",
    "Player",
    "Team",
    "Pos",
    "Price £m",
    "Own %",
    "Avg FDR",
    "Fixtures",
    "Projected Pts",
    "Pts / £m",
    "FVI",
    "Recommendation",
]

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Price £m": st.column_config.NumberColumn(format="£%.1f"),
        "Own %": st.column_config.NumberColumn(format="%.1f%%"),
        "Avg FDR": st.column_config.NumberColumn(format="%.2f"),
        "Projected Pts": st.column_config.NumberColumn(format="%.1f"),
        "Pts / £m": st.column_config.NumberColumn(format="%.2f"),
        "FVI": st.column_config.NumberColumn(format="%.1f"),
    },
)

# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Why the model likes them")

for _, row in display.iterrows():
    with st.expander(
        f"#{int(row['rank'])} {row['player']} — FVI {row['fvi']:.1f} — {row['recommendation']}"
    ):
        st.write(agent.explain_player(row))
        st.write(
            f"**Component scores** → "
            f"Fixture {row['fixture_score']:.1f} · "
            f"Points {row['points_score']:.1f} · "
            f"Value {row['value_score']:.1f} · "
            f"Differential {row['differential_score']:.1f}"
        )
        if row.get("seasons_played", 0) > 0 and row.get("hist_ppg", 0) > 0.1:
            st.caption(
                f"Historical → {row['hist_ppg']:.1f} PPG · "
                f"Last season {row['last_season_ppg']:.1f} PPG · "
                f"{row['hist_minutes_pct']*100:.0f}% minutes · "
                f"{int(row['seasons_played'])} seasons"
            )
        if row["news"]:
            st.warning(f"FPL news: {row['news']}")

# ---------------------------------------------------------------------------
# Differentials section
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🔍 Find differentials (≤ 10% ownership)")

diffs = filtered[filtered["ownership"] <= 10].head(10)

if not diffs.empty:
    st.dataframe(
        diffs[
            [
                "player",
                "team_short",
                "position",
                "price",
                "ownership",
                "avg_fdr",
                "projected_points",
                "fvi",
                "recommendation",
            ]
        ].rename(
            columns={
                "player": "Player",
                "team_short": "Team",
                "position": "Pos",
                "price": "Price £m",
                "ownership": "Own %",
                "avg_fdr": "Avg FDR",
                "projected_points": "Projected Pts",
                "fvi": "FVI",
                "recommendation": "Recommendation",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No sub-10% ownership players matched the current filters.")

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
csv = display.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download results as CSV",
    data=csv,
    file_name=f"fvi_gw{start_gw}_gw{end_gw}.csv",
    mime="text/csv",
)

st.caption(
    "FVI is a decision-support model, not a guarantee of future FPL points. "
    "Live data from the public Fantasy Premier League API. "
    "Historical baselines from the open vaastav/Fantasy-Premier-League dataset."
)
