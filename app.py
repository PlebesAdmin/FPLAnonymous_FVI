import streamlit as st
import pandas as pd

from fvi_engine import FPLFixtureValueAgent, FVIConfig


st.set_page_config(
    page_title="FPL Fixture Value Index",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚽ FPL Fixture Value Index")
st.caption(
    "Transparent player rankings using fixtures, current form, historical performance "
    "(home/away + consistency), price and ownership."
)

with st.expander("📖 How to use this tool (click to open)", expanded=False):
    st.markdown(
        """
**Quick start**
1. Choose a **Start** and **End** gameweek in the sidebar.
2. Filter by position, price, ownership or minutes if you want.
3. Rankings update automatically.
4. Expand any player to see the full reasoning (including historical home/away and consistency).

**What the model uses**
- Live FPL data (price, ownership, form, fixtures, availability)
- Historical PPG from previous seasons
- Home vs Away historical rates applied to each fixture
- Minutes reliability and consistency score

**Tips**
- Longer windows put more weight on historical performance.
- Lower maximum ownership to surface differentials.
- Blank gameweeks contribute zero projected points.
        """
    )

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

with st.sidebar:
    st.header("⚙️ Filters")

    start_gw = st.selectbox(
        "Start gameweek",
        available_gws,
        index=available_gws.index(next_gw) if next_gw in available_gws else 0,
        help="First gameweek in the window.",
    )

    end_options = [gw for gw in available_gws if gw >= start_gw]
    default_end_idx = min(4, len(end_options) - 1)
    end_gw = st.selectbox(
        "End gameweek",
        end_options,
        index=default_end_idx,
        help="Last gameweek. Double gameweeks are counted automatically.",
    )

    st.divider()

    positions = st.multiselect(
        "Positions",
        ["GKP", "DEF", "MID", "FWD"],
        default=["GKP", "DEF", "MID", "FWD"],
    )

    max_price = st.slider("Maximum price (£m)", 4.0, 15.0, 15.0, 0.1)
    min_minutes = st.slider("Minimum minutes this season", 0, 900, 90, 30)
    max_ownership = st.slider("Maximum ownership (%)", 1.0, 100.0, 100.0, 1.0)
    top_n = st.slider("Players to display", 5, 40, 15)

    st.divider()
    st.markdown("### FVI weights")
    st.caption("Adjust to suit your style. Values are normalised automatically.")

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

if end_gw < start_gw:
    st.error("End gameweek must be ≥ Start gameweek.")
    st.stop()

if not positions:
    st.warning("Please select at least one position.")
    st.stop()

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
    st.warning("No players matched the selected filters.")
    st.stop()

filtered = df[
    df["position"].isin(positions)
    & (df["price"] <= max_price)
    & (df["ownership"] <= max_ownership)
].copy()

if filtered.empty:
    st.warning("No players match these filters. Try relaxing price, ownership or minutes.")
    st.stop()

display = filtered.head(top_n).copy()
display["fixtures"] = display.apply(agent.fixtures_string, axis=1)

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
            hist_ppg = row.get("hist_ppg", 0)
            hist_home = row.get("hist_ppg_home", hist_ppg)
            hist_away = row.get("hist_ppg_away", hist_ppg)
            consistency = row.get("consistency_score", 50)
            minutes_pct = row.get("hist_minutes_pct", 0)
            seasons = int(row.get("seasons_played", 0))
            n_starts = int(row.get("n_starts", 0))

            st.caption(
                f"Historical → {hist_ppg:.1f} PPG "
                f"(Home {hist_home:.1f} / Away {hist_away:.1f}) · "
                f"Consistency {consistency:.0f}/100 · "
                f"{minutes_pct*100:.0f}% minutes · "
                f"{seasons} seasons · "
                f"{n_starts} starts sampled"
            )
        if row.get("news"):
            st.warning(f"FPL news: {row['news']}")

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
    "Historical baselines from vaastav/Fantasy-Premier-League."
)
