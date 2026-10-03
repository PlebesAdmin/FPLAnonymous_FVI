# FPL Fixture Value Index

A transparent Python / Streamlit tool that ranks Fantasy Premier League players for any gameweek range.

## What it does

You choose:

- Start & End gameweek
- Positions
- Maximum price
- Maximum ownership
- Minimum minutes played

The app pulls the latest public FPL data and ranks players using a clear **Fixture Value Index (FVI)**.

### Default FVI weights

- **35%** Fixture Ease  
- **30%** Points Potential  
- **20%** Value (points per £m)  
- **15%** Differential Potential  

You can change the weights in the sidebar. The score is always 0–100.

### How points are projected

A baseline is built from FPL’s own `ep_next`, recent form and season points-per-game.  
The blend changes automatically with the length of the window you select (short horizon trusts `ep_next` more; longer horizons lean on form + PPG).

Then each fixture applies:

- Fixture difficulty adjustment
- Mild home / away factor
- Availability (injury / suspension chance)

Double gameweeks are counted automatically. Blank gameweeks contribute zero points.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud (free)

1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → select the repo → set Main file path to `app.py`.
4. Click **Deploy**.

Your public link will look like:  
`https://yourname-fpl-fixture-value-index.streamlit.app`

## Project structure

```text
fpl_fixture_value_index/
├── app.py                 # Streamlit UI
├── fvi_engine.py          # Scoring engine
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml        # Dark theme
```

A `cache/` folder is created automatically on first run.

## Design principle

This is deliberately **not** an opaque AI model.  
Every recommendation can be explained by the four component scores and the fixture list.

Future versions can add extra signals (xG, set-piece takers, rotation risk, etc.) as new transparent components of the same index.

## Data source

Public Fantasy Premier League API:

- https://fantasy.premierleague.com/api/bootstrap-static/
- https://fantasy.premierleague.com/api/fixtures/

The API is unofficial, so caching and defensive parsing are intentional.
