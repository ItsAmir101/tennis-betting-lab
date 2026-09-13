
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Tennis Betting Lab",
    page_icon="🎾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width: 1000px; padding: 1rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.stButton button {width: 100%; min-height: 48px;}
div[data-baseweb="select"] {font-size: 1rem;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
DATA_FILES = {
    "ATP": BASE / "2026-atp-season.csv",
    "WTA": BASE / "2026-wta-season.csv",
}

@st.cache_data(ttl=21600, show_spinner="Loading 2026 tennis database…")
def load_matches(tour):
    path = DATA_FILES[tour]
    if not path.exists():
        st.error(f"Missing database file: {path.name}. Upload it to the same GitHub repository as app.py.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date_human"], errors="coerce")
        df["surface"] = df["surface"].astype(str).str.title()
        # Only completed matches are used for historical statistics.
        # Keep other rows available for transparency, but filter here.
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.upper().eq("FINISHED")].copy()
        return df.sort_values("date", ascending=False)
    except Exception as e:
        st.error(f"Could not load {tour} data: {e}")
        return pd.DataFrame()

def player_matches(df, player, surface="All"):
    if df.empty:
        return pd.DataFrame()

    w = df[df["home_name"].eq(player)].copy()
    w["result"] = np.where(w["winner_code"].eq(1), "W",
                           np.where(w["winner_code"].eq(2), "L", np.nan))
    w["opponent"] = w["away_name"]
    w["player_rank"] = w["home_rank"]
    w["opp_rank"] = w["away_rank"]
    w["aces_player"] = w["home_aces"]
    w["df_player"] = w["home_double_faults"]
    w["serve_pts_won_pct"] = w["home_service_points_won_perc"]
    w["return_pts_won_pct"] = w["home_return_points_won_perc"]
    w["bp_conversion_pct"] = w["home_break_points_won_perc"]
    w["bp_save_pct"] = w["home_break_points_saved_perc"]
    w["sets_won"] = w["home_set_score"]
    w["sets_lost"] = w["away_set_score"]

    l = df[df["away_name"].eq(player)].copy()
    l["result"] = np.where(l["winner_code"].eq(2), "W",
                           np.where(l["winner_code"].eq(1), "L", np.nan))
    l["opponent"] = l["home_name"]
    l["player_rank"] = l["away_rank"]
    l["opp_rank"] = l["home_rank"]
    l["aces_player"] = l["away_aces"]
    l["df_player"] = l["away_double_faults"]
    l["serve_pts_won_pct"] = l["away_service_points_won_perc"]
    l["return_pts_won_pct"] = l["away_return_points_won_perc"]
    l["bp_conversion_pct"] = l["away_break_points_won_perc"]
    l["bp_save_pct"] = l["away_break_points_saved_perc"]
    l["sets_won"] = l["away_set_score"]
    l["sets_lost"] = l["home_set_score"]

    d = pd.concat([w, l], ignore_index=True)
    d = d[d["result"].isin(["W", "L"])]
    if surface != "All":
        d = d[d["surface"].eq(surface)]
    return d.sort_values("date", ascending=False)

def safe_mean(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan

def pct(num, den):
    return float(num / den) if den else np.nan

def summarize(d, n=20):
    d = d.head(n)
    if d.empty:
        return {}
    wins = int((d.result == "W").sum())
    losses = int((d.result == "L").sum())
    return {
        "sample": len(d),
        "wins": wins,
        "losses": losses,
        "win_rate": pct(wins, len(d)),
        "aces": safe_mean(d.aces_player),
        "df": safe_mean(d.df_player),
        "serve_pts_won_pct": safe_mean(d.serve_pts_won_pct),
        "return_pts_won_pct": safe_mean(d.return_pts_won_pct),
        "bp_conversion_pct": safe_mean(d.bp_conversion_pct),
        "bp_save_pct": safe_mean(d.bp_save_pct),
    }

def h2h(df, p1, p2, surface="All"):
    if df.empty:
        return pd.DataFrame()
    d = df[
        ((df.home_name.eq(p1)) & (df.away_name.eq(p2))) |
        ((df.home_name.eq(p2)) & (df.away_name.eq(p1)))
    ].copy()
    if surface != "All":
        d = d[d.surface.eq(surface)]
    return d.sort_values("date", ascending=False)

def model(df, p1, p2, surface):
    a = summarize(player_matches(df, p1, surface), 20)
    b = summarize(player_matches(df, p2, surface), 20)
    if not a or not b:
        return .5, .5, a, b

    wr_a, wr_b = a.get("win_rate", .5), b.get("win_rate", .5)
    serve_a, serve_b = a.get("serve_pts_won_pct", np.nan), b.get("serve_pts_won_pct", np.nan)
    ace_a, ace_b = a.get("aces", np.nan), b.get("aces", np.nan)
    ret_a, ret_b = a.get("return_pts_won_pct", np.nan), b.get("return_pts_won_pct", np.nan)

    serve_signal = 0 if np.isnan(serve_a) or np.isnan(serve_b) else (serve_a - serve_b) / 100
    return_signal = 0 if np.isnan(ret_a) or np.isnan(ret_b) else (ret_a - ret_b) / 100
    ace_signal = 0 if np.isnan(ace_a) or np.isnan(ace_b) else np.tanh((ace_a - ace_b) / 5)
    confidence = min(1, min(a.get("sample", 0), b.get("sample", 0)) / 20)

    # Transparent research model, not a sportsbook-grade pricing model.
    score = (
        0.55 * (wr_a - wr_b)
        + 0.20 * serve_signal
        + 0.15 * return_signal
        + 0.10 * ace_signal
    )
    score *= (0.65 + 0.35 * confidence)
    p = 1 / (1 + np.exp(-5.0 * score))
    return float(p), float(1 - p), a, b

def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.1%}"

def fmt_num(x):
    return "N/A" if pd.isna(x) else f"{x:.2f}"

st.title("🎾 Tennis Betting Lab")
st.caption("2026 ATP/WTA Tour + Challenger database • matchup intelligence • Kalshi edge scanner")

tour_filter = st.selectbox("Tour", ["All", "ATP", "WTA"])
surface = st.selectbox("Surface", ["All", "Hard", "Clay", "Grass", "Carpet"])

atp_df = load_matches("ATP")
wta_df = load_matches("WTA")

if tour_filter == "ATP":
    df = atp_df
elif tour_filter == "WTA":
    df = wta_df
else:
    df = pd.concat([atp_df, wta_df], ignore_index=True)

if df.empty:
    st.stop()

# Universal directory from every player appearing in the supplied 2026 files.
def build_player_directory(atp, wta):
    frames = []
    for tour_name, d in [("ATP", atp), ("WTA", wta)]:
        if d.empty:
            continue
        names = pd.concat(
            [d["home_name"], d["away_name"]], ignore_index=True
        ).dropna().astype(str)
        frames.append(pd.DataFrame({"player": names, "tour": tour_name}))
    if not frames:
        return pd.DataFrame(columns=["player", "tour"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(["tour", "player"])
        .reset_index(drop=True)
    )

directory = build_player_directory(atp_df, wta_df)
if tour_filter != "All":
    directory = directory[directory["tour"].eq(tour_filter)].copy()

st.caption(
    f"Player directory: {len(directory):,} players • "
    f"2026 ATP Tour/Challenger + WTA Tour/Challenger data"
)

search = st.text_input(
    "Search any player",
    placeholder="Type a surname: Rybakina, Sabalenka, Alcaraz, Sinner…",
).strip().lower()

if search:
    candidates = directory[
        directory["player"].str.lower().str.contains(search, na=False)
    ].copy()
else:
    candidates = directory.copy()

if candidates.empty:
    st.warning(
        f"No {tour_filter} player found for '{search}'. "
        "Try the surname only."
    )
    st.stop()

candidate_labels = [
    f"{r.player} — {r.tour}" for r in candidates.itertuples(index=False)
]
p1_label = st.selectbox("Player 1", candidate_labels, key="p1_universal")
p1_name, p1_tour = p1_label.rsplit(" — ", 1)

remaining = candidates[
    ~((candidates["player"].eq(p1_name)) & (candidates["tour"].eq(p1_tour)))
].copy()

if remaining.empty:
    st.warning("Search for another player to create a matchup.")
    st.stop()

p2_labels = [
    f"{r.player} — {r.tour}" for r in remaining.itertuples(index=False)
]
p2_label = st.selectbox("Player 2", p2_labels, key="p2_universal")
p2_name, p2_tour = p2_label.rsplit(" — ", 1)

if p1_tour != p2_tour:
    st.error(
        "Player 1 and Player 2 must be from the same tour. "
        "Select two ATP players or two WTA players."
    )

p1, p2 = p1_name, p2_name

if st.button("ANALYZE MATCHUP", type="primary"):
    if not p1 or not p2:
        st.error("Select two players first.")
        st.stop()

    p1p, p2p, s1, s2 = model(df, p1, p2, surface)
    hh = h2h(df, p1, p2, surface)
    r1 = player_matches(df, p1, surface).head(10)
    r2 = player_matches(df, p2, surface).head(10)

    st.subheader("Model Probability")
    c1, c2 = st.columns(2)
    c1.metric(p1, fmt_pct(p1p))
    c2.metric(p2, fmt_pct(p2p))

    st.subheader("Kalshi Edge")
    price = st.number_input(f"Current YES price for {p1} (¢)", min_value=1, max_value=99, value=50)
    market = price / 100
    edge = p1p - market
    c1, c2, c3 = st.columns(3)
    c1.metric("Market implied", fmt_pct(market))
    c2.metric("Model edge", f"{edge:+.1%}")
    signal = "STRONG VALUE" if edge >= .08 else "VALUE" if edge >= .05 else "FAIR" if edge > -0.03 else "EXPENSIVE"
    c3.metric("Signal", signal)

    if edge >= .08:
        st.success("Meaningful model-vs-market gap. Verify current match conditions, liquidity, and the market rules before trading.")
    elif edge >= .05:
        st.info("Potential edge. Check price movement, matchup context, and market depth.")
    elif edge > -0.03:
        st.warning("Price is close to model fair value.")
    else:
        st.error("Player 1 is priced above the model estimate.")

    st.subheader("Head-to-Head")
    if hh.empty:
        st.write("No H2H matches found in the loaded 2026 database for this surface.")
    else:
        p1wins = int(((hh.home_name.eq(p1)) & (hh.winner_code.eq(1))).sum() +
                     ((hh.away_name.eq(p1)) & (hh.winner_code.eq(2))).sum())
        p2wins = len(hh) - p1wins
        st.write(f"**{p1}: {p1wins}** wins  |  **{p2}: {p2wins}** wins")
        hview = hh[["date", "tournament", "surface", "round", "home_name", "away_name",
                    "home_set_score", "away_set_score"]].copy()
        st.dataframe(hview, hide_index=True, use_container_width=True)

    st.subheader("Recent Form")
    a, b = st.columns(2)
    a.write(f"**{p1} — last 10 {surface.lower()} matches**")
    a.dataframe(
        r1[["date", "tournament", "surface", "round", "result", "opponent",
            "player_rank", "opp_rank", "aces_player", "df_player"]],
        hide_index=True, use_container_width=True
    )
    b.write(f"**{p2} — last 10 {surface.lower()} matches**")
    b.dataframe(
        r2[["date", "tournament", "surface", "round", "result", "opponent",
            "player_rank", "opp_rank", "aces_player", "df_player"]],
        hide_index=True, use_container_width=True
    )

    st.subheader("Stat Comparison — last 20")
    stat_rows = []
    for label, key, kind in [
        ("Win rate", "win_rate", "pct"),
        ("Aces / match", "aces", "num"),
        ("Double faults / match", "df", "num"),
        ("Service points won", "serve_pts_won_pct", "pct"),
        ("Return points won", "return_pts_won_pct", "pct"),
        ("Break-point conversion", "bp_conversion_pct", "pct"),
        ("Break points saved", "bp_save_pct", "pct"),
    ]:
        stat_rows.append({
            "Metric": label,
            p1: fmt_pct(s1.get(key, np.nan)) if kind == "pct" else fmt_num(s1.get(key, np.nan)),
            p2: fmt_pct(s2.get(key, np.nan)) if kind == "pct" else fmt_num(s2.get(key, np.nan)),
        })
    st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

    st.subheader("Ace / Double-Fault Prop Baseline")
    prop_rows = []
    for label, key in [("Aces / match", "aces"), ("Double faults / match", "df")]:
        prop_rows.append({
            "Prop": label,
            p1: fmt_num(s1.get(key, np.nan)),
            p2: fmt_num(s2.get(key, np.nan)),
        })
    st.dataframe(pd.DataFrame(prop_rows), hide_index=True, use_container_width=True)
    st.caption("The uploaded 2026 files contain aces, double faults, serve/return percentages, and break-point percentages. They do not contain shot-by-shot net touches or unforced-error counts.")

with st.expander("How to use this for Kalshi"):
    st.markdown("""
**DATA → FORM → H2H → MATCHUP → PROP → PROBABILITY → MARKET → EDGE**

1. Pick the tour and surface.
2. Select the two players.
3. Run the matchup.
4. Compare recent surface form, H2H, serve/return numbers, aces, and double faults.
5. Enter the current Kalshi YES price.
6. The app compares that market price with its model probability.
7. Treat the model as a research estimate, not a guarantee.
""")

with st.expander("Database notes"):
    st.write("The app uses the two uploaded 2026 season CSV files stored locally in the same repository as app.py. Historical statistics are calculated from finished matches in those files.")
    st.write("No external tennis database is required. The Kalshi API is not used to place trades.")
