
import os
import re
import math
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Tennis Betting Lab V6", page_icon="🎾", layout="wide")

DATA_DIR = os.path.dirname(__file__)

def load_local_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing local database file: {filename}")
    return pd.read_csv(path, low_memory=False)

def build_match_table(raw):
    """Convert the local TennisData-style home/away schema into winner/loser form."""
    d = raw.copy()

    # Keep completed matches for historical analysis.
    if "status" in d.columns:
        finished = d["status"].astype(str).str.upper().eq("FINISHED")
        if finished.any():
            d = d.loc[finished].copy()

    d["date"] = pd.to_datetime(d.get("date_human"), errors="coerce")

    # winner_code: 1 = home, 2 = away
    code = pd.to_numeric(d.get("winner_code"), errors="coerce")
    d["winner_name"] = np.where(code.eq(1), d["home_name"], np.where(code.eq(2), d["away_name"], np.nan))
    d["loser_name"] = np.where(code.eq(1), d["away_name"], np.where(code.eq(2), d["home_name"], np.nan))

    # Winner/loser-oriented stat columns.
    stat_map = {
        "ace": ("home_aces", "away_aces"),
        "df": ("home_double_faults", "away_double_faults"),
        "serve_pct": ("home_service_points_won_perc", "away_service_points_won_perc"),
        "return_pct": ("home_return_points_won_perc", "away_return_points_won_perc"),
        "bp_won_pct": ("home_break_points_won_perc", "away_break_points_won_perc"),
        "bp_saved_pct": ("home_break_points_saved_perc", "away_break_points_saved_perc"),
        "rank": ("home_rank", "away_rank"),
        "points": ("home_points", "away_points"),
    }
    for out_name, (home_col, away_col) in stat_map.items():
        hv = pd.to_numeric(d.get(home_col), errors="coerce")
        av = pd.to_numeric(d.get(away_col), errors="coerce")
        d["winner_" + out_name] = np.where(code.eq(1), hv, np.where(code.eq(2), av, np.nan))
        d["loser_" + out_name] = np.where(code.eq(1), av, np.where(code.eq(2), hv, np.nan))

    for c in ["surface", "tournament", "round"]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str)

    return d.dropna(subset=["winner_name", "loser_name"]).sort_values("date", ascending=False)

@st.cache_data(ttl=86400, show_spinner=False)
def load_matches(tour):
    filename = "2026-atp-season.csv" if tour == "ATP" else "2026-wta-season.csv"
    return build_match_table(load_local_csv(filename))

def player_directory(d):
    names = pd.concat([d["winner_name"], d["loser_name"]], ignore_index=True)
    return pd.DataFrame({"display": names.dropna().astype(str).str.strip().drop_duplicates().sort_values().tolist()})

def normalize_query(q):
    return re.sub(r"[^a-z0-9 ]+", "", str(q).lower()).strip()

def find_player_options(players, query):
    names = players["display"].tolist()
    if not query:
        return names
    q = normalize_query(query)
    exact = [n for n in names if normalize_query(n) == q]
    if exact:
        return exact
    return [n for n in names if q in normalize_query(n)]

def player_matches(d, player, surface="All"):
    x = d[(d["winner_name"].eq(player)) | (d["loser_name"].eq(player))].copy()
    if surface != "All":
        x = x[x["surface"].eq(surface)]
    return x.sort_values("date", ascending=False)

def summarize(d, player, surface="All", n=20):
    x = player_matches(d, player, surface).head(n)
    if x.empty:
        return {"n": 0, "wins": 0, "losses": 0, "win_pct": np.nan,
                "aces": np.nan, "df": np.nan, "serve_pct": np.nan,
                "return_pct": np.nan, "bp_won_pct": np.nan, "bp_saved_pct": np.nan}

    is_winner = x["winner_name"].eq(player)
    wins = int(is_winner.sum())

    vals = {}
    for metric in ["ace", "df", "serve_pct", "return_pct", "bp_won_pct", "bp_saved_pct"]:
        vals[metric] = []
        for _, r in x.iterrows():
            if r["winner_name"] == player:
                v = r.get("winner_" + metric, np.nan)
            else:
                v = r.get("loser_" + metric, np.nan)
            if pd.notna(v):
                vals[metric].append(float(v))

    return {
        "n": len(x),
        "wins": wins,
        "losses": len(x) - wins,
        "win_pct": wins / len(x),
        "aces": np.nanmean(vals["ace"]) if vals["ace"] else np.nan,
        "df": np.nanmean(vals["df"]) if vals["df"] else np.nan,
        "serve_pct": np.nanmean(vals["serve_pct"]) / 100 if vals["serve_pct"] else np.nan,
        "return_pct": np.nanmean(vals["return_pct"]) / 100 if vals["return_pct"] else np.nan,
        "bp_won_pct": np.nanmean(vals["bp_won_pct"]) / 100 if vals["bp_won_pct"] else np.nan,
        "bp_saved_pct": np.nanmean(vals["bp_saved_pct"]) / 100 if vals["bp_saved_pct"] else np.nan,
    }

def h2h(d, p1, p2, surface="All"):
    x = d[((d["winner_name"].eq(p1)) & (d["loser_name"].eq(p2))) |
          ((d["winner_name"].eq(p2)) & (d["loser_name"].eq(p1)))].copy()
    if surface != "All":
        x = x[x["surface"].eq(surface)]
    return x.sort_values("date", ascending=False)

def elo_prob(d, p1, p2, surface="All"):
    x = d.dropna(subset=["winner_name", "loser_name", "date"]).sort_values("date")
    ratings = {}
    surface_ratings = {}

    for _, r in x.iterrows():
        a, b = r["winner_name"], r["loser_name"]
        surf = r["surface"]
        ratings.setdefault(a, 1500.0)
        ratings.setdefault(b, 1500.0)
        surface_ratings.setdefault((a, surf), 1500.0)
        surface_ratings.setdefault((b, surf), 1500.0)

        if surface == "All" or surf == surface:
            ra, rb = ratings[a], ratings[b]
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            k = 24
            ratings[a] += k * (1 - ea)
            ratings[b] += k * (0 - (1 - ea))

            rsa, rsb = surface_ratings[(a, surf)], surface_ratings[(b, surf)]
            es = 1 / (1 + 10 ** ((rsb - rsa) / 400))
            ks = 28
            surface_ratings[(a, surf)] += ks * (1 - es)
            surface_ratings[(b, surf)] += ks * (0 - (1 - es))

    base = 1 / (1 + 10 ** ((ratings.get(p2, 1500) - ratings.get(p1, 1500)) / 400))
    if surface != "All":
        sp = 1 / (1 + 10 ** ((surface_ratings.get((p2, surface), 1500) -
                               surface_ratings.get((p1, surface), 1500)) / 400))
        return 0.35 * base + 0.65 * sp
    return base

def model(d, p1, p2, surface):
    s1 = summarize(d, p1, surface, 20)
    s2 = summarize(d, p2, surface, 20)
    r1 = summarize(d, p1, "All", 30)
    r2 = summarize(d, p2, "All", 30)

    elo = elo_prob(d, p1, p2, surface)
    f1 = s1["win_pct"] if pd.notna(s1["win_pct"]) else r1["win_pct"]
    f2 = s2["win_pct"] if pd.notna(s2["win_pct"]) else r2["win_pct"]
    wr = 0.5 if pd.isna(f1) or pd.isna(f2) else 0.5 + 0.5 * (f1 - f2)

    serve = 0.5 if pd.isna(s1["serve_pct"]) or pd.isna(s2["serve_pct"]) else 0.5 + 2.2 * (s1["serve_pct"] - s2["serve_pct"])
    ret = 0.5 if pd.isna(s1["return_pct"]) or pd.isna(s2["return_pct"]) else 0.5 + 2.0 * (s1["return_pct"] - s2["return_pct"])
    ace = 0.5 if pd.isna(s1["aces"]) or pd.isna(s2["aces"]) else 0.5 + 0.035 * (s1["aces"] - s2["aces"])
    bp = 0.5 if pd.isna(s1["bp_won_pct"]) or pd.isna(s2["bp_won_pct"]) else 0.5 + 0.8 * (s1["bp_won_pct"] - s2["bp_won_pct"])

    h = h2h(d, p1, p2, surface)
    hp = 0.5 if h.empty else float((h["winner_name"] == p1).mean())

    n = min(s1["n"], s2["n"])
    confidence = min(1.0, n / 12.0)

    raw = 0.40 * elo + 0.25 * wr + 0.12 * serve + 0.08 * ret + 0.05 * ace + 0.05 * bp + 0.05 * hp
    p1prob = 0.5 + (raw - 0.5) * (0.45 + 0.55 * confidence)

    return {
        "p1": float(np.clip(p1prob, 0.01, 0.99)),
        "p2": float(np.clip(1 - p1prob, 0.01, 0.99)),
        "s1": s1, "s2": s2, "elo": elo, "h2h": h, "confidence": confidence
    }

def pct(x):
    return "—" if pd.isna(x) else f"{x * 100:.1f}%"

def num(x):
    return "—" if pd.isna(x) else f"{x:.2f}"

st.title("🎾 Tennis Betting Lab V6")
st.caption("Local 2026 ATP/WTA match database • player search • matchup intelligence • Kalshi edge calculator")

with st.sidebar:
    st.header("Data")
    if st.button("Refresh local database"):
        st.cache_data.clear()
        st.rerun()
    st.success("No paid tennis API required.")
    st.write("Source: CSV files stored in this repository.")
    st.caption("This version reads the repository-root CSV files directly.")

tour = st.selectbox("Tour", ["ATP", "WTA"])

try:
    with st.spinner(f"Loading {tour} database…"):
        matches = load_matches(tour)
        players_df = player_directory(matches)
except Exception as e:
    st.error(f"Database load failed: {e}")
    st.info("Make sure the matching 2026 CSV is in the repository root.")
    st.stop()

surface_opts = ["All"] + sorted([x for x in matches["surface"].dropna().unique().tolist() if str(x).strip()])
surface = st.selectbox("Surface", surface_opts)
st.write(f"**Player directory:** {len(players_df):,} names • **finished matches:** {len(matches):,}")

q = st.text_input("Search any player", placeholder="Alcaraz, Sinner, Sabalenka…")
filtered = find_player_options(players_df, q)

if q and not filtered:
    st.warning("No player matched that search.")
    st.stop()

filtered = filtered[:500]
c1, c2 = st.columns(2)
with c1:
    p1 = st.selectbox("Player 1", filtered, key="p1")
with c2:
    default_idx = 1 if len(filtered) > 1 else 0
    p2 = st.selectbox("Player 2", filtered, key="p2", index=default_idx)

if st.button("ANALYZE MATCHUP", type="primary", use_container_width=True):
    if p1 == p2:
        st.error("Choose two different players.")
    else:
        st.session_state["analysis"] = (p1, p2, surface, model(matches, p1, p2, surface))

if "analysis" in st.session_state:
    p1, p2, surface, m = st.session_state["analysis"]

    st.divider()
    st.subheader(f"{p1} vs {p2}")

    a, b = st.columns(2)
    with a:
        st.metric(p1, f"{m['p1'] * 100:.1f}%", f"Fair price: {m['p1'] * 100:.0f}¢")
    with b:
        st.metric(p2, f"{m['p2'] * 100:.1f}%", f"Fair price: {m['p2'] * 100:.0f}¢")

    st.info("Model output is a research estimate, not a guarantee.")

    st.subheader("Recent form")
    form = pd.DataFrame({
        "Metric": ["Sample", "Win %", "Aces / match", "Double faults / match",
                   "Serve points won", "Return points won", "Break points won", "Break points saved"],
        p1: [m["s1"]["n"], pct(m["s1"]["win_pct"]), num(m["s1"]["aces"]), num(m["s1"]["df"]),
             pct(m["s1"]["serve_pct"]), pct(m["s1"]["return_pct"]), pct(m["s1"]["bp_won_pct"]), pct(m["s1"]["bp_saved_pct"])],
        p2: [m["s2"]["n"], pct(m["s2"]["win_pct"]), num(m["s2"]["aces"]), num(m["s2"]["df"]),
             pct(m["s2"]["serve_pct"]), pct(m["s2"]["return_pct"]), pct(m["s2"]["bp_won_pct"]), pct(m["s2"]["bp_saved_pct"])],
    })
    st.dataframe(form, use_container_width=True, hide_index=True)

    st.subheader("Head-to-head")
    h = m["h2h"]
    if h.empty:
        st.write("No head-to-head matches found in this local database.")
    else:
        h_view = h[["date", "tournament", "round", "surface", "winner_name", "loser_name"]].copy()
        h_view["date"] = h_view["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(h_view, use_container_width=True, hide_index=True)

    st.subheader("Kalshi edge calculator")
    k1, k2 = st.columns(2)
    with k1:
        price1 = st.number_input(f"{p1} YES price (¢)", min_value=1, max_value=99, value=int(round(m["p1"] * 100)), key="price1")
    with k2:
        price2 = st.number_input(f"{p2} YES price (¢)", min_value=1, max_value=99, value=int(round(m["p2"] * 100)), key="price2")

    edge1 = m["p1"] * 100 - price1
    edge2 = m["p2"] * 100 - price2
    e1, e2 = st.columns(2)
    with e1:
        st.metric(f"{p1} edge", f"{edge1:+.1f}¢")
    with e2:
        st.metric(f"{p2} edge", f"{edge2:+.1f}¢")

    st.caption(f"Model confidence: {m['confidence'] * 100:.0f}% based on the smaller recent sample.")
