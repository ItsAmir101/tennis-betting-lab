
import io, re, math, time
from datetime import datetime, timezone
from difflib import SequenceMatcher

import pandas as pd
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Tennis Betting Lab", page_icon="🎾", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1.2rem;}
h1 {font-size: 2.2rem;}
.small {color:#8f96a3;font-size:.85rem;}
.card {padding:1rem;border:1px solid #30343b;border-radius:12px;margin:.4rem 0;}
.good {font-weight:700;}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# FREE PUBLIC DATA SOURCES
# -------------------------------------------------------------------
ATP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/"
WTA_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/"
MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/"

# Annual tour-level files. The app also attempts Challenger/qualifying/futures files.
CURRENT_YEAR = datetime.now().year
YEARS = list(range(max(1968, CURRENT_YEAR-10), CURRENT_YEAR+1))

session = requests.Session()
session.headers.update({"User-Agent":"Tennis-Betting-Lab/4.0"})

def get_csv(url, timeout=30):
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), low_memory=False)

@st.cache_data(ttl=86400, show_spinner=False)
def load_database():
    frames = []
    status = []

    # Player master files
    for tour, base, candidates in [
        ("ATP", ATP_BASE, ["atp_players.csv", "atp_players.csv"]),
        ("WTA", WTA_BASE, ["wta_players.csv", "wta_players.csv"]),
    ]:
        loaded = False
        for name in candidates:
            try:
                p = get_csv(base + name)
                p["tour_source"] = tour
                frames.append(("players", p))
                status.append(f"{tour} player file: OK")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            status.append(f"{tour} player file: unavailable; names will be built from match files")

    # Annual match files. Include tour, qualifying/challenger, and futures when present.
    for year in YEARS:
        for tour, base, names in [
            ("ATP", ATP_BASE, [
                f"atp_matches_{year}.csv",
                f"atp_matches_qual_chall_{year}.csv",
                f"atp_matches_futures_{year}.csv",
            ]),
            ("WTA", WTA_BASE, [
                f"wta_matches_{year}.csv",
                f"wta_matches_qual_chall_{year}.csv",
            ]),
        ]:
            for name in names:
                try:
                    df = get_csv(base + name)
                    df["tour_source"] = tour
                    # classify level from filename
                    if "futures" in name:
                        df["level_source"] = "Futures"
                    elif "qual_chall" in name:
                        df["level_source"] = "Challenger/Qualifying"
                    else:
                        df["level_source"] = "Tour"
                    frames.append(("matches", df))
                    status.append(f"{name}: OK")
                except Exception:
                    status.append(f"{name}: not available")

    # Build combined match table
    match_frames = [x[1] for x in frames if x[0] == "matches"]
    if not match_frames:
        raise RuntimeError("No public match files could be downloaded. Check Streamlit internet access.")

    matches = pd.concat(match_frames, ignore_index=True, sort=False)

    # Normalize dates
    if "tourney_date" in matches:
        matches["tourney_date"] = pd.to_datetime(matches["tourney_date"], errors="coerce")

    # Normalize names and IDs
    for c in ["winner_name","loser_name","winner_id","loser_id","surface","tourney_name","round","score"]:
        if c not in matches:
            matches[c] = np.nan

    # Player directory from master files + every name found in matches
    player_rows = []
    for kind, df in frames:
        if kind != "players":
            continue
        cols = {c.lower():c for c in df.columns}
        if "player_id" in cols and "first_name" in cols and "last_name" in cols:
            x = df[[cols["player_id"],cols["first_name"],cols["last_name"]]].copy()
            x.columns = ["player_id","first_name","last_name"]
            x["name"] = x["first_name"].fillna("").astype(str).str.strip()+" "+x["last_name"].fillna("").astype(str).str.strip()
            x["tour"] = df.get("tour_source","").values
            player_rows.append(x[["player_id","name","tour"]])

    # Also construct names from matches, ensuring current players appear even if master file is absent.
    for side in ["winner","loser"]:
        x = matches[[f"{side}_name", f"{side}_id", "tour_source"]].copy()
        x.columns = ["name","player_id","tour"]
        x = x[x["name"].notna()]
        x["name"] = x["name"].astype(str).str.strip()
        player_rows.append(x)

    players = pd.concat(player_rows, ignore_index=True) if player_rows else pd.DataFrame(columns=["player_id","name","tour"])
    players["name"] = players["name"].fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
    players = players[players["name"].str.len() > 2].drop_duplicates(["name","tour"]).reset_index(drop=True)

    return matches, players, status

@st.cache_data(ttl=86400, show_spinner=False)
def load_mcp():
    # MCP is supplemental. We try both men's and women's aggregate match files.
    out = []
    status = []
    for fname, tour in [("charting-m-matches.csv","ATP"),("charting-w-matches.csv","WTA")]:
        try:
            x = get_csv(MCP_BASE + fname)
            x["tour_source"] = tour
            out.append(x)
            status.append(f"{fname}: OK")
        except Exception as e:
            status.append(f"{fname}: unavailable")
    if out:
        return pd.concat(out, ignore_index=True, sort=False), status
    return pd.DataFrame(), status

def clean_name(s):
    return re.sub(r"[^a-z0-9 ]","",str(s).lower()).strip()

def player_search(players, query, tour):
    q = clean_name(query)
    p = players.copy()
    if tour != "All":
        p = p[p["tour"] == tour]
    if not q:
        return p.sort_values("name").head(100)
    p["_n"] = p["name"].map(clean_name)
    p["_score"] = p["_n"].map(lambda x: 1.0 if q in x else SequenceMatcher(None,q,x).ratio())
    return p[p["_score"] >= 0.35].sort_values(["_score","name"], ascending=[False,True]).head(80).drop(columns=["_n","_score"])

def safe_num(df, col):
    if col not in df:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

def add_player_perspective(matches, name):
    m = matches[(matches["winner_name"] == name) | (matches["loser_name"] == name)].copy()
    if m.empty:
        return m
    m["win"] = np.where(m["winner_name"] == name, 1, 0)
    m["opponent"] = np.where(m["winner_name"] == name, m["loser_name"], m["winner_name"])
    m["aces"] = np.where(m["winner_name"] == name, safe_num(m,"w_ace"), safe_num(m,"l_ace"))
    m["df"] = np.where(m["winner_name"] == name, safe_num(m,"w_df"), safe_num(m,"l_df"))
    m["opp_aces"] = np.where(m["winner_name"] == name, safe_num(m,"l_ace"), safe_num(m,"w_ace"))
    m["opp_df"] = np.where(m["winner_name"] == name, safe_num(m,"l_df"), safe_num(m,"w_df"))
    # Serve point win proxies where match stats exist
    won = np.where(m["winner_name"] == name, safe_num(m,"w_1stWon")+safe_num(m,"w_2ndWon"),
                   safe_num(m,"l_1stWon")+safe_num(m,"l_2ndWon"))
    svpt = np.where(m["winner_name"] == name, safe_num(m,"w_svpt"), safe_num(m,"l_svpt"))
    m["serve_points_won_pct"] = np.where(svpt > 0, won/svpt*100, np.nan)
    return m.sort_values("tourney_date", ascending=False)

def summarize(pm):
    if pm.empty:
        return {}
    def avg(c):
        x = pd.to_numeric(pm[c], errors="coerce") if c in pm else pd.Series(dtype=float)
        return float(x.dropna().mean()) if x.notna().any() else np.nan
    return {
        "matches": len(pm),
        "wins": int(pm["win"].sum()),
        "losses": int(len(pm)-pm["win"].sum()),
        "win_pct": float(pm["win"].mean()*100),
        "aces": avg("aces"),
        "df": avg("df"),
        "serve_pct": avg("serve_points_won_pct"),
    }

def surface_filter(pm, surface):
    if surface == "All":
        return pm
    return pm[pm["surface"].astype(str).str.lower() == surface.lower()]

def h2h(matches, p1, p2):
    x = matches[((matches["winner_name"]==p1)&(matches["loser_name"]==p2)) |
                ((matches["winner_name"]==p2)&(matches["loser_name"]==p1))].copy()
    if x.empty: return x
    x["winner_display"] = x["winner_name"]
    return x.sort_values("tourney_date", ascending=False)

def model_probability(pm1, pm2, surface):
    a = summarize(surface_filter(pm1,surface))
    b = summarize(surface_filter(pm2,surface))
    if not a or not b or a["matches"] == 0 or b["matches"] == 0:
        return 0.5, "Insufficient sample"
    # Transparent model: recent surface W/L, serve efficiency, ace signal, and sample confidence.
    wr = (a["win_pct"] - b["win_pct"]) / 100
    sv = ((a["serve_pct"] if not np.isnan(a["serve_pct"]) else 50) -
          (b["serve_pct"] if not np.isnan(b["serve_pct"]) else 50)) / 100
    ac = ((a["aces"] if not np.isnan(a["aces"]) else 0) -
          (b["aces"] if not np.isnan(b["aces"]) else 0))
    ac_signal = np.tanh(ac/5) * 0.04
    sample = min(1.0, math.sqrt(min(a["matches"],b["matches"])/20))
    score = (0.70*wr + 0.26*sv + ac_signal) * (0.55 + 0.45*sample)
    p = 1/(1+math.exp(-5*score))
    return float(np.clip(p,0.05,0.95)), f"Surface sample: {a['matches']} vs {b['matches']} matches"

def pct(x):
    return "—" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.1f}%"

# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------
st.title("🎾 Tennis Betting Lab")
st.caption("Free public tennis databases • matchup intelligence • Kalshi edge calculator")

with st.sidebar:
    st.header("Data")
    if st.button("Refresh public databases"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data is downloaded directly from public GitHub repositories. No paid tennis API is required.")
    st.markdown("**Primary:** Jeff Sackmann ATP/WTA")
    st.markdown("**Advanced:** Match Charting Project")
    st.markdown("**Validation/reference:** Ultimate Tennis Statistics")

try:
    matches, players, source_status = load_database()
except Exception as e:
    st.error(f"Could not load the free public tennis database: {e}")
    st.stop()

mcp, mcp_status = load_mcp()

st.write(f"**Player directory:** {len(players):,} names • **match rows loaded:** {len(matches):,}")
with st.expander("Database/source status"):
    for s in source_status:
        st.write("•", s)
    for s in mcp_status:
        st.write("• MCP:", s)

tour = st.selectbox("Tour / database", ["All","ATP","WTA"])
surface = st.selectbox("Surface", ["All","Hard","Clay","Grass"])

c1,c2 = st.columns(2)
with c1:
    st.subheader("Player 1")
    q1 = st.text_input("Search player 1", placeholder="Rybakina, Sabalenka, Alcaraz, Sinner…")
    r1 = player_search(players,q1,tour)
    if r1.empty:
        st.warning("No matching player found.")
        st.stop()
    names1 = r1["name"].tolist()
    p1 = st.selectbox("Select exact player 1", names1, key="p1")
with c2:
    st.subheader("Player 2")
    q2 = st.text_input("Search player 2", placeholder="Tjen, Vidmanova, Sinner…")
    r2 = player_search(players,q2,tour)
    if r2.empty:
        st.warning("No matching player found.")
        st.stop()
    names2 = r2["name"].tolist()
    p2 = st.selectbox("Select exact player 2", names2, key="p2")

if p1 == p2:
    st.warning("Choose two different players.")
    st.stop()

if st.button("ANALYZE MATCHUP", type="primary", use_container_width=True):
    pm1 = add_player_perspective(matches,p1)
    pm2 = add_player_perspective(matches,p2)
    sf1 = surface_filter(pm1,surface)
    sf2 = surface_filter(pm2,surface)
    prob, note = model_probability(pm1,pm2,surface)

    st.divider()
    st.header(f"{p1} vs {p2}")
    a,b,c = st.columns(3)
    with a: st.metric(p1, f"{prob*100:.1f}% model win probability")
    with b: st.metric(p2, f"{(1-prob)*100:.1f}% model win probability")
    with c: st.metric("Model note", note)

    # H2H
    hh = h2h(matches,p1,p2)
    st.subheader("Head-to-head")
    if hh.empty:
        st.info("No H2H found in the loaded public match database.")
    else:
        wins1 = int((hh["winner_name"]==p1).sum())
        wins2 = int((hh["winner_name"]==p2).sum())
        st.write(f"**Overall:** {p1} {wins1}–{wins2} {p2}")
        st.dataframe(hh[["tourney_date","tourney_name","surface","winner_name","loser_name","score"]].head(20), use_container_width=True, hide_index=True)

    # Comparison
    st.subheader("Form & matchup statistics")
    s1 = summarize(sf1); s2 = summarize(sf2)
    rows = [
        ("Surface matches", s1.get("matches",0), s2.get("matches",0)),
        ("Surface win %", pct(s1.get("win_pct",np.nan)), pct(s2.get("win_pct",np.nan))),
        ("Avg aces", "—" if np.isnan(s1.get("aces",np.nan)) else f"{s1['aces']:.2f}", "—" if np.isnan(s2.get("aces",np.nan)) else f"{s2['aces']:.2f}"),
        ("Avg double faults", "—" if np.isnan(s1.get("df",np.nan)) else f"{s1['df']:.2f}", "—" if np.isnan(s2.get("df",np.nan)) else f"{s2['df']:.2f}"),
        ("Serve points won", pct(s1.get("serve_pct",np.nan)), pct(s2.get("serve_pct",np.nan))),
    ]
    st.table(pd.DataFrame(rows, columns=["Metric",p1,p2]))

    # Recent matches
    st.subheader("Recent matches")
    rr = []
    for name, pm in [(p1,pm1),(p2,pm2)]:
        x = pm.head(10).copy()
        x["Player"] = name
        x["Result"] = np.where(x["win"].eq(1),"W","L")
        rr.append(x[["Player","tourney_date","tourney_name","surface","Result","opponent","score","aces","df"]])
    recent = pd.concat(rr,ignore_index=True).sort_values("tourney_date",ascending=False)
    st.dataframe(recent, use_container_width=True, hide_index=True)

    # Advanced MCP supplement
    st.subheader("Advanced Match Charting Project supplement")
    if mcp.empty:
        st.info("MCP data could not be loaded right now.")
    else:
        # show any charted matches for either player; column names vary, so identify name columns dynamically
        text = mcp.astype(str)
        mask = text.apply(lambda col: col.str.contains(re.escape(p1),case=False,na=False)).any(axis=1) | \
               text.apply(lambda col: col.str.contains(re.escape(p2),case=False,na=False)).any(axis=1)
        mc = mcp[mask].head(20)
        if mc.empty:
            st.info("No charted matches found for these players in MCP. MCP is selective, not a complete match database.")
        else:
            st.dataframe(mc.head(20), use_container_width=True, hide_index=True)

    # Kalshi calculator, not a URL/API dependency
    st.subheader("Kalshi edge calculator")
    st.caption("Enter the current YES price you see on Kalshi. The app compares it with the model probability.")
    k1,k2 = st.columns(2)
    with k1:
        price1 = st.number_input(f"{p1} YES price (¢)", min_value=1, max_value=99, value=int(round(prob*100)), step=1)
    with k2:
        price2 = st.number_input(f"{p2} YES price (¢)", min_value=1, max_value=99, value=int(round((1-prob)*100)), step=1)
    p1m = price1/100
    p2m = price2/100
    e1 = prob-p1m
    e2 = (1-prob)-p2m
    ec1,ec2 = st.columns(2)
    with ec1:
        st.metric(f"{p1} edge", f"{e1*100:+.1f} pts", delta="Potential value" if e1>0.03 else "No clear edge")
    with ec2:
        st.metric(f"{p2} edge", f"{e2*100:+.1f} pts", delta="Potential value" if e2>0.03 else "No clear edge")

    st.info("This is a research/modeling tool, not a guarantee of outcome. Market price, injury/news, scheduling, and match conditions can change the true probability.")

with st.expander("What this database covers"):
    st.write("""
    The primary database is built from public Jeff Sackmann ATP/WTA files. The ATP repository includes tour-level results plus qualifying/Challenger and Futures files where available; WTA coverage is primarily tour-level. The app also adds Match Charting Project data when available for deeper shot/error/net-point context. Ultimate Tennis Statistics is treated as an external validation/reference source rather than scraped as a required dependency.
    """)
    st.write("The app refreshes public source files every 24 hours and has a manual refresh button.")
