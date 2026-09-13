
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime

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

# Use the GitHub Contents API instead of raw.githubusercontent.com.
# This avoids the 404 issue that can occur from Streamlit's runtime fetching
# the raw CSV URL.
GITHUB_API = {
    "ATP": "https://api.github.com/repos/JeffSackmann/tennis_atp/contents/atp_matches_2026.csv?ref=master",
    "WTA": "https://api.github.com/repos/JeffSackmann/tennis_wta/contents/wta_matches_2026.csv?ref=master",
}

@st.cache_data(ttl=21600, show_spinner="Loading tennis database…")
def load_matches(tour):
    url = GITHUB_API[tour]
    try:
        r = requests.get(url, timeout=30, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        payload = r.json()

        # GitHub returns the file as base64 through the Contents API.
        import base64
        content = base64.b64decode(payload["content"]).decode("utf-8")
        df = pd.read_csv(StringIO(content))
        df["date"] = pd.to_datetime(
            df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        return df
    except Exception as e:
        st.error(
            f"Could not load {tour} data. The tennis database request failed: {e}"
        )
        return pd.DataFrame()

def player_matches(df, player, surface="All"):
    if df.empty:
        return pd.DataFrame()
    w = df[df.winner.eq(player)].copy()
    w["result"] = "W"
    w["opponent"] = w["loser"]
    w["player_rank"] = w["w_rank"]
    w["opp_rank"] = w["l_rank"]
    w["aces_player"] = w["w_ace"]
    w["df_player"] = w["w_df"]
    w["svpt_player"] = w["w_svpt"]
    w["first_in"] = w["w_1stIn"]
    w["first_won"] = w["w_1stWon"]
    w["second_won"] = w["w_2ndWon"]
    w["sv_gms"] = w["w_SvGms"]
    w["bp_won"] = np.nan
    w["bp_saved"] = w["w_bpSaved"]
    w["bp_faced"] = w["w_bpFaced"]

    l = df[df.loser.eq(player)].copy()
    l["result"] = "L"
    l["opponent"] = l["winner"]
    l["player_rank"] = l["l_rank"]
    l["opp_rank"] = l["w_rank"]
    l["aces_player"] = l["l_ace"]
    l["df_player"] = l["l_df"]
    l["svpt_player"] = l["l_svpt"]
    l["first_in"] = l["l_1stIn"]
    l["first_won"] = l["l_1stWon"]
    l["second_won"] = l["l_2ndWon"]
    l["sv_gms"] = l["l_SvGms"]
    l["bp_won"] = np.nan
    l["bp_saved"] = l["l_bpSaved"]
    l["bp_faced"] = l["l_bpFaced"]

    d = pd.concat([w, l], ignore_index=True)
    if surface != "All":
        d = d[d.surface.eq(surface)]
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
    wins = int((d.result=="W").sum())
    losses = int((d.result=="L").sum())
    return {
        "sample": len(d),
        "wins": wins,
        "losses": losses,
        "win_rate": pct(wins, len(d)),
        "aces": safe_mean(d.aces_player),
        "df": safe_mean(d.df_player),
        "first_serve_pct": pct(d.first_in.sum(), d.svpt_player.sum()),
        "first_serve_points_won": pct(d.first_won.sum(), d.first_in.sum()),
        "second_serve_points_won": pct(d.second_won.sum(), (d.svpt_player-d.first_in).sum()),
        "bp_conversion": pct(d.bp_won.sum(), d.bp_faced.sum()),
        "bp_save": pct(d.bp_saved.sum(), d.bp_faced.sum()),
    }

def h2h(df, p1, p2, surface="All"):
    if df.empty:
        return pd.DataFrame()
    d = df[((df.winner.eq(p1)) & (df.loser.eq(p2))) |
           ((df.winner.eq(p2)) & (df.loser.eq(p1)))].copy()
    if surface != "All":
        d = d[d.surface.eq(surface)]
    return d.sort_values("date", ascending=False)

def model(df, p1, p2, surface):
    a = summarize(player_matches(df,p1,surface), 20)
    b = summarize(player_matches(df,p2,surface), 20)
    if not a or not b:
        return .5, .5, a, b

    # Transparent model:
    # 70% recent surface win-rate difference
    # 20% serve/ace signal
    # 10% data/sample confidence
    wr_a, wr_b = a.get("win_rate",.5), b.get("win_rate",.5)
    ace_a, ace_b = a.get("aces",np.nan), b.get("aces",np.nan)
    serve_a = np.nanmean([a.get("first_serve_points_won",np.nan), a.get("second_serve_points_won",np.nan)])
    serve_b = np.nanmean([b.get("first_serve_points_won",np.nan), b.get("second_serve_points_won",np.nan)])
    ace_signal = 0 if np.isnan(ace_a) or np.isnan(ace_b) else np.tanh((ace_a-ace_b)/5)
    serve_signal = 0 if np.isnan(serve_a) or np.isnan(serve_b) else (serve_a-serve_b)
    confidence = min(1, min(a.get("sample",0),b.get("sample",0))/20)
    score = 0.70*(wr_a-wr_b) + 0.08*ace_signal + 0.22*serve_signal
    score *= (0.65 + 0.35*confidence)
    p = 1/(1+np.exp(-4.2*score))
    return float(p), float(1-p), a, b

@st.cache_data(ttl=60, show_spinner=False)
def search_kalshi(p1,p2):
    url="https://external-api.kalshi.com/trade-api/v2/markets"
    try:
        r=requests.get(url, params={"limit":1000,"status":"open"}, timeout=15)
        r.raise_for_status()
        markets=r.json().get("markets",[])
        a,b=p1.lower(),p2.lower()
        hits=[]
        for m in markets:
            title=(m.get("title") or "").lower()
            if a in title and b in title:
                hits.append(m)
        return hits
    except Exception:
        return []

def fmt_pct(x):
    return "N/A" if pd.isna(x) else f"{x:.1%}"

st.title("🎾 Tennis Betting Lab")
st.caption("Matchup intelligence + Kalshi edge scanner")

tour = st.selectbox("Tour", ["ATP","WTA"])
surface = st.selectbox("Surface", ["All","Hard","Clay","Grass","Carpet"])

df = load_matches(tour)
if df.empty:
    st.stop()

players = sorted(set(df.winner.dropna()) | set(df.loser.dropna()))
p1 = st.selectbox("Player 1", players)
p2 = st.selectbox("Player 2", [p for p in players if p != p1])

if st.button("ANALYZE MATCHUP", type="primary"):
    p1p,p2p,s1,s2 = model(df,p1,p2,surface)
    hh=h2h(df,p1,p2,surface)
    r1=player_matches(df,p1,surface).head(10)
    r2=player_matches(df,p2,surface).head(10)

    st.subheader("Model")
    c1,c2=st.columns(2)
    c1.metric(p1,fmt_pct(p1p))
    c2.metric(p2,fmt_pct(p2p))

    st.subheader("Kalshi Edge")
    price=st.number_input(f"Current YES price for {p1} (¢)",1,99,50)
    market=price/100
    edge=p1p-market
    c1,c2,c3=st.columns(3)
    c1.metric("Market implied",fmt_pct(market))
    c2.metric("Model edge",f"{edge:+.1%}")
    signal="STRONG VALUE" if edge>=.08 else "VALUE" if edge>=.05 else "FAIR" if edge>-0.03 else "EXPENSIVE"
    c3.metric("Signal",signal)

    if edge >= .08:
        st.success("Model shows a meaningful positive edge. Verify live conditions and liquidity before trading.")
    elif edge >= .05:
        st.info("Potential edge. Check price movement, matchup context and market depth.")
    elif edge > -0.03:
        st.warning("Price is close to model fair value.")
    else:
        st.error("Player 1 is priced above the model estimate.")

    st.subheader("Head-to-Head")
    if hh.empty:
        st.write("No H2H matches found in the loaded database for this surface.")
    else:
        st.write(f"**{p1}: {(hh.winner==p1).sum()}** wins  |  **{p2}: {(hh.winner==p2).sum()}** wins")
        st.dataframe(hh[["date","tourney_name","surface","round","winner","loser","score"]],hide_index=True,use_container_width=True)

    st.subheader("Recent Form")
    a,b=st.columns(2)
    a.write(f"**{p1} — last 10**")
    a.dataframe(r1[["date","tourney_name","surface","round","result","opponent","score"]],hide_index=True,use_container_width=True)
    b.write(f"**{p2} — last 10**")
    b.dataframe(r2[["date","tourney_name","surface","round","result","opponent","score"]],hide_index=True,use_container_width=True)

    st.subheader("Stat Comparison — last 20")
    stat_rows=[]
    for label,key in [
        ("Win rate","win_rate"),("Aces / match","aces"),("Double faults / match","df"),
        ("1st serve %","first_serve_pct"),("1st serve points won","first_serve_points_won"),
        ("2nd serve points won","second_serve_points_won"),
        ("Break-point conversion","bp_conversion"),("Break points saved","bp_save")]:
        stat_rows.append({
            "Metric":label,
            p1: fmt_pct(s1.get(key,np.nan)) if "rate" in key or "%" in label or "points" in label.lower() or "conversion" in label.lower() or "saved" in label.lower()
                else ("N/A" if pd.isna(s1.get(key,np.nan)) else round(s1.get(key),2)),
            p2: fmt_pct(s2.get(key,np.nan)) if "rate" in key or "%" in label or "points" in label.lower() or "conversion" in label.lower() or "saved" in label.lower()
                else ("N/A" if pd.isna(s2.get(key,np.nan)) else round(s2.get(key),2)),
        })
    st.dataframe(pd.DataFrame(stat_rows),hide_index=True,use_container_width=True)

    st.subheader("Kalshi Market Search")
    markets=search_kalshi(p1,p2)
    if markets:
        rows=[]
        for m in markets[:15]:
            rows.append({
                "Market":m.get("title"),
                "YES bid":m.get("yes_bid"),
                "YES ask":m.get("yes_ask"),
                "Volume":m.get("volume"),
                "Ticker":m.get("ticker")
            })
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    else:
        st.caption("No matching open market was found automatically. Enter the current Kalshi price manually above.")

with st.expander("How I use this for Kalshi"):
    st.markdown("""
**DATA → FORM → H2H → MATCHUP → PROP → PROBABILITY → MARKET → EDGE**

1. Start with surface and recent form.
2. Use H2H as supporting evidence, not the whole prediction.
3. Compare serve/return indicators and opponent quality.
4. Treat the model probability as an estimate.
5. Compare it with the Kalshi price.
6. Look for a meaningful gap rather than forcing a trade.
""")

with st.expander("Data notes"):
    st.write("Historical match data is loaded from Jeff Sackmann's ATP/WTA datasets. Some matches do not contain complete point/stat fields, so missing values are left blank rather than estimated.")
    st.write("The Kalshi search uses public market data only and does not place orders.")
