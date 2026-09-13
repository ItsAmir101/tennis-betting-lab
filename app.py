
import io, re, math, unicodedata
from urllib.parse import quote
import requests
import pandas as pd
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Tennis Betting Lab V8", page_icon="🎾", layout="wide")
BASE = "https://www.tennisabstract.com"
PLAYER_URL = BASE + "/cgi-bin/player-classic.cgi?p={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TennisBettingLab/8.0)"}

def slug_candidates(name):
    n = unicodedata.normalize("NFKD", str(name).strip())
    n = "".join(c for c in n if not unicodedata.combining(c))
    compact = re.sub(r"[^A-Za-z0-9]", "", n)
    vals = [compact]
    if compact.lower().endswith("jr"):
        vals.append(compact[:-2])
    return list(dict.fromkeys(x for x in vals if x))

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_player(name_or_url):
    raw = str(name_or_url).strip()
    if not raw:
        raise ValueError("Enter a player name.")
    if raw.startswith(("http://", "https://")):
        candidates = [raw]
    else:
        candidates = [PLAYER_URL.format(quote(x)) for x in slug_candidates(raw)]
    last = ""
    for url in candidates:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            last = f"HTTP {r.status_code}"
            if r.status_code != 200:
                continue
            if "Tennis Abstract" not in r.text:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.get_text(" ", strip=True) if soup.title else ""
            if "Player Not Found" in title or "404" in title:
                continue
            try:
                tables = pd.read_html(io.StringIO(r.text))
            except Exception:
                tables = []
            return {"url": r.url, "html": r.text, "text": soup.get_text(" ", strip=True), "tables": tables}
        except Exception as e:
            last = str(e)
    raise ValueError(f"Could not find '{raw}' on Tennis Abstract ({last}). Use the optional Tennis Abstract URL field if needed.")

def num(x):
    if x is None: return np.nan
    s = str(x).replace(",", "").replace("%", "").strip()
    if s in ("", "-", "—", "nan", "NaN"): return np.nan
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan

def profile(data):
    t = data["text"]
    out = {"name": None, "rank": np.nan, "elo": np.nan, "elo_rank": np.nan, "url": data["url"]}
    m = re.search(r"Tennis Abstract:\s*(.*?)\s+Match Results", data["html"], re.I | re.S)
    if m: out["name"] = BeautifulSoup(m.group(1), "html.parser").get_text(" ", strip=True)
    for pat, key in [
        (r"Current rank:\s*(\d+)", "rank"),
        (r"Elo rank:\s*(\d+)", "elo_rank"),
        (r"Elo rank:\s*\d+\s*\(rating:\s*([0-9.]+)\)", "elo"),
    ]:
        m = re.search(pat, t, re.I)
        if m: out[key] = num(m.group(1))
    # Tennis Abstract's Last-52 summary: W-L, TB W-L, then Ace%, 1stIn%, 1st%, 2nd%, Hold%, SPW, Brk%, RPW, TPW, DR.
    m = re.search(
        r"Last 52\s+(\d+)-(\d+)\s+\((\d+)%\)\s+"
        r"([0-9.]+)-([0-9.]+)\s+\((\d+)%\)\s+"
        r"([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+"
        r"([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)",
        t, re.I)
    if m:
        g = m.groups()
        keys = ["wins52","losses52","win_pct52","tb_wins","tb_losses","tb_pct","ace_pct","first_in","first_pct","second_pct","hold_pct","spw","brk_pct","rpw","tpw","dr"]
        for k,v in zip(keys,g): out[k] = num(v)
    # Surface win-loss summaries.
    for s in ["Hard","Clay","Grass","Grand Slams","vs Top 10"]:
        m = re.search(rf"{re.escape(s)}\s+(\d+)-(\d+)\s+\((\d+)%\)", t, re.I)
        if m:
            key = s.lower().replace(" ","_")
            out[key+"_wins"], out[key+"_losses"], out[key+"_win_pct"] = map(num, m.groups())
    return out

def match_table(data):
    best, score = None, -1
    for df in data["tables"]:
        joined = " ".join(str(c).lower() for c in df.columns)
        sc = sum(term in joined for term in ["date","opponent","result","surface","score","round"])
        if sc > score and sc >= 2:
            best, score = df.copy(), sc
    return best if best is not None else pd.DataFrame()

def normalize_matches(df):
    if df.empty: return df
    df.columns = [str(c).strip() for c in df.columns]
    low = {str(c).lower(): c for c in df.columns}
    def pick(words):
        for k,c in low.items():
            if any(w in k for w in words): return c
        return None
    datec, oppc, resc, surfc = pick(["date"]), pick(["opponent","opp"]), pick(["result","w/l"]), pick(["surface"])
    out = df.copy()
    if datec: out["_date"] = pd.to_datetime(out[datec], errors="coerce")
    else: out["_date"] = pd.NaT
    if resc: out["_win"] = out[resc].astype(str).str.upper().str.startswith("W")
    else: out["_win"] = np.nan
    out["_surface"] = out[surfc].astype(str) if surfc else ""
    out["_opponent"] = out[oppc].astype(str) if oppc else ""
    return out

def recent(df, n=10, surface=None):
    if df.empty: return None
    x = df.dropna(subset=["_date"]).sort_values("_date", ascending=False)
    if surface: x = x[x["_surface"].str.lower() == surface.lower()]
    x = x.head(n)
    if x.empty or x["_win"].isna().all(): return None
    w = int(x["_win"].fillna(False).sum())
    return {"n":len(x), "wins":w, "losses":len(x)-w, "pct":100*w/len(x)}

def model(a,b,f1=None,f2=None):
    ea, eb = num(a.get("elo")), num(b.get("elo"))
    ea = ea if np.isfinite(ea) else 1500
    eb = eb if np.isfinite(eb) else 1500
    elo = 1/(1+10**((eb-ea)/400))
    vals, weights = [elo], [0.62]
    wa, wb = num(a.get("win_pct52")), num(b.get("win_pct52"))
    if np.isfinite(wa) and np.isfinite(wb):
        vals.append((wa/100)/((wa/100)+(wb/100))); weights.append(.18)
    sa, sb = num(a.get("spw")), num(b.get("spw"))
    if np.isfinite(sa) and np.isfinite(sb):
        vals.append(.5+(sa-sb)/100); weights.append(.10)
    ra, rb = num(a.get("rpw")), num(b.get("rpw"))
    if np.isfinite(ra) and np.isfinite(rb):
        vals.append(.5+(ra-rb)/100); weights.append(.06)
    if f1 and f2 and f1["n"] >= 3 and f2["n"] >= 3:
        vals.append((f1["pct"]/100)/((f1["pct"]/100)+(f2["pct"]/100))); weights.append(.04)
    return min(.95,max(.05,float(np.average(vals,weights=weights))))

st.title("🎾 Tennis Betting Lab V8")
st.caption("Tennis Abstract-powered matchup research • no paid tennis API")

with st.sidebar:
    tour = st.selectbox("Tour", ["ATP","WTA"])
    surface = st.selectbox("Surface", ["Hard","Clay","Grass","Carpet"])
    st.markdown("### Data")
    st.write("Primary source: Tennis Abstract player pages")
    st.caption("Tennis Abstract pages/rankings are updated by Tennis Abstract. The probability below is an independent research model.")

a_col,b_col = st.columns(2)
with a_col: p1 = st.text_input("Player 1", placeholder="Jannik Sinner")
with b_col: p2 = st.text_input("Player 2", placeholder="Carlos Alcaraz")

with st.expander("Fallback for unusual player names"):
    u1 = st.text_input("Player 1 Tennis Abstract URL")
    u2 = st.text_input("Player 2 Tennis Abstract URL")

x,y,z = st.columns(3)
with x: go = st.button("Analyze matchup", type="primary", use_container_width=True)
with y: k1 = st.number_input("Player 1 Kalshi YES (¢)",0.0,100.0,50.0,.5)
with z: k2 = st.number_input("Player 2 Kalshi YES (¢)",0.0,100.0,50.0,.5)

if go:
    q1, q2 = u1.strip() or p1.strip(), u2.strip() or p2.strip()
    if not q1 or not q2:
        st.error("Enter both players."); st.stop()
    try:
        with st.spinner("Loading Tennis Abstract..."):
            d1,d2 = fetch_player(q1),fetch_player(q2)
            a,b = profile(d1),profile(d2)
            m1,m2 = normalize_matches(match_table(d1)),normalize_matches(match_table(d2))
    except Exception as e:
        st.error(str(e)); st.stop()

    n1 = a.get("name") or p1
    n2 = b.get("name") or p2
    f1,f2 = recent(m1,10),recent(m2,10)
    s1,s2 = recent(m1,10,surface),recent(m2,10,surface)
    pr1 = model(a,b,s1 or f1,s2 or f2); pr2=1-pr1
    fair1,fair2=pr1*100,pr2*100
    edge1,edge2=fair1-k1,fair2-k2

    st.subheader(f"{n1} vs {n2}")
    st.caption(f"Source: Tennis Abstract • {surface} • player pages linked below")
    c1,c2,c3,c4=st.columns(4)
    c1.metric(n1,f"{pr1*100:.1f}%")
    c2.metric(n2,f"{pr2*100:.1f}%")
    c3.metric(f"{n1} fair price",f"{fair1:.1f}¢")
    c4.metric(f"{n2} fair price",f"{fair2:.1f}¢")
    if edge1 > edge2: st.success(f"Kalshi edge lean: {n1} ({edge1:+.1f}¢)")
    elif edge2 > edge1: st.success(f"Kalshi edge lean: {n2} ({edge2:+.1f}¢)")
    else: st.info("The two entered prices produce the same edge.")

    rows=[
        ("Current rank",a.get("rank"),b.get("rank")),
        ("Tennis Abstract Elo",a.get("elo"),b.get("elo")),
        ("Elo rank",a.get("elo_rank"),b.get("elo_rank")),
        ("Last 52 win %",a.get("win_pct52"),b.get("win_pct52")),
        ("Ace %",a.get("ace_pct"),b.get("ace_pct")),
        ("1st serve in %",a.get("first_in"),b.get("first_in")),
        ("1st serve points won %",a.get("first_pct"),b.get("first_pct")),
        ("2nd serve points won %",a.get("second_pct"),b.get("second_pct")),
        ("Hold %",a.get("hold_pct"),b.get("hold_pct")),
        ("Service points won %",a.get("spw"),b.get("spw")),
        ("Break %",a.get("brk_pct"),b.get("brk_pct")),
        ("Return points won %",a.get("rpw"),b.get("rpw")),
        ("Total points won %",a.get("tpw"),b.get("tpw")),
    ]
    st.markdown("### Player comparison")
    st.dataframe(pd.DataFrame(rows,columns=["Metric",n1,n2]),use_container_width=True,hide_index=True)

    st.markdown("### Recent form")
    q1,q2=st.columns(2)
    with q1:
        st.write(f"**{n1}**")
        st.write(f"Last 10: {f1['wins']}-{f1['losses']} ({f1['pct']:.0f}%)" if f1 else "Recent match table unavailable.")
        st.write(f"{surface}: {s1['wins']}-{s1['losses']} ({s1['pct']:.0f}%)" if s1 else f"{surface}: insufficient parsed matches.")
    with q2:
        st.write(f"**{n2}**")
        st.write(f"Last 10: {f2['wins']}-{f2['losses']} ({f2['pct']:.0f}%)" if f2 else "Recent match table unavailable.")
        st.write(f"{surface}: {s2['wins']}-{s2['losses']} ({s2['pct']:.0f}%)" if s2 else f"{surface}: insufficient parsed matches.")

    st.markdown("### Kalshi edge")
    st.dataframe(pd.DataFrame([
        [n1,f"{pr1*100:.1f}%",f"{fair1:.1f}¢",f"{k1:.1f}¢",f"{edge1:+.1f}¢"],
        [n2,f"{pr2*100:.1f}%",f"{fair2:.1f}¢",f"{k2:.1f}¢",f"{edge2:+.1f}¢"],
    ],columns=["Player","Model probability","Fair price","Kalshi price","Edge"]),use_container_width=True,hide_index=True)

    st.markdown("### Source pages")
    st.write(d1["url"]); st.write(d2["url"])
    st.warning("Research tool only. Probabilities are estimates, not guarantees. Verify the Kalshi contract, surface, start time, injuries/withdrawals, and market price before trading.")

st.markdown("---")
st.caption("Tennis Abstract: https://www.tennisabstract.com/")
