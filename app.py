
import base64
import io
import os
import re
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Tennis Betting Lab V7", page_icon="🎾", layout="wide")

# V7 uses Jeff Sackmann's public tennis datasets directly through GitHub's
# Contents API. No paid tennis API and no local CSV folder are required.
ATP_REPO = "JeffSackmann/tennis_atp"
WTA_REPO = "JeffSackmann/tennis_wta"
API = "https://api.github.com"

HEADERS = {
    "Accept": "application/vnd.github.raw+json",
    "User-Agent": "Tennis-Betting-Lab-V7",
}

def gh_get_json(url):
    r = requests.get(url, headers={"Accept": "application/vnd.github+json",
                                   "User-Agent": "Tennis-Betting-Lab-V7"}, timeout=45)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=86400, show_spinner=False)
def github_csv(repo, path):
    # Contents API returns base64 content for repository files.
    url = f"{API}/repos/{repo}/contents/{path}"
    r = requests.get(url, headers={"Accept": "application/vnd.github+json",
                                   "User-Agent": "Tennis-Betting-Lab-V7"}, timeout=60)
    r.raise_for_status()
    obj = r.json()
    if obj.get("encoding") != "base64":
        raise ValueError(f"Unexpected GitHub response for {path}")
    raw = base64.b64decode(obj["content"])
    return pd.read_csv(io.BytesIO(raw), low_memory=False)

def available_years(repo):
    items = gh_get_json(f"{API}/repos/{repo}/contents/")
    years = set()
    for item in items:
        m = re.match(r"(?:atp|wta)_matches_(\d{4})\.csv$", item.get("name", ""))
        if m:
            years.add(int(m.group(1)))
    return sorted(years)

@st.cache_data(ttl=86400, show_spinner=False)
def load_player_master(tour):
    repo = ATP_REPO if tour == "ATP" else WTA_REPO
    filename = "atp_players.csv" if tour == "ATP" else "wta_players.csv"
    p = github_csv(repo, filename)
    # Sackmann player files use player_id, first_name, last_name.
    p = p.rename(columns={c: c.lower() for c in p.columns})
    p["first_name"] = p.get("first_name", "").fillna("").astype(str)
    p["last_name"] = p.get("last_name", "").fillna("").astype(str)
    p["player_id"] = p.get("player_id", "").astype(str)
    p["display_name"] = (p["first_name"].str.strip() + " " + p["last_name"].str.strip()).str.strip()
    p["search_name"] = (p["display_name"] + " " + p["last_name"]).str.lower()
    return p.drop_duplicates("player_id")

@st.cache_data(ttl=86400, show_spinner=False)
def load_matches(tour, years, include_lower=True):
    repo = ATP_REPO if tour == "ATP" else WTA_REPO
    prefix = "atp" if tour == "ATP" else "wta"
    frames = []
    for y in years:
        candidates = [f"{prefix}_matches_{y}.csv"]
        if include_lower:
            candidates.append(
                f"{prefix}_matches_qual_chall_{y}.csv" if tour == "ATP"
                else f"{prefix}_matches_qual_itf_{y}.csv"
            )
        for path in candidates:
            try:
                d = github_csv(repo, path)
                d["season_year"] = y
                d["source_file"] = path
                frames.append(d)
            except requests.HTTPError:
                # Some years do not have the optional lower-level file.
                continue
    if not frames:
        raise RuntimeError("No match files could be loaded from the public dataset.")
    d = pd.concat(frames, ignore_index=True, sort=False)
    return normalize_sackmann(d)

def normalize_sackmann(d):
    d = d.copy()
    if "tourney_date" in d.columns:
        d["date"] = pd.to_datetime(d["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        d["date"] = pd.NaT

    for c in ["winner_name", "loser_name", "surface", "tourney_name", "round"]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].fillna("").astype(str)

    numeric = [
        "winner_id","loser_id","winner_rank","loser_rank","winner_rank_points",
        "loser_rank_points","w_ace","l_ace","w_df","l_df",
        "w_svpt","l_svpt","w_1stIn","l_1stIn","w_1stWon","l_1stWon",
        "w_2ndWon","l_2ndWon","w_SvGms","l_SvGms","w_bpSaved","l_bpSaved",
        "w_bpFaced","l_bpFaced","w_bpWon","l_bpWon"
    ]
    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.sort_values("date", ascending=False).reset_index(drop=True)

def normalize_query(q):
    return re.sub(r"[^a-z0-9 ]+", "", str(q).lower()).strip()

def player_options(master, query):
    names = master[["player_id","display_name"]].copy()
    q = normalize_query(query)
    if not q:
        return names.sort_values("display_name").to_dict("records")

    last = master["last_name"].str.lower()
    first = master["first_name"].str.lower()
    full = master["display_name"].str.lower()

    mask = (
        full.str.contains(q, regex=False, na=False) |
        last.str.contains(q, regex=False, na=False) |
        first.str.contains(q, regex=False, na=False)
    )
    return master.loc[mask, ["player_id","display_name"]].sort_values("display_name").to_dict("records")

def player_matches(d, player_id, player_name):
    pid = str(player_id)
    w_id = d["winner_id"].astype(str) if "winner_id" in d.columns else pd.Series("", index=d.index)
    l_id = d["loser_id"].astype(str) if "loser_id" in d.columns else pd.Series("", index=d.index)
    x = d[(w_id == pid) | (l_id == pid)].copy()
    if x.empty:
        x = d[(d["winner_name"] == player_name) | (d["loser_name"] == player_name)].copy()
    return x.sort_values("date", ascending=False)

def side_value(row, player_id, player_name, winner_col, loser_col):
    is_w = str(row.get("winner_id", "")) == str(player_id) or row.get("winner_name") == player_name
    return row.get(winner_col, np.nan) if is_w else row.get(loser_col, np.nan)

def summarize(d, pid, pname, surface="All", n=20):
    x = player_matches(d, pid, pname)
    if surface != "All":
        x = x[x["surface"].eq(surface)]
    x = x.head(n)
    if x.empty:
        return {"n":0,"wins":0,"losses":0,"win_pct":np.nan,"aces":np.nan,"df":np.nan,
                "first_serve_pct":np.nan,"first_serve_won_pct":np.nan,
                "second_serve_won_pct":np.nan,"bp_saved_pct":np.nan,"bp_won_pct":np.nan}

    wins = 0
    vals = {k: [] for k in ["ace","df","first_serve_pct","first_serve_won_pct",
                             "second_serve_won_pct","bp_saved_pct","bp_won_pct"]}

    for _, r in x.iterrows():
        is_w = str(r.get("winner_id","")) == str(pid) or r.get("winner_name") == pname
        wins += int(is_w)

        pairs = {
            "ace": ("w_ace","l_ace"),
            "df": ("w_df","l_df"),
        }
        for key,(wc,lc) in pairs.items():
            v = r.get(wc if is_w else lc, np.nan)
            if pd.notna(v): vals[key].append(float(v))

        svpt = r.get("w_svpt" if is_w else "l_svpt", np.nan)
        first_in = r.get("w_1stIn" if is_w else "l_1stIn", np.nan)
        first_won = r.get("w_1stWon" if is_w else "l_1stWon", np.nan)
        second_won = r.get("w_2ndWon" if is_w else "l_2ndWon", np.nan)
        bp_saved = r.get("w_bpSaved" if is_w else "l_bpSaved", np.nan)
        bp_faced = r.get("w_bpFaced" if is_w else "l_bpFaced", np.nan)
        bp_won = r.get("w_bpWon" if is_w else "l_bpWon", np.nan)
        opp_bp_faced = r.get("l_bpFaced" if is_w else "w_bpFaced", np.nan)

        if pd.notna(svpt) and svpt > 0:
            if pd.notna(first_in): vals["first_serve_pct"].append(float(first_in/svpt))
            if pd.notna(first_won) and pd.notna(first_in) and first_in > 0:
                vals["first_serve_won_pct"].append(float(first_won/first_in))
            if pd.notna(second_won) and pd.notna(svpt) and pd.notna(first_in) and svpt-first_in > 0:
                vals["second_serve_won_pct"].append(float(second_won/(svpt-first_in)))
        if pd.notna(bp_saved) and pd.notna(bp_faced) and bp_faced > 0:
            vals["bp_saved_pct"].append(float(bp_saved/bp_faced))
        if pd.notna(bp_won) and pd.notna(opp_bp_faced) and opp_bp_faced > 0:
            vals["bp_won_pct"].append(float(bp_won/opp_bp_faced))

    def mean(k):
        return np.mean(vals[k]) if vals[k] else np.nan

    return {
        "n":len(x), "wins":wins, "losses":len(x)-wins,
        "win_pct":wins/len(x),
        "aces":mean("ace"), "df":mean("df"),
        "first_serve_pct":mean("first_serve_pct"),
        "first_serve_won_pct":mean("first_serve_won_pct"),
        "second_serve_won_pct":mean("second_serve_won_pct"),
        "bp_saved_pct":mean("bp_saved_pct"), "bp_won_pct":mean("bp_won_pct")
    }

def h2h(d, p1id, p1, p2id, p2, surface="All"):
    w1 = d["winner_id"].astype(str).eq(str(p1id)) & d["loser_id"].astype(str).eq(str(p2id))
    w2 = d["winner_id"].astype(str).eq(str(p2id)) & d["loser_id"].astype(str).eq(str(p1id))
    x = d[w1 | w2].copy()
    # Fallback for datasets where IDs are missing.
    if x.empty:
        x = d[((d["winner_name"].eq(p1)) & (d["loser_name"].eq(p2))) |
              ((d["winner_name"].eq(p2)) & (d["loser_name"].eq(p1)))].copy()
    if surface != "All":
        x = x[x["surface"].eq(surface)]
    return x.sort_values("date", ascending=False)

def build_elo(d, surface="All"):
    x = d.dropna(subset=["date"]).sort_values("date")
    ratings = {}
    sr = {}
    for _, r in x.iterrows():
        a,b = str(r.get("winner_id","")), str(r.get("loser_id",""))
        if a == "nan" or b == "nan" or not a or not b:
            a,b = r["winner_name"], r["loser_name"]
        surf = r["surface"]
        ratings.setdefault(a,1500.0); ratings.setdefault(b,1500.0)
        sr.setdefault((a,surf),1500.0); sr.setdefault((b,surf),1500.0)
        if surface == "All" or surf == surface:
            ra,rb=ratings[a],ratings[b]
            ea=1/(1+10**((rb-ra)/400))
            ratings[a]+=24*(1-ea); ratings[b]+=24*(0-(1-ea))
            rsa,rsb=sr[(a,surf)],sr[(b,surf)]
            es=1/(1+10**((rsb-rsa)/400))
            sr[(a,surf)]+=28*(1-es); sr[(b,surf)]+=28*(0-(1-es))
    return ratings,sr

def model(d,p1id,p1,p2id,p2,surface):
    ratings,sr=build_elo(d,surface)
    key1=str(p1id); key2=str(p2id)
    r1=ratings.get(key1,ratings.get(p1,1500)); r2=ratings.get(key2,ratings.get(p2,1500))
    elo=1/(1+10**((r2-r1)/400))
    if surface != "All":
        s1r=sr.get((key1,surface),sr.get((p1,surface),1500))
        s2r=sr.get((key2,surface),sr.get((p2,surface),1500))
        se=1/(1+10**((s2r-s1r)/400))
        elo=.35*elo+.65*se

    s1=summarize(d,p1id,p1,surface,20); s2=summarize(d,p2id,p2,surface,20)
    f1=s1["win_pct"]; f2=s2["win_pct"]
    wr=.5 if pd.isna(f1) or pd.isna(f2) else .5+.5*(f1-f2)

    def diff_prob(a,b,scale):
        if pd.isna(a) or pd.isna(b): return .5
        return float(np.clip(.5+scale*(a-b),.05,.95))
    serve=diff_prob(s1["first_serve_won_pct"],s2["first_serve_won_pct"],1.8)
    bp=diff_prob(s1["bp_won_pct"],s2["bp_won_pct"],1.0)
    ace=diff_prob(s1["aces"],s2["aces"],.025)
    h=h2h(d,p1id,p1,p2id,p2,surface)
    hp=.5 if h.empty else float((h["winner_id"].astype(str)==str(p1id)).mean())
    if hp == 0.5 and not h.empty:
        hp=float((h["winner_name"]==p1).mean())

    n=min(s1["n"],s2["n"])
    conf=min(1,n/15)
    raw=.45*elo+.22*wr+.13*serve+.08*bp+.05*ace+.07*hp
    p=.5+(raw-.5)*(.45+.55*conf)
    return np.clip(p,.01,.99),1-np.clip(p,.01,.99),s1,s2,h,conf

def pct(x): return "—" if pd.isna(x) else f"{x*100:.1f}%"
def num(x): return "—" if pd.isna(x) else f"{x:.2f}"

st.title("🎾 Tennis Betting Lab V7")
st.caption("Full public ATP/WTA player directory • multi-year matches • surface form • H2H • Elo • Kalshi edge")

with st.sidebar:
    st.header("Database")
    st.write("Public Jeff Sackmann datasets via GitHub Contents API.")
    st.caption("No paid tennis API is required.")
    include_lower = st.checkbox("Include Challenger / qualifying / ITF", value=True)
    if st.button("Clear cached database"):
        st.cache_data.clear()
        st.rerun()

tour=st.selectbox("Tour",["ATP","WTA"])

try:
    master=load_player_master(tour)
    years=available_years(ATP_REPO if tour=="ATP" else WTA_REPO)
except Exception as e:
    st.error(f"Could not load the public player database: {e}")
    st.stop()

current_year=max(years)
year_options=[y for y in years if y>=2021]
year_start=st.selectbox("Match history starting year",year_options,index=max(0,len(year_options)-6))
selected_years=list(range(year_start,current_year+1))

st.write(f"**Player directory:** {len(master):,} players • **Years:** {year_start}–{current_year}")

query=st.text_input("Search player",placeholder="Try a full name, last name, or part of a name")
opts=player_options(master,query)
if query and not opts:
    st.warning("No player matched. Try another spelling or just the last name.")
    st.stop()

# Keep selectboxes manageable.
opts=opts[:300]
labels=[x["display_name"] for x in opts]
if not labels:
    labels=master["display_name"].sort_values().head(300).tolist()

c1,c2=st.columns(2)
with c1:
    p1name=st.selectbox("Player 1",labels,key="p1")
with c2:
    p2default=1 if len(labels)>1 else 0
    p2name=st.selectbox("Player 2",labels,index=p2default,key="p2")

p1row=master[master["display_name"].eq(p1name)].iloc[0]
p2row=master[master["display_name"].eq(p2name)].iloc[0]

if st.button("LOAD MATCH DATA + ANALYZE",type="primary",use_container_width=True):
    if p1name==p2name:
        st.error("Choose two different players.")
        st.stop()
    try:
        with st.spinner(f"Loading {year_start}–{current_year} {tour} match data…"):
            matches=load_matches(tour,selected_years,include_lower)
        st.session_state["matches"]=matches
        st.session_state["analysis"]=model(
            matches,p1row["player_id"],p1name,p2row["player_id"],p2name,"All"
        )
        st.session_state["p1id"]=p1row["player_id"]
        st.session_state["p2id"]=p2row["player_id"]
    except Exception as e:
        st.error(f"Database load failed: {e}")
        st.stop()

if "matches" not in st.session_state:
    st.info("Search for two players, choose them, then tap **LOAD MATCH DATA + ANALYZE**.")
    st.stop()

matches=st.session_state["matches"]

surfaces=["All"]+sorted([s for s in matches["surface"].dropna().unique() if str(s).strip()])
surface=st.selectbox("Surface",surfaces)
p1id=st.session_state["p1id"]; p2id=st.session_state["p2id"]

p1prob,p2prob,s1,s2,h,conf=model(matches,p1id,p1name,p2id,p2name,surface)

st.divider()
st.subheader(f"{p1name} vs {p2name}")

a,b=st.columns(2)
with a:
    st.metric(p1name,f"{p1prob*100:.1f}%",f"Fair price {p1prob*100:.0f}¢")
with b:
    st.metric(p2name,f"{p2prob*100:.1f}%",f"Fair price {p2prob*100:.0f}¢")

st.caption(f"Model confidence: {conf*100:.0f}% • Analysis uses {year_start}–{current_year} data and the selected surface.")

st.subheader("Recent form")
form=pd.DataFrame({
    "Metric":["Matches","Wins","Win %","Aces / match","Double faults / match",
              "1st serve in","1st serve points won","2nd serve points won",
              "Break points won","Break points saved"],
    p1name:[s1["n"],s1["wins"],pct(s1["win_pct"]),num(s1["aces"]),num(s1["df"]),
            pct(s1["first_serve_pct"]),pct(s1["first_serve_won_pct"]),pct(s1["second_serve_won_pct"]),
            pct(s1["bp_won_pct"]),pct(s1["bp_saved_pct"])],
    p2name:[s2["n"],s2["wins"],pct(s2["win_pct"]),num(s2["aces"]),num(s2["df"]),
            pct(s2["first_serve_pct"]),pct(s2["first_serve_won_pct"]),pct(s2["second_serve_won_pct"]),
            pct(s2["bp_won_pct"]),pct(s2["bp_saved_pct"])]
})
st.dataframe(form,use_container_width=True,hide_index=True)

st.subheader("Head-to-head")
if h.empty:
    st.write("No head-to-head matches found in the selected data.")
else:
    hv=h[["date","tourney_name","round","surface","winner_name","loser_name"]].copy()
    hv["date"]=hv["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(hv,use_container_width=True,hide_index=True)

st.subheader("Kalshi edge calculator")
k1,k2=st.columns(2)
with k1:
    price1=st.number_input(f"{p1name} YES price (¢)",1,99,int(round(p1prob*100)),key="k1")
with k2:
    price2=st.number_input(f"{p2name} YES price (¢)",1,99,int(round(p2prob*100)),key="k2")

e1,e2=st.columns(2)
with e1: st.metric(f"{p1name} edge",f"{p1prob*100-price1:+.1f}¢")
with e2: st.metric(f"{p2name} edge",f"{p2prob*100-price2:+.1f}¢")

st.caption("Research model only. It does not guarantee an outcome or account for every factor, including injuries, withdrawals, weather, or live market movement.")
