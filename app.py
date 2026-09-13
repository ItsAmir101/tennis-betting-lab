import os, io, re, math, time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title='Tennis Betting Lab V6', page_icon='🎾', layout='wide')

YEARS = list(range(2021, 2027))
HEADERS = {
    'User-Agent': 'Tennis-Betting-Lab/6.0',
    'Accept': 'application/vnd.github.raw+json',
}

# GitHub's API is used instead of raw.githubusercontent.com/jsDelivr.
# It is public and requires no paid tennis API key.
DATA_DIR = os.path.dirname(__file__)  # CSV files are stored in the repository root

def load_local_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing local database file: {filename}")
    return pd.read_csv(path, low_memory=False)


@st.cache_data(ttl=86400, show_spinner=False)
def load_players(tour):
    # Build the searchable player directory directly from the local match database.
    # This keeps the app independent of external tennis APIs.
    frames = []
    if tour == "ATP":
        frames.append(load_local_csv("2026-atp-season.csv"))
    else:
        frames.append(load_local_csv("2026-wta-season.csv"))
    d = pd.concat(frames, ignore_index=True, sort=False)

    names = pd.concat([
        d[["winner_name"]].rename(columns={"winner_name": "display"}),
        d[["loser_name"]].rename(columns={"loser_name": "display"})
    ], ignore_index=True)
    names["display"] = names["display"].fillna("").astype(str).str.strip()
    return names[names["display"].ne("")].drop_duplicates("display").sort_values("display")


@st.cache_data(ttl=86400, show_spinner=False)
def load_all_matches(tour):
    filename = "2026-atp-season.csv" if tour == "ATP" else "2026-wta-season.csv"
    return load_local_csv(filename)


def clean_matches(df):
    d = df.copy()
    if 'tourney_date' in d.columns:
        d['tourney_date'] = pd.to_datetime(d['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
    numeric = [
        'winner_rank','loser_rank','w_ace','l_ace','w_df','l_df','w_svpt','l_svpt',
        'w_1stIn','l_1stIn','w_1stWon','l_1stWon','w_2ndWon','l_2ndWon',
        'w_SvGms','l_SvGms','w_bpWon','l_bpWon','w_bpSaved','l_bpSaved',
        'w_bpFaced','l_bpFaced'
    ]
    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')
    for c, default in [('surface','Unknown'), ('tourney_name',''), ('tourney_level','')]:
        if c not in d.columns:
            d[c] = default
        d[c] = d[c].fillna(default)
    if 'winner_name' not in d.columns or 'loser_name' not in d.columns:
        raise RuntimeError('The downloaded match data does not contain winner_name/loser_name columns.')
    return d.sort_values('tourney_date', ascending=False)


def normalize_query(q):
    return re.sub(r'[^a-z0-9 ]+', '', q.lower()).strip()


def find_player_options(players, query):
    if not query:
        return players['display'].tolist()
    q = normalize_query(query)
    names = players['display'].tolist()
    exact = [n for n in names if normalize_query(n) == q]
    if exact:
        return exact
    return [n for n in names if q in normalize_query(n)]


def player_matches(d, player, surface='All'):
    x = d[(d['winner_name'].eq(player)) | (d['loser_name'].eq(player))].copy()
    if surface != 'All':
        x = x[x['surface'].eq(surface)]
    return x.sort_values('tourney_date', ascending=False)


def result_for(row, player):
    return 'W' if row['winner_name'] == player else 'L'


def summarize(d, player, surface='All', n=20):
    x = player_matches(d, player, surface).head(n)
    if x.empty:
        return {'n':0,'wins':0,'losses':0,'win_pct':np.nan,'aces':np.nan,'df':np.nan,'serve_pts_won':np.nan,'hold_pct':np.nan,'break_pct':np.nan}
    wins = int((x.winner_name == player).sum())
    aces, dfs, spw, holds, breaks = [], [], [], [], []
    for _, r in x.iterrows():
        is_w = r['winner_name'] == player
        prefix = 'w_' if is_w else 'l_'
        opp = 'l_' if is_w else 'w_'
        if pd.notna(r.get(prefix+'ace', np.nan)): aces.append(r.get(prefix+'ace'))
        if pd.notna(r.get(prefix+'df', np.nan)): dfs.append(r.get(prefix+'df'))
        svpt = r.get(prefix+'svpt', np.nan)
        first = r.get(prefix+'1stIn', np.nan)
        fw = r.get(prefix+'1stWon', np.nan)
        sw = r.get(prefix+'2ndWon', np.nan)
        if pd.notna(svpt) and pd.notna(first) and pd.notna(fw) and pd.notna(sw) and svpt > 0:
            spw.append((fw + sw) / svpt)
        opp_bpw = r.get(opp+'bpWon', np.nan)
        opp_bpf = r.get(opp+'bpFaced', np.nan)
        if pd.notna(opp_bpw) and pd.notna(opp_bpf) and opp_bpf > 0:
            holds.append(1 - opp_bpw / opp_bpf)
        bpw = r.get(prefix+'bpWon', np.nan)
        bpf = r.get(opp+'bpFaced', np.nan)
        if pd.notna(bpw) and pd.notna(bpf) and bpf > 0:
            breaks.append(bpw / bpf)
    return {
        'n': len(x), 'wins': wins, 'losses': len(x)-wins, 'win_pct': wins/len(x),
        'aces': np.nanmean(aces) if aces else np.nan,
        'df': np.nanmean(dfs) if dfs else np.nan,
        'serve_pts_won': np.nanmean(spw) if spw else np.nan,
        'hold_pct': np.nanmean(holds) if holds else np.nan,
        'break_pct': np.nanmean(breaks) if breaks else np.nan,
    }


def h2h(d, p1, p2, surface='All'):
    x = d[((d.winner_name==p1)&(d.loser_name==p2)) | ((d.winner_name==p2)&(d.loser_name==p1))].copy()
    if surface != 'All': x = x[x.surface == surface]
    return x.sort_values('tourney_date', ascending=False)


def elo_prob(d, p1, p2, surface='All'):
    x = d.dropna(subset=['winner_name','loser_name','tourney_date']).sort_values('tourney_date')
    ratings, sr = {}, {}
    for _, r in x.iterrows():
        a, b, surf = r.winner_name, r.loser_name, r.surface
        ratings.setdefault(a,1500); ratings.setdefault(b,1500)
        sr.setdefault((a,surf),1500); sr.setdefault((b,surf),1500)
        if surface == 'All' or surf == surface:
            ra, rb = ratings[a], ratings[b]
            ea = 1/(1+10**((rb-ra)/400)); k=24
            ratings[a] += k*(1-ea); ratings[b] += k*(0-(1-ea))
            rsa, rsb = sr[(a,surf)], sr[(b,surf)]
            es = 1/(1+10**((rsb-rsa)/400)); ks=28
            sr[(a,surf)] += ks*(1-es); sr[(b,surf)] += ks*(0-(1-es))
    base = 1/(1+10**((ratings.get(p2,1500)-ratings.get(p1,1500))/400))
    if surface != 'All':
        surfp = 1/(1+10**((sr.get((p2,surface),1500)-sr.get((p1,surface),1500))/400))
        return 0.35*base + 0.65*surfp
    return base


def model(d, p1, p2, surface):
    s1, s2 = summarize(d,p1,surface,20), summarize(d,p2,surface,20)
    r1, r2 = summarize(d,p1,'All',30), summarize(d,p2,'All',30)
    e = elo_prob(d,p1,p2,surface)
    f1 = s1['win_pct'] if pd.notna(s1['win_pct']) else r1['win_pct']
    f2 = s2['win_pct'] if pd.notna(s2['win_pct']) else r2['win_pct']
    wr = 0.5 if pd.isna(f1) or pd.isna(f2) else 0.5 + 0.5*(f1-f2)
    sp = 0.5 if pd.isna(s1['serve_pts_won']) or pd.isna(s2['serve_pts_won']) else 0.5 + 2.2*(s1['serve_pts_won']-s2['serve_pts_won'])
    ace = 0.5 if pd.isna(s1['aces']) or pd.isna(s2['aces']) else 0.5 + 0.035*(s1['aces']-s2['aces'])
    h = h2h(d,p1,p2,surface)
    hp = 0.5 if h.empty else (h.winner_name==p1).mean()
    n = min(s1['n'], s2['n']); conf = min(1,n/12)
    raw = 0.45*e + 0.30*wr + 0.15*sp + 0.05*ace + 0.05*hp
    p1prob = 0.5 + (raw-0.5)*(0.45+0.55*conf)
    return {'p1':p1prob,'p2':1-p1prob,'s1':s1,'s2':s2,'elo':e,'h2h':h,'confidence':conf}


def pct(x): return '—' if pd.isna(x) else f'{x*100:.1f}%'
def num(x): return '—' if pd.isna(x) else f'{x:.2f}'

st.title('🎾 Tennis Betting Lab V6')
st.caption('Free public ATP/WTA + Challenger/qualifying data • player search • matchup intelligence • Kalshi edge calculator')

with st.sidebar:
    st.header('Data')
    if st.button('Refresh local databases'):
        st.cache_data.clear(); st.rerun()
    st.success('No paid tennis API required.')
    st.write('Source: Local CSV database in this repository')
    st.caption('The app reads the ATP/WTA CSV files stored in the repo, so Streamlit Cloud does not need to reach an external tennis-data endpoint.')

# Load selected tour only.
tour = st.selectbox('Tour', ['ATP','WTA'])
try:
    with st.spinner(f'Loading {tour} player directory and 2021–2026 match database…'):
        players_df = load_players(tour)
        matches = clean_matches(load_all_matches(tour))
except Exception as e:
    st.error(f'Database load failed: {e}')
    st.info('Make sure the required CSV files are present in the repository under the data/ folder.')
    st.stop()

surface_opts = ['All'] + sorted([str(x) for x in matches.surface.dropna().unique() if str(x) not in ('Unknown','nan','')])
surface = st.selectbox('Surface', surface_opts)
st.write(f'**Player directory:** {len(players_df):,} names • **match records:** {len(matches):,} across 2021–2026')

q = st.text_input('Search any player', placeholder='Rybakina, Sabalenka, Alcaraz, Sinner, Tien…')
filtered = find_player_options(players_df, q)
if q and not filtered:
    st.warning('No player matched that search.')
    st.stop()

# If the search has many results, keep dropdown usable. If exact/surname result exists, show all matches.
filtered = filtered[:500]
c1,c2 = st.columns(2)
with c1:
    p1 = st.selectbox('Player 1', filtered, key='p1')
with c2:
    p2 = st.selectbox('Player 2', filtered, key='p2', index=min(1,len(filtered)-1))

if st.button('ANALYZE MATCHUP', type='primary', use_container_width=True):
    if p1 == p2:
        st.error('Choose two different players.')
    else:
        st.session_state['analysis'] = (p1,p2,surface,model(matches,p1,p2,surface))

if 'analysis' in st.session_state:
    p1,p2,surface,m = st.session_state['analysis']
    st.divider(); st.subheader(f'{p1} vs {p2}')
    a,b = st.columns(2)
    with a: st.metric(p1, f"{m['p1']*100:.1f}%", f"Fair price: {m['p1']*100:.0f}¢")
    with b: st.metric(p2, f"{m['p2']*100:.1f}%", f"Fair price: {m['p2']*100:.0f}¢")
    st.caption(f"Model confidence: {m['confidence']*100:.0f}% based on the smaller recent sample. Research model; not a guarantee.")

    s1,s2 = m['s1'],m['s2']
    st.subheader('Form & matchup')
    table = pd.DataFrame({
        'Metric':[f'Recent {surface} record (20)','Win %','Aces / match','Double faults / match','Serve points won','Hold %','Break %','All-surface recent win %'],
        p1:[f"{s1['wins']}-{s1['losses']}",pct(s1['win_pct']),num(s1['aces']),num(s1['df']),pct(s1['serve_pts_won']),pct(s1['hold_pct']),pct(s1['break_pct']),pct(summarize(matches,p1,'All',30)['win_pct'])],
        p2:[f"{s2['wins']}-{s2['losses']}",pct(s2['win_pct']),num(s2['aces']),num(s2['df']),pct(s2['serve_pts_won']),pct(s2['hold_pct']),pct(s2['break_pct']),pct(summarize(matches,p2,'All',30)['win_pct'])]
    })
    st.dataframe(table,use_container_width=True,hide_index=True)

    h=m['h2h']; st.subheader('Head-to-head')
    if h.empty: st.write('No H2H found in the loaded 2021–2026 database for this surface filter.')
    else:
        st.write(f"{p1}: {(h.winner_name==p1).sum()} wins • {p2}: {(h.winner_name==p2).sum()} wins")
        cols=[c for c in ['tourney_date','tourney_name','surface','tourney_level','round','winner_name','loser_name','score','w_ace','l_ace','w_df','l_df'] if c in h.columns]
        st.dataframe(h[cols].head(20),use_container_width=True,hide_index=True)

    st.subheader('Recent matches')
    recent = pd.concat([
        player_matches(matches,p1,surface).head(10).assign(player=p1),
        player_matches(matches,p2,surface).head(10).assign(player=p2)
    ]).sort_values('tourney_date',ascending=False)
    cols=[c for c in ['tourney_date','player','tourney_name','surface','tourney_level','round','winner_name','loser_name','score'] if c in recent.columns]
    st.dataframe(recent[cols],use_container_width=True,hide_index=True)

    st.subheader('Kalshi edge calculator')
    k1,k2=st.columns(2)
    with k1: market1=st.number_input(f'{p1} Kalshi YES price (¢)',1,99,int(round(m['p1']*100)),1)
    with k2: market2=st.number_input(f'{p2} Kalshi YES price (¢)',1,99,int(round(m['p2']*100)),1)
    e1=m['p1']-market1/100; e2=m['p2']-market2/100
    ec1,ec2=st.columns(2)
    with ec1: st.metric(f'{p1} edge',f'{e1*100:+.1f} pts',f'Model {m["p1"]*100:.1f}% vs market {market1}%')
    with ec2: st.metric(f'{p2} edge',f'{e2*100:+.1f} pts',f'Model {m["p2"]*100:.1f}% vs market {market2}%')
    best = p1 if e1 >= e2 else p2; bestedge=max(e1,e2)
    if bestedge >= 0.05: st.success(f'Largest model-vs-market edge: {best} at {bestedge*100:+.1f} percentage points.')
    elif bestedge >= 0.02: st.info(f'Modest edge: {best} at {bestedge*100:+.1f} percentage points. Check sample size, injuries and liquidity.')
    else: st.warning('No strong model edge at these prices.')

    st.subheader('Data provenance')
    st.write('Player and match data are loaded from Jeff Sackmann’s public ATP/WTA repositories. The repositories include season match files and player master files. The current app accesses those public files through GitHub’s public Contents API. Verify the source license before commercial use.')
    st.caption(f'Last calculation: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
