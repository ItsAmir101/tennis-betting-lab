import io, re, math, time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title='Tennis Betting Lab V5', page_icon='🎾', layout='wide')

# ----------------------------
# Public/free data sources
# ----------------------------
ATP = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/'
WTA = 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/'
# jsDelivr is a useful fallback when raw.githubusercontent.com is unavailable.
ATP_CDN = 'https://cdn.jsdelivr.net/gh/JeffSackmann/tennis_atp@master/'
WTA_CDN = 'https://cdn.jsdelivr.net/gh/JeffSackmann/tennis_wta@master/'

YEARS = [2026, 2025, 2024, 2023, 2022, 2021]

HEADERS = {'User-Agent': 'Tennis-Betting-Lab/5.0'}


def _get_csv(url, timeout=30):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), low_memory=False)


def get_csv(filename, tour):
    bases = [ATP, ATP_CDN] if tour == 'ATP' else [WTA, WTA_CDN]
    last = None
    for base in bases:
        try:
            return _get_csv(base + filename)
        except Exception as e:
            last = e
    raise RuntimeError(f'Unable to download {filename}: {last}')


@st.cache_data(ttl=86400, show_spinner=False)
def load_year(tour, year):
    if tour == 'ATP':
        files = [f'atp_matches_{year}.csv', f'atp_matches_qual_chall_{year}.csv']
    else:
        files = [f'wta_matches_{year}.csv', f'wta_matches_qual_itf_{year}.csv']
    frames = []
    errors = []
    for f in files:
        try:
            d = get_csv(f, tour)
            d['_source_file'] = f
            frames.append(d)
        except Exception as e:
            errors.append(str(e))
    if not frames:
        raise RuntimeError('; '.join(errors))
    return pd.concat(frames, ignore_index=True, sort=False)


@st.cache_data(ttl=86400, show_spinner=False)
def load_players(tour):
    fn = 'atp_players.csv' if tour == 'ATP' else 'wta_players.csv'
    try:
        return get_csv(fn, tour)
    except Exception:
        # Build a current directory from recent match files if player master file is unavailable.
        d = load_year(tour, 2026)
        names = pd.concat([d[['winner_name']].rename(columns={'winner_name':'name'}),
                           d[['loser_name']].rename(columns={'loser_name':'name'})], ignore_index=True)
        names = names.dropna().drop_duplicates().sort_values('name')
        return names


@st.cache_data(ttl=86400, show_spinner=False)
def load_all_recent(tour):
    frames=[]
    for y in YEARS:
        try:
            frames.append(load_year(tour, y))
        except Exception:
            continue
    if not frames:
        raise RuntimeError('No public match files could be downloaded.')
    return pd.concat(frames, ignore_index=True, sort=False)


def clean_matches(df):
    d=df.copy()
    d['tourney_date']=pd.to_datetime(d['tourney_date'].astype(str), format='%Y%m%d', errors='coerce')
    for c in ['winner_rank','loser_rank','w_ace','l_ace','w_df','l_df','w_svpt','l_svpt','w_1stIn','l_1stIn','w_1stWon','l_1stWon','w_2ndWon','l_2ndWon','w_bpSaved','l_bpSaved','w_bpFaced','l_bpFaced','w_SvGms','l_SvGms','w_bpWon','l_bpWon','w_bpFaced','l_bpFaced']:
        if c in d.columns: d[c]=pd.to_numeric(d[c], errors='coerce')
    d['surface']=d.get('surface', pd.Series(index=d.index, dtype='object')).fillna('Unknown')
    d['level']=d.get('tourney_level', pd.Series(index=d.index, dtype='object')).fillna('')
    d['tourney_name']=d.get('tourney_name', pd.Series(index=d.index, dtype='object')).fillna('')
    return d.sort_values('tourney_date', ascending=False)


def player_directory(tour, matches):
    try:
        p=load_players(tour)
        if {'first_name','last_name'}.issubset(p.columns):
            p['display']=p['first_name'].fillna('')+' '+p['last_name'].fillna('')
        elif 'name' in p.columns:
            p['display']=p['name']
        else:
            p['display']=p.iloc[:,0].astype(str)
        names=p['display'].dropna().astype(str).str.replace(r'\s+',' ',regex=True).str.strip().drop_duplicates().tolist()
    except Exception:
        names=[]
    for c in ['winner_name','loser_name']:
        if c in matches.columns: names += matches[c].dropna().astype(str).tolist()
    return sorted(set(n for n in names if n), key=str.lower)


def find_name(name, matches):
    # Normalize punctuation/case and return the best exact/contains match.
    names=sorted(set(pd.concat([matches.get('winner_name',pd.Series(dtype=str)),matches.get('loser_name',pd.Series(dtype=str))]).dropna().astype(str)))
    target=re.sub(r'[^a-z0-9 ]','',name.lower()).strip()
    exact=[n for n in names if re.sub(r'[^a-z0-9 ]','',n.lower()).strip()==target]
    if exact: return exact[0]
    contains=[n for n in names if target in re.sub(r'[^a-z0-9 ]','',n.lower())]
    if len(contains)==1: return contains[0]
    return name


def p_matches(d, player, surface='All'):
    x=d[(d['winner_name'].eq(player)) | (d['loser_name'].eq(player))].copy()
    if surface!='All': x=x[x['surface'].eq(surface)]
    return x.sort_values('tourney_date', ascending=False)


def result_for(row, player):
    return 'W' if row['winner_name']==player else 'L'


def summarize(d, player, surface='All', n=20):
    x=p_matches(d,player,surface).head(n)
    if x.empty: return {'n':0,'wins':0,'losses':0,'win_pct':np.nan,'aces':np.nan,'df':np.nan,'serve_pts_won':np.nan,'hold_pct':np.nan,'break_pct':np.nan}
    wins=(x.winner_name==player).sum(); losses=len(x)-wins
    ac=[]; dfs=[]; spw=[]; holds=[]; breaks=[]
    for _,r in x.iterrows():
        w=r.winner_name==player
        if 'w_ace' in r and pd.notna(r.w_ace): ac.append(r.w_ace if w else r.get('l_ace',np.nan))
        if 'w_df' in r and pd.notna(r.w_df): dfs.append(r.w_df if w else r.get('l_df',np.nan))
        # Serve points won percentage when components exist
        if w:
            svpt=r.get('w_svpt',np.nan); first=r.get('w_1stIn',np.nan); fw=r.get('w_1stWon',np.nan); sw=r.get('w_2ndWon',np.nan)
            sg=r.get('w_SvGms',np.nan); bpw=r.get('w_bpWon',np.nan)
        else:
            svpt=r.get('l_svpt',np.nan); first=r.get('l_1stIn',np.nan); fw=r.get('l_1stWon',np.nan); sw=r.get('l_2ndWon',np.nan)
            sg=r.get('l_SvGms',np.nan); bpw=r.get('l_bpWon',np.nan)
        if pd.notna(svpt) and pd.notna(first) and pd.notna(fw) and pd.notna(sw) and svpt>0:
            second=max(svpt-first,0); pts=fw+sw; spw.append(pts/svpt)
        if pd.notna(sg) and sg>0:
            # Breaks are not directly available as games held; infer from opponent break won when possible.
            opp='l_' if w else 'w_'
            opp_bpw=r.get(opp+'bpWon',np.nan)
            opp_bpf=r.get(opp+'bpFaced',np.nan)
            if pd.notna(opp_bpw) and pd.notna(opp_bpf): holds.append(1-(opp_bpw/max(opp_bpf,1)))
        if pd.notna(bpw):
            opp='l_' if w else 'w_'; opp_bpf=r.get(opp+'bpFaced',np.nan)
            if pd.notna(opp_bpf) and opp_bpf>0: breaks.append(bpw/opp_bpf)
    return {'n':len(x),'wins':int(wins),'losses':int(losses),'win_pct':wins/len(x),'aces':np.nanmean(ac) if ac else np.nan,'df':np.nanmean(dfs) if dfs else np.nan,'serve_pts_won':np.nanmean(spw) if spw else np.nan,'hold_pct':np.nanmean(holds) if holds else np.nan,'break_pct':np.nanmean(breaks) if breaks else np.nan}


def h2h(d,p1,p2,surface='All'):
    x=d[((d.winner_name==p1)&(d.loser_name==p2))|((d.winner_name==p2)&(d.loser_name==p1))].copy()
    if surface!='All': x=x[x.surface==surface]
    return x.sort_values('tourney_date',ascending=False)


def elo_prob(d,p1,p2,surface='All'):
    # Rebuild a simple rolling Elo from 2021 onward. Surface-specific model when selected.
    x=d.dropna(subset=['winner_name','loser_name','tourney_date']).sort_values('tourney_date')
    ratings={}
    sr={}
    for _,r in x.iterrows():
        a,b=r.winner_name,r.loser_name
        surf=r.surface
        for key in [a,b]: ratings.setdefault(key,1500); sr.setdefault((key,surf),1500)
        if surface=='All' or surf==surface:
            ra,rb=ratings[a],ratings[b]; ea=1/(1+10**((rb-ra)/400)); k=24
            ratings[a]+=k*(1-ea); ratings[b]+=k*(0-(1-ea))
            rsa,rsb=sr[(a,surf)],sr[(b,surf)]; es=1/(1+10**((rsb-rsa)/400)); ks=28
            sr[(a,surf)]+=ks*(1-es); sr[(b,surf)]+=ks*(0-(1-es))
    ra=ratings.get(p1,1500); rb=ratings.get(p2,1500)
    base=1/(1+10**((rb-ra)/400))
    if surface!='All':
        rsa=sr.get((p1,surface),1500); rsb=sr.get((p2,surface),1500)
        surfp=1/(1+10**((rsb-rsa)/400))
        return 0.35*base+0.65*surfp
    return base


def model(d,p1,p2,surface):
    s1=summarize(d,p1,surface,20); s2=summarize(d,p2,surface,20)
    r1=summarize(d,p1,'All',30); r2=summarize(d,p2,'All',30)
    e=elo_prob(d,p1,p2,surface)
    f1=s1['win_pct'] if pd.notna(s1['win_pct']) else r1['win_pct']
    f2=s2['win_pct'] if pd.notna(s2['win_pct']) else r2['win_pct']
    wr=0.5 if pd.isna(f1) or pd.isna(f2) else 0.5 + 0.5*(f1-f2)
    sp=0.5
    if pd.notna(s1['serve_pts_won']) and pd.notna(s2['serve_pts_won']): sp=0.5+2.2*(s1['serve_pts_won']-s2['serve_pts_won'])
    ace=0.5
    if pd.notna(s1['aces']) and pd.notna(s2['aces']): ace=0.5+0.035*(s1['aces']-s2['aces'])
    h=h2h(d,p1,p2,surface)
    hp=0.5 if h.empty else (h.winner_name==p1).mean()
    n=min(s1['n'],s2['n'])
    conf=min(1,n/12)
    raw=0.45*e+0.30*wr+0.15*sp+0.05*ace+0.05*hp
    # Pull toward 50 when samples are weak.
    p1prob=0.5+(raw-0.5)*(0.45+0.55*conf)
    return {'p1':p1prob,'p2':1-p1prob,'s1':s1,'s2':s2,'elo':e,'h2h':h,'confidence':conf}


def pct(x): return '—' if pd.isna(x) else f'{x*100:.1f}%'
def num(x): return '—' if pd.isna(x) else f'{x:.2f}'

# ---------------------------- UI ----------------------------
st.title('🎾 Tennis Betting Lab V5')
st.caption('Free public ATP/WTA + Challenger/qualifying data • matchup model • Kalshi edge calculator')

with st.sidebar:
    st.header('Data')
    if st.button('Refresh public databases'):
        st.cache_data.clear(); st.rerun()
    st.write('Primary: Jeff Sackmann ATP/WTA')
    st.write('Fallback: jsDelivr mirror of the same public repositories')
    st.caption('No paid tennis API is required.')
    st.divider()
    st.caption('Data sources are public and licensed for non-commercial use with attribution. Verify the source license before commercializing the app.')

# Load both tours lazily; this makes first load faster.
tour=st.selectbox('Tour', ['ATP','WTA'])
try:
    with st.spinner('Loading free public tennis database…'):
        matches=clean_matches(load_all_recent(tour))
        players=player_directory(tour,matches)
except Exception as e:
    st.error(f'Database load failed: {e}')
    st.info('If Streamlit Cloud cannot reach both GitHub endpoints, use the Refresh button. The app has a second public CDN fallback.')
    st.stop()

surface_opts=['All']+sorted([str(x) for x in matches.surface.dropna().unique() if str(x) not in ['Unknown','nan']])
surface=st.selectbox('Surface',surface_opts)
st.write(f'**Player directory:** {len(players):,} names • **match records loaded:** {len(matches):,} across 2021–2026')

q=st.text_input('Search any player', placeholder='Try Rybakina, Sabalenka, Alcaraz, Sinner, Tien…')
filtered=[n for n in players if q.lower() in n.lower()] if q else players
if q and not filtered:
    st.warning('No player matched that search in the currently loaded public database.')
    st.stop()

c1,c2=st.columns(2)
with c1:
    p1=st.selectbox('Player 1',filtered,key='p1')
with c2:
    p2=st.selectbox('Player 2',filtered,key='p2',index=min(1,len(filtered)-1))

if st.button('ANALYZE MATCHUP',type='primary',use_container_width=True):
    if p1==p2:
        st.error('Choose two different players.')
        st.stop()
    m=model(matches,p1,p2,surface)
    st.session_state['analysis']=(p1,p2,surface,m)

if 'analysis' in st.session_state:
    p1,p2,surface,m=st.session_state['analysis']
    st.divider()
    st.subheader(f'{p1} vs {p2}')
    a,b=st.columns(2)
    with a:
        st.metric(p1,f"{m['p1']*100:.1f}%",f"Fair price: {m['p1']*100:.0f}¢")
    with b:
        st.metric(p2,f"{m['p2']*100:.1f}%",f"Fair price: {m['p2']*100:.0f}¢")
    st.caption(f'Model confidence: {m["confidence"]*100:.0f}% based primarily on the smaller recent sample. This is a research model, not a guarantee.')

    s1,s2=m['s1'],m['s2']
    st.subheader('Form & matchup')
    table=pd.DataFrame({
        'Metric':[f'Recent {surface} record (20)','Win %','Aces / match','Double faults / match','Serve points won','All-surface recent win %'],
        p1:[f"{s1['wins']}-{s1['losses']}",pct(s1['win_pct']),num(s1['aces']),num(s1['df']),pct(s1['serve_pts_won']),pct(summarize(matches,p1,'All',30)['win_pct'])],
        p2:[f"{s2['wins']}-{s2['losses']}",pct(s2['win_pct']),num(s2['aces']),num(s2['df']),pct(s2['serve_pts_won']),pct(summarize(matches,p2,'All',30)['win_pct'])]
    })
    st.dataframe(table,use_container_width=True,hide_index=True)

    h=m['h2h']
    st.subheader('Head-to-head')
    if h.empty: st.write('No H2H found in the loaded 2021–2026 database for this surface filter.')
    else:
        st.write(f'{p1}: {(h.winner_name==p1).sum()} wins • {p2}: {(h.winner_name==p2).sum()} wins')
        cols=[c for c in ['tourney_date','tourney_name','surface','round','winner_name','loser_name','score','w_ace','l_ace','w_df','l_df'] if c in h.columns]
        st.dataframe(h[cols].head(20),use_container_width=True,hide_index=True)

    st.subheader('Recent matches')
    recent=pd.concat([p_matches(matches,p1,surface).head(10).assign(player=p1),p_matches(matches,p2,surface).head(10).assign(player=p2)]).sort_values('tourney_date',ascending=False)
    cols=[c for c in ['tourney_date','player','tourney_name','surface','round','winner_name','loser_name','score'] if c in recent.columns]
    st.dataframe(recent[cols],use_container_width=True,hide_index=True)

    st.subheader('Kalshi edge calculator')
    k1,k2=st.columns(2)
    with k1: market1=st.number_input(f'{p1} Kalshi YES price (¢)',min_value=1,max_value=99,value=int(round(m['p1']*100)),step=1)
    with k2: market2=st.number_input(f'{p2} Kalshi YES price (¢)',min_value=1,max_value=99,value=int(round(m['p2']*100)),step=1)
    # Prices can be entered independently; edge is model probability minus market price.
    e1=m['p1']-market1/100; e2=m['p2']-market2/100
    ec1,ec2=st.columns(2)
    with ec1: st.metric(f'{p1} edge',f'{e1*100:+.1f} pts',f'Model {m["p1"]*100:.1f}% vs market {market1}%')
    with ec2: st.metric(f'{p2} edge',f'{e2*100:+.1f} pts',f'Model {m["p2"]*100:.1f}% vs market {market2}%')
    best=p1 if e1>e2 else p2
    bestedge=max(e1,e2)
    if bestedge>=0.05: st.success(f'Largest model-vs-market edge: {best} at {bestedge*100:+.1f} percentage points.')
    elif bestedge>=0.02: st.info(f'Modest edge: {best} at {bestedge*100:+.1f} percentage points. Consider sample size and market liquidity.')
    else: st.warning('No strong model edge at these prices.')

    st.subheader('Data provenance')
    st.write('ATP/WTA results, rankings and match statistics are sourced from Jeff Sackmann’s public tennis repositories. The ATP repository documents tour-level matches plus qualifying/Challenger and Futures files; the WTA repository documents tour-level and qualifying/ITF files. cite links are shown in the project documentation.')
    st.caption(f'Last app calculation: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
