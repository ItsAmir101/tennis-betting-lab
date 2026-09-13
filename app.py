import io,re,unicodedata
from urllib.parse import quote,urlparse
import requests,pandas as pd,numpy as np,streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title='Tennis Betting Lab V9',page_icon='🎾',layout='wide')
KAPI='https://external-api.kalshi.com/trade-api/v2'
TA='https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p={}'
H={'User-Agent':'Mozilla/5.0 (compatible; TennisBettingLab/9.0)'}

def clean(x): return re.sub(r'\s+',' ',str(x or '')).strip(' -|:')
def norm(x):
 x=unicodedata.normalize('NFKD',clean(x)); return re.sub(r'\s+',' ',re.sub(r'[^A-Za-z0-9 ]',' ',''.join(c for c in x if not unicodedata.combining(c)))).lower().strip()
def slugs(name):
 n=unicodedata.normalize('NFKD',clean(name)); n=''.join(c for c in n if not unicodedata.combining(c)); c=re.sub(r'[^A-Za-z0-9]','',n)
 return list(dict.fromkeys([c,re.sub(r'[^A-Za-z0-9]','',n.replace("'",''))]))
def num(x):
 try:
  m=re.search(r'-?\d+(?:\.\d+)?',str(x).replace(',','').replace('%','')); return float(m.group()) if m else np.nan
 except: return np.nan

def ticker_from_url(url):
 for x in re.findall(r'[A-Za-z0-9_-]{6,}',url):
  if x.upper().startswith('KX'): return x
 parts=[x for x in urlparse(url).path.split('/') if x]
 return parts[-1] if parts and parts[-1].lower() not in ('markets','events') else None

@st.cache_data(ttl=60,show_spinner=False)
def kalshi_api(ticker):
 r=requests.get(f'{KAPI}/markets/{quote(ticker,safe="")}',headers=H,timeout=20)
 if r.status_code!=200: raise ValueError(f'Kalshi API returned HTTP {r.status_code}.')
 m=r.json().get('market',{})
 if not m: raise ValueError('Kalshi returned no market data.')
 return m

def names(text):
 text=clean(text)
 p=re.split(r'\s+(?:vs?\.?|v\.?)\s+|\s+@\s+',text,flags=re.I)
 if len(p)==2:return clean(p[0]),clean(p[1])
 m=re.search(r'(?:will\s+)?(.+?)\s+(?:beat|defeat|win\s+against)\s+(.+?)(?:\?|$)',text,re.I)
 return (clean(m.group(1)),clean(m.group(2))) if m else (None,None)

def kalshi_input(raw):
 raw=clean(raw)
 if raw.startswith(('http://','https://')):
  t=ticker_from_url(raw)
  if t:
   try:
    m=kalshi_api(t); a,b=names(' '.join([m.get('title',''),m.get('yes_sub_title',''),m.get('subtitle','')]))
    if a and b:return t,m,a,b,'Kalshi API'
   except: pass
  r=requests.get(raw,headers=H,timeout=20)
  soup=BeautifulSoup(r.text,'html.parser'); title=soup.title.get_text(' ',strip=True) if soup.title else ''
  a,b=names(title)
  if not a:
   a,b=names(soup.get_text(' ',strip=True)[:6000])
  if a and b:return t,{},a,b,'Kalshi webpage fallback'
  raise ValueError('Could not identify both player names from the Kalshi link.')
 if re.fullmatch(r'[A-Za-z0-9_-]{6,}',raw):
  m=kalshi_api(raw); a,b=names(' '.join([m.get('title',''),m.get('yes_sub_title',''),m.get('subtitle','')]))
  if a and b:return raw,m,a,b,'Kalshi API'
 raise ValueError('Not a recognizable Kalshi URL or ticker.')

@st.cache_data(ttl=21600,show_spinner=False)
def ta(name):
 urls=[name] if str(name).startswith(('http://','https://')) else [TA.format(quote(s)) for s in slugs(name)]
 last=''
 for u in urls:
  try:
   r=requests.get(u,headers=H,timeout=25); last=f'HTTP {r.status_code}'
   if r.status_code!=200 or 'Tennis Abstract' not in r.text: continue
   soup=BeautifulSoup(r.text,'html.parser'); tables=[]
   try: tables=pd.read_html(io.StringIO(r.text))
   except: pass
   return r.url,r.text,soup.get_text(' ',strip=True),tables
  except Exception as e:last=str(e)
 raise ValueError(f"Tennis Abstract could not find '{name}' ({last}).")

def profile(data):
 url,html,text,tables=data; o={k:np.nan for k in ['rank','elo','elo_rank','win52','spw','rpw','tpw']}; o['name']=None
 title=BeautifulSoup(html,'html.parser').title
 if title:
  s=title.get_text(' ',strip=True); m=re.search(r'Tennis Abstract:\s*(.*?)\s+(?:Match Results|Results)',s,re.I)
  if m:o['name']=clean(m.group(1))
 for pat,k in [(r'Current rank:\s*(\d+)','rank'),(r'Elo rank:\s*(\d+)','elo_rank'),(r'Elo rank:\s*\d+\s*\(rating:\s*([0-9.]+)','elo')]:
  m=re.search(pat,text,re.I)
  if m:o[k]=num(m.group(1))
 m=re.search(r'Last 52\s+\d+-\d+\s+\((\d+)%\)',text,re.I)
 if m:o['win52']=num(m.group(1))
 for pat,k in [(r'Service points won\s*[: ]\s*(\d+(?:\.\d+)?)%', 'spw'),(r'Return points won\s*[: ]\s*(\d+(?:\.\d+)?)%','rpw'),(r'Total points won\s*[: ]\s*(\d+(?:\.\d+)?)%','tpw')]:
  m=re.search(pat,text,re.I)
  if m:o[k]=num(m.group(1))
 return o

def match_df(data):
 tables=data[3]; best=pd.DataFrame(); score=-1
 for d in tables:
  c=' '.join(map(lambda x:str(x).lower(),d.columns)); s=sum(x in c for x in ['date','opponent','result','surface','score'])
  if s>score and s>=2:best=d.copy();score=s
 if best.empty:return best
 best.columns=[str(c).strip() for c in best.columns]; low={str(c).lower():c for c in best.columns}
 def pick(keys):
  for k,c in low.items():
   if any(x in k for x in keys):return c
  return None
 dc,rc,sc,oc=pick(['date']),pick(['result','w/l']),pick(['surface']),pick(['opponent','opp'])
 best['_date']=pd.to_datetime(best[dc],errors='coerce') if dc else pd.NaT
 best['_win']=best[rc].astype(str).str.upper().str.startswith('W') if rc else np.nan
 best['_surface']=best[sc].astype(str) if sc else ''
 best['_opp']=best[oc].astype(str) if oc else ''
 return best

def recent(d,n=10,surf=None):
 if d.empty:return None
 x=d.dropna(subset=['_date']).sort_values('_date',ascending=False)
 if surf:x=x[x['_surface'].str.lower()==surf.lower()]
 x=x.head(n)
 if x.empty:return None
 w=int(x['_win'].fillna(False).sum()); return {'n':len(x),'w':w,'l':len(x)-w,'pct':100*w/len(x)}

def prob(a,b,r1,r2):
 ea,eb=a['elo'],b['elo']; ea=ea if np.isfinite(ea) else 1500; eb=eb if np.isfinite(eb) else 1500
 vals=[1/(1+10**((eb-ea)/400))]; ws=[.68]
 for k,weight in [('win52',.14),('spw',.08),('rpw',.06)]:
  x,y=a[k],b[k]
  if np.isfinite(x) and np.isfinite(y): vals.append((x/100)/((x/100)+(y/100)) if k=='win52' else .5+(x-y)/100); ws.append(weight)
 if r1 and r2 and r1['n']>=3 and r2['n']>=3: vals.append((r1['pct']/100)/((r1['pct']/100)+(r2['pct']/100)));ws.append(.04)
 return min(.95,max(.05,float(np.average(vals,weights=ws))))

st.title('🎾 Tennis Betting Lab V9')
st.caption('Paste a Kalshi tennis link → identify players → pull Tennis Abstract stats → estimate fair probability and edge.')
with st.sidebar:
 surface=st.selectbox('Surface',['Hard','Clay','Grass','Carpet'])
 st.markdown('**Backup:** If the link cannot be parsed, enter both player names below.')
link=st.text_input('Kalshi market URL or ticker',placeholder='Paste Kalshi link here')
c1,c2=st.columns(2)
with c1: p1=st.text_input('Backup Player 1',placeholder='Martin Borisiouk')
with c2: p2=st.text_input('Backup Player 2',placeholder='Dong Ju Kim')
a,b=st.columns(2)
with a: go=st.button('Analyze Kalshi link',type='primary',use_container_width=True)
with b: fallback=st.button('Use backup player names',use_container_width=True)
if go:
 try:
  t,m,x,y,src=kalshi_input(link);st.session_state['m']=(t,m,x,y,src);st.success(f'Identified: {x} vs {y} • {src}')
 except Exception as e:st.error(str(e));st.info('Use the backup player-name fields. That bypasses Kalshi parsing.')
if fallback:
 if p1.strip() and p2.strip():st.session_state['m']=(None,{},p1.strip(),p2.strip(),'Manual fallback')
 else:st.error('Enter both player names.')
if 'm' in st.session_state:
 t,m,x,y,src=st.session_state['m']
 try:
  with st.spinner('Loading Tennis Abstract...'):
   d1,d2=ta(x),ta(y); a1,a2=profile(d1),profile(d2); md1,md2=match_df(d1),match_df(d2)
   r1,r2=recent(md1,10),recent(md2,10); s1,s2=recent(md1,10,surface),recent(md2,10,surface); pr=prob(a1,a2,s1 or r1,s2 or r2)
 except Exception as e:st.error(str(e));st.stop()
 n1=a1['name'] or x;n2=a2['name'] or y;market=m or {}
 ky=num(market.get('yes_ask_dollars'))*100 if market.get('yes_ask_dollars') is not None else np.nan
 if not np.isfinite(ky) and market.get('last_price_dollars') is not None:ky=num(market['last_price_dollars'])*100
 st.divider();st.subheader(f'{n1} vs {n2}');st.caption(f'Identification: {src}' + (f' • Kalshi ticker: {t}' if t else ''))
 q1,q2,q3=st.columns(3);q1.metric(n1,f'{pr*100:.1f}%');q2.metric(n2,f'{(1-pr)*100:.1f}%');q3.metric('Kalshi YES',f'{ky:.1f}¢' if np.isfinite(ky) else 'Not loaded')
 rows=[('Current rank',a1['rank'],a2['rank']),('Tennis Abstract Elo',a1['elo'],a2['elo']),('Elo rank',a1['elo_rank'],a2['elo_rank']),('Last-52 win %',a1['win52'],a2['win52']),('Service points won %',a1['spw'],a2['spw']),('Return points won %',a1['rpw'],a2['rpw']),('Total points won %',a1['tpw'],a2['tpw'])]
 st.markdown('### Player statistics');st.dataframe(pd.DataFrame(rows,columns=['Metric',n1,n2]),use_container_width=True,hide_index=True)
 st.markdown('### Recent form');q1,q2=st.columns(2)
 for col,n,r,s in [(q1,n1,r1,s1),(q2,n2,r2,s2)]:
  with col:
   st.write(f'**{n}**');st.write(f"Last 10: {r['w']}-{r['l']} ({r['pct']:.0f}%)" if r else 'Last 10 unavailable');st.write(f"{surface}: {s['w']}-{s['l']} ({s['pct']:.0f}%)" if s else f'{surface}: insufficient data')
 if np.isfinite(ky):
  e1=pr*100-ky;e2=(1-pr)*100-(100-ky);st.markdown('### Kalshi edge');st.dataframe(pd.DataFrame([[n1,f'{pr*100:.1f}%',f'{ky:.1f}¢',f'{e1:+.1f}¢'],[n2,f'{(1-pr)*100:.1f}%',f'{100-ky:.1f}¢',f'{e2:+.1f}¢']],columns=['Player','Model probability','Market YES','Edge']),use_container_width=True,hide_index=True)
  st.success(f'Model edge lean: {n1} ({e1:+.1f}¢)' if e1>e2 else f'Model edge lean: {n2} ({e2:+.1f}¢)')
 st.markdown('### Tennis Abstract source pages');st.write(d1[0]);st.write(d2[0])
 st.warning('Research estimate only. Verify contract wording, matchup, surface, start time, injuries/withdrawals, and live price before trading.')
