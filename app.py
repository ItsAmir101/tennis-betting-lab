
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tennis Betting Lab", page_icon="🎾", layout="centered")

BASE = "https://api.livetennisapi.com/api/public/v1"
KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"

st.markdown("""
<style>
.block-container {max-width: 900px; padding-top: 1.2rem;}
h1 {font-size: 2.1rem;}
.small {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)

st.title("🎾 Tennis Betting Lab")
st.caption("Paste a Kalshi tennis market → identify the players → pull current tennis data → estimate probability → compare to market price.")

with st.sidebar:
    st.header("Data connection")
    api_key = st.text_input(
        "Live Tennis API key",
        type="password",
        help="Get a key from Live Tennis API. The key is used only for this app session."
    )
    st.caption("The API covers ATP, WTA, Challenger and ITF. Free access provides current players, live matches and fixtures; historical match data requires a history-enabled plan.")
    st.divider()
    st.markdown("**Kalshi:** public market-data endpoints are used; no Kalshi trading credentials are required.")

def headers():
    return {"Authorization": f"Bearer {api_key.strip()}"} if api_key.strip() else {}

def api_get(path, params=None, timeout=20):
    if not api_key.strip():
        return None, "Enter your Live Tennis API key in the sidebar."
    try:
        r = requests.get(BASE + path, headers=headers(), params=params, timeout=timeout)
        if r.status_code == 401:
            return None, "Live Tennis API rejected the key (401). Check the key."
        if r.status_code == 403:
            return None, "This endpoint requires a higher Live Tennis API plan (403)."
        if r.status_code == 429:
            return None, "Live Tennis API rate limit reached (429)."
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"Live Tennis API request failed: {e}"

def kalshi_get(path, params=None, timeout=15):
    try:
        r = requests.get(KALSHI_BASE + path, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, f"Kalshi market request failed: {e}"

def parse_kalshi_url(url):
    m = re.search(r"kalshi\.com/[^?#\s]+/([^/?#\s]+)(?:[?#].*)?$", url.strip())
    if not m:
        return None
    candidate = m.group(1)
    # The actual ticker in Kalshi links is commonly the final uppercase segment.
    if re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{5,}", candidate):
        return candidate
    # Also support links where the ticker is the final path component after a slug.
    parts = [p for p in url.split("/") if p]
    for p in reversed(parts):
        p = p.split("?")[0].split("#")[0]
        if re.fullmatch(r"[A-Z0-9][A-Z0-9_-]{5,}", p):
            return p
    return None

def kalshi_market_from_url(url):
    ticker = parse_kalshi_url(url)
    if not ticker:
        return None, "I couldn't detect a Kalshi market ticker from that URL."
    data, err = kalshi_get(f"/markets/{ticker}")
    if err:
        return None, err
    market = data.get("market")
    if not market:
        return None, "Kalshi returned no market for that ticker."
    return market, None

def extract_names(text):
    # Supports common Kalshi titles such as "Vidmanova vs Tjen", "X vs Y",
    # and "Will X win vs Y?" without requiring exact formatting.
    clean = re.sub(r"\s+", " ", text or "").strip()
    clean = re.sub(r"(?i)\b(will|win|to win|match|women|men|wta|atp)\b", " ", clean)
    patterns = [
        r"(.+?)\s+(?:vs\.?|versus|-)\s+(.+)",
        r"(.+?)\s+(?:beats|defeats)\s+(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, clean, flags=re.I)
        if m:
            a, b = m.group(1).strip(" -:|"), m.group(2).strip(" -:|?.")
            if 2 <= len(a) <= 60 and 2 <= len(b) <= 60:
                return a, b
    return None

@st.cache_data(ttl=300)
def search_players_cached(query, key):
    # key is part of the cache signature so results are never shared between credentials.
    try:
        r = requests.get(BASE + "/players", headers={"Authorization": f"Bearer {key}"},
                         params={"search": query, "limit": 20}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def resolve_player(name):
    if not name:
        return []
    data = search_players_cached(name, api_key.strip())
    return (data or {}).get("data", [])

def pick_best_match(name, candidates):
    if not candidates:
        return None
    q = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
    def score(p):
        pn = re.sub(r"[^a-z0-9 ]", "", str(p.get("name","")).lower()).strip()
        s = 0
        if pn == q: s += 100
        if q and q in pn: s += 40
        qparts = set(q.split())
        pparts = set(pn.split())
        s += 20 * len(qparts & pparts)
        if p.get("ranking") is not None: s += 2
        return s
    return sorted(candidates, key=score, reverse=True)[0]

def price_to_probability(market):
    # Prefer executable side prices. Kalshi binary prices are dollars.
    yask = market.get("yes_ask_dollars")
    ybid = market.get("yes_bid_dollars")
    last = market.get("last_price_dollars")
    vals = []
    for v in (yask, ybid, last):
        try:
            vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return None
    return vals[0]

def profile(player_id):
    return api_get(f"/players/{int(player_id)}")

def render_player(p, label):
    st.subheader(label)
    cols = st.columns(4)
    cols[0].metric("Player", p.get("name","—"))
    cols[1].metric("Ranking", p.get("ranking") if p.get("ranking") is not None else "—")
    cols[2].metric("Elo", (p.get("stats") or {}).get("ratings",{}).get("overall") if isinstance((p.get("stats") or {}).get("ratings"), dict) else "—")
    cols[3].metric("Hand", p.get("hand") or "—")
    st.caption(f"Tour record: {p.get('tour') or '—'} • Country: {p.get('country') or '—'}")

url = st.text_input(
    "Kalshi tennis market URL",
    placeholder="Paste the full Kalshi match URL here"
)

if st.button("ANALYZE KALSHI MARKET", type="primary", use_container_width=True):
    if not api_key.strip():
        st.error("Enter your Live Tennis API key in the sidebar first.")
        st.stop()
    if not url.strip():
        st.error("Paste a Kalshi market URL first.")
        st.stop()

    market, err = kalshi_market_from_url(url)
    if err:
        st.error(err)
        st.stop()

    st.session_state["market"] = market

market = st.session_state.get("market")

if market:
    st.divider()
    st.subheader("Kalshi market")
    st.write(f"**{market.get('title') or market.get('subtitle') or 'Untitled market'}**")
    if market.get("subtitle"):
        st.caption(market["subtitle"])

    pmarket = price_to_probability(market)
    if pmarket is not None:
        st.metric("YES market price", f"{pmarket:.0%}")
    else:
        st.warning("No current YES price was returned by Kalshi.")

    names = extract_names((market.get("title") or "") + " " + (market.get("subtitle") or "") + " " + (market.get("yes_sub_title") or ""))
    if not names:
        st.warning("I found the Kalshi market, but couldn't reliably split the title into two player names. Use the manual names below.")
        c1, c2 = st.columns(2)
        manual1 = c1.text_input("Player 1 name")
        manual2 = c2.text_input("Player 2 name")
        names = (manual1, manual2) if manual1 and manual2 else None

    if names:
        n1, n2 = names
        st.write(f"**Detected:** {n1} vs {n2}")
        cand1 = resolve_player(n1)
        cand2 = resolve_player(n2)
        p1 = pick_best_match(n1, cand1)
        p2 = pick_best_match(n2, cand2)

        if not p1 or not p2:
            st.error("One or both players could not be resolved through the live player database.")
            st.write("Search candidates returned:", {"Player 1": cand1[:5], "Player 2": cand2[:5]})
            st.stop()

        d1, e1 = profile(p1["id"])
        d2, e2 = profile(p2["id"])
        if e1 or e2:
            st.error(e1 or e2)
            st.stop()

        pp1 = d1.get("player", d1)
        pp2 = d2.get("player", d2)

        render_player(pp1, "Player 1")
        render_player(pp2, "Player 2")

        r1 = pp1.get("ranking")
        r2 = pp2.get("ranking")
        elo1 = ((pp1.get("stats") or {}).get("ratings") or {}).get("overall")
        elo2 = ((pp2.get("stats") or {}).get("ratings") or {}).get("overall")

        # Transparent baseline. When both current Elo values exist, use them;
        # otherwise use ranking. This is intentionally a baseline, not a claim
        # of a proprietary model.
        if isinstance(elo1, (int,float)) and isinstance(elo2, (int,float)):
            model_p1 = 1 / (1 + 10 ** (-(elo1 - elo2) / 400))
            basis = "current Elo"
        elif isinstance(r1, int) and isinstance(r2, int):
            # Convert ranking ratio to a gentle probability signal.
            strength = np.log((r2 + 20) / (r1 + 20))
            model_p1 = 1 / (1 + np.exp(-1.15 * strength))
            basis = "current ranking"
        else:
            model_p1 = 0.5
            basis = "no comparable ranking/Elo"

        model_p1 = float(np.clip(model_p1, 0.02, 0.98))
        model_p2 = 1 - model_p1

        st.divider()
        st.subheader("Baseline probability")
        a, b = st.columns(2)
        a.metric(pp1.get("name","Player 1"), f"{model_p1:.1%}")
        b.metric(pp2.get("name","Player 2"), f"{model_p2:.1%}")
        st.caption(f"Baseline uses {basis}. It is not a guarantee and should be combined with surface/form/history when those data are available.")

        if pmarket is not None:
            st.subheader("Kalshi edge")
            # Assume YES corresponds to player 1 only when the market subtitle/title
            # explicitly names player 1; otherwise label the raw YES edge.
            edge = model_p1 - pmarket
            st.metric("Model probability − YES price", f"{edge:+.1%}")
            if edge > 0.05:
                st.success("Positive baseline edge on YES, subject to market mapping and data quality.")
            elif edge < -0.05:
                st.warning("Baseline model is below the YES price.")
            else:
                st.info("No meaningful baseline edge from the current inputs.")

        st.divider()
        st.subheader("Next-level data")
        st.write(
            "With a history-enabled Live Tennis API plan, this page can add completed-match form, "
            "surface-specific results, H2H, point-by-point history and deeper backtesting. "
            "The API's coverage spans ATP, WTA, Challenger and ITF."
        )
else:
    st.info("Paste a Kalshi match URL above. The app will try to identify the market and both players automatically.")
    st.markdown("""
**Workflow**

1. Paste the Kalshi URL.
2. The app extracts the market ticker.
3. It pulls the live Kalshi market data.
4. It identifies both players.
5. It searches the live tennis player database.
6. It pulls current ranking/Elo/profile data.
7. It calculates a transparent baseline probability.
8. It compares that probability with the Kalshi price.

This version is designed to expand into the full ATP/WTA/Challenger/ITF analysis engine.
""")
