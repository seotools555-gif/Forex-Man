
# app_forex_pro_click_tools.py — Forex Pro (IST) with per‑pair click/expand tools
# - Per‑pair UI: toggles for Candle Patterns, MATLAB tools (Bands/Stoch/ADX), News Sentiment
# - Clear "Trade / Don't Trade now" message per pair
# - Keeps real-time IST chart, ASI arrows, RSI/MACD, position sizing, margin

import os, time, math, re
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests, feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Optional SciPy for smoothing (MATLAB-like)
try:
    from scipy.signal import savgol_filter
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------- Config ----------
DEFAULT_NEWSAPI_KEY = "1502cba32d134f4095aa03d4bd5bfe3c"
def cfg(name, default=""):
    val = os.getenv(name)
    if val not in (None, "", "None"): return val
    try:
        if name in st.secrets:
            sv = st.secrets.get(name)
            if sv not in (None, "", "None"): return str(sv)
    except Exception:
        pass
    return default

NEWSAPI_KEY = cfg("NEWSAPI_KEY", DEFAULT_NEWSAPI_KEY)
ACCOUNT_BALANCE_DEFAULT = float(cfg("ACCOUNT_BALANCE", "100000"))
IST = pytz.timezone("Asia/Kolkata")

# Ensure VADER
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ---------- Theme ----------
st.set_page_config(page_title="Forex Pro — Click Tools", layout="wide")
st.markdown("""
<style>
.card{background:linear-gradient(180deg,#0d1b2a,#0b2338);padding:14px;border-radius:14px;border:1px solid rgba(255,255,255,.05);box-shadow:0 6px 18px rgba(0,0,0,.45)}
.small{color:#9fb0c8;font-size:13px}
.good{color:#10b981} .bad{color:#ef4444} .warn{color:#f59e0b}
</style>
""", unsafe_allow_html=True)
st.title("💹 Forex Pro — India (Click Tools)")
st.caption("Toggle MATLAB tools, candles, and news per pair. Educational only — not financial advice.")

# ---------- Pairs & Sidebar ----------
MAJOR = ["EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X","NZDUSD=X","USDCAD=X"]
CROSS  = ["EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X","CHFJPY=X","NZDJPY=X","EURAUD=X","GBPAUD=X"]
EXOTIC = ["USDINR=X","USDSGD=X","USDHKD=X","USDTRY=X"]
ALL_PAIRS = MAJOR + CROSS + EXOTIC

st.sidebar.header("⚙️ Settings")
pairs = st.sidebar.multiselect("Pairs to analyze", ALL_PAIRS, MAJOR)
rt_pair = st.sidebar.selectbox("Real‑Time Pair", ALL_PAIRS, index=0)
account_balance = st.sidebar.number_input("Account balance (USD)", value=ACCOUNT_BALANCE_DEFAULT, step=1000.0)
leverage = st.sidebar.number_input("Leverage", min_value=1, max_value=500, value=30, step=1)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.1, 2.0, 0.5, 0.1)
refresh = st.sidebar.slider("Auto-refresh (seconds)", 30, 1800, 120)
if st.sidebar.button("Force refresh now"): st.cache_data.clear()

# ---------- Data helpers ----------
@st.cache_data(ttl=60, show_spinner=False)
def download_history(symbol: str, period="45d", interval="1h"):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, actions=False, auto_adjust=False)
        if df is None or df.empty: return None
        for c in ("Open","High","Low","Close"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open","High","Low","Close"])
        return df if not df.empty else None
    except Exception:
        return None

@st.cache_data(ttl=20, show_spinner=False)
def download_intraday(symbol: str, period="1d", interval="5m"):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval, actions=False, auto_adjust=False)
        if df is None or df.empty: return None
        for c in ("Open","High","Low","Close"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Open","High","Low","Close"])
        return df if not df.empty else None
    except Exception:
        return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # RSI
    d = df["Close"].diff(); up = d.clip(lower=0); down = -d.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / (down.ewm(alpha=1/14, adjust=False).mean().replace(0,1e-10))
    df["RSI"] = 100 - (100/(1+rs))
    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    # ATR
    tr = pd.concat([(df["High"]-df["Low"]).abs(), (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14, min_periods=1).mean()
    # Bands for MATLAB toggle
    ma = df["Close"].rolling(20).mean(); std = df["Close"].rolling(20).std()
    df["BB_UPPER"] = ma + 2*std; df["BB_LOWER"] = ma - 2*std
    # ADX rough
    up_move = df["High"].diff(); down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = tr.rolling(14, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / (atr14 + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / (atr14 + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    df["ADX14"] = dx.rolling(14).mean()
    return df

def detect_candle(df: pd.DataFrame) -> str:
    if df is None or len(df) < 3: return "No Data"
    last = df.iloc[-1]; prev = df.iloc[-2]
    body = float(last["Close"] - last["Open"]); prev_body = float(prev["Close"] - prev["Open"])
    rng = float(last["High"] - last["Low"]); 
    if rng <= 0: return "No Data"
    upper = float(last["High"] - max(last["Close"], last["Open"]))
    lower = float(min(last["Close"], last["Open"]) - last["Low"])
    if body > 0 and prev_body < 0 and last["Close"] > prev["Open"] and last["Open"] < prev["Close"]:
        return "Bullish Engulfing"
    if body < 0 and prev_body > 0 and last["Open"] > prev["Close"] and last["Close"] < prev["Open"]:
        return "Bearish Engulfing"
    if abs(body) <= 0.1 * rng: return "Doji"
    if abs(body) < (rng * 0.3) and lower > (2 * abs(body)): return "Hammer"
    if abs(body) < (rng * 0.3) and upper > (2 * abs(body)): return "Inverted Hammer"
    if body > 0 and upper > 2 * abs(body) and lower < abs(body): return "Shooting Star"
    return "No Clear Pattern"

def asi(df: pd.DataFrame):
    if df is None or len(df) < 3: return pd.Series([0], index=df.index if df is not None else None)
    df = df.copy(); vals=[np.nan]
    for i in range(1, len(df)):
        H,L,C,O = df["High"].iloc[i], df["Low"].iloc[i], df["Close"].iloc[i], df["Open"].iloc[i]
        Cp,Op = df["Close"].iloc[i-1], df["Open"].iloc[i-1]
        A = abs(H - Cp); B = abs(L - Cp); Cw = abs(H - L)
        R = Cw if (L <= Cp <= H) else abs(H - Cp) if (Cp < L) else abs(Cp - L)
        if R == 0: vals.append(0.0); continue
        K = max(A, B); s = 50.0 * ((C - Cp) + 0.5*(C - O) + 0.25*(Cp - Op)) / R
        if (A + B) != 0: s *= (K / (A + B))
        vals.append(s)
    return pd.Series(vals, index=df.index).fillna(0).cumsum()

def trade_sizing(price, sl, balance, risk_pct, leverage):
    if price is None or sl is None or price<=0 or sl<=0 or price==sl: return None,None,None
    risk_amt = balance * (risk_pct/100.0); per_unit = abs(price - sl)
    if per_unit <= 0: return None,None,None
    units = risk_amt / per_unit; notional = units * price; margin = notional / max(1, leverage)
    return round(units,2), round(notional,2), round(margin,2)

def news_points(pair: str):
    FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/marketsNews",
        "https://www.fxstreet.com/rss",
        "https://www.investing.com/rss/news_25.rss",
    ]
    KEYS = {
        "USD":["usd","fed","powell","cpi","pce","fomc","treasury"],
        "EUR":["eur","ecb","lagarde","eurozone","germany","euro"],
        "GBP":["gbp","boe","uk","britain"],
        "JPY":["jpy","boj","yen","japan"],
        "AUD":["aud","rba","australia"],
        "CAD":["cad","boc","canada","oil"],
        "CHF":["chf","snb","swiss","switzerland"],
        "INR":["inr","rbi","india","rupee"]
    }
    def fetch_newsapi(q="forex OR currency", page_size=25):
        if not NEWSAPI_KEY: return []
        try:
            r = requests.get("https://newsapi.org/v2/everything",
                             params={"q":q,"language":"en","pageSize":page_size,"sortBy":"publishedAt","apiKey":NEWSAPI_KEY},
                             timeout=10)
            out=[]; 
            for a in r.json().get("articles", []):
                out.append({"title":a.get("title",""),"desc":a.get("description",""),"source":(a.get("source") or {}).get("name",""),"url":a.get("url","")})
            return out
        except Exception:
            return []
    def fetch_rss(limit=25):
        items=[]
        for u in FEEDS:
            try:
                f=feedparser.parse(u)
                for e in f.entries[:max(3, limit//len(FEEDS))]:
                    items.append({"title":e.get("title",""),"desc":e.get("summary",""),"source":u,"url":e.get("link","")})
            except Exception:
                pass
        seen=set(); out=[]
        for it in items:
            t=it.get("title","").strip()
            if t and t not in seen: seen.add(t); out.append(it)
        return out[:limit]
    items = fetch_newsapi(f"{pair.split('=')[0]} OR forex OR currency", 30) or fetch_rss(30)
    pos=neg=neu=0; per={k:0.0 for k in KEYS}; details=[]
    for it in items:
        text = (it["title"] + " " + it["desc"]).strip()
        if not text: continue
        sc = float(sia.polarity_scores(text)["compound"])
        if sc >= 0.05: pos+=1
        elif sc <= -0.05: neg+=1
        else: neu+=1
        low=text.lower()
        for cur,kws in KEYS.items():
            if any(kw in low for kw in kws):
                per[cur] += (1 if sc>=0 else -1) * abs(sc)
        details.append({"title":it["title"],"score":sc,"source":it.get("source",""),"url":it.get("url","")})
    p = pair.replace("=X","").upper(); base,quote = p[:3], p[3:6]
    bias = float(per.get(base,0.0) - per.get(quote,0.0))
    return {"pos":pos,"neg":neg,"neu":neu,"bias":bias,"items":details}

def composite_decision(action, news_bias, atr, adx):
    tech = 1.0 if action=="BUY" else -1.0 if action=="SELL" else 0.0
    vol = 0.5 if atr>0 else 0.0
    trend = (adx or 0)/50.0
    score = 0.28*(news_bias/3.0) + 0.24*tech + 0.16*vol + 0.12*trend
    score = float(max(-1.0, min(1.0, score)))
    if score >= 0.45: final="STRONG BUY"
    elif score >= 0.12: final="BUY"
    elif score <= -0.45: final="STRONG SELL"
    elif score <= -0.12: final="SELL"
    else: final="HOLD"
    conf = "High" if abs(score)>=0.62 else "Medium" if abs(score)>=0.3 else "Low"
    # Trade / Don't Trade rule
    trade_ok = (final!="HOLD") and not (conf=="Low" or (adx is not None and adx<18))
    trade_msg = "✅ Trade" if trade_ok else "⏸️ Don't trade now"
    return score, final, conf, trade_msg

# ---------- Real-Time panel (IST) ----------
st.markdown("### ⚡ Real‑Time Intraday (IST)")
c1,c2,c3 = st.columns([1,1,1])
with c1:
    rt_interval = st.selectbox("Interval", ["1m","2m","5m","15m","30m","60m"], index=1)
with c2:
    rt_period = st.selectbox("Lookback", ["1d","5d","7d"], index=0)
with c3:
    show_panels = st.checkbox("Show RSI/MACD panels", value=True)

df_rt = download_intraday(rt_pair, period=rt_period, interval=rt_interval)
if df_rt is not None:
    df_rt = compute_indicators(df_rt)
    df_rt = df_rt.tz_localize("UTC").tz_convert(IST) if df_rt.index.tz is None else df_rt.tz_convert(IST)
    fig = make_subplots(rows=3 if show_panels else 1, cols=1, shared_xaxes=True, row_heights=[0.62,0.19,0.19] if show_panels else [1])
    fig.add_trace(go.Candlestick(x=df_rt.index, open=df_rt["Open"], high=df_rt["High"], low=df_rt["Low"], close=df_rt["Close"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_rt.index, y=df_rt["EMA20"], mode="lines", name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_rt.index, y=df_rt["EMA50"], mode="lines", name="EMA50"), row=1, col=1)
    if SCIPY_OK and len(df_rt)>15:
        sm = savgol_filter(df_rt["Close"].values, 15, 2)
        fig.add_trace(go.Scatter(x=df_rt.index, y=sm, mode="lines", name="Smooth(15,2)"), row=1, col=1)
    if show_panels:
        fig.add_trace(go.Scatter(x=df_rt.index, y=df_rt["RSI"], mode="lines", name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.add_trace(go.Bar(x=df_rt.index, y=df_rt["MACD_HIST"], name="MACD Hist"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_rt.index, y=df_rt["MACD"], name="MACD", mode="lines"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_rt.index, y=df_rt["MACD_SIGNAL"], name="Signal", mode="lines"), row=3, col=1)
    now_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    fig.update_layout(height=600 if show_panels else 420, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0), title=f"{rt_pair} · {now_ist}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No intraday data for selected pair/interval.")

st.markdown("---")
st.subheader("📈 Intraday Cards — click tools per pair")

# ---------- Per pair cards with click tools ----------
for pair in pairs:
    df = download_history(pair, period="45d", interval="1h")
    if df is None:
        st.warning(f"{pair}: no price data"); 
        continue
    df = compute_indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    last_close, prev_close = float(last["Close"]), float(prev["Close"])

    # Base action (EMA + momentum)
    action = "HOLD"
    if last_close > prev_close and last["Close"] >= last["EMA20"]:
        action = "BUY"
    elif last_close < prev_close and last["Close"] <= last["EMA20"]:
        action = "SELL"

    atr = float(last["ATR"]) if not math.isnan(float(last["ATR"])) else max(df["Close"].pct_change().std() * last_close, last_close*0.0001)
    sl=tp1=tp2=None
    if action=="BUY":
        sl = last_close - 1.2*atr; tp1 = last_close + 1.5*(last_close - sl); tp2 = last_close + 2.5*(last_close - sl)
    elif action=="SELL":
        sl = last_close + 1.2*atr; tp1 = last_close - 1.5*(sl - last_close); tp2 = last_close - 2.5*(sl - last_close)

    pattern = detect_candle(df)
    news = news_points(pair)
    score, final, conf, trade_msg = composite_decision(action, news["bias"], atr, float(last.get("ADX14", np.nan)) if "ADX14" in df.columns else 0.0)
    units, notional, margin = trade_sizing(last_close, sl, account_balance, risk_pct, leverage)

    # Header card
    st.markdown(f"<div class='card'><h4 style='margin:0'>{pair} — <span class='good'>{final}</span>  <span class='small'>({trade_msg})</span></h4>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Conf: {conf} • Price: {round(last_close,6)} • RSI: {round(float(last['RSI']),2)} • ATR: {round(atr,6)} • ADX14: {round(float(last.get('ADX14',np.nan)),2) if 'ADX14' in df.columns else '-'} • Pattern: {pattern}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Entry <b>{round(last_close,6)}</b> • SL <b>{None if sl is None else round(sl,6)}</b> • TP1 <b>{None if tp1 is None else round(tp1,6)}</b> • TP2 <b>{None if tp2 is None else round(tp2,6)}</b> • Units≈ {units} • Margin≈ {margin}</div>", unsafe_allow_html=True)

    # Tool toggles (simple checkboxes)
    c1,c2,c3 = st.columns(3)
    with c1:
        show_candles = st.checkbox(f"🔍 Candle analysis — {pair}", value=False, key=f"cand_{pair}")
    with c2:
        show_matlab = st.checkbox(f"🧪 MATLAB tools — {pair}", value=False, key=f"mat_{pair}")
    with c3:
        show_news = st.checkbox(f"📰 News sentiment — {pair}", value=True, key=f"news_{pair}")

    # Candle pattern explanation
    if show_candles:
        st.write("**Candlestick insight:**", pattern)
        st.write("Rules: Bullish patterns above EMA20 and rising RSI favor **longs**; bearish patterns below EMA20 with falling RSI favor **shorts**. Doji suggests patience.")

    # MATLAB tools chart (Bands + smoothing + EMA)
    if show_matlab:
        dfc = download_intraday(pair, period="1d", interval="5m")
        if dfc is not None:
            dfc = compute_indicators(dfc)
            if SCIPY_OK and len(dfc)>15:
                dfc["SMOOTH"] = savgol_filter(dfc["Close"].values, 15, 2)
            figm = go.Figure()
            figm.add_trace(go.Candlestick(x=dfc.index, open=dfc['Open'], high=dfc['High'], low=dfc['Low'], close=dfc['Close']))
            figm.add_trace(go.Scatter(x=dfc.index, y=dfc['EMA20'], mode='lines', name='EMA20'))
            figm.add_trace(go.Scatter(x=dfc.index, y=dfc['EMA50'], mode='lines', name='EMA50'))
            figm.add_trace(go.Scatter(x=dfc.index, y=dfc['BB_UPPER'], mode='lines', name='BB Upper'))
            figm.add_trace(go.Scatter(x=dfc.index, y=dfc['BB_LOWER'], mode='lines', name='BB Lower'))
            if "SMOOTH" in dfc.columns:
                figm.add_trace(go.Scatter(x=dfc.index, y=dfc['SMOOTH'], mode='lines', name='Smooth(15,2)'))
            x_last = dfc.index[-1]; y_last = float(dfc["Close"].iloc[-1])
            if final in ("BUY","STRONG BUY"):
                figm.add_annotation(x=x_last, y=y_last, text="▲", showarrow=True, arrowhead=2, ax=0, ay=-30)
            elif final in ("SELL","STRONG SELL"):
                figm.add_annotation(x=x_last, y=y_last, text="▼", showarrow=True, arrowhead=2, ax=0, ay=30)
            figm.update_layout(height=320, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0), title=f"{pair} · MATLAB overlays")
            st.plotly_chart(figm, use_container_width=True)
        else:
            st.info("No 5m data for MATLAB tools right now.")

    # News sentiment list + trade/don't trade summary
    if show_news:
        bias = news["bias"]; pos,neg,neu = news["pos"], news["neg"], news["neu"]
        st.write(f"**News Sentiment:** Bias {bias:+.3f}  |  Headlines +{pos} / −{neg} / ◼ {neu}")
        with st.expander("Show latest headlines"):
            for h in news["items"][:10]:
                txt = f"({h.get('score'):+.3f}) {h.get('title')} — {h.get('source','')}"
                url = h.get("url","")
                if url: st.markdown(f"- [{txt}]({url})")
                else: st.write(f"- {txt}")

    st.markdown("</div>", unsafe_allow_html=True)  # close card
    st.markdown("---")

st.caption("Toggle tools per pair. IST time. Data: Yahoo Finance; News: NewsAPI/RSS.")

# ---- Auto-refresh (version-safe) ----
import time as _t
_t.sleep(refresh)
try:
    st.rerun()
except Exception:
    try:
        st.experimental_rerun()
    except Exception:
        pass
