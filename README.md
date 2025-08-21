# Forex Pro — Click Tools (Ready to Run)

This app provides **real-time intraday forex analysis** with per-pair expandable tools:

## Features
- ✅ Final signal: Buy / Sell / Hold with confidence
- 🔔 Clear advice: **Trade** or **Don't trade now** (based on signal, ADX, confidence)
- 📈 Candlestick pattern detection (Doji, Engulfing, Hammer, etc.)
- 🧮 MATLAB-inspired tools: Bollinger Bands, EMA20/50, Savitzky–Golay smoothing (if SciPy installed)
- 📰 News sentiment analysis with clickable headlines
- 📊 Real-time intraday chart (IST) with EMA/RSI/MACD
- 🧠 Position sizing + margin based on balance, risk%, leverage

## Local Run (Windows/Mac/Linux)
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

# optional for richer news:
# Windows (CMD)
set NEWSAPI_KEY=1502cba32d134f4095aa03d4bd5bfe3c
# macOS/Linux (bash)
export NEWSAPI_KEY=1502cba32d134f4095aa03d4bd5bfe3c

python -m streamlit run app_forex_pro_click_tools.py
```

## Streamlit Cloud
- Main file: `app_forex_pro_click_tools.py`
- Requirements: `requirements.txt`
- (Optional) Secrets → `NEWSAPI_KEY`

---
**Educational only. Not financial advice.**
