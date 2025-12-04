import requests
import numpy as np
import pandas as pd
import time
import math
import os
from scipy.stats import t as student_t

# ==========================================
# 🔐 CONFIGURATION (Env Variables)
# ==========================================
# GitHub Actions will inject these securely
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Global Report String (We will add text to this instead of just printing)
FULL_REPORT = ""

def log(text):
    """Helper to print to console AND add to Telegram report"""
    global FULL_REPORT
    print(text)
    FULL_REPORT += text + "\n"

# ==========================================
# 📡 DATA ENGINE (US-Friendly)
# ==========================================

def get_kraken_data(interval_mins):
    try:
        url_ohlc = "https://api.kraken.com/0/public/OHLC"
        params = {'pair': 'XBTUSD', 'interval': interval_mins}
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        resp_ohlc = requests.get(url_ohlc, params=params, headers=headers, timeout=10).json()
        if resp_ohlc.get('error'): return None, None

        pair = list(resp_ohlc['result'].keys())[0]
        candles = resp_ohlc['result'][pair]
        
        closes = [float(c[4]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        opens = [float(c[1]) for c in candles]
        
        df = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})

        url_book = "https://api.kraken.com/0/public/Depth"
        resp_book = requests.get(url_book, params={'pair': 'XBTUSD', 'count': 50}, headers=headers, timeout=10).json()
        bids = np.array(resp_book['result'][pair]['bids'], dtype=float)
        asks = np.array(resp_book['result'][pair]['asks'], dtype=float)
        
        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        imbalance = bid_vol / (bid_vol + ask_vol)

        return df, imbalance
    except:
        return None, None

def calculate_parkinson_volatility(df, interval_mins):
    df['hl_log'] = np.log(df['High'] / df['Low']) ** 2
    variance = df['hl_log'].mean() / (4 * np.log(2))
    period_vol = np.sqrt(variance)
    periods_per_day = 1440 / interval_mins
    return period_vol * np.sqrt(periods_per_day)

def calculate_adx(df, period=14):
    df['tr0'] = abs(df['High'] - df['Low'])
    df['tr1'] = abs(df['High'] - df['Close'].shift(1))
    df['tr2'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         df['High'] - df['High'].shift(1), 0)
    df['+DM'] = np.where(df['+DM'] < 0, 0, df['+DM'])
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                         df['Low'].shift(1) - df['Low'], 0)
    df['-DM'] = np.where(df['-DM'] < 0, 0, df['-DM'])

    df['TR_smooth'] = df['TR'].rolling(period).mean()
    df['+DM_smooth'] = df['+DM'].rolling(period).mean()
    df['-DM_smooth'] = df['-DM'].rolling(period).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
        df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    
    df['ADX'] = df['DX'].rolling(period).mean()
    return df.iloc[-1]

def calculate_fat_tail_probs(price, vol_daily, duration_days, tp_pct, sl_pct, drift_score):
    period_vol = vol_daily * math.sqrt(duration_days)
    df_crypto = 3 
    expected_mean = price * (1 + (drift_score * period_vol))
    
    t_long = (price * (1 + tp_pct) - expected_mean) / (price * period_vol)
    t_short = (price * (1 - sl_pct) - expected_mean) / (price * period_vol)
    
    prob_long = student_t.sf(t_long, df_crypto) * 100 
    prob_short = student_t.cdf(t_short, df_crypto) * 100
    prob_stable = 100 - (prob_long + prob_short)
    
    return max(0, prob_long), max(0, prob_short), max(0, prob_stable)

# ==========================================
# 🚀 STRATEGY ENGINE
# ==========================================

def analyze_strategy(mode):
    if mode == "scalp":
        title = "⚡ SCALP (1-4 Hours)"
        interval = 5; duration = 0.1; tp=0.008; sl=0.004
    elif mode == "swing":
        title = "🌊 SWING (5-7 Days)"
        interval = 1440; duration = 7; tp=0.10; sl=0.05
    else: # Day
        title = "📅 DAY (24 Hours)"
        interval = 60; duration = 1; tp=0.02; sl=0.008

    df, imbalance = get_kraken_data(interval)
    if df is None:
        log(f"⚠️ Error fetching data for {mode}")
        return

    latest = calculate_adx(df)
    current_price = latest['Close']
    adx = latest['ADX']
    if np.isnan(adx): adx = 20.0
    
    daily_vol = calculate_parkinson_volatility(df, interval)
    vol_msg = f"{daily_vol*100:.1f}%"

    book_bias = (imbalance - 0.5) * 2 
    trend_bias = 0.5 if latest['+DI'] > latest['-DI'] else -0.5
    total_drift = (book_bias * 0.4) + (trend_bias * 0.6)

    p_long, p_short, p_stable = calculate_fat_tail_probs(
        current_price, daily_vol, duration, tp, sl, total_drift
    )

    log(f"\n" + "="*40)
    log(f" {title} | ${current_price:,.0f}")
    log(f"="*40)
    log(f"📊 ADX: {adx:.1f} | Vol: {vol_msg} | Flow: {imbalance*100:.0f}% Buys")
    log(f"🎲 Long: {p_long:.1f}% | Short: {p_short:.1f}% | Stable: {p_stable:.1f}%")
    
    signal = "✋ WAIT"
    
    if adx < 20: signal = "✋ WAIT (Dead Market)"
    elif p_stable > 60: signal = "✋ WAIT (Low Volatility)"
    elif p_long > p_short and p_long > 45:
        signal = "✅ LONG"
    elif p_short > p_long and p_short > 45:
        signal = "🔻 SHORT"
        
    log(f"📢 SIGNAL: {signal}")
    
    if "LONG" in signal or "SHORT" in signal:
        target = tp if "LONG" in signal else sl
        stop = sl if "LONG" in signal else tp
        # For print clarity, I am showing pure percentages here
        # In actual calculation, prices are derived below
        pass

    if "LONG" in signal:
        log(f"   👉 TP: ${current_price*(1+tp):,.0f} (+{tp*100}%)")
        log(f"   👉 SL: ${current_price*(1-sl):,.0f} (-{sl*100}%)")
    elif "SHORT" in signal:
        log(f"   👉 TP: ${current_price*(1-tp):,.0f} (-{tp*100}%)")
        log(f"   👉 SL: ${current_price*(1+sl):,.0f} (+{sl*100}%)")

def send_to_telegram():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram keys missing. Check GitHub Secrets.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "🤖 *BITCOIN STRATEGY REPORT* 🤖\n" + FULL_REPORT,
        "parse_mode": "Markdown" # Or use empty string if special chars fail
    }
    try:
        requests.post(url, json=payload)
        print("✅ Telegram sent successfully.")
    except Exception as e:
        print(f"❌ Telegram failed: {e}")

if __name__ == "__main__":
    log("🔬 STARTING SCAN (Parkinson Vol)...")
    for s in ["scalp", "day", "swing"]:
        analyze_strategy(s)
        time.sleep(1)
    
    send_to_telegram()
