import requests
import numpy as np
import pandas as pd
import time
import math
import os
import json
from scipy.stats import t as student_t

# ==========================================
# 🔐 CONFIGURATION & LOGGING
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
STATE_FILE = "trade_state.json"

# Global string to accumulate the report for Telegram
FULL_REPORT = ""

def log(text):
    """Prints to console AND adds to Telegram report"""
    global FULL_REPORT
    print(text)
    FULL_REPORT += text + "\n"

# ==========================================
# 💾 MEMORY ENGINE (Load/Save JSON)
# ==========================================
def load_state():
    """Loads the balance and active positions from file"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    else:
        # Initial Start: $100.00 for each strategy
        return {
            "scalp": {"balance": 100.0, "position": None, "entry_price": 0, "tp": 0, "sl": 0},
            "day":   {"balance": 100.0, "position": None, "entry_price": 0, "tp": 0, "sl": 0},
            "swing": {"balance": 100.0, "position": None, "entry_price": 0, "tp": 0, "sl": 0}
        }

def save_state(state):
    """Saves the current state back to JSON"""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

# ==========================================
# 📡 DATA ENGINE (US-Friendly / Kraken)
# ==========================================
def get_kraken_data(interval_mins):
    """Fetches Candles and Order Book Imbalance"""
    try:
        # 1. Get OHLC Candles
        url_ohlc = "https://api.kraken.com/0/public/OHLC"
        params = {'pair': 'XBTUSD', 'interval': interval_mins}
        headers = {'User-Agent': 'Mozilla/5.0'} # Pretend to be a browser
        
        resp_ohlc = requests.get(url_ohlc, params=params, headers=headers, timeout=10).json()
        if resp_ohlc.get('error'): return None, None

        pair = list(resp_ohlc['result'].keys())[0]
        candles = resp_ohlc['result'][pair]
        
        # Parse into DataFrame
        closes = [float(c[4]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        opens = [float(c[1]) for c in candles]
        
        df = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})

        # 2. Get Order Book (Imbalance)
        url_book = "https://api.kraken.com/0/public/Depth"
        resp_book = requests.get(url_book, params={'pair': 'XBTUSD', 'count': 50}, headers=headers, timeout=10).json()
        bids = np.array(resp_book['result'][pair]['bids'], dtype=float)
        asks = np.array(resp_book['result'][pair]['asks'], dtype=float)
        
        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        imbalance = bid_vol / (bid_vol + ask_vol)

        return df, imbalance
    except Exception as e:
        print(f"Data Error: {e}")
        return None, None

# ==========================================
# 🧮 MATH & INDICATOR ENGINE
# ==========================================
def calculate_parkinson_volatility(df, interval_mins):
    """Smart Volatility using High/Low ranges (Best for Crypto)"""
    df['hl_log'] = np.log(df['High'] / df['Low']) ** 2
    variance = df['hl_log'].mean() / (4 * np.log(2))
    period_vol = np.sqrt(variance)
    periods_per_day = 1440 / interval_mins
    return period_vol * np.sqrt(periods_per_day)

def calculate_adx(df, period=14):
    """Calculates Trend Strength (ADX)"""
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
    """Calculates Probabilities using Student's t-Distribution (df=3)"""
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
# 💸 PAPER TRADE ENGINE (Smart Replay)
# ==========================================
def manage_position(mode, state, df, interval_mins):
    """
    Replays recent candles to check if TP/SL was hit inside the 1-hour gap.
    """
    pos = state[mode]
    if pos["position"] is None:
        return "NO_POSITION"

    entry = pos["entry_price"]
    tp_price = pos["tp"]
    sl_price = pos["sl"]
    balance = pos["balance"]
    
    # How many candles fit in 1 hour? (The bot runs hourly)
    # Scalp (5m) = 12 candles. Day (60m) = 1 candle.
    candles_to_check = int(60 / interval_mins)
    if candles_to_check < 1: candles_to_check = 1
    
    # Get the slice of history since the last run
    history_window = df.iloc[-(candles_to_check + 1):-1]
    
    result = None
    
    # Replay Loop
    for index, row in history_window.iterrows():
        high = row['High']
        low = row['Low']
        
        if pos["position"] == "LONG":
            if low <= sl_price:   # Loss
                result = "LOSS"
                pct = (sl_price - entry) / entry
                balance = balance * (1 + pct)
                break
            elif high >= tp_price: # Win
                result = "WIN"
                pct = (tp_price - entry) / entry
                balance = balance * (1 + pct)
                break 

        elif pos["position"] == "SHORT":
            if high >= sl_price:  # Loss
                result = "LOSS"
                pct = (entry - sl_price) / entry
                balance = balance * (1 + pct)
                break
            elif low <= tp_price: # Win
                result = "WIN"
                pct = (entry - tp_price) / entry
                balance = balance * (1 + pct)
                break

    # Update State
    if result:
        pos["balance"] = balance
        pos["position"] = None
        pos["entry_price"] = 0
        pos["tp"] = 0
        pos["sl"] = 0
        return f"{result} (Bal: ${balance:.2f})"
    
    # Floating PnL
    current_price = df.iloc[-1]['Close']
    if pos["position"] == "LONG":
        floating_pnl = (current_price - entry) / entry * 100
    else:
        floating_pnl = (entry - current_price) / entry * 100
        
    return f"HOLD ({floating_pnl:+.2f}%)"

# ==========================================
# 🚀 MAIN STRATEGY LOOP
# ==========================================
def analyze_strategy(mode, state):
    # 1. Define Parameters
    if mode == "scalp":
        title = "⚡ SCALP (1-4h)"
        interval = 5; duration = 0.1; tp=0.008; sl=0.004
    elif mode == "swing":
        title = "🌊 SWING (5-7d)"
        interval = 1440; duration = 7; tp=0.10; sl=0.05
    else: # Day
        title = "📅 DAY (24h)"
        interval = 60; duration = 1; tp=0.02; sl=0.008

    # 2. Get Data
    df, imbalance = get_kraken_data(interval)
    if df is None:
        log(f"⚠️ Error fetching data for {mode}")
        return

    # 3. Indicators
    latest = calculate_adx(df)
    current_price = latest['Close']
    adx = latest['ADX']
    if np.isnan(adx): adx = 20.0
    
    daily_vol = calculate_parkinson_volatility(df, interval)
    
    # 4. Drift (Order Book + Trend)
    book_bias = (imbalance - 0.5) * 2 
    trend_bias = 0.5 if latest['+DI'] > latest['-DI'] else -0.5
    total_drift = (book_bias * 0.4) + (trend_bias * 0.6)

    # 5. Math (Fat Tails)
    p_long, p_short, p_stable = calculate_fat_tail_probs(
        current_price, daily_vol, duration, tp, sl, total_drift
    )

    # 6. Check Active Trades (Replay Engine)
    trade_status = manage_position(mode, state, df, interval)
    
    # 7. Print Report
    log(f"\n" + "="*30)
    log(f" {title}")
    log(f" 💰 Balance: ${state[mode]['balance']:.2f}")
    
    if state[mode]['position']:
        log(f" 🚩 Position: {state[mode]['position']} @ ${state[mode]['entry_price']:.0f}")
        log(f"    Status: {trade_status}")
    else:
        log(f" ⚪ Status: Cash")
        
    log(f"-"*30)
    log(f" 📊 ADX: {adx:.1f} | Buyers: {imbalance*100:.0f}%")
    log(f" 🎲 L:{p_long:.0f}% | S:{p_short:.0f}% | Stable:{p_stable:.0f}%")
    
    # 8. Signal Logic (Entry)
    if state[mode]['position'] is None:
        signal = "WAIT"
        
        if adx < 20: signal = "WAIT (Dead)"
        elif p_stable > 60: signal = "WAIT (Stable)"
        elif p_long > p_short and p_long > 45: signal = "LONG"
        elif p_short > p_long and p_short > 45: signal = "SHORT"
        
        # Execute Entry
        if signal == "LONG":
            state[mode]['position'] = "LONG"
            state[mode]['entry_price'] = current_price
            state[mode]['tp'] = current_price * (1 + tp)
            state[mode]['sl'] = current_price * (1 - sl)
            log(f" ✅ OPEN LONG! TP: ${state[mode]['tp']:.0f} (+{tp*100}%)")
        elif signal == "SHORT":
            state[mode]['position'] = "SHORT"
            state[mode]['entry_price'] = current_price
            state[mode]['tp'] = current_price * (1 - tp)
            state[mode]['sl'] = current_price * (1 + sl)
            log(f" 🔻 OPEN SHORT! TP: ${state[mode]['tp']:.0f} (-{tp*100}%)")
        else:
            log(f" ✋ Action: {signal}")

def send_to_telegram():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Telegram keys missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": FULL_REPORT}
    try: requests.post(url, json=payload)
    except Exception as e: print(f"Telegram Error: {e}")

if __name__ == "__main__":
    current_state = load_state()
    log("🤖 INSTITUTIONAL BTC SCANNER 🤖")
    
    for s in ["scalp", "day", "swing"]:
        analyze_strategy(s, current_state)
        time.sleep(1) # Pause to respect API
    
    save_state(current_state)
    send_to_telegram()
