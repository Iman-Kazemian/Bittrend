import requests
import numpy as np
import pandas as pd
import time
import math
import os
import json
from scipy.stats import t as student_t

# ==========================================
# 🔐 CONFIGURATION
# ==========================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
STATE_FILE = "trade_state.json"
FULL_REPORT = ""

def log(text):
    global FULL_REPORT
    print(text)
    FULL_REPORT += text + "\n"

# ==========================================
# 💾 MEMORY ENGINE (Load/Save JSON)
# ==========================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    else:
        # Default Start: $100 for each strategy
        return {
            "scalp": {"balance": 100.0, "position": None, "entry_price": 0, "tp": 0, "sl": 0},
            "day":   {"balance": 100.0, "position": None, "entry_price": 0, "tp": 0, "sl": 0},
            "swing": {"balance": 100.0, "position": None, "entry_price": 0, "tp": 0, "sl": 0}
        }

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

# ==========================================
# 📡 DATA ENGINE
# ==========================================
def get_kraken_data(interval_mins):
    try:
        url_ohlc = "https://api.kraken.com/0/public/OHLC"
        params = {'pair': 'XBTUSD', 'interval': interval_mins}
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        resp_ohlc = requests.get(url_ohlc, params=params, headers=headers, timeout=10).json()
        if resp_ohlc.get('error'): return None, None, None

        pair = list(resp_ohlc['result'].keys())[0]
        candles = resp_ohlc['result'][pair]
        
        # Parse DataFrame
        closes = [float(c[4]) for c in candles]
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        opens = [float(c[1]) for c in candles]
        
        df = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': closes})

        # Order Book Imbalance
        url_book = "https://api.kraken.com/0/public/Depth"
        resp_book = requests.get(url_book, params={'pair': 'XBTUSD', 'count': 50}, headers=headers, timeout=10).json()
        bids = np.array(resp_book['result'][pair]['bids'], dtype=float)
        asks = np.array(resp_book['result'][pair]['asks'], dtype=float)
        
        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        imbalance = bid_vol / (bid_vol + ask_vol)

        # Get the LAST COMPLETED CANDLE (Index -2) to check if we hit TP/SL during the hour
        last_candle_high = highs[-2]
        last_candle_low = lows[-2]

        return df, imbalance, (last_candle_high, last_candle_low)
    except:
        return None, None, None

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
        df['+DI'] = 100 * (df['+DM_s'] / df['TR_s']) if 'TR_s' in df else 100 * (df['+DM_smooth'] / df['TR_smooth'])
        df['-DI'] = 100 * (df['-DM_s'] / df['TR_s']) if 'TR_s' in df else 100 * (df['-DM_smooth'] / df['TR_smooth'])
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
# 💸 PAPER TRADING LOGIC
# ==========================================
def manage_position(mode, state, current_price, last_high, last_low):
    """
    Checks if active trades hit TP or SL.
    Returns the updated State string (e.g., "HOLD", "WIN", "LOSS")
    """
    pos = state[mode]
    if pos["position"] is None:
        return "NO_POSITION"

    entry = pos["entry_price"]
    tp_price = pos["tp"]
    sl_price = pos["sl"]
    balance = pos["balance"]
    
    # 1. Check if Hit (Prioritize SL logic for safety)
    # We check 'last_high' and 'last_low' to see if price wicked there during the hour
    
    pnl = 0
    result = None
    
    if pos["position"] == "LONG":
        if last_low <= sl_price:   # Stopped Out
            result = "LOSS"
            # Calculate % loss
            pct_change = (sl_price - entry) / entry
            balance = balance * (1 + pct_change)
        elif last_high >= tp_price: # Take Profit
            result = "WIN"
            pct_change = (tp_price - entry) / entry
            balance = balance * (1 + pct_change)
            
    elif pos["position"] == "SHORT":
        if last_high >= sl_price:  # Stopped Out
            result = "LOSS"
            pct_change = (entry - sl_price) / entry
            balance = balance * (1 + pct_change)
        elif last_low <= tp_price: # Take Profit
            result = "WIN"
            pct_change = (entry - tp_price) / entry
            balance = balance * (1 + pct_change)

    # 2. Update State if trade ended
    if result:
        pos["balance"] = balance
        pos["position"] = None
        pos["entry_price"] = 0
        pos["tp"] = 0
        pos["sl"] = 0
        return f"{result} (Bal: ${balance:.2f})"
    
    # 3. If trade is still open, calculate floating PnL
    if pos["position"] == "LONG":
        floating_pnl = (current_price - entry) / entry * 100
    else:
        floating_pnl = (entry - current_price) / entry * 100
        
    return f"HOLD (PnL: {floating_pnl:+.2f}%)"

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
def analyze_strategy(mode, state):
    if mode == "scalp":
        title = "⚡ SCALP (1-4h)"
        interval = 5; duration = 0.1; tp=0.008; sl=0.004
    elif mode == "swing":
        title = "🌊 SWING (5-7d)"
        interval = 1440; duration = 7; tp=0.10; sl=0.05
    else: # Day
        title = "📅 DAY (24h)"
        interval = 60; duration = 1; tp=0.02; sl=0.008

    df, imbalance, last_candles = get_kraken_data(interval)
    if df is None: return

    latest = calculate_adx(df)
    current_price = latest['Close']
    adx = latest['ADX']
    if np.isnan(adx): adx = 20.0
    
    daily_vol = calculate_parkinson_volatility(df, interval)
    
    # Drift
    book_bias = (imbalance - 0.5) * 2 
    trend_bias = 0.5 if latest['+DI'] > latest['-DI'] else -0.5
    total_drift = (book_bias * 0.4) + (trend_bias * 0.6)

    # Math
    p_long, p_short, p_stable = calculate_fat_tail_probs(
        current_price, daily_vol, duration, tp, sl, total_drift
    )

    # --- PAPER TRADE CHECK ---
    trade_status = manage_position(mode, state, current_price, last_candles[0], last_candles[1])
    
    # --- REPORTING ---
    log(f"\n" + "="*30)
    log(f" {title}")
    log(f" 💰 Balance: ${state[mode]['balance']:.2f}")
    if state[mode]['position']:
        log(f" 🚩 Position: {state[mode]['position']} @ ${state[mode]['entry_price']:.0f}")
        log(f"    Status: {trade_status}")
    else:
        log(f" ⚪ Status: Cash")
    log(f"-"*30)
    
    # Signal Logic
    signal = "WAIT"
    if state[mode]['position'] is None: # Only look for signals if we are flat
        if adx < 20: signal = "WAIT (Dead Mkt)"
        elif p_stable > 60: signal = "WAIT (Low Vol)"
        elif p_long > p_short and p_long > 45: signal = "LONG"
        elif p_short > p_long and p_short > 45: signal = "SHORT"
        
        # EXECUTE NEW TRADE
        if signal == "LONG":
            state[mode]['position'] = "LONG"
            state[mode]['entry_price'] = current_price
            state[mode]['tp'] = current_price * (1 + tp)
            state[mode]['sl'] = current_price * (1 - sl)
            log(f" ✅ OPENING LONG! Target: ${state[mode]['tp']:.0f}")
        elif signal == "SHORT":
            state[mode]['position'] = "SHORT"
            state[mode]['entry_price'] = current_price
            state[mode]['tp'] = current_price * (1 - tp)
            state[mode]['sl'] = current_price * (1 + sl)
            log(f" 🔻 OPENING SHORT! Target: ${state[mode]['tp']:.0f}")
    
    # Print Probs
    log(f" 📊 Probs: L:{p_long:.0f}% S:{p_short:.0f}% Stable:{p_stable:.0f}%")

def send_to_telegram():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": FULL_REPORT}
    try: requests.post(url, json=payload)
    except: pass

if __name__ == "__main__":
    current_state = load_state()
    log("🤖 HOURLY PAPER TRADING REPORT 🤖")
    
    for s in ["scalp", "day", "swing"]:
        analyze_strategy(s, current_state)
        time.sleep(1)
    
    save_state(current_state)
    send_to_telegram()
