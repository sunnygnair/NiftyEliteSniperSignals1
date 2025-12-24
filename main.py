import streamlit as st
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import numpy as np
import datetime
import json
import os
import glob
import logging
import signal
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. PAGE CONFIG & ULTRA-COMPACT STYLING
# ==========================================
st.set_page_config(page_title="Nifty 100 Elite Sniper", layout="wide", page_icon="🎯")

st.markdown(
    """
    <style>
        /* 1. Global Spacing - Remove Top Whitespace */
        .block-container {
            padding-top: 0.1rem !important; /* Minimized top padding */
            padding-bottom: 0rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* 2. Tighten Vertical Gaps between Widgets */
        div[data-testid="stVerticalBlock"] {
            gap: 0.4rem !important; /* Reduced from 1rem to 0.4rem */
        }
        
        /* 3. Headers - Reduce Margins */
        h3 {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-bottom: 0px !important;
        }
        p {
            margin-bottom: 0px !important;
        }
        
        /* 4. Custom Metric Cards (Compact) */
        div[data-testid="stMetric"] {
            background-color: #161920;
            border: 1px solid #303030;
            padding: 8px 12px; /* Reduced padding */
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            min-height: 70px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            color: #9aa0a6;
            margin-bottom: 0px !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
            color: #ffffff;
            font-weight: 700;
        }
        
        /* 5. Dataframe - Remove internal padding/margins */
        .stDataFrame {
            border: 1px solid #303030;
            border-radius: 5px;
            margin-top: 0px !important;
        }
        
        /* 6. Primary Button Styling */
        div.stButton > button {
            width: 100%;
            border-radius: 5px;
            font-weight: 600;
            background-color: #ff4b4b;
            color: white;
            border: none;
            height: 2.5rem; /* Slightly shorter */
            margin-top: 5px;
        }
        div.stButton > button:hover {
            background-color: #ff3333;
            border: none;
        }

        /* 7. AI Box */
        .ai-box {
            background-color: #0e1117;
            padding: 10px 15px;
            border-radius: 6px;
            border-left: 3px solid #00d4ff;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.85rem;
            line-height: 1.4;
            color: #e0e0e0;
            margin-top: 5px;
        }
        
        /* 8. Hide Footer/Header */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. CONFIG & LOGGING
# ==========================================
logging.basicConfig(filename="screener.log", level=logging.INFO)

def load_api_key():
    # First check Render/System Environment Variable
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    
    # Fallback to config.json (for local testing)
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                return json.load(f).get("api_keys", {}).get("gemini", "")
    except: return ""
    return ""
    
    API_KEY = load_api_key() 

# ==========================================
# 3. BACKEND LOGIC (UNCHANGED)
# ==========================================
def calculate_adx(df, period=14):
    try:
        df = df.copy()
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
        df['+DM'] = np.where(df['+DM'] < 0, 0, df['+DM'])
        df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
        df['-DM'] = np.where(df['-DM'] < 0, 0, df['-DM'])

        df['TR14'] = df['TR'].rolling(window=period).sum()
        df['+DM14'] = df['+DM'].rolling(window=period).sum()
        df['-DM14'] = df['-DM'].rolling(window=period).sum()

        df['+DI'] = 100 * (df['+DM14'] / df['TR14'])
        df['-DI'] = 100 * (df['-DM14'] / df['TR14'])
        
        df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
        df['ADX'] = df['DX'].rolling(window=period).mean()
        return df['ADX'].iloc[-1]
    except: return 0

@st.cache_data(ttl=600)
def load_symbols_from_csv():
    files = glob.glob("MW-NIFTY-100*.csv")
    if not files: files = glob.glob("*.csv")
    if not files: return []

    latest_file = max(files, key=os.path.getctime)
    try:
        df = pd.read_csv(latest_file)
        df.columns = [str(c).strip().replace('"', '').replace('\n', '') for c in df.columns]
        target_col = next((c for c in df.columns if "SYMBOL" in c), None)
        if target_col:
            raw = df[target_col].dropna().unique().tolist()
            return [f"{str(s).strip()}.NS" for s in raw if isinstance(s, str) and s.strip() != "SYMBOL"]
        return []
    except: return []

@st.cache_data(ttl=300)
def run_elite_scan(tickers):
    if not tickers: return pd.DataFrame()
    try:
        data = yf.download(tickers, period="5d", interval="5m", group_by="ticker", threads=True, progress=False, auto_adjust=False)
    except: return pd.DataFrame()
    
    scored_results = []
    
    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if df.empty or len(df) < 50: continue
            
            today_df = df[df.index.date == df.index.date[-1]]
            if today_df.empty: continue
            
            # Daily Ref
            daily_stats = df.resample('D').agg({'High':'max', 'Low':'min'})
            try:
                pd_stats = daily_stats.iloc[-2] 
                pd_high, pd_low = pd_stats['High'], pd_stats['Low']
            except: pd_high, pd_low = 999999, 0

            curr = today_df["Close"].iloc[-1]
            op = today_df["Open"].iloc[0]
            hi = today_df["High"].max()
            lo = today_df["Low"].min()
            vol_now = today_df["Volume"].iloc[-1]
            
            cum_pv = (today_df["Close"] * today_df["Volume"]).cumsum()
            cum_vol = today_df["Volume"].cumsum()
            vwap = (cum_pv / cum_vol).iloc[-1]
            
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            
            ema20 = df["Close"].ewm(span=20).mean().iloc[-1]
            ema50 = df["Close"].ewm(span=50).mean().iloc[-1]
            
            adx = calculate_adx(df)
            avg_vol_20 = df["Volume"].rolling(20).mean().iloc[-1]
            rvol = vol_now / avg_vol_20 if avg_vol_20 > 0 else 0

            # SCORING ENGINE
            if adx < 20: continue 

            bullish_points = 0
            if curr > vwap: bullish_points += 15
            if curr > ema20 > ema50: bullish_points += 15
            if curr > pd_high: bullish_points += 20
            if abs(op - lo) < (curr * 0.001): bullish_points += 25
            if 60 <= rsi <= 75: bullish_points += 10
            if rvol > 2.0: bullish_points += 15

            bearish_points = 0
            if curr < vwap: bearish_points += 15
            if curr < ema20 < ema50: bearish_points += 15
            if curr < pd_low: bearish_points += 20
            if abs(op - hi) < (curr * 0.001): bearish_points += 25
            if 25 <= rsi <= 40: bearish_points += 10
            if rvol > 2.0: bearish_points += 15

            final_bias = "NEUTRAL"
            final_score = 0
            reasons = []
            
            if bullish_points > 60 and bullish_points > bearish_points:
                final_bias = "BUY"
                final_score = bullish_points
                if curr > pd_high: reasons.append("PDH Breakout")
                if abs(op - lo) < (curr * 0.001): reasons.append("Open=Low")
                if rvol > 2.0: reasons.append(f"Vol {rvol:.1f}x")
                
            elif bearish_points > 60 and bearish_points > bullish_points:
                final_bias = "SELL"
                final_score = bearish_points
                if curr < pd_low: reasons.append("PDL Breakdown")
                if abs(op - hi) < (curr * 0.001): reasons.append("Open=High")
                if rvol > 2.0: reasons.append(f"Vol {rvol:.1f}x")

            if final_score >= 65:
                pct_chg = ((curr - op)/op)*100
                scored_results.append({
                    "Stock": ticker.replace(".NS", ""),
                    "Price": curr,
                    "Chg%": pct_chg,
                    "Bias": final_bias,
                    "Conviction": final_score,
                    "Pattern": ", ".join(reasons) if reasons else "Trend Play",
                    "RSI": round(rsi, 1),
                    "ADX": round(adx, 1),
                    "RVOL": round(rvol, 1)
                })
        except Exception as e: continue
    
    df_res = pd.DataFrame(scored_results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="Conviction", ascending=False)
        return df_res.head(10)
    return pd.DataFrame()

def run_elite_ai(key, df):
    if not key or df.empty: return "AI Unavailable."
    top_picks = df.to_dict(orient="records")
    prompt = f"""
    You are a Hedge Fund Algo Trader.
    Top {len(top_picks)} High-Prob Setups: {top_picks}
    
    Task: Pick the safest 3 trades and provide a plan.
    Format:
    | Stock | Action | Entry | Stop | Target | Logic |
    |---|---|---|---|---|---|
    """
    try:
        genai.configure(api_key=key)
        m = genai.GenerativeModel("gemini-3-flash-preview")
        return m.generate_content(prompt).text
    except Exception as e: return f"AI Error: {e}"

def style_elite_table(df):
    def color_rows(row):
        if row['Bias'] == 'BUY': return ['background-color: #0d2e0d; color: #e6ffe6; font-weight: 500'] * len(row)
        if row['Bias'] == 'SELL': return ['background-color: #2e0d0d; color: #ffe6e6; font-weight: 500'] * len(row)
        return [''] * len(row)
    
    return df.style.apply(color_rows, axis=1).format({
        "Price": "{:,.2f}", 
        "Chg%": "{:+.2f}", 
        "RSI": "{:.0f}",
        "ADX": "{:.0f}",
        "RVOL": "{:.1f}x"
    })

# ==========================================
# 4. MAIN LAYOUT
# ==========================================
def main():
    st.sidebar.title("🎯 Controls")
    
    if st.sidebar.button("↻ Force Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    if st.sidebar.button("🛑 Stop App"): os.kill(os.getpid(), signal.SIGTERM)

    # 1. HEADER SECTION (Inline Title, Info & Button to save vertical space)
    symbols = load_symbols_from_csv()
    
    c1, c2, c3 = st.columns([2.5, 0.8, 1.2])
    with c1:
        st.markdown("### 🎯 Nifty 100 Elite Sniper")
        st.caption(f"Filters: ADX>20 | RVOL>2x | EMA Align | Breakouts (Universe: {len(symbols)})")
    with c3:
        scan_click = st.button("🚀 FIND TOP 10 TRADES")
        if scan_click: st.session_state['scan_active'] = True

    # 2. EXECUTION & RESULTS
    if not symbols:
        st.warning("Please place 'MW-NIFTY-100.csv' in the folder.")
        return

    if st.session_state.get('scan_active', False):
        with st.spinner("Analyzing Market Structure..."):
            elite_df = run_elite_scan(symbols)
        
        if not elite_df.empty:
            # 2a. TOP PICK METRICS (Fluid Layout)
            # Reduced padding above by CSS
            top = elite_df.iloc[0]
            
            m1, m2, m3 = st.columns([1, 1, 2])
            m1.metric("🔥 Top Pick", f"{top['Stock']}", f"{top['Bias']}")
            m2.metric("📊 Score", f"{top['Conviction']}/100")
            m3.metric("📈 Pattern Detected", top['Pattern'])
            
            # 2b. DATAFRAME (Height increased to 450px to eliminate scrollbar)
            st.markdown("###### 🏆 The Elite Shortlist (Top 10)")
            st.dataframe(
                style_elite_table(elite_df), 
                hide_index=True, 
                use_container_width=True, 
                height=450  # Increased height for 10 rows
            )
            
            # 2c. AI PLAN
            if API_KEY:
                with st.spinner("🤖 AI Strategizing..."):
                    ai_plan = run_elite_ai(API_KEY, elite_df)
                    st.markdown(f"<div class='ai-box'>{ai_plan}</div>", unsafe_allow_html=True)
            else:
                st.info("💡 Add Gemini API Key to config.json for AI Trade Plans")
                
        else:
            st.warning("⚠️ No stocks met the 'Elite' criteria (Score > 65) right now.")
            st.caption("Market might be sideways or low volume. Filters are strict to save capital.")

if __name__ == "__main__":

    main()
