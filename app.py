import streamlit as st
import pandas as pd
import yfinance as yf
import google.generativeai as genai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests 
import feedparser
import datetime
import numpy as np
import time
import os

# ==========================================
# 🔑【金鑰設定區】
try:
    GEMINI_API_KEY_GLOBAL = st.secrets["GEMINI_KEY"]
    FINMIND_TOKEN_GLOBAL = st.secrets["FINMIND_TOKEN"]
except:
    # 如果找不到保險箱(例如第一次在本地跑)，先給空值避免報錯
    GEMINI_API_KEY_GLOBAL = ""
    FINMIND_TOKEN_GLOBAL = ""
# ==========================================

st.set_page_config(page_title="Alpha Strategist AI", layout="wide", page_icon="🚀")

# CSS 優化
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    header[data-testid="stHeader"] { background-color: #0f172a; }
    h1, h2, h3, h4, h5, h6, span, div, label, p, li { color: #f1f5f9 !important; }
    div[data-testid="stMetricLabel"] p { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] div { color: #38bdf8 !important; }
    section[data-testid="stSidebar"] { background-color: #1e293b; }
    .stTextInput input, .stTextArea textarea { background-color: #334155; color: #ffffff; border: 1px solid #475569; }
    button[data-baseweb="tab"] { background-color: transparent !important; color: #94a3b8 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #334155 !important; color: #ffffff !important; }
    div[data-testid="stTable"] { color: white !important; }
    thead tr th { background-color: #1e293b !important; color: #38bdf8 !important; }
    
    .role-box { padding: 15px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid; font-size: 0.95rem; line-height: 1.6; }
    .blue-team { background-color: #1e293b; border-color: #3b82f6; color: #e2e8f0; }
    .grok-synergy { background-color: #2e1065; border-color: #a855f7; color: #e9d5ff; font-family: 'Segoe UI', sans-serif; }
    .red-team { background-color: #3f1818; border-color: #ef4444; color: #fecaca; }
    .commander { background-color: #143328; border-color: #10b981; color: #d1fae5; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Alpha Strategist AI")
st.markdown("##### ⚡ Powered by Gemini Auto-Adapt | v19.2 模型自適應版")

# 🔥 全域變數初始化
target_stock_sidebar = "2330"
target_stock = "2330"
enable_wargame = False
wargame_mode = "單一模式"
scanner_list = "2330 2317 2454 2603 2376 3231"
valid_gemini = "".join(GEMINI_API_KEY_GLOBAL.split())
valid_finmind = "".join(FINMIND_TOKEN_GLOBAL.split())

# 🔥 新增：自動尋找最佳模型
@st.cache_resource
def get_best_model_name(api_key):
    try:
        genai.configure(api_key=api_key)
        # 列出所有可用模型
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1. 優先找 2.5 或 2.0 Flash (速度快、省額度)
        for m in models:
            if 'flash' in m.lower() and 'legacy' not in m.lower() and ('2.5' in m or '2.0' in m):
                return m
        
        # 2. 其次找任何 Flash
        for m in models:
            if 'flash' in m.lower() and 'legacy' not in m.lower():
                return m
        
        # 3. 找 Pro
        for m in models:
            if 'pro' in m.lower() and 'legacy' not in m.lower():
                return m
        
        # 4. 如果都找不到，回傳第一個可用的
        if models: return models[0]
    except:
        pass
    return "gemini-1.5-flash" # 最終保底 (如果 API 連線失敗)

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 戰術設定")
    
    if valid_gemini: 
        st.success("✅ Gemini 金鑰鎖定")
        # 自動偵測模型並顯示
        best_model = get_best_model_name(valid_gemini)
        st.caption(f"🤖 目前使用模型：`{best_model}`")
    else: st.error("❌ 缺 Gemini Key")
    
    if valid_finmind: st.success("✅ FinMind Token 鎖定")
    else: st.warning("⚠️ 缺 FinMind Token")

    st.markdown("---")
    app_mode = st.radio("📡 戰術模式", ["🎯 單兵作戰 (深度分析)", "📡 戰情雷達 (多股掃描)"])

    st.markdown("---")
    if app_mode == "🎯 單兵作戰 (深度分析)":
        st.subheader("📋 自選監控")
        default_list = ["2330 台積電", "2317 鴻海", "2603 長榮", "2376 技嘉", "3231 緯創", "2454 聯發科"]
        selected_ticker_raw = st.radio("快速切換", default_list)
        target_stock_sidebar = selected_ticker_raw.split(" ")[0]
        
        st.subheader("🎯 兵棋推演")
        enable_wargame = st.toggle("啟動「紅藍軍對抗」", value=True)
        if enable_wargame:
            wargame_mode = st.radio("紅軍風格", ["🔴 傳統主力 (理性)", "🟣 Grok 合作 (安全)"], index=1)
    else:
        st.subheader("📡 掃描清單")
        scanner_list = st.text_area("輸入代號 (空白隔開)", scanner_list)
        st.caption("AI 將會批次掃描並評比這些股票。")

    st.markdown("---")
    strategy_profile = st.radio("投資輪廓", ["穩健價值型", "激進動能型"], index=0)

# --- 工具函數 (保留 v19.1 的優化) ---

def safe_api_call(url, params, max_retries=2):
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code == 200: return r.json()
            elif r.status_code == 429: time.sleep(1); continue
        except: time.sleep(1)
    return None

def calculate_indicators(df):
    if df.empty or len(df) < 60: return df
    df = df.copy()
    df['9_High'] = df['High'].rolling(9).max(); df['9_Low'] = df['Low'].rolling(9).min()
    denominator = df['9_High'] - df['9_Low']
    df['RSV'] = np.where(denominator != 0, (df['Close'] - df['9_Low']) / denominator * 100, 50)
    df['K'] = df['RSV'].ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean(); df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']; df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    return df

def calculate_breakout_probs(df, step_percent=1.0):
    if df.empty: return None
    df = df.copy()
    df['Prev_Close'] = df['Close'].shift(1); df['Prev_Open'] = df['Open'].shift(1); df['Prev_High'] = df['High'].shift(1); df['Prev_Low'] = df['Low'].shift(1)
    df['Is_Up'] = df['Prev_Close'] > df['Prev_Open']; df['Is_Down'] = df['Prev_Close'] <= df['Prev_Open']
    n = len(df); df['Weight'] = np.exp(np.linspace(-2, 0, n))
    stats = []
    for i in range(1, 4):
        dist = df['Prev_Close'] * (step_percent * i / 100)
        hit_high = (df['High'] >= df['Prev_High'] + dist).astype(int); hit_low = (df['Low'] <= df['Prev_Low'] - dist).astype(int)
        def get_prob(mask_col, hit_series):
            mask = df[mask_col]; valid_hits = hit_series[mask]; valid_weights = df.loc[mask, 'Weight']
            return np.average(valid_hits, weights=valid_weights) * 100 if len(valid_hits) > 0 else 0.0
        stats.append({'Level': i, 'Up_Bull': get_prob('Is_Up', hit_high), 'Down_Bull': get_prob('Is_Up', hit_low), 'Up_Bear': get_prob('Is_Down', hit_high), 'Down_Bear': get_prob('Is_Down', hit_low)})
    return pd.DataFrame(stats)

@st.cache_data(ttl=300) 
def get_technical_chips(stock_id, days):
    end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=days + 150)
    df_chips = pd.DataFrame()
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": valid_finmind}
        data = safe_api_call(url, params)
        if data and "data" in data:
            raw_inst = pd.DataFrame(data["data"])
            if not raw_inst.empty:
                foreign = raw_inst[raw_inst['name'] == 'Foreign_Investor'].copy(); foreign['外資'] = foreign['buy'] - foreign['sell']
                trust = raw_inst[raw_inst['name'] == 'Investment_Trust'].copy(); trust['投信'] = trust['buy'] - trust['sell']
                df_chips = pd.merge(foreign[['date', '外資']], trust[['date', '投信']], on='date', how='outer').fillna(0)
    except: pass

    try:
        df_price = yf.download(f"{stock_id}.TW", start=start_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
        if df_price.empty: return None, None, None
        if isinstance(df_price.columns, pd.MultiIndex): df_price.columns = df_price.columns.get_level_values(0)
        df_price = df_price.reset_index(); df_price['date'] = df_price['Date'].dt.strftime('%Y-%m-%d')
        if len(df_price) < 60: return None, None, None
        df_price['MA5'] = df_price['Close'].rolling(window=5).mean(); df_price['MA20'] = df_price['Close'].rolling(window=20).mean(); df_price['MA60'] = df_price['Close'].rolling(window=60).mean()
        df_price = calculate_indicators(df_price)
    except: return None, None, None

    df_probs = calculate_breakout_probs(df_price.copy(), 1.0)
    if not df_chips.empty: merged = pd.merge(df_price, df_chips, on='date', how='left').fillna(0)
    else: merged = df_price; merged['外資'] = 0; merged['投信'] = 0
    return merged.tail(days), df_chips, df_probs

@st.cache_data(ttl=3600)
def get_finmind_per(stock_id):
    url = "https://api.finmindtrade.com/api/v4/data"
    end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=14)
    params = {"dataset": "TaiwanStockPER", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": valid_finmind}
    data = safe_api_call(url, params)
    if data and "data" in data and data["data"]: return {"P/E": data["data"][-1].get("PER", 0), "Yield": data["data"][-1].get("dividend_yield", 0)}
    return None

def get_fundamentals(stock_id):
    try:
        stock = yf.Ticker(f"{stock_id}.TW"); info = stock.info
        return {"P/E": info.get('trailingPE', 'N/A'), "EPS": info.get('trailingEps', 'N/A'), "Yield": info.get('dividendYield', 'N/A'), "Cap": info.get('marketCap', 'N/A'), "Name": info.get('longName', stock_id)}
    except: return {}

@st.cache_data(ttl=3600)
def get_revenue_data(stock_id):
    try:
        url = "https://api.finmindtrade.com/api/v4/data"; end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=730)
        params = {"dataset": "TaiwanStockMonthRevenue", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": valid_finmind}
        data = safe_api_call(url, params)
        if data and "data" in data:
            df = pd.DataFrame(data["data"]); df['date'] = pd.to_datetime(df['date']); df = df.sort_values('date', ascending=False).head(12)
            return pd.DataFrame({'期間': df['date'].dt.strftime('%Y-%m'), '營收': round(df['revenue']/100000000, 2)})
    except: return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_google_news(stock_id):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={stock_id}+TW+Stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
        return [{"title": e.title, "url": e.link} for e in feed.entries[:6]]
    except: return []

def compress_data_for_ai(df, max_rows=15):
    if len(df) <= max_rows: return df.to_string(index=False)
    return df.tail(max_rows).to_string(index=False)

def save_report_to_md(stock_id, price, content):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"# {stock_id} 策略研報\n- **日期：** {date_str}\n- **收盤價：** {price}\n\n---\n## AI 決策摘要\n{content}\n\n---\n*Created by Alpha Strategist AI*"

# --- 批次掃描 ---
def run_batch_scan(ticker_list, model_name):
    summary_data = []
    progress_bar = st.progress(0); status_text = st.empty()
    tickers = [t.strip() for t in ticker_list.replace(',', ' ').split(' ') if t.strip()]
    total = len(tickers)
    
    for i, stock_id in enumerate(tickers):
        status_text.text(f"📡 正在掃描 {stock_id} ... ({i+1}/{total})")
        try:
            df, _, _ = get_technical_chips(stock_id, 60)
            finmind_per = get_finmind_per(stock_id)
            if df is not None and not df.empty:
                last = df.iloc[-1]
                trend = "🟢 多頭" if last['Close'] > last['MA60'] else "🔴 空頭"
                if last['Close'] < last['MA20']: trend = "⚪ 整理"
                chips_sum = df['外資'].tail(5).sum() if '外資' in df.columns else 0
                chips_status = "🔥 外資買" if chips_sum > 2000 else ("🧊 外資賣" if chips_sum < -2000 else "➖ 觀望")
                pe = finmind_per['P/E'] if finmind_per else "N/A"
                summary_data.append({"代號": stock_id, "收盤": last['Close'], "趨勢": trend, "籌碼": chips_status, "P/E": pe})
        except: pass
        progress_bar.progress((i + 1) / total)
        time.sleep(0.5) 
    status_text.empty(); progress_bar.empty()
    return pd.DataFrame(summary_data)

# --- 主介面 ---
if app_mode == "🎯 單兵作戰 (深度分析)":
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1: 
        manual_input = st.text_input("股票代號", target_stock_sidebar, label_visibility="collapsed")
        target_stock = manual_input if manual_input else target_stock_sidebar
    with col2: analysis_days = st.slider("回溯天數", 30, 180, 90, label_visibility="collapsed")
    with col3: run_analysis = st.button("🔥 啟動兵棋推演", type="primary", use_container_width=True)

    if run_analysis:
        if not valid_gemini: st.error("⛔ 請檢查 Gemini Key")
        else:
            with st.spinner(f"📡 戰情室連線中..."):
                df, _, df_probs = get_technical_chips(target_stock, analysis_days)
                fundamentals = get_fundamentals(target_stock)
                finmind_per = get_finmind_per(target_stock)
                
                if finmind_per and df is not None:
                    current_price = df.iloc[-1]['Close']
                    fundamentals['P/E'] = finmind_per['P/E']; fundamentals['Yield'] = finmind_per['Yield']
                    if finmind_per['P/E'] > 0: fundamentals['EPS'] = round(current_price / finmind_per['P/E'], 2)
                
                news_list = get_google_news(target_stock)
                df_revenue = get_revenue_data(target_stock)
                
                if df is not None and not df.empty:
                    st.markdown("---")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("名稱", fundamentals.get("Name", target_stock))
                    m2.metric("P/E", fundamentals.get("P/E"))
                    m3.metric("EPS", fundamentals.get("EPS"))
                    m4.metric("殖利率", f"{fundamentals.get('Yield')}%")
                    m5.metric("市值(億)", f"{fundamentals.get('Cap')}")
                    st.markdown("---")

                    chart_col, ai_col = st.columns([2, 1])
                    with chart_col:
                        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2])
                        fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], name='MA60', line=dict(color='blue')), row=1, col=1)
                        
                        if df_probs is not None:
                            last_c = df.iloc[-1]['Close']
                            for i, row in df_probs.iterrows():
                                target = last_c * (1 + row['Level']/100)
                                fig.add_hline(y=target, line_dash="dot", line_color="yellow", row=1, col=1)

                        if '外資' in df.columns: fig.add_trace(go.Bar(x=df['date'], y=df['外資'], name='外資', marker_color='cyan'), row=2, col=1)
                        if '投信' in df.columns: fig.add_trace(go.Bar(x=df['date'], y=df['投信'], name='投信', marker_color='orange'), row=2, col=1)
                        fig.add_trace(go.Bar(x=df['date'], y=df['MACD_Hist'], name='MACD', marker_color='red'), row=3, col=1)
                        fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K', line=dict(color='orange')), row=4, col=1)
                        fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D', line=dict(color='purple')), row=4, col=1)
                        fig.update_layout(template='plotly_dark', height=800, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                        st.plotly_chart(fig, use_container_width=True)

                    with ai_col:
                        data_for_ai = compress_data_for_ai(df)
                        news_str = "\n".join([f"- {n['title']}" for n in news_list[:5]]) 
                        
                        prompt = f"分析 {target_stock}。\n數據：{data_for_ai}\n新聞：{news_str}\n請給出操作建議。"
                        try:
                            genai.configure(api_key=valid_gemini)
                            # 🔥 使用自動偵測到的最佳模型
                            model = genai.GenerativeModel(best_model)
                            
                            with st.status(f"🧠 AI 思考中 ({best_model})..."):
                                response = model.generate_content(prompt)
                                st.markdown(response.text)
                                st.download_button("💾 下載報告", response.text, file_name="report.md")
                                
                        except Exception as e: st.error(f"AI Error: {e}")
                else: st.error("⚠️ 查無數據")

else:
    # 戰情雷達
    st.subheader("📡 板塊戰情雷達")
    col1, col2 = st.columns([3, 1])
    with col1: run_scan = st.button("🚀 啟動全域掃描", type="primary", use_container_width=True)
    if run_scan:
        with st.spinner("📡 掃描中..."):
            res = run_batch_scan(scanner_list, best_model) # 傳入模型名稱
            if not res.empty:
                st.dataframe(res, use_container_width=True)
                try:
                    genai.configure(api_key=valid_gemini)
                    model = genai.GenerativeModel(best_model)
                    prompt = f"評比這些股票：\n{res.to_string()}\n選出 MVP 和 危險名單。"
                    resp = model.generate_content(prompt)
                    st.markdown(f"<div class='role-box commander'>{resp.text}</div>", unsafe_allow_html=True)
                except: st.warning("AI 額度不足，無法評比")
