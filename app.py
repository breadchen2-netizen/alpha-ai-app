import streamlit as st
import os
import subprocess
import sys
import time

# ==========================================
# 🔥【暴力修復模組】強制檢查並安裝新版 SDK
# 這是為了確保 Streamlit Cloud 絕對不會用舊版驅動程式
# ==========================================
try:
    import google.generativeai as genai
    from packaging import version
    # 檢查版本是否過舊 (低於 0.5.2 就無法使用 Flash 模型)
    current_ver = getattr(genai, "__version__", "0.0.0")
    if version.parse(current_ver) < version.parse("0.5.2"):
        print(f"⚠️ 偵測到舊版 SDK ({current_ver})，正在強制升級...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "google-generativeai>=0.5.2"])
        import google.generativeai as genai # 重新載入
        print("✅ SDK 更新完成！")
except Exception as e:
    # 如果根本沒安裝或 import 失敗，直接暴力安裝
    print("⚠️ 環境初始化中，正在安裝 AI 驅動程式...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai>=0.5.2"])
    import google.generativeai as genai

# ==========================================
# 📦 標準套件載入
# ==========================================
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import feedparser
import datetime
import numpy as np

# ==========================================
# 🔑【金鑰設定區】
# ==========================================
try:
    GEMINI_API_KEY_GLOBAL = st.secrets["GEMINI_KEY"]
    FINMIND_TOKEN_GLOBAL = st.secrets["FINMIND_TOKEN"]
except:
    GEMINI_API_KEY_GLOBAL = ""
    FINMIND_TOKEN_GLOBAL = ""

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(page_title="Alpha Strategist AI", layout="wide", page_icon="🚀")

# CSS 優化 (黑底風格)
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    header[data-testid="stHeader"] { background-color: #0f172a; }
    h1, h2, h3, h4, h5, h6, span, div, label, p, li { color: #f1f5f9 !important; }
    div[data-testid="stMetricLabel"] p { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] div { color: #38bdf8 !important; }
    section[data-testid="stSidebar"] { background-color: #1e293b; }
    .stTextInput input { background-color: #334155; color: #ffffff; border: 1px solid #475569; }
    button[data-baseweb="tab"] { background-color: transparent !important; color: #94a3b8 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #334155 !important; color: #ffffff !important; }
    div[data-testid="stTable"] { color: white !important; }
    
    /* 角色對話框樣式 */
    .role-box { padding: 15px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid; font-size: 0.95rem; line-height: 1.6; }
    .report-content { background-color: #1e293b; border-color: #3b82f6; color: #e2e8f0; padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Alpha Strategist AI")

# --- 診斷區塊 (可選) ---
with st.expander("🔍 工程師診斷模式：查看 SDK 版本"):
    st.write(f"當前 SDK 版本: {genai.__version__}")
    if GEMINI_API_KEY_GLOBAL:
        st.success("API Key 已載入")
    else:
        st.error("API Key 未載入")

st.markdown("##### ⚡ Powered by Gemini 1.5 Flash | v25.0 戰術合成版")

# ==========================================
# 📊 數據處理函數
# ==========================================
def calculate_indicators(df):
    df['9_High'] = df['High'].rolling(9).max(); df['9_Low'] = df['Low'].rolling(9).min()
    denominator = df['9_High'] - df['9_Low']
    df['RSV'] = np.where(denominator != 0, (df['Close'] - df['9_Low']) / denominator * 100, 50)
    df['K'] = df['RSV'].ewm(com=2).mean(); df['D'] = df['K'].ewm(com=2).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean(); df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']; df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    return df

def calculate_breakout_probs(df, step_percent=1.0):
    df = df.copy()
    df['Prev_Close'] = df['Close'].shift(1); df['Prev_Open'] = df['Open'].shift(1); df['Prev_High'] = df['High'].shift(1); df['Prev_Low'] = df['Low'].shift(1)
    df['Is_Up'] = df['Prev_Close'] > df['Prev_Open']; df['Is_Down'] = df['Prev_Close'] <= df['Prev_Open']
    n = len(df); df['Weight'] = np.linspace(0.1, 1.0, n)
    stats = []
    for i in range(1, 4):
        dist = df['Prev_Close'] * (step_percent * i / 100)
        target_high = df['Prev_High'] + dist; target_low = df['Prev_Low'] - dist
        hit_high = (df['High'] >= target_high).astype(int); hit_low = (df['Low'] <= target_low).astype(int)
        def get_prob(mask_col, hit_series):
            mask = df[mask_col]; valid_hits = hit_series[mask]; valid_weights = df.loc[mask, 'Weight']
            return np.average(valid_hits, weights=valid_weights) * 100 if len(valid_hits) > 0 else 0.0
        stats.append({'Level': i, 'Up_Bull': get_prob('Is_Up', hit_high), 'Down_Bull': get_prob('Is_Up', hit_low), 'Up_Bear': get_prob('Is_Down', hit_high), 'Down_Bear': get_prob('Is_Down', hit_low)})
    return pd.DataFrame(stats)

@st.cache_data(ttl=300)
def get_comprehensive_data(stock_id, days):
    end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=days + 730)
    df_chips = pd.DataFrame()
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": FINMIND_TOKEN_GLOBAL}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200 and "data" in r.json():
            raw_inst = pd.DataFrame(r.json()["data"])
            if not raw_inst.empty:
                foreign = raw_inst[raw_inst['name'] == 'Foreign_Investor'].copy(); foreign['外資'] = foreign['buy'] - foreign['sell']
                trust = raw_inst[raw_inst['name'] == 'Investment_Trust'].copy(); trust['投信'] = trust['buy'] - trust['sell']
                df_chips = pd.merge(foreign[['date', '外資']], trust[['date', '投信']], on='date', how='outer').fillna(0)
    except: pass
    
    try:
        # 加入 threads=False 增加穩定性
        df_price = yf.download(f"{stock_id}.TW", start=start_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True, threads=False)
        if df_price is None or df_price.empty: return None, None, None
        
        if isinstance(df_price.columns, pd.MultiIndex): df_price.columns = df_price.columns.get_level_values(0)
        df_price = df_price.reset_index()
        
        if 'Date' in df_price.columns: df_price['date'] = df_price['Date'].dt.strftime('%Y-%m-%d')
        elif 'date' in df_price.columns: df_price['date'] = pd.to_datetime(df_price['date']).dt.strftime('%Y-%m-%d')
        else: return None, None, None

        df_price['MA5'] = df_price['Close'].rolling(window=5).mean(); df_price['MA20'] = df_price['Close'].rolling(window=20).mean(); df_price['MA60'] = df_price['Close'].rolling(window=60).mean()
        df_price = calculate_indicators(df_price)
    except Exception as e: 
        print(f"Stock Data Error: {e}")
        return None, None, None
        
    df_probs = calculate_breakout_probs(df_price.copy(), 1.0)
    if not df_chips.empty: merged = pd.merge(df_price, df_chips, on='date', how='left').fillna(0)
    else: merged = df_price; merged['外資'] = 0; merged['投信'] = 0
    return merged.tail(days), df_chips, df_probs

def get_finmind_per(stock_id):
    try:
        end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=7)
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockPER", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": FINMIND_TOKEN_GLOBAL}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200 and "data" in r.json():
            data = r.json()["data"]
            if data: return {"P/E": data[-1].get("PER", 0), "Yield": data[-1].get("dividend_yield", 0)}
    except: pass
    return None

def get_fundamentals(stock_id):
    try:
        stock = yf.Ticker(f"{stock_id}.TW")
        info = stock.fast_info
        return {
            "P/E": "N/A", 
            "EPS": "N/A", 
            "Yield": "N/A", 
            "Cap": round(info.market_cap/100000000, 2) if info.market_cap else 'N/A', 
            "Name": stock_id, 
            "Sector": "TW Stock"
        }
    except: return {}

def get_revenue_data(stock_id):
    try:
        end_date = datetime.date.today(); start_date = end_date - datetime.timedelta(days=730)
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockMonthRevenue", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": FINMIND_TOKEN_GLOBAL}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"]); df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=True)
                df['MoM'] = df['revenue'].pct_change() * 100; df['YoY'] = df['revenue'].pct_change(periods=12) * 100
                df = df.sort_values('date', ascending=False).head(12)
                return pd.DataFrame({'期間': df['date'].dt.strftime('%Y-%m'), '營收(億)': round(df['revenue']/100000000, 2), '月增%': df['MoM'].map('{:,.2f}'.format), '年增%': df['YoY'].map('{:,.2f}'.format)})
    except: pass
    return pd.DataFrame()

def get_google_news(stock_id):
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={stock_id}+TW+Stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
        return [{"title": e.title, "url": e.link, "date": f"{e.published_parsed.tm_mon}/{e.published_parsed.tm_mday}"} for e in feed.entries[:6]]
    except: return []

# ==========================================
# 🧠 AI 核心函數 (含快取與合併 Prompt)
# ==========================================
@st.cache_data(ttl=3600) # 🔥 快取 1 小時，省 Quota！
def ask_gemini_combined_strategy(ticker, profile, wargame_on, red_style, data_context):
    """
    將三方會談合併為一次請求，節省 API 呼叫次數。
    """
    if not GEMINI_API_KEY_GLOBAL:
        return "⚠️ 請先設定 Gemini API Key"

    # 定義紅軍角色
    if "Grok" in red_style:
        red_persona = "Grok (馬斯克的 AI)"
        red_tone = "極度理性、科技視角、強調第一性原理，尋找被忽略的系統性風險。"
    else:
        red_persona = "華爾街空頭主力"
        red_tone = "冷血、無情、專找泡沫與估值過高點，用最嚴苛的標準審視。"

    # 合併 Prompt
    prompt = f"""
    你現在是 Alpha Strategist AI。請針對台股 {ticker} 進行一場深度的「兵棋推演」。
    
    【投資人輪廓】：{profile}
    
    【市場情報】：
    {data_context}
    
    請依照以下結構，進行三方辯論與決策，並直接輸出為 Markdown 格式：

    ---
    ### 🔵 第一章：藍軍參謀報告 (基本面與多頭)
    *角色：資深產業分析師，樂觀但有據。*
    * **優勢分析**：從財報、技術面金叉、籌碼集中度分析。
    * **機會點**：未來的催化劑 (Catalyst) 是什麼？
    * **目標價位**：根據斐波那契或技術支撐給出預期。

    ---
    ### 🟣 第二章：紅軍 ({red_persona}) 批判
    *角色：{red_tone}*
    * **盲點戳破**：藍軍忽略了什麼致命風險？(例如：外資大賣、營收衰退、乖離過大)
    * **下檔風險**：最壞情況會跌到哪裡？
    * **靈魂拷問**：給投資人一個尖銳的問題。

    ---
    ### ⚔️ 第三章：總司令最終決策
    *角色：冷靜的操盤手，整合上述觀點。*
    * **戰場定調**：現在是進攻還是防守時刻？
    * **SOP 操作指引**：
        1.  **建倉策略**：(例如：分批 3-3-4，或等待回檔)
        2.  **關鍵點位**：進場價、停損價、停利價。
        3.  **每日任務**：明天開盤該盯什麼？
    """

    try:
        genai.configure(api_key=GEMINI_API_KEY_GLOBAL)
        # 🔥 使用最穩定的 1.5 Flash 模型
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI 思考中斷：{str(e)}"

# ==========================================
# 🖥️ 主介面
# ==========================================
with st.sidebar:
    st.header("⚙️ 戰術設定")
    if GEMINI_API_KEY_GLOBAL: st.success(f"✅ Gemini 金鑰已載入")
    else: st.error("❌ 未偵測到 Gemini Key")
    if FINMIND_TOKEN_GLOBAL: st.success(f"✅ FinMind Token 已載入")
    else: st.warning("⚠️ 未偵測到 FinMind Token")

    st.markdown("---")
    st.subheader("📋 自選監控")
    default_list = ["2330 台積電", "2317 鴻海", "2603 長榮", "2376 技嘉", "3231 緯創", "2454 聯發科"]
    selected_ticker_raw = st.radio("快速切換", default_list)
    target_stock_sidebar = selected_ticker_raw.split(" ")[0]

    st.markdown("---")
    st.subheader("🎯 兵棋推演")
    enable_wargame = st.toggle("啟動「紅藍軍對抗」", value=True)
    if enable_wargame:
        wargame_mode = st.radio("紅軍風格", ["🔴 傳統主力 (理性)", "🟣 Grok 合作 (安全)"], index=1)
    else: wargame_mode = "單一模式"
    
    st.markdown("---")
    strategy_profile = st.radio("投資輪廓", ["穩健價值型", "激進動能型"], index=0)

# --- 主畫面 ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1: 
    manual_input = st.text_input("股票代號", target_stock_sidebar, label_visibility="collapsed")
    target_stock = manual_input if manual_input else target_stock_sidebar
with col2: analysis_days = st.slider("回溯天數", 30, 180, 90, label_visibility="collapsed")
with col3: run_analysis = st.button("🔥 啟動兵棋推演", type="primary", use_container_width=True)

if run_analysis:
    if not GEMINI_API_KEY_GLOBAL: st.error("⛔ 請檢查 Gemini Key")
    else:
        with st.spinner(f"📡 戰情室連線中... 調閱 {target_stock} 全維度數據..."):
            
            # 1. 抓取數據
            df, _, df_probs = get_comprehensive_data(target_stock, analysis_days)
            fundamentals = get_fundamentals(target_stock)
            finmind_per = get_finmind_per(target_stock)
            
            if finmind_per and df is not None and not df.empty:
                current_price = df.iloc[-1]['Close']
                fundamentals['P/E'] = finmind_per['P/E']; fundamentals['Yield'] = finmind_per['Yield']
                if finmind_per['P/E'] > 0: fundamentals['EPS'] = round(current_price / finmind_per['P/E'], 2)
            
            news_list = get_google_news(target_stock)
            df_revenue = get_revenue_data(target_stock)
            
            # 2. 顯示數據儀表板
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
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2], subplot_titles=("價量 & 機率軌道", "法人籌碼", "MACD", "KD"))
                    fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價', increasing_line_color='#ef4444', decreasing_line_color='#10b981'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5', line=dict(color='#fbbf24', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20', line=dict(color='#a855f7', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], name='MA60', line=dict(color='#3b82f6', width=2)), row=1, col=1)
                    
                    # 機率軌道
                    last_close = df.iloc[-1]['Close']; last_high = df.iloc[-1]['High']; last_low = df.iloc[-1]['Low']
                    if df_probs is not None:
                        for i, row_prob in df_probs.iterrows():
                            level = row_prob['Level']; dist = last_close * (1.0 * level / 100); target_up = last_high + dist
                            fig.add_shape(type="line", x0=df['date'].iloc[-5], x1=df['date'].iloc[-1], y0=target_up, y1=target_up, line=dict(color='yellow', width=1, dash="dot"), row=1, col=1)
                            target_down = last_low - dist
                            fig.add_shape(type="line", x0=df['date'].iloc[-5], x1=df['date'].iloc[-1], y0=target_down, y1=target_down, line=dict(color='cyan', width=1, dash="dot"), row=1, col=1)
                    
                    fig.add_trace(go.Bar(x=df['date'], y=df['外資'], name='外資', marker_color='cyan'), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['date'], y=df['投信'], name='投信', marker_color='orange'), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['date'], y=df['MACD_Hist'], name='MACD柱', marker_color=np.where(df['MACD_Hist']<0, 'green', 'red')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], name='DIF', line=dict(color='yellow', width=1)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], name='DEA', line=dict(color='blue', width=1)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K值', line=dict(color='orange', width=1)), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D值', line=dict(color='purple', width=1)), row=4, col=1)
                    fig.update_layout(template='plotly_dark', height=1000, xaxis_rangeslider_visible=False, showlegend=True, paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font=dict(color='#f8fafc'), margin=dict(t=30, b=30, l=60, r=40))
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("")
                    info_tab1, info_tab2, info_tab3 = st.tabs(["📰 新聞", "💰 營收", "🎲 機率表"])
                    with info_tab1:
                        for n in news_list: st.markdown(f"**[{n['title']}]({n.get('url', '#')})**")
                    with info_tab2: st.dataframe(df_revenue, use_container_width=True, hide_index=True)
                    with info_tab3: st.dataframe(df_probs.style.format("{:.1f}%"), use_container_width=True)

                # 3. AI 分析模組
                with ai_col:
                    # 準備數據 Context
                    data_for_ai = df[['date', 'Close', 'MA60', '外資', '投信', 'K', 'D', 'MACD_Hist']].tail(12).to_string(index=False)
                    news_str = "\n".join([f"- {n['title']}" for n in news_list[:5]]) 
                    rev_str = df_revenue.head(6).to_string() if not df_revenue.empty else "無"
                    
                    full_context = f"""
                    【技術指標】：\n{data_for_ai}
                    【基本面】：P/E {fundamentals.get('P/E')}, 殖利率 {fundamentals.get('Yield')}%
                    【近期營收】：\n{rev_str}
                    【新聞焦點】：\n{news_str}
                    """

                    st.subheader("⚔️ 戰情推演報告")
                    
                    # 呼叫 AI (這裡會用到快取，第二次點擊不扣額度)
                    with st.status("🧠 戰情室運算中 (整合分析)...", expanded=True):
                        ai_report = ask_gemini_combined_strategy(target_stock, strategy_profile, enable_wargame, wargame_mode, full_context)
                        
                        # 顯示結果
                        st.markdown(f"<div class='report-content'>{ai_report}</div>", unsafe_allow_html=True)
                        
                        # 下載按鈕
                        st.download_button(
                            label="💾 下載完整戰報 (Markdown)",
                            data=f"# {target_stock} 深度戰報\n{datetime.date.today()}\n\n{ai_report}",
                            file_name=f"{target_stock}_report.md",
                            mime="text/markdown"
                        )

            else: st.error("⚠️ 查無數據，請確認股票代號是否正確。")
