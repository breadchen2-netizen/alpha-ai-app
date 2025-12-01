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

# ==========================================
# 🔑【金鑰設定區 - 混合安全版】
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
    .stTextInput input { background-color: #334155; color: #ffffff; border: 1px solid #475569; }
    button[data-baseweb="tab"] { background-color: transparent !important; color: #94a3b8 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #334155 !important; color: #ffffff !important; }
    div[data-testid="stTable"] { color: white !important; }
    thead tr th { background-color: #1e293b !important; color: #38bdf8 !important; }
    
    /* 兵推對話框 */
    .role-box { padding: 15px; border-radius: 8px; margin-bottom: 12px; border-left: 5px solid; font-size: 0.95rem; line-height: 1.6; }
    .blue-team { background-color: #1e293b; border-color: #3b82f6; color: #e2e8f0; }
    .grok-synergy { background-color: #2e1065; border-color: #a855f7; color: #e9d5ff; font-family: 'Segoe UI', sans-serif; }
    .red-team { background-color: #3f1818; border-color: #ef4444; color: #fecaca; }
    .commander { background-color: #143328; border-color: #10b981; color: #d1fae5; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Alpha Strategist AI")
st.markdown("##### ⚡ Powered by Gemini 2.5 Pro | v15.1 精準財報修復版")

# --- 側邊欄 ---
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
    st.subheader("🎯 兵棋推演模式")
    
    enable_wargame = st.toggle("啟動「紅藍軍對抗」", value=True)
    
    if enable_wargame:
        wargame_mode = st.radio("選擇紅軍風格", ["🔴 傳統主力 (理性博弈)", "🟣 Grok 合作模式 (安全獲利)"], index=1)
    else:
        wargame_mode = "單一模式"
    
    st.markdown("---")
    strategy_profile = st.radio("您的投資輪廓 (藍軍)", ["穩健價值型 (巴菲特)", "激進動能型 (李佛摩)"], index=0)

# --- 數據函數 ---

def calculate_indicators(df):
    df['9_High'] = df['High'].rolling(9).max()
    df['9_Low'] = df['Low'].rolling(9).min()
    df['RSV'] = (df['Close'] - df['9_Low']) / (df['9_High'] - df['9_Low']) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = (df['DIF'] - df['DEA']) * 2
    return df

def calculate_breakout_probs(df, step_percent=1.0):
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Is_Up'] = df['Prev_Close'] > df['Prev_Open']
    df['Is_Down'] = df['Prev_Close'] <= df['Prev_Open']
    n = len(df)
    df['Weight'] = np.linspace(0.1, 1.0, n)
    stats = []
    for i in range(1, 4):
        dist = df['Prev_Close'] * (step_percent * i / 100)
        target_high = df['Prev_High'] + dist
        target_low = df['Prev_Low'] - dist
        hit_high = (df['High'] >= target_high).astype(int)
        hit_low = (df['Low'] <= target_low).astype(int)
        def get_prob(mask_col, hit_series):
            mask = df[mask_col]
            valid_hits = hit_series[mask]
            valid_weights = df.loc[mask, 'Weight']
            if len(valid_hits) == 0: return 0.0
            return np.average(valid_hits, weights=valid_weights) * 100
        stats.append({'Level': i, 'Up_Bull': get_prob('Is_Up', hit_high), 'Down_Bull': get_prob('Is_Up', hit_low), 'Up_Bear': get_prob('Is_Down', hit_high), 'Down_Bear': get_prob('Is_Down', hit_low)})
    return pd.DataFrame(stats)

def get_comprehensive_data(stock_id, days):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days + 730)
    df_chips = pd.DataFrame()
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        token = "".join(FINMIND_TOKEN_GLOBAL.split())
        params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": token}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200 and "data" in r.json():
            raw_inst = pd.DataFrame(r.json()["data"])
            if not raw_inst.empty:
                foreign = raw_inst[raw_inst['name'] == 'Foreign_Investor'].copy()
                foreign['外資'] = foreign['buy'] - foreign['sell']
                trust = raw_inst[raw_inst['name'] == 'Investment_Trust'].copy()
                trust['投信'] = trust['buy'] - trust['sell']
                df_chips = pd.merge(foreign[['date', '外資']], trust[['date', '投信']], on='date', how='outer').fillna(0)
    except Exception: pass

    try:
        df_price = yf.download(f"{stock_id}.TW", start=start_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
        if isinstance(df_price.columns, pd.MultiIndex): df_price.columns = df_price.columns.get_level_values(0)
        df_price = df_price.reset_index()
        df_price['date'] = df_price['Date'].dt.strftime('%Y-%m-%d')
        df_price['MA5'] = df_price['Close'].rolling(window=5).mean()
        df_price['MA20'] = df_price['Close'].rolling(window=20).mean()
        df_price['MA60'] = df_price['Close'].rolling(window=60).mean()
        df_price = calculate_indicators(df_price)
    except: return None, None, None

    df_probs = calculate_breakout_probs(df_price.copy(), 1.0)

    if not df_chips.empty:
        merged = pd.merge(df_price, df_chips, on='date', how='left').fillna(0)
    else:
        merged = df_price
        merged['外資'] = 0
        merged['投信'] = 0
    return merged.tail(days), df_chips, df_probs

# 🔥 修正：恢復成只接受 stock_id
def get_fundamentals(stock_id):
    try:
        stock = yf.Ticker(f"{stock_id}.TW")
        info = stock.info
        # 先抓 Yahoo 的數據當備用
        raw_yield = info.get('dividendYield', 0)
        fmt_yield = round(raw_yield * 100, 2) if raw_yield and raw_yield < 1 else (round(raw_yield, 2) if raw_yield else 'N/A')
        pe = round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A'
        eps = round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else 'N/A'
        
        return {
            "P/E": pe, "EPS": eps, "Yield": fmt_yield, 
            "Cap": round(info.get('marketCap', 0)/100000000, 2) if info.get('marketCap') else 'N/A', 
            "Name": info.get('longName', stock_id), 
            "Sector": info.get('sector', 'N/A'), 
            "Summary": info.get('longBusinessSummary', '暫無描述')
        }
    except: return {}

# 🔥 新增：FinMind PER 查詢函數
def get_finmind_per(stock_id):
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=7)
        url = "https://api.finmindtrade.com/api/v4/data"
        token = "".join(FINMIND_TOKEN_GLOBAL.split())
        params = {
            "dataset": "TaiwanStockPER",
            "data_id": stock_id,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "token": token
        }
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200 and "data" in r.json():
            data = r.json()["data"]
            if data:
                latest = data[-1] # 取最新一筆
                return {
                    "P/E": latest.get("PER", 0),
                    "Yield": latest.get("dividend_yield", 0),
                    "P/B": latest.get("PBR", 0)
                }
    except: pass
    return None

def get_revenue_data(stock_id):
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=730)
        url = "https://api.finmindtrade.com/api/v4/data"
        token = "".join(FINMIND_TOKEN_GLOBAL.split())
        params = {"dataset": "TaiwanStockMonthRevenue", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": token}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if "data" in data and data["data"]:
                df = pd.DataFrame(data["data"])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=True)
                df['MoM'] = df['revenue'].pct_change() * 100
                df['YoY'] = df['revenue'].pct_change(periods=12) * 100
                df = df.sort_values('date', ascending=False).head(12)
                return pd.DataFrame({'期間': df['date'].dt.strftime('%Y-%m'), '營收(億)': round(df['revenue']/100000000, 2), '月增%': df['MoM'].map('{:,.2f}'.format), '年增%': df['YoY'].map('{:,.2f}'.format), '來源': 'FinMind'})
    except: pass
    try:
        stock = yf.Ticker(f"{stock_id}.TW")
        rev = stock.quarterly_financials.loc['Total Revenue'].sort_index()
        df_y = pd.DataFrame({'revenue': rev})
        df_y['qoq'] = df_y['revenue'].pct_change() * 100
        df_y['yoy'] = df_y['revenue'].pct_change(periods=4) * 100
        df_y = df_y.sort_index(ascending=False).head(4)
        return pd.DataFrame({'期間': df_y.index.strftime('%Y-%m'), '營收(億)': round(df_y['revenue']/100000000, 2), '月增%': df_y['qoq'].map('{:,.2f}'.format), '年增%': df_y['yoy'].map('{:,.2f}'.format), '來源': 'Yahoo (季)'})
    except: return pd.DataFrame()

def get_google_news(stock_id):
    try:
        from duckduckgo_search import DDGS
        results = DDGS().news(keywords=f"{stock_id} 台股 營收 展望", region="wt-wt", safesearch="off", max_results=6)
        return results if results else []
    except:
        try:
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={stock_id}+TW+Stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
            return [{"title": e.title, "url": e.link, "date": "近期"} for e in feed.entries[:6]]
        except: return []

# --- 主介面 ---
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
            
            # 1. 抓股價與籌碼
            df, _, df_probs = get_comprehensive_data(target_stock, analysis_days)
            
            # 2. 抓基本面 (先用 Yahoo)
            fundamentals = get_fundamentals(target_stock)
            
            # 🔥 3. 抓 FinMind 官方 P/E 並更新 (這是在主程式執行，不會有參數錯誤)
            finmind_per = get_finmind_per(target_stock)
            if finmind_per and df is not None and not df.empty:
                current_price = df.iloc[-1]['Close']
                # 用 FinMind 數據覆蓋 Yahoo
                fundamentals['P/E'] = finmind_per['P/E']
                fundamentals['Yield'] = finmind_per['Yield']
                # 反推 EPS
                if finmind_per['P/E'] > 0:
                    fundamentals['EPS'] = round(current_price / finmind_per['P/E'], 2)
            
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
                    fig = make_subplots(
                        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2], 
                        subplot_titles=("價量 & 機率軌道", "法人籌碼", "MACD", "KD")
                    )
                    # K線
                    fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價', increasing_line_color='#ef4444', decreasing_line_color='#10b981'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5', line=dict(color='#fbbf24', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20', line=dict(color='#a855f7', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], name='MA60', line=dict(color='#3b82f6', width=2)), row=1, col=1)
                    
                    last_close = df.iloc[-1]['Close']; last_high = df.iloc[-1]['High']; last_low = df.iloc[-1]['Low']
                    is_last_up = last_close > df.iloc[-1]['Open']
                    prob_col_up = 'Up_Bull' if is_last_up else 'Up_Bear'
                    prob_col_down = 'Down_Bull' if is_last_up else 'Down_Bear'
                    if df_probs is not None:
                        for i, row_prob in df_probs.iterrows():
                            level = row_prob['Level']; dist = last_close * (1.0 * level / 100)
                            target_up = last_high + dist; prob_up = row_prob[prob_col_up]
                            fig.add_shape(type="line", x0=df['date'].iloc[-5], x1=df['date'].iloc[-1], y0=target_up, y1=target_up, line=dict(color='yellow', width=1, dash="dot"), row=1, col=1)
                            fig.add_annotation(x=df['date'].iloc[-1], y=target_up, text=f"L{level} ({prob_up:.0f}%)", showarrow=False, xanchor="left", font=dict(color="yellow", size=10), row=1, col=1)
                            target_down = last_low - dist; prob_down = row_prob[prob_col_down]
                            fig.add_shape(type="line", x0=df['date'].iloc[-5], x1=df['date'].iloc[-1], y0=target_down, y1=target_down, line=dict(color='cyan', width=1, dash="dot"), row=1, col=1)
                            fig.add_annotation(x=df['date'].iloc[-1], y=target_down, text=f"L{level} ({prob_down:.0f}%)", showarrow=False, xanchor="left", font=dict(color="cyan", size=10), row=1, col=1)

                    fig.add_trace(go.Bar(x=df['date'], y=df['外資'], name='外資', marker_color='cyan'), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['date'], y=df['投信'], name='投信', marker_color='orange'), row=2, col=1)

                    fig.add_trace(go.Bar(x=df['date'], y=df['MACD_Hist'], name='MACD柱', marker_color=np.where(df['MACD_Hist']<0, 'green', 'red')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], name='DIF', line=dict(color='yellow', width=1)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], name='DEA', line=dict(color='blue', width=1)), row=3, col=1)

                    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K值', line=dict(color='orange', width=1)), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D值', line=dict(color='purple', width=1)), row=4, col=1)
                    fig.add_hline(y=80, line_dash="dot", row=4, col=1, line_color="gray"); fig.add_hline(y=20, line_dash="dot", row=4, col=1, line_color="gray")

                    fig.update_layout(template='plotly_dark', height=1000, xaxis_rangeslider_visible=False, showlegend=True, paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font=dict(color='#f8fafc', size=12), legend=dict(orientation="h", y=1.01, x=0, font=dict(color="#f8fafc"), bgcolor="rgba(0,0,0,0.5)"), margin=dict(t=30, b=30, l=60, r=40))
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("")
                    info_tab1, info_tab2, info_tab3 = st.tabs(["📰 新聞", "💰 營收", "🎲 機率表"])
                    with info_tab1:
                        for n in news_list: st.markdown(f"**[{n['title']}]({n.get('url', '#')})**")
                    with info_tab2: st.dataframe(df_revenue, use_container_width=True, hide_index=True)
                    with info_tab3: st.dataframe(df_probs.style.format("{:.1f}%"), use_container_width=True)

                with ai_col:
                    data_for_ai = df[['date', 'Close', 'MA60', '外資', '投信', 'K', 'D', 'MACD_Hist']].tail(12).to_string(index=False)
                    news_str = "\n".join([f"- {n['title']}" for n in news_list[:8]]) 
                    rev_str = df_revenue.head(6).to_string() if not df_revenue.empty else "無"
                    
                    if "穩健" in strategy_profile:
                        investor_profile = "基本面驅動的戰術型投資人。策略：左側低接，重視估值與安全邊際。"
                    else:
                        investor_profile = "動能驅動的交易型投資人。策略：右側追價，重視量能與趨勢。"

                    prompt_blue = f"""
                    你現在是 Alpha Strategist AI (v6.4 深度復刻版)。
                    你的任務是執行【七大核心模組】分析，為 {target_stock} 撰寫一份深度研報。

                    **預載投資者輪廓：**
                    {investor_profile}

                    **【輸入情報】**
                    1. 技術籌碼：\n{data_for_ai}
                    2. 基本面 (P/E, EPS, 殖利率)：{fundamentals}
                    3. 營收趨勢：\n{rev_str}
                    4. 宏觀/新聞：\n{news_str}

                    **請依照以下架構輸出報告 (Markdown)：**

                    ### 1. 🔍 基本面與宏觀掃描 (Fundamental Scan)
                    * **估值評估：** P/E ({fundamentals.get('P/E')}) 與 EPS 相比，股價是便宜還是貴？(請參考歷史經驗)
                    * **營收動能：** 近期營收是成長還是衰退？(引用數據)
                    * **宏觀/新聞解讀：** 新聞標題透露了什麼產業趨勢？(利多/利空/雜訊)

                    ### 2. ⚖️ 技術與籌碼診斷 (Tech & Chips)
                    * **趨勢判讀：** 目前股價在季線 (MA60) 之上還是之下？均線排列為何？
                    * **籌碼意圖：** 外資與投信是在「吃貨」、「倒貨」還是「觀望」？(請引用買賣超張數)
                    * **指標訊號：** KD 與 MACD 是否出現背離或黃金/死亡交叉？

                    ### 3. 🎲 風險與情境 (Risk & Scenarios)
                    * **主要風險：** * **情境推演：** 若股價跌破關鍵支撐，下檔看哪裡？若突破壓力，目標看哪裡？

                    ### 4. 🚀 戰略合成 (Strategy)
                    * **操作建議：** 基於投資者輪廓，現在該做什麼？(買進/觀望/賣出)
                    * **防守點位：** (必填) 給出明確的止損價位。
                    """

                    try:
                        genai.configure(api_key=GEMINI_API_KEY_GLOBAL)
                        model = genai.GenerativeModel('models/gemini-2.5-pro')
                        
                        if enable_wargame:
                            with st.status("🔵 藍軍參謀：分析理想面...", expanded=True) as status:
                                response_analyst = model.generate_content(prompt_blue).text
                                st.markdown(f"<div class='role-box blue-team'>{response_analyst}</div>", unsafe_allow_html=True)
                                status.update(label="✅ 藍軍報告完成", state="complete", expanded=False)
                                time.sleep(1)

                            if "Grok" in wargame_mode:
                                red_class = "grok-synergy" # 紫色合作風格
                                red_persona = "Grok (合作戰友)"
                                red_mission = """
                                你是 xAI 的 Grok，但這次你是站在使用者這邊的「超級軍師」。
                                你的風格：【機智、反骨、但極度實用】。
                                你的任務：
                                1. 承認藍軍的基本面分析有道理，但指出市場的「非理性風險」。
                                2. 提出「三步安全獲利藍圖」：(1) 觀望與觸發條件 (2) 分批進場點 (3) 紀律出場點。
                                3. 確保方案是「低風險、高勝率」的，目標年化報酬 10-15%。
                                """
                            else:
                                red_class = "red-team"
                                red_persona = "主力操盤手"
                                red_mission = "無情批判藍軍盲點，提出如何製造假突破或假跌破來修理散戶的劇本。"

                            with st.status(f"🟣 紅軍 ({red_persona})：擬定獲利藍圖...", expanded=True) as status:
                                prompt_predator = f"""
                                角色：{red_persona}。
                                任務：{red_mission}
                                藍軍觀點：{response_analyst}
                                數據：\n{data_for_ai}
                                """
                                response_predator = model.generate_content(prompt_predator).text
                                st.markdown(f"<div class='role-box {red_class}'>{response_predator}</div>", unsafe_allow_html=True)
                                status.update(label="✅ 紅軍策略完成", state="complete", expanded=False)
                                time.sleep(1)

                            st.subheader("⚔️ 總司令決策")
                            with st.spinner("🧠 綜合推演中..."):
                                prompt_commander = f"""
                                角色：Alpha Strategist 總司令。
                                藍軍(理性)：{response_analyst}
                                紅軍(實戰)：{response_predator}
                                
                                請整合兩者觀點，並參考 Grok 的實戰風格，給出最終的「傻瓜執行清單」。
                                
                                請嚴格依照以下格式輸出 (Markdown)：
                                ### 1. 🛡️ 戰場動態 (Risk Level 0-10)
                                * (一句話定調目前的風險程度)
                                
                                ### 2. 🦅 每日看盤 SOP (10秒檢查法)
                                * **A. 價格訊號：** (例如：收盤是否站上 xxx 元？)
                                * **B. 籌碼訊號：** (例如：投信賣超是否縮減至 xxx 張？)
                                * **C. 量能訊號：** (例如：成交量是否大於 xxx 萬張？)
                                * **行動：** 若滿足則...；若未滿足則...
                                
                                ### 3. 🎯 預掛單設定 (Set & Forget)
                                * **第一批進場：** 價格 xxx，資金 %
                                * **加碼條件：** 價格 xxx
                                * **鐵律停損單：** 觸發價 xxx (市價出場)
                                * **分批停利單：** 目標一 xxx，目標二 xxx
                                """
                                response_commander = model.generate_content(prompt_commander, stream=True)
                                response_container = st.empty()
                                full_response = ""
                                for chunk in response_commander:
                                    full_response += chunk.text
                                    response_container.markdown(full_response)
                        else:
                            with st.status("🧠 深度分析中...", expanded=True):
                                response = model.generate_content(prompt_blue)
                                st.markdown(response.text)

                    except Exception as e: st.error(f"AI Error: {e}")

            else: st.error("⚠️ 查無數據")
