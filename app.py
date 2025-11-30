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

# ==========================================
# 🔑【金鑰設定區】
# ✅ 安全寫法
GEMINI_API_KEY_GLOBAL = st.secrets["GEMINI_KEY"]
FINMIND_TOKEN_GLOBAL = st.secrets["FINMIND_TOKEN"]
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
</style>
""", unsafe_allow_html=True)

st.title("🚀 Alpha Strategist AI")
st.markdown("##### ⚡ Powered by Gemini 2.5 Pro | v9.6 運算修復版")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 戰術設定")
    valid_gemini = "".join(GEMINI_API_KEY_GLOBAL.split())
    valid_finmind = "".join(FINMIND_TOKEN_GLOBAL.split())
    
    if valid_gemini: st.success("✅ Gemini 金鑰鎖定")
    else: st.error("❌ 缺 Gemini Key")
    if valid_finmind: st.success("✅ FinMind Token 鎖定")
    else: st.warning("⚠️ 缺 FinMind Token")

    st.markdown("---")
    st.subheader("📋 自選監控")
    default_list = ["2330 台積電", "2317 鴻海", "2603 長榮", "2376 技嘉", "3231 緯創", "2454 聯發科"]
    selected_ticker_raw = st.radio("快速切換", default_list)
    target_stock_sidebar = selected_ticker_raw.split(" ")[0]

    st.markdown("---")
    st.subheader("🎲 機率參數 (時間加權)")
    breakout_step = st.slider("波動間距 (%)", 0.5, 5.0, 1.0, 0.5)

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

# 🔥 v9.6 修復：改寫加權平均邏輯，避免 KeyError
def calculate_breakout_probs(df, step_percent=1.0):
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Open'] = df['Open'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    
    df['Is_Up'] = df['Prev_Close'] > df['Prev_Open']
    df['Is_Down'] = df['Prev_Close'] <= df['Prev_Open']
    
    # 建立時間權重 (越新越重)
    n = len(df)
    df['Weight'] = np.linspace(0.1, 1.0, n)
    
    stats = []
    for i in range(1, 4):
        dist = df['Prev_Close'] * (step_percent * i / 100)
        target_high = df['Prev_High'] + dist
        target_low = df['Prev_Low'] - dist
        
        # 產生 0/1 訊號
        hit_high = (df['High'] >= target_high).astype(int)
        hit_low = (df['Low'] <= target_low).astype(int)
        
        # 🔥 修復後的加權計算函數 (直接傳入 Series，不傳名稱)
        def get_prob(mask_col, hit_series):
            # 篩選出符合條件 (紅K/黑K) 的數據
            mask = df[mask_col]
            valid_hits = hit_series[mask]
            valid_weights = df.loc[mask, 'Weight']
            
            if len(valid_hits) == 0: return 0.0
            return np.average(valid_hits, weights=valid_weights) * 100

        stats.append({
            'Level': i,
            'Up_Bull': get_prob('Is_Up', hit_high),
            'Down_Bull': get_prob('Is_Up', hit_low),
            'Up_Bear': get_prob('Is_Down', hit_high),
            'Down_Bear': get_prob('Is_Down', hit_low)
        })
        
    return pd.DataFrame(stats)

def get_comprehensive_data(stock_id, days):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=days + 730)
    
    df_chips = pd.DataFrame()
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockInstitutionalInvestorsBuySell", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": valid_finmind}
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

    # 使用修復後的計算函數
    df_probs = calculate_breakout_probs(df_price.copy(), breakout_step)

    if not df_chips.empty:
        merged = pd.merge(df_price, df_chips, on='date', how='left').fillna(0)
    else:
        merged = df_price
        merged['外資'] = 0
        merged['投信'] = 0
        
    return merged.tail(days), df_chips, df_probs

def get_fundamentals(stock_id):
    try:
        stock = yf.Ticker(f"{stock_id}.TW")
        info = stock.info
        raw_yield = info.get('dividendYield', 0)
        fmt_yield = round(raw_yield * 100, 2) if raw_yield and raw_yield < 1 else (round(raw_yield, 2) if raw_yield else 'N/A')
        pe = round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A'
        eps = round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else 'N/A'
        return {"P/E": pe, "EPS": eps, "Yield": fmt_yield, "Cap": round(info.get('marketCap', 0)/100000000, 2) if info.get('marketCap') else 'N/A', "Name": info.get('longName', stock_id), "Sector": info.get('sector', 'N/A'), "Summary": info.get('longBusinessSummary', '暫無描述')}
    except: return {}

def get_revenue_data(stock_id):
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=730)
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {"dataset": "TaiwanStockMonthRevenue", "data_id": stock_id, "start_date": start_date.strftime('%Y-%m-%d'), "end_date": end_date.strftime('%Y-%m-%d'), "token": valid_finmind}
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
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={stock_id}+TW+Stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant")
        return [{"title": e.title, "url": e.link, "date": f"{e.published_parsed.tm_mon}/{e.published_parsed.tm_mday}"} for e in feed.entries[:6]]
    except: return []

# --- 主介面 ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1: 
    manual_input = st.text_input("股票代號", target_stock_sidebar, label_visibility="collapsed")
    target_stock = manual_input if manual_input else target_stock_sidebar
with col2: analysis_days = st.slider("回溯天數", 30, 180, 90, label_visibility="collapsed")
with col3: run_analysis = st.button("🔥 啟動 Alpha 分析", type="primary", use_container_width=True)

if run_analysis:
    if not valid_gemini: st.error("⛔ 請檢查 Gemini Key")
    else:
        with st.spinner(f"📡 戰情室連線中... 調閱 {target_stock} ..."):
            
            df, _, df_probs = get_comprehensive_data(target_stock, analysis_days)
            fundamentals = get_fundamentals(target_stock)
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
                        rows=4, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.5, 0.15, 0.15, 0.2], 
                        subplot_titles=("價量 & 機率軌道", "法人籌碼", "MACD", "KD")
                    )
                    
                    # 1. K線
                    fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價', increasing_line_color='#ef4444', decreasing_line_color='#10b981'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA5'], name='MA5', line=dict(color='#fbbf24', width=1)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20', line=dict(color='#a855f7', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], name='MA60', line=dict(color='#3b82f6', width=2)), row=1, col=1)

                    # 加權機率軌道
                    last_close = df.iloc[-1]['Close']
                    last_open = df.iloc[-1]['Open']
                    last_high = df.iloc[-1]['High']
                    last_low = df.iloc[-1]['Low']
                    is_last_up = last_close > last_open
                    prob_col_up = 'Up_Bull' if is_last_up else 'Up_Bear'
                    prob_col_down = 'Down_Bull' if is_last_up else 'Down_Bear'
                    
                    if df_probs is not None:
                        for i, row_prob in df_probs.iterrows():
                            level = row_prob['Level']
                            dist = last_close * (breakout_step * level / 100)
                            target_up = last_high + dist
                            prob_up = row_prob[prob_col_up]
                            fig.add_shape(type="line", x0=df['date'].iloc[-5], x1=df['date'].iloc[-1], y0=target_up, y1=target_up, line=dict(color='yellow', width=1, dash="dot"), row=1, col=1)
                            fig.add_annotation(x=df['date'].iloc[-1], y=target_up, text=f"L{level}: {target_up:.1f} ({prob_up:.0f}%)", showarrow=False, xanchor="left", font=dict(color="yellow", size=10), row=1, col=1)

                            target_down = last_low - dist
                            prob_down = row_prob[prob_col_down]
                            fig.add_shape(type="line", x0=df['date'].iloc[-5], x1=df['date'].iloc[-1], y0=target_down, y1=target_down, line=dict(color='cyan', width=1, dash="dot"), row=1, col=1)
                            fig.add_annotation(x=df['date'].iloc[-1], y=target_down, text=f"L{level}: {target_down:.1f} ({prob_down:.0f}%)", showarrow=False, xanchor="left", font=dict(color="cyan", size=10), row=1, col=1)

                    # 2. 籌碼
                    fig.add_trace(go.Bar(x=df['date'], y=df['外資'], name='外資', marker_color='cyan'), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['date'], y=df['投信'], name='投信', marker_color='orange'), row=2, col=1)

                    # 3. MACD
                    fig.add_trace(go.Bar(x=df['date'], y=df['MACD_Hist'], name='MACD柱', marker_color=np.where(df['MACD_Hist']<0, 'green', 'red')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['DIF'], name='DIF', line=dict(color='yellow', width=1)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['DEA'], name='DEA', line=dict(color='blue', width=1)), row=3, col=1)

                    # 4. KD
                    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K值', line=dict(color='orange', width=1)), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D值', line=dict(color='purple', width=1)), row=4, col=1)
                    fig.add_hline(y=80, line_dash="dot", row=4, col=1, line_color="gray")
                    fig.add_hline(y=20, line_dash="dot", row=4, col=1, line_color="gray")

                    fig.update_layout(
                        template='plotly_dark', height=1000, xaxis_rangeslider_visible=False, showlegend=True,
                        paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font=dict(color='#f8fafc', size=12),
                        legend=dict(orientation="h", y=1.01, x=0, font=dict(color="#f8fafc"), bgcolor="rgba(0,0,0,0.5)"),
                        margin=dict(t=30, b=30, l=60, r=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.write("")
                    info_tab1, info_tab2, info_tab3 = st.tabs(["📰 新聞", "💰 營收", "🎲 機率表 (加權)"])
                    with info_tab1:
                        for n in news_list: st.markdown(f"📅 {n['date']} | [{n['title']}]({n.get('url', '#')})")
                    with info_tab2:
                        st.dataframe(df_revenue, use_container_width=True, hide_index=True)
                    with info_tab3:
                        st.write("📊 時間加權統計 (權重：今日100% -> 2年前10%)")
                        st.dataframe(df_probs.style.format("{:.1f}%"), use_container_width=True)

                with ai_col:
                    st.subheader("🤖 Alpha 全能研報")
                    
                    data_for_ai = df[['date', 'Close', 'MA60', '外資', '投信', 'K', 'D', 'MACD_Hist']].tail(12).to_string(index=False)
                    prob_str = df_probs.to_string(index=False)
                    last_k_color = "紅K" if is_last_up else "黑K"

                    strategy_mode = st.session_state.get('strategy_mode', '穩健') # 預設穩健
                    if "穩健" in strategy_mode: role = "巴菲特流派"
                    else: role = "李佛摩流派"

                    prompt = f"""
                    **角色：** Alpha Strategist 首席投資官 ({role})。
                    **標的：** {target_stock}
                    
                    **【全方位數據】**
                    * 技術籌碼：\n{data_for_ai}
                    * 統計機率 ({last_k_color}後，採時間加權)：\n{prob_str}
                    * 說明：Level 1 = 漲跌{breakout_step}%
                    
                    **任務：撰寫【決策研報】(Markdown)：**

                    ### 1. 🎲 統計學視角 (Statistical Edge)
                    * **明日預測：** 根據加權統計，明日上漲/下跌機率？
                    * **關鍵價位：** 引用圖表上的 L1/L2 機率價位。

                    ### 2. 🕵️‍♂️ 籌碼與技術
                    * **主力動向：** 外資投信態度。
                    * **技術結構：** KD/MACD 位置。

                    ### 3. ⚔️ 最終指令
                    * **決策：** 【買進 / 觀望 / 賣出 / 放空】
                    * **鐵律止損：** (必填)

                    ---
                    *Alpha Strategist v9.6*
                    """
                    
                    try:
                        genai.configure(api_key=valid_gemini)
                        model = genai.GenerativeModel('models/gemini-2.5-pro')
                        with st.status("🧠 AI 正在計算勝率...", expanded=True) as status:
                            response_container = st.empty()
                            full_response = ""
                            response = model.generate_content(prompt, stream=True)
                            for chunk in response:
                                full_response += chunk.text
                                response_container.markdown(full_response)
                            status.update(label="✅ 分析完成", state="complete", expanded=True)
                    except Exception as e: st.error(f"AI Error: {e}")


            else: st.error("⚠️ 查無數據")
