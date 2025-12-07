import streamlit as st
import os
import subprocess
import sys
import time
import re # 用來切割 AI 回應

# ==========================================
# 🔥【暴力修復模組】
# ==========================================
try:
    import google.generativeai as genai
    from packaging import version
    current_ver = getattr(genai, "__version__", "0.0.0")
    if version.parse(current_ver) < version.parse("0.5.2"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "google-generativeai>=0.5.2"])
        import google.generativeai as genai
except Exception as e:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-generativeai>=0.5.2"])
    import google.generativeai as genai

# ==========================================
# 📦 標準套件
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
# 🔑【金鑰設定】
# ==========================================
try:
    GEMINI_API_KEY_GLOBAL = st.secrets["GEMINI_KEY"]
    FINMIND_TOKEN_GLOBAL = st.secrets["FINMIND_TOKEN"]
except:
    GEMINI_API_KEY_GLOBAL = ""
    FINMIND_TOKEN_GLOBAL = ""

# ==========================================
# ⚙️ UI 設定
# ==========================================
st.set_page_config(page_title="Alpha Strategist AI", layout="wide", page_icon="🚀")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    h1, h2, h3, h4, h5, h6, span, div, label, p, li { color: #f1f5f9 !important; }
    div[data-testid="stMetricLabel"] p { color: #94a3b8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] div { color: #38bdf8 !important; }
    section[data-testid="stSidebar"] { background-color: #1e293b; }
    .stTextInput input { background-color: #334155; color: #ffffff; border: 1px solid #475569; }
    button[data-baseweb="tab"] { background-color: transparent !important; color: #94a3b8 !important; }
    div[data-testid="stTable"] { color: white !important; }
    
    /* 角色對話框優化 */
    .role-box { padding: 18px; border-radius: 10px; margin-bottom: 15px; border-left: 6px solid; font-size: 1rem; line-height: 1.7; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .blue-team { background-color: #1e293b; border-color: #3b82f6; color: #e2e8f0; }
    .red-team { background-color: #3f1818; border-color: #ef4444; color: #fecaca; }
    .commander { background-color: #143328; border-color: #10b981; color: #d1fae5; }
    .grok { background-color: #2e1065; border-color: #a855f7; color: #e9d5ff; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Alpha Strategist AI")
st.markdown("##### ⚡ Powered by Gemini 2.5 Flash | v30.0 完美佈局版")

# ==========================================
# 📊 數據函數
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
        df_price = yf.download(f"{stock_id}.TW", start=start_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True, threads=False)
        if df_price is None or df_price.empty: return None, None, None
        
        if isinstance(df_price.columns, pd.MultiIndex): df_price.columns = df_price.columns.get_level_values(0)
        df_price = df_price.reset_index()
        
        if 'Date' in df_price.columns: df_price['date'] = df_price['Date'].dt.strftime('%Y-%m-%d')
        elif 'date' in df_price.columns: df_price['date'] = pd.to_datetime(df_price['date']).dt.strftime('%Y-%m-%d')
        else: return None, None, None

        df_price['MA5'] = df_price['Close'].rolling(window=5).mean(); df_price['MA20'] = df_price['Close'].rolling(window=20).mean(); df_price['MA60'] = df_price['Close'].rolling(window=60).mean()
        df_price = calculate_indicators(df_price)
    except Exception as e: return None, None, None
        
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
            "P/E": "N/A", "EPS": "N/A", "Yield": "N/A", 
            "Cap": round(info.market_cap/100000000, 2) if info.market_cap else 'N/A', 
            "Name": stock_id, # 預設回傳代號，如果抓不到中文名
        }
    except: return {"Name": stock_id}

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
        return [{"title": e.title, "url": e.link} for e in feed.entries[:6]]
    except: return []

# ==========================================
# 🧠 AI 核心與解析器 (修復幻覺與格式)
# ==========================================
@st.cache_data(ttl=3600) 
def ask_gemini_combined_strategy(ticker, stock_name, profile, wargame_on, red_style, data_context):
    """
    接收 stock_name 參數，強制解決 AI 認錯股票的問題。
    """
    if not GEMINI_API_KEY_GLOBAL: return "⚠️ 請先設定 Gemini API Key"

    if "Grok" in red_style:
        red_persona = "Grok (馬斯克的 AI)"; red_tone = "極度理性、科技視角、第一性原理。"
    else:
        red_persona = "華爾街空頭主力"; red_tone = "冷血、無情、專找泡沫。"

    # 🔥 關鍵修改：Prompt 強制注入中文名稱
    prompt = f"""
    你現在是 Alpha Strategist AI。請針對台股 {ticker} ({stock_name}) 進行深度的「兵棋推演」。
    注意：請務必確認分析對象是 {stock_name}，不要誤判為其他同產業公司。
    
    【投資人輪廓】：{profile}
    【市場情報】：{data_context}
    
    請嚴格依照以下標記格式輸出 (不要改變 Tag)：
    
    <BLUE_TEAM>
    (在此處撰寫藍軍參謀報告：基本面優勢、技術面金叉、目標價位)
    </BLUE_TEAM>

    <RED_TEAM>
    (在此處撰寫紅軍 {red_persona} 批判：盲點戳破、下檔風險、靈魂拷問。風格：{red_tone})
    </RED_TEAM>

    <COMMANDER>
    (在此處撰寫總司令最終決策：進攻或防守、建倉SOP、關鍵點位)
    </COMMANDER>
    """
    
    # 候選模型 (包含你的隱藏版權限)
    candidate_models = [
        'models/gemini-1.5-flash', 'models/gemini-2.5-flash', 
        'models/gemini-1.5-pro', 'models/gemini-pro'
    ]

    genai.configure(api_key=GEMINI_API_KEY_GLOBAL)
    
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text # 成功就回傳
        except: continue
            
    return "❌ AI 連線失敗，請稍後再試。"

def parse_ai_response(text):
    """將 AI 的長文切回三個區塊，還原視覺效果"""
    blue = re.search(r"<BLUE_TEAM>(.*?)</BLUE_TEAM>", text, re.DOTALL)
    red = re.search(r"<RED_TEAM>(.*?)</RED_TEAM>", text, re.DOTALL)
    commander = re.search(r"<COMMANDER>(.*?)</COMMANDER>", text, re.DOTALL)
    
    return {
        "blue": blue.group(1).strip() if blue else "藍軍數據不足...",
        "red": red.group(1).strip() if red else "紅軍暫無意見...",
        "commander": commander.group(1).strip() if commander else text # 如果格式跑掉，就全顯示
    }

# ==========================================
# 🖥️ 主介面
# ==========================================
with st.sidebar:
    st.header("⚙️ 戰術設定")
    if GEMINI_API_KEY_GLOBAL: st.success("✅ Gemini Ready")
    else: st.error("❌ No Gemini Key")

    st.markdown("---")
    st.subheader("📋 自選監控")
    # 🔥 這裡把名稱寫死，傳給後端用
    default_list = ["2330 台積電", "2317 鴻海", "2603 長榮", "2376 技嘉", "3231 緯創", "2454 聯發科"]
    selected_ticker_raw = st.radio("快速切換", default_list)
    target_stock_sidebar = selected_ticker_raw.split(" ")[0]
    # 自動抓取對應的中文名稱
    target_name_sidebar = selected_ticker_raw.split(" ")[1]

    st.markdown("---")
    st.subheader("🎯 兵棋推演")
    enable_wargame = st.toggle("啟動「紅藍軍對抗」", value=True)
    wargame_mode = st.radio("紅軍風格", ["🔴 傳統主力", "🟣 Grok 合作"], index=1) if enable_wargame else "單一模式"
    st.markdown("---")
    strategy_profile = st.radio("投資輪廓", ["穩健價值型", "激進動能型"], index=0)

# --- 主畫面佈局 (T型佈局) ---
col1, col2, col3 = st.columns([1, 1, 2])
with col1: 
    manual_input = st.text_input("股票代號", target_stock_sidebar, label_visibility="collapsed")
    target_stock = manual_input if manual_input else target_stock_sidebar
    # 如果是手動輸入，名稱暫時用代號代替，除非去查表 (這裡簡化)
    target_name = target_name_sidebar if manual_input == target_stock_sidebar else target_stock 
with col2: analysis_days = st.slider("回溯天數", 30, 180, 90, label_visibility="collapsed")
with col3: run_analysis = st.button("🔥 啟動兵棋推演", type="primary", use_container_width=True)

if run_analysis:
    if not GEMINI_API_KEY_GLOBAL: st.error("⛔ 請檢查 Gemini Key")
    else:
        with st.spinner(f"📡 戰情室連線中... 正在分析 {target_stock} {target_name}"):
            
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
            
            if df is not None and not df.empty:
                # --- Row 1: 核心指標 ---
                st.markdown("---")
                m1, m2, m3, m4, m5 = st.columns(5)
                # 優先使用 sidebar 傳進來的正確名稱
                display_name = target_name if target_name != target_stock else fundamentals.get("Name", target_stock)
                m1.metric("名稱", display_name)
                m2.metric("P/E", fundamentals.get("P/E"))
                m3.metric("EPS", fundamentals.get("EPS"))
                m4.metric("殖利率", f"{fundamentals.get('Yield')}%")
                m5.metric("市值(億)", f"{fundamentals.get('Cap')}")
                st.markdown("---")

                # --- Row 2: 圖表 (左) + 新聞/籌碼 (右) ---
                chart_col, data_col = st.columns([2, 1]) # 2:1 比例，圖表大一點

                with chart_col:
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.15, 0.15, 0.2], subplot_titles=("價量 & 機率軌道", "法人籌碼", "MACD", "KD"))
                    fig.add_trace(go.Candlestick(x=df['date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價', increasing_line_color='#ef4444', decreasing_line_color='#10b981'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA20'], name='MA20', line=dict(color='#a855f7', width=1.5)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['MA60'], name='MA60', line=dict(color='#3b82f6', width=2)), row=1, col=1)
                    # 機率軌道
                    last_c = df.iloc[-1]['Close']; last_h = df.iloc[-1]['High']; last_l = df.iloc[-1]['Low']
                    if df_probs is not None:
                        for i, r in df_probs.iterrows():
                            dist = last_c * (r['Level']/100)
                            fig.add_hline(y=last_h+dist, line_dash="dot", line_color="yellow", row=1, col=1)
                            fig.add_hline(y=last_l-dist, line_dash="dot", line_color="cyan", row=1, col=1)
                    
                    fig.add_trace(go.Bar(x=df['date'], y=df['外資'], name='外資', marker_color='cyan'), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['date'], y=df['投信'], name='投信', marker_color='orange'), row=2, col=1)
                    fig.add_trace(go.Bar(x=df['date'], y=df['MACD_Hist'], name='MACD', marker_color=np.where(df['MACD_Hist']<0,'green','red')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['K'], name='K', line=dict(color='orange')), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df['date'], y=df['D'], name='D', line=dict(color='purple')), row=4, col=1)
                    fig.update_layout(template='plotly_dark', height=800, xaxis_rangeslider_visible=False, showlegend=False, paper_bgcolor='#0f172a', plot_bgcolor='#0f172a', font=dict(color='#f8fafc'), margin=dict(t=30, b=30, l=40, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                with data_col:
                    st.subheader("📰 市場情報")
                    tab1, tab2 = st.tabs(["新聞", "營收"])
                    with tab1:
                        for n in news_list: st.markdown(f"- [{n['title']}]({n.get('url', '#')})")
                    with tab2:
                        st.dataframe(df_revenue, use_container_width=True, hide_index=True)
                    
                    st.subheader("🎲 機率分佈")
                    st.dataframe(df_probs.style.format("{:.1f}%"), use_container_width=True, hide_index=True)

                # --- Row 3: 全寬度 AI 戰報 (還原紅藍軍視覺) ---
                st.markdown("---")
                st.subheader("⚔️ 戰情推演報告")
                
                # 準備 Context
                data_for_ai = df[['date', 'Close', 'MA60', '外資', '投信', 'K', 'D']].tail(10).to_string(index=False)
                news_str = "\n".join([f"- {n['title']}" for n in news_list[:5]])
                full_context = f"數據:\n{data_for_ai}\n新聞:\n{news_str}\n基本面: P/E {fundamentals.get('P/E')}"

                # 呼叫 AI
                raw_response = ask_gemini_combined_strategy(target_stock, display_name, strategy_profile, enable_wargame, wargame_mode, full_context)
                
                # 解析並還原視覺
                parsed = parse_ai_response(raw_response)
                
                # 顯示藍軍
                st.markdown(f"<div class='role-box blue-team'><b>🔵 藍軍參謀報告：</b><br>{parsed['blue']}</div>", unsafe_allow_html=True)
                
                # 顯示紅軍 (根據模式切換顏色)
                red_class = "grok" if "Grok" in wargame_mode else "red-team"
                red_title = "🟣 Grok 觀點：" if "Grok" in wargame_mode else "🔴 紅軍批判："
                st.markdown(f"<div class='role-box {red_class}'><b>{red_title}</b><br>{parsed['red']}</div>", unsafe_allow_html=True)
                
                # 顯示總司令
                st.markdown(f"<div class='role-box commander'><b>⚔️ 總司令最終決策：</b><br>{parsed['commander']}</div>", unsafe_allow_html=True)

                # 下載報告
                final_md = f"# {display_name} 分析報告\n\n## 藍軍\n{parsed['blue']}\n\n## 紅軍\n{parsed['red']}\n\n## 結論\n{parsed['commander']}"
                st.download_button("💾 下載報告", final_md, file_name=f"{target_stock}_report.md")

            else: st.error("⚠️ 查無數據")
