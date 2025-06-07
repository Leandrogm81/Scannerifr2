# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import traceback

# --- LISTA DE ATIVOS DO IBrA (Índice Brasil Amplo) ---
# Uma lista mais completa para a função de scanner.
IBRA_STOCKS = [
    'ABEV3.SA', 'ALPA4.SA', 'AMER3.SA', 'ARZZ3.SA', 'ASAI3.SA', 'AZUL4.SA', 
    'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBSE3.SA', 'BEEF3.SA', 
    'BPAC11.SA', 'BRAP4.SA', 'BRFS3.SA', 'BRKM5.SA', 'CASH3.SA', 'CCRO3.SA', 
    'CIEL3.SA', 'CMIG4.SA', 'CMIN3.SA', 'COGN3.SA', 'CPFE3.SA', 'CPLE6.SA', 
    'CRFB3.SA', 'CSAN3.SA', 'CSNA3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DXCO3.SA', 
    'ECOR3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENBR3.SA', 
    'ENEV3.SA', 'ENGIE3.SA', 'EQTL3.SA', 'EZTC3.SA', 'FLRY3.SA', 'GGBR4.SA', 
    'GOAU4.SA', 'GOLL4.SA', 'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 
    'ITSA4.SA', 'ITUB4.SA', 'JBSS3.SA', 'KLBN11.SA', 'LREN3.SA', 'LWSA3.SA', 
    'MGLU3.SA', 'MRFG3.SA', 'MRVE3.SA', 'MULT3.SA', 'NTCO3.SA', 'PCAR3.SA', 
    'PETR3.SA', 'PETR4.SA', 'PETZ3.SA', 'PRIO3.SA', 'RADL3.SA', 'RAIL3.SA', 
    'RAIZ4.SA', 'RDOR3.SA', 'RENT3.SA', 'RRRP3.SA', 'SANB11.SA', 'SBSP3.SA', 
    'SLCE3.SA', 'SOMA3.SA', 'SUZB3.SA', 'TAEE11.SA', 'TIMS3.SA', 'TOTS3.SA', 
    'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'VBBR3.SA', 'VIVT3.SA', 'WEGE3.SA', 
    'YDUQ3.SA', 'AURE3.SA', 'RECV3.SA', 'ALOS3.SA', 'VAMO3.SA', 'SMFT3.SA',
    'KEPL3.SA', 'VIVA3.SA', 'MDIA3.SA', 'HBSA3.SA'
]

# --- Função para Obter Dados de Ações ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date_str, end_date_str):
    try:
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d") + timedelta(days=1)
        end_date_yf_str = end_date_dt.strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start_date_str, end=end_date_yf_str, progress=False, auto_adjust=True)

        if data.empty: return None
        data.columns = [str(col).title() for col in data.columns]
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_cols): return None
            
        df = data[required_cols].copy()
        df.index = pd.to_datetime(df.index).date
        df.index.name = "Date"
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        return df
    except Exception:
        return None

# --- Função de Cálculo do RSI --- 
def calculate_rsi(data, n=2):
    if "Close" not in data.columns or len(data) < n + 1: return data
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=n, min_periods=n).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=n, min_periods=n).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi[loss == 0] = 100
    rsi[rsi.isna()] = 50
    data[f"RSI_{n}"] = rsi
    return data

# --- Lógica do Backtest --- 
def run_ifr2_backtest(data, oversold_level, target_days, time_stop_days, shares_per_trade):
    if f"RSI_2" not in data.columns or data.empty: return []
    trades = []
    in_position = False
    entry_price, entry_date, target_price = 0.0, None, 0.0
    days_in_trade = 0
    data['RSI_2_Prev'] = data['RSI_2'].shift(1)

    for i in range(1, len(data)):
        current_date = data.index[i]
        current_open = data['Open'].iloc[i]
        current_high = data['High'].iloc[i]
        prev_rsi = data['RSI_2_Prev'].iloc[i]

        if in_position:
            days_in_trade += 1
            exit_price, exit_reason = None, None
            if current_high >= target_price:
                exit_price, exit_reason = target_price, "Alvo"
            elif days_in_trade >= time_stop_days:
                exit_price, exit_reason = current_open, "Tempo"

            if exit_reason:
                trades.append({
                    "Entry Date": entry_date, "Entry Price": entry_price,
                    "Exit Date": current_date, "Exit Price": exit_price,
                    "Result Fin (R$)": (exit_price - entry_price) * shares_per_trade
                })
                in_position = False

        if not in_position and prev_rsi < oversold_level:
            target_slice = data['High'].iloc[max(0, i - target_days):i]
            if not target_slice.empty:
                target_price = target_slice.max()
                entry_price, entry_date = current_open, current_date
                in_position, days_in_trade = True, 0
    return trades

# --- Interface do Usuário com Streamlit --- 
st.set_page_config(layout="wide")
st.title("Scanner de Estratégia IFR2")
st.write("Analise o desempenho da estratégia IFR2 em uma vasta gama de ativos da B3 e descubra os mais rentáveis.")

st.sidebar.header("Parâmetros Globais da Análise")
start_date_input = st.sidebar.date_input("Data de Início", datetime.now() - timedelta(days=365*5))
end_date_input = st.sidebar.date_input("Data de Fim", datetime.now())
param_oversold = st.sidebar.slider("Nível Sobrevenda IFR(2)", 1, 50, 20)
param_target_days = st.sidebar.number_input("Dias para Alvo (Máx. X Dias)", 1, 10, 3)
param_time_stop = st.sidebar.number_input("Stop no Tempo (Dias)", 1, 20, 7)
param_shares = st.sidebar.number_input("Lote (Ações por Trade)", 1, 10000, 100)
initial_capital = st.sidebar.number_input("Capital Inicial Hipotético (R$)", 10000, 1000000, 50000)

if st.sidebar.button("✔️ Iniciar Análise Completa"):
    if start_date_input >= end_date_input:
        st.error("A data de início deve ser anterior à data de fim.")
    else:
        results_summary = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(IBRA_STOCKS):
            status_text.text(f"Analisando {i+1}/{len(IBRA_STOCKS)}: {ticker}")
            progress_bar.progress((i + 1) / len(IBRA_STOCKS))

            stock_data = get_stock_data(ticker, start_date_input.strftime("%Y-%m-%d"), end_date_input.strftime("%Y-%m-%d"))
            if stock_data is not None:
                stock_data_rsi = calculate_rsi(stock_data, n=2)
                trades = run_ifr2_backtest(
                    stock_data_rsi, param_oversold, param_target_days, param_time_stop, param_shares
                )
                
                if trades:
                    trades_df = pd.DataFrame(trades)
                    total_pnl = trades_df["Result Fin (R$)"].sum()
                    return_pct = (total_pnl / initial_capital) * 100
                    
                    results_summary.append({
                        "Ativo": ticker,
                        "Retorno (%)": return_pct,
                        "Lucro Total (R$)": total_pnl,
                        "Nº de Trades": len(trades_df),
                        "Taxa de Acerto (%)": (len(trades_df[trades_df['Result Fin (R$)'] > 0]) / len(trades_df)) * 100
                    })
        
        status_text.success("Análise completa!")
        
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            top_20_df = summary_df.sort_values(by="Retorno (%)", ascending=False).head(20)

            st.header("🏆 Top 20 Ações Mais Rentáveis")
            st.write("Ranking das ações com o maior retorno percentual sobre o capital inicial para os parâmetros fornecidos.")
            
            st.dataframe(top_20_df.style.format({
                "Retorno (%)": "{:.2f}%",
                "Lucro Total (R$)": "R$ {:,.2f}",
                "Taxa de Acerto (%)": "{:.2f}%"
            }))
            
            st.header("Visualização do Ranking")
            chart_data = top_20_df.set_index('Ativo')['Retorno (%)']
            st.bar_chart(chart_data)

        else:
            st.info("Nenhuma operação foi executada para nenhum dos ativos com os parâmetros fornecidos.")
else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Iniciar Análise Completa' para começar.")

