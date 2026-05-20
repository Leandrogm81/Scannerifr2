"""Aplicativo Streamlit para screening e backtesting de ações (IFR2 Miner & Screener)."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# Importar módulos refatorados
from src.universe_registry import load_universe
from src.data_fetcher import fetch_all_data
from src.indicators import calc_indicators
from src.backtester import run_backtest
from src.screener import screen_tickers
from config.settings import (
    DEFAULT_IBOV,
    DEFAULT_SMLL,
    DEFAULT_RSI_THRESHOLD,
)

# Configuração da página para estética premium
st.set_page_config(
    page_title="IFR2 Miner & Screener | B3 & EUA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilização CSS adicional para vibe "Quant"
st.markdown(
    """
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e313d;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2e313d;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Caching de carregamento de universos
@st.cache_data(ttl=3600)
def get_universe_cached(code: str):
    return load_universe(code)


# Sidebar com opções
st.sidebar.title("💎 IFR2 Miner & Screener")
st.sidebar.markdown("Explore o mercado com dados reais de alta qualidade.")

# Seleção de mercado
market_choice = st.sidebar.radio("Escolha o Mercado", ["B3", "NYSE/NASDAQ", "Ambos"])

# Ativos e registros de metadados
active_tickers_meta = {}

# Carregar o universo de ativos de acordo com a escolha
if market_choice == "B3":
    index_options = [
        "Top 100 B3 (Snapshot)",
        "IBOVESPA (Legado)",
        "SMLL (Legado)",
        "Customizado",
    ]
    index_choice = st.sidebar.selectbox("Escolha a Cesta de Ativos", index_options)

    if index_choice == "Top 100 B3 (Snapshot)":
        snapshot = get_universe_cached("b3_top100")
        universe = snapshot.tickers
        for record in snapshot.records:
            active_tickers_meta[record.ticker] = record
        st.sidebar.success(f"📦 Universo: {snapshot.definition.label}")
        st.sidebar.info(f"✨ Ativos no snapshot: {snapshot.count}")
        st.sidebar.caption(
            f"📅 Atualizado em: {snapshot.records[0].updated_at if snapshot.count > 0 else 'N/A'}"
        )
    elif index_choice == "IBOVESPA (Legado)":
        universe = DEFAULT_IBOV
        st.sidebar.info(f"✨ Ativos legados: {len(universe)}")
    elif index_choice == "SMLL (Legado)":
        universe = DEFAULT_SMLL
        st.sidebar.info(f"✨ Ativos legados: {len(universe)}")
    else:
        custom_assets = st.sidebar.text_area(
            "Insira os Tickers (um por linha)", "PETR4.SA\nVALE3.SA\nWEGE3.SA"
        )
        universe = [
            ticker.strip()
            for ticker in custom_assets.strip().split("\n")
            if ticker.strip()
        ]

elif market_choice == "NYSE/NASDAQ":
    index_options = ["Top 500 US (Snapshot)", "Customizado"]
    index_choice = st.sidebar.selectbox("Escolha a Cesta de Ativos", index_options)

    if index_choice == "Top 500 US (Snapshot)":
        snapshot = get_universe_cached("us_top500")
        universe = snapshot.tickers
        for record in snapshot.records:
            active_tickers_meta[record.ticker] = record
        st.sidebar.success(f"📦 Universo: {snapshot.definition.label}")
        st.sidebar.info(f"✨ Ativos no snapshot: {snapshot.count}")
        st.sidebar.caption(
            f"📅 Atualizado em: {snapshot.records[0].updated_at if snapshot.count > 0 else 'N/A'}"
        )
    else:
        custom_assets = st.sidebar.text_area(
            "Insira os Tickers (um por linha)", "AAPL\nMSFT\nTSLA\nNVDA"
        )
        universe = [
            ticker.strip()
            for ticker in custom_assets.strip().split("\n")
            if ticker.strip()
        ]

else:  # Ambos
    index_options = ["B3 Top 100 + US Top 500 (Snapshots)", "Customizado"]
    index_choice = st.sidebar.selectbox("Escolha a Cesta de Ativos", index_options)

    if index_choice == "B3 Top 100 + US Top 500 (Snapshots)":
        snapshot_b3 = get_universe_cached("b3_top100")
        snapshot_us = get_universe_cached("us_top500")
        universe = snapshot_b3.tickers + snapshot_us.tickers
        for record in snapshot_b3.records:
            active_tickers_meta[record.ticker] = record
        for record in snapshot_us.records:
            active_tickers_meta[record.ticker] = record
        st.sidebar.success("📦 Universos Combinados")
        st.sidebar.info(f"✨ Ativos totais: {len(universe)}")
    else:
        custom_assets = st.sidebar.text_area(
            "Insira os Tickers (um por linha)", "PETR4.SA\nAAPL\nTSLA\nWEGE3.SA"
        )
        universe = [
            ticker.strip()
            for ticker in custom_assets.strip().split("\n")
            if ticker.strip()
        ]

# Multiselect para refinar ativos
selected_universe = st.sidebar.multiselect(
    "Selecione/Filtre os Ativos para Execução", universe, default=universe
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros da Estratégia")
rsi_threshold = st.sidebar.slider(
    "Threshold IFR2 de Compra",
    1,
    30,
    DEFAULT_RSI_THRESHOLD,
    help="Nível de sobrevenda para sinal de entrada.",
)

# Configuração dinâmica de volumes mínimos de liquidez
if market_choice == "B3":
    min_vol_fin_brl = st.sidebar.number_input(
        "Volume Fin. Médio Mínimo (R$)",
        value=1000000,
        step=500000,
        help="Filtra ativos sem liquidez mínima na B3.",
    )
    min_vol_fin_usd = 200000.0
    min_vol_fin_param = min_vol_fin_brl
elif market_choice == "NYSE/NASDAQ":
    min_vol_fin_usd = st.sidebar.number_input(
        "Volume Fin. Médio Mínimo (US$)",
        value=200000,
        step=50000,
        help="Filtra ativos sem liquidez mínima nos EUA.",
    )
    min_vol_fin_brl = 1000000.0
    min_vol_fin_param = min_vol_fin_usd
else:  # Ambos
    min_vol_fin_brl = st.sidebar.number_input(
        "Volume Mínimo B3 (R$)",
        value=1000000,
        step=500000,
        help="Liquidez para ativos brasileiros.",
    )
    min_vol_fin_usd = st.sidebar.number_input(
        "Volume Mínimo US (US$)",
        value=200000,
        step=50000,
        help="Liquidez para ativos americanos.",
    )
    min_vol_fin_param = {"BRL": min_vol_fin_brl, "USD": min_vol_fin_usd}

period_backtest = st.sidebar.selectbox(
    "Período Histórico (Dados)", ["1y", "2y", "3y", "5y"], index=1
)

# Tabs para screening e backtesting
tab1, tab2 = st.tabs(["📊 Screener Diário", "📈 O Garimpo (Mining)"])

with tab1:
    st.header("Screener em Tempo Real")
    st.write(
        f"Ativos em região de compra (IFR2 < {rsi_threshold} e Preço > SMA200) com liquidez adequada."
    )

    if st.button("Executar Varredura", type="primary"):
        start_time = time.time()

        # Baixar dados usando a versão otimizada com progress bar real
        progress_bar = st.progress(0.0)
        with st.spinner(f"Baixando dados de {len(selected_universe)} ativos..."):
            all_data = fetch_all_data(
                selected_universe,
                period_backtest,
                progress_callback=lambda p: progress_bar.progress(p),
            )

        elapsed = time.time() - start_time
        errors_count = sum(1 for df in all_data.values() if df.empty)

        # Processar indicadores de forma centralizada e acoplar metadados
        processed_data = {}
        for ticker, df in all_data.items():
            if df is not None and not df.empty:
                df = calc_indicators(df)

                # Resolver metadados para este ticker
                meta = active_tickers_meta.get(ticker)
                if meta:
                    df["market"] = meta.market
                    df["currency"] = meta.currency
                else:
                    # Heurística para listas customizadas/legadas
                    if ticker.endswith(".SA"):
                        df["market"] = "B3"
                        df["currency"] = "BRL"
                    else:
                        df["market"] = "NYSE/NASDAQ"
                        df["currency"] = "USD"

                processed_data[ticker] = df

        # Executar screening centralizado multi-moeda
        results = screen_tickers(processed_data, rsi_threshold, min_vol_fin_param)

        if results:
            res_df = pd.DataFrame(results)

            def highlight_signal(val):
                return "color: #00FF00; font-weight: bold" if val == "COMPRA!" else ""

            msg = f"Varredura concluída em {elapsed:.1f}s | {len(results)} ativos qualificados"
            if errors_count > 0:
                msg += f" | ⚠️ {errors_count} falhas de download"
                st.warning(msg)
            else:
                st.success(msg)

            st.dataframe(
                res_df.style.map(highlight_signal, subset=["Sinal"]),
                use_container_width=True,
            )
        else:
            msg = "Nenhum resultado encontrado que atenda aos critérios."
            if errors_count > 0:
                msg += f" (⚠️ {errors_count} falhas de download registradas)"
            st.warning(msg)

with tab2:
    st.header("O Garimpo - Backtester em Lote")
    st.info("Comparação histórica dos ativos para validar a eficiência da estratégia.")

    if st.button("Iniciar Mineração", type="primary"):
        start_time = time.time()

        # Baixar dados usando a versão otimizada com progress bar real
        progress_mine = st.progress(0.0)
        with st.spinner(f"Processando {len(selected_universe)} ativos..."):
            all_data = fetch_all_data(
                selected_universe,
                period_backtest,
                progress_callback=lambda p: progress_mine.progress(p),
            )

        elapsed = time.time() - start_time
        errors_count = sum(1 for df in all_data.values() if df.empty)

        # Processar indicadores e executar backtests
        mining_results = []
        curves = {}

        for ticker, df in all_data.items():
            if df is not None and not df.empty:
                df = calc_indicators(df)

                # Identificar moeda do ativo
                meta = active_tickers_meta.get(ticker)
                if meta:
                    df["market"] = meta.market
                    df["currency"] = meta.currency
                else:
                    if ticker.endswith(".SA"):
                        df["market"] = "B3"
                        df["currency"] = "BRL"
                    else:
                        df["market"] = "NYSE/NASDAQ"
                        df["currency"] = "USD"

                currency = df["currency"].iloc[-1]
                market = df["market"].iloc[-1]

                # Resolver threshold de liquidez correto
                threshold = min_vol_fin_brl if currency == "BRL" else min_vol_fin_usd

                if (
                    "Vol_Fin_Medio" in df.columns
                    and df["Vol_Fin_Medio"].iloc[-1] >= threshold
                ):
                    backtest = run_backtest(df, buy_threshold=rsi_threshold)

                    if backtest and backtest["Total_Trades"] > 0:
                        diff = backtest["Cum_Return"] - backtest["Buy_Hold_Return"]
                        mining_results.append(
                            {
                                "Ticker": ticker,
                                "Win Rate %": round(backtest["Win_Rate"] * 100, 1),
                                "Profit Factor": round(backtest["Profit_Factor"], 2),
                                "Exp. Math %": round(backtest["Exp_Math"] * 100, 2),
                                "Trades": backtest["Total_Trades"],
                                "Retorno Sist. %": round(
                                    backtest["Cum_Return"] * 100, 1
                                ),
                                "Buy & Hold %": round(
                                    backtest["Buy_Hold_Return"] * 100, 1
                                ),
                                "Alpha %": round(diff * 100, 1),
                                "Moeda": currency,
                                "Mercado": market,
                            }
                        )
                        curves[ticker] = backtest["Equity_Curve"]

        if mining_results:
            msg = f"Mineração concluída em {elapsed:.1f}s | {len(mining_results)} ativos minerados"
            if errors_count > 0:
                msg += f" | ⚠️ {errors_count} falhas de download"
                st.warning(msg)
            else:
                st.success(msg)

            mine_df = pd.DataFrame(mining_results).sort_values(
                by="Alpha %", ascending=False
            )

            # Reorganizar colunas para legibilidade premium
            cols = [
                "Ticker",
                "Mercado",
                "Moeda",
                "Win Rate %",
                "Profit Factor",
                "Exp. Math %",
                "Trades",
                "Retorno Sist. %",
                "Buy & Hold %",
                "Alpha %",
            ]
            mine_df = mine_df[cols]

            def highlight_diff(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "color: #00FF00; font-weight: bold"
                    elif val < 0:
                        return "color: #FF4B4B"
                return ""

            st.subheader("Ranking de Desempenho (Ordenado por Alpha)")
            st.dataframe(
                mine_df.style.map(
                    highlight_diff, subset=["Alpha %", "Retorno Sist. %"]
                ),
                use_container_width=True,
            )

            st.subheader("Curvas de Capital (Top 5 Ativos)")
            fig = go.Figure()
            top_5_tickers = mine_df.head(5)["Ticker"].tolist()

            for ticker in top_5_tickers:
                if ticker in curves:
                    fig.add_trace(
                        go.Scatter(
                            y=curves[ticker],
                            mode="lines",
                            name=ticker,
                            connectgaps=True,
                        )
                    )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Sequência de Trades",
                yaxis_title="Evolução do Patrimônio (1.0 = Base)",
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title="Ativos",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            msg = "Nenhum dado válido para gerar estatísticas."
            if errors_count > 0:
                msg += f" (⚠️ {errors_count} falhas de download registradas)"
            st.warning(msg)

st.markdown("---")
st.caption(
    "Desenvolvido para análise quantitativa no mercado financeiro. Lembre-se: Retorno passado não garante retorno futuro."
)
