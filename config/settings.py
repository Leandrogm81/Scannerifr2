"""Configurações e constantes do projeto."""

from pathlib import Path

from src.universe_registry import load_universe_tickers

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Listas legadas preservadas por compatibilidade temporária
DEFAULT_IBOV = [
    "PETR4.SA",
    "VALE3.SA",
    "ITUB4.SA",
    "BBDC4.SA",
    "ABEV3.SA",
    "BBAS3.SA",
    "JBSS3.SA",
    "WEGE3.SA",
    "RENT3.SA",
    "LREN3.SA",
    "GGBR4.SA",
    "CMIG4.SA",
    "SUZB3.SA",
    "ELET3.SA",
    "PRIO3.SA",
    "RADL3.SA",
    "BRFS3.SA",
    "CSNA3.SA",
]

DEFAULT_SMLL = [
    "COGN3.SA",
    "SOMA3.SA",
    "RAIZ4.SA",
    "MRFG3.SA",
    "CVCB3.SA",
    "MOVI3.SA",
    "QUAL3.SA",
    "MYPK3.SA",
    "STBP3.SA",
    "LOGG3.SA",
    "WIZC3.SA",
    "ANIM3.SA",
]

LEGACY_TOP100_B3 = DEFAULT_IBOV + DEFAULT_SMLL
LEGACY_US_WATCHLIST = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "BRK.B",
    "JPM",
    "V",
]

# Aliases novos baseados em snapshots versionados
TOP100_B3 = load_universe_tickers("b3_top100", fallback=LEGACY_TOP100_B3)
TOP500_US = load_universe_tickers("us_top500", fallback=LEGACY_US_WATCHLIST)
B3_TOP100 = TOP100_B3
US_TOP500 = TOP500_US

# Parâmetros padrão da estratégia
DEFAULT_RSI_THRESHOLD = 10
DEFAULT_MIN_VOL_FIN = 1_000_000  # R$ 1 milhão
DEFAULT_PERIOD = "2y"

# Configurações de cache
CACHE_TTL = 3600  # 1 hora em segundos

# Mercado alvo
MARKETS = ["B3", "NYSE/NASDAQ"]
