from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "AlphaPulse"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    REDIS_URL: str = "redis://localhost:6379"
    
    # TimescaleDB specific settings
    TIMESCALE_COMPRESSION_ENABLED: bool = True
    TIMESCALE_CHUNK_INTERVAL: str = "1 day"
    TIMESCALE_COMPRESSION_AFTER: str = "7 days"
    
    # API Keys (set these in environment variables)
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET_KEY: Optional[str] = None
    
    # Twitter API (for sentiment analysis)
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET: Optional[str] = None
    TWITTER_ACCESS_TOKEN: Optional[str] = None
    TWITTER_ACCESS_TOKEN_SECRET: Optional[str] = None
    
    # Reddit API (for sentiment analysis)
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "AlphaPulse/1.0"
    
    # News API
    NEWS_API_KEY: Optional[str] = None
    
    # Trading settings
    DEFAULT_LEVERAGE: int = 1
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    MAX_DAILY_LOSS: float = 0.05    # 5% of portfolio
    MAX_CONSECUTIVE_LOSSES: int = 3
    MIN_RISK_REWARD_RATIO: float = 2.0
    RISK_PER_TRADE: float = 0.02
    
    # Technical Analysis settings
    EMA_PERIODS: List[int] = [9, 21, 50, 200]
    EMA_SHORT_PERIOD: int = 12
    EMA_LONG_PERIOD: int = 26
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: int = 70
    RSI_OVERSOLD: int = 30
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: int = 2
    ATR_PERIOD: int = 14
    ADX_PERIOD: int = 14
    
    # Timeframes for multi-timeframe analysis
    TIMEFRAMES: List[str] = ["5m", "15m", "1h", "4h"]
    PRIMARY_TIMEFRAME: str = "15m"
    CONFLUENCE_REQUIRED: int = 2
    
    # Market regime detection
    VOLATILITY_THRESHOLD: float = 0.02
    TREND_STRENGTH_THRESHOLD: float = 0.6
    ATR_THRESHOLD_HIGH: float = 0.03
    ATR_THRESHOLD_LOW: float = 0.01
    KAMA_PERIOD: int = 10
    
    # Sentiment analysis
    SENTIMENT_UPDATE_INTERVAL: int = 300  # 5 minutes
    SENTIMENT_THRESHOLD: float = 0.1
    SENTIMENT_WEIGHT_TWITTER: float = 0.4
    SENTIMENT_WEIGHT_REDDIT: float = 0.3
    SENTIMENT_WEIGHT_NEWS: float = 0.3
    
    # Risk management
    TRAILING_STOP_PERCENTAGE: float = 0.5
    TRAILING_STOP_ENABLED: bool = True
    TRAILING_STOP_ACTIVATION: float = 0.01
    TRAILING_STOP_DISTANCE: float = 0.005
    BREAK_EVEN_THRESHOLD: float = 0.3
    BREAK_EVEN_ENABLED: bool = True
    BREAK_EVEN_ACTIVATION: float = 0.02
    
    # System monitoring
    HEALTH_CHECK_INTERVAL: int = 60
    API_LATENCY_THRESHOLD: int = 1000
    MAX_EXECUTION_ERRORS: int = 5
    
    # Inference optimization settings
    ONNX_ENABLED: bool = True
    BATCH_SIZE: int = 100
    MIXED_PRECISION_ENABLED: bool = True
    MODEL_CACHE_SIZE: int = 10
    INFERENCE_TIMEOUT: float = 0.1  # 100ms
    MAX_LATENCY_MS: int = 1000
    
    # WebSocket settings
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 100
    
    # Pine Script integration
    PINE_SCRIPT_ENABLED: bool = True
    PINE_SCRIPT_UPDATE_INTERVAL: int = 60
    PINE_SCRIPT_SIGNAL_TIMEOUT: int = 300
    
    # Connection pool settings
    DB_MIN_CONNECTIONS: int = 5
    DB_MAX_CONNECTIONS: int = 20
    DB_CONNECTION_TIMEOUT: float = 30.0
    DB_HEALTH_CHECK_INTERVAL: float = 30.0
    DB_MAX_CONNECTION_LIFETIME: float = 3600.0
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


# Trading pairs configuration
TRADING_PAIRS = [
    "BTC/USDT",
    "ETH/USDT", 
    "BNB/USDT",
    "ADA/USDT",
    "SOL/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "MATIC/USDT"
]

# Economic calendar events to avoid
ECONOMIC_EVENTS = [
    "FOMC",
    "CPI",
    "NFP",
    "GDP",
    "Fed Rate Decision",
    "ECB Rate Decision",
    "BOE Rate Decision"
]

# News keywords for sentiment analysis
NEWS_KEYWORDS = [
    "bitcoin", "crypto", "cryptocurrency", "blockchain",
    "ethereum", "defi", "nft", "regulation", "adoption"
]

# Environment-specific overrides
def get_settings_for_environment(env: str = None) -> Settings:
    """Get settings for specific environment with overrides"""
    if env is None:
        env = os.getenv("ENVIRONMENT", "production")
    
    if env == "development":
        # Development overrides
        settings.DEBUG = True
        settings.HEALTH_CHECK_INTERVAL = 30
        settings.DB_MIN_CONNECTIONS = 2
        settings.DB_MAX_CONNECTIONS = 10
    
    elif env == "testing":
        # Testing overrides
        settings.DEBUG = True
        settings.DATABASE_URL = "postgresql://test:test@localhost/test_alphapulse"
        settings.HEALTH_CHECK_INTERVAL = 10
        settings.DB_MIN_CONNECTIONS = 1
        settings.DB_MAX_CONNECTIONS = 5
    
    return settings

# Auto-detect environment and apply overrides
current_env = os.getenv("ENVIRONMENT", "production")
settings = get_settings_for_environment(current_env)
