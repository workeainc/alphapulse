"""
Centralized Configuration Management for AlphaPlus
Handles all application settings, environment variables, and configuration validation
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="alphapulse", env="DB_NAME")
    username: str = Field(default="alpha_emon", env="DB_USER")
    password: str = Field(default="Emon_@17711", env="DB_PASSWORD")
    min_size: int = Field(default=5, env="DB_MIN_SIZE")
    max_size: int = Field(default=20, env="DB_MAX_SIZE")
    command_timeout: int = Field(default=60, env="DB_COMMAND_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class RedisSettings(BaseSettings):
    """Redis configuration settings"""
    enabled: bool = Field(default=False, env="REDIS_ENABLED")
    host: str = Field(default="redis", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class ExchangeSettings(BaseSettings):
    """Exchange API configuration settings"""
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    binance_sandbox: bool = Field(default=True, env="BINANCE_SANDBOX")
    ccxt_rate_limit: bool = Field(default=True, env="CCXT_RATE_LIMIT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class AISettings(BaseSettings):
    """AI/ML configuration settings"""
    model_cache_size: int = Field(default=100, env="AI_MODEL_CACHE_SIZE")
    inference_batch_size: int = Field(default=32, env="AI_INFERENCE_BATCH_SIZE")
    enable_gpu: bool = Field(default=False, env="AI_ENABLE_GPU")
    model_update_interval: int = Field(default=3600, env="AI_MODEL_UPDATE_INTERVAL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class WebSocketSettings(BaseSettings):
    """WebSocket configuration settings"""
    symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT"], env="WS_SYMBOLS")
    timeframes: List[str] = Field(default=["1m", "5m", "15m", "1h"], env="WS_TIMEFRAMES")
    performance_mode: str = Field(default="enhanced", env="WS_PERFORMANCE_MODE")
    enable_shared_memory: bool = Field(default=False, env="WS_ENABLE_SHARED_MEMORY")
    max_connections: int = Field(default=100, env="WS_MAX_CONNECTIONS")
    ping_interval: int = Field(default=30, env="WS_PING_INTERVAL")
    ping_timeout: int = Field(default=10, env="WS_PING_TIMEOUT")
    batch_size: int = Field(default=50, env="WS_BATCH_SIZE")
    batch_timeout: float = Field(default=0.1, env="WS_BATCH_TIMEOUT")
    
    @validator('symbols', 'timeframes', pre=True)
    def parse_list(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class LoggingSettings(BaseSettings):
    """Logging configuration settings"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    max_file_size: int = Field(default=10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class TradingSettings(BaseSettings):
    """Trading configuration settings"""
    default_symbols: List[str] = Field(default=["BTC/USDT", "ETH/USDT"], env="TRADING_SYMBOLS")
    default_timeframes: List[str] = Field(default=["1m", "5m", "15m", "1h"], env="TRADING_TIMEFRAMES")
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    risk_per_trade: float = Field(default=0.02, env="RISK_PER_TRADE")
    
    @validator('default_symbols', 'default_timeframes', pre=True)
    def parse_list(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class APISettings(BaseSettings):
    """API configuration settings for external services"""
    # Market Data APIs
    coinglass_api_key: str = Field(default="", env="COINGLASS_API_KEY")
    polygon_api_key: str = Field(default="", env="POLYGON_API_KEY")
    coinmarketcap_api_key: str = Field(default="", env="COINMARKETCAP_API_KEY")
    coingecko_api_key: str = Field(default="", env="COINGECKO_API_KEY")
    
    # Social Media & News APIs
    news_api_key: str = Field(default="", env="NEWS_API_KEY")
    twitter_api_key: str = Field(default="", env="TWITTER_API_KEY")
    twitter_api_secret: str = Field(default="", env="TWITTER_API_SECRET")
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    
    # AI/ML APIs
    huggingface_api_key: str = Field(default="", env="HUGGINGFACE_API_KEY")
    
    # Reddit API (for social sentiment)
    reddit_client_id: Optional[str] = Field(default=None, env="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, env="REDDIT_CLIENT_SECRET")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

class Settings(BaseSettings):
    """Main application settings"""
    
    # Application settings
    app_name: str = Field(default="AlphaPlus Trading System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    exchange: ExchangeSettings = ExchangeSettings()
    ai: AISettings = AISettings()
    websocket: WebSocketSettings = WebSocketSettings()
    logging: LoggingSettings = LoggingSettings()
    trading: TradingSettings = TradingSettings()
    api: APISettings = APISettings()
    
    # Feature flags
    enable_ai: bool = Field(default=True, env="ENABLE_AI")
    enable_trading: bool = Field(default=False, env="ENABLE_TRADING")
    enable_websocket: bool = Field(default=True, env="ENABLE_WEBSOCKET")
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.database.username}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"
    
    def validate_configuration(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate database settings
            if not self.database.host or not self.database.database:
                logger.error("Invalid database configuration")
                return False
            
            # Validate trading settings
            if self.enable_trading and not self.exchange.binance_api_key:
                logger.warning("Trading enabled but no API key provided")
            
            # Validate AI settings
            if self.enable_ai and self.ai.model_cache_size <= 0:
                logger.error("Invalid AI model cache size")
                return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "debug": self.debug,
            "environment": self.environment,
            "host": self.host,
            "port": self.port,
            "database": self.database.dict(),
            "redis": self.redis.dict(),
            "exchange": self.exchange.dict(),
            "ai": self.ai.dict(),
            "websocket": self.websocket.dict(),
            "logging": self.logging.dict(),
            "trading": self.trading.dict(),
            "feature_flags": {
                "enable_ai": self.enable_ai,
                "enable_trading": self.enable_trading,
                "enable_websocket": self.enable_websocket,
                "enable_monitoring": self.enable_monitoring
            }
        }

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings

def get_config() -> Settings:
    """Get the global settings instance (alias for get_settings)"""
    return settings

def load_config_from_file(config_path: str) -> bool:
    """Load configuration from JSON file"""
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update environment variables
            for key, value in config_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        env_key = f"{key.upper()}_{sub_key.upper()}"
                        os.environ[env_key] = str(sub_value)
                else:
                    os.environ[key.upper()] = str(value)
            
            logger.info(f"✅ Configuration loaded from {config_path}")
            return True
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        return False

def setup_logging():
    """Setup logging configuration"""
    log_config = {
        'level': getattr(logging, settings.logging.level.upper()),
        'format': settings.logging.format,
        'handlers': []
    }
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(settings.logging.format))
    log_config['handlers'].append(console_handler)
    
    # File handler (if specified)
    if settings.logging.file_path:
        try:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                settings.logging.file_path,
                maxBytes=settings.logging.max_file_size,
                backupCount=settings.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(settings.logging.format))
            log_config['handlers'].append(file_handler)
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")
    
    # Apply configuration
    logging.basicConfig(**log_config)
    
    # Set specific logger levels
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    logger.info(f"✅ Logging configured with level: {settings.logging.level}")

# Initialize logging on import
setup_logging()
