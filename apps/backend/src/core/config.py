"""
AlphaPlus Configuration Management
Central configuration for all API keys, database settings, and system parameters
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Database Configuration
    TIMESCALEDB_HOST: str = Field(default="postgres", env="TIMESCALEDB_HOST")
    TIMESCALEDB_PORT: int = Field(default=5432, env="TIMESCALEDB_PORT")
    TIMESCALEDB_DATABASE: str = Field(default="alphapulse", env="TIMESCALEDB_DATABASE")
    TIMESCALEDB_USERNAME: str = Field(default="alpha_emon", env="TIMESCALEDB_USERNAME")
    TIMESCALEDB_PASSWORD: str = Field(default="Emon_@17711", env="TIMESCALEDB_PASSWORD")
    TIMESCALEDB_URL: str = Field(
        default="postgresql://alpha_emon:Emon_@17711@postgres:5432/alphapulse",
        env="TIMESCALEDB_URL"
    )
    
    # Market Data API Keys
    COINGLASS_API_KEY: str = Field(default="", env="COINGLASS_API_KEY")
    POLYGON_API_KEY: str = Field(default="", env="POLYGON_API_KEY")
    COINMARKETCAP_API_KEY: str = Field(default="", env="COINMARKETCAP_API_KEY")
    NEWS_API_KEY: str = Field(default="", env="NEWS_API_KEY")
    TWITTER_API_KEY: str = Field(default="", env="TWITTER_API_KEY")
    TWITTER_API_SECRET: str = Field(default="", env="TWITTER_API_SECRET")
    HUGGINGFACE_API_KEY: str = Field(default="", env="HUGGINGFACE_API_KEY")
    COINGECKO_API_KEY: str = Field(default="", env="COINGECKO_API_KEY")
    
    # Binance Configuration
    BINANCE_API_KEY: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    BINANCE_SECRET_KEY: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    BINANCE_TESTNET: bool = Field(default=True, env="BINANCE_TESTNET")
    
    # System Configuration
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    
    # Trading Configuration
    TRADING_ENABLED: bool = Field(default=False, env="TRADING_ENABLED")
    PAPER_TRADING: bool = Field(default=True, env="PAPER_TRADING")
    RISK_LIMIT_PERCENT: float = Field(default=2.0, env="RISK_LIMIT_PERCENT")
    
    # Database Performance
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=30, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    
    # Streaming Configuration
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_SSL: bool = Field(default=False, env="REDIS_SSL")
    
    # Streaming Performance
    STREAM_BATCH_SIZE: int = Field(default=100, env="STREAM_BATCH_SIZE")
    STREAM_FLUSH_INTERVAL: float = Field(default=5.0, env="STREAM_FLUSH_INTERVAL")
    STREAM_MAX_CONCURRENT: int = Field(default=10, env="STREAM_MAX_CONCURRENT")
    
    # Streaming Features
    ENABLE_NORMALIZATION: bool = Field(default=True, env="ENABLE_NORMALIZATION")
    ENABLE_CANDLE_BUILDING: bool = Field(default=True, env="ENABLE_CANDLE_BUILDING")
    ENABLE_ROLLING_STATE: bool = Field(default=True, env="ENABLE_ROLLING_STATE")
    ENABLE_INDICATORS: bool = Field(default=True, env="ENABLE_INDICATORS")
    
    # Phase 4: Data Lifecycle Management
    LIFECYCLE_ENABLED: bool = Field(default=True, env="LIFECYCLE_ENABLED")
    LIFECYCLE_RETENTION_DAYS: int = Field(default=365, env="LIFECYCLE_RETENTION_DAYS")
    LIFECYCLE_COMPRESSION_DAYS: int = Field(default=7, env="LIFECYCLE_COMPRESSION_DAYS")
    LIFECYCLE_CLEANUP_ENABLED: bool = Field(default=True, env="LIFECYCLE_CLEANUP_ENABLED")
    LIFECYCLE_ARCHIVE_ENABLED: bool = Field(default=True, env="LIFECYCLE_ARCHIVE_ENABLED")
    
    # Phase 5: Security Enhancement
    SECURITY_ENABLED: bool = Field(default=True, env="SECURITY_ENABLED")
    SECURITY_AUDIT_LOGGING: bool = Field(default=True, env="SECURITY_AUDIT_LOGGING")
    SECURITY_ACCESS_CONTROL: bool = Field(default=True, env="SECURITY_ACCESS_CONTROL")
    SECURITY_SECRETS_ROTATION: bool = Field(default=True, env="SECURITY_SECRETS_ROTATION")
    SECURITY_MONITORING: bool = Field(default=True, env="SECURITY_MONITORING")
    SECURITY_AUDIT_RETENTION_DAYS: int = Field(default=2555, env="SECURITY_AUDIT_RETENTION_DAYS")
    SECURITY_EVENT_RETENTION_DAYS: int = Field(default=365, env="SECURITY_EVENT_RETENTION_DAYS")
    SECURITY_KEY_ROTATION_INTERVAL_DAYS: int = Field(default=30, env="SECURITY_KEY_ROTATION_INTERVAL_DAYS")
    
    # Additional Configuration Fields
    TWITTER_BEARER_TOKEN: str = Field(default="", env="TWITTER_BEARER_TOKEN")
    REDDIT_CLIENT_ID: str = Field(default="", env="REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET: str = Field(default="", env="REDDIT_CLIENT_SECRET")
    SECRET_KEY: str = Field(default="", env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(default="", env="JWT_SECRET_KEY")
    DEFAULT_TIMEFRAMES: str = Field(default='["1m", "5m", "15m", "1h", "4h", "1d"]', env="DEFAULT_TIMEFRAMES")
    MAX_OPEN_POSITIONS: int = Field(default=10, env="MAX_OPEN_POSITIONS")
    DEFAULT_POSITION_SIZE: float = Field(default=0.01, env="DEFAULT_POSITION_SIZE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields but ignore them

# Create global settings instance
settings = Settings()

# Database connection string
DATABASE_URL = settings.TIMESCALEDB_URL

# Streaming configuration dictionary
STREAMING_CONFIG = {
    'redis_host': settings.REDIS_HOST,
    'redis_port': settings.REDIS_PORT,
    'redis_db': settings.REDIS_DB,
    'redis_password': settings.REDIS_PASSWORD,
    'redis_ssl': settings.REDIS_SSL,
    'stream_prefix': 'alphapulse',
    'max_stream_length': 10000,
    'batch_size': settings.STREAM_BATCH_SIZE,
}

# Phase 4: Data Lifecycle Management configuration
DATA_LIFECYCLE_CONFIG = {
    'lifecycle_enabled': settings.LIFECYCLE_ENABLED,
    'retention_policies': {
        'stream_data': 30,
        'signals': settings.LIFECYCLE_RETENTION_DAYS,
        'signal_outcomes': settings.LIFECYCLE_RETENTION_DAYS,
        'feature_snapshot_versions': 180,
        'lifecycle_executions': 90,
        'compression_metrics': 90,
        'cleanup_operations': 90,
    },
    'compression_policies': {
        'stream_data': 7,
        'signals': settings.LIFECYCLE_COMPRESSION_DAYS,
        'signal_outcomes': settings.LIFECYCLE_COMPRESSION_DAYS,
        'feature_snapshot_versions': 14,
        'lifecycle_executions': 7,
        'compression_metrics': 7,
        'cleanup_operations': 7,
    },
    'cleanup_policies': {
        'enabled': settings.LIFECYCLE_CLEANUP_ENABLED,
        'orphaned_records': {'min_age_days': 30},
        'duplicate_records': {'enabled': True},
        'corrupted_records': {'enabled': True},
        'expired_records': {'max_age_days': 365},
    },
    'archive_policies': {
        'enabled': settings.LIFECYCLE_ARCHIVE_ENABLED,
        'archive_format': 'parquet',
        'compression_type': 'gzip',
        'archive_schedule': 'monthly',
    },
    'monitoring': {
        'statistics_retention_days': 90,
        'alert_on_failures': True,
        'performance_tracking': True,
    }
}

# Phase 5: Security Enhancement configuration
SECURITY_CONFIG = {
    'security_enabled': settings.SECURITY_ENABLED,
    'audit_logging': {
        'enabled': settings.SECURITY_AUDIT_LOGGING,
        'retention_days': settings.SECURITY_AUDIT_RETENTION_DAYS,
        'log_level': 'INFO',
        'include_metadata': True,
    },
    'access_control': {
        'enabled': settings.SECURITY_ACCESS_CONTROL,
        'session_timeout_minutes': 30,
        'max_failed_attempts': 5,
        'lockout_duration_minutes': 15,
    },
    'secrets_management': {
        'enabled': settings.SECURITY_SECRETS_ROTATION,
        'rotation_interval_days': settings.SECURITY_KEY_ROTATION_INTERVAL_DAYS,
        'encryption_algorithm': 'AES-256',
        'auto_rotation': True,
    },
    'security_monitoring': {
        'enabled': settings.SECURITY_MONITORING,
        'alert_threshold': 10,
        'notification_channels': ['email', 'slack'],
        'threat_detection': True,
    },
    'policies': {
        'default_audit_policy': {'enabled': True, 'retention_days': 2555, 'log_level': 'INFO'},
        'default_access_policy': {'enabled': True, 'session_timeout_minutes': 30, 'max_failed_attempts': 5},
        'default_secrets_policy': {'enabled': True, 'rotation_interval_days': 30, 'encryption_algorithm': 'AES-256'},
        'default_monitoring_policy': {'enabled': True, 'alert_threshold': 10, 'notification_channels': ['email', 'slack']},
    }
}

# Streaming configuration dictionary
STREAMING_CONFIG = {
    'redis_host': settings.REDIS_HOST,
    'redis_port': settings.REDIS_PORT,
    'redis_db': settings.REDIS_DB,
    'redis_password': settings.REDIS_PASSWORD,
    'redis_ssl': settings.REDIS_SSL,
    'stream_prefix': 'alphapulse',
    'max_stream_length': 10000,
    'batch_size': settings.STREAM_BATCH_SIZE,
    'flush_interval': settings.STREAM_FLUSH_INTERVAL,
    'connection_pool_size': 10,
    'retry_attempts': 3,
    'retry_delay': 1.0,
    'health_check_interval': 30,
    'db_batch_size': 1000,
    'db_flush_interval': 5.0,
    'enable_deduplication': True,
    'enable_outlier_detection': True,
    'enable_validation': True,
    'enable_timestamp_normalization': True,
    'deduplication_window': 300,
    'hash_cache_size': 10000,
    'outlier_threshold': 3.0,
    'outlier_window_size': 1000,
    'max_price_change': 0.5,
    'min_volume': 0.0,
    'max_timestamp_drift': 60,
    'processing_timeout': 5.0,
    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'exact_close': True,
    'enable_vwap': True,
    'enable_trade_count': True,
    'max_candle_age': 3600,
    'default_window_size': 100,
    'max_window_size': 10000,
    'memory_limit_mb': 1024,
    'indicator_periods': {
        'sma': [20, 50, 200],
        'ema': [12, 26],
        'rsi': [14],
        'macd': [12, 26, 9],
        'bollinger': [20, 2]
    },
    'update_interval': 0.1,
    'cleanup_interval': 300,
    'max_concurrent_processing': settings.STREAM_MAX_CONCURRENT,
    'enable_normalization': settings.ENABLE_NORMALIZATION,
    'enable_candle_building': settings.ENABLE_CANDLE_BUILDING,
    'enable_rolling_state': settings.ENABLE_ROLLING_STATE,
    'enable_indicators': settings.ENABLE_INDICATORS,
    'collection_interval': 5.0,
    'retention_hours': 24,
    'max_history_size': 10000,
    'enable_alerts': True,
    'alert_thresholds': {
        'cpu_percent': {'threshold': 80.0, 'comparison': 'gt', 'severity': 'warning'},
        'memory_percent': {'threshold': 85.0, 'comparison': 'gt', 'severity': 'warning'},
        'processing_latency_ms': {'threshold': 100.0, 'comparison': 'gt', 'severity': 'error'},
        'error_rate': {'threshold': 0.05, 'comparison': 'gt', 'severity': 'error'},
        'queue_size': {'threshold': 1000, 'comparison': 'gt', 'severity': 'warning'}
    }
}
