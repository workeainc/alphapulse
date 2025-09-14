#!/usr/bin/env python3
"""
Enhanced Sentiment Analysis Configuration
Configuration for API keys, Redis, and sentiment analysis settings
"""

import os
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RedisConfig:
    """Redis configuration for caching and real-time data"""
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30

@dataclass
class DatabaseConfig:
    """Database configuration for TimescaleDB"""
    host: str = 'localhost'
    port: int = 5432
    database: str = 'alphapulse'
    username: str = 'alpha_emon'
    password: str = 'Emon_@17711'
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class SentimentAPIConfig:
    """API configuration for sentiment data sources"""
    # Twitter API (for real-time sentiment)
    twitter_api_key: str = ''
    twitter_api_secret: str = ''
    twitter_bearer_token: str = ''
    twitter_access_token: str = ''
    twitter_access_token_secret: str = ''
    
    # Reddit API
    reddit_client_id: str = ''
    reddit_client_secret: str = ''
    reddit_user_agent: str = 'AlphaPlus-Sentiment-Analysis/1.0'
    
    # News API
    news_api_key: str = ''
    news_api_url: str = 'https://newsapi.org/v2'
    
    # Telegram API (for crypto channels)
    telegram_bot_token: str = ''
    telegram_api_id: str = ''
    telegram_api_hash: str = ''
    
    # Discord API (for crypto communities)
    discord_bot_token: str = ''
    discord_guild_id: str = ''
    
    # Alternative News Sources
    crypto_panic_api_key: str = ''
    cryptocompare_api_key: str = ''
    
    # Fear & Greed Index
    fear_greed_api_url: str = 'https://api.alternative.me/fng/'

@dataclass
class SentimentModelConfig:
    """Configuration for sentiment analysis models"""
    # Transformer Models
    default_model: str = 'distilbert-base-uncased-finetuned-sst-2-english'
    finbert_model: str = 'ProsusAI/finbert'
    roberta_model: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    
    # Model Settings
    max_text_length: int = 512
    batch_size: int = 32
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    
    # Confidence Thresholds
    min_confidence: float = 0.6
    high_confidence_threshold: float = 0.8
    
    # Sentiment Thresholds
    positive_threshold: float = 0.1
    negative_threshold: float = -0.1

@dataclass
class SentimentProcessingConfig:
    """Configuration for sentiment processing"""
    # Collection Intervals
    collection_interval_seconds: int = 60  # 1 minute
    aggregation_interval_seconds: int = 300  # 5 minutes
    correlation_interval_seconds: int = 600  # 10 minutes
    
    # Window Sizes for Aggregation
    window_sizes: List[str] = None
    
    # Rate Limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    
    # Caching
    cache_timeout_seconds: int = 300  # 5 minutes
    sentiment_cache_prefix: str = 'sentiment:'
    
    # Quality Filters
    min_volume_threshold: int = 10
    min_confidence_threshold: float = 0.5
    outlier_detection_enabled: bool = True
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = ['1min', '5min', '15min', '1hour']

@dataclass
class AlertConfig:
    """Configuration for sentiment alerts"""
    # Alert Thresholds
    sentiment_spike_threshold: float = 0.3
    trend_reversal_threshold: float = 0.2
    anomaly_threshold: float = 2.0  # Standard deviations
    
    # Alert Severity Levels
    low_severity_threshold: float = 0.1
    medium_severity_threshold: float = 0.3
    high_severity_threshold: float = 0.5
    critical_severity_threshold: float = 0.8
    
    # Alert Channels
    email_alerts_enabled: bool = False
    webhook_alerts_enabled: bool = False
    webhook_url: str = ''
    
    # Alert Cooldown
    alert_cooldown_minutes: int = 15

class EnhancedSentimentConfig:
    """Main configuration class for enhanced sentiment analysis"""
    
    def __init__(self):
        self.redis = RedisConfig()
        self.database = DatabaseConfig()
        self.api = SentimentAPIConfig()
        self.model = SentimentModelConfig()
        self.processing = SentimentProcessingConfig()
        self.alerts = AlertConfig()
        
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Redis Configuration
        self.redis.host = os.getenv('REDIS_HOST', self.redis.host)
        self.redis.port = int(os.getenv('REDIS_PORT', str(self.redis.port)))
        self.redis.db = int(os.getenv('REDIS_DB', str(self.redis.db)))
        self.redis.password = os.getenv('REDIS_PASSWORD', self.redis.password)
        
        # Database Configuration
        self.database.host = os.getenv('DB_HOST', self.database.host)
        self.database.port = int(os.getenv('DB_PORT', str(self.database.port)))
        self.database.database = os.getenv('DB_NAME', self.database.database)
        self.database.username = os.getenv('DB_USER', self.database.username)
        self.database.password = os.getenv('DB_PASSWORD', self.database.password)
        
        # Twitter API
        self.api.twitter_api_key = os.getenv('TWITTER_API_KEY', '')
        self.api.twitter_api_secret = os.getenv('TWITTER_API_SECRET', '')
        self.api.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
        self.api.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN', '')
        self.api.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
        
        # Reddit API
        self.api.reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
        self.api.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
        self.api.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', self.api.reddit_user_agent)
        
        # News API
        self.api.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.api.news_api_url = os.getenv('NEWS_API_URL', self.api.news_api_url)
        
        # Telegram API
        self.api.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.api.telegram_api_id = os.getenv('TELEGRAM_API_ID', '')
        self.api.telegram_api_hash = os.getenv('TELEGRAM_API_HASH', '')
        
        # Discord API
        self.api.discord_bot_token = os.getenv('DISCORD_BOT_TOKEN', '')
        self.api.discord_guild_id = os.getenv('DISCORD_GUILD_ID', '')
        
        # Alternative APIs
        self.api.crypto_panic_api_key = os.getenv('CRYPTO_PANIC_API_KEY', '')
        self.api.cryptocompare_api_key = os.getenv('CRYPTOCOMPARE_API_KEY', '')
        
        # Model Configuration
        self.model.default_model = os.getenv('SENTIMENT_MODEL', self.model.default_model)
        self.model.device = os.getenv('SENTIMENT_DEVICE', self.model.device)
        self.model.max_text_length = int(os.getenv('SENTIMENT_MAX_LENGTH', str(self.model.max_text_length)))
        
        # Processing Configuration
        self.processing.collection_interval_seconds = int(os.getenv('SENTIMENT_COLLECTION_INTERVAL', str(self.processing.collection_interval_seconds)))
        self.processing.aggregation_interval_seconds = int(os.getenv('SENTIMENT_AGGREGATION_INTERVAL', str(self.processing.aggregation_interval_seconds)))
        self.processing.cache_timeout_seconds = int(os.getenv('SENTIMENT_CACHE_TIMEOUT', str(self.processing.cache_timeout_seconds)))
        
        # Alert Configuration
        self.alerts.sentiment_spike_threshold = float(os.getenv('SENTIMENT_SPIKE_THRESHOLD', str(self.alerts.sentiment_spike_threshold)))
        self.alerts.webhook_url = os.getenv('SENTIMENT_WEBHOOK_URL', '')
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.redis.password:
            return f"redis://:{self.redis.password}@{self.redis.host}:{self.redis.port}/{self.redis.db}"
        return f"redis://{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(self.database.password)
        return f"postgresql+asyncpg://{self.database.username}:{encoded_password}@{self.database.host}:{self.database.port}/{self.database.database}"
    
    def is_twitter_configured(self) -> bool:
        """Check if Twitter API is configured"""
        return bool(self.api.twitter_bearer_token or (self.api.twitter_api_key and self.api.twitter_api_secret))
    
    def is_reddit_configured(self) -> bool:
        """Check if Reddit API is configured"""
        return bool(self.api.reddit_client_id and self.api.reddit_client_secret)
    
    def is_news_configured(self) -> bool:
        """Check if News API is configured"""
        return bool(self.api.news_api_key)
    
    def is_telegram_configured(self) -> bool:
        """Check if Telegram API is configured"""
        return bool(self.api.telegram_bot_token)
    
    def is_discord_configured(self) -> bool:
        """Check if Discord API is configured"""
        return bool(self.api.discord_bot_token)
    
    def get_configured_sources(self) -> List[str]:
        """Get list of configured data sources"""
        sources = []
        if self.is_twitter_configured():
            sources.append('twitter')
        if self.is_reddit_configured():
            sources.append('reddit')
        if self.is_news_configured():
            sources.append('news')
        if self.is_telegram_configured():
            sources.append('telegram')
        if self.is_discord_configured():
            sources.append('discord')
        sources.append('onchain')  # Always available
        return sources

# Global configuration instance
config = EnhancedSentimentConfig()

def get_config() -> EnhancedSentimentConfig:
    """Get the global configuration instance"""
    return config
