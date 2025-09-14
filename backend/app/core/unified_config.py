"""
Unified Configuration for AlphaPulse Trading Bot
Consolidates all configuration settings into a single source of truth
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Environment detection
is_dev = os.getenv("ENV", "production").lower() == "development"

def setup_logging(log_level: str = "INFO") -> None:
    """Setup standardized logging configuration"""
    # Define log format based on environment
    if is_dev:
        # Development format with more details
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    else:
        # Production format (no emojis, structured)
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('alphapulse.log')
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

class UnifiedConfig:
    """Unified configuration class for AlphaPulse"""
    
    def __init__(self):
        # Application settings
        self.APP_NAME = os.getenv("APP_NAME", "AlphaPulse Trading Bot")
        self.VERSION = os.getenv("VERSION", "1.0.0")
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Database settings
        self.DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost:5432/alphapulse")
        self.TIMESCALEDB_HOST = os.getenv("TIMESCALEDB_HOST", "localhost")
        self.TIMESCALEDB_PORT = int(os.getenv("TIMESCALEDB_PORT", "5432"))
        self.TIMESCALEDB_DATABASE = os.getenv("TIMESCALEDB_DATABASE", "alphapulse")
        self.TIMESCALEDB_USERNAME = os.getenv("TIMESCALEDB_USERNAME", "alpha_emon")
        self.TIMESCALEDB_PASSWORD = os.getenv("TIMESCALEDB_PASSWORD", "Emon_@17711")
        
        # Redis settings
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Trading settings
        self.UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", "30"))
        self.API_KEYS = self._parse_json_env("API_KEYS", {})
        self.TRADING_PAIRS = self._parse_list_env("TRADING_PAIRS", ["BTC", "ETH"])
        self.STRATEGY_PARAMS = self._parse_json_env("STRATEGY_PARAMS", {})
        
        # Security settings
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # Performance settings
        self.MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
        
        # Setup logging
        setup_logging(self.LOG_LEVEL)
    
    def _parse_json_env(self, key: str, default: Any) -> Any:
        """Parse JSON environment variable"""
        value = os.getenv(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return default
    
    def _parse_list_env(self, key: str, default: List[str]) -> List[str]:
        """Parse comma-separated environment variable as list"""
        value = os.getenv(key)
        if value:
            return [item.strip() for item in value.split(",")]
        return default
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with standardized configuration"""
        logger = logging.getLogger(name)
        
        # Add custom formatter for development
        if is_dev and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

# Global configuration instance
unified_settings = UnifiedConfig()

def get_settings() -> UnifiedConfig:
    """Get the unified configuration instance"""
    return unified_settings

def get_logger(name: str) -> logging.Logger:
    """Get a logger with standardized configuration"""
    return unified_settings.get_logger(name)
