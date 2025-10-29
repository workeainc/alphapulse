#!/usr/bin/env python3
"""
Test Environment Configuration
Bypasses .env file issues for testing the consolidated system
"""

import os
import sys

def setup_test_environment():
    """Setup test environment variables"""
    
    # Database Configuration
    os.environ['DATABASE_URL'] = 'postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse'
    os.environ['TIMESCALEDB_HOST'] = 'localhost'
    os.environ['TIMESCALEDB_PORT'] = '5432'
    os.environ['TIMESCALEDB_DATABASE'] = 'alphapulse'
    os.environ['TIMESCALEDB_USERNAME'] = 'alpha_emon'
    os.environ['TIMESCALEDB_PASSWORD'] = 'Emon_@17711'
    os.environ['TIMESCALEDB_URL'] = 'postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse'
    
    # Redis Configuration
    os.environ['REDIS_HOST'] = 'localhost'
    os.environ['REDIS_PORT'] = '6379'
    os.environ['REDIS_PASSWORD'] = ''
    os.environ['REDIS_URL'] = 'redis://localhost:6379'
    
    # Application Settings
    os.environ['APP_NAME'] = 'AlphaPulse'
    os.environ['APP_VERSION'] = '1.0.0'
    os.environ['APP_ENV'] = 'development'
    os.environ['DEBUG'] = 'true'
    os.environ['LOG_LEVEL'] = 'INFO'
    
    # Server Configuration
    os.environ['HOST'] = '0.0.0.0'
    os.environ['PORT'] = '8000'
    os.environ['WORKERS'] = '4'
    
    # Security
    os.environ['SECRET_KEY'] = 'alphapulse_secret_key_2024_development_only'
    os.environ['JWT_SECRET_KEY'] = 'alphapulse_jwt_secret_2024_development_only'
    
    # API Keys
    os.environ['COINGECKO_API_KEY'] = 'test_coingecko_key'
    os.environ['NEWS_API_KEY'] = 'test_news_key'
    os.environ['HUGGINGFACE_API_KEY'] = 'test_huggingface_key'
    
    # Trading Configuration
    os.environ['DEFAULT_TIMEFRAMES'] = '["1m", "5m", "15m", "1h", "4h", "1d"]'
    os.environ['MAX_OPEN_POSITIONS'] = '10'
    os.environ['DEFAULT_POSITION_SIZE'] = '0.01'
    os.environ['RISK_PER_TRADE'] = '0.02'
    
    # Pattern Detection
    os.environ['PATTERN_CONFIDENCE_THRESHOLD'] = '0.7'
    os.environ['MIN_VOLUME_CONFIRMATION'] = '1.5'
    os.environ['MAX_PATTERN_LOOKBACK'] = '100'
    
    # Technical Indicators
    os.environ['RSI_PERIOD'] = '14'
    os.environ['RSI_OVERBOUGHT'] = '70'
    os.environ['RSI_OVERSOLD'] = '30'
    os.environ['MACD_FAST'] = '12'
    os.environ['MACD_SLOW'] = '26'
    os.environ['MACD_SIGNAL'] = '9'
    os.environ['BOLLINGER_PERIOD'] = '20'
    os.environ['BOLLINGER_STD'] = '2'
    
    # Performance & Monitoring
    os.environ['METRICS_ENABLED'] = 'true'
    os.environ['HEALTH_CHECK_INTERVAL'] = '30'
    os.environ['PERFORMANCE_MONITORING'] = 'true'
    os.environ['CACHE_TTL_DEFAULT'] = '300'
    os.environ['CACHE_TTL_MARKET_DATA'] = '60'
    os.environ['CACHE_TTL_NEWS'] = '1800'
    os.environ['CACHE_TTL_SENTIMENT'] = '3600'
    
    # Environment
    os.environ['NODE_ENV'] = 'development'
    os.environ['PYTHONPATH'] = './backend'
    os.environ['PYTHON_VERSION'] = '3.9'
    
    print("✅ Test environment variables set successfully")

if __name__ == "__main__":
    setup_test_environment()
