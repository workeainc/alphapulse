#!/usr/bin/env python3
"""
Feast Configuration for Feature Store
Phase 2B: Feast Framework Integration
"""

import os
from pathlib import Path
from typing import Dict, Any

# Feast configuration
FEAST_CONFIG = {
    "project": "alphapulse",
    "provider": "local",
    "online_store": {
        "type": "redis",
        "connection_string": "redis://localhost:6379",
        "ttl": 3600  # 1 hour TTL for online features
    },
    "offline_store": {
        "type": "timescaledb",
        "connection_string": "postgresql://alphapulse:alphapulse@localhost:5432/alphapulse",
        "database": "alphapulse"
    },
    "feature_server": {
        "host": "0.0.0.0",
        "port": 6566,
        "enable_ssl": False
    },
    "registry": {
        "type": "local",
        "path": "data/alphapulse/feast/registry.db"
    },
    "feature_store_yaml_path": "data/alphapulse/feast/feature_store.yaml"
}

# Feature definitions for Feast
FEATURE_DEFINITIONS = {
    "technical_indicators": {
        "description": "Technical analysis indicators for trading",
        "features": {
            "rsi_14": {
                "description": "14-period Relative Strength Index",
                "type": "float",
                "tags": ["technical", "momentum", "oscillator"]
            },
            "macd": {
                "description": "Moving Average Convergence Divergence",
                "type": "float",
                "tags": ["technical", "trend", "momentum"]
            },
            "ema_20": {
                "description": "20-period Exponential Moving Average",
                "type": "float",
                "tags": ["technical", "trend", "smoothing"]
            },
            "bollinger_bands_position": {
                "description": "Position within Bollinger Bands (0-1)",
                "type": "float",
                "tags": ["technical", "volatility", "mean_reversion"]
            },
            "atr": {
                "description": "Average True Range",
                "type": "float",
                "tags": ["technical", "volatility", "risk"]
            },
            "volume_sma_ratio": {
                "description": "Volume to Simple Moving Average ratio",
                "type": "float",
                "tags": ["technical", "volume", "confirmation"]
            }
        }
    },
    "market_features": {
        "description": "Market microstructure features",
        "features": {
            "bid_ask_spread": {
                "description": "Bid-Ask spread percentage",
                "type": "float",
                "tags": ["market", "liquidity", "costs"]
            },
            "order_book_imbalance": {
                "description": "Buy vs Sell order book imbalance",
                "type": "float",
                "tags": ["market", "order_flow", "sentiment"]
            },
            "trade_size_avg": {
                "description": "Average trade size",
                "type": "float",
                "tags": ["market", "activity", "institutional"]
            }
        }
    },
    "sentiment_features": {
        "description": "Market sentiment indicators",
        "features": {
            "news_sentiment": {
                "description": "Aggregated news sentiment score",
                "type": "float",
                "tags": ["sentiment", "news", "external"]
            },
            "social_sentiment": {
                "description": "Social media sentiment score",
                "type": "float",
                "tags": ["sentiment", "social", "crowd"]
            },
            "fear_greed_index": {
                "description": "Market fear/greed index",
                "type": "float",
                "tags": ["sentiment", "market", "psychology"]
            }
        }
    }
}

# Entity definitions
ENTITY_DEFINITIONS = {
    "symbol": {
        "description": "Trading symbol (e.g., BTCUSDT, ETHUSDT)",
        "type": "string",
        "join_keys": ["symbol"]
    },
    "timeframe": {
        "description": "Trading timeframe (e.g., 1h, 4h, 1d)",
        "type": "string",
        "join_keys": ["tf"]
    },
    "timestamp": {
        "description": "Feature timestamp",
        "type": "timestamp",
        "join_keys": ["ts"]
    }
}

# Data source definitions
DATA_SOURCES = {
    "candles": {
        "description": "OHLCV candlestick data",
        "table": "candles",
        "timestamp_column": "ts",
        "join_keys": ["symbol", "tf"],
        "features": [
            "o", "h", "l", "c", "v", "vwap", "taker_buy_vol"
        ]
    },
    "signals": {
        "description": "Trading signals and predictions",
        "table": "signals",
        "timestamp_column": "ts",
        "join_keys": ["symbol", "tf"],
        "features": [
            "label", "pred", "proba", "outcome", "realized_rr", "latency_ms"
        ]
    },
    "features": {
        "description": "Computed technical features",
        "table": "feature_values",
        "timestamp_column": "timestamp",
        "join_keys": ["entity_id"],
        "features": [
            "value", "metadata"
        ]
    }
}

# Feature view definitions
FEATURE_VIEWS = {
    "technical_indicators_view": {
        "description": "Technical indicators for trading decisions",
        "entities": ["symbol", "timeframe"],
        "features": [
            "rsi_14", "macd", "ema_20", "bollinger_bands_position", "atr", "volume_sma_ratio"
        ],
        "ttl": 3600,  # 1 hour
        "online": True
    },
    "market_features_view": {
        "description": "Market microstructure features",
        "entities": ["symbol", "timeframe"],
        "features": [
            "bid_ask_spread", "order_book_imbalance", "trade_size_avg"
        ],
        "ttl": 300,  # 5 minutes
        "online": True
    },
    "sentiment_features_view": {
        "description": "Market sentiment indicators",
        "entities": ["symbol"],
        "features": [
            "news_sentiment", "social_sentiment", "fear_greed_index"
        ],
        "ttl": 1800,  # 30 minutes
        "online": True
    },
    "comprehensive_view": {
        "description": "All features combined for model training",
        "entities": ["symbol", "timeframe"],
        "features": [
            "rsi_14", "macd", "ema_20", "bollinger_bands_position", "atr", "volume_sma_ratio",
            "bid_ask_spread", "order_book_imbalance", "trade_size_avg",
            "news_sentiment", "social_sentiment", "fear_greed_index"
        ],
        "ttl": 86400,  # 24 hours
        "online": False  # Offline only for training
    }
}

# Feast YAML configuration
def generate_feast_yaml() -> str:
    """Generate Feast feature store YAML configuration"""
    yaml_content = f"""
project: {FEAST_CONFIG['project']}
provider: {FEAST_CONFIG['provider']}
online_store:
    type: {FEAST_CONFIG['online_store']['type']}
    connection_string: {FEAST_CONFIG['online_store']['connection_string']}
    ttl: {FEAST_CONFIG['online_store']['ttl']}

offline_store:
    type: {FEAST_CONFIG['offline_store']['type']}
    connection_string: {FEAST_CONFIG['offline_store']['connection_string']}
    database: {FEAST_CONFIG['offline_store']['database']}

feature_server:
    host: {FEAST_CONFIG['feature_server']['host']}
    port: {FEAST_CONFIG['feature_server']['port']}
    enable_ssl: {FEAST_CONFIG['feature_server']['enable_ssl']}

registry:
    type: {FEAST_CONFIG['registry']['type']}
    path: {FEAST_CONFIG['registry']['path']}
"""
    return yaml_content

# Feature store paths
def get_feast_paths() -> Dict[str, Path]:
    """Get Feast-related file paths"""
    base_path = Path("data/alphapulse/feast")
    
    return {
        "base": base_path,
        "registry": base_path / "registry.db",
        "feature_store_yaml": base_path / "feature_store.yaml",
        "feature_definitions": base_path / "feature_definitions.py",
        "data_sources": base_path / "data_sources.py",
        "feature_views": base_path / "feature_views.py"
    }

# Create Feast directory structure
def create_feast_structure():
    """Create Feast directory structure and configuration files"""
    paths = get_feast_paths()
    
    # Create directories
    paths["base"].mkdir(parents=True, exist_ok=True)
    
    # Create feature store YAML
    with open(paths["feature_store_yaml"], "w") as f:
        f.write(generate_feast_yaml())
    
    print(f"✅ Created Feast configuration at {paths['feature_store_yaml']}")

if __name__ == "__main__":
    create_feast_structure()
