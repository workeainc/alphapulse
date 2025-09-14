#!/usr/bin/env python3
"""
Feast Feature Definitions
Phase 2B: Feast Framework Integration
"""

from datetime import timedelta
from feast import (
    Entity, Feature, FeatureView, FeatureService, Field,
    FileSource, RequestSource, ValueType, FeatureService
)
from feast.types import Float32, Float64, Int64, String, Bool
from feast.data_source import RequestSource
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgresSource

# Entity definitions
symbol_entity = Entity(
    name="symbol",
    value_type=ValueType.STRING,
    description="Trading symbol (e.g., BTCUSDT, ETHUSDT)",
    join_keys=["symbol"],
)

timeframe_entity = Entity(
    name="timeframe",
    value_type=ValueType.STRING,
    description="Trading timeframe (e.g., 1h, 4h, 1d)",
    join_keys=["tf"],
)

timestamp_entity = Entity(
    name="timestamp",
    value_type=ValueType.UNIX_TIMESTAMP,
    description="Feature timestamp",
    join_keys=["ts"],
)

# Data sources
candles_source = PostgresSource(
    name="candles_source",
    query="""
        SELECT 
            symbol, tf, ts,
            o, h, l, c, v, vwap, taker_buy_vol,
            features
        FROM candles
        WHERE ts >= '{{{{ start_date }}}}' AND ts < '{{{{ end_date }}}}'
    """,
    timestamp_field="ts",
    created_timestamp_column="created_at",
    updated_timestamp_column="updated_at"
)

signals_source = PostgresSource(
    name="signals_source",
    query="""
        SELECT 
            symbol, tf, ts,
            label, pred, proba, outcome, realized_rr, latency_ms,
            features
        FROM signals
        WHERE ts >= '{{{{ start_date }}}}' AND ts < '{{{{ end_date }}}}'
    """,
    timestamp_field="ts",
    created_timestamp_column="created_at",
    updated_timestamp_column="updated_at"
)

features_source = PostgresSource(
    name="features_source",
    query="""
        SELECT 
            entity_id, timestamp,
            feature_name, value, metadata
        FROM feature_values
        WHERE timestamp >= '{{{{ start_date }}}}' AND timestamp < '{{{{ end_date }}}}'
    """,
    timestamp_field="timestamp",
    created_timestamp_column="created_at"
)

# Feature definitions for technical indicators
rsi_14_feature = Feature(
    name="rsi_14",
    dtype=Float32,
    description="14-period Relative Strength Index"
)

macd_feature = Feature(
    name="macd",
    dtype=Float32,
    description="Moving Average Convergence Divergence"
)

ema_20_feature = Feature(
    name="ema_20",
    dtype=Float32,
    description="20-period Exponential Moving Average"
)

bollinger_bands_position_feature = Feature(
    name="bollinger_bands_position",
    dtype=Float32,
    description="Position within Bollinger Bands (0-1)"
)

atr_feature = Feature(
    name="atr",
    dtype=Float32,
    description="Average True Range"
)

volume_sma_ratio_feature = Feature(
    name="volume_sma_ratio",
    dtype=Float32,
    description="Volume to Simple Moving Average ratio"
)

# Feature definitions for market features
bid_ask_spread_feature = Feature(
    name="bid_ask_spread",
    dtype=Float32,
    description="Bid-Ask spread percentage"
)

order_book_imbalance_feature = Feature(
    name="order_book_imbalance",
    dtype=Float32,
    description="Buy vs Sell order book imbalance"
)

trade_size_avg_feature = Feature(
    name="trade_size_avg",
    dtype=Float32,
    description="Average trade size"
)

# Feature definitions for sentiment features
news_sentiment_feature = Feature(
    name="news_sentiment",
    dtype=Float32,
    description="Aggregated news sentiment score"
)

social_sentiment_feature = Feature(
    name="social_sentiment",
    dtype=Float32,
    description="Social media sentiment score"
)

fear_greed_index_feature = Feature(
    name="fear_greed_index",
    dtype=Float32,
    description="Market fear/greed index"
)

# Feature views
technical_indicators_view = FeatureView(
    name="technical_indicators_view",
    description="Technical indicators for trading decisions",
    entities=[symbol_entity, timeframe_entity],
    ttl=timedelta(hours=1),
    schema=[
        Field(name="symbol", dtype=String),
        Field(name="tf", dtype=String),
        Field(name="ts", dtype=String),
        Field(name="rsi_14", dtype=Float32),
        Field(name="macd", dtype=Float32),
        Field(name="ema_20", dtype=Float32),
        Field(name="bollinger_bands_position", dtype=Float32),
        Field(name="atr", dtype=Float32),
        Field(name="volume_sma_ratio", dtype=Float32),
    ],
    source=candles_source,
    online=True,
    tags={
        "category": "technical",
        "purpose": "trading_decisions",
        "update_frequency": "1h"
    }
)

market_features_view = FeatureView(
    name="market_features_view",
    description="Market microstructure features",
    entities=[symbol_entity, timeframe_entity],
    ttl=timedelta(minutes=5),
    schema=[
        Field(name="symbol", dtype=String),
        Field(name="tf", dtype=String),
        Field(name="ts", dtype=String),
        Field(name="bid_ask_spread", dtype=Float32),
        Field(name="order_book_imbalance", dtype=Float32),
        Field(name="trade_size_avg", dtype=Float32),
    ],
    source=candles_source,
    online=True,
    tags={
        "category": "market",
        "purpose": "microstructure_analysis",
        "update_frequency": "5m"
    }
)

sentiment_features_view = FeatureView(
    name="sentiment_features_view",
    description="Market sentiment indicators",
    entities=[symbol_entity],
    ttl=timedelta(minutes=30),
    schema=[
        Field(name="symbol", dtype=String),
        Field(name="ts", dtype=String),
        Field(name="news_sentiment", dtype=Float32),
        Field(name="social_sentiment", dtype=Float32),
        Field(name="fear_greed_index", dtype=Float32),
    ],
    source=signals_source,
    online=True,
    tags={
        "category": "sentiment",
        "purpose": "market_sentiment",
        "update_frequency": "30m"
    }
)

comprehensive_features_view = FeatureView(
    name="comprehensive_features_view",
    description="All features combined for model training",
    entities=[symbol_entity, timeframe_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="symbol", dtype=String),
        Field(name="tf", dtype=String),
        Field(name="ts", dtype=String),
        # Technical indicators
        Field(name="rsi_14", dtype=Float32),
        Field(name="macd", dtype=Float32),
        Field(name="ema_20", dtype=Float32),
        Field(name="bollinger_bands_position", dtype=Float32),
        Field(name="atr", dtype=Float32),
        Field(name="volume_sma_ratio", dtype=Float32),
        # Market features
        Field(name="bid_ask_spread", dtype=Float32),
        Field(name="order_book_imbalance", dtype=Float32),
        Field(name="trade_size_avg", dtype=Float32),
        # Sentiment features
        Field(name="news_sentiment", dtype=Float32),
        Field(name="social_sentiment", dtype=Float32),
        Field(name="fear_greed_index", dtype=Float32),
    ],
    source=candles_source,
    online=False,  # Offline only for training
    tags={
        "category": "comprehensive",
        "purpose": "model_training",
        "update_frequency": "1d"
    }
)

# Feature services
technical_indicators_service = FeatureService(
    name="technical_indicators_service",
    description="Technical indicators for real-time trading",
    features=[
        technical_indicators_view
    ],
    tags={
        "category": "trading",
        "latency": "real_time",
        "use_case": "live_trading"
    }
)

market_analysis_service = FeatureService(
    name="market_analysis_service",
    description="Market microstructure analysis",
    features=[
        market_features_view
    ],
    tags={
        "category": "analysis",
        "latency": "near_real_time",
        "use_case": "market_analysis"
    }
)

sentiment_analysis_service = FeatureService(
    name="sentiment_analysis_service",
    description="Market sentiment analysis",
    features=[
        sentiment_features_view
    ],
    tags={
        "category": "sentiment",
        "latency": "near_real_time",
        "use_case": "sentiment_analysis"
    }
)

comprehensive_training_service = FeatureService(
    name="comprehensive_training_service",
    description="All features for model training",
    features=[
        comprehensive_features_view
    ],
    tags={
        "category": "training",
        "latency": "batch",
        "use_case": "model_training"
    }
)

# Export all definitions
__all__ = [
    # Entities
    "symbol_entity", "timeframe_entity", "timestamp_entity",
    
    # Data sources
    "candles_source", "signals_source", "features_source",
    
    # Features
    "rsi_14_feature", "macd_feature", "ema_20_feature",
    "bollinger_bands_position_feature", "atr_feature", "volume_sma_ratio_feature",
    "bid_ask_spread_feature", "order_book_imbalance_feature", "trade_size_avg_feature",
    "news_sentiment_feature", "social_sentiment_feature", "fear_greed_index_feature",
    
    # Feature views
    "technical_indicators_view", "market_features_view", "sentiment_features_view", "comprehensive_features_view",
    
    # Feature services
    "technical_indicators_service", "market_analysis_service", "sentiment_analysis_service", "comprehensive_training_service"
]
