"""
Migration 050: Advanced Feature Engineering Phase 6
Add multi-timeframe feature fusion, market regime-aware features, news sentiment integration, and volume profile features
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the advanced feature engineering migration"""
    try:
        # Database connection
        db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='alpha_emon',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        async with db_pool.acquire() as conn:
            logger.info("🚀 Starting Advanced Feature Engineering Phase 6 migration...")
            
            # Create sde_multitimeframe_features table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_multitimeframe_features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    base_timeframe VARCHAR(10) NOT NULL,
                    feature_timestamp TIMESTAMP NOT NULL,
                    
                    -- Multi-timeframe Feature Vectors
                    mtf_1m_features JSONB,
                    mtf_5m_features JSONB,
                    mtf_15m_features JSONB,
                    mtf_1h_features JSONB,
                    mtf_4h_features JSONB,
                    mtf_1d_features JSONB,
                    
                    -- Feature Fusion Results
                    fused_features JSONB,
                    fusion_method VARCHAR(30),
                    fusion_weights JSONB,
                    fusion_confidence DECIMAL(6,4),
                    
                    -- Feature Quality Metrics
                    feature_completeness DECIMAL(6,4),
                    feature_freshness_seconds INTEGER,
                    feature_quality_score DECIMAL(6,4),
                    
                    -- Processing Metadata
                    processing_time_ms INTEGER,
                    cache_hit BOOLEAN,
                    feature_source VARCHAR(30),
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_multitimeframe_features table")
            
            # Create sde_market_regime_features table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_market_regime_features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    feature_timestamp TIMESTAMP NOT NULL,
                    
                    -- Market Regime Classification
                    regime_type VARCHAR(20), -- trending, ranging, volatile, breakout
                    regime_confidence DECIMAL(6,4),
                    regime_duration_hours INTEGER,
                    regime_transition_probability DECIMAL(6,4),
                    
                    -- Regime-Specific Features
                    trending_features JSONB, -- momentum, trend_strength, adx
                    ranging_features JSONB, -- support_resistance, bollinger_position
                    volatile_features JSONB, -- atr_percentile, volatility_ratio
                    breakout_features JSONB, -- breakout_strength, volume_confirmation
                    
                    -- Regime-Aware Technical Indicators
                    regime_adjusted_rsi DECIMAL(6,4),
                    regime_adjusted_macd DECIMAL(6,4),
                    regime_adjusted_bollinger_position DECIMAL(6,4),
                    regime_adjusted_atr_percentile DECIMAL(6,4),
                    
                    -- Regime Transition Signals
                    transition_probability DECIMAL(6,4),
                    next_regime_prediction VARCHAR(20),
                    transition_confidence DECIMAL(6,4),
                    
                    -- Metadata
                    regime_model_version VARCHAR(20),
                    feature_quality_score DECIMAL(6,4),
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_market_regime_features table")
            
            # Create sde_news_sentiment_features table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_news_sentiment_features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    feature_timestamp TIMESTAMP NOT NULL,
                    
                    -- News Sentiment Scores
                    overall_sentiment_score DECIMAL(6,4),
                    news_sentiment_score DECIMAL(6,4),
                    social_sentiment_score DECIMAL(6,4),
                    technical_sentiment_score DECIMAL(6,4),
                    
                    -- Sentiment Breakdown
                    positive_news_count INTEGER,
                    negative_news_count INTEGER,
                    neutral_news_count INTEGER,
                    total_news_count INTEGER,
                    
                    -- Sentiment Features
                    sentiment_momentum DECIMAL(6,4), -- change over time
                    sentiment_volatility DECIMAL(6,4),
                    sentiment_trend DECIMAL(6,4),
                    fear_greed_index DECIMAL(6,4),
                    
                    -- News Impact Features
                    high_impact_news_count INTEGER,
                    medium_impact_news_count INTEGER,
                    low_impact_news_count INTEGER,
                    news_impact_score DECIMAL(6,4),
                    
                    -- Sentiment Sources
                    news_sources JSONB, -- list of sources
                    social_sources JSONB, -- social media sources
                    technical_sources JSONB, -- technical analysis sources
                    
                    -- Sentiment Confidence
                    sentiment_confidence DECIMAL(6,4),
                    sentiment_freshness_minutes INTEGER,
                    
                    -- Metadata
                    sentiment_model_version VARCHAR(20),
                    feature_quality_score DECIMAL(6,4),
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_news_sentiment_features table")
            
            # Create sde_volume_profile_features table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_volume_profile_features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    feature_timestamp TIMESTAMP NOT NULL,
                    
                    -- Volume Profile Analysis
                    volume_profile_data JSONB, -- price levels with volume
                    poc_price DECIMAL(12,4), -- Point of Control
                    value_area_high DECIMAL(12,4),
                    value_area_low DECIMAL(12,4),
                    value_area_percentage DECIMAL(6,4),
                    
                    -- Volume Profile Features
                    volume_nodes JSONB, -- high volume nodes
                    volume_gaps JSONB, -- low volume areas
                    volume_clusters JSONB, -- volume clusters
                    volume_trend DECIMAL(6,4),
                    
                    -- Price-Volume Relationship
                    price_volume_correlation DECIMAL(6,4),
                    volume_price_trend DECIMAL(6,4),
                    volume_weighted_avg_price DECIMAL(12,4),
                    volume_profile_skew DECIMAL(6,4),
                    
                    -- Volume Profile Zones
                    high_volume_zones JSONB,
                    low_volume_zones JSONB,
                    balanced_zones JSONB,
                    zone_strength_scores JSONB,
                    
                    -- Volume Profile Signals
                    volume_breakout_signal DECIMAL(6,4),
                    volume_support_resistance JSONB,
                    volume_momentum DECIMAL(6,4),
                    
                    -- Metadata
                    profile_period_days INTEGER,
                    profile_resolution DECIMAL(6,4),
                    feature_quality_score DECIMAL(6,4),
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_volume_profile_features table")
            
            # Create sde_feature_fusion_config table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_feature_fusion_config (
                    id SERIAL PRIMARY KEY,
                    fusion_method VARCHAR(50) UNIQUE NOT NULL,
                    method_description TEXT,
                    
                    -- Fusion Configuration
                    fusion_params JSONB,
                    feature_weights JSONB,
                    quality_thresholds JSONB,
                    
                    -- Performance Metrics
                    fusion_accuracy DECIMAL(6,4),
                    fusion_latency_ms INTEGER,
                    fusion_success_rate DECIMAL(6,4),
                    
                    -- Method Status
                    is_active BOOLEAN DEFAULT true,
                    is_default BOOLEAN DEFAULT false,
                    priority INTEGER DEFAULT 0,
                    
                    -- Usage Statistics
                    total_usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_feature_fusion_config table")
            
            # Create sde_feature_quality_tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_feature_quality_tracking (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    feature_type VARCHAR(30) NOT NULL,
                    
                    -- Quality Metrics
                    completeness_score DECIMAL(6,4),
                    freshness_score DECIMAL(6,4),
                    accuracy_score DECIMAL(6,4),
                    consistency_score DECIMAL(6,4),
                    overall_quality_score DECIMAL(6,4),
                    
                    -- Quality Issues
                    missing_features_count INTEGER,
                    stale_features_count INTEGER,
                    inconsistent_features_count INTEGER,
                    quality_issues JSONB,
                    
                    -- Performance Metrics
                    processing_time_ms INTEGER,
                    cache_hit_rate DECIMAL(6,4),
                    error_rate DECIMAL(6,4),
                    
                    -- Time Period
                    evaluation_start_time TIMESTAMP,
                    evaluation_end_time TIMESTAMP,
                    sample_count INTEGER,
                    
                    -- Metadata
                    quality_model_version VARCHAR(20),
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_feature_quality_tracking table")
            
            # Insert default feature fusion configurations
            await conn.execute("""
                INSERT INTO sde_feature_fusion_config (
                    fusion_method, method_description, fusion_params, feature_weights,
                    quality_thresholds, fusion_accuracy, fusion_latency_ms, is_default
                ) VALUES 
                ('weighted_concatenation', 'Weighted concatenation of features from different timeframes',
                 '{"normalize_features": true, "handle_missing": "interpolate"}',
                 '{"1m": 0.1, "5m": 0.2, "15m": 0.3, "1h": 0.25, "4h": 0.1, "1d": 0.05}',
                 '{"min_completeness": 0.8, "max_freshness_seconds": 300}',
                 0.85, 50, true),
                
                ('hierarchical_fusion', 'Hierarchical fusion based on timeframe importance',
                 '{"hierarchy_levels": ["1d", "4h", "1h", "15m", "5m", "1m"]}',
                 '{"1d": 0.3, "4h": 0.25, "1h": 0.25, "15m": 0.15, "5m": 0.05}',
                 '{"min_completeness": 0.7, "max_freshness_seconds": 600}',
                 0.82, 75, false),
                
                ('adaptive_fusion', 'Adaptive fusion based on market conditions',
                 '{"adaptation_threshold": 0.1, "lookback_period": 24}',
                 '{"trending": {"1h": 0.4, "15m": 0.3, "5m": 0.2, "1m": 0.1}, "ranging": {"15m": 0.4, "5m": 0.3, "1m": 0.2, "1h": 0.1}}',
                 '{"min_completeness": 0.75, "max_freshness_seconds": 450}',
                 0.88, 60, false)
                ON CONFLICT (fusion_method) DO NOTHING
            """)
            logger.info("✅ Inserted default feature fusion configurations")
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_multitimeframe_features_symbol_timeframe 
                ON sde_multitimeframe_features(symbol, base_timeframe, feature_timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_market_regime_features_symbol_regime 
                ON sde_market_regime_features(symbol, timeframe, regime_type)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_news_sentiment_features_symbol_timestamp 
                ON sde_news_sentiment_features(symbol, feature_timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_volume_profile_features_symbol_timeframe 
                ON sde_volume_profile_features(symbol, timeframe, feature_timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_feature_fusion_config_active 
                ON sde_feature_fusion_config(is_active, is_default)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_feature_quality_tracking_symbol_type 
                ON sde_feature_quality_tracking(symbol, timeframe, feature_type)
            """)
            
            logger.info("✅ Created performance indexes")
            
            logger.info("🎉 Advanced Feature Engineering Phase 6 migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
