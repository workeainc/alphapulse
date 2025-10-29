#!/usr/bin/env python3
"""
Migration: Create Enhanced Market Intelligence Tables for Advanced Market Analysis
TimescaleDB schema for storing comprehensive market intelligence, inflow/outflow analysis, 
whale tracking, and advanced correlation data
"""

import asyncio
import logging
import os
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for psql authentication
os.environ['PGPASSWORD'] = 'Emon_@17711'

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def create_enhanced_market_intelligence_tables():
    """Create enhanced market intelligence tables with TimescaleDB optimizations"""
    
    # 1. Enhanced Market Intelligence Table (expanded from existing)
    enhanced_market_intelligence_table = """
    CREATE TABLE IF NOT EXISTS enhanced_market_intelligence (
        id SERIAL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Core Market Metrics
        btc_dominance NUMERIC(10,4),
        total2_value NUMERIC(20,8),
        total3_value NUMERIC(20,8),
        total_market_cap NUMERIC(20,8),
        btc_market_cap NUMERIC(20,8),
        eth_market_cap NUMERIC(20,8),
        
        -- Advanced Market Metrics
        total2_total3_ratio NUMERIC(10,6),
        btc_eth_ratio NUMERIC(10,6),
        market_structure_score NUMERIC(4,3),
        market_efficiency_ratio NUMERIC(6,4),
        
        -- Sentiment & Fear Metrics
        market_sentiment_score NUMERIC(4,3),
        news_sentiment_score NUMERIC(4,3),
        social_sentiment_score NUMERIC(4,3),
        volume_positioning_score NUMERIC(4,3),
        fear_greed_index INTEGER,
        
        -- Market Regime & Volatility
        market_regime VARCHAR(50), -- 'bullish', 'bearish', 'sideways', 'volatile', 'accumulation', 'distribution'
        volatility_index NUMERIC(6,4),
        trend_strength NUMERIC(4,3),
        momentum_score NUMERIC(4,3),
        
        -- Composite Indices
        composite_market_strength NUMERIC(4,3),
        risk_on_risk_off_score NUMERIC(4,3),
        market_confidence_index NUMERIC(4,3),
        
        -- Enhanced Features (Phase 1)
        sector_rotation_strength NUMERIC(4,3),
        capital_flow_heatmap JSONB,
        sector_performance_ranking JSONB,
        rotation_confidence NUMERIC(4,3),
        
        -- Sentiment Enhancement
        weighted_coin_sentiment JSONB,
        whale_sentiment_proxy NUMERIC(4,3),
        sentiment_divergence_score NUMERIC(4,3),
        multi_timeframe_sentiment JSONB,
        
        -- Phase 2: Advanced Analytics
        rolling_beta_btc_eth NUMERIC(6,4),
        rolling_beta_btc_altcoins NUMERIC(6,4),
        lead_lag_analysis JSONB,
        correlation_breakdown_alerts JSONB,
        optimal_timing_signals JSONB,
        monte_carlo_scenarios JSONB,
        confidence_bands JSONB,
        feature_importance_scores JSONB,
        ensemble_model_weights JSONB,
        prediction_horizons JSONB,
        
        -- Metadata
        data_quality_score NUMERIC(4,3),
        source VARCHAR(50),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # 2. Inflow/Outflow Analysis Table
    inflow_outflow_table = """
    CREATE TABLE IF NOT EXISTS inflow_outflow_analysis (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Exchange Flow Metrics
        exchange_inflow_24h NUMERIC(20,8),
        exchange_outflow_24h NUMERIC(20,8),
        net_exchange_flow NUMERIC(20,8),
        exchange_flow_ratio NUMERIC(6,4),
        
        -- Whale Movement Metrics
        whale_inflow_24h NUMERIC(20,8),
        whale_outflow_24h NUMERIC(20,8),
        net_whale_flow NUMERIC(20,8),
        whale_flow_ratio NUMERIC(6,4),
        large_transaction_count INTEGER,
        avg_transaction_size NUMERIC(20,8),
        
        -- Network Activity Metrics
        active_addresses_24h INTEGER,
        new_addresses_24h INTEGER,
        transaction_count_24h INTEGER,
        network_activity_score NUMERIC(4,3),
        
        -- Enhanced Flow Analysis (Phase 1)
        stablecoin_flow_24h NUMERIC(20,8),
        derivatives_flow_24h NUMERIC(20,8),
        spot_flow_24h NUMERIC(20,8),
        exchange_specific_flows JSONB,
        on_chain_exchange_flow NUMERIC(20,8),
        
        -- Supply Distribution
        supply_concentration_top_10 NUMERIC(6,4),
        supply_concentration_top_100 NUMERIC(6,4),
        supply_distribution_score NUMERIC(4,3),
        
        -- Flow Analysis
        flow_direction VARCHAR(20), -- 'inflow', 'outflow', 'neutral'
        flow_strength VARCHAR(20), -- 'weak', 'moderate', 'strong', 'extreme'
        flow_confidence NUMERIC(4,3),
        flow_anomaly BOOLEAN DEFAULT FALSE,
        
        -- Metadata
        exchange VARCHAR(50),
        data_source VARCHAR(50),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # 3. Whale Movement Tracking Table
    whale_movement_table = """
    CREATE TABLE IF NOT EXISTS whale_movement_tracking (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Whale Transaction Details
        transaction_hash VARCHAR(100),
        from_address VARCHAR(100),
        to_address VARCHAR(100),
        transaction_value NUMERIC(20,8),
        transaction_fee NUMERIC(20,8),
        
        -- Whale Classification
        whale_type VARCHAR(50), -- 'exchange', 'institutional', 'retail', 'unknown'
        whale_category VARCHAR(50), -- 'whale', 'shark', 'dolphin', 'fish'
        wallet_age_days INTEGER,
        
        -- Movement Analysis
        movement_type VARCHAR(50), -- 'accumulation', 'distribution', 'transfer', 'exchange_deposit', 'exchange_withdrawal'
        movement_direction VARCHAR(20), -- 'inflow', 'outflow', 'internal'
        movement_significance VARCHAR(20), -- 'low', 'medium', 'high', 'extreme'
        
        -- Impact Analysis
        price_impact_estimate NUMERIC(6,4),
        market_impact_score NUMERIC(4,3),
        correlation_with_price NUMERIC(6,4),
        
        -- Metadata
        blockchain VARCHAR(50),
        exchange_detected VARCHAR(50),
        confidence_score NUMERIC(4,3),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # 4. Advanced Correlation Analysis Table
    correlation_analysis_table = """
    CREATE TABLE IF NOT EXISTS correlation_analysis (
        id SERIAL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Asset Pair Correlations
        btc_eth_correlation NUMERIC(6,4),
        btc_altcoin_correlation NUMERIC(6,4),
        eth_altcoin_correlation NUMERIC(6,4),
        defi_correlation NUMERIC(6,4),
        meme_correlation NUMERIC(6,4),
        
        -- Market Segment Correlations
        large_cap_correlation NUMERIC(6,4),
        mid_cap_correlation NUMERIC(6,4),
        small_cap_correlation NUMERIC(6,4),
        sector_correlation_matrix JSONB,
        
        -- Cross-Asset Correlations
        crypto_gold_correlation NUMERIC(6,4),
        crypto_sp500_correlation NUMERIC(6,4),
        crypto_dxy_correlation NUMERIC(6,4),
        crypto_vix_correlation NUMERIC(6,4),
        
        -- Rolling Correlations
        btc_eth_rolling_7d NUMERIC(6,4),
        btc_eth_rolling_30d NUMERIC(6,4),
        btc_altcoin_rolling_7d NUMERIC(6,4),
        btc_altcoin_rolling_30d NUMERIC(6,4),
        
        -- Enhanced Correlation Analysis (Phase 1)
        rolling_beta_btc_eth NUMERIC(6,4),
        rolling_beta_btc_altcoins NUMERIC(6,4),
        lead_lag_analysis JSONB,
        correlation_breakdown_alerts JSONB,
        optimal_timing_signals JSONB,
        
        -- Phase 2: Advanced Correlation
        cross_market_correlations JSONB,
        beta_regime VARCHAR(50),
        lead_lag_confidence NUMERIC(4,3),
        
        -- Correlation Regime
        correlation_regime VARCHAR(50), -- 'high_correlation', 'low_correlation', 'diverging', 'converging'
        correlation_strength VARCHAR(20), -- 'weak', 'moderate', 'strong'
        correlation_trend VARCHAR(20), -- 'increasing', 'decreasing', 'stable'
        
        -- Metadata
        correlation_window_days INTEGER,
        data_quality_score NUMERIC(4,3),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # 5. Predictive Market Regime Table
    predictive_regime_table = """
    CREATE TABLE IF NOT EXISTS predictive_market_regime (
        id SERIAL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Current Regime
        current_regime VARCHAR(50), -- 'bullish', 'bearish', 'sideways', 'volatile', 'accumulation', 'distribution'
        regime_confidence NUMERIC(4,3),
        regime_strength NUMERIC(4,3),
        
        -- Predicted Regime
        predicted_regime VARCHAR(50),
        prediction_confidence NUMERIC(4,3),
        prediction_horizon_hours INTEGER,
        regime_change_probability NUMERIC(4,3),
        
        -- Regime Features
        btc_dominance_trend NUMERIC(6,4),
        total2_total3_trend NUMERIC(6,4),
        volume_trend NUMERIC(6,4),
        sentiment_trend NUMERIC(6,4),
        volatility_trend NUMERIC(6,4),
        
        -- Enhanced Prediction Engine (Phase 1)
        monte_carlo_scenarios JSONB,
        confidence_bands JSONB,
        feature_importance_scores JSONB,
        ensemble_model_weights JSONB,
        prediction_horizons JSONB,
        
        -- Phase 2: Advanced ML
        xgboost_prediction NUMERIC(6,4),
        catboost_prediction NUMERIC(6,4),
        ensemble_prediction NUMERIC(6,4),
        prediction_confidence NUMERIC(4,3),
        model_performance_metrics JSONB,
        
        -- ML Model Features
        feature_vector JSONB,
        model_version VARCHAR(50),
        model_performance_score NUMERIC(4,3),
        
        -- Regime Transitions
        previous_regime VARCHAR(50),
        regime_duration_hours INTEGER,
        transition_probability NUMERIC(4,3),
        
        -- Metadata
        model_type VARCHAR(50), -- 'ensemble', 'lstm', 'transformer', 'statistical'
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # 6. Market Anomaly Detection Table
    anomaly_detection_table = """
    CREATE TABLE IF NOT EXISTS market_anomaly_detection (
        id SERIAL,
        symbol VARCHAR(20) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Anomaly Types
        anomaly_type VARCHAR(100), -- 'volume_spike', 'price_spike', 'whale_movement', 'correlation_breakdown', 'sentiment_divergence'
        anomaly_severity VARCHAR(20), -- 'low', 'medium', 'high', 'extreme'
        anomaly_confidence NUMERIC(4,3),
        
        -- Anomaly Metrics
        baseline_value NUMERIC(20,8),
        current_value NUMERIC(20,8),
        deviation_percentage NUMERIC(10,4),
        z_score NUMERIC(8,4),
        
        -- Context Information
        market_context VARCHAR(200),
        related_events JSONB,
        impact_assessment VARCHAR(200),
        
        -- Detection Method
        detection_method VARCHAR(50), -- 'statistical', 'ml', 'rule_based', 'ensemble'
        detection_model VARCHAR(50),
        false_positive_probability NUMERIC(4,3),
        
        -- Resolution
        resolved BOOLEAN DEFAULT FALSE,
        resolution_time TIMESTAMPTZ,
        resolution_type VARCHAR(50), -- 'natural', 'intervention', 'false_alarm'
        
        -- Metadata
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # 7. Market Intelligence Alerts Table (NEW)
    market_alerts_table = """
    CREATE TABLE IF NOT EXISTS market_intelligence_alerts (
        id SERIAL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Alert Information
        alert_type VARCHAR(50), -- 'regime_change', 'whale_movement', 'correlation_breakdown', 'sentiment_divergence', 'flow_anomaly'
        severity VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
        message TEXT,
        actionable_insight TEXT,
        
        -- Risk Assessment
        risk_level VARCHAR(20), -- 'low', 'medium', 'high', 'extreme'
        confidence_score NUMERIC(4,3),
        related_metrics JSONB,
        
        -- Alert Context
        affected_assets JSONB,
        market_impact_assessment TEXT,
        recommended_action TEXT,
        
        -- Metadata
        source VARCHAR(50),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # 8. Risk/Reward Analysis Table (NEW)
    risk_reward_table = """
    CREATE TABLE IF NOT EXISTS risk_reward_analysis (
        id SERIAL,
        timestamp TIMESTAMPTZ NOT NULL,
        
        -- Risk Assessment
        market_risk_score NUMERIC(4,3),
        recommended_leverage NUMERIC(4,3),
        portfolio_risk_level VARCHAR(20), -- 'low', 'medium', 'high', 'extreme'
        
        -- Liquidation Analysis
        liquidation_heatmap JSONB,
        liquidation_risk_score NUMERIC(4,3),
        
        -- Risk/Reward Setups
        risk_reward_setups JSONB,
        optimal_entry_points JSONB,
        stop_loss_recommendations JSONB,
        
        -- Confidence Intervals
        confidence_interval JSONB,
        risk_adjusted_returns NUMERIC(6,4),
        
        -- Market Context
        current_regime VARCHAR(50),
        sentiment_context VARCHAR(100),
        flow_context VARCHAR(100),
        
        -- Metadata
        analysis_version VARCHAR(50),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        PRIMARY KEY (timestamp, id)
    );
    """
    
    # Execute table creation using asyncpg
    tables = [
        ("Enhanced Market Intelligence", enhanced_market_intelligence_table),
        ("Inflow/Outflow Analysis", inflow_outflow_table),
        ("Whale Movement Tracking", whale_movement_table),
        ("Correlation Analysis", correlation_analysis_table),
        ("Predictive Market Regime", predictive_regime_table),
        ("Market Anomaly Detection", anomaly_detection_table),
        ("Market Intelligence Alerts", market_alerts_table),
        ("Risk/Reward Analysis", risk_reward_table)
    ]
    
    try:
        # Connect to database using asyncpg
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database successfully")
        
        for i, (table_name, command) in enumerate(tables, 1):
            try:
                logger.info(f"Creating {table_name} table {i}/8...")
                await conn.execute(command)
                logger.info(f"✅ {table_name} table created successfully")
                    
            except Exception as e:
                logger.error(f"❌ Error creating {table_name} table: {e}")
                await conn.close()
                return False
        
        # Create TimescaleDB hypertables
        hypertable_commands = [
            "SELECT create_hypertable('enhanced_market_intelligence', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('inflow_outflow_analysis', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('whale_movement_tracking', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('correlation_analysis', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('predictive_market_regime', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('market_anomaly_detection', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('market_intelligence_alerts', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
            "SELECT create_hypertable('risk_reward_analysis', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
        ]
        
        for i, command in enumerate(hypertable_commands, 1):
            try:
                logger.info(f"Creating enhanced market intelligence hypertable {i}/6...")
                await conn.execute(command)
                logger.info(f"✅ Enhanced market intelligence hypertable {i} created successfully")
                    
            except Exception as e:
                logger.warning(f"⚠️ Hypertable {i} creation warning: {e}")
        
        # Create indexes for performance
        index_commands = [
            # Enhanced Market Intelligence indexes
            "CREATE INDEX IF NOT EXISTS idx_enhanced_market_intelligence_timestamp ON enhanced_market_intelligence (timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_market_intelligence_regime ON enhanced_market_intelligence (market_regime, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_market_intelligence_btc_dominance ON enhanced_market_intelligence (btc_dominance, timestamp DESC);",
            
            # Inflow/Outflow indexes
            "CREATE INDEX IF NOT EXISTS idx_inflow_outflow_symbol_timestamp ON inflow_outflow_analysis (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_inflow_outflow_direction ON inflow_outflow_analysis (flow_direction, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_inflow_outflow_anomaly ON inflow_outflow_analysis (flow_anomaly, timestamp DESC);",
            
            # Whale Movement indexes
            "CREATE INDEX IF NOT EXISTS idx_whale_movement_symbol_timestamp ON whale_movement_tracking (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_whale_movement_type ON whale_movement_tracking (movement_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_whale_movement_significance ON whale_movement_tracking (movement_significance, timestamp DESC);",
            
            # Correlation indexes
            "CREATE INDEX IF NOT EXISTS idx_correlation_analysis_timestamp ON correlation_analysis (timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_correlation_analysis_regime ON correlation_analysis (correlation_regime, timestamp DESC);",
            
            # Predictive Regime indexes
            "CREATE INDEX IF NOT EXISTS idx_predictive_regime_timestamp ON predictive_market_regime (timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_predictive_regime_current ON predictive_market_regime (current_regime, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_predictive_regime_predicted ON predictive_market_regime (predicted_regime, timestamp DESC);",
            
            # Anomaly Detection indexes
            "CREATE INDEX IF NOT EXISTS idx_anomaly_detection_symbol_timestamp ON market_anomaly_detection (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_detection_type ON market_anomaly_detection (anomaly_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_detection_severity ON market_anomaly_detection (anomaly_severity, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_detection_resolved ON market_anomaly_detection (resolved, timestamp DESC);"
        ]
        
        for i, command in enumerate(index_commands, 1):
            try:
                logger.info(f"Creating enhanced market intelligence index {i}/16...")
                await conn.execute(command)
                logger.info(f"✅ Enhanced market intelligence index {i} created successfully")
                    
            except Exception as e:
                logger.warning(f"⚠️ Index {i} creation warning: {e}")
        
        # Create continuous aggregates for real-time analysis
        continuous_aggregate_commands = [
            # 5-minute aggregates for market intelligence
            """
            CREATE MATERIALIZED VIEW enhanced_market_intelligence_5m
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('5 minutes', timestamp) AS bucket,
                AVG(btc_dominance) as avg_btc_dominance,
                AVG(total2_value) as avg_total2_value,
                AVG(total3_value) as avg_total3_value,
                AVG(market_sentiment_score) as avg_market_sentiment,
                AVG(fear_greed_index) as avg_fear_greed_index,
                AVG(composite_market_strength) as avg_market_strength,
                COUNT(*) as data_points
            FROM enhanced_market_intelligence
            GROUP BY bucket
            ORDER BY bucket;
            """,
            
            # 1-hour aggregates for correlation analysis
            """
            CREATE MATERIALIZED VIEW correlation_analysis_1h
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 hour', timestamp) AS bucket,
                AVG(btc_eth_correlation) as avg_btc_eth_correlation,
                AVG(btc_altcoin_correlation) as avg_btc_altcoin_correlation,
                AVG(crypto_gold_correlation) as avg_crypto_gold_correlation,
                AVG(crypto_sp500_correlation) as avg_crypto_sp500_correlation,
                COUNT(*) as data_points
            FROM correlation_analysis
            GROUP BY bucket
            ORDER BY bucket;
            """,
            
            # 1-hour aggregates for inflow/outflow
            """
            CREATE MATERIALIZED VIEW inflow_outflow_1h
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 hour', timestamp) AS bucket,
                symbol,
                SUM(exchange_inflow_24h) as total_inflow,
                SUM(exchange_outflow_24h) as total_outflow,
                SUM(net_exchange_flow) as net_flow,
                AVG(flow_confidence) as avg_flow_confidence,
                COUNT(*) as data_points
            FROM inflow_outflow_analysis
            GROUP BY bucket, symbol
            ORDER BY bucket, symbol;
            """
        ]
        
        for i, command in enumerate(continuous_aggregate_commands, 1):
            try:
                logger.info(f"Creating continuous aggregate {i}/3...")
                await conn.execute(command)
                logger.info(f"✅ Continuous aggregate {i} created successfully")
                    
            except Exception as e:
                logger.warning(f"⚠️ Continuous aggregate {i} creation warning: {e}")
        
        # Setup compression and retention policies
        policy_commands = [
            # Compression policies (compress after 7 days)
            "SELECT add_compression_policy('enhanced_market_intelligence', INTERVAL '7 days');",
            "SELECT add_compression_policy('inflow_outflow_analysis', INTERVAL '7 days');",
            "SELECT add_compression_policy('whale_movement_tracking', INTERVAL '7 days');",
            "SELECT add_compression_policy('correlation_analysis', INTERVAL '7 days');",
            "SELECT add_compression_policy('predictive_market_regime', INTERVAL '7 days');",
            "SELECT add_compression_policy('market_anomaly_detection', INTERVAL '7 days');",
            
            # Retention policies (keep data for 90 days)
            "SELECT add_retention_policy('enhanced_market_intelligence', INTERVAL '90 days');",
            "SELECT add_retention_policy('inflow_outflow_analysis', INTERVAL '90 days');",
            "SELECT add_retention_policy('whale_movement_tracking', INTERVAL '90 days');",
            "SELECT add_retention_policy('correlation_analysis', INTERVAL '90 days');",
            "SELECT add_retention_policy('predictive_market_regime', INTERVAL '90 days');",
            "SELECT add_retention_policy('market_anomaly_detection', INTERVAL '90 days');"
        ]
        
        for i, command in enumerate(policy_commands, 1):
            try:
                logger.info(f"Setting up policy {i}/12...")
                await conn.execute(command)
                logger.info(f"✅ Policy {i} set up successfully")
                    
            except Exception as e:
                logger.warning(f"⚠️ Policy {i} setup warning: {e}")
        
        await conn.close()
        logger.info("✅ Enhanced Market Intelligence Tables Migration completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Enhanced Market Intelligence Tables Migration...")
    
    try:
        success = await create_enhanced_market_intelligence_tables()
        if success:
            logger.info("✅ Enhanced Market Intelligence Tables Migration completed successfully!")
        else:
            logger.error("❌ Migration failed")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
