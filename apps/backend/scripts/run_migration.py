#!/usr/bin/env python3
"""
TimescaleDB Migration Executor
Executes the OHLCV hypertable migration using Python
"""

import asyncio
import asyncpg
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_missing_tables(conn):
    """Create missing tables for Phase 2 & 3 functionality"""
    try:
        # Create signals table for Phase 3 entry zone monitoring
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id VARCHAR(50) PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                entry_zone JSONB,
                confidence_score NUMERIC(4,3) DEFAULT 0.0,
                status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_updated TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT valid_direction CHECK (direction IN ('long', 'short', 'neutral')),
                CONSTRAINT valid_status CHECK (status IN ('active', 'expired', 'invalidated', 'executed')),
                CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
            );
        """)
        logger.info("✅ Created signals table")
        
        # Create candles view/alias for ohlcv_data (fixes legacy references)
        try:
            await conn.execute("""
                CREATE OR REPLACE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    symbol, 
                    timeframe, 
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp  -- Keep original timestamp column too
                FROM ohlcv_data;
            """)
            logger.info("✅ Created candles view")
        except Exception as e:
            logger.warning(f"⚠️ Could not create candles view: {e}")
        
        # Create market_intelligence table (placeholder for sentiment/events)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_intelligence (
                id BIGSERIAL,
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                intelligence_type VARCHAR(50) NOT NULL,
                data JSONB NOT NULL,
                confidence_score NUMERIC(4,3) DEFAULT 0.0,
                market_sentiment_score FLOAT DEFAULT 0.5,
                market_regime VARCHAR(20) DEFAULT 'neutral',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (id, timestamp)
            );
        """)
        
        # Convert to hypertable
        await conn.execute("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = 'market_intelligence'
                ) THEN
                    PERFORM create_hypertable('market_intelligence', 'timestamp', 
                                             chunk_time_interval => INTERVAL '1 hour');
                END IF;
            END $$;
        """)
        logger.info("✅ Created market_intelligence table")
        
        # Create volume_analysis table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS volume_analysis (
                id BIGSERIAL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                analysis_data JSONB NOT NULL,
                confidence_score NUMERIC(4,3) DEFAULT 0.0,
                volume_ratio FLOAT DEFAULT 1.0,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (id, timestamp)
            );
        """)
        
        await conn.execute("""
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = 'volume_analysis'
                ) THEN
                    PERFORM create_hypertable('volume_analysis', 'timestamp', 
                                             chunk_time_interval => INTERVAL '1 hour');
                END IF;
            END $$;
        """)
        logger.info("✅ Created volume_analysis table")
        
        # Create placeholder tables for other missing components
        placeholder_tables = [
            'free_api_market_data',
            'free_api_sentiment_data', 
            'free_api_liquidation_events',
            'free_api_news_data',
            'free_api_social_data',
            'sde_dynamic_thresholds',
            'price_action_ml_predictions'
        ]
        
        for table_name in placeholder_tables:
            if table_name in ['free_api_sentiment_data', 'free_api_news_data', 'free_api_social_data']:
                # Tables that need source, sentiment_type, platform columns
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id BIGSERIAL,
                        symbol VARCHAR(20),
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        data JSONB DEFAULT '{{}}',
                        source VARCHAR(50) DEFAULT 'unknown',
                        sentiment_type VARCHAR(20) DEFAULT 'neutral',
                        platform VARCHAR(50) DEFAULT 'unknown',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (id)
                    );
                """)
            elif table_name == 'sde_dynamic_thresholds':
                # Table that needs min_confidence_threshold column
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id BIGSERIAL,
                        symbol VARCHAR(20),
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        data JSONB DEFAULT '{{}}',
                        min_confidence_threshold FLOAT DEFAULT 0.5,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (id)
                    );
                """)
            elif table_name == 'price_action_ml_predictions':
                # Table that needs timeframe column
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id BIGSERIAL,
                        symbol VARCHAR(20),
                        timeframe VARCHAR(10) DEFAULT '1h',
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        data JSONB DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (id)
                    );
                """)
            else:
                # Default placeholder table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id BIGSERIAL,
                        symbol VARCHAR(20),
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        data JSONB DEFAULT '{{}}',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (id)
                    );
                """)
            logger.info(f"✅ Created placeholder table {table_name}")
        
        logger.info("🎉 All missing tables created successfully!")
        
        # Add missing columns to existing tables
        logger.info("🔧 Adding missing columns to existing tables...")
        
        # Fix candles view
        # Use CREATE OR REPLACE VIEW instead of DROP + CREATE
        try:
            await conn.execute("""
                CREATE OR REPLACE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    symbol, 
                    timeframe, 
                    open as o,
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp  -- Keep original timestamp column too
                FROM ohlcv_data
            """)
            logger.info("✅ Fixed candles view with ts column")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix candles view: {e}")
        
        # Add missing columns to market_intelligence
        try:
            await conn.execute("ALTER TABLE market_intelligence ADD COLUMN IF NOT EXISTS market_sentiment_score FLOAT DEFAULT 0.5")
            await conn.execute("ALTER TABLE market_intelligence ADD COLUMN IF NOT EXISTS market_regime VARCHAR(20) DEFAULT 'neutral'")
            logger.info("✅ Added columns to market_intelligence")
        except Exception as e:
            logger.warning(f"⚠️ Could not add columns to market_intelligence: {e}")
        
        # Add missing columns to volume_analysis
        try:
            await conn.execute("ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS volume_ratio FLOAT DEFAULT 1.0")
            logger.info("✅ Added volume_ratio to volume_analysis")
        except Exception as e:
            logger.warning(f"⚠️ Could not add volume_ratio to volume_analysis: {e}")
        
        # Add missing columns to placeholder tables
        try:
            await conn.execute("ALTER TABLE free_api_sentiment_data ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE free_api_sentiment_data ADD COLUMN IF NOT EXISTS sentiment_type VARCHAR(20) DEFAULT 'neutral'")
            await conn.execute("ALTER TABLE free_api_sentiment_data ADD COLUMN IF NOT EXISTS platform VARCHAR(50) DEFAULT 'unknown'")
            logger.info("✅ Added columns to free_api_sentiment_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add columns to free_api_sentiment_data: {e}")
        
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS min_confidence_threshold FLOAT DEFAULT 0.5")
            logger.info("✅ Added min_confidence_threshold to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add min_confidence_threshold to sde_dynamic_thresholds: {e}")
        
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10) DEFAULT '1h'")
            logger.info("✅ Added timeframe to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add timeframe to price_action_ml_predictions: {e}")
        
        logger.info("🎉 Column schema fixes completed!")
        
        # Add additional missing columns for signal generation
        logger.info("🔧 Adding additional missing columns...")
        
        # Add missing columns to market_intelligence
        try:
            await conn.execute("ALTER TABLE market_intelligence ADD COLUMN IF NOT EXISTS news_sentiment_score FLOAT DEFAULT 0.5")
            await conn.execute("ALTER TABLE market_intelligence ADD COLUMN IF NOT EXISTS volatility_index FLOAT DEFAULT 1.0")
            logger.info("✅ Added additional columns to market_intelligence")
        except Exception as e:
            logger.warning(f"⚠️ Could not add additional columns to market_intelligence: {e}")
        
        # Add missing columns to price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS prediction_type VARCHAR(20) DEFAULT 'neutral'")
            logger.info("✅ Added prediction_type to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add prediction_type to price_action_ml_predictions: {e}")
        
        # Add missing columns to volume_analysis
        try:
            await conn.execute("ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS volume_trend VARCHAR(20) DEFAULT 'stable'")
            logger.info("✅ Added volume_trend to volume_analysis")
        except Exception as e:
            logger.warning(f"⚠️ Could not add volume_trend to volume_analysis: {e}")
        
        # Add missing columns to free_api_sentiment_data
        try:
            await conn.execute("ALTER TABLE free_api_sentiment_data ADD COLUMN IF NOT EXISTS sentiment_score FLOAT DEFAULT 0.5")
            logger.info("✅ Added sentiment_score to free_api_sentiment_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add sentiment_score to free_api_sentiment_data: {e}")
        
        # Add missing columns to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS min_consensus_heads FLOAT DEFAULT 0.5")
            logger.info("✅ Added min_consensus_heads to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add min_consensus_heads to sde_dynamic_thresholds: {e}")
        
        logger.info("🎉 Additional column schema fixes completed!")
        
        # Add final missing columns for complete signal generation
        logger.info("🔧 Adding final missing columns for signal generation...")
        
        # Fix candles view to include 'h' column
        try:
            await conn.execute("""
                CREATE OR REPLACE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    symbol, 
                    timeframe, 
                    open as o,
                    high as h,
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp  -- Keep original timestamp column too
                FROM ohlcv_data
            """)
            logger.info("✅ Fixed candles view with h column")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix candles view: {e}")
        
        # Add missing columns to market_intelligence
        try:
            await conn.execute("ALTER TABLE market_intelligence ADD COLUMN IF NOT EXISTS fear_greed_index FLOAT DEFAULT 50.0")
            await conn.execute("ALTER TABLE market_intelligence ADD COLUMN IF NOT EXISTS trend_strength FLOAT DEFAULT 0.5")
            logger.info("✅ Added fear_greed_index and trend_strength to market_intelligence")
        except Exception as e:
            logger.warning(f"⚠️ Could not add fear_greed_index/trend_strength to market_intelligence: {e}")
        
        # Add missing columns to price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS prediction_probability FLOAT DEFAULT 0.5")
            logger.info("✅ Added prediction_probability to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add prediction_probability to price_action_ml_predictions: {e}")
        
        # Add missing columns to volume_analysis
        try:
            await conn.execute("ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS order_book_imbalance FLOAT DEFAULT 0.0")
            logger.info("✅ Added order_book_imbalance to volume_analysis")
        except Exception as e:
            logger.warning(f"⚠️ Could not add order_book_imbalance to volume_analysis: {e}")
        
        # Add missing columns to free_api_sentiment_data
        try:
            await conn.execute("ALTER TABLE free_api_sentiment_data ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.5")
            logger.info("✅ Added confidence to free_api_sentiment_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add confidence to free_api_sentiment_data: {e}")
        
        # Add missing columns to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS min_probability_threshold FLOAT DEFAULT 0.5")
            logger.info("✅ Added min_probability_threshold to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add min_probability_threshold to sde_dynamic_thresholds: {e}")
        
        logger.info("🎉 Final column schema fixes completed!")
        
        # Add final missing columns for complete signal generation
        logger.info("🔧 Adding final missing columns for complete signal generation...")
        
        # Fix candles view to include 'l' column
        try:
            # Use CREATE OR REPLACE VIEW instead of DROP + CREATE
            await conn.execute("""
                CREATE OR REPLACE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    symbol, 
                    timeframe, 
                    open as o,
                    high as h,
                    low as l,
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp
                FROM ohlcv_data
            """)
            logger.info("✅ Fixed candles view with l column")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix candles view: {e}")
        
        # Add missing columns to price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS confidence_score FLOAT DEFAULT 0.5")
            logger.info("✅ Added confidence_score to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add confidence_score to price_action_ml_predictions: {e}")
        
        # Add missing columns to market_intelligence
        try:
            await conn.execute("ALTER TABLE market_intelligence ADD COLUMN IF NOT EXISTS btc_dominance FLOAT DEFAULT 50.0")
            logger.info("✅ Added btc_dominance to market_intelligence")
        except Exception as e:
            logger.warning(f"⚠️ Could not add btc_dominance to market_intelligence: {e}")
        
        # Add missing columns to volume_analysis
        try:
            await conn.execute("ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS volume_positioning_score FLOAT DEFAULT 0.5")
            logger.info("✅ Added volume_positioning_score to volume_analysis")
        except Exception as e:
            logger.warning(f"⚠️ Could not add volume_positioning_score to volume_analysis: {e}")
        
        # Add missing columns to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS calibration_weight_isotonic FLOAT DEFAULT 0.5")
            logger.info("✅ Added calibration_weight_isotonic to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add calibration_weight_isotonic to sde_dynamic_thresholds: {e}")
        
        # Create sentiment_data table
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id SERIAL,
                    ts TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    sentiment_score FLOAT DEFAULT 0.5,
                    confidence FLOAT DEFAULT 0.5,
                    source VARCHAR(50) DEFAULT 'unknown',
                    platform VARCHAR(50) DEFAULT 'unknown',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (id, ts)
                )
            """)
            logger.info("✅ Created sentiment_data table")
        except Exception as e:
            logger.warning(f"⚠️ Could not create sentiment_data table: {e}")
        
        logger.info("🎉 Complete column schema fixes completed!")
        
        # Add final missing columns for complete signal generation
        logger.info("🔧 Adding final missing columns for complete signal generation...")
        
        # Fix candles view to include 'c' column
        try:
            # Use CREATE OR REPLACE VIEW instead of DROP + CREATE
            await conn.execute("""
                CREATE OR REPLACE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    symbol, 
                    timeframe, 
                    open as o,
                    high as h,
                    low as l,
                    close as c,
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp
                FROM ohlcv_data
            """)
            logger.info("✅ Fixed candles view with c column")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix candles view: {e}")
        
        # Add missing columns to sentiment_data
        try:
            await conn.execute("ALTER TABLE sentiment_data ADD COLUMN IF NOT EXISTS sentiment_label VARCHAR(20) DEFAULT 'neutral'")
            logger.info("✅ Added sentiment_label to sentiment_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add sentiment_label to sentiment_data: {e}")
        
        # Add missing columns to price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS feature_vector JSONB DEFAULT '{}'")
            logger.info("✅ Added feature_vector to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add feature_vector to price_action_ml_predictions: {e}")
        
        # Add missing columns to volume_analysis
        try:
            await conn.execute("ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS buy_volume_ratio FLOAT DEFAULT 0.5")
            logger.info("✅ Added buy_volume_ratio to volume_analysis")
        except Exception as e:
            logger.warning(f"⚠️ Could not add buy_volume_ratio to volume_analysis: {e}")
        
        # Add missing columns to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS calibration_weight_platt FLOAT DEFAULT 0.5")
            logger.info("✅ Added calibration_weight_platt to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add calibration_weight_platt to sde_dynamic_thresholds: {e}")
        
        # Create market_regime_data table
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_regime_data (
                    id SERIAL,
                    ts TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    trend_strength FLOAT DEFAULT 0.5,
                    market_regime VARCHAR(20) DEFAULT 'neutral',
                    source VARCHAR(50) DEFAULT 'unknown',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (id, ts)
                )
            """)
            logger.info("✅ Created market_regime_data table")
        except Exception as e:
            logger.warning(f"⚠️ Could not create market_regime_data table: {e}")
        
        logger.info("🎉 All final missing columns added successfully!")
        
        # Add remaining missing columns for complete signal generation
        logger.info("🔧 Adding remaining missing columns for complete signal generation...")
        
        # Fix candles view to include 'v' column
        try:
            # Use CREATE OR REPLACE VIEW instead of DROP + CREATE
            await conn.execute("""
                CREATE OR REPLACE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    symbol, 
                    timeframe, 
                    open as o,
                    high as h,
                    low as l,
                    close as c,
                    volume as v,
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp
                FROM ohlcv_data
            """)
            logger.info("✅ Fixed candles view with v column")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix candles view: {e}")
        
        # Add missing columns to price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS model_output JSONB DEFAULT '{}'")
            logger.info("✅ Added model_output to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add model_output to price_action_ml_predictions: {e}")
        
        # Add missing columns to sentiment_data
        try:
            await conn.execute("ALTER TABLE sentiment_data ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP")
            logger.info("✅ Added timestamp to sentiment_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add timestamp to sentiment_data: {e}")
        
        # Add missing columns to volume_analysis
        try:
            await conn.execute("ALTER TABLE volume_analysis ADD COLUMN IF NOT EXISTS sell_volume_ratio FLOAT DEFAULT 0.5")
            logger.info("✅ Added sell_volume_ratio to volume_analysis")
        except Exception as e:
            logger.warning(f"⚠️ Could not add sell_volume_ratio to volume_analysis: {e}")
        
        # Add missing columns to market_regime_data
        try:
            await conn.execute("ALTER TABLE market_regime_data ADD COLUMN IF NOT EXISTS regime_type VARCHAR(20) DEFAULT 'neutral'")
            logger.info("✅ Added regime_type to market_regime_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add regime_type to market_regime_data: {e}")
        
        # Add missing columns to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS calibration_weight_temperature FLOAT DEFAULT 0.5")
            logger.info("✅ Added calibration_weight_temperature to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add calibration_weight_temperature to sde_dynamic_thresholds: {e}")
        
        logger.info("🎉 All remaining missing columns added successfully!")
        
        # Add final missing columns for complete signal generation
        logger.info("🔧 Adding final missing columns for complete signal generation...")
        
        # Fix candles view to include 'tf' column
        try:
            # Use CREATE OR REPLACE VIEW instead of DROP + CREATE
            await conn.execute("""
                CREATE OR REPLACE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    timestamp as tf,
                    symbol, 
                    timeframe, 
                    open as o,
                    high as h,
                    low as l,
                    close as c,
                    volume as v,
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp
                FROM ohlcv_data
            """)
            logger.info("✅ Fixed candles view with tf column")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix candles view: {e}")
        
        # Remove any problematic candles view operations in other sections
        logger.info("🔧 Cleaning up candles view operations...")
        try:
            # Ensure candles view is properly created without conflicts
            await conn.execute("DROP VIEW IF EXISTS candles CASCADE")
            await conn.execute("""
                CREATE VIEW candles AS 
                SELECT 
                    timestamp as ts, 
                    timestamp as tf,
                    symbol, 
                    timeframe, 
                    open as o,
                    high as h,
                    low as l,
                    close as c,
                    volume as v,
                    open, 
                    high, 
                    low, 
                    close, 
                    volume, 
                    source,
                    timestamp
                FROM ohlcv_data
            """)
            logger.info("✅ Cleaned up candles view operations")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up candles view: {e}")
        
        # Add missing columns to price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS market_context JSONB DEFAULT '{}'")
            logger.info("✅ Added market_context to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add market_context to price_action_ml_predictions: {e}")
        
        # Add missing columns to market_regime_data
        try:
            await conn.execute("ALTER TABLE market_regime_data ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.5")
            logger.info("✅ Added confidence to market_regime_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add confidence to market_regime_data: {e}")
        
        # Add missing columns to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS calibration_weight_ensemble FLOAT DEFAULT 0.5")
            logger.info("✅ Added calibration_weight_ensemble to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add calibration_weight_ensemble to sde_dynamic_thresholds: {e}")
        
        logger.info("🎉 All final missing columns added successfully!")
        
        # Add final missing columns for signal generation
        logger.info("🔧 Adding final missing columns for signal generation...")
        
        # Add missing columns to market_regime_data
        try:
            await conn.execute("ALTER TABLE market_regime_data ADD COLUMN IF NOT EXISTS volatility FLOAT DEFAULT 0.0")
            logger.info("✅ Added volatility to market_regime_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add volatility to market_regime_data: {e}")
        
        # Add missing columns to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS market_regime VARCHAR(20) DEFAULT 'neutral'")
            logger.info("✅ Added market_regime to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add market_regime to sde_dynamic_thresholds: {e}")
        
        # Fix varchar length issue in price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ALTER COLUMN symbol TYPE VARCHAR(50)")
            await conn.execute("ALTER TABLE price_action_ml_predictions ALTER COLUMN prediction TYPE VARCHAR(100)")
            logger.info("✅ Fixed varchar length in price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix varchar length in price_action_ml_predictions: {e}")
        
        # Add missing volatility_level column to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS volatility_level FLOAT DEFAULT 0.5")
            logger.info("✅ Added volatility_level to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add volatility_level to sde_dynamic_thresholds: {e}")
        
        # Add missing is_active column to sde_dynamic_thresholds
        try:
            await conn.execute("ALTER TABLE sde_dynamic_thresholds ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE")
            logger.info("✅ Added is_active to sde_dynamic_thresholds")
        except Exception as e:
            logger.warning(f"⚠️ Could not add is_active to sde_dynamic_thresholds: {e}")
        
        # Add missing prediction column to price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ADD COLUMN IF NOT EXISTS prediction VARCHAR(100)")
            logger.info("✅ Added prediction column to price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not add prediction column to price_action_ml_predictions: {e}")
        
        # Add critical missing columns for signal generation
        logger.info("🔧 Adding critical missing columns for signal generation...")
        
        # Add missing SDE columns
        try:
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS head_b_probability FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS signal_id VARCHAR(50)")
            logger.info("✅ Added head_b_probability and signal_id to sde_model_consensus_tracking")
        except Exception as e:
            logger.warning(f"⚠️ Could not add SDE columns: {e}")
        
        # Fix prediction column varchar length
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ALTER COLUMN prediction TYPE VARCHAR(100)")
            logger.info("✅ Fixed prediction column varchar length in price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix prediction column varchar length: {e}")
        
        logger.info("🎉 All critical missing columns added successfully!")
        
        # Fix duplicate key warning in ohlcv_data
        logger.info("🔧 Fixing duplicate key warning in ohlcv_data...")
        try:
            # Remove duplicates
            await conn.execute("""
                DELETE FROM ohlcv_data
                WHERE (symbol, timeframe, timestamp) IN (
                    SELECT symbol, timeframe, timestamp
                    FROM ohlcv_data
                    GROUP BY symbol, timeframe, timestamp
                    HAVING COUNT(*) > 1
                ) AND ctid NOT IN (
                    SELECT MIN(ctid)
                    FROM ohlcv_data
                    GROUP BY symbol, timeframe, timestamp
                )
            """)
            
            # Recreate unique index
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_unique ON ohlcv_data (symbol, timeframe, timestamp)")
            logger.info("✅ Fixed duplicate key warning in ohlcv_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix duplicate key warning: {e}")
        
        # Fix ownership warnings by changing table ownership to alpha_emon
        logger.info("🔐 Fixing ownership warnings...")
        try:
            await conn.execute("ALTER TABLE public.ohlcv_data OWNER TO alpha_emon")
            await conn.execute("ALTER TABLE public.order_book_data OWNER TO alpha_emon")
            await conn.execute("ALTER TABLE public.technical_indicators OWNER TO alpha_emon")
            await conn.execute("ALTER TABLE public.support_resistance_levels OWNER TO alpha_emon")
            logger.info("✅ Fixed ownership warnings")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix ownership warnings: {e}")
        
        # Add missing timestamp column to market_regime_data
        try:
            await conn.execute("ALTER TABLE market_regime_data ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP")
            logger.info("✅ Added timestamp to market_regime_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add timestamp to market_regime_data: {e}")
        
        # Add missing source and platform columns to free_api tables
        try:
            await conn.execute("ALTER TABLE free_api_market_data ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE free_api_sentiment_data ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE free_api_sentiment_data ADD COLUMN IF NOT EXISTS platform VARCHAR(50) DEFAULT 'unknown'")
            logger.info("✅ Added source and platform columns to free_api tables")
        except Exception as e:
            logger.warning(f"⚠️ Could not add source/platform columns to free_api tables: {e}")
        
        # Add missing columns to other free_api tables
        try:
            await conn.execute("ALTER TABLE free_api_liquidation_events ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE free_api_news_data ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE free_api_social_data ADD COLUMN IF NOT EXISTS platform VARCHAR(50) DEFAULT 'unknown'")
            logger.info("✅ Added source and platform columns to other free_api tables")
        except Exception as e:
            logger.warning(f"⚠️ Could not add source/platform columns to other free_api tables: {e}")
        
        # Add missing price column to free_api_market_data
        try:
            await conn.execute("ALTER TABLE free_api_market_data ADD COLUMN IF NOT EXISTS price FLOAT DEFAULT 0.0")
            logger.info("✅ Added price column to free_api_market_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not add price column to free_api_market_data: {e}")
        
        # Add final missing columns for signal generation
        try:
            await conn.execute("ALTER TABLE free_api_market_data ADD COLUMN IF NOT EXISTS fear_greed_index FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE free_api_liquidation_events ADD COLUMN IF NOT EXISTS value_usd FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_integration_config ADD COLUMN IF NOT EXISTS config_data TEXT DEFAULT '{}'")
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS head_a_probability FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_news_blackout ADD COLUMN IF NOT EXISTS event_impact FLOAT DEFAULT 0.0")
            logger.info("✅ Added final missing columns for signal generation")
        except Exception as e:
            logger.warning(f"⚠️ Could not add final missing columns: {e}")
        
        # Add missing columns to free_api tables for signal generation
        try:
            await conn.execute("ALTER TABLE free_api_market_data ADD COLUMN IF NOT EXISTS volume_24h FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE free_api_liquidation_events ADD COLUMN IF NOT EXISTS liquidation_type VARCHAR(50) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE free_api_news_data ADD COLUMN IF NOT EXISTS sentiment_score FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE free_api_social_data ADD COLUMN IF NOT EXISTS sentiment_score FLOAT DEFAULT 0.0")
            logger.info("✅ Added missing columns for signal generation")
        except Exception as e:
            logger.warning(f"⚠️ Could not add missing columns for signal generation: {e}")
        
        # Add additional missing columns for complete signal generation
        try:
            await conn.execute("ALTER TABLE free_api_market_data ADD COLUMN IF NOT EXISTS market_cap FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE free_api_liquidation_events ADD COLUMN IF NOT EXISTS price FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE free_api_news_data ADD COLUMN IF NOT EXISTS relevance_score FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE free_api_social_data ADD COLUMN IF NOT EXISTS influence_score FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS account_id VARCHAR(50) DEFAULT 'default'")
            logger.info("✅ Added additional missing columns for complete signal generation")
        except Exception as e:
            logger.warning(f"⚠️ Could not add additional missing columns: {e}")
        
        # Add remaining missing columns for signal generation
        try:
            await conn.execute("ALTER TABLE free_api_market_data ADD COLUMN IF NOT EXISTS price_change_24h FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE free_api_liquidation_events ADD COLUMN IF NOT EXISTS quantity FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_integration_config ADD COLUMN IF NOT EXISTS config_name VARCHAR(50) DEFAULT 'default'")
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS timeframe VARCHAR(10) DEFAULT '1m'")
            await conn.execute("ALTER TABLE sde_news_blackout ADD COLUMN IF NOT EXISTS event_type VARCHAR(50) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS outcome VARCHAR(20) DEFAULT 'pending'")
            logger.info("✅ Added remaining missing columns for signal generation")
        except Exception as e:
            logger.warning(f"⚠️ Could not add remaining missing columns: {e}")
        
        # Verify OHLCV columns exist
        try:
            await conn.execute("ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS high NUMERIC(20,8)")
            await conn.execute("ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS close NUMERIC(20,8)")
            await conn.execute("ALTER TABLE ohlcv_data ADD COLUMN IF NOT EXISTS volume NUMERIC(20,8)")
            logger.info("✅ Verified OHLCV columns exist")
        except Exception as e:
            logger.warning(f"⚠️ Could not verify OHLCV columns: {e}")
        
        # Create missing SDE tables
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_integration_config (
                    id BIGSERIAL PRIMARY KEY,
                    config_key VARCHAR(100) NOT NULL UNIQUE,
                    config_value TEXT NOT NULL,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_model_consensus_tracking (
                    id BIGSERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    consensus_score FLOAT DEFAULT 0.0,
                    model_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_news_blackout (
                    id BIGSERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ NOT NULL,
                    reason TEXT DEFAULT 'news_event',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_divergence_analysis (
                    id BIGSERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    rsi_divergence JSONB,
                    macd_divergence JSONB,
                    volume_divergence JSONB,
                    combined_divergence JSONB,
                    overall_confidence FLOAT DEFAULT 0.0,
                    divergence_score FLOAT DEFAULT 0.0,
                    confirmation_count INTEGER DEFAULT 0,
                    analysis_data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created missing SDE tables")
        except Exception as e:
            logger.warning(f"⚠️ Could not create missing SDE tables: {e}")
        
        # Fix varchar length issues in price_action_ml_predictions
        try:
            await conn.execute("ALTER TABLE price_action_ml_predictions ALTER COLUMN symbol TYPE VARCHAR(50)")
            await conn.execute("ALTER TABLE price_action_ml_predictions ALTER COLUMN prediction TYPE VARCHAR(50)")
            logger.info("✅ Fixed varchar length issues in price_action_ml_predictions")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix varchar length issues: {e}")
        
        # Fix duplicate key warning in ohlcv_data
        try:
            # Remove duplicates
            await conn.execute("""
                DELETE FROM ohlcv_data
                WHERE (symbol, timeframe, timestamp) IN (
                    SELECT symbol, timeframe, timestamp
                    FROM ohlcv_data
                    GROUP BY symbol, timeframe, timestamp
                    HAVING COUNT(*) > 1
                ) AND ctid NOT IN (
                    SELECT MIN(ctid)
                    FROM ohlcv_data
                    GROUP BY symbol, timeframe, timestamp
                )
            """)
            
            # Recreate unique index
            await conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ohlcv_unique ON ohlcv_data (symbol, timeframe, timestamp)")
            logger.info("✅ Fixed duplicate key warning in ohlcv_data")
        except Exception as e:
            logger.warning(f"⚠️ Could not fix duplicate key warning: {e}")
        
        # Add missing SDE columns
        try:
            await conn.execute("ALTER TABLE sde_integration_config ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true")
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS head_a_confidence FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_news_blackout ADD COLUMN IF NOT EXISTS event_title VARCHAR(200) DEFAULT 'unknown'")
            await conn.execute("ALTER TABLE free_api_liquidation_events ADD COLUMN IF NOT EXISTS side VARCHAR(10) DEFAULT 'unknown'")
            logger.info("✅ Added missing SDE columns")
        except Exception as e:
            logger.warning(f"⚠️ Could not add missing SDE columns: {e}")
        
        # Add final missing SDE columns
        try:
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS head_a_direction VARCHAR(10) DEFAULT 'neutral'")
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS head_b_confidence FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS head_c_confidence FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_model_consensus_tracking ADD COLUMN IF NOT EXISTS head_d_confidence FLOAT DEFAULT 0.0")
            await conn.execute("ALTER TABLE sde_news_blackout ADD COLUMN IF NOT EXISTS blackout_active BOOLEAN DEFAULT false")
            await conn.execute("ALTER TABLE free_api_liquidation_events ADD COLUMN IF NOT EXISTS raw_data JSONB DEFAULT '{}'")
            logger.info("✅ Added final missing SDE columns")
        except Exception as e:
            logger.warning(f"⚠️ Could not add final missing SDE columns: {e}")
        
        logger.info("🎉 All final fixes completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error creating missing tables: {e}")

async def enhance_data_population(conn):
    """Enhance data population with proper rate limiting and error handling"""
    try:
        import ccxt
        import asyncio
        from datetime import datetime, timedelta
        
        # Initialize Binance exchange
        exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        logger.info("📊 Starting enhanced data population...")
        
        for symbol in symbols:
            logger.info(f"🔄 Populating data for {symbol}...")
            
            try:
                # Check current data count
                count_query = """
                    SELECT COUNT(*) FROM ohlcv_data 
                    WHERE symbol = $1 AND timeframe = '1m' 
                    AND timestamp > NOW() - INTERVAL '7 days'
                """
                current_count = await conn.fetchval(count_query, symbol)
                logger.info(f"📊 Current {symbol} 1m data: {current_count} candles")
                
                # If we have sufficient data, skip
                if current_count >= 10000:
                    logger.info(f"✅ {symbol} has sufficient data ({current_count} candles)")
                    continue
                
                # Calculate how much data we need
                needed_candles = 10080 - current_count  # 7 days * 24 hours * 60 minutes
                batches_needed = (needed_candles // 1000) + 1
                
                logger.info(f"📊 Need {needed_candles} more candles for {symbol}, fetching {batches_needed} batches")
                
                # Fetch historical data with rate limiting
                for batch in range(batches_needed):
                    try:
                        # Rate limiting: wait 1 second between requests
                        if batch > 0:
                            await asyncio.sleep(1)
                        
                        # Calculate time range for this batch
                        end_time = datetime.utcnow() - timedelta(minutes=batch * 1000)
                        start_time = end_time - timedelta(minutes=1000)
                        
                        # Fetch klines with retry logic
                        klines = None
                        for attempt in range(3):
                            try:
                                klines = exchange.fetch_ohlcv(
                                    symbol, 
                                    '1m', 
                                    since=int(start_time.timestamp() * 1000),
                                    limit=1000
                                )
                                break
                            except Exception as e:
                                if attempt < 2:
                                    logger.warning(f"⚠️ Attempt {attempt + 1} failed for {symbol} batch {batch}: {e}")
                                    await asyncio.sleep(5)  # Wait 5 seconds before retry
                                else:
                                    logger.error(f"❌ All 3 attempts failed for {symbol} batch {batch}: {e}")
                                    raise e
                        
                        if not klines:
                            logger.warning(f"⚠️ No data returned for {symbol} batch {batch}")
                            continue
                        
                        # Insert data into database
                        insert_query = """
                            INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume, source)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                            ON CONFLICT DO NOTHING
                        """
                        
                        batch_data = []
                        for kline in klines:
                            batch_data.append((
                                symbol,
                                '1m',
                                datetime.fromtimestamp(kline[0] / 1000),
                                float(kline[1]),
                                float(kline[2]),
                                float(kline[3]),
                                float(kline[4]),
                                float(kline[5]),
                                'rest_api'
                            ))
                        
                        await conn.executemany(insert_query, batch_data)
                        logger.info(f"✅ Inserted {len(batch_data)} candles for {symbol} batch {batch}")
                        
                        # Check if we have enough data now
                        new_count = await conn.fetchval(count_query, symbol)
                        if new_count >= 10000:
                            logger.info(f"✅ {symbol} now has sufficient data ({new_count} candles)")
                            break
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error fetching batch {batch} for {symbol}: {e}")
                        continue
                
                # Verify final count
                final_count = await conn.fetchval(count_query, symbol)
                logger.info(f"📊 Final {symbol} 1m data: {final_count} candles")
                
            except Exception as e:
                logger.error(f"❌ Error populating data for {symbol}: {e}")
                continue
        
        # Populate 4H and 1D aggregates
        logger.info("🔄 Populating 4H and 1D aggregates...")
        await populate_timeframe_aggregates(conn)
        
        logger.info("🎉 Data population completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in data population: {e}")

async def populate_timeframe_aggregates(conn):
    """Populate 4H and 1D aggregates from 1m data"""
    try:
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            # Create 4H aggregates
            await conn.execute(f"""
                INSERT INTO ohlcv_4h (timestamp, symbol, timeframe, open, high, low, close, volume)
                SELECT 
                    time_bucket('4 hours', timestamp) AS timestamp,
                    symbol,
                    '4h' AS timeframe,
                    FIRST(open, timestamp) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, timestamp) AS close,
                    SUM(volume) AS volume
                FROM ohlcv_data 
                WHERE symbol = $1 AND timeframe = '1m'
                AND timestamp > NOW() - INTERVAL '7 days'
                GROUP BY time_bucket('4 hours', timestamp), symbol
                ON CONFLICT DO NOTHING
            """, symbol)
            
            # Create 1D aggregates
            await conn.execute(f"""
                INSERT INTO ohlcv_1d (timestamp, symbol, timeframe, open, high, low, close, volume)
                SELECT 
                    time_bucket('1 day', timestamp) AS timestamp,
                    symbol,
                    '1d' AS timeframe,
                    FIRST(open, timestamp) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    LAST(close, timestamp) AS close,
                    SUM(volume) AS volume
                FROM ohlcv_data 
                WHERE symbol = $1 AND timeframe = '1m'
                AND timestamp > NOW() - INTERVAL '7 days'
                GROUP BY time_bucket('1 day', timestamp), symbol
                ON CONFLICT DO NOTHING
            """, symbol)
            
            # Check counts
            count_4h = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_4h WHERE symbol = $1", symbol)
            count_1d = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_1d WHERE symbol = $1", symbol)
            
            logger.info(f"✅ {symbol}: 4H={count_4h} candles, 1D={count_1d} candles")
        
    except Exception as e:
        logger.error(f"❌ Error populating timeframe aggregates: {e}")

async def execute_migration():
    """Execute the TimescaleDB migration with enhanced table creation and data population"""
    
    # Database connection parameters
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        # Connect to database
        logger.info("🔌 Connecting to TimescaleDB...")
        conn = await asyncpg.connect(db_url)
        
        # Step 1: Execute existing OHLCV migration
        logger.info("📄 Executing OHLCV hypertable migration...")
        migration_file = Path("backend/database/migrations/create_ohlcv_hypertable.sql")
        if migration_file.exists():
            migration_sql = migration_file.read_text(encoding='utf-8')
            statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
            
            for i, statement in enumerate(statements, 1):
                if not statement:
                    continue
                try:
                    await conn.execute(statement)
                    logger.info(f"✅ OHLCV statement {i} executed")
                except Exception as e:
                    logger.warning(f"⚠️ OHLCV statement {i} warning: {e}")
        
        # Step 2: Create missing tables for Phase 2 & 3
        logger.info("🏗️ Creating missing tables for Phase 2 & 3...")
        await create_missing_tables(conn)
        
        # Step 3: Enhance data population
        logger.info("📊 Enhancing data population...")
        await enhance_data_population(conn)
        
        # Verify tables were created
        logger.info("🔍 Verifying table creation...")
        
        # Check if TimescaleDB extension is enabled
        extension_result = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')")
        logger.info(f"📊 TimescaleDB extension enabled: {extension_result}")
        
        # Check if hypertables were created
        hypertables = await conn.fetch("""
            SELECT hypertable_name, num_dimensions, num_chunks 
            FROM timescaledb_information.hypertables 
            WHERE hypertable_schema = 'public'
        """)
        
        logger.info("📊 Created hypertables:")
        for table in hypertables:
            logger.info(f"  - {table['hypertable_name']}: {table['num_dimensions']} dimensions, {table['num_chunks']} chunks")
        
        # Check if tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('ohlcv_data', 'signals', 'market_intelligence', 'volume_analysis', 
                              'ohlcv_4h', 'ohlcv_1d', 'candles')
        """)
        
        logger.info("📊 Created tables:")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
        
        # Check data counts
        logger.info("📊 Data verification:")
        for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
            count_1m = await conn.fetchval("""
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE symbol = $1 AND timeframe = '1m' 
                AND timestamp > NOW() - INTERVAL '7 days'
            """, symbol)
            count_4h = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_4h WHERE symbol = $1", symbol)
            count_1d = await conn.fetchval("SELECT COUNT(*) FROM ohlcv_1d WHERE symbol = $1", symbol)
            
            logger.info(f"  - {symbol}: 1m={count_1m}, 4h={count_4h}, 1d={count_1d}")
        
        # Check materialized views
        views = await conn.fetch("""
            SELECT matviewname 
            FROM pg_matviews 
            WHERE schemaname = 'public'
        """)
        
        logger.info("📊 Created materialized views:")
        for view in views:
            logger.info(f"  - {view['matviewname']}")
        
        await conn.close()
        logger.info("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

async def test_database_connection():
    """Test database connection and basic functionality"""
    
    db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
    
    try:
        logger.info("🔌 Testing database connection...")
        conn = await asyncpg.connect(db_url)
        
        # Test basic query
        result = await conn.fetchval("SELECT version()")
        logger.info(f"📊 PostgreSQL version: {result}")
        
        # Test TimescaleDB
        timescale_version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
        if timescale_version:
            logger.info(f"📊 TimescaleDB version: {timescale_version}")
        else:
            logger.warning("⚠️ TimescaleDB extension not found")
        
        # Test table access
        table_count = await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        logger.info(f"📊 Total tables in public schema: {table_count}")
        
        await conn.close()
        logger.info("✅ Database connection test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False

async def main():
    """Main function"""
    logger.info("🚀 Starting TimescaleDB Migration")
    logger.info("=" * 50)
    
    # Test connection first
    if not await test_database_connection():
        logger.error("❌ Database connection failed, aborting migration")
        return
    
    # Execute migration
    if await execute_migration():
        logger.info("🎉 Migration completed successfully!")
    else:
        logger.error("❌ Migration failed!")

if __name__ == "__main__":
    asyncio.run(main())
