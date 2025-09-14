#!/usr/bin/env python3
"""
Migration: ML + Risk Integration Tables
Create tables for storing actionable trade signals with ML + Risk integration
"""

import logging
from sqlalchemy import create_engine, text
from datetime import datetime
import os

logger = logging.getLogger(__name__)

def run_migration():
    """Run the ML + Risk integration migration"""
    
    # Database connection
    database_url = os.getenv("DATABASE_URL", "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse")
    engine = create_engine(database_url)
    
    logger.info("🚀 Starting ML + Risk Integration Migration...")
    
    try:
        # Create actionable_trade_signals table
        logger.info("Creating actionable_trade_signals table...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS actionable_trade_signals (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    signal_strength VARCHAR(20) NOT NULL,
                    confidence_score DECIMAL(5,4) NOT NULL,
                    risk_level VARCHAR(20) NOT NULL,
                    market_regime VARCHAR(20) NOT NULL,
                    recommended_leverage INTEGER NOT NULL,
                    position_size_usdt DECIMAL(15,2) NOT NULL,
                    stop_loss_price DECIMAL(15,6) NOT NULL,
                    take_profit_price DECIMAL(15,6) NOT NULL,
                    risk_reward_ratio DECIMAL(8,4) NOT NULL,
                    ml_confidence DECIMAL(5,4) NOT NULL,
                    ml_prediction VARCHAR(20) NOT NULL,
                    model_contributions JSONB,
                    risk_score DECIMAL(5,2) NOT NULL,
                    liquidation_risk DECIMAL(5,2) NOT NULL,
                    portfolio_impact DECIMAL(8,6) NOT NULL,
                    volatility_score DECIMAL(5,4) NOT NULL,
                    liquidity_score DECIMAL(5,4) NOT NULL,
                    market_depth_analysis JSONB,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            conn.commit()
        
        # Create TimescaleDB hypertable
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable('actionable_trade_signals', 'timestamp', 
                                           if_not_exists => TRUE)
                """))
                conn.commit()
                logger.info("✅ Created TimescaleDB hypertable for actionable_trade_signals")
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Create indexes for actionable_trade_signals
        logger.info("Creating indexes for actionable_trade_signals...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_actionable_signals_symbol 
                ON actionable_trade_signals (symbol, timestamp DESC)
            """))
            conn.commit()
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_actionable_signals_signal_type 
                ON actionable_trade_signals (signal_type, timestamp DESC)
            """))
            conn.commit()
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_actionable_signals_confidence 
                ON actionable_trade_signals (confidence_score DESC, timestamp DESC)
            """))
            conn.commit()
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_actionable_signals_risk_level 
                ON actionable_trade_signals (risk_level, timestamp DESC)
            """))
            conn.commit()
        
        # Create ml_risk_integration_metrics table
        logger.info("Creating ml_risk_integration_metrics table...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ml_risk_integration_metrics (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    service_name VARCHAR(50) NOT NULL,
                    signals_generated INTEGER DEFAULT 0,
                    signals_executed INTEGER DEFAULT 0,
                    average_confidence DECIMAL(5,4) DEFAULT 0.0,
                    average_risk_score DECIMAL(5,2) DEFAULT 0.0,
                    success_rate DECIMAL(5,4) DEFAULT 0.0,
                    total_pnl DECIMAL(15,2) DEFAULT 0.0,
                    processing_time_ms INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            conn.commit()
        
        # Create TimescaleDB hypertable for metrics
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable('ml_risk_integration_metrics', 'timestamp', 
                                           if_not_exists => TRUE)
                """))
                conn.commit()
                logger.info("✅ Created TimescaleDB hypertable for ml_risk_integration_metrics")
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Create indexes for metrics
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_ml_risk_metrics_service 
                ON ml_risk_integration_metrics (service_name, timestamp DESC)
            """))
            conn.commit()
        
        # Create signal_execution_logs table
        logger.info("Creating signal_execution_logs table...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS signal_execution_logs (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    signal_id INTEGER NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    execution_status VARCHAR(20) NOT NULL,
                    execution_price DECIMAL(15,6),
                    execution_size DECIMAL(15,2),
                    execution_leverage INTEGER,
                    actual_stop_loss DECIMAL(15,6),
                    actual_take_profit DECIMAL(15,6),
                    execution_time_ms INTEGER,
                    error_message TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            conn.commit()
        
        # Create TimescaleDB hypertable for execution logs
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable('signal_execution_logs', 'timestamp', 
                                           if_not_exists => TRUE)
                """))
                conn.commit()
                logger.info("✅ Created TimescaleDB hypertable for signal_execution_logs")
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Create indexes for execution logs
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_execution_logs_signal_id 
                ON signal_execution_logs (signal_id, timestamp DESC)
            """))
            conn.commit()
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_execution_logs_status 
                ON signal_execution_logs (execution_status, timestamp DESC)
            """))
            conn.commit()
        
        # Create risk_adjusted_positions table
        logger.info("Creating risk_adjusted_positions table...")
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS risk_adjusted_positions (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    base_position_size DECIMAL(15,2) NOT NULL,
                    risk_adjusted_size DECIMAL(15,2) NOT NULL,
                    leverage_multiplier DECIMAL(8,4) NOT NULL,
                    stop_loss_adjustment DECIMAL(8,4) NOT NULL,
                    take_profit_adjustment DECIMAL(8,4) NOT NULL,
                    risk_factors JSONB,
                    confidence_boost DECIMAL(5,4) NOT NULL,
                    market_regime VARCHAR(20) NOT NULL,
                    volatility_score DECIMAL(5,4) NOT NULL,
                    liquidity_score DECIMAL(5,4) NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            conn.commit()
        
        # Create TimescaleDB hypertable for risk adjusted positions
        try:
            with engine.connect() as conn:
                conn.execute(text("""
                    SELECT create_hypertable('risk_adjusted_positions', 'timestamp', 
                                           if_not_exists => TRUE)
                """))
                conn.commit()
                logger.info("✅ Created TimescaleDB hypertable for risk_adjusted_positions")
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Create indexes for risk adjusted positions
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_risk_positions_symbol 
                ON risk_adjusted_positions (symbol, timestamp DESC)
            """))
            conn.commit()
        
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_risk_positions_signal_type 
                ON risk_adjusted_positions (signal_type, timestamp DESC)
            """))
            conn.commit()
        
        # Insert default configuration
        logger.info("Inserting default configuration...")
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO ml_risk_integration_metrics (
                    service_name, signals_generated, signals_executed, 
                    average_confidence, average_risk_score, success_rate, total_pnl
                ) VALUES (
                    'ml_risk_integration_service', 0, 0, 0.0, 0.0, 0.0, 0.0
                ) ON CONFLICT DO NOTHING
            """))
            conn.commit()
        
        logger.info("✅ ML + Risk Integration Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migration()
