#!/usr/bin/env python3
"""
Database Setup for Noise Filtering and Adaptive Learning
Creates tables for pattern performance tracking, market regime classification, and adaptive confidence models
"""

import psycopg2
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_noise_filtering_database():
    """Setup database tables for noise filtering and adaptive learning"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',
        'user': 'postgres',
        'password': 'Emon_@17711'
    }
    
    conn = None
    cursor = None
    
    try:
        print("🔌 Connecting to database...")
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        print("✅ Database connection established")
        
        # Create pattern performance tracking table
        print("📝 Creating pattern_performance_tracking table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_performance_tracking (
                timestamp TIMESTAMPTZ NOT NULL,
                tracking_id VARCHAR(100) NOT NULL,
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_name VARCHAR(50) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                pattern_confidence DECIMAL(5,4) NOT NULL,
                predicted_outcome VARCHAR(20) NOT NULL, -- 'success', 'failure', 'neutral'
                actual_outcome VARCHAR(20), -- 'success', 'failure', 'neutral', NULL if not yet known
                market_regime VARCHAR(20) NOT NULL, -- 'trending', 'sideways', 'volatile', 'consolidation'
                volume_ratio DECIMAL(8,4) NOT NULL, -- volume / average_volume
                volatility_level DECIMAL(8,4) NOT NULL, -- ATR / price
                spread_impact DECIMAL(8,4) NOT NULL, -- bid_ask_spread / price
                noise_filter_score DECIMAL(5,4) NOT NULL, -- 0-1 score from noise filter
                performance_score DECIMAL(5,4), -- calculated performance score
                outcome_timestamp TIMESTAMPTZ, -- when actual outcome was determined
                outcome_price DECIMAL(18,8), -- price at outcome determination
                profit_loss DECIMAL(18,8), -- P&L if applicable
                market_conditions JSONB, -- additional market condition data
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, tracking_id)
            )
        """)
        
        # Create market regime classification table
        print("📝 Creating market_regime_classification table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regime_classification (
                timestamp TIMESTAMPTZ NOT NULL,
                regime_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                regime_type VARCHAR(20) NOT NULL, -- 'trending', 'sideways', 'volatile', 'consolidation'
                regime_confidence DECIMAL(5,4) NOT NULL,
                trend_strength DECIMAL(5,4) NOT NULL, -- 0-1 strength of trend
                volatility_level DECIMAL(8,4) NOT NULL, -- ATR / price
                volume_profile VARCHAR(20) NOT NULL, -- 'normal', 'high', 'low', 'spike'
                momentum_score DECIMAL(5,4) NOT NULL, -- momentum indicator score
                support_resistance_proximity DECIMAL(5,4) NOT NULL, -- distance to S/R levels
                market_microstructure JSONB, -- spread, depth, liquidity metrics
                regime_features JSONB, -- technical indicators for regime classification
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, regime_id)
            )
        """)
        
        # Create adaptive confidence models table
        print("📝 Creating adaptive_confidence_models table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_confidence_models (
                timestamp TIMESTAMPTZ NOT NULL,
                model_id VARCHAR(100) NOT NULL,
                pattern_name VARCHAR(50) NOT NULL,
                market_regime VARCHAR(20) NOT NULL,
                model_version VARCHAR(20) NOT NULL,
                model_type VARCHAR(30) NOT NULL, -- 'linear', 'ensemble', 'neural_network'
                feature_weights JSONB NOT NULL, -- weights for different features
                performance_metrics JSONB NOT NULL, -- accuracy, precision, recall, f1
                training_data_size INTEGER NOT NULL,
                last_training_timestamp TIMESTAMPTZ NOT NULL,
                model_parameters JSONB, -- model-specific parameters
                validation_score DECIMAL(5,4) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, model_id)
            )
        """)
        
        # Create noise filter settings table
        print("📝 Creating noise_filter_settings table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS noise_filter_settings (
                timestamp TIMESTAMPTZ NOT NULL,
                setting_id VARCHAR(100) NOT NULL,
                filter_type VARCHAR(30) NOT NULL, -- 'volume', 'volatility', 'time', 'spread'
                filter_name VARCHAR(50) NOT NULL,
                filter_parameters JSONB NOT NULL, -- configurable parameters
                is_active BOOLEAN DEFAULT TRUE,
                priority INTEGER DEFAULT 1, -- filter priority order
                description TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, setting_id)
            )
        """)
        
        # Create pattern quality metrics table
        print("📝 Creating pattern_quality_metrics table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_quality_metrics (
                timestamp TIMESTAMPTZ NOT NULL,
                metric_id VARCHAR(100) NOT NULL,
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_name VARCHAR(50) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                quality_score DECIMAL(5,4) NOT NULL, -- overall quality score 0-1
                volume_quality DECIMAL(5,4) NOT NULL, -- volume-based quality
                volatility_quality DECIMAL(5,4) NOT NULL, -- volatility-based quality
                spread_quality DECIMAL(5,4) NOT NULL, -- spread-based quality
                time_quality DECIMAL(5,4) NOT NULL, -- time-based quality
                candlestick_quality DECIMAL(5,4) NOT NULL, -- candlestick formation quality
                market_context_quality DECIMAL(5,4) NOT NULL, -- market context quality
                noise_level DECIMAL(5,4) NOT NULL, -- estimated noise level 0-1
                filter_reasons JSONB, -- reasons for filtering decisions
                quality_factors JSONB, -- detailed quality factor breakdown
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, metric_id)
            )
        """)
        
        # Convert tables to TimescaleDB hypertables
        print("🔄 Converting tables to TimescaleDB hypertables...")
        
        tables_to_convert = [
            'pattern_performance_tracking',
            'market_regime_classification', 
            'adaptive_confidence_models',
            'noise_filter_settings',
            'pattern_quality_metrics'
        ]
        
        for table_name in tables_to_convert:
            try:
                cursor.execute(f"SELECT create_hypertable('{table_name}', 'timestamp', if_not_exists => TRUE)")
                print(f"✅ Converted {table_name} to hypertable")
            except Exception as e:
                print(f"⚠️ Warning: Could not convert {table_name} to hypertable: {e}")
        
        # Create indexes for performance
        print("📊 Creating performance indexes...")
        
        # Pattern performance tracking indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_performance_symbol_timeframe ON pattern_performance_tracking (symbol, timeframe, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_performance_pattern_name ON pattern_performance_tracking (pattern_name, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_performance_outcome ON pattern_performance_tracking (actual_outcome, timestamp DESC)")
        
        # Market regime indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_regime_symbol_timeframe ON market_regime_classification (symbol, timeframe, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_regime_type ON market_regime_classification (regime_type, timestamp DESC)")
        
        # Adaptive confidence models indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_models_pattern_regime ON adaptive_confidence_models (pattern_name, market_regime, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_models_active ON adaptive_confidence_models (is_active, timestamp DESC)")
        
        # Pattern quality metrics indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_quality_symbol_pattern ON pattern_quality_metrics (symbol, pattern_name, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_quality_score ON pattern_quality_metrics (quality_score DESC, timestamp DESC)")
        
        # Insert default noise filter settings
        print("⚙️ Inserting default noise filter settings...")
        
        default_filters = [
            {
                'filter_type': 'volume',
                'filter_name': 'low_volume_filter',
                'filter_parameters': {
                    'min_volume_ratio': 0.5,
                    'volume_period': 20,
                    'enabled': True
                },
                'priority': 1,
                'description': 'Filter patterns with volume below 50% of 20-period average'
            },
            {
                'filter_type': 'volatility',
                'filter_name': 'low_volatility_filter',
                'filter_parameters': {
                    'min_atr_ratio': 0.005,
                    'atr_period': 14,
                    'enabled': True
                },
                'priority': 2,
                'description': 'Filter patterns when ATR is below 0.5% of price'
            },
            {
                'filter_type': 'time',
                'filter_name': 'low_liquidity_hours_filter',
                'filter_parameters': {
                    'low_liquidity_start': '02:00',
                    'low_liquidity_end': '06:00',
                    'timezone': 'UTC',
                    'enabled': True
                },
                'priority': 3,
                'description': 'Reduce pattern sensitivity during low-liquidity hours'
            },
            {
                'filter_type': 'spread',
                'filter_name': 'high_spread_filter',
                'filter_parameters': {
                    'max_spread_ratio': 0.001,
                    'enabled': True
                },
                'priority': 4,
                'description': 'Filter patterns when bid/ask spread is too high'
            }
        ]
        
        for filter_config in default_filters:
            import json
            cursor.execute("""
                INSERT INTO noise_filter_settings (
                    timestamp, setting_id, filter_type, filter_name, filter_parameters, 
                    priority, description
                ) VALUES (
                    NOW(), %s, %s, %s, %s::jsonb, %s, %s
                ) ON CONFLICT DO NOTHING
            """, (
                f"{filter_config['filter_type']}_{filter_config['filter_name']}_{int(datetime.now().timestamp())}",
                filter_config['filter_type'],
                filter_config['filter_name'],
                json.dumps(filter_config['filter_parameters']),
                filter_config['priority'],
                filter_config['description']
            ))
        
        conn.commit()
        print("✅ All tables created and configured successfully")
        
        # Verify table creation
        print("\n📖 Verifying table creation...")
        for table_name in tables_to_convert:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   {table_name}: {count} records")
        
        print("\n🎯 Noise Filtering Database Setup Complete!")
        print("✅ All tables created and configured")
        print("✅ TimescaleDB hypertables configured")
        print("✅ Performance indexes created")
        print("✅ Default noise filter settings inserted")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("✅ Database connection closed")

if __name__ == "__main__":
    setup_noise_filtering_database()
