#!/usr/bin/env python3
"""
Setup Advanced Pattern Recognition Database - Simple Version
Creates advanced pattern recognition tables in existing alphapulse database
"""

import psycopg2
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_advanced_pattern_database():
    """Setup advanced pattern recognition database tables"""
    try:
        print("🚀 Setting up Advanced Pattern Recognition Database...")
        
        # Database configuration - using existing alphapulse database
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',  # Use existing database
            'user': 'postgres',
            'password': 'Emon_@17711'
        }
        
        # Connect to database
        conn = psycopg2.connect(**db_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        print(f"✅ Connected to database: {db_config['database']}")
        
        # Check if TimescaleDB is available
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'timescaledb'")
        if not cursor.fetchone():
            print("❌ TimescaleDB extension not found. Please install TimescaleDB first.")
            return False
        
        print("✅ TimescaleDB extension found")
        
        # Create advanced pattern tables
        tables_created = []
        
        # 1. Multi-Timeframe Patterns Table
        print("\n📊 Creating multi_timeframe_patterns table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_timeframe_patterns (
                timestamp TIMESTAMPTZ NOT NULL,
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                primary_timeframe VARCHAR(10) NOT NULL,
                pattern_name VARCHAR(50) NOT NULL,
                pattern_type VARCHAR(20) NOT NULL,
                primary_confidence DECIMAL(5,4) NOT NULL,
                primary_strength VARCHAR(20) NOT NULL,
                price_level DECIMAL(20,8) NOT NULL,
                confirmation_timeframes TEXT[] NOT NULL,
                timeframe_confidences JSONB NOT NULL,
                timeframe_alignments JSONB NOT NULL,
                overall_confidence DECIMAL(5,4) NOT NULL,
                confirmation_score DECIMAL(5,2) NOT NULL,
                trend_alignment VARCHAR(20) NOT NULL,
                failure_probability DECIMAL(5,4) NOT NULL,
                detection_method VARCHAR(50) DEFAULT 'multi_timeframe',
                processing_latency_ms INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, pattern_id)
            )
        """)
        tables_created.append('multi_timeframe_patterns')
        print("✅ multi_timeframe_patterns table created")
        
        # 2. Pattern Failure Predictions Table
        print("\n📊 Creating pattern_failure_predictions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_failure_predictions (
                timestamp TIMESTAMPTZ NOT NULL,
                prediction_id VARCHAR(100) NOT NULL,
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_name VARCHAR(50) NOT NULL,
                failure_probability DECIMAL(5,4) NOT NULL,
                failure_confidence DECIMAL(5,4) NOT NULL,
                failure_reasons TEXT[] NOT NULL,
                risk_factors JSONB NOT NULL,
                market_volatility DECIMAL(10,8) NOT NULL,
                volume_profile VARCHAR(20) NOT NULL,
                liquidity_score DECIMAL(5,4) NOT NULL,
                support_resistance_proximity DECIMAL(5,4) NOT NULL,
                rsi_value DECIMAL(5,2) NOT NULL,
                macd_signal VARCHAR(20) NOT NULL,
                bollinger_position VARCHAR(20) NOT NULL,
                atr_value DECIMAL(20,8) NOT NULL,
                prediction_model VARCHAR(50) DEFAULT 'ensemble_ml',
                feature_importance JSONB NOT NULL,
                processing_latency_ms INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, prediction_id)
            )
        """)
        tables_created.append('pattern_failure_predictions')
        print("✅ pattern_failure_predictions table created")
        
        # 3. Pattern Strength Scores Table
        print("\n📊 Creating pattern_strength_scores table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_strength_scores (
                timestamp TIMESTAMPTZ NOT NULL,
                score_id VARCHAR(100) NOT NULL,
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_name VARCHAR(50) NOT NULL,
                pattern_type VARCHAR(20) NOT NULL,
                strength_score DECIMAL(5,4) NOT NULL,
                volume_score DECIMAL(5,4) NOT NULL,
                trend_alignment_score DECIMAL(5,4) NOT NULL,
                support_resistance_score DECIMAL(5,4) NOT NULL,
                market_regime_score DECIMAL(5,4) NOT NULL,
                historical_success_rate DECIMAL(5,4) NOT NULL,
                weighted_score DECIMAL(5,4) NOT NULL,
                strength_category VARCHAR(20) NOT NULL,
                confidence_level DECIMAL(5,4) NOT NULL,
                feature_weights JSONB NOT NULL,
                processing_latency_ms INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, score_id)
            )
        """)
        tables_created.append('pattern_strength_scores')
        print("✅ pattern_strength_scores table created")
        
        # 4. Advanced Pattern Signals Table
        print("\n📊 Creating advanced_pattern_signals table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advanced_pattern_signals (
                timestamp TIMESTAMPTZ NOT NULL,
                signal_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_id VARCHAR(100) NOT NULL,
                signal_type VARCHAR(20) NOT NULL,
                signal_strength DECIMAL(5,4) NOT NULL,
                entry_price DECIMAL(20,8) NOT NULL,
                stop_loss DECIMAL(20,8),
                take_profit DECIMAL(20,8),
                risk_reward_ratio DECIMAL(5,2),
                confidence_score DECIMAL(5,4) NOT NULL,
                failure_probability DECIMAL(5,4) NOT NULL,
                market_conditions JSONB NOT NULL,
                technical_indicators JSONB NOT NULL,
                signal_metadata JSONB,
                processing_latency_ms INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (timestamp, signal_id)
            )
        """)
        tables_created.append('advanced_pattern_signals')
        print("✅ advanced_pattern_signals table created")
        
        # Convert tables to TimescaleDB hypertables
        print("\n🏗️ Converting tables to TimescaleDB hypertables...")
        for table in tables_created:
            try:
                cursor.execute(f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE)")
                print(f"✅ {table} converted to hypertable")
            except Exception as e:
                print(f"⚠️ {table} hypertable creation: {e}")
        
        # Create basic indexes
        print("\n📈 Creating basic indexes...")
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_multi_timeframe_symbol ON multi_timeframe_patterns (symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_multi_timeframe_pattern ON multi_timeframe_patterns (pattern_name, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_failure_predictions_symbol ON pattern_failure_predictions (symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_failure_predictions_pattern ON pattern_failure_predictions (pattern_name, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strength_scores_symbol ON pattern_strength_scores (symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_strength_scores_pattern ON pattern_strength_scores (pattern_name, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_advanced_signals_symbol ON advanced_pattern_signals (symbol, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_advanced_signals_type ON advanced_pattern_signals (signal_type, timestamp DESC)"
        ]
        
        for query in index_queries:
            try:
                cursor.execute(query)
                print(f"✅ Index created: {query.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                print(f"⚠️ Index creation: {e}")
        
        # Verify tables were created
        print("\n🔍 Verifying table creation...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('multi_timeframe_patterns', 'pattern_failure_predictions', 'pattern_strength_scores', 'advanced_pattern_signals')
            ORDER BY table_name
        """)
        
        created_tables = [row[0] for row in cursor.fetchall()]
        print(f"✅ Tables created: {created_tables}")
        
        # Check hypertables
        cursor.execute("""
            SELECT hypertable_name 
            FROM timescaledb_information.hypertables 
            WHERE hypertable_schema = 'public'
            AND hypertable_name IN ('multi_timeframe_patterns', 'pattern_failure_predictions', 'pattern_strength_scores', 'advanced_pattern_signals')
        """)
        
        hypertables = [row[0] for row in cursor.fetchall()]
        print(f"✅ Hypertables created: {hypertables}")
        
        cursor.close()
        conn.close()
        
        print(f"\n🎉 Advanced Pattern Recognition Database setup completed!")
        print(f"📊 Created {len(created_tables)} tables")
        print(f"🏗️ Created {len(hypertables)} hypertables")
        print(f"📈 Database: {db_config['database']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_advanced_pattern_database()
    if success:
        print("\n✅ Advanced Pattern Recognition Database is ready!")
    else:
        print("\n❌ Database setup failed. Please check the logs.")
