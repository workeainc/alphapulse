#!/usr/bin/env python3
"""
Advanced Pattern Recognition Database Setup for AlphaPlus
Creates all tables for multi-timeframe patterns, failure predictions, strength scores, correlations, and adaptive settings
"""

import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_advanced_pattern_database():
    """Setup advanced pattern recognition database schema"""
    try:
        logger.info("🚀 Starting Advanced Pattern Recognition Database Setup...")
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="alphapulse",
            user="alpha_emon",
            password="Emon_@17711"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Step 1: Create multi-timeframe patterns table
        logger.info("📊 Creating multi-timeframe patterns table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_timeframe_patterns (
                pattern_id VARCHAR(100),
                symbol VARCHAR(20) NOT NULL,
                primary_timeframe VARCHAR(10) NOT NULL,
                pattern_name VARCHAR(100) NOT NULL,
                pattern_type VARCHAR(20) NOT NULL,
                primary_confidence DECIMAL(4,3) NOT NULL,
                primary_strength VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                price_level DECIMAL(20,8) NOT NULL,
                confirmation_timeframes JSONB NOT NULL,
                timeframe_confidences JSONB NOT NULL,
                timeframe_alignments JSONB NOT NULL,
                overall_confidence DECIMAL(4,3) NOT NULL,
                confirmation_score DECIMAL(5,2) NOT NULL,
                trend_alignment VARCHAR(20) NOT NULL,
                failure_probability DECIMAL(4,3) NOT NULL,
                detection_method VARCHAR(50) NOT NULL DEFAULT 'multi_timeframe',
                processing_latency_ms INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (timestamp, pattern_id)
            );
        """)
        
        # Convert to TimescaleDB hypertable
        cursor.execute("""
            SELECT create_hypertable('multi_timeframe_patterns', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE
            );
        """)
        
        # Step 2: Create pattern failure predictions table
        logger.info("⚠️ Creating pattern failure predictions table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_failure_predictions (
                prediction_id VARCHAR(100),
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                failure_probability DECIMAL(4,3) NOT NULL,
                failure_confidence DECIMAL(4,3) NOT NULL,
                failure_reasons JSONB NOT NULL,
                risk_factors JSONB NOT NULL,
                market_volatility DECIMAL(6,4) NOT NULL,
                volume_profile VARCHAR(20) NOT NULL,
                liquidity_score DECIMAL(4,3) NOT NULL,
                support_resistance_proximity DECIMAL(4,3) NOT NULL,
                rsi_value DECIMAL(5,2) NOT NULL,
                macd_signal VARCHAR(20) NOT NULL,
                bollinger_position VARCHAR(20) NOT NULL,
                atr_value DECIMAL(10,4) NOT NULL,
                prediction_model VARCHAR(50) NOT NULL,
                feature_importance JSONB NOT NULL,
                processing_latency_ms INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (timestamp, prediction_id)
            );
        """)
        
        # Convert to TimescaleDB hypertable
        cursor.execute("""
            SELECT create_hypertable('pattern_failure_predictions', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE
            );
        """)
        
        # Step 3: Create pattern strength scores table
        logger.info("💪 Creating pattern strength scores table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_strength_scores (
                score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                strength_score DECIMAL(5,2) NOT NULL,  -- 0-100
                volume_score DECIMAL(4,3) NOT NULL,
                trend_alignment_score DECIMAL(4,3) NOT NULL,
                support_resistance_score DECIMAL(4,3) NOT NULL,
                market_regime_score DECIMAL(4,3) NOT NULL,
                historical_success_score DECIMAL(4,3) NOT NULL,
                factor_breakdown JSONB NOT NULL,
                strength_category VARCHAR(20) NOT NULL,  -- weak, moderate, strong
                confidence_level DECIMAL(4,3) NOT NULL,
                processing_latency_ms INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        
        # Convert to TimescaleDB hypertable
        cursor.execute("""
            SELECT create_hypertable('pattern_strength_scores', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE
            );
        """)
        
        # Step 4: Create pattern correlations table
        logger.info("🔗 Creating pattern correlations table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_correlations (
                correlation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pattern_a VARCHAR(100) NOT NULL,
                pattern_b VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                correlation_coefficient DECIMAL(4,3) NOT NULL,  -- -1 to 1
                correlation_significance DECIMAL(4,3) NOT NULL,
                correlation_type VARCHAR(50) NOT NULL,  -- co_occurrence, sequential, conflict
                time_lag_minutes INTEGER,
                co_occurrence_count INTEGER NOT NULL,
                total_occurrences INTEGER NOT NULL,
                confidence_interval_lower DECIMAL(4,3),
                confidence_interval_upper DECIMAL(4,3),
                market_conditions JSONB,
                timestamp TIMESTAMPTZ NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(pattern_a, pattern_b, symbol, correlation_type)
            );
        """)
        
        # Convert to TimescaleDB hypertable
        cursor.execute("""
            SELECT create_hypertable('pattern_correlations', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE
            );
        """)
        
        # Step 5: Create adaptive pattern settings table
        logger.info("🔄 Creating adaptive pattern settings table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_pattern_settings (
                setting_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                market_regime VARCHAR(50) NOT NULL,
                pattern_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                confidence_threshold DECIMAL(4,3) NOT NULL,
                volume_threshold DECIMAL(4,3) NOT NULL,
                volatility_adjustment DECIMAL(4,3) NOT NULL,
                trend_strength_adjustment DECIMAL(4,3) NOT NULL,
                adaptive_parameters JSONB NOT NULL,
                performance_metrics JSONB NOT NULL,
                success_rate DECIMAL(4,3) NOT NULL,
                avg_profit_loss DECIMAL(8,4) NOT NULL,
                total_signals INTEGER NOT NULL,
                last_updated TIMESTAMPTZ NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(market_regime, pattern_type, symbol, timeframe)
            );
        """)
        
        # Step 6: Create advanced pattern signals table
        logger.info("📈 Creating advanced pattern signals table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS advanced_pattern_signals (
                signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pattern_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                pattern_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                signal_type VARCHAR(20) NOT NULL,  -- buy, sell, hold
                confidence_score DECIMAL(4,3) NOT NULL,
                strength_score DECIMAL(5,2) NOT NULL,
                failure_probability DECIMAL(4,3) NOT NULL,
                entry_price DECIMAL(20,8) NOT NULL,
                stop_loss DECIMAL(20,8),
                take_profit DECIMAL(20,8),
                risk_reward_ratio DECIMAL(6,2),
                position_size DECIMAL(8,4),
                multi_timeframe_confirmation BOOLEAN NOT NULL DEFAULT FALSE,
                correlation_conflicts JSONB,
                adaptive_settings_applied JSONB,
                ensemble_score DECIMAL(4,3) NOT NULL,
                market_regime VARCHAR(50),
                processing_latency_ms INTEGER,
                metadata JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)
        
        # Convert to TimescaleDB hypertable
        cursor.execute("""
            SELECT create_hypertable('advanced_pattern_signals', 'timestamp', 
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE
            );
        """)
        
        # Step 7: Create basic indexes
        logger.info("🔍 Creating basic indexes...")
        
        # Multi-timeframe patterns indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mtf_patterns_symbol_time 
            ON multi_timeframe_patterns (symbol, timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mtf_patterns_confirmation_score 
            ON multi_timeframe_patterns (confirmation_score DESC);
        """)
        
        # Failure predictions indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_failure_pred_symbol_time 
            ON pattern_failure_predictions (symbol, timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_failure_pred_probability 
            ON pattern_failure_predictions (failure_probability DESC);
        """)
        
        # Strength scores indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strength_scores_symbol_time 
            ON pattern_strength_scores (symbol, timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strength_scores_score 
            ON pattern_strength_scores (strength_score DESC);
        """)
        
        # Pattern correlations indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_correlations_symbol 
            ON pattern_correlations (symbol, correlation_coefficient DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_correlations_patterns 
            ON pattern_correlations (pattern_a, pattern_b);
        """)
        
        # Advanced signals indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_advanced_signals_symbol_time 
            ON advanced_pattern_signals (symbol, timestamp DESC);
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_advanced_signals_ensemble_score 
            ON advanced_pattern_signals (ensemble_score DESC);
        """)
        
        # Step 8: Create continuous aggregates for performance monitoring
        logger.info("📊 Creating continuous aggregates...")
        
        # Multi-timeframe patterns hourly stats
        cursor.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS mtf_patterns_hourly_stats
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 hour', timestamp) AS bucket,
                symbol,
                pattern_name,
                COUNT(*) as pattern_count,
                AVG(confirmation_score) as avg_confirmation_score,
                AVG(overall_confidence) as avg_confidence,
                AVG(failure_probability) as avg_failure_probability,
                AVG(processing_latency_ms) as avg_latency_ms
            FROM multi_timeframe_patterns
            GROUP BY bucket, symbol, pattern_name;
        """)
        
        # Failure predictions hourly stats
        cursor.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS failure_pred_hourly_stats
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 hour', timestamp) AS bucket,
                symbol,
                pattern_name,
                COUNT(*) as prediction_count,
                AVG(failure_probability) as avg_failure_probability,
                AVG(failure_confidence) as avg_confidence,
                AVG(processing_latency_ms) as avg_latency_ms
            FROM pattern_failure_predictions
            GROUP BY bucket, symbol, pattern_name;
        """)
        
        # Strength scores hourly stats
        cursor.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS strength_scores_hourly_stats
            WITH (timescaledb.continuous) AS
            SELECT 
                time_bucket('1 hour', timestamp) AS bucket,
                symbol,
                pattern_name,
                COUNT(*) as score_count,
                AVG(strength_score) as avg_strength_score,
                AVG(confidence_level) as avg_confidence,
                AVG(processing_latency_ms) as avg_latency_ms
            FROM pattern_strength_scores
            GROUP BY bucket, symbol, pattern_name;
        """)
        
        # Step 9: Insert initial adaptive settings
        logger.info("⚙️ Initializing adaptive pattern settings...")
        
        # Market regimes
        market_regimes = ["trending", "ranging", "volatile", "low_volatility"]
        pattern_types = ["doji", "hammer", "shooting_star", "engulfing", "morning_star", "evening_star"]
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        for regime in market_regimes:
            for pattern_type in pattern_types:
                for symbol in symbols:
                    for timeframe in timeframes:
                        # Adjust thresholds based on market regime
                        if regime == "trending":
                            confidence_threshold = 0.6
                            volume_threshold = 0.5
                            volatility_adjustment = 0.1
                        elif regime == "ranging":
                            confidence_threshold = 0.7
                            volume_threshold = 0.6
                            volatility_adjustment = 0.2
                        elif regime == "volatile":
                            confidence_threshold = 0.8
                            volume_threshold = 0.7
                            volatility_adjustment = 0.3
                        else:  # low_volatility
                            confidence_threshold = 0.6
                            volume_threshold = 0.5
                            volatility_adjustment = 0.0
                        
                        cursor.execute("""
                            INSERT INTO adaptive_pattern_settings (
                                market_regime, pattern_type, symbol, timeframe,
                                confidence_threshold, volume_threshold, volatility_adjustment,
                                trend_strength_adjustment, adaptive_parameters, performance_metrics,
                                success_rate, avg_profit_loss, total_signals, last_updated
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                            ) ON CONFLICT (market_regime, pattern_type, symbol, timeframe) DO NOTHING;
                        """, (
                            regime, pattern_type, symbol, timeframe,
                            confidence_threshold, volume_threshold, volatility_adjustment,
                            0.5,  # trend_strength_adjustment
                            json.dumps({"base_threshold": 0.5, "regime_multiplier": 1.0}),
                            json.dumps({"total_trades": 0, "winning_trades": 0, "losing_trades": 0}),
                            0.5,  # success_rate
                            0.0,  # avg_profit_loss
                            0     # total_signals
                        ))
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Advanced Pattern Recognition Database Setup completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced Pattern Recognition Database Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_advanced_tables():
    """Verify that all advanced pattern recognition tables exist"""
    try:
        logger.info("🔍 Verifying Advanced Pattern Recognition Database Tables...")
        
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="alphapulse",
            user="alpha_emon",
            password="Emon_@17711"
        )
        cursor = conn.cursor()
        
        # Check for required tables
        required_tables = [
            "multi_timeframe_patterns",
            "pattern_failure_predictions",
            "pattern_strength_scores",
            "pattern_correlations",
            "adaptive_pattern_settings",
            "advanced_pattern_signals"
        ]
        
        for table in required_tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table,))
            
            exists = cursor.fetchone()[0]
            if not exists:
                logger.error(f"❌ Required table {table} not found")
                return False
            else:
                logger.info(f"✅ Table {table} exists")
        
        # Check for hypertables
        cursor.execute("""
            SELECT hypertable_name 
            FROM timescaledb_information.hypertables 
            WHERE hypertable_name IN ('multi_timeframe_patterns', 'pattern_failure_predictions', 
                                    'pattern_strength_scores', 'pattern_correlations', 'advanced_pattern_signals');
        """)
        
        hypertables = [row[0] for row in cursor.fetchall()]
        logger.info(f"✅ Found hypertables: {hypertables}")
        
        # Check for continuous aggregates
        cursor.execute("""
            SELECT view_name 
            FROM timescaledb_information.continuous_aggregates 
            WHERE view_name LIKE '%_hourly_stats';
        """)
        
        continuous_aggregates = [row[0] for row in cursor.fetchall()]
        logger.info(f"✅ Found continuous aggregates: {continuous_aggregates}")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Advanced Pattern Recognition Database verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced Pattern Recognition Database verification failed: {e}")
        return False

def main():
    """Main function to run advanced pattern recognition database setup"""
    try:
        # Run setup
        if setup_advanced_pattern_database():
            # Verify setup
            if verify_advanced_tables():
                logger.info("🎉 Advanced Pattern Recognition Database setup and verification completed successfully!")
                return True
            else:
                logger.error("❌ Advanced Pattern Recognition Database verification failed")
                return False
        else:
            logger.error("❌ Advanced Pattern Recognition Database setup failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Advanced Pattern Recognition Database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
