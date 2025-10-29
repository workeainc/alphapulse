import asyncio
import asyncpg
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_phase8_advanced_ml():
    """Create Phase 8 advanced ML features database structures"""
    
    # Database connection
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="alpha_emon",
        password="Emon_@17711",
        database="alphapulse"
    )
    
    try:
        logger.info("🚀 Creating Phase 8 Advanced ML Features...")
        
        # Anomaly Detection Tables
        anomaly_detection_table = """
        CREATE TABLE IF NOT EXISTS anomaly_detection (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            anomaly_type VARCHAR(30) NOT NULL, -- 'manipulation', 'news_event', 'technical_anomaly'
            anomaly_score DECIMAL(8,6) NOT NULL,
            confidence_score DECIMAL(3,2) NOT NULL,
            anomaly_features JSONB,
            detection_method VARCHAR(30), -- 'isolation_forest', 'autoencoder', 'statistical'
            severity_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Reinforcement Learning Tables
        rl_agent_states_table = """
        CREATE TABLE IF NOT EXISTS rl_agent_states (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            agent_id VARCHAR(50) NOT NULL,
            state_features JSONB NOT NULL,
            action_taken VARCHAR(30), -- 'buy', 'sell', 'hold', 'close'
            reward_received DECIMAL(8,6),
            next_state_features JSONB,
            episode_id VARCHAR(50),
            step_number INTEGER,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        rl_policy_performance_table = """
        CREATE TABLE IF NOT EXISTS rl_policy_performance (
            id SERIAL,
            agent_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            episode_id VARCHAR(50) NOT NULL,
            total_reward DECIMAL(8,6) NOT NULL,
            episode_length INTEGER NOT NULL,
            win_rate DECIMAL(5,4),
            sharpe_ratio DECIMAL(8,6),
            max_drawdown DECIMAL(8,6),
            profit_factor DECIMAL(8,6),
            policy_version VARCHAR(20),
            training_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Enhanced Multi-Timeframe Pattern Analysis Table
        enhanced_patterns_table = """
        CREATE TABLE IF NOT EXISTS enhanced_patterns (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            primary_timeframe VARCHAR(10) NOT NULL,
            pattern_name VARCHAR(100) NOT NULL,
            pattern_type VARCHAR(20) NOT NULL, -- 'bullish', 'bearish', 'neutral'
            confidence DECIMAL(3,2) NOT NULL,
            strength VARCHAR(20) NOT NULL, -- 'weak', 'medium', 'strong', 'extreme'
            price_level DECIMAL(20, 8) NOT NULL,
            volume_confirmation BOOLEAN NOT NULL,
            volume_confidence DECIMAL(3,2) NOT NULL,
            trend_alignment VARCHAR(20) NOT NULL,
            confirmation_timeframes JSONB, -- Array of timeframes
            timeframe_confidences JSONB, -- Object of timeframe -> confidence
            timeframe_alignments JSONB, -- Object of timeframe -> alignment
            failure_probability DECIMAL(3,2) NOT NULL,
            processing_latency_ms DECIMAL(10,2) NOT NULL,
            
            -- Volume Profile Analysis
            poc_level DECIMAL(20, 8),
            value_area_high DECIMAL(20, 8),
            value_area_low DECIMAL(20, 8),
            volume_profile_confidence DECIMAL(3,2),
            volume_nodes_count INTEGER,
            volume_gaps_count INTEGER,
            
            -- Elliott Wave Analysis
            current_wave VARCHAR(20),
            wave_count INTEGER,
            pattern_type_elliott VARCHAR(20),
            trend_direction_elliott VARCHAR(20),
            next_target_elliott DECIMAL(20, 8),
            elliott_confidence DECIMAL(3,2),
            fibonacci_levels JSONB,
            
            -- Wyckoff Analysis
            wyckoff_pattern VARCHAR(50),
            wyckoff_confidence DECIMAL(3,2),
            wyckoff_phase VARCHAR(30),
            
            -- SMC Analysis
            smc_patterns JSONB, -- Array of SMC pattern types
            smc_confidence DECIMAL(3,2),
            order_blocks_count INTEGER,
            fair_value_gaps_count INTEGER,
            
            -- Metadata
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Advanced Pattern Recognition Tables
        advanced_patterns_table = """
        CREATE TABLE IF NOT EXISTS advanced_patterns (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            pattern_type VARCHAR(50) NOT NULL, -- 'lstm_sequence', 'transformer_pattern', 'cnn_formation'
            pattern_confidence DECIMAL(3,2) NOT NULL,
            pattern_features JSONB NOT NULL,
            sequence_length INTEGER,
            prediction_horizon INTEGER,
            model_version VARCHAR(50),
            pattern_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Trading System Integration Tables
        trading_signals_table = """
        CREATE TABLE IF NOT EXISTS trading_signals (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            signal_type VARCHAR(30) NOT NULL, -- 'entry', 'exit', 'stop_loss', 'take_profit'
            signal_strength DECIMAL(3,2) NOT NULL,
            signal_source VARCHAR(50), -- 'volume_analysis', 'ml_prediction', 'rl_agent', 'anomaly_detection'
            entry_price DECIMAL(12,6),
            stop_loss_price DECIMAL(12,6),
            take_profit_price DECIMAL(12,6),
            position_size DECIMAL(8,4),
            risk_reward_ratio DECIMAL(5,2),
            confidence_score DECIMAL(3,2),
            signal_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        position_optimization_table = """
        CREATE TABLE IF NOT EXISTS position_optimization (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            optimization_type VARCHAR(30) NOT NULL, -- 'position_sizing', 'stop_loss', 'take_profit'
            current_position_size DECIMAL(8,4),
            recommended_position_size DECIMAL(8,4),
            current_stop_loss DECIMAL(12,6),
            recommended_stop_loss DECIMAL(12,6),
            current_take_profit DECIMAL(12,6),
            recommended_take_profit DECIMAL(12,6),
            optimization_factors JSONB,
            confidence_score DECIMAL(3,2),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Alert Priority System Table
        alert_priority_table = """
        CREATE TABLE IF NOT EXISTS alert_priority (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            alert_type VARCHAR(50) NOT NULL,
            priority_level VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
            alert_score DECIMAL(3,2) NOT NULL,
            contributing_factors JSONB,
            alert_message TEXT,
            is_acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_by VARCHAR(50),
            acknowledged_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """
        
        # Create all tables
        tables = [
            ("anomaly_detection", anomaly_detection_table),
            ("rl_agent_states", rl_agent_states_table),
            ("rl_policy_performance", rl_policy_performance_table),
            ("enhanced_patterns", enhanced_patterns_table),
            ("advanced_patterns", advanced_patterns_table),
            ("trading_signals", trading_signals_table),
            ("position_optimization", position_optimization_table),
            ("alert_priority", alert_priority_table)
        ]
        
        for table_name, table_sql in tables:
            await conn.execute(table_sql)
            logger.info(f"✅ Created table: {table_name}")
        
        # Create TimescaleDB hypertables
        hypertable_commands = [
            "SELECT create_hypertable('anomaly_detection', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
            "SELECT create_hypertable('rl_agent_states', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
            "SELECT create_hypertable('rl_policy_performance', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
            "SELECT create_hypertable('advanced_patterns', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
            "SELECT create_hypertable('position_optimization', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');",
            "SELECT create_hypertable('alert_priority', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');"
        ]
        
        for cmd in hypertable_commands:
            try:
                await conn.execute(cmd)
            except Exception as e:
                logger.warning(f"⚠️ Hypertable creation warning: {e}")
        
        # Handle trading_signals table separately (might already exist)
        try:
            await conn.execute("SELECT create_hypertable('trading_signals', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');")
        except Exception as e:
            logger.warning(f"⚠️ Trading signals hypertable warning: {e}")
        
        logger.info("✅ Created TimescaleDB hypertables")
        
        # Create indexes for performance
        index_commands = [
            "CREATE INDEX IF NOT EXISTS idx_anomaly_detection_symbol_time ON anomaly_detection (symbol, timeframe, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_detection_type ON anomaly_detection (anomaly_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_rl_agent_states_agent ON rl_agent_states (agent_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_rl_policy_performance_episode ON rl_policy_performance (episode_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_advanced_patterns_type ON advanced_patterns (pattern_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_position_optimization_type ON position_optimization (optimization_type, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_alert_priority_level ON alert_priority (priority_level, timestamp DESC);"
        ]
        
        for cmd in index_commands:
            try:
                await conn.execute(cmd)
            except Exception as e:
                logger.warning(f"⚠️ Index creation warning: {e}")
        
        # Handle trading_signals index separately
        try:
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_type ON trading_signals (signal_type, timestamp DESC);")
        except Exception as e:
            logger.warning(f"⚠️ Trading signals index warning: {e}")
        
        logger.info("✅ Created performance indexes")
        
        # Create materialized views for real-time monitoring
        materialized_views = [
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS real_time_anomalies AS
            SELECT 
                symbol,
                timeframe,
                timestamp,
                anomaly_type,
                anomaly_score,
                confidence_score,
                severity_level
            FROM anomaly_detection
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC;
            """,
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS high_priority_alerts AS
            SELECT 
                symbol,
                timeframe,
                timestamp,
                alert_type,
                priority_level,
                alert_score,
                alert_message
            FROM alert_priority
            WHERE priority_level IN ('high', 'critical') 
            AND timestamp >= NOW() - INTERVAL '1 hour'
            ORDER BY alert_score DESC, timestamp DESC;
            """
        ]
        
        for view_sql in materialized_views:
            try:
                await conn.execute(view_sql)
            except Exception as e:
                logger.warning(f"⚠️ Materialized view creation warning: {e}")
        
        # Handle trading signals view separately (might have different schema)
        try:
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS active_trading_signals AS
                SELECT 
                    symbol,
                    timeframe,
                    timestamp,
                    signal_type,
                    signal_strength,
                    signal_source,
                    entry_price,
                    stop_loss_price,
                    take_profit_price,
                    risk_reward_ratio,
                    confidence_score
                FROM trading_signals
                WHERE timestamp >= NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC;
            """)
        except Exception as e:
            logger.warning(f"⚠️ Trading signals view warning: {e}")
        
        logger.info("✅ Created materialized views")
        
        # Enable compression for older data
        compression_commands = [
            "SELECT add_compression_policy('anomaly_detection', INTERVAL '7 days');",
            "SELECT add_compression_policy('rl_agent_states', INTERVAL '7 days');",
            "SELECT add_compression_policy('rl_policy_performance', INTERVAL '7 days');",
            "SELECT add_compression_policy('advanced_patterns', INTERVAL '7 days');",
            "SELECT add_compression_policy('trading_signals', INTERVAL '7 days');",
            "SELECT add_compression_policy('position_optimization', INTERVAL '7 days');",
            "SELECT add_compression_policy('alert_priority', INTERVAL '7 days');"
        ]
        
        for cmd in compression_commands:
            try:
                await conn.execute(cmd)
            except Exception as e:
                logger.warning(f"⚠️ Compression policy warning: {e}")
        
        logger.info("✅ Enabled TimescaleDB compression")
        
        logger.info("🎉 Phase 8 Advanced ML Features database setup completed!")
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 8 setup: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_phase8_advanced_ml())
