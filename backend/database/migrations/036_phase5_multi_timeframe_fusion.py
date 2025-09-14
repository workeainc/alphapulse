"""
Migration: Phase 5 Multi-Timeframe Fusion Integration
Phase 5: Database schema updates for multi-timeframe signal fusion and analysis
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def upgrade(connection: asyncpg.Connection):
    """Upgrade database schema"""
    try:
        logger.info("🔄 Adding Phase 5 Multi-Timeframe Fusion tables...")
        
        # Phase 5: Multi-Timeframe Fusion Tables
        
        # Create multi-timeframe signals table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS multi_timeframe_signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                base_timeframe VARCHAR(10) NOT NULL,
                signal_id VARCHAR(50) NOT NULL,
                timeframe_signals JSONB NOT NULL,
                fusion_confidence FLOAT NOT NULL,
                timeframe_agreement FLOAT NOT NULL,
                signal_consistency FLOAT NOT NULL,
                market_condition VARCHAR(20),
                fusion_metadata JSONB,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create timeframe analysis table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS timeframe_analysis (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                analysis_type VARCHAR(30) NOT NULL,
                confidence_score FLOAT NOT NULL,
                volume_quality FLOAT,
                pattern_clarity FLOAT,
                risk_reward_ratio FLOAT,
                analysis_data JSONB,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create MTF fusion results table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS mtf_fusion_results (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                fusion_id VARCHAR(50) NOT NULL,
                primary_direction VARCHAR(10) NOT NULL,
                overall_strength VARCHAR(20) NOT NULL,
                weighted_confidence FLOAT NOT NULL,
                timeframe_weights JSONB,
                signal_breakdown JSONB,
                fusion_algorithm VARCHAR(30),
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create timeframe weights table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS timeframe_weights (
                id SERIAL PRIMARY KEY,
                market_condition VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                base_weight FLOAT NOT NULL,
                adjusted_weight FLOAT NOT NULL,
                weight_factor FLOAT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create signal consistency table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_consistency (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                signal_id VARCHAR(50) NOT NULL,
                consistency_score FLOAT NOT NULL,
                coefficient_variation FLOAT,
                signal_strengths JSONB,
                consistency_analysis TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create timeframe agreement table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS timeframe_agreement (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                signal_id VARCHAR(50) NOT NULL,
                agreement_score FLOAT NOT NULL,
                agreeing_timeframes JSONB,
                disagreeing_timeframes JSONB,
                agreement_analysis TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for efficient querying
        try:
            # Multi-timeframe signals indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_multi_timeframe_signals_symbol 
                ON multi_timeframe_signals(symbol)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_multi_timeframe_signals_timestamp 
                ON multi_timeframe_signals(timestamp DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_multi_timeframe_signals_fusion_confidence 
                ON multi_timeframe_signals(fusion_confidence DESC)
            """)
            
            # Timeframe analysis indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeframe_analysis_symbol_timeframe 
                ON timeframe_analysis(symbol, timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeframe_analysis_timestamp 
                ON timeframe_analysis(timestamp DESC)
            """)
            
            # MTF fusion results indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_fusion_results_symbol 
                ON mtf_fusion_results(symbol)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_mtf_fusion_results_timestamp 
                ON mtf_fusion_results(timestamp DESC)
            """)
            
            # Timeframe weights indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeframe_weights_market_condition 
                ON timeframe_weights(market_condition)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeframe_weights_timeframe 
                ON timeframe_weights(timeframe)
            """)
            
            # Signal consistency indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_consistency_symbol 
                ON signal_consistency(symbol)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_consistency_timestamp 
                ON signal_consistency(timestamp DESC)
            """)
            
            # Timeframe agreement indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeframe_agreement_symbol 
                ON timeframe_agreement(symbol)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_timeframe_agreement_timestamp 
                ON timeframe_agreement(timestamp DESC)
            """)
            
            logger.info("✅ Indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not create some indexes: {e}")
        
        # Add columns to existing signals table for Phase 5 tracking
        try:
            await connection.execute("""
                ALTER TABLE signals 
                ADD COLUMN IF NOT EXISTS timeframe_agreement_score FLOAT,
                ADD COLUMN IF NOT EXISTS signal_consistency_score FLOAT,
                ADD COLUMN IF NOT EXISTS mtf_fusion_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS timeframe_weights_used JSONB,
                ADD COLUMN IF NOT EXISTS higher_timeframe_context JSONB
            """)
            logger.info("✅ Added Phase 5 tracking columns to signals table")
        except Exception as e:
            logger.warning(f"⚠️ Could not add Phase 5 tracking columns to signals table: {e}")
        
        # Create TimescaleDB hypertables for time-series data
        try:
            await connection.execute("""
                SELECT create_hypertable('multi_timeframe_signals', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ multi_timeframe_signals converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for multi_timeframe_signals: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('timeframe_analysis', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ timeframe_analysis converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for timeframe_analysis: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('mtf_fusion_results', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ mtf_fusion_results converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for mtf_fusion_results: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('signal_consistency', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ signal_consistency converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for signal_consistency: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('timeframe_agreement', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ timeframe_agreement converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for timeframe_agreement: {e}")
        
        # Insert default timeframe weights
        try:
            await connection.execute("""
                INSERT INTO timeframe_weights (market_condition, timeframe, base_weight, adjusted_weight, weight_factor)
                VALUES 
                ('trending', '1m', 0.05, 0.03, 0.6),
                ('trending', '5m', 0.10, 0.08, 0.8),
                ('trending', '15m', 0.15, 0.18, 1.2),
                ('trending', '1h', 0.25, 0.30, 1.2),
                ('trending', '4h', 0.25, 0.30, 1.2),
                ('trending', '1d', 0.20, 0.11, 0.55),
                ('ranging', '1m', 0.05, 0.03, 0.6),
                ('ranging', '5m', 0.10, 0.08, 0.8),
                ('ranging', '15m', 0.15, 0.20, 1.33),
                ('ranging', '1h', 0.25, 0.30, 1.2),
                ('ranging', '4h', 0.25, 0.25, 1.0),
                ('ranging', '1d', 0.20, 0.14, 0.7),
                ('volatile', '1m', 0.05, 0.025, 0.5),
                ('volatile', '5m', 0.10, 0.05, 0.5),
                ('volatile', '15m', 0.15, 0.12, 0.8),
                ('volatile', '1h', 0.25, 0.25, 1.0),
                ('volatile', '4h', 0.25, 0.30, 1.2),
                ('volatile', '1d', 0.20, 0.255, 1.275)
                ON CONFLICT DO NOTHING
            """)
            logger.info("✅ Default timeframe weights inserted")
        except Exception as e:
            logger.warning(f"⚠️ Could not insert default timeframe weights: {e}")
        
        logger.info("✅ Phase 5 Multi-Timeframe Fusion tables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating Phase 5 tables: {e}")
        raise

async def downgrade(connection: asyncpg.Connection):
    """Downgrade database schema"""
    try:
        logger.info("🔄 Removing Phase 5 Multi-Timeframe Fusion tables...")
        
        # Drop tables
        await connection.execute("DROP TABLE IF EXISTS multi_timeframe_signals CASCADE")
        await connection.execute("DROP TABLE IF EXISTS timeframe_analysis CASCADE")
        await connection.execute("DROP TABLE IF EXISTS mtf_fusion_results CASCADE")
        await connection.execute("DROP TABLE IF EXISTS timeframe_weights CASCADE")
        await connection.execute("DROP TABLE IF EXISTS signal_consistency CASCADE")
        await connection.execute("DROP TABLE IF EXISTS timeframe_agreement CASCADE")
        
        # Remove columns from signals table
        await connection.execute("""
            ALTER TABLE signals 
            DROP COLUMN IF EXISTS timeframe_agreement_score,
            DROP COLUMN IF EXISTS signal_consistency_score,
            DROP COLUMN IF EXISTS mtf_fusion_confidence,
            DROP COLUMN IF EXISTS timeframe_weights_used,
            DROP COLUMN IF EXISTS higher_timeframe_context
        """)
        
        logger.info("✅ Phase 5 Multi-Timeframe Fusion tables removed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error removing Phase 5 tables: {e}")
        raise

async def main():
    """Run migration"""
    try:
        # Connect to database
        connection = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='alpha_emon',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        # Run upgrade
        await upgrade(connection)
        
        # Close connection
        await connection.close()
        
        logger.info("🎉 Phase 5 migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Phase 5 migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
