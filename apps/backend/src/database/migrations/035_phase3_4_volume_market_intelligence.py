"""
Migration: Phase 3-4 Volume Analysis and Market Intelligence Tables
Phase 3.4: Database schema updates for volume analysis and market intelligence
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def upgrade(connection: asyncpg.Connection):
    """Upgrade database schema"""
    try:
        logger.info("🔄 Adding Phase 3-4 Volume Analysis and Market Intelligence tables...")
        
        # Phase 3: Volume Analysis Tables
        
        # Create volume profile data table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS volume_profile_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                price_level FLOAT NOT NULL,
                volume_at_level FLOAT NOT NULL,
                volume_type VARCHAR(20), -- 'bid', 'ask', 'total'
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create order book analysis table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS order_book_analysis (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                bid_ask_imbalance FLOAT NOT NULL,
                order_flow_toxicity FLOAT,
                depth_pressure FLOAT,
                liquidity_walls JSONB,
                order_clusters JSONB,
                spread_analysis JSONB,
                order_book_analysis TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create liquidity analysis table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS liquidity_analysis (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                liquidity_score FLOAT NOT NULL,
                bid_liquidity FLOAT,
                ask_liquidity FLOAT,
                liquidity_walls JSONB,
                order_clusters JSONB,
                depth_pressure FLOAT,
                spread_analysis JSONB,
                liquidity_analysis TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Phase 4: Market Intelligence Tables
        
        # Create BTC dominance table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS btc_dominance (
                id SERIAL PRIMARY KEY,
                dominance_value FLOAT NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                ts TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create market cap data table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS market_cap_data (
                id SERIAL PRIMARY KEY,
                total2_value FLOAT,
                total3_value FLOAT,
                timeframe VARCHAR(10) NOT NULL,
                ts TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create market sentiment table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS market_sentiment (
                id SERIAL PRIMARY KEY,
                fear_greed_index FLOAT,
                social_sentiment FLOAT,
                timeframe VARCHAR(10) NOT NULL,
                ts TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create asset correlations table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS asset_correlations (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                btc_correlation FLOAT,
                eth_correlation FLOAT,
                market_correlation FLOAT,
                timeframe VARCHAR(10) NOT NULL,
                ts TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for efficient querying
        try:
            # Volume analysis indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_volume_profile_data_symbol_timeframe 
                ON volume_profile_data(symbol, timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_volume_profile_data_timestamp 
                ON volume_profile_data(timestamp DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_order_book_analysis_symbol 
                ON order_book_analysis(symbol)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_order_book_analysis_timestamp 
                ON order_book_analysis(timestamp DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_liquidity_analysis_symbol 
                ON liquidity_analysis(symbol)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_liquidity_analysis_timestamp 
                ON liquidity_analysis(timestamp DESC)
            """)
            
            # Market intelligence indexes
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_btc_dominance_timeframe 
                ON btc_dominance(timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_btc_dominance_ts 
                ON btc_dominance(ts DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_cap_data_timeframe 
                ON market_cap_data(timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_cap_data_ts 
                ON market_cap_data(ts DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_sentiment_timeframe 
                ON market_sentiment(timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_sentiment_ts 
                ON market_sentiment(ts DESC)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_asset_correlations_symbol_timeframe 
                ON asset_correlations(symbol, timeframe)
            """)
            
            await connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_asset_correlations_ts 
                ON asset_correlations(ts DESC)
            """)
            
            logger.info("✅ Indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not create some indexes: {e}")
        
        # Add columns to existing signals table for Phase 3-4 tracking
        try:
            await connection.execute("""
                ALTER TABLE signals 
                ADD COLUMN IF NOT EXISTS volume_analysis_score FLOAT,
                ADD COLUMN IF NOT EXISTS volume_health_score FLOAT,
                ADD COLUMN IF NOT EXISTS btc_dominance_score FLOAT,
                ADD COLUMN IF NOT EXISTS market_cap_correlation FLOAT,
                ADD COLUMN IF NOT EXISTS market_sentiment_score FLOAT
            """)
            logger.info("✅ Added Phase 3-4 tracking columns to signals table")
        except Exception as e:
            logger.warning(f"⚠️ Could not add Phase 3-4 tracking columns to signals table: {e}")
        
        # Create TimescaleDB hypertables for time-series data
        try:
            await connection.execute("""
                SELECT create_hypertable('volume_profile_data', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ volume_profile_data converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for volume_profile_data: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('order_book_analysis', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ order_book_analysis converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for order_book_analysis: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('liquidity_analysis', 'timestamp', if_not_exists => TRUE)
            """)
            logger.info("✅ liquidity_analysis converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for liquidity_analysis: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('btc_dominance', 'ts', if_not_exists => TRUE)
            """)
            logger.info("✅ btc_dominance converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for btc_dominance: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('market_cap_data', 'ts', if_not_exists => TRUE)
            """)
            logger.info("✅ market_cap_data converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for market_cap_data: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('market_sentiment', 'ts', if_not_exists => TRUE)
            """)
            logger.info("✅ market_sentiment converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for market_sentiment: {e}")
        
        try:
            await connection.execute("""
                SELECT create_hypertable('asset_correlations', 'ts', if_not_exists => TRUE)
            """)
            logger.info("✅ asset_correlations converted to hypertable")
        except Exception as e:
            logger.warning(f"⚠️ Could not create hypertable for asset_correlations: {e}")
        
        logger.info("✅ Phase 3-4 Volume Analysis and Market Intelligence tables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating Phase 3-4 tables: {e}")
        raise

async def downgrade(connection: asyncpg.Connection):
    """Downgrade database schema"""
    try:
        logger.info("🔄 Removing Phase 3-4 Volume Analysis and Market Intelligence tables...")
        
        # Drop tables
        await connection.execute("DROP TABLE IF EXISTS volume_profile_data CASCADE")
        await connection.execute("DROP TABLE IF EXISTS order_book_analysis CASCADE")
        await connection.execute("DROP TABLE IF EXISTS liquidity_analysis CASCADE")
        await connection.execute("DROP TABLE IF EXISTS btc_dominance CASCADE")
        await connection.execute("DROP TABLE IF EXISTS market_cap_data CASCADE")
        await connection.execute("DROP TABLE IF EXISTS market_sentiment CASCADE")
        await connection.execute("DROP TABLE IF EXISTS asset_correlations CASCADE")
        
        # Remove columns from signals table
        await connection.execute("""
            ALTER TABLE signals 
            DROP COLUMN IF EXISTS volume_analysis_score,
            DROP COLUMN IF EXISTS volume_health_score,
            DROP COLUMN IF EXISTS btc_dominance_score,
            DROP COLUMN IF EXISTS market_cap_correlation,
            DROP COLUMN IF EXISTS market_sentiment_score
        """)
        
        logger.info("✅ Phase 3-4 Volume Analysis and Market Intelligence tables removed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error removing Phase 3-4 tables: {e}")
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
        
        logger.info("🎉 Phase 3-4 migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Phase 3-4 migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
