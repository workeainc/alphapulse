"""
Database Setup Script for AlphaPulse
Creates all necessary tables for the intelligent signal generator
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def setup_database_tables():
    """Setup all necessary database tables"""
    
    # Database connection parameters
    db_config = {
        'host': 'postgres',  # Docker service name
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**db_config)
        logger.info("✅ Connected to database")
        
        # 1. Create market_intelligence table
        market_intelligence_table = """
        CREATE TABLE IF NOT EXISTS market_intelligence (
            timestamp TIMESTAMPTZ NOT NULL,
            btc_dominance DECIMAL(8,4),
            total2_value DECIMAL(30,8),
            total3_value DECIMAL(30,8),
            market_sentiment_score DECIMAL(8,4),
            news_sentiment_score DECIMAL(8,4),
            volume_positioning_score DECIMAL(8,4),
            fear_greed_index INTEGER,
            market_regime VARCHAR(20),
            volatility_index DECIMAL(8,4),
            trend_strength DECIMAL(8,4),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        await conn.execute(market_intelligence_table)
        logger.info("✅ Created market_intelligence table")
        
        # 2. Create volume_analysis table
        volume_analysis_table = """
        CREATE TABLE IF NOT EXISTS volume_analysis (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) DEFAULT '1h',
            volume_ratio DECIMAL(8,4),
            volume_trend VARCHAR(20),
            order_book_imbalance DECIMAL(8,4),
            volume_positioning_score DECIMAL(8,4),
            buy_volume_ratio DECIMAL(8,4),
            sell_volume_ratio DECIMAL(8,4),
            volume_breakout BOOLEAN DEFAULT FALSE,
            volume_analysis TEXT,
            analysis_confidence DECIMAL(5,4) DEFAULT 0.5,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        await conn.execute(volume_analysis_table)
        logger.info("✅ Created volume_analysis table")
        
        # 3. Create candles table
        candles_table = """
        CREATE TABLE IF NOT EXISTS candles (
            ts TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            tf VARCHAR(10) NOT NULL,
            o DECIMAL(20,8),
            h DECIMAL(20,8),
            l DECIMAL(20,8),
            c DECIMAL(20,8),
            v DECIMAL(20,8),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        await conn.execute(candles_table)
        logger.info("✅ Created candles table")
        
        # 4. Create price_action_ml_predictions table
        price_action_table = """
        CREATE TABLE IF NOT EXISTS price_action_ml_predictions (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            prediction_type VARCHAR(50),
            prediction_probability DECIMAL(8,4),
            confidence_score DECIMAL(8,4),
            feature_vector JSONB,
            model_output JSONB,
            market_context JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        await conn.execute(price_action_table)
        logger.info("✅ Created price_action_ml_predictions table")
        
        # 5. Create market_regime_data table (if referenced)
        market_regime_data_table = """
        CREATE TABLE IF NOT EXISTS market_regime_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            regime_type VARCHAR(20),
            confidence DECIMAL(8,4),
            volatility DECIMAL(8,4),
            trend_strength DECIMAL(8,4),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        await conn.execute(market_regime_data_table)
        logger.info("✅ Created market_regime_data table")
        
        # Create TimescaleDB hypertables
        try:
            hypertables = [
                "SELECT create_hypertable('market_intelligence', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
                "SELECT create_hypertable('volume_analysis', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
                "SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
                "SELECT create_hypertable('price_action_ml_predictions', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
                "SELECT create_hypertable('market_regime_data', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
            ]
            
            for hypertable in hypertables:
                await conn.execute(hypertable)
            
            logger.info("✅ Created TimescaleDB hypertables")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation warnings: {e}")
        
        # Create indexes
        indexes = [
            # market_intelligence indexes
            "CREATE INDEX IF NOT EXISTS idx_market_intelligence_timestamp ON market_intelligence (timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_intelligence_regime ON market_intelligence (market_regime);",
            
            # volume_analysis indexes
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_symbol_timestamp ON volume_analysis (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_timeframe ON volume_analysis (timeframe);",
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_volume_trend ON volume_analysis (volume_trend);",
            "CREATE INDEX IF NOT EXISTS idx_volume_analysis_confidence ON volume_analysis (analysis_confidence DESC);",
            
            # candles indexes
            "CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts ON candles (symbol, tf, ts DESC);",
            "CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles (symbol);",
            
            # price_action_ml_predictions indexes
            "CREATE INDEX IF NOT EXISTS idx_price_action_symbol_timestamp ON price_action_ml_predictions (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_price_action_prediction_type ON price_action_ml_predictions (prediction_type);",
            
            # market_regime_data indexes
            "CREATE INDEX IF NOT EXISTS idx_market_regime_symbol_timestamp ON market_regime_data (symbol, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_regime_type ON market_regime_data (regime_type);"
        ]
        
        for index in indexes:
            await conn.execute(index)
        
        logger.info("✅ Created all indexes")
        
        # Insert sample data
        now = datetime.now()
        
        # Sample market intelligence data
        market_intelligence_data = [
            (now, 45.2, 1200000000000, 8500000000000, 0.65, 0.58, 0.72, 55, 'bullish', 0.035, 0.68),
            (now, 48.1, 1180000000000, 8400000000000, 0.62, 0.61, 0.69, 52, 'sideways', 0.028, 0.45),
            (now, 52.3, 1150000000000, 8300000000000, 0.58, 0.54, 0.65, 48, 'bearish', 0.042, 0.72)
        ]
        
        await conn.executemany("""
            INSERT INTO market_intelligence (
                timestamp, btc_dominance, total2_value, total3_value,
                market_sentiment_score, news_sentiment_score, volume_positioning_score,
                fear_greed_index, market_regime, volatility_index, trend_strength
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        """, market_intelligence_data)
        
        # Sample volume analysis data
        volume_analysis_data = [
            (now, 'BTC/USDT', '1h', 1.5, 'increasing', 0.2, 0.7, 0.6, 0.4, True, 'High volume with bullish positioning', 0.8),
            (now, 'ETH/USDT', '1h', 0.8, 'stable', -0.1, 0.5, 0.5, 0.5, False, 'Normal volume conditions', 0.6),
            (now, 'SOL/USDT', '1h', 2.1, 'increasing', 0.3, 0.8, 0.7, 0.3, True, 'Volume breakout with strong buying', 0.9)
        ]
        
        await conn.executemany("""
            INSERT INTO volume_analysis (
                timestamp, symbol, timeframe, volume_ratio, volume_trend, 
                order_book_imbalance, volume_positioning_score, buy_volume_ratio, 
                sell_volume_ratio, volume_breakout, volume_analysis, analysis_confidence
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """, volume_analysis_data)
        
        # Sample candles data
        candles_data = [
            (now, 'BTC/USDT', '1h', 45000.0, 45200.0, 44800.0, 45100.0, 1000000.0),
            (now, 'ETH/USDT', '1h', 2800.0, 2820.0, 2780.0, 2810.0, 500000.0),
            (now, 'SOL/USDT', '1h', 95.0, 96.0, 94.0, 95.5, 200000.0)
        ]
        
        await conn.executemany("""
            INSERT INTO candles (
                ts, symbol, tf, o, h, l, c, v
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, candles_data)
        
        # Sample market regime data
        market_regime_data = [
            (now, 'BTC/USDT', 'bullish', 0.75, 0.035, 0.68),
            (now, 'ETH/USDT', 'sideways', 0.65, 0.028, 0.45),
            (now, 'SOL/USDT', 'bearish', 0.70, 0.042, 0.72)
        ]
        
        await conn.executemany("""
            INSERT INTO market_regime_data (
                timestamp, symbol, regime_type, confidence, volatility, trend_strength
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """, market_regime_data)
        
        logger.info("✅ Inserted sample data into all tables")
        
        await conn.close()
        logger.info("✅ Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error setting up database tables: {e}")
        raise

async def main():
    """Main function to run the database setup"""
    logger.info("🚀 Starting database table setup...")
    await setup_database_tables()
    logger.info("✅ Database setup completed!")

if __name__ == "__main__":
    asyncio.run(main())
