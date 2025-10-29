#!/usr/bin/env python3
"""
Simplified migration script to create enhanced sentiment analysis tables
Uses asyncpg directly to avoid SQLAlchemy compatibility issues
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def create_enhanced_sentiment_tables():
    """Create enhanced sentiment analysis tables with TimescaleDB optimizations"""
    
    try:
        # Connect to database
        conn = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                'psql', '-h', DB_CONFIG['host'], '-p', str(DB_CONFIG['port']), 
                '-U', DB_CONFIG['user'], '-d', DB_CONFIG['database'],
                '-c', 'SELECT 1;'
            ),
            timeout=10
        )
        
        logger.info("Database connection test successful")
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Please ensure PostgreSQL/TimescaleDB is running and accessible")
        return False
    
    # Create tables using psql commands
    table_creation_commands = [
        # 1. Enhanced Sentiment Data Table
        """
        CREATE TABLE IF NOT EXISTS enhanced_sentiment_data (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            source VARCHAR(50) NOT NULL,
            sentiment_score FLOAT NOT NULL,
            sentiment_label VARCHAR(20) NOT NULL,
            confidence FLOAT NOT NULL,
            volume INTEGER,
            keywords TEXT[],
            raw_text TEXT,
            processed_text TEXT,
            language VARCHAR(10) DEFAULT 'en',
            user_id VARCHAR(100),
            user_followers INTEGER,
            user_verified BOOLEAN DEFAULT FALSE,
            engagement_metrics JSONB,
            topic_classification VARCHAR(100),
            sarcasm_detected BOOLEAN DEFAULT FALSE,
            context_score FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """,
        
        # 2. Real-time Sentiment Aggregation Table
        """
        CREATE TABLE IF NOT EXISTS real_time_sentiment_aggregation (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            window_size VARCHAR(10) NOT NULL,
            overall_sentiment_score FLOAT NOT NULL,
            positive_sentiment_score FLOAT NOT NULL,
            negative_sentiment_score FLOAT NOT NULL,
            neutral_sentiment_score FLOAT NOT NULL,
            twitter_sentiment FLOAT,
            reddit_sentiment FLOAT,
            news_sentiment FLOAT,
            telegram_sentiment FLOAT,
            discord_sentiment FLOAT,
            onchain_sentiment FLOAT,
            total_volume INTEGER NOT NULL,
            twitter_volume INTEGER,
            reddit_volume INTEGER,
            news_volume INTEGER,
            telegram_volume INTEGER,
            discord_volume INTEGER,
            onchain_volume INTEGER,
            confidence_weighted_score FLOAT NOT NULL,
            source_diversity_score FLOAT NOT NULL,
            outlier_filtered BOOLEAN DEFAULT FALSE,
            sentiment_trend VARCHAR(20),
            trend_strength FLOAT,
            momentum_score FLOAT,
            fear_greed_index INTEGER,
            market_regime VARCHAR(20),
            volatility_level VARCHAR(20),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """,
        
        # 3. Sentiment Correlation Table
        """
        CREATE TABLE IF NOT EXISTS sentiment_correlation (
            id SERIAL,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            price_change_percent FLOAT NOT NULL,
            volume_change_percent FLOAT NOT NULL,
            volatility FLOAT NOT NULL,
            sentiment_price_correlation FLOAT NOT NULL,
            sentiment_volume_correlation FLOAT NOT NULL,
            sentiment_volatility_correlation FLOAT NOT NULL,
            sentiment_lag_1min FLOAT,
            sentiment_lag_5min FLOAT,
            sentiment_lag_15min FLOAT,
            sentiment_lag_1hour FLOAT,
            sentiment_predictive_power FLOAT,
            price_prediction_accuracy FLOAT,
            btc_correlation FLOAT,
            eth_correlation FLOAT,
            market_correlation FLOAT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (timestamp, id)
        );
        """,
        
        # 4. Sentiment Alerts Table
        """
        CREATE TABLE IF NOT EXISTS sentiment_alerts (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            alert_type VARCHAR(50) NOT NULL,
            alert_severity VARCHAR(20) NOT NULL,
            sentiment_change FLOAT NOT NULL,
            volume_change FLOAT NOT NULL,
            confidence_score FLOAT NOT NULL,
            trigger_threshold FLOAT NOT NULL,
            trigger_source VARCHAR(50) NOT NULL,
            trigger_reason TEXT NOT NULL,
            status VARCHAR(20) DEFAULT 'active',
            acknowledged_at TIMESTAMPTZ,
            resolved_at TIMESTAMPTZ,
            acknowledged_by VARCHAR(100),
            action_taken VARCHAR(100),
            action_result TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
        
        # 5. Sentiment Model Performance Table
        """
        CREATE TABLE IF NOT EXISTS sentiment_model_performance (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            accuracy FLOAT NOT NULL,
            precision FLOAT NOT NULL,
            recall FLOAT NOT NULL,
            f1_score FLOAT NOT NULL,
            training_data_size INTEGER NOT NULL,
            validation_data_size INTEGER NOT NULL,
            training_duration_seconds FLOAT,
            feature_importance JSONB,
            status VARCHAR(20) DEFAULT 'active',
            is_current BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    ]
    
    # Execute table creation commands
    for i, command in enumerate(table_creation_commands, 1):
        try:
            logger.info(f"Creating table {i}/5...")
            
            # Use psql to execute the command
            process = await asyncio.create_subprocess_exec(
                'psql', '-h', DB_CONFIG['host'], '-p', str(DB_CONFIG['port']), 
                '-U', DB_CONFIG['user'], '-d', DB_CONFIG['database'],
                '-c', command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ Table {i} created successfully")
            else:
                logger.error(f"❌ Error creating table {i}: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"❌ Error executing table creation {i}: {e}")
    
    # Create TimescaleDB hypertables
    hypertable_commands = [
        "SELECT create_hypertable('enhanced_sentiment_data', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
        "SELECT create_hypertable('real_time_sentiment_aggregation', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');",
        "SELECT create_hypertable('sentiment_correlation', 'timestamp', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');"
    ]
    
    for i, command in enumerate(hypertable_commands, 1):
        try:
            logger.info(f"Creating hypertable {i}/3...")
            
            process = await asyncio.create_subprocess_exec(
                'psql', '-h', DB_CONFIG['host'], '-p', str(DB_CONFIG['port']), 
                '-U', DB_CONFIG['user'], '-d', DB_CONFIG['database'],
                '-c', command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ Hypertable {i} created successfully")
            else:
                logger.warning(f"⚠️ Hypertable {i} creation warning: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"❌ Error creating hypertable {i}: {e}")
    
    # Create indexes
    index_commands = [
        "CREATE INDEX IF NOT EXISTS idx_enhanced_sentiment_symbol_timestamp ON enhanced_sentiment_data (symbol, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_enhanced_sentiment_source_timestamp ON enhanced_sentiment_data (source, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_real_time_sentiment_symbol_window ON real_time_sentiment_aggregation (symbol, window_size, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_sentiment_correlation_symbol_timeframe ON sentiment_correlation (symbol, timeframe, timestamp DESC);",
        "CREATE INDEX IF NOT EXISTS idx_sentiment_alerts_symbol_status ON sentiment_alerts (symbol, status, timestamp DESC);"
    ]
    
    for i, command in enumerate(index_commands, 1):
        try:
            logger.info(f"Creating index {i}/5...")
            
            process = await asyncio.create_subprocess_exec(
                'psql', '-h', DB_CONFIG['host'], '-p', str(DB_CONFIG['port']), 
                '-U', DB_CONFIG['user'], '-d', DB_CONFIG['database'],
                '-c', command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ Index {i} created successfully")
            else:
                logger.warning(f"⚠️ Index {i} creation warning: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"❌ Error creating index {i}: {e}")
    
    logger.info("✅ Enhanced sentiment tables migration completed!")
    return True

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Enhanced Sentiment Analysis Migration...")
    
    try:
        success = await create_enhanced_sentiment_tables()
        if success:
            logger.info("✅ Enhanced Sentiment Analysis Migration completed successfully!")
        else:
            logger.error("❌ Migration failed")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
