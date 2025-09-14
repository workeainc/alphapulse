#!/usr/bin/env python3
"""
Migration script for Enhanced WebSocket System
Phase 2: Enhanced WebSocket Tables and Optimizations
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string
DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def create_enhanced_websocket_tables():
    """Create enhanced WebSocket system tables and optimizations"""
    
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            logger.info("Starting Enhanced WebSocket tables creation...")
            
            # 1. Update signals table to match enhanced WebSocket requirements
            logger.info("Updating signals table for enhanced WebSocket...")
            await conn.execute(text("""
                -- Add missing columns if they don't exist
                DO $$ 
                BEGIN
                    -- Add signal_id column if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'signal_id') THEN
                        ALTER TABLE signals ADD COLUMN signal_id VARCHAR(100);
                        CREATE INDEX IF NOT EXISTS idx_signals_signal_id ON signals (signal_id);
                    END IF;
                    
                    -- Add metadata column if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'metadata') THEN
                        ALTER TABLE signals ADD COLUMN metadata JSONB;
                        CREATE INDEX IF NOT EXISTS idx_signals_metadata ON signals USING GIN (metadata);
                    END IF;
                    
                    -- Add timeframe column if it doesn't exist (rename tf if exists)
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'timeframe') THEN
                        IF EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'tf') THEN
                            ALTER TABLE signals RENAME COLUMN tf TO timeframe;
                        ELSE
                            ALTER TABLE signals ADD COLUMN timeframe VARCHAR(10);
                        END IF;
                        CREATE INDEX IF NOT EXISTS idx_signals_timeframe ON signals (timeframe);
                    END IF;
                    
                    -- Add direction column if it doesn't exist (rename label if exists)
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'direction') THEN
                        IF EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'label') THEN
                            ALTER TABLE signals RENAME COLUMN label TO direction;
                        ELSE
                            ALTER TABLE signals ADD COLUMN direction VARCHAR(10);
                        END IF;
                        CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals (direction);
                    END IF;
                    
                    -- Add entry_price column if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'entry_price') THEN
                        ALTER TABLE signals ADD COLUMN entry_price FLOAT;
                    END IF;
                    
                    -- Add stop_loss column if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'stop_loss') THEN
                        ALTER TABLE signals ADD COLUMN stop_loss FLOAT;
                    END IF;
                    
                    -- Add tp1 column if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'tp1') THEN
                        ALTER TABLE signals ADD COLUMN tp1 FLOAT;
                    END IF;
                    
                    -- Add pattern_type column if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'pattern_type') THEN
                        ALTER TABLE signals ADD COLUMN pattern_type VARCHAR(50);
                    END IF;
                    
                    -- Add indicators column if it doesn't exist (rename features if exists)
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'indicators') THEN
                        IF EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'features') THEN
                            ALTER TABLE signals RENAME COLUMN features TO indicators;
                        ELSE
                            ALTER TABLE signals ADD COLUMN indicators JSONB;
                        END IF;
                        CREATE INDEX IF NOT EXISTS idx_signals_indicators ON signals USING GIN (indicators);
                    END IF;
                    
                    -- Add outcome column if it doesn't exist
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                 WHERE table_name = 'signals' AND column_name = 'outcome') THEN
                        ALTER TABLE signals ADD COLUMN outcome VARCHAR(20) DEFAULT 'pending';
                    END IF;
                    
                END $$;
            """))
            
            # 2. Create WebSocket connection tracking table
            logger.info("Creating WebSocket connections table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS websocket_connections (
                    id SERIAL PRIMARY KEY,
                    client_id VARCHAR(100) UNIQUE NOT NULL,
                    connection_type VARCHAR(50) NOT NULL, -- 'dashboard', 'binance', 'external'
                    status VARCHAR(20) NOT NULL DEFAULT 'connected', -- 'connected', 'disconnected', 'error'
                    connected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    disconnected_at TIMESTAMPTZ,
                    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    messages_sent INTEGER DEFAULT 0,
                    messages_received INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    user_agent TEXT,
                    ip_address INET,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """))
            
            # 3. Create WebSocket performance metrics table
            logger.info("Creating WebSocket performance metrics table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS websocket_performance (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    metric_unit VARCHAR(20), -- 'ms', 'count', 'percentage'
                    connection_type VARCHAR(50),
                    client_id VARCHAR(100),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # 4. Create TimescaleDB hypertable for performance metrics
            await conn.execute(text("""
                SELECT create_hypertable('websocket_performance', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE)
            """))
            
            # 5. Create indexes for optimal performance
            logger.info("Creating performance indexes...")
            
            # WebSocket connections indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_websocket_connections_client_id 
                ON websocket_connections (client_id)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_websocket_connections_status 
                ON websocket_connections (status, last_activity)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_websocket_connections_type_status 
                ON websocket_connections (connection_type, status)
            """))
            
            # WebSocket performance indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_websocket_performance_metric_name 
                ON websocket_performance (metric_name, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_websocket_performance_connection_type 
                ON websocket_performance (connection_type, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_websocket_performance_metadata 
                ON websocket_performance USING GIN (metadata)
            """))
            
            # 6. Enable compression for performance table
            logger.info("Enabling compression for performance table...")
            await conn.execute(text("""
                ALTER TABLE websocket_performance SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'metric_name,connection_type',
                    timescaledb.compress_orderby = 'timestamp DESC'
                )
            """))
            
            # 7. Set up compression and retention policies
            logger.info("Setting up compression and retention policies...")
            
            # Compress performance data older than 1 day
            try:
                await conn.execute(text("""
                    SELECT add_compression_policy('websocket_performance', INTERVAL '1 day', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Compression policy for websocket_performance already exists")
            
            # Keep performance data for 30 days
            try:
                await conn.execute(text("""
                    SELECT add_retention_policy('websocket_performance', INTERVAL '30 days', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Retention policy for websocket_performance already exists")
            
            # 8. Create Redis pub/sub tracking table
            logger.info("Creating Redis pub/sub tracking table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS redis_pubsub_events (
                    id SERIAL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    channel VARCHAR(100) NOT NULL,
                    event_type VARCHAR(50) NOT NULL, -- 'publish', 'subscribe', 'unsubscribe'
                    message_size INTEGER,
                    client_count INTEGER,
                    latency_ms DOUBLE PRECISION,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (timestamp, id)
                )
            """))
            
            # 9. Create TimescaleDB hypertable for Redis events
            await conn.execute(text("""
                SELECT create_hypertable('redis_pubsub_events', 'timestamp', 
                    chunk_time_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE)
            """))
            
            # 10. Create indexes for Redis events
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_redis_pubsub_events_channel 
                ON redis_pubsub_events (channel, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_redis_pubsub_events_type 
                ON redis_pubsub_events (event_type, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_redis_pubsub_events_success 
                ON redis_pubsub_events (success, timestamp DESC)
            """))
            
            # 11. Enable compression for Redis events
            await conn.execute(text("""
                ALTER TABLE redis_pubsub_events SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'channel,event_type',
                    timescaledb.compress_orderby = 'timestamp DESC'
                )
            """))
            
            # 12. Set up compression and retention for Redis events
            try:
                await conn.execute(text("""
                    SELECT add_compression_policy('redis_pubsub_events', INTERVAL '1 day', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Compression policy for redis_pubsub_events already exists")
            
            try:
                await conn.execute(text("""
                    SELECT add_retention_policy('redis_pubsub_events', INTERVAL '30 days', if_not_exists => TRUE)
                """))
            except Exception as e:
                if "already exists" not in str(e):
                    raise
                logger.info("Retention policy for redis_pubsub_events already exists")
            
            # 13. Create function to update connection last activity
            logger.info("Creating utility functions...")
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION update_websocket_connection_activity()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.last_activity = NOW();
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # 14. Create trigger for connection activity updates
            await conn.execute(text("""
                DROP TRIGGER IF EXISTS trigger_update_websocket_connection_activity 
                ON websocket_connections
            """))
            
            await conn.execute(text("""
                CREATE TRIGGER trigger_update_websocket_connection_activity
                    BEFORE UPDATE ON websocket_connections
                    FOR EACH ROW
                    EXECUTE FUNCTION update_websocket_connection_activity()
            """))
            
            # 15. Create function to clean up old connections
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION cleanup_old_websocket_connections()
                RETURNS INTEGER AS $$
                DECLARE
                    deleted_count INTEGER;
                BEGIN
                    DELETE FROM websocket_connections 
                    WHERE last_activity < NOW() - INTERVAL '1 hour'
                    AND status = 'disconnected';
                    
                    GET DIAGNOSTICS deleted_count = ROW_COUNT;
                    RETURN deleted_count;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            logger.info("Enhanced WebSocket tables created successfully!")
            
            # 16. Verify table creation
            logger.info("Verifying table creation...")
            result = await conn.execute(text("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_name IN ('websocket_connections', 'websocket_performance', 'redis_pubsub_events')
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            for table in tables:
                logger.info(f"   - {table[0]} ({table[1]})")
            
            # 17. Verify hypertables
            result = await conn.execute(text("""
                SELECT hypertable_name
                FROM timescaledb_information.hypertables
                WHERE hypertable_name IN ('websocket_performance', 'redis_pubsub_events')
                ORDER BY hypertable_name
            """))
            
            hypertables = result.fetchall()
            for ht in hypertables:
                logger.info(f"   - Hypertable: {ht[0]}")
            
            # 18. Show signals table structure
            logger.info("Current signals table structure:")
            result = await conn.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'signals'
                ORDER BY ordinal_position
            """))
            
            columns = result.fetchall()
            for col in columns:
                logger.info(f"   - {col[0]}: {col[1]}")
            
            logger.info("Enhanced WebSocket migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        raise
    finally:
        await engine.dispose()

async def main():
    """Main migration function"""
    try:
        await create_enhanced_websocket_tables()
        logger.info("Enhanced WebSocket tables migration completed!")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
