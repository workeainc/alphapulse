"""
Migration: Real-Time Signal Enhancement
Add real-time processing capabilities to existing signal tables
"""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import asyncpg

logger = logging.getLogger(__name__)

async def upgrade(connection: asyncpg.Connection) -> None:
    """Upgrade database schema for real-time signal processing"""
    
    logger.info("🔄 Starting real-time signal enhancement migration...")
    
    try:
        # 1. Add missing columns to signals table
        await connection.execute("""
            ALTER TABLE signals 
            ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS risk_reward_ratio FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS volume_confirmation BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS trend_alignment BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS market_regime VARCHAR(50),
            ADD COLUMN IF NOT EXISTS validation_metrics JSONB,
            ADD COLUMN IF NOT EXISTS ensemble_votes JSONB,
            ADD COLUMN IF NOT EXISTS confidence_breakdown JSONB,
            ADD COLUMN IF NOT EXISTS news_impact_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS sentiment_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS health_score FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS signal_priority INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP,
            ADD COLUMN IF NOT EXISTS cancelled_reason TEXT,
            ADD COLUMN IF NOT EXISTS real_time_processing_time_ms FLOAT DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS notification_sent BOOLEAN DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS external_alert_sent BOOLEAN DEFAULT FALSE
        """)
        
        # 2. Add indexes for real-time queries
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_active_status 
            ON signals(is_active, symbol, ts DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_priority 
            ON signals(signal_priority DESC, confidence DESC, ts DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_health_score 
            ON signals(health_score DESC, ts DESC)
        """)
        
        # 3. Create or enhance performance_metrics table
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                test_run_id VARCHAR(100) NOT NULL,
                latency_avg_ms FLOAT DEFAULT 0.0,
                latency_max_ms FLOAT DEFAULT 0.0,
                throughput_signals_sec FLOAT DEFAULT 0.0,
                accuracy FLOAT DEFAULT 0.0,
                filter_rate FLOAT DEFAULT 0.0,
                real_time_latency_avg_ms FLOAT DEFAULT 0.0,
                real_time_latency_max_ms FLOAT DEFAULT 0.0,
                signal_generation_rate_per_min FLOAT DEFAULT 0.0,
                ensemble_accuracy FLOAT DEFAULT 0.0,
                news_reaction_time_ms FLOAT DEFAULT 0.0,
                notification_delivery_time_ms FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # 4. Create real-time signal queue table (regular table, not hypertable)
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS real_time_signal_queue (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100),
                symbol VARCHAR(20) NOT NULL,
                priority INTEGER DEFAULT 0,
                confidence FLOAT NOT NULL,
                health_score FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                processed_at TIMESTAMP,
                notification_sent BOOLEAN DEFAULT FALSE,
                status VARCHAR(20) DEFAULT 'pending'
            )
        """)
        
        # 5. Add indexes for signal queue
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_queue_priority 
            ON real_time_signal_queue(priority DESC, confidence DESC, created_at DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_queue_status 
            ON real_time_signal_queue(status, created_at DESC)
        """)
        
        # 6. Create notification tracking table (regular table, not hypertable)
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_notifications (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100),
                notification_type VARCHAR(50) NOT NULL,
                channel VARCHAR(50) NOT NULL,
                sent_at TIMESTAMP DEFAULT NOW(),
                delivery_status VARCHAR(20) DEFAULT 'sent',
                delivery_time_ms FLOAT DEFAULT 0.0,
                error_message TEXT
            )
        """)
        
        # 7. Add indexes for notifications
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_notifications_signal_id 
            ON signal_notifications(signal_id, sent_at DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_notifications_channel 
            ON signal_notifications(channel, sent_at DESC)
        """)
        
        # 8. Create ensemble model voting table (regular table, not hypertable)
        await connection.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_model_votes (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100),
                model_name VARCHAR(100) NOT NULL,
                vote_confidence FLOAT NOT NULL,
                vote_direction VARCHAR(10) NOT NULL,
                model_weight FLOAT DEFAULT 1.0,
                processing_time_ms FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # 9. Add indexes for model votes
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_votes_signal_id 
            ON ensemble_model_votes(signal_id, created_at DESC)
        """)
        
        await connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_votes_model_name 
            ON ensemble_model_votes(model_name, created_at DESC)
        """)
        
        logger.info("✅ Real-time signal enhancement migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

async def downgrade(connection: asyncpg.Connection) -> None:
    """Downgrade database schema"""
    
    logger.info("🔄 Rolling back real-time signal enhancement migration...")
    
    try:
        # Remove new tables
        await connection.execute("DROP TABLE IF EXISTS ensemble_model_votes CASCADE")
        await connection.execute("DROP TABLE IF EXISTS signal_notifications CASCADE")
        await connection.execute("DROP TABLE IF EXISTS real_time_signal_queue CASCADE")
        
        # Remove new columns from signals table
        await connection.execute("""
            ALTER TABLE signals 
            DROP COLUMN IF EXISTS confidence,
            DROP COLUMN IF EXISTS risk_reward_ratio,
            DROP COLUMN IF EXISTS volume_confirmation,
            DROP COLUMN IF EXISTS trend_alignment,
            DROP COLUMN IF EXISTS market_regime,
            DROP COLUMN IF EXISTS validation_metrics,
            DROP COLUMN IF EXISTS ensemble_votes,
            DROP COLUMN IF EXISTS confidence_breakdown,
            DROP COLUMN IF EXISTS news_impact_score,
            DROP COLUMN IF EXISTS sentiment_score,
            DROP COLUMN IF EXISTS health_score,
            DROP COLUMN IF EXISTS signal_priority,
            DROP COLUMN IF EXISTS is_active,
            DROP COLUMN IF EXISTS expires_at,
            DROP COLUMN IF EXISTS cancelled_reason,
            DROP COLUMN IF EXISTS real_time_processing_time_ms,
            DROP COLUMN IF EXISTS notification_sent,
            DROP COLUMN IF EXISTS external_alert_sent
        """)
        
        # Remove new columns from performance_metrics table
        await connection.execute("""
            ALTER TABLE performance_metrics 
            DROP COLUMN IF EXISTS real_time_latency_avg_ms,
            DROP COLUMN IF EXISTS real_time_latency_max_ms,
            DROP COLUMN IF EXISTS signal_generation_rate_per_min,
            DROP COLUMN IF EXISTS ensemble_accuracy,
            DROP COLUMN IF EXISTS news_reaction_time_ms,
            DROP COLUMN IF EXISTS notification_delivery_time_ms
        """)
        
        # Remove indexes
        await connection.execute("DROP INDEX IF EXISTS idx_signals_active_status")
        await connection.execute("DROP INDEX IF EXISTS idx_signals_priority")
        await connection.execute("DROP INDEX IF EXISTS idx_signals_health_score")
        await connection.execute("DROP INDEX IF EXISTS idx_signal_queue_priority")
        await connection.execute("DROP INDEX IF EXISTS idx_signal_queue_status")
        await connection.execute("DROP INDEX IF EXISTS idx_notifications_signal_id")
        await connection.execute("DROP INDEX IF EXISTS idx_notifications_channel")
        await connection.execute("DROP INDEX IF EXISTS idx_model_votes_signal_id")
        await connection.execute("DROP INDEX IF EXISTS idx_model_votes_model_name")
        
        logger.info("✅ Rollback completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Rollback failed: {e}")
        raise
