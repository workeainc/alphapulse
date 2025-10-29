#!/usr/bin/env python3
"""
Migration script to create active learning tables
Phase 3: Active Learning Loop Implementation
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection string (update this with your actual connection)
DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def create_active_learning_tables():
    """Create the active learning queue table for manual labeling"""
    
    engine = create_async_engine(DATABASE_URL)
    
    try:
        async with engine.begin() as conn:
            logger.info("🚀 Starting active learning tables creation...")
            
            # 1. Create active_learning_queue table
            logger.info("📊 Creating active_learning_queue table...")
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS active_learning_queue (
                    id SERIAL PRIMARY KEY,
                    signal_id INTEGER,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    prediction_confidence FLOAT NOT NULL,
                    predicted_label VARCHAR(10),
                    predicted_probability FLOAT,
                    features JSONB,
                    market_data JSONB,
                    model_id VARCHAR(50),
                    timestamp TIMESTAMPTZ NOT NULL,
                    
                    -- Manual labeling fields
                    manual_label VARCHAR(10),
                    labeled_by VARCHAR(100),
                    labeled_at TIMESTAMPTZ,
                    labeling_notes TEXT,
                    
                    -- Status tracking
                    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'labeled', 'processed', 'skipped'
                    priority INTEGER DEFAULT 1,
                    
                    -- Integration with retrain queue
                    retrain_queue_id INTEGER,
                    
                    -- Metadata
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Constraints
                    CONSTRAINT valid_confidence CHECK (prediction_confidence >= 0.0 AND prediction_confidence <= 1.0),
                    CONSTRAINT valid_status CHECK (status IN ('pending', 'labeled', 'processed', 'skipped')),
                    CONSTRAINT valid_labels CHECK (predicted_label IN ('BUY', 'SELL', 'HOLD') AND 
                                                 (manual_label IS NULL OR manual_label IN ('BUY', 'SELL', 'HOLD')))
                )
            """))
            
            # 2. Create indexes for optimal performance
            logger.info("🔍 Creating performance indexes...")
            
            # Primary query indexes
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_learning_status_priority 
                ON active_learning_queue (status, priority DESC, created_at)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_learning_confidence 
                ON active_learning_queue (prediction_confidence)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_learning_symbol_timestamp 
                ON active_learning_queue (symbol, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_learning_model_id 
                ON active_learning_queue (model_id, timestamp DESC)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_learning_features 
                ON active_learning_queue USING GIN (features)
            """))
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_active_learning_retrain_queue 
                ON active_learning_queue (retrain_queue_id)
            """))
            
            # 3. Create view for easy querying of pending items
            logger.info("👁️ Creating active learning views...")
            await conn.execute(text("""
                CREATE OR REPLACE VIEW active_learning_pending AS
                SELECT 
                    id,
                    symbol,
                    timeframe,
                    prediction_confidence,
                    predicted_label,
                    predicted_probability,
                    features,
                    model_id,
                    timestamp,
                    created_at,
                    priority
                FROM active_learning_queue 
                WHERE status = 'pending'
                ORDER BY priority DESC, created_at ASC
            """))
            
            # 4. Create view for labeling statistics
            await conn.execute(text("""
                CREATE OR REPLACE VIEW active_learning_stats AS
                SELECT 
                    status,
                    COUNT(*) as count,
                    AVG(prediction_confidence) as avg_confidence,
                    MIN(prediction_confidence) as min_confidence,
                    MAX(prediction_confidence) as max_confidence,
                    COUNT(CASE WHEN manual_label IS NOT NULL THEN 1 END) as labeled_count,
                    COUNT(CASE WHEN retrain_queue_id IS NOT NULL THEN 1 END) as processed_count
                FROM active_learning_queue 
                GROUP BY status
            """))
            
            # 5. Create function to automatically add low-confidence predictions
            logger.info("⚙️ Creating helper functions...")
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION add_low_confidence_prediction(
                    p_signal_id INTEGER,
                    p_symbol VARCHAR(20),
                    p_timeframe VARCHAR(10),
                    p_prediction_confidence FLOAT,
                    p_predicted_label VARCHAR(10),
                    p_predicted_probability FLOAT,
                    p_features JSONB,
                    p_market_data JSONB,
                    p_model_id VARCHAR(50),
                    p_timestamp TIMESTAMPTZ
                ) RETURNS INTEGER AS $$
                DECLARE
                    queue_id INTEGER;
                BEGIN
                    -- Only add if confidence is in the low-confidence range (0.45-0.55)
                    IF p_prediction_confidence >= 0.45 AND p_prediction_confidence <= 0.55 THEN
                        INSERT INTO active_learning_queue (
                            signal_id, symbol, timeframe, prediction_confidence,
                            predicted_label, predicted_probability, features,
                            market_data, model_id, timestamp, priority
                        ) VALUES (
                            p_signal_id, p_symbol, p_timeframe, p_prediction_confidence,
                            p_predicted_label, p_predicted_probability, p_features,
                            p_market_data, p_model_id, p_timestamp,
                            CASE 
                                WHEN p_prediction_confidence BETWEEN 0.48 AND 0.52 THEN 3  -- Highest priority
                                WHEN p_prediction_confidence BETWEEN 0.46 AND 0.54 THEN 2  -- Medium priority
                                ELSE 1  -- Lower priority
                            END
                        ) RETURNING id INTO queue_id;
                        
                        RETURN queue_id;
                    ELSE
                        RETURN NULL;
                    END IF;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            # 6. Create function to process labeled items
            await conn.execute(text("""
                CREATE OR REPLACE FUNCTION process_labeled_item(
                    p_queue_id INTEGER,
                    p_manual_label VARCHAR(10),
                    p_labeled_by VARCHAR(100),
                    p_labeling_notes TEXT DEFAULT NULL
                ) RETURNS INTEGER AS $$
                DECLARE
                    retrain_id INTEGER;
                    signal_id_val INTEGER;
                BEGIN
                    -- Update the queue item
                    UPDATE active_learning_queue 
                    SET 
                        manual_label = p_manual_label,
                        labeled_by = p_labeled_by,
                        labeled_at = NOW(),
                        labeling_notes = p_labeling_notes,
                        status = 'labeled',
                        updated_at = NOW()
                    WHERE id = p_queue_id;
                    
                    -- Get the signal_id for retrain queue
                    SELECT signal_id INTO signal_id_val 
                    FROM active_learning_queue 
                    WHERE id = p_queue_id;
                    
                    -- Add to retrain queue if we have a signal_id
                    IF signal_id_val IS NOT NULL THEN
                        INSERT INTO retrain_queue (
                            signal_id, reason, priority, status
                        ) VALUES (
                            signal_id_val, 
                            'active_learning_labeled', 
                            2,  -- Medium priority for active learning
                            'pending'
                        ) RETURNING id INTO retrain_id;
                        
                        -- Update the queue item with retrain queue reference
                        UPDATE active_learning_queue 
                        SET retrain_queue_id = retrain_id, status = 'processed'
                        WHERE id = p_queue_id;
                        
                        RETURN retrain_id;
                    ELSE
                        RETURN NULL;
                    END IF;
                END;
                $$ LANGUAGE plpgsql;
            """))
            
            logger.info("✅ Active learning tables created successfully!")
            
            # 7. Verify table creation
            logger.info("🔍 Verifying table creation...")
            result = await conn.execute(text("""
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_name = 'active_learning_queue'
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            for table in tables:
                logger.info(f"   - {table[0]} ({table[1]})")
            
            # 8. Verify views
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_name LIKE 'active_learning_%'
                ORDER BY table_name
            """))
            
            views = result.fetchall()
            for view in views:
                logger.info(f"   - View: {view[0]}")
            
            # 9. Verify functions
            result = await conn.execute(text("""
                SELECT routine_name 
                FROM information_schema.routines 
                WHERE routine_name LIKE 'add_low_confidence_prediction' 
                   OR routine_name LIKE 'process_labeled_item'
                ORDER BY routine_name
            """))
            
            functions = result.fetchall()
            for func in functions:
                logger.info(f"   - Function: {func[0]}")
            
            logger.info("🎉 Active learning migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error during migration: {e}")
        raise
    finally:
        await engine.dispose()

async def main():
    """Main migration function"""
    try:
        await create_active_learning_tables()
        logger.info("🚀 Active learning tables migration completed!")
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
