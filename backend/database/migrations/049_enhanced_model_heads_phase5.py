"""
Migration 049: Enhanced Model Heads Phase 5
Add enhanced model head tracking, ONNX model management, and feature storage
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the enhanced model heads migration"""
    try:
        # Database connection
        db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='alpha_emon',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        async with db_pool.acquire() as conn:
            logger.info("🚀 Starting enhanced model heads migration...")
            
            # Create sde_enhanced_model_heads table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_enhanced_model_heads (
                    id SERIAL PRIMARY KEY,
                    signal_id VARCHAR(50),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    
                    -- Model Head Results
                    head_a_type VARCHAR(30) DEFAULT 'catboost_technical',
                    head_a_direction VARCHAR(10),
                    head_a_probability DECIMAL(6,4),
                    head_a_confidence DECIMAL(6,4),
                    head_a_features_used JSONB,
                    head_a_reasoning TEXT,
                    head_a_model_version VARCHAR(20),
                    
                    head_b_type VARCHAR(30) DEFAULT 'logistic_sentiment',
                    head_b_direction VARCHAR(10),
                    head_b_probability DECIMAL(6,4),
                    head_b_confidence DECIMAL(6,4),
                    head_b_features_used JSONB,
                    head_b_reasoning TEXT,
                    head_b_model_version VARCHAR(20),
                    
                    head_c_type VARCHAR(30) DEFAULT 'tree_orderflow',
                    head_c_direction VARCHAR(10),
                    head_c_probability DECIMAL(6,4),
                    head_c_confidence DECIMAL(6,4),
                    head_c_features_used JSONB,
                    head_c_reasoning TEXT,
                    head_c_model_version VARCHAR(20),
                    
                    head_d_type VARCHAR(30) DEFAULT 'rule_based',
                    head_d_direction VARCHAR(10),
                    head_d_probability DECIMAL(6,4),
                    head_d_confidence DECIMAL(6,4),
                    head_d_features_used JSONB,
                    head_d_reasoning TEXT,
                    head_d_model_version VARCHAR(20),
                    
                    -- Consensus Results
                    consensus_achieved BOOLEAN,
                    consensus_direction VARCHAR(10),
                    consensus_score DECIMAL(6,4),
                    agreeing_heads_count INTEGER,
                    disagreeing_heads_count INTEGER,
                    
                    -- Processing Metadata
                    processing_time_ms INTEGER,
                    model_creation_success BOOLEAN,
                    onnx_inference_used BOOLEAN,
                    feature_engineering_used BOOLEAN,
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_enhanced_model_heads table")
            
            # Create sde_onnx_model_registry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_onnx_model_registry (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(50) UNIQUE NOT NULL,
                    model_type VARCHAR(30) NOT NULL,
                    model_version VARCHAR(20) NOT NULL,
                    
                    -- Model Configuration
                    model_path VARCHAR(255),
                    model_hash VARCHAR(64),
                    model_size_bytes BIGINT,
                    input_shape JSONB,
                    output_shape JSONB,
                    
                    -- Performance Metrics
                    inference_time_ms DECIMAL(8,3),
                    accuracy_score DECIMAL(6,4),
                    precision_score DECIMAL(6,4),
                    recall_score DECIMAL(6,4),
                    f1_score DECIMAL(6,4),
                    
                    -- Deployment Status
                    is_active BOOLEAN DEFAULT true,
                    is_production BOOLEAN DEFAULT false,
                    deployment_timestamp TIMESTAMP,
                    
                    -- Metadata
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_onnx_model_registry table")
            
            # Create sde_enhanced_features table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_enhanced_features (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    feature_timestamp TIMESTAMP NOT NULL,
                    
                    -- Technical Features
                    technical_features JSONB,
                    sentiment_features JSONB,
                    orderflow_features JSONB,
                    rulebased_features JSONB,
                    
                    -- Feature Engineering Metadata
                    feature_engineering_version VARCHAR(20),
                    feature_count INTEGER,
                    feature_quality_score DECIMAL(6,4),
                    missing_features_count INTEGER,
                    
                    -- Processing Metadata
                    processing_time_ms INTEGER,
                    cache_hit BOOLEAN,
                    feature_source VARCHAR(30),
                    
                    -- Timestamps
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_enhanced_features table")
            
            # Create sde_model_head_performance table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_model_head_performance (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    head_type VARCHAR(30) NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    
                    -- Performance Metrics
                    accuracy DECIMAL(6,4),
                    precision DECIMAL(6,4),
                    recall DECIMAL(6,4),
                    f1_score DECIMAL(6,4),
                    win_rate DECIMAL(6,4),
                    profit_factor DECIMAL(6,4),
                    avg_win DECIMAL(10,4),
                    avg_loss DECIMAL(10,4),
                    max_drawdown DECIMAL(6,4),
                    
                    -- Sample Size
                    total_predictions INTEGER,
                    correct_predictions INTEGER,
                    incorrect_predictions INTEGER,
                    
                    -- Time Period
                    start_date DATE,
                    end_date DATE,
                    
                    -- Model Version
                    model_version VARCHAR(20),
                    
                    -- Metadata
                    analysis_timestamp TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_model_head_performance table")
            
            # Insert default ONNX model configurations
            await conn.execute("""
                INSERT INTO sde_onnx_model_registry (
                    model_name, model_type, model_version, model_path, input_shape, output_shape,
                    inference_time_ms, accuracy_score, is_active, is_production
                ) VALUES 
                ('catboost_technical', 'catboost', '1.0.0', '/models/catboost_technical.onnx', 
                 '{"features": 6}', '{"probability": 1}', 5.0, 0.75, true, false),
                ('logistic_sentiment', 'logistic', '1.0.0', '/models/logistic_sentiment.onnx',
                 '{"features": 4}', '{"probability": 1}', 2.0, 0.70, true, false),
                ('tree_orderflow', 'decision_tree', '1.0.0', '/models/tree_orderflow.onnx',
                 '{"features": 4}', '{"probability": 1}', 3.0, 0.65, true, false)
                ON CONFLICT (model_name) DO NOTHING
            """)
            logger.info("✅ Inserted default ONNX model configurations")
            
            # Create indexes for performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_enhanced_model_heads_symbol_timeframe 
                ON sde_enhanced_model_heads(symbol, timeframe)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_enhanced_model_heads_consensus 
                ON sde_enhanced_model_heads(consensus_achieved, consensus_score)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_onnx_model_registry_active 
                ON sde_onnx_model_registry(is_active, is_production)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_enhanced_features_symbol_timeframe 
                ON sde_enhanced_features(symbol, timeframe, feature_timestamp)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sde_model_head_performance_symbol_head 
                ON sde_model_head_performance(symbol, head_type)
            """)
            
            logger.info("✅ Created performance indexes")
            
            logger.info("🎉 Enhanced model heads migration completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'db_pool' in locals():
            await db_pool.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
