"""
SDE Integration Phase 1: Model Consensus Integration with Signal Generation
Implements model consensus tracking, enhanced signal validation, and integration metrics
"""

import asyncio
import logging
import asyncpg
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migration():
    """Run SDE Integration Phase 1 migration"""
    try:
        # Database connection
        pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='alpha_emon',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        async with pool.acquire() as conn:
            logger.info("🚀 Starting SDE Integration Phase 1 Migration")
            
            # 1. Create SDE Model Consensus Tracking Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_model_consensus_tracking (
                    consensus_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id VARCHAR(50),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Model Head Results
                    head_a_direction VARCHAR(10),
                    head_a_probability DECIMAL(5,4),
                    head_a_confidence DECIMAL(5,4),
                    
                    head_b_direction VARCHAR(10),
                    head_b_probability DECIMAL(5,4),
                    head_b_confidence DECIMAL(5,4),
                    
                    head_c_direction VARCHAR(10),
                    head_c_probability DECIMAL(5,4),
                    head_c_confidence DECIMAL(5,4),
                    
                    head_d_direction VARCHAR(10),
                    head_d_probability DECIMAL(5,4),
                    head_d_confidence DECIMAL(5,4),
                    
                    -- Consensus Results
                    consensus_achieved BOOLEAN NOT NULL,
                    consensus_direction VARCHAR(10),
                    consensus_probability DECIMAL(5,4),
                    agreeing_heads_count INTEGER NOT NULL,
                    min_agreeing_heads INTEGER NOT NULL,
                    min_head_probability DECIMAL(5,4) NOT NULL,
                    
                    -- Integration Metadata
                    processing_time_ms INTEGER,
                    consensus_reason TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_model_consensus_tracking table")
            
            # 2. Create Enhanced Signal Validation Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_signal_validation (
                    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id VARCHAR(50),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- SDE Framework Results
                    consensus_passed BOOLEAN NOT NULL,
                    confluence_passed BOOLEAN NOT NULL,
                    execution_passed BOOLEAN NOT NULL,
                    news_blackout_passed BOOLEAN NOT NULL,
                    signal_limits_passed BOOLEAN NOT NULL,
                    
                    -- Detailed Scores
                    consensus_score DECIMAL(5,4),
                    confluence_score DECIMAL(5,4),
                    execution_quality_score DECIMAL(5,4),
                    
                    -- Gate Results
                    final_confidence DECIMAL(5,4),
                    confidence_threshold DECIMAL(5,4) NOT NULL,
                    confidence_passed BOOLEAN NOT NULL,
                    
                    -- Integration Metadata
                    sde_version VARCHAR(20) DEFAULT '1.0.0',
                    processing_time_ms INTEGER,
                    validation_reason TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_signal_validation table")
            
            # 3. Create SDE Integration Metrics Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_integration_metrics (
                    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    metric_date DATE NOT NULL,
                    symbol VARCHAR(20),
                    timeframe VARCHAR(10),
                    
                    -- Signal Counts
                    total_signals_generated INTEGER DEFAULT 0,
                    signals_passing_consensus INTEGER DEFAULT 0,
                    signals_passing_confluence INTEGER DEFAULT 0,
                    signals_passing_execution INTEGER DEFAULT 0,
                    signals_passing_news_check INTEGER DEFAULT 0,
                    signals_passing_limits INTEGER DEFAULT 0,
                    signals_passing_confidence INTEGER DEFAULT 0,
                    final_signals_emitted INTEGER DEFAULT 0,
                    
                    -- Quality Metrics
                    avg_consensus_score DECIMAL(5,4),
                    avg_confluence_score DECIMAL(5,4),
                    avg_execution_quality DECIMAL(5,4),
                    avg_final_confidence DECIMAL(5,4),
                    
                    -- Performance Metrics
                    avg_processing_time_ms INTEGER,
                    max_processing_time_ms INTEGER,
                    min_processing_time_ms INTEGER,
                    
                    -- Rejection Reasons
                    consensus_rejections INTEGER DEFAULT 0,
                    confluence_rejections INTEGER DEFAULT 0,
                    execution_rejections INTEGER DEFAULT 0,
                    news_rejections INTEGER DEFAULT 0,
                    limits_rejections INTEGER DEFAULT 0,
                    confidence_rejections INTEGER DEFAULT 0,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_integration_metrics table")
            
            # 4. Create SDE Integration Configuration Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_integration_config (
                    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    config_name VARCHAR(100) UNIQUE NOT NULL,
                    config_type VARCHAR(50) NOT NULL,
                    config_data JSONB NOT NULL,
                    is_active BOOLEAN DEFAULT true,
                    description TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_integration_config table")
            
            # 5. Create Performance Indexes
            indexes = [
                ("idx_sde_consensus_signal_id", "sde_model_consensus_tracking", "signal_id"),
                ("idx_sde_consensus_symbol_time", "sde_model_consensus_tracking", "symbol, timestamp"),
                ("idx_sde_consensus_achieved", "sde_model_consensus_tracking", "consensus_achieved"),
                
                ("idx_sde_validation_signal_id", "sde_signal_validation", "signal_id"),
                ("idx_sde_validation_symbol_time", "sde_signal_validation", "symbol, timestamp"),
                ("idx_sde_validation_confidence", "sde_signal_validation", "confidence_passed"),
                
                ("idx_sde_metrics_date", "sde_integration_metrics", "metric_date"),
                ("idx_sde_metrics_symbol", "sde_integration_metrics", "symbol"),
                ("idx_sde_metrics_timeframe", "sde_integration_metrics", "timeframe"),
                
                ("idx_sde_config_active", "sde_integration_config", "is_active"),
                ("idx_sde_config_type", "sde_integration_config", "config_type")
            ]
            
            for index_name, table_name, columns in indexes:
                try:
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})")
                    logger.info(f"✅ Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create index {index_name}: {e}")
            
            # 6. Insert Default Integration Configurations
            default_configs = [
                {
                    'config_name': 'sde_integration_default',
                    'config_type': 'integration',
                    'config_data': {
                        'confidence_threshold': 0.85,
                        'consensus_threshold': 0.70,
                        'confluence_threshold': 8.0,
                        'execution_quality_threshold': 7.0,
                        'max_processing_time_ms': 100,
                        'enable_news_blackout': True,
                        'enable_signal_limits': True,
                        'enable_consensus_tracking': True,
                        'enable_validation_tracking': True,
                        'enable_metrics_tracking': True
                    },
                    'description': 'Default SDE integration configuration'
                },
                {
                    'config_name': 'sde_performance_default',
                    'config_type': 'performance',
                    'config_data': {
                        'target_processing_time_ms': 50,
                        'max_processing_time_ms': 100,
                        'cache_consensus_results': True,
                        'cache_confluence_results': True,
                        'parallel_processing': True,
                        'batch_processing': False,
                        'enable_profiling': True
                    },
                    'description': 'Default SDE performance configuration'
                },
                {
                    'config_name': 'sde_quality_default',
                    'config_type': 'quality',
                    'config_data': {
                        'min_consensus_score': 0.70,
                        'min_confluence_score': 8.0,
                        'min_execution_quality': 7.0,
                        'min_final_confidence': 0.85,
                        'quality_decay_factor': 0.95,
                        'enable_quality_tracking': True,
                        'enable_quality_alerts': True
                    },
                    'description': 'Default SDE quality configuration'
                },
                {
                    'config_name': 'sde_monitoring_default',
                    'config_type': 'monitoring',
                    'config_data': {
                        'metrics_collection_interval': 300,  # 5 minutes
                        'alert_threshold_processing_time': 100,
                        'alert_threshold_rejection_rate': 0.3,
                        'alert_threshold_confidence_drop': 0.1,
                        'enable_real_time_monitoring': True,
                        'enable_alert_notifications': True,
                        'enable_performance_tracking': True
                    },
                    'description': 'Default SDE monitoring configuration'
                }
            ]
            
            for config in default_configs:
                try:
                    await conn.execute("""
                        INSERT INTO sde_integration_config (config_name, config_type, config_data, description)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (config_name) DO UPDATE SET
                            config_data = EXCLUDED.config_data,
                            description = EXCLUDED.description,
                            updated_at = NOW()
                    """, config['config_name'], config['config_type'], 
                         json.dumps(config['config_data']), config['description'])
                    logger.info(f"✅ Inserted config: {config['config_name']}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to insert config {config['config_name']}: {e}")
            
            logger.info("🎉 SDE Integration Phase 1 Migration Completed Successfully!")
            
        await pool.close()
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
