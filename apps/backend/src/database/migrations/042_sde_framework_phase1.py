"""
Phase 1: SDE Framework Implementation - Model Consensus & Confluence Scoring
Database migration for Single-Decision Engine framework
"""

import asyncio
import logging
import asyncpg
import json
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'database': 'alphapulse'
}

async def create_sde_tables(pool: asyncpg.Pool):
    """Create SDE framework tables"""
    
    tables = [
        # Model Consensus Tracking
        """
        CREATE TABLE IF NOT EXISTS sde_model_consensus (
            id SERIAL PRIMARY KEY,
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Model Head Results
            head_a_direction VARCHAR(10), -- 'LONG', 'SHORT', 'FLAT'
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
            consensus_achieved BOOLEAN DEFAULT FALSE,
            consensus_direction VARCHAR(10),
            consensus_probability DECIMAL(5,4),
            consensus_score DECIMAL(5,4),
            agreeing_heads_count INTEGER DEFAULT 0,
            
            -- Metadata
            processing_time_ms INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Confluence Scoring
        """
        CREATE TABLE IF NOT EXISTS sde_confluence_scores (
            id SERIAL PRIMARY KEY,
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Individual Component Scores (0-10)
            zone_score DECIMAL(3,1),
            htf_bias_score DECIMAL(3,1),
            trigger_quality_score DECIMAL(3,1),
            fvg_confluence_score DECIMAL(3,1),
            orderbook_confluence_score DECIMAL(3,1),
            sentiment_confluence_score DECIMAL(3,1),
            volume_confluence_score DECIMAL(3,1),
            pattern_confluence_score DECIMAL(3,1),
            
            -- Overall Confluence
            total_confluence_score DECIMAL(3,1),
            confluence_gate_passed BOOLEAN DEFAULT FALSE,
            confluence_breakdown JSONB,
            
            -- Metadata
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Execution Quality Filters
        """
        CREATE TABLE IF NOT EXISTS sde_execution_quality (
            id SERIAL PRIMARY KEY,
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Spread Analysis
            current_spread DECIMAL(10,8),
            atr_value DECIMAL(10,8),
            spread_atr_ratio DECIMAL(5,4),
            spread_gate_passed BOOLEAN DEFAULT FALSE,
            
            -- Volatility Analysis
            atr_percentile DECIMAL(5,2),
            volatility_regime VARCHAR(20), -- 'low', 'normal', 'high', 'extreme'
            volatility_gate_passed BOOLEAN DEFAULT FALSE,
            
            -- Impact Analysis
            orderbook_impact_cost DECIMAL(10,8),
            estimated_slippage DECIMAL(10,8),
            impact_gate_passed BOOLEAN DEFAULT FALSE,
            
            -- Overall Execution Quality
            execution_quality_score DECIMAL(3,1),
            all_gates_passed BOOLEAN DEFAULT FALSE,
            
            -- Metadata
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # SDE Signal Configuration
        """
        CREATE TABLE IF NOT EXISTS sde_config (
            id SERIAL PRIMARY KEY,
            config_name VARCHAR(100) UNIQUE NOT NULL,
            config_type VARCHAR(50) NOT NULL, -- 'consensus', 'confluence', 'execution', 'general'
            config_data JSONB NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # SDE Signal History
        """
        CREATE TABLE IF NOT EXISTS sde_signal_history (
            id SERIAL PRIMARY KEY,
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Signal Details
            signal_type VARCHAR(10), -- 'LONG', 'SHORT', 'FLAT'
            confidence_score DECIMAL(5,4),
            confluence_score DECIMAL(3,1),
            execution_quality_score DECIMAL(3,1),
            
            -- Entry/Exit Levels
            entry_price DECIMAL(15,8),
            stop_loss DECIMAL(15,8),
            take_profit_1 DECIMAL(15,8),
            take_profit_2 DECIMAL(15,8),
            take_profit_3 DECIMAL(15,8),
            take_profit_4 DECIMAL(15,8),
            
            -- Risk Management
            risk_reward_ratio DECIMAL(5,2),
            position_size_percentage DECIMAL(5,2),
            
            -- SDE Framework Results
            consensus_achieved BOOLEAN DEFAULT FALSE,
            confluence_gate_passed BOOLEAN DEFAULT FALSE,
            execution_gate_passed BOOLEAN DEFAULT FALSE,
            sde_final_decision VARCHAR(10), -- 'LONG', 'SHORT', 'FLAT'
            
            -- Explainability
            model_consensus_breakdown JSONB,
            confluence_breakdown JSONB,
            execution_breakdown JSONB,
            natural_language_reasons JSONB,
            
            -- Metadata
            processing_time_ms INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # SDE Performance Tracking
        """
        CREATE TABLE IF NOT EXISTS sde_performance_metrics (
            id SERIAL PRIMARY KEY,
            metric_date DATE NOT NULL,
            symbol VARCHAR(20),
            timeframe VARCHAR(10),
            
            -- Signal Metrics
            total_signals_generated INTEGER DEFAULT 0,
            consensus_achieved_count INTEGER DEFAULT 0,
            confluence_passed_count INTEGER DEFAULT 0,
            execution_passed_count INTEGER DEFAULT 0,
            final_signals_emitted INTEGER DEFAULT 0,
            
            -- Quality Metrics
            avg_consensus_score DECIMAL(5,4),
            avg_confluence_score DECIMAL(3,1),
            avg_execution_quality DECIMAL(3,1),
            avg_confidence_score DECIMAL(5,4),
            
            -- Performance Metrics
            signal_accuracy DECIMAL(5,4),
            avg_risk_reward DECIMAL(5,2),
            max_drawdown DECIMAL(5,2),
            
            -- Processing Metrics
            avg_processing_time_ms INTEGER,
            cache_hit_rate DECIMAL(5,4),
            
            -- Metadata
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """
    ]
    
    for i, table_sql in enumerate(tables, 1):
        try:
            async with pool.acquire() as conn:
                await conn.execute(table_sql)
                logger.info(f"✅ Created SDE table {i}/{len(tables)}")
        except Exception as e:
            logger.error(f"❌ Failed to create SDE table {i}: {e}")
            raise

async def create_sde_indexes(pool: asyncpg.Pool):
    """Create performance indexes for SDE tables"""
    
    indexes = [
        # Model Consensus Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_consensus_signal_id ON sde_model_consensus(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_consensus_symbol_timeframe ON sde_model_consensus(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_sde_consensus_timestamp ON sde_model_consensus(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_consensus_achieved ON sde_model_consensus(consensus_achieved) WHERE consensus_achieved = true",
        
        # Confluence Score Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_confluence_signal_id ON sde_confluence_scores(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_confluence_score ON sde_confluence_scores(total_confluence_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_confluence_gate ON sde_confluence_scores(confluence_gate_passed) WHERE confluence_gate_passed = true",
        
        # Execution Quality Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_execution_signal_id ON sde_execution_quality(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_execution_quality_score ON sde_execution_quality(execution_quality_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_execution_gates ON sde_execution_quality(all_gates_passed) WHERE all_gates_passed = true",
        
        # Signal History Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_history_signal_id ON sde_signal_history(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_history_symbol_timeframe ON sde_signal_history(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_sde_history_timestamp ON sde_signal_history(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_history_decision ON sde_signal_history(sde_final_decision)",
        
        # Performance Metrics Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_performance_date ON sde_performance_metrics(metric_date DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_performance_symbol ON sde_performance_metrics(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_sde_performance_accuracy ON sde_performance_metrics(signal_accuracy DESC)"
    ]
    
    for i, index_sql in enumerate(indexes, 1):
        try:
            async with pool.acquire() as conn:
                await conn.execute(index_sql)
                logger.info(f"✅ Created SDE index {i}/{len(indexes)}")
        except Exception as e:
            logger.error(f"❌ Failed to create SDE index {i}: {e}")
            raise

async def insert_default_configs(pool: asyncpg.Pool):
    """Insert default SDE configuration"""
    
    configs = [
        {
            'config_name': 'sde_consensus_default',
            'config_type': 'consensus',
            'config_data': {
                'min_agreeing_heads': 3,
                'min_head_probability': 0.70,
                'consensus_threshold': 0.75,
                'head_weights': {
                    'head_a': 0.30,  # CatBoost
                    'head_b': 0.25,  # Logistic
                    'head_c': 0.25,  # OB-tree
                    'head_d': 0.20   # Rule-scoring
                }
            },
            'description': 'Default consensus configuration for SDE framework'
        },
        {
            'config_name': 'sde_confluence_default',
            'config_type': 'confluence',
            'config_data': {
                'min_confluence_score': 8.0,
                'component_weights': {
                    'zone_score': 0.25,
                    'htf_bias_score': 0.20,
                    'trigger_quality_score': 0.20,
                    'fvg_confluence_score': 0.15,
                    'orderbook_confluence_score': 0.10,
                    'sentiment_confluence_score': 0.10
                },
                'score_thresholds': {
                    'zone_score_min': 7.0,
                    'htf_bias_score_min': 6.0,
                    'trigger_quality_score_min': 7.0
                }
            },
            'description': 'Default confluence scoring configuration'
        },
        {
            'config_name': 'sde_execution_default',
            'config_type': 'execution',
            'config_data': {
                'spread_atr_ratio_max': 0.12,
                'atr_percentile_min': 25.0,
                'atr_percentile_max': 75.0,
                'impact_cost_max': 0.15,
                'min_execution_quality': 0.8
            },
            'description': 'Default execution quality configuration'
        },
        {
            'config_name': 'sde_general_default',
            'config_type': 'general',
            'config_data': {
                'max_open_signals_per_symbol': 1,
                'max_open_signals_per_account': 3,
                'min_confidence_threshold': 0.85,
                'min_risk_reward_ratio': 2.0,
                'news_blackout_minutes': 15,
                'funding_rate_impact_threshold': 0.001
            },
            'description': 'General SDE framework configuration'
        }
    ]
    
    for config in configs:
        try:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sde_config (config_name, config_type, config_data, description)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (config_name) DO UPDATE SET
                        config_data = EXCLUDED.config_data,
                        updated_at = NOW()
                """, config['config_name'], config['config_type'], json.dumps(config['config_data']), config['description'])
                logger.info(f"✅ Inserted/updated config: {config['config_name']}")
        except Exception as e:
            logger.error(f"❌ Failed to insert config {config['config_name']}: {e}")
            raise

async def run_migration():
    """Run the SDE framework migration"""
    logger.info("🚀 Starting SDE Framework Phase 1 Migration")
    
    try:
        # Create database connection
        pool = await asyncpg.create_pool(**db_config)
        logger.info("✅ Database connection established")
        
        # Create tables
        await create_sde_tables(pool)
        logger.info("✅ SDE tables created")
        
        # Wait for tables to be fully created
        await asyncio.sleep(3)
        
        # Create indexes
        await create_sde_indexes(pool)
        logger.info("✅ SDE indexes created")
        
        # Insert default configurations
        await insert_default_configs(pool)
        logger.info("✅ Default configurations inserted")
        
        # Close connection
        await pool.close()
        
        logger.info("🎉 SDE Framework Phase 1 Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
