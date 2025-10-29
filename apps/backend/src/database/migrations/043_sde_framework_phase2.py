"""
Phase 2: SDE Framework Implementation - Execution Quality Enhancement
Database migration for Single-Decision Engine framework Phase 2
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

async def create_phase2_tables(pool: asyncpg.Pool):
    """Create Phase 2 SDE framework tables"""
    
    tables = [
        # News/Funding Blackout Tracking
        """
        CREATE TABLE IF NOT EXISTS sde_news_blackout (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            event_type VARCHAR(50) NOT NULL, -- 'news', 'funding', 'earnings', 'economic'
            event_impact VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
            event_title TEXT NOT NULL,
            event_description TEXT,
            start_time TIMESTAMP WITH TIME ZONE NOT NULL,
            end_time TIMESTAMP WITH TIME ZONE NOT NULL,
            blackout_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Signal Limits and Quota Management
        """
        CREATE TABLE IF NOT EXISTS sde_signal_limits (
            id SERIAL PRIMARY KEY,
            account_id VARCHAR(50) NOT NULL,
            symbol VARCHAR(20),
            timeframe VARCHAR(10),
            limit_type VARCHAR(30) NOT NULL, -- 'per_symbol', 'per_account', 'per_timeframe'
            max_signals INTEGER NOT NULL,
            current_signals INTEGER DEFAULT 0,
            reset_period VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'weekly'
            last_reset TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Four TP Structure Management
        """
        CREATE TABLE IF NOT EXISTS sde_tp_structure (
            id SERIAL PRIMARY KEY,
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Entry and Stop
            entry_price DECIMAL(15,8) NOT NULL,
            stop_loss DECIMAL(15,8) NOT NULL,
            risk_distance DECIMAL(15,8) NOT NULL, -- distance from entry to stop
            
            -- Take Profit Levels
            tp1_price DECIMAL(15,8),
            tp1_distance DECIMAL(15,8), -- in ATR or fixed distance
            tp1_percentage DECIMAL(5,2), -- % of position to close
            
            tp2_price DECIMAL(15,8),
            tp2_distance DECIMAL(15,8),
            tp2_percentage DECIMAL(5,2),
            
            tp3_price DECIMAL(15,8),
            tp3_distance DECIMAL(15,8),
            tp3_percentage DECIMAL(5,2),
            
            tp4_price DECIMAL(15,8),
            tp4_distance DECIMAL(15,8),
            tp4_percentage DECIMAL(5,2),
            
            -- Execution Status
            tp1_hit BOOLEAN DEFAULT FALSE,
            tp2_hit BOOLEAN DEFAULT FALSE,
            tp3_hit BOOLEAN DEFAULT FALSE,
            tp4_hit BOOLEAN DEFAULT FALSE,
            stop_hit BOOLEAN DEFAULT FALSE,
            
            -- Partial Exit Tracking
            total_position_size DECIMAL(10,4) NOT NULL,
            remaining_position_size DECIMAL(10,4) NOT NULL,
            total_exited_size DECIMAL(10,4) DEFAULT 0.0,
            
            -- Stop Movement
            stop_moved_to_breakeven BOOLEAN DEFAULT FALSE,
            breakeven_stop_price DECIMAL(15,8),
            trailing_stop_active BOOLEAN DEFAULT FALSE,
            trailing_stop_price DECIMAL(15,8),
            
            -- Metadata
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Enhanced Execution Quality Metrics
        """
        CREATE TABLE IF NOT EXISTS sde_enhanced_execution (
            id SERIAL PRIMARY KEY,
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Spread Analysis
            bid_price DECIMAL(15,8),
            ask_price DECIMAL(15,8),
            spread_absolute DECIMAL(15,8),
            spread_percentage DECIMAL(10,6),
            atr_value DECIMAL(15,8),
            spread_atr_ratio DECIMAL(10,6),
            spread_gate_passed BOOLEAN DEFAULT FALSE,
            
            -- Volatility Analysis
            atr_percentile DECIMAL(5,2),
            volatility_regime VARCHAR(20), -- 'low', 'normal', 'high', 'extreme'
            volatility_score DECIMAL(3,1), -- 0-10
            volatility_gate_passed BOOLEAN DEFAULT FALSE,
            
            -- Impact Analysis
            orderbook_depth DECIMAL(15,2),
            estimated_impact DECIMAL(10,6),
            impact_cost DECIMAL(10,6),
            impact_gate_passed BOOLEAN DEFAULT FALSE,
            
            -- Market Microstructure
            bid_ask_imbalance DECIMAL(5,4), -- -1 to +1
            order_flow_direction VARCHAR(10), -- 'buying', 'selling', 'neutral'
            liquidity_score DECIMAL(3,1), -- 0-10
            market_impact_score DECIMAL(3,1), -- 0-10
            
            -- Overall Quality
            execution_quality_score DECIMAL(3,1), -- 0-10
            all_gates_passed BOOLEAN DEFAULT FALSE,
            quality_breakdown JSONB,
            
            -- Metadata
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Signal Queue Management
        """
        CREATE TABLE IF NOT EXISTS sde_signal_queue (
            id SERIAL PRIMARY KEY,
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            -- Queue Information
            queue_position INTEGER NOT NULL,
            priority_score DECIMAL(5,4) NOT NULL, -- 0-1, higher is better
            queue_status VARCHAR(20) NOT NULL, -- 'pending', 'processing', 'emitted', 'rejected'
            
            -- Signal Quality Metrics
            confidence_score DECIMAL(5,4),
            confluence_score DECIMAL(3,1),
            execution_quality_score DECIMAL(3,1),
            risk_reward_ratio DECIMAL(5,2),
            
            -- Queue Management
            max_queue_size INTEGER DEFAULT 10,
            queue_full BOOLEAN DEFAULT FALSE,
            replacement_candidate BOOLEAN DEFAULT FALSE,
            replaced_signal_id UUID,
            
            -- Timing
            queue_entry_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            processing_start_time TIMESTAMP WITH TIME ZONE,
            emission_time TIMESTAMP WITH TIME ZONE,
            
            -- Metadata
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        # Performance Tracking Enhancement
        """
        CREATE TABLE IF NOT EXISTS sde_enhanced_performance (
            id SERIAL PRIMARY KEY,
            metric_date DATE NOT NULL,
            symbol VARCHAR(20),
            timeframe VARCHAR(10),
            
            -- Signal Generation Metrics
            total_signals_generated INTEGER DEFAULT 0,
            consensus_achieved_count INTEGER DEFAULT 0,
            confluence_passed_count INTEGER DEFAULT 0,
            execution_passed_count INTEGER DEFAULT 0,
            final_signals_emitted INTEGER DEFAULT 0,
            signals_rejected INTEGER DEFAULT 0,
            
            -- Quality Metrics
            avg_consensus_score DECIMAL(5,4),
            avg_confluence_score DECIMAL(3,1),
            avg_execution_quality DECIMAL(3,1),
            avg_confidence_score DECIMAL(5,4),
            avg_risk_reward DECIMAL(5,2),
            
            -- Queue Metrics
            avg_queue_time_ms INTEGER,
            queue_efficiency DECIMAL(5,4), -- signals emitted / signals queued
            queue_replacement_rate DECIMAL(5,4), -- signals replaced / total signals
            
            -- Execution Metrics
            avg_spread_atr_ratio DECIMAL(10,6),
            avg_impact_cost DECIMAL(10,6),
            avg_volatility_score DECIMAL(3,1),
            
            -- Performance Metrics
            signal_accuracy DECIMAL(5,4),
            avg_profit_loss DECIMAL(10,4),
            max_drawdown DECIMAL(5,2),
            sharpe_ratio DECIMAL(5,4),
            
            -- Processing Metrics
            avg_processing_time_ms INTEGER,
            cache_hit_rate DECIMAL(5,4),
            parallel_processing_efficiency DECIMAL(5,4),
            
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
                logger.info(f"✅ Created Phase 2 table {i}/{len(tables)}")
        except Exception as e:
            logger.error(f"❌ Failed to create Phase 2 table {i}: {e}")
            raise

async def create_phase2_indexes(pool: asyncpg.Pool):
    """Create performance indexes for Phase 2 tables"""
    
    indexes = [
        # News Blackout Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_news_symbol ON sde_news_blackout(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_sde_news_time ON sde_news_blackout(start_time, end_time)",
        "CREATE INDEX IF NOT EXISTS idx_sde_news_active ON sde_news_blackout(blackout_active) WHERE blackout_active = true",
        "CREATE INDEX IF NOT EXISTS idx_sde_news_impact ON sde_news_blackout(event_impact)",
        
        # Signal Limits Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_limits_account ON sde_signal_limits(account_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_limits_symbol ON sde_signal_limits(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_sde_limits_type ON sde_signal_limits(limit_type)",
        "CREATE INDEX IF NOT EXISTS idx_sde_limits_reset ON sde_signal_limits(last_reset)",
        
        # TP Structure Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_tp_signal_id ON sde_tp_structure(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_tp_symbol_timeframe ON sde_tp_structure(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_sde_tp_status ON sde_tp_structure(tp1_hit, tp2_hit, tp3_hit, tp4_hit, stop_hit)",
        "CREATE INDEX IF NOT EXISTS idx_sde_tp_trailing ON sde_tp_structure(trailing_stop_active) WHERE trailing_stop_active = true",
        
        # Enhanced Execution Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_signal_id ON sde_enhanced_execution(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_quality ON sde_enhanced_execution(execution_quality_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_gates ON sde_enhanced_execution(all_gates_passed) WHERE all_gates_passed = true",
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_spread ON sde_enhanced_execution(spread_atr_ratio)",
        
        # Signal Queue Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_queue_signal_id ON sde_signal_queue(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_sde_queue_priority ON sde_signal_queue(priority_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_queue_status ON sde_signal_queue(queue_status)",
        "CREATE INDEX IF NOT EXISTS idx_sde_queue_position ON sde_signal_queue(queue_position)",
        
        # Enhanced Performance Indexes
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_perf_date ON sde_enhanced_performance(metric_date DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_perf_symbol ON sde_enhanced_performance(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_perf_accuracy ON sde_enhanced_performance(signal_accuracy DESC)",
        "CREATE INDEX IF NOT EXISTS idx_sde_enhanced_perf_sharpe ON sde_enhanced_performance(sharpe_ratio DESC)"
    ]
    
    for i, index_sql in enumerate(indexes, 1):
        try:
            async with pool.acquire() as conn:
                await conn.execute(index_sql)
                logger.info(f"✅ Created Phase 2 index {i}/{len(indexes)}")
        except Exception as e:
            logger.error(f"❌ Failed to create Phase 2 index {i}: {e}")
            raise

async def insert_phase2_configs(pool: asyncpg.Pool):
    """Insert Phase 2 SDE configuration"""
    
    configs = [
        {
            'config_name': 'sde_news_blackout_default',
            'config_type': 'news_blackout',
            'config_data': {
                'blackout_minutes_before': 15,
                'blackout_minutes_after': 15,
                'impact_thresholds': {
                    'low': 0.0,
                    'medium': 0.3,
                    'high': 0.6,
                    'critical': 0.8
                },
                'event_types': ['news', 'funding', 'earnings', 'economic', 'regulatory'],
                'symbol_specific_blackouts': True,
                'global_blackouts': True
            },
            'description': 'News and funding rate blackout configuration'
        },
        {
            'config_name': 'sde_signal_limits_default',
            'config_type': 'signal_limits',
            'config_data': {
                'max_open_signals_per_symbol': 1,
                'max_open_signals_per_account': 3,
                'max_signals_per_hour': 4,
                'max_signals_per_day': 10,
                'priority_scoring': {
                    'confidence_weight': 0.4,
                    'confluence_weight': 0.3,
                    'execution_quality_weight': 0.2,
                    'risk_reward_weight': 0.1
                },
                'queue_management': {
                    'max_queue_size': 10,
                    'replacement_threshold': 0.1,  # 10% improvement required
                    'queue_timeout_minutes': 30
                }
            },
            'description': 'Signal limits and quota management configuration'
        },
        {
            'config_name': 'sde_tp_structure_default',
            'config_type': 'tp_structure',
            'config_data': {
                'tp_levels': {
                    'tp1': {'distance': 0.5, 'percentage': 25},  # 0.5R, 25% of position
                    'tp2': {'distance': 1.0, 'percentage': 25},  # 1.0R, 25% of position
                    'tp3': {'distance': 2.0, 'percentage': 25},  # 2.0R, 25% of position
                    'tp4': {'distance': 4.0, 'percentage': 25}   # 4.0R, 25% of position
                },
                'stop_management': {
                    'breakeven_trigger': 'tp2_hit',  # Move to BE after TP2
                    'breakeven_buffer': 0.1,  # 10% buffer above entry
                    'trailing_start': 'tp3_hit',  # Start trailing after TP3
                    'trailing_distance': 1.0  # 1.0 ATR trailing distance
                },
                'position_sizing': {
                    'base_position_size': 0.02,  # 2% of account
                    'max_position_size': 0.05,   # 5% of account
                    'risk_per_trade': 0.01       # 1% risk per trade
                }
            },
            'description': 'Four TP structure and position management configuration'
        },
        {
            'config_name': 'sde_enhanced_execution_default',
            'config_type': 'enhanced_execution',
            'config_data': {
                'spread_analysis': {
                    'max_spread_atr_ratio': 0.12,
                    'max_spread_percentage': 0.05,
                    'min_atr_value': 0.001
                },
                'volatility_analysis': {
                    'atr_percentile_min': 25.0,
                    'atr_percentile_max': 75.0,
                    'volatility_score_weights': {
                        'atr_percentile': 0.4,
                        'regime_alignment': 0.3,
                        'stability': 0.3
                    }
                },
                'impact_analysis': {
                    'max_impact_cost': 0.15,
                    'min_orderbook_depth': 1000,
                    'impact_score_weights': {
                        'depth': 0.4,
                        'imbalance': 0.3,
                        'flow': 0.3
                    }
                },
                'quality_thresholds': {
                    'min_execution_quality': 8.0,
                    'min_liquidity_score': 7.0,
                    'min_market_impact_score': 6.0
                }
            },
            'description': 'Enhanced execution quality assessment configuration'
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
                logger.info(f"✅ Inserted/updated Phase 2 config: {config['config_name']}")
        except Exception as e:
            logger.error(f"❌ Failed to insert Phase 2 config {config['config_name']}: {e}")
            raise

async def run_migration():
    """Run the SDE framework Phase 2 migration"""
    logger.info("🚀 Starting SDE Framework Phase 2 Migration")
    
    try:
        # Create database connection
        pool = await asyncpg.create_pool(**db_config)
        logger.info("✅ Database connection established")
        
        # Create Phase 2 tables
        await create_phase2_tables(pool)
        logger.info("✅ Phase 2 tables created")
        
        # Wait for tables to be fully created
        await asyncio.sleep(3)
        
        # Create Phase 2 indexes
        await create_phase2_indexes(pool)
        logger.info("✅ Phase 2 indexes created")
        
        # Insert Phase 2 configurations
        await insert_phase2_configs(pool)
        logger.info("✅ Phase 2 configurations inserted")
        
        # Close connection
        await pool.close()
        
        logger.info("🎉 SDE Framework Phase 2 Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
