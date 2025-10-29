"""
Phase 4: Advanced Price Action Integration Migration
Integrates sophisticated price action models with signal generator
"""

import asyncio
import logging
import asyncpg
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'port': 5432,
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'database': 'alphapulse'
}

async def create_connection_pool():
    """Create database connection pool"""
    try:
        pool = await asyncpg.create_pool(**db_config)
        logger.info("✅ Database connection pool created successfully")
        return pool
    except Exception as e:
        logger.error(f"❌ Failed to create connection pool: {e}")
        raise

async def create_price_action_tables(pool):
    """Create tables for advanced price action integration"""
    
    tables = [
        # Price Action ML Model Registry
        """
        CREATE TABLE IF NOT EXISTS price_action_ml_models (
            model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL, -- 'support_resistance', 'demand_supply', 'market_structure', 'pattern_ml'
            model_version VARCHAR(20) NOT NULL,
            model_path TEXT NOT NULL,
            feature_set JSONB NOT NULL,
            performance_metrics JSONB NOT NULL,
            training_config JSONB NOT NULL,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_trained_at TIMESTAMP WITH TIME ZONE,
            accuracy_score FLOAT,
            precision_score FLOAT,
            recall_score FLOAT,
            f1_score FLOAT,
            training_samples INTEGER,
            validation_samples INTEGER,
            model_size_mb FLOAT,
            inference_latency_ms FLOAT,
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        # Support & Resistance Levels
        """
        CREATE TABLE IF NOT EXISTS support_resistance_levels (
            level_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            level_type VARCHAR(20) NOT NULL, -- 'support', 'resistance', 'dynamic_support', 'dynamic_resistance'
            price_level DECIMAL(20, 8) NOT NULL,
            strength_score FLOAT NOT NULL, -- 0.0 to 1.0
            confidence_score FLOAT NOT NULL, -- 0.0 to 1.0
            touch_count INTEGER DEFAULT 0,
            last_touched_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_active BOOLEAN DEFAULT true,
            break_count INTEGER DEFAULT 0,
            hold_count INTEGER DEFAULT 0,
            volume_profile JSONB,
            order_flow_data JSONB,
            ml_prediction_confidence FLOAT,
            market_structure_context JSONB,
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        # Market Structure Analysis (HH, HL, LH, LL)
        """
        CREATE TABLE IF NOT EXISTS market_structure_analysis (
            structure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            structure_type VARCHAR(20) NOT NULL, -- 'HH', 'HL', 'LH', 'LL', 'breakout', 'breakdown'
            price_level DECIMAL(20, 8) NOT NULL,
            volume_level DECIMAL(20, 8),
            strength_score FLOAT NOT NULL, -- 0.0 to 1.0
            confirmation_score FLOAT NOT NULL, -- 0.0 to 1.0
            trend_alignment VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
            momentum_score FLOAT,
            volume_confirmation BOOLEAN,
            pattern_confirmation BOOLEAN,
            support_resistance_context JSONB,
            demand_supply_context JSONB,
            ml_enhanced_confidence FLOAT,
            market_regime_context JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        # Demand & Supply Zones
        """
        CREATE TABLE IF NOT EXISTS demand_supply_zones (
            zone_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            zone_type VARCHAR(20) NOT NULL, -- 'demand', 'supply', 'equilibrium'
            upper_bound DECIMAL(20, 8) NOT NULL,
            lower_bound DECIMAL(20, 8) NOT NULL,
            center_price DECIMAL(20, 8) NOT NULL,
            zone_strength VARCHAR(20) NOT NULL, -- 'weak', 'moderate', 'strong', 'very_strong'
            volume_profile JSONB NOT NULL,
            order_flow_data JSONB,
            breakout_probability FLOAT,
            hold_probability FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_active BOOLEAN DEFAULT true,
            touch_count INTEGER DEFAULT 0,
            break_count INTEGER DEFAULT 0,
            ml_zone_confidence FLOAT,
            market_structure_alignment JSONB,
            support_resistance_overlap JSONB,
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        # Price Action ML Predictions
        """
        CREATE TABLE IF NOT EXISTS price_action_ml_predictions (
            prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            model_id UUID REFERENCES price_action_ml_models(model_id),
            prediction_type VARCHAR(50) NOT NULL, -- 'support_resistance', 'breakout', 'reversal', 'continuation'
            prediction_probability FLOAT NOT NULL, -- 0.0 to 1.0
            confidence_score FLOAT NOT NULL, -- 0.0 to 1.0
            target_price DECIMAL(20, 8),
            stop_loss_price DECIMAL(20, 8),
            time_horizon_hours INTEGER,
            feature_vector JSONB NOT NULL,
            model_output JSONB NOT NULL,
            market_context JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        # Price Action Signal Integration
        """
        CREATE TABLE IF NOT EXISTS price_action_signal_integration (
            integration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            signal_id UUID NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            
            -- Price Action Components
            support_resistance_score FLOAT,
            market_structure_score FLOAT,
            demand_supply_score FLOAT,
            pattern_ml_score FLOAT,
            
            -- Integration Weights
            support_resistance_weight FLOAT DEFAULT 0.25,
            market_structure_weight FLOAT DEFAULT 0.25,
            demand_supply_weight FLOAT DEFAULT 0.25,
            pattern_ml_weight FLOAT DEFAULT 0.25,
            
            -- Combined Price Action Score
            combined_price_action_score FLOAT NOT NULL,
            price_action_confidence FLOAT NOT NULL,
            
            -- Enhanced Signal Metrics
            enhanced_confidence_score FLOAT NOT NULL,
            enhanced_risk_reward_ratio FLOAT,
            enhanced_entry_price DECIMAL(20, 8),
            enhanced_stop_loss DECIMAL(20, 8),
            enhanced_take_profit DECIMAL(20, 8),
            
            -- Context Data
            support_resistance_context JSONB,
            market_structure_context JSONB,
            demand_supply_context JSONB,
            pattern_ml_context JSONB,
            
            -- Performance Tracking
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        # Price Action Performance Tracking
        """
        CREATE TABLE IF NOT EXISTS price_action_performance (
            performance_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            
            -- Performance Metrics
            accuracy_score FLOAT,
            precision_score FLOAT,
            recall_score FLOAT,
            f1_score FLOAT,
            profit_factor FLOAT,
            win_rate FLOAT,
            avg_win FLOAT,
            avg_loss FLOAT,
            max_drawdown FLOAT,
            sharpe_ratio FLOAT,
            
            -- Signal Quality
            signal_count INTEGER,
            successful_signals INTEGER,
            failed_signals INTEGER,
            avg_confidence FLOAT,
            avg_risk_reward FLOAT,
            
            -- Model Performance
            inference_latency_ms FLOAT,
            feature_importance JSONB,
            model_confidence FLOAT,
            
            -- Market Context
            market_regime VARCHAR(50),
            volatility_level VARCHAR(20),
            trend_strength FLOAT,
            
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        # Price Action Configuration
        """
        CREATE TABLE IF NOT EXISTS price_action_config (
            config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            config_name VARCHAR(100) NOT NULL UNIQUE,
            config_type VARCHAR(50) NOT NULL, -- 'support_resistance', 'market_structure', 'demand_supply', 'integration'
            config_data JSONB NOT NULL,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            description TEXT,
            version VARCHAR(20) DEFAULT '1.0.0'
        )
        """,
        
        # Price Action Alerts
        """
        CREATE TABLE IF NOT EXISTS price_action_alerts (
            alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            symbol VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            alert_type VARCHAR(50) NOT NULL, -- 'support_break', 'resistance_break', 'HH_formation', 'LL_formation', 'zone_breakout'
            price_level DECIMAL(20, 8) NOT NULL,
            alert_strength FLOAT NOT NULL, -- 0.0 to 1.0
            alert_message TEXT NOT NULL,
            is_triggered BOOLEAN DEFAULT false,
            triggered_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            expires_at TIMESTAMP WITH TIME ZONE,
            metadata JSONB DEFAULT '{}'::jsonb
        )
        """
    ]
    
    for i, table_sql in enumerate(tables, 1):
        try:
            async with pool.acquire() as conn:
                await conn.execute(table_sql)
                logger.info(f"✅ Created table {i}/{len(tables)}")
        except Exception as e:
            logger.error(f"❌ Failed to create table {i}: {e}")
            raise

async def create_performance_indexes(pool):
    """Create performance indexes for price action tables"""
    
    indexes = [
        # Support Resistance Indexes
        "CREATE INDEX IF NOT EXISTS idx_support_resistance_symbol_timeframe ON support_resistance_levels(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_support_resistance_price_level ON support_resistance_levels(price_level)",
        "CREATE INDEX IF NOT EXISTS idx_support_resistance_strength ON support_resistance_levels(strength_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_support_resistance_active ON support_resistance_levels(is_active) WHERE is_active = true",
        
        # Market Structure Indexes
        "CREATE INDEX IF NOT EXISTS idx_market_structure_symbol_timeframe ON market_structure_analysis(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_market_structure_timestamp ON market_structure_analysis(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_market_structure_type ON market_structure_analysis(market_structure_type)",
        "CREATE INDEX IF NOT EXISTS idx_market_structure_strength ON market_structure_analysis(structure_strength DESC)",
        
        # Demand Supply Indexes
        "CREATE INDEX IF NOT EXISTS idx_demand_supply_symbol_timeframe ON demand_supply_zones(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_demand_supply_bounds ON demand_supply_zones(zone_start_price, zone_end_price)",
        "CREATE INDEX IF NOT EXISTS idx_demand_supply_strength ON demand_supply_zones(zone_strength)",
        "CREATE INDEX IF NOT EXISTS idx_demand_supply_active ON demand_supply_zones(zone_metadata) WHERE zone_metadata->>'is_active' = 'true'",
        
        # ML Predictions Indexes
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_timeframe ON price_action_ml_predictions(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON price_action_ml_predictions(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_confidence ON price_action_ml_predictions(confidence_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON price_action_ml_predictions(model_id)",
        
        # Signal Integration Indexes
        "CREATE INDEX IF NOT EXISTS idx_signal_integration_signal_id ON price_action_signal_integration(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_signal_integration_symbol_timeframe ON price_action_signal_integration(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_signal_integration_confidence ON price_action_signal_integration(enhanced_confidence_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_signal_integration_timestamp ON price_action_signal_integration(timestamp DESC)",
        
        # Performance Indexes
        "CREATE INDEX IF NOT EXISTS idx_performance_symbol_timeframe ON price_action_performance(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON price_action_performance(timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_performance_model_type ON price_action_performance(model_type)",
        
        # Alerts Indexes
        "CREATE INDEX IF NOT EXISTS idx_alerts_symbol_timeframe ON price_action_alerts(symbol, timeframe)",
        "CREATE INDEX IF NOT EXISTS idx_alerts_triggered ON price_action_alerts(is_triggered)",
        "CREATE INDEX IF NOT EXISTS idx_alerts_expires_at ON price_action_alerts(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_alerts_strength ON price_action_alerts(alert_strength DESC)"
    ]
    
    for i, index_sql in enumerate(indexes, 1):
        try:
            async with pool.acquire() as conn:
                await conn.execute(index_sql)
                logger.info(f"✅ Created index {i}/{len(indexes)}")
        except Exception as e:
            logger.error(f"❌ Failed to create index {i}: {e}")
            raise

async def insert_default_configurations(pool):
    """Insert default configurations for price action integration"""
    
    configs = [
        {
            "config_name": "support_resistance_default",
            "config_type": "support_resistance",
            "config_data": {
                "min_strength_threshold": 0.6,
                "min_confidence_threshold": 0.7,
                "max_levels_per_symbol": 10,
                "touch_validation_periods": True,
                "break_validation_periods": 2,
                "volume_confirmation_required": True,
                "ml_enhancement_enabled": True,
                "dynamic_level_adjustment": True
            },
            "description": "Default support and resistance configuration"
        },
        {
            "config_name": "market_structure_default",
            "config_type": "market_structure",
            "config_data": {
                "min_strength_threshold": 0.65,
                "min_confirmation_score": 0.7,
                "volume_confirmation_required": True,
                "pattern_confirmation_required": True,
                "trend_alignment_weight": 0.3,
                "momentum_weight": 0.2,
                "volume_weight": 0.25,
                "pattern_weight": 0.25,
                "ml_enhancement_enabled": True
            },
            "description": "Default market structure analysis configuration"
        },
        {
            "config_name": "demand_supply_default",
            "config_type": "demand_supply",
            "config_data": {
                "min_zone_strength": "moderate",
                "volume_profile_required": True,
                "order_flow_analysis_enabled": True,
                "breakout_probability_threshold": 0.6,
                "hold_probability_threshold": 0.7,
                "ml_zone_confidence_threshold": 0.65,
                "market_structure_alignment_required": True
            },
            "description": "Default demand and supply zone configuration"
        },
        {
            "config_name": "price_action_integration_default",
            "config_type": "integration",
            "config_data": {
                "support_resistance_weight": 0.25,
                "market_structure_weight": 0.25,
                "demand_supply_weight": 0.25,
                "pattern_ml_weight": 0.25,
                "min_combined_score": 0.75,
                "min_confidence_threshold": 0.8,
                "enhancement_factor": 1.2,
                "risk_reward_improvement": 0.1,
                "ml_prediction_required": True,
                "market_context_required": True
            },
            "description": "Default price action integration configuration"
        }
    ]
    
    for config in configs:
        try:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO price_action_config (config_name, config_type, config_data, description)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (config_name) DO NOTHING
                """, config["config_name"], config["config_type"], 
                     json.dumps(config["config_data"]), config["description"])
                logger.info(f"✅ Inserted config: {config['config_name']}")
        except Exception as e:
            logger.error(f"❌ Failed to insert config {config['config_name']}: {e}")

async def run_migration():
    """Run the complete Phase 4 migration"""
    logger.info("🚀 Starting Phase 4: Advanced Price Action Integration Migration")
    
    try:
        # Create connection pool
        pool = await create_connection_pool()
        
        # Create tables
        logger.info("📋 Creating price action tables...")
        await create_price_action_tables(pool)
        
        # Wait for tables to be fully created
        await asyncio.sleep(3)
        
        # Create indexes
        logger.info("🔍 Creating performance indexes...")
        await create_performance_indexes(pool)
        
        # Insert default configurations
        logger.info("⚙️ Inserting default configurations...")
        await insert_default_configurations(pool)
        
        logger.info("✅ Phase 4 migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        if 'pool' in locals():
            await pool.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
