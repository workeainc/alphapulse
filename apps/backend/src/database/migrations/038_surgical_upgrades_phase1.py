#!/usr/bin/env python3
"""
Surgical Upgrades Phase 1: Interface Standardization & Hard Gating System
Database Migration for Enhanced Signal Quality and Risk Management

This migration implements:
1. Interface standardization tables for consistent component interfaces
2. Confidence calibration system for reliability correction
3. Hard gating system for signal validation and quota management
4. Enhanced signal payload structure for complete transparency
5. Real-time news override system for market risk management
"""

import asyncio
import asyncpg
import logging
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgicalUpgradesPhase1Migration:
    """Phase 1 migration for surgical upgrades"""
    
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        self.connection = None
        
    async def connect(self):
        """Connect to database"""
        try:
            self.connection = await asyncpg.connect(**self.connection_params)
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    async def run_migration(self):
        """Run the complete Phase 1 migration"""
        try:
            if not await self.connect():
                return False
            
            logger.info("🚀 Starting Surgical Upgrades Phase 1 Migration...")
            
            # Step 1: Interface Standardization Tables
            await self._create_interface_standardization_tables()
            
            # Step 2: Confidence Calibration System
            await self._create_confidence_calibration_tables()
            
            # Step 3: Hard Gating System
            await self._create_hard_gating_tables()
            
            # Step 4: Enhanced Signal Payload Structure
            await self._create_enhanced_signal_tables()
            
            # Step 5: Real-time News Override System
            await self._create_news_override_tables()
            
            # Step 6: Quota Management System
            await self._create_quota_management_tables()
            
            # Step 7: Mutex and Cooldown System
            await self._create_mutex_cooldown_tables()
            
            # Step 8: Data Sanity Validation Tables
            await self._create_data_sanity_tables()
            
            # Step 9: Create Indexes for Performance
            await self._create_performance_indexes()
            
            # Step 10: Insert Default Configuration
            await self._insert_default_configuration()
            
            logger.info("✅ Surgical Upgrades Phase 1 Migration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
        finally:
            if self.connection:
                await self.connection.close()
    
    async def _create_interface_standardization_tables(self):
        """Create tables for interface standardization"""
        logger.info("📋 Creating interface standardization tables...")
        
        # Component Interface Registry
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS component_interface_registry (
                id SERIAL PRIMARY KEY,
                component_name VARCHAR(100) NOT NULL,
                interface_type VARCHAR(50) NOT NULL, -- 'onnx', 'drift', 'pattern', 'volume', 'sentiment'
                interface_version VARCHAR(20) NOT NULL,
                method_name VARCHAR(100) NOT NULL,
                input_signature JSONB NOT NULL,
                output_signature JSONB NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(component_name, interface_type, method_name)
            )
        """)
        
        # Interface Performance Metrics
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS interface_performance_metrics (
                id SERIAL PRIMARY KEY,
                component_name VARCHAR(100) NOT NULL,
                interface_type VARCHAR(50) NOT NULL,
                method_name VARCHAR(100) NOT NULL,
                execution_time_ms FLOAT NOT NULL,
                success_rate FLOAT NOT NULL,
                error_count INTEGER DEFAULT 0,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB
            )
        """)
        
        # Standardized Interface Results
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS standardized_interface_results (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                component_name VARCHAR(100) NOT NULL,
                interface_type VARCHAR(50) NOT NULL,
                input_data JSONB NOT NULL,
                output_data JSONB NOT NULL,
                confidence_score FLOAT NOT NULL,
                processing_time_ms FLOAT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_component 
            ON standardized_interface_results (signal_id, component_name);
        """)
        
        logger.info("✅ Interface standardization tables created")
    
    async def _create_confidence_calibration_tables(self):
        """Create tables for confidence calibration system"""
        logger.info("📋 Creating confidence calibration tables...")
        
        # Calibration Models
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS calibration_models (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                calibration_type VARCHAR(50) NOT NULL, -- 'isotonic', 'platt', 'temperature'
                model_data BYTEA NOT NULL,
                calibration_metrics JSONB NOT NULL,
                brier_score FLOAT NOT NULL,
                reliability_score FLOAT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(model_name, calibration_type)
            )
        """)
        
        # Calibration History
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS calibration_history (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                calibration_type VARCHAR(50) NOT NULL,
                raw_confidence FLOAT NOT NULL,
                calibrated_confidence FLOAT NOT NULL,
                actual_outcome BOOLEAN,
                reliability_bucket VARCHAR(20), -- '0.8-0.85', '0.85-0.9', etc.
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_calibration 
            ON calibration_history (model_name, calibration_type);
        """)
        
        # Reliability Buckets
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS reliability_buckets (
                id SERIAL PRIMARY KEY,
                bucket_range VARCHAR(20) NOT NULL, -- '0.8-0.85', '0.85-0.9', etc.
                expected_win_rate FLOAT NOT NULL,
                actual_win_rate FLOAT NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(bucket_range)
            )
        """)
        
        logger.info("✅ Confidence calibration tables created")
    
    async def _create_hard_gating_tables(self):
        """Create tables for hard gating system"""
        logger.info("📋 Creating hard gating tables...")
        
        # Signal Gates Configuration
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_gates_config (
                id SERIAL PRIMARY KEY,
                gate_name VARCHAR(100) NOT NULL,
                gate_type VARCHAR(50) NOT NULL, -- 'data_sanity', 'quota', 'mutex', 'news', 'spread'
                is_enabled BOOLEAN DEFAULT TRUE,
                threshold_value FLOAT NOT NULL,
                threshold_type VARCHAR(20) NOT NULL, -- 'absolute', 'percentage', 'ratio'
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(gate_name)
            )
        """)
        
        # Gate Validation Results
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS gate_validation_results (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                gate_name VARCHAR(100) NOT NULL,
                gate_type VARCHAR(50) NOT NULL,
                passed BOOLEAN NOT NULL,
                actual_value FLOAT NOT NULL,
                threshold_value FLOAT NOT NULL,
                validation_time_ms FLOAT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_gate 
            ON gate_validation_results (signal_id, gate_name);
        """)
        
        # Gate Performance Metrics
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS gate_performance_metrics (
                id SERIAL PRIMARY KEY,
                gate_name VARCHAR(100) NOT NULL,
                total_validations INTEGER DEFAULT 0,
                passed_validations INTEGER DEFAULT 0,
                failed_validations INTEGER DEFAULT 0,
                avg_validation_time_ms FLOAT DEFAULT 0,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(gate_name)
            )
        """)
        
        logger.info("✅ Hard gating tables created")
    
    async def _create_enhanced_signal_tables(self):
        """Create tables for enhanced signal payload structure"""
        logger.info("📋 Creating enhanced signal tables...")
        
        # Enhanced Signal Payloads
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_signal_payloads (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                confidence_breakdown JSONB NOT NULL, -- Detailed confidence scores
                mtf_agreement JSONB NOT NULL, -- Multi-timeframe confirmation details
                orderbook_analysis JSONB NOT NULL, -- Liquidity walls, imbalance
                news_impact JSONB NOT NULL, -- Real-time news sentiment
                data_health_metrics JSONB NOT NULL, -- Data quality indicators
                reasoning_chain JSONB NOT NULL, -- Step-by-step reasoning
                lifecycle_info JSONB NOT NULL, -- Expiry, cooldown info
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(signal_id)
            )
        """)
        
        # Signal Reasoning Chain
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_reasoning_chain (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                reasoning_step INTEGER NOT NULL,
                component_name VARCHAR(100) NOT NULL,
                input_data JSONB NOT NULL,
                output_data JSONB NOT NULL,
                confidence_contribution FLOAT NOT NULL,
                reasoning_text TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_reasoning 
            ON signal_reasoning_chain (signal_id, reasoning_step);
        """)
        
        # MTF Agreement Details
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS mtf_agreement_details (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                timeframe VARCHAR(10) NOT NULL, -- '15m', '1h', '4h', '1d'
                agreement_score FLOAT NOT NULL,
                trend_direction VARCHAR(20) NOT NULL, -- 'bullish', 'bearish', 'neutral'
                confirmation_strength FLOAT NOT NULL,
                weight_factor FLOAT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_timeframe 
            ON mtf_agreement_details (signal_id, timeframe);
        """)
        
        logger.info("✅ Enhanced signal tables created")
    
    async def _create_news_override_tables(self):
        """Create tables for real-time news override system"""
        logger.info("📋 Creating news override tables...")
        
        # News Override Rules
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS news_override_rules (
                id SERIAL PRIMARY KEY,
                rule_name VARCHAR(100) NOT NULL,
                sentiment_threshold FLOAT NOT NULL, -- -1.0 to +1.0
                impact_threshold FLOAT NOT NULL, -- 0.0 to 1.0
                action_type VARCHAR(20) NOT NULL, -- 'block_long', 'block_short', 'allow_early'
                symbol_pattern VARCHAR(100), -- Pattern for matching symbols
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(rule_name)
            )
        """)
        
        # News Override Events
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS news_override_events (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                rule_name VARCHAR(100) NOT NULL,
                news_id INTEGER,
                sentiment_score FLOAT NOT NULL,
                impact_score FLOAT NOT NULL,
                action_taken VARCHAR(20) NOT NULL,
                override_reason TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_override 
            ON news_override_events (signal_id, rule_name);
        """)
        
        # Real-time News Feed
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS real_time_news_feed (
                id SERIAL PRIMARY KEY,
                news_id INTEGER NOT NULL,
                symbol VARCHAR(20),
                headline TEXT NOT NULL,
                sentiment_score FLOAT NOT NULL,
                impact_score FLOAT NOT NULL,
                source VARCHAR(100) NOT NULL,
                published_at TIMESTAMP WITH TIME ZONE NOT NULL,
                processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_processed BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_sentiment 
            ON real_time_news_feed (symbol, sentiment_score);
        """)
        
        logger.info("✅ News override tables created")
    
    async def _create_quota_management_tables(self):
        """Create tables for quota management system"""
        logger.info("📋 Creating quota management tables...")
        
        # Quota Configuration
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS quota_configuration (
                id SERIAL PRIMARY KEY,
                quota_type VARCHAR(50) NOT NULL, -- 'daily', 'hourly', 'symbol', 'system'
                quota_limit INTEGER NOT NULL,
                quota_window_hours INTEGER NOT NULL,
                priority_threshold FLOAT NOT NULL, -- Minimum confidence for quota consideration
                replacement_strategy VARCHAR(20) NOT NULL, -- 'drop_worst', 'fifo', 'priority'
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(quota_type)
            )
        """)
        
        # Quota Usage Tracking
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS quota_usage_tracking (
                id SERIAL PRIMARY KEY,
                quota_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20),
                signal_id VARCHAR(100) NOT NULL,
                confidence_score FLOAT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                window_start TIMESTAMP WITH TIME ZONE NOT NULL,
                window_end TIMESTAMP WITH TIME ZONE NOT NULL
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_quota_usage 
            ON quota_usage_tracking (quota_type, window_start, window_end);
        """)
        
        # Quota Replacement Events
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS quota_replacement_events (
                id SERIAL PRIMARY KEY,
                quota_type VARCHAR(50) NOT NULL,
                replaced_signal_id VARCHAR(100) NOT NULL,
                replacing_signal_id VARCHAR(100) NOT NULL,
                replaced_confidence FLOAT NOT NULL,
                replacing_confidence FLOAT NOT NULL,
                replacement_reason TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_replacement 
            ON quota_replacement_events (replaced_signal_id, replacing_signal_id);
        """)
        
        logger.info("✅ Quota management tables created")
    
    async def _create_mutex_cooldown_tables(self):
        """Create tables for mutex and cooldown system"""
        logger.info("📋 Creating mutex and cooldown tables...")
        
        # Active Signal Mutex
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS active_signal_mutex (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                signal_id VARCHAR(100) NOT NULL,
                signal_direction VARCHAR(10) NOT NULL, -- 'long', 'short'
                entry_price FLOAT NOT NULL,
                stop_loss FLOAT NOT NULL,
                take_profit FLOAT NOT NULL,
                confidence_score FLOAT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                status VARCHAR(20) DEFAULT 'active', -- 'active', 'closed', 'expired'
                UNIQUE(symbol)
            )
        """)
        
        # Cooldown Tracking
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS cooldown_tracking (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                last_signal_id VARCHAR(100) NOT NULL,
                last_signal_direction VARCHAR(10) NOT NULL,
                cooldown_duration_minutes INTEGER NOT NULL,
                cooldown_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                cooldown_end TIMESTAMP WITH TIME ZONE NOT NULL,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe 
            ON cooldown_tracking (symbol, timeframe);
        """)
        
        # Signal Lifecycle Events
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_lifecycle_events (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                event_type VARCHAR(20) NOT NULL, -- 'created', 'activated', 'closed', 'expired', 'cancelled'
                event_reason TEXT,
                event_data JSONB,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_lifecycle 
            ON signal_lifecycle_events (signal_id, event_type);
        """)
        
        logger.info("✅ Mutex and cooldown tables created")
    
    async def _create_data_sanity_tables(self):
        """Create tables for data sanity validation"""
        logger.info("📋 Creating data sanity tables...")
        
        # Data Sanity Rules
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS data_sanity_rules (
                id SERIAL PRIMARY KEY,
                rule_name VARCHAR(100) NOT NULL,
                data_type VARCHAR(50) NOT NULL, -- 'orderbook', 'candlestick', 'volume', 'sentiment'
                validation_type VARCHAR(50) NOT NULL, -- 'staleness', 'spread', 'volume', 'price'
                threshold_value FLOAT NOT NULL,
                threshold_operator VARCHAR(10) NOT NULL, -- 'gt', 'lt', 'eq', 'gte', 'lte'
                is_critical BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(rule_name)
            )
        """)
        
        # Data Sanity Validation Results
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS data_sanity_validation_results (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                rule_name VARCHAR(100) NOT NULL,
                data_type VARCHAR(50) NOT NULL,
                actual_value FLOAT NOT NULL,
                threshold_value FLOAT NOT NULL,
                passed BOOLEAN NOT NULL,
                validation_time_ms FLOAT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_sanity 
            ON data_sanity_validation_results (signal_id, rule_name);
        """)
        
        # Orderbook Health Metrics
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_health_metrics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                spread_bps FLOAT NOT NULL, -- Spread in basis points
                is_stale BOOLEAN NOT NULL,
                staleness_seconds INTEGER NOT NULL,
                liquidity_score FLOAT NOT NULL,
                imbalance_score FLOAT NOT NULL,
                wall_count INTEGER NOT NULL,
                health_score FLOAT NOT NULL
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_health 
            ON orderbook_health_metrics (symbol, timestamp);
        """)
        
        logger.info("✅ Data sanity tables created")
    
    async def _create_performance_indexes(self):
        """Create performance indexes for the new tables"""
        logger.info("📋 Creating performance indexes...")
        
        # Interface performance indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_interface_performance_component 
            ON interface_performance_metrics (component_name, interface_type);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_interface_performance_timestamp 
            ON interface_performance_metrics (timestamp);
        """)
        
        # Calibration indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibration_history_model 
            ON calibration_history (model_name, calibration_type);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibration_history_timestamp 
            ON calibration_history (timestamp);
        """)
        
        # Gate validation indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_gate_validation_gate 
            ON gate_validation_results (gate_name, gate_type);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_gate_validation_timestamp 
            ON gate_validation_results (timestamp);
        """)
        
        # Signal payload indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_enhanced_signal_payloads_timestamp 
            ON enhanced_signal_payloads (created_at);
        """)
        
        # News override indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_override_events_timestamp 
            ON news_override_events (timestamp);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_real_time_news_feed_symbol 
            ON real_time_news_feed (symbol, published_at);
        """)
        
        # Quota indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_quota_usage_window 
            ON quota_usage_tracking (quota_type, window_start, window_end);
        """)
        
        # Mutex indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_active_signal_mutex_symbol 
            ON active_signal_mutex (symbol, status);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_cooldown_tracking_symbol 
            ON cooldown_tracking (symbol, timeframe, is_active);
        """)
        
        # Data sanity indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_sanity_validation_signal 
            ON data_sanity_validation_results (signal_id, passed);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_orderbook_health_symbol 
            ON orderbook_health_metrics (symbol, timestamp);
        """)
        
        logger.info("✅ Performance indexes created")
    
    async def _insert_default_configuration(self):
        """Insert default configuration for the surgical upgrades"""
        logger.info("📋 Inserting default configuration...")
        
        # Default gate configurations
        await self.connection.execute("""
            INSERT INTO signal_gates_config (gate_name, gate_type, threshold_value, threshold_type, description) VALUES
            ('data_health_minimum', 'data_sanity', 0.9, 'absolute', 'Minimum data health score required'),
            ('orderbook_staleness_max', 'data_sanity', 3.0, 'absolute', 'Maximum orderbook staleness in seconds'),
            ('spread_max_atr_ratio', 'data_sanity', 0.12, 'ratio', 'Maximum spread as ratio of ATR'),
            ('daily_signal_quota', 'quota', 10, 'absolute', 'Maximum signals per day system-wide'),
            ('hourly_signal_quota', 'quota', 4, 'absolute', 'Maximum signals per hour system-wide'),
            ('symbol_cooldown_minutes', 'mutex', 30, 'absolute', 'Cooldown period after signal closes'),
            ('news_negative_threshold', 'news', -0.6, 'absolute', 'Negative news sentiment threshold'),
            ('news_positive_threshold', 'news', 0.6, 'absolute', 'Positive news sentiment threshold'),
            ('min_risk_reward_ratio', 'risk', 2.0, 'absolute', 'Minimum risk/reward ratio required'),
            ('confidence_threshold', 'quality', 0.85, 'absolute', 'Minimum confidence score required')
            ON CONFLICT (gate_name) DO NOTHING;
        """)
        
        # Default quota configurations
        await self.connection.execute("""
            INSERT INTO quota_configuration (quota_type, quota_limit, quota_window_hours, priority_threshold, replacement_strategy) VALUES
            ('daily', 10, 24, 0.85, 'drop_worst'),
            ('hourly', 4, 1, 0.85, 'drop_worst'),
            ('symbol', 1, 24, 0.80, 'fifo'),
            ('system', 50, 24, 0.90, 'priority')
            ON CONFLICT (quota_type) DO NOTHING;
        """)
        
        # Default news override rules
        await self.connection.execute("""
            INSERT INTO news_override_rules (rule_name, sentiment_threshold, impact_threshold, action_type, symbol_pattern) VALUES
            ('block_negative_news', -0.6, 0.3, 'block_long', '%'),
            ('allow_positive_news', 0.6, 0.3, 'allow_early', '%'),
            ('block_high_impact_negative', -0.4, 0.7, 'block_long', '%'),
            ('allow_high_impact_positive', 0.4, 0.7, 'allow_early', '%')
            ON CONFLICT (rule_name) DO NOTHING;
        """)
        
        # Default data sanity rules
        await self.connection.execute("""
            INSERT INTO data_sanity_rules (rule_name, data_type, validation_type, threshold_value, threshold_operator, is_critical) VALUES
            ('orderbook_not_stale', 'orderbook', 'staleness', 3.0, 'lt', TRUE),
            ('spread_reasonable', 'orderbook', 'spread', 0.12, 'lt', TRUE),
            ('volume_sufficient', 'volume', 'volume', 1000.0, 'gt', FALSE),
            ('price_change_reasonable', 'candlestick', 'price', 0.1, 'lt', TRUE),
            ('sentiment_data_fresh', 'sentiment', 'staleness', 300.0, 'lt', FALSE)
            ON CONFLICT (rule_name) DO NOTHING;
        """)
        
        # Default reliability buckets
        await self.connection.execute("""
            INSERT INTO reliability_buckets (bucket_range, expected_win_rate, actual_win_rate, sample_count) VALUES
            ('0.80-0.85', 0.825, 0.825, 0),
            ('0.85-0.90', 0.875, 0.875, 0),
            ('0.90-0.95', 0.925, 0.925, 0),
            ('0.95-1.00', 0.975, 0.975, 0)
            ON CONFLICT (bucket_range) DO NOTHING;
        """)
        
        logger.info("✅ Default configuration inserted")

async def run_surgical_upgrades_phase1_migration():
    """Run the Phase 1 migration for surgical upgrades"""
    connection_params = {
        'host': 'localhost',
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    migration = SurgicalUpgradesPhase1Migration(connection_params)
    success = await migration.run_migration()
    
    if success:
        logger.info("🎉 Surgical Upgrades Phase 1 Migration completed successfully!")
    else:
        logger.error("❌ Surgical Upgrades Phase 1 Migration failed!")
    
    return success

if __name__ == "__main__":
    asyncio.run(run_surgical_upgrades_phase1_migration())
