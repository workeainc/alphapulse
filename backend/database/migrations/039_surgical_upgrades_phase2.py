#!/usr/bin/env python3
"""
Surgical Upgrades Phase 2: Confidence Calibration & Enhanced Signal Generation
Database Migration for Advanced Signal Quality and Risk Management

This migration implements:
1. Confidence calibration system with reliability tracking
2. Enhanced signal generation with calibrated fusion
3. Hard gating implementation with real-time validation
4. Signal lifecycle management with cooldown and mutex
5. Advanced quota management with priority-based replacement
"""

import asyncio
import asyncpg
import logging
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgicalUpgradesPhase2Migration:
    """Phase 2 migration for surgical upgrades"""
    
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
        """Run the complete Phase 2 migration"""
        try:
            if not await self.connect():
                return False
            
            logger.info("🚀 Starting Surgical Upgrades Phase 2 Migration...")
            
            # Step 1: Enhanced Signal Generation Tables
            await self._create_enhanced_signal_generation_tables()
            
            # Step 2: Confidence Calibration Implementation
            await self._create_confidence_calibration_implementation()
            
            # Step 3: Hard Gating Implementation
            await self._create_hard_gating_implementation()
            
            # Step 4: Signal Lifecycle Management
            await self._create_signal_lifecycle_management()
            
            # Step 5: Advanced Quota Management
            await self._create_advanced_quota_management()
            
            # Step 6: Real-time Validation System
            await self._create_real_time_validation_system()
            
            # Step 7: Performance Monitoring Enhancement
            await self._create_performance_monitoring_enhancement()
            
            # Step 8: Create Indexes for Performance
            await self._create_enhanced_performance_indexes()
            
            # Step 9: Insert Default Calibration Data
            await self._insert_default_calibration_data()
            
            logger.info("✅ Surgical Upgrades Phase 2 Migration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
        finally:
            if self.connection:
                await self.connection.close()
    
    async def _create_enhanced_signal_generation_tables(self):
        """Create tables for enhanced signal generation"""
        logger.info("📋 Creating enhanced signal generation tables...")
        
        # Calibrated Signal Generation
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS calibrated_signal_generation (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                raw_confidence FLOAT NOT NULL,
                calibrated_confidence FLOAT NOT NULL,
                calibration_model VARCHAR(100) NOT NULL,
                calibration_type VARCHAR(50) NOT NULL, -- 'isotonic', 'platt', 'temperature'
                reliability_score FLOAT NOT NULL,
                brier_score FLOAT NOT NULL,
                calibration_metrics JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(signal_id)
            )
        """)
        
        # Confidence Fusion Components
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS confidence_fusion_components (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                component_name VARCHAR(100) NOT NULL,
                component_weight FLOAT NOT NULL,
                raw_score FLOAT NOT NULL,
                normalized_score FLOAT NOT NULL,
                drift_adjustment FLOAT DEFAULT 0.0,
                final_contribution FLOAT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_component 
            ON confidence_fusion_components (signal_id, component_name);
        """)
        
        # Ensemble Voting Results
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_voting_results (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                model_weight FLOAT NOT NULL,
                prediction FLOAT NOT NULL,
                confidence FLOAT NOT NULL,
                drift_score FLOAT DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_model 
            ON ensemble_voting_results (signal_id, model_name);
        """)
        
        logger.info("✅ Enhanced signal generation tables created")
    
    async def _create_confidence_calibration_implementation(self):
        """Create tables for confidence calibration implementation"""
        logger.info("📋 Creating confidence calibration implementation...")
        
        # Calibration Training Data
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS calibration_training_data (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                calibration_type VARCHAR(50) NOT NULL,
                raw_confidence FLOAT NOT NULL,
                actual_outcome BOOLEAN NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_used_for_training BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_calibration 
            ON calibration_training_data (model_name, calibration_type);
        """)
        
        # Calibration Model Performance
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS calibration_model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                calibration_type VARCHAR(50) NOT NULL,
                brier_score FLOAT NOT NULL,
                reliability_score FLOAT NOT NULL,
                calibration_error FLOAT NOT NULL,
                sample_count INTEGER NOT NULL,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(model_name, calibration_type)
            )
        """)
        
        # Dynamic Threshold Adjustment
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_threshold_adjustment (
                id SERIAL PRIMARY KEY,
                threshold_type VARCHAR(50) NOT NULL, -- 'confidence', 'risk_reward', 'volume'
                current_threshold FLOAT NOT NULL,
                target_win_rate FLOAT NOT NULL,
                actual_win_rate FLOAT NOT NULL,
                adjustment_factor FLOAT NOT NULL,
                sample_size INTEGER NOT NULL,
                last_adjusted TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE,
                UNIQUE(threshold_type)
            )
        """)
        
        logger.info("✅ Confidence calibration implementation created")
    
    async def _create_hard_gating_implementation(self):
        """Create tables for hard gating implementation"""
        logger.info("📋 Creating hard gating implementation...")
        
        # Gate Execution Engine
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS gate_execution_engine (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                gate_sequence INTEGER NOT NULL,
                gate_name VARCHAR(100) NOT NULL,
                gate_type VARCHAR(50) NOT NULL,
                input_data JSONB NOT NULL,
                output_data JSONB NOT NULL,
                execution_time_ms FLOAT NOT NULL,
                passed BOOLEAN NOT NULL,
                failure_reason TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_gate_sequence 
            ON gate_execution_engine (signal_id, gate_sequence);
        """)
        
        # Gate Dependencies
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS gate_dependencies (
                id SERIAL PRIMARY KEY,
                gate_name VARCHAR(100) NOT NULL,
                depends_on_gate VARCHAR(100),
                dependency_type VARCHAR(20) NOT NULL, -- 'required', 'optional', 'conditional'
                condition_expression TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_gate_dependencies 
            ON gate_dependencies (gate_name, depends_on_gate);
        """)
        
        # Gate Performance Analytics
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS gate_performance_analytics (
                id SERIAL PRIMARY KEY,
                gate_name VARCHAR(100) NOT NULL,
                total_executions INTEGER DEFAULT 0,
                passed_executions INTEGER DEFAULT 0,
                failed_executions INTEGER DEFAULT 0,
                avg_execution_time_ms FLOAT DEFAULT 0,
                success_rate FLOAT DEFAULT 0,
                failure_patterns JSONB,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(gate_name)
            )
        """)
        
        logger.info("✅ Hard gating implementation created")
    
    async def _create_signal_lifecycle_management(self):
        """Create tables for signal lifecycle management"""
        logger.info("📋 Creating signal lifecycle management...")
        
        # Signal State Machine
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_state_machine (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                current_state VARCHAR(50) NOT NULL, -- 'generated', 'validated', 'active', 'closed', 'expired'
                previous_state VARCHAR(50),
                state_transition_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                transition_reason TEXT,
                state_data JSONB
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_state 
            ON signal_state_machine (signal_id, current_state);
        """)
        
        # Signal Expiry Management
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_expiry_management (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                expiry_type VARCHAR(50) NOT NULL, -- 'time_based', 'condition_based', 'manual'
                expiry_condition TEXT,
                expiry_time TIMESTAMP WITH TIME ZONE,
                is_expired BOOLEAN DEFAULT FALSE,
                expiry_reason TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(signal_id)
            )
        """)
        
        # Signal Cooldown Management
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS signal_cooldown_management (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                last_signal_id VARCHAR(100) NOT NULL,
                cooldown_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                cooldown_end TIMESTAMP WITH TIME ZONE NOT NULL,
                cooldown_duration_minutes INTEGER NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                cooldown_reason TEXT
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_timeframe_cooldown 
            ON signal_cooldown_management (symbol, timeframe, is_active);
        """)
        
        logger.info("✅ Signal lifecycle management created")
    
    async def _create_advanced_quota_management(self):
        """Create tables for advanced quota management"""
        logger.info("📋 Creating advanced quota management...")
        
        # Priority-based Signal Queue
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS priority_signal_queue (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                confidence_score FLOAT NOT NULL,
                priority_score FLOAT NOT NULL,
                queue_position INTEGER NOT NULL,
                queue_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE,
                replacement_candidate BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_priority_queue 
            ON priority_signal_queue (priority_score DESC, queue_timestamp);
        """)
        
        # Quota Replacement History
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS quota_replacement_history (
                id SERIAL PRIMARY KEY,
                quota_type VARCHAR(50) NOT NULL,
                replaced_signal_id VARCHAR(100) NOT NULL,
                replacing_signal_id VARCHAR(100) NOT NULL,
                replaced_priority FLOAT NOT NULL,
                replacing_priority FLOAT NOT NULL,
                replacement_reason TEXT NOT NULL,
                replacement_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_replacement_history 
            ON quota_replacement_history (quota_type, replacement_timestamp);
        """)
        
        # Dynamic Quota Adjustment
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_quota_adjustment (
                id SERIAL PRIMARY KEY,
                quota_type VARCHAR(50) NOT NULL,
                current_limit INTEGER NOT NULL,
                adjusted_limit INTEGER NOT NULL,
                adjustment_reason TEXT NOT NULL,
                performance_metrics JSONB,
                adjustment_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                is_active BOOLEAN DEFAULT TRUE
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_quota_adjustment 
            ON dynamic_quota_adjustment (quota_type, adjustment_timestamp);
        """)
        
        logger.info("✅ Advanced quota management created")
    
    async def _create_real_time_validation_system(self):
        """Create tables for real-time validation system"""
        logger.info("📋 Creating real-time validation system...")
        
        # Real-time Validation Rules
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS real_time_validation_rules (
                id SERIAL PRIMARY KEY,
                rule_name VARCHAR(100) NOT NULL,
                rule_type VARCHAR(50) NOT NULL, -- 'data_quality', 'market_condition', 'risk_management'
                rule_expression TEXT NOT NULL,
                validation_function TEXT,
                is_critical BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(rule_name)
            )
        """)
        
        # Validation Rule Execution
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS validation_rule_execution (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                rule_name VARCHAR(100) NOT NULL,
                rule_type VARCHAR(50) NOT NULL,
                input_data JSONB NOT NULL,
                output_data JSONB NOT NULL,
                execution_time_ms FLOAT NOT NULL,
                passed BOOLEAN NOT NULL,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_rule 
            ON validation_rule_execution (signal_id, rule_name);
        """)
        
        # Market Condition Validation
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS market_condition_validation (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                validation_type VARCHAR(50) NOT NULL, -- 'volatility', 'liquidity', 'spread', 'volume'
                current_value FLOAT NOT NULL,
                threshold_value FLOAT NOT NULL,
                is_valid BOOLEAN NOT NULL,
                validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_validation 
            ON market_condition_validation (symbol, validation_type);
        """)
        
        logger.info("✅ Real-time validation system created")
    
    async def _create_performance_monitoring_enhancement(self):
        """Create tables for performance monitoring enhancement"""
        logger.info("📋 Creating performance monitoring enhancement...")
        
        # Enhanced Performance Metrics
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_performance_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_type VARCHAR(50) NOT NULL, -- 'latency', 'throughput', 'accuracy', 'reliability'
                metric_value FLOAT NOT NULL,
                metric_unit VARCHAR(20) NOT NULL,
                component_name VARCHAR(100),
                signal_id VARCHAR(100),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_metric_component 
            ON enhanced_performance_metrics (metric_name, component_name, timestamp);
        """)
        
        # Performance Alerting
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS performance_alerting (
                id SERIAL PRIMARY KEY,
                alert_name VARCHAR(100) NOT NULL,
                alert_type VARCHAR(50) NOT NULL, -- 'threshold', 'trend', 'anomaly'
                metric_name VARCHAR(100) NOT NULL,
                threshold_value FLOAT NOT NULL,
                current_value FLOAT NOT NULL,
                severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
                alert_message TEXT NOT NULL,
                is_acknowledged BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        # Create index separately
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_alert_metric 
            ON performance_alerting (alert_name, metric_name, created_at);
        """)
        
        # System Health Monitoring
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS system_health_monitoring (
                id SERIAL PRIMARY KEY,
                component_name VARCHAR(100) NOT NULL,
                health_score FLOAT NOT NULL,
                status VARCHAR(20) NOT NULL, -- 'healthy', 'warning', 'critical', 'down'
                error_count INTEGER DEFAULT 0,
                last_error TEXT,
                uptime_seconds INTEGER DEFAULT 0,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(component_name)
            )
        """)
        
        logger.info("✅ Performance monitoring enhancement created")
    
    async def _create_enhanced_performance_indexes(self):
        """Create enhanced performance indexes"""
        logger.info("📋 Creating enhanced performance indexes...")
        
        # Calibrated signal generation indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibrated_signal_confidence 
            ON calibrated_signal_generation (calibrated_confidence DESC);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibrated_signal_model 
            ON calibrated_signal_generation (calibration_model, calibration_type);
        """)
        
        # Confidence fusion indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence_fusion_signal 
            ON confidence_fusion_components (signal_id, component_weight DESC);
        """)
        
        # Ensemble voting indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_ensemble_voting_signal 
            ON ensemble_voting_results (signal_id, model_weight DESC);
        """)
        
        # Gate execution indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_gate_execution_signal 
            ON gate_execution_engine (signal_id, gate_sequence);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_gate_execution_passed 
            ON gate_execution_engine (passed, execution_time_ms);
        """)
        
        # Signal state machine indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_state_current 
            ON signal_state_machine (current_state, state_transition_time);
        """)
        
        # Priority queue indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_priority_queue_active 
            ON priority_signal_queue (is_active, priority_score DESC);
        """)
        
        # Performance metrics indexes
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp 
            ON enhanced_performance_metrics (timestamp DESC);
        """)
        
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_component 
            ON enhanced_performance_metrics (component_name, metric_name);
        """)
        
        logger.info("✅ Enhanced performance indexes created")
    
    async def _insert_default_calibration_data(self):
        """Insert default calibration data"""
        logger.info("📋 Inserting default calibration data...")
        
        try:
            # Default dynamic thresholds
            await self.connection.execute("""
                INSERT INTO dynamic_threshold_adjustment 
                (threshold_type, current_threshold, target_win_rate, actual_win_rate, adjustment_factor, sample_size) VALUES
                ('confidence', 0.85, 0.85, 0.85, 1.0, 0),
                ('risk_reward', 2.0, 0.80, 0.80, 1.0, 0),
                ('volume', 1000.0, 0.75, 0.75, 1.0, 0);
            """)
        except Exception as e:
            logger.info(f"Dynamic thresholds already exist: {e}")
        
        try:
            # Default validation rules
            await self.connection.execute("""
                INSERT INTO real_time_validation_rules 
                (rule_name, rule_type, rule_expression, is_critical) VALUES
                ('data_freshness', 'data_quality', 'data_age < 300', TRUE),
                ('spread_reasonable', 'market_condition', 'spread_bps < 50', TRUE),
                ('volume_sufficient', 'market_condition', 'volume > 1000', FALSE),
                ('volatility_acceptable', 'risk_management', 'atr_ratio < 0.1', TRUE);
            """)
        except Exception as e:
            logger.info(f"Validation rules already exist: {e}")
        
        try:
            # Default gate dependencies
            await self.connection.execute("""
                INSERT INTO gate_dependencies 
                (gate_name, depends_on_gate, dependency_type) VALUES
                ('data_sanity', NULL, 'required'),
                ('quota_check', 'data_sanity', 'required'),
                ('mutex_check', 'quota_check', 'required'),
                ('news_override', 'mutex_check', 'conditional'),
                ('final_validation', 'news_override', 'required');
            """)
        except Exception as e:
            logger.info(f"Gate dependencies already exist: {e}")
        
        try:
            # Default system health
            await self.connection.execute("""
                INSERT INTO system_health_monitoring 
                (component_name, health_score, status, uptime_seconds) VALUES
                ('ONNXInferenceEngine', 1.0, 'healthy', 0),
                ('FeatureDriftDetector', 1.0, 'healthy', 0),
                ('SignalGenerator', 1.0, 'healthy', 0),
                ('GateExecutionEngine', 1.0, 'healthy', 0);
            """)
        except Exception as e:
            logger.info(f"System health data already exists: {e}")
        
        logger.info("✅ Default calibration data inserted")

async def run_surgical_upgrades_phase2_migration():
    """Run the Phase 2 migration for surgical upgrades"""
    connection_params = {
        'host': 'localhost',
        'port': 5432,
        'user': 'alpha_emon',
        'password': 'Emon_@17711',
        'database': 'alphapulse'
    }
    
    migration = SurgicalUpgradesPhase2Migration(connection_params)
    success = await migration.run_migration()
    
    if success:
        logger.info("🎉 Surgical Upgrades Phase 2 Migration completed successfully!")
    else:
        logger.error("❌ Surgical Upgrades Phase 2 Migration failed!")
    
    return success

if __name__ == "__main__":
    asyncio.run(run_surgical_upgrades_phase2_migration())
