"""
SDE Integration Phase 2: Enhanced Signal Integration with Explainability
Implements explainability payload, advanced validation, and signal lifecycle management
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
    """Run SDE Integration Phase 2 migration"""
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
            logger.info("🚀 Starting SDE Integration Phase 2 Migration")
            
            # 1. Create SDE Explainability Payload Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_explainability_payload (
                    explainability_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id VARCHAR(50),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Model Consensus Breakdown
                    head_a_direction VARCHAR(10),
                    head_a_probability DECIMAL(5,4),
                    head_a_confidence DECIMAL(5,4),
                    head_a_reasoning TEXT,
                    
                    head_b_direction VARCHAR(10),
                    head_b_probability DECIMAL(5,4),
                    head_b_confidence DECIMAL(5,4),
                    head_b_reasoning TEXT,
                    
                    head_c_direction VARCHAR(10),
                    head_c_probability DECIMAL(5,4),
                    head_c_confidence DECIMAL(5,4),
                    head_c_reasoning TEXT,
                    
                    head_d_direction VARCHAR(10),
                    head_d_probability DECIMAL(5,4),
                    head_d_confidence DECIMAL(5,4),
                    head_d_reasoning TEXT,
                    
                    -- Feature Importance
                    top_features JSONB,
                    feature_importance_weights JSONB,
                    
                    -- Confluence Breakdown
                    zone_score DECIMAL(5,4),
                    htf_bias_score DECIMAL(5,4),
                    trigger_score DECIMAL(5,4),
                    ob_confluence_score DECIMAL(5,4),
                    sentiment_score DECIMAL(5,4),
                    
                    -- Historical Analogs
                    historical_analogs JSONB,
                    similar_setups_count INTEGER,
                    avg_outcome_score DECIMAL(5,4),
                    
                    -- Natural Language Reasoning
                    reasoning_summary TEXT,
                    confidence_factors JSONB,
                    risk_factors JSONB,
                    
                    -- Expected Utility
                    eu_long DECIMAL(5,4),
                    eu_short DECIMAL(5,4),
                    expected_rr_ratio DECIMAL(5,4),
                    gate_pass_info JSONB,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_explainability_payload table")
            
            # 2. Create SDE Advanced Validation Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_advanced_validation (
                    validation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id VARCHAR(50),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Advanced Quality Gates
                    data_sanity_passed BOOLEAN NOT NULL,
                    spread_validation_passed BOOLEAN NOT NULL,
                    volatility_regime_passed BOOLEAN NOT NULL,
                    liquidity_check_passed BOOLEAN NOT NULL,
                    market_microstructure_passed BOOLEAN NOT NULL,
                    
                    -- Detailed Validation Scores
                    data_sanity_score DECIMAL(5,4),
                    spread_score DECIMAL(5,4),
                    volatility_score DECIMAL(5,4),
                    liquidity_score DECIMAL(5,4),
                    microstructure_score DECIMAL(5,4),
                    
                    -- Market Conditions
                    current_spread DECIMAL(10,8),
                    atr_value DECIMAL(10,8),
                    atr_percentile DECIMAL(5,4),
                    volume_profile_health DECIMAL(5,4),
                    orderbook_depth DECIMAL(10,8),
                    
                    -- Validation Metadata
                    validation_version VARCHAR(20) DEFAULT '2.0.0',
                    processing_time_ms INTEGER,
                    validation_reason TEXT,
                    failed_gates JSONB,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_advanced_validation table")
            
            # 3. Create SDE Signal Lifecycle Management Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_signal_lifecycle (
                    lifecycle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id VARCHAR(50),
                    symbol VARCHAR(20) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    
                    -- Lifecycle States
                    current_state VARCHAR(50) NOT NULL,
                    previous_state VARCHAR(50),
                    state_transition_reason TEXT,
                    
                    -- Signal Lifecycle Events
                    signal_generated_at TIMESTAMPTZ,
                    consensus_checked_at TIMESTAMPTZ,
                    confluence_calculated_at TIMESTAMPTZ,
                    execution_validated_at TIMESTAMPTZ,
                    news_checked_at TIMESTAMPTZ,
                    limits_checked_at TIMESTAMPTZ,
                    signal_emitted_at TIMESTAMPTZ,
                    signal_expired_at TIMESTAMPTZ,
                    signal_cancelled_at TIMESTAMPTZ,
                    
                    -- Performance Tracking
                    time_to_consensus_ms INTEGER,
                    time_to_confluence_ms INTEGER,
                    time_to_execution_ms INTEGER,
                    time_to_emission_ms INTEGER,
                    total_processing_time_ms INTEGER,
                    
                    -- Lifecycle Metadata
                    lifecycle_version VARCHAR(20) DEFAULT '2.0.0',
                    state_history JSONB,
                    performance_metrics JSONB,
                    
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_signal_lifecycle table")
            
            # 4. Create SDE Performance Analytics Table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sde_performance_analytics (
                    analytics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    analytics_date DATE NOT NULL,
                    symbol VARCHAR(20),
                    timeframe VARCHAR(10),
                    
                    -- Signal Quality Metrics
                    total_signals_processed INTEGER DEFAULT 0,
                    signals_passing_all_gates INTEGER DEFAULT 0,
                    signals_rejected_by_consensus INTEGER DEFAULT 0,
                    signals_rejected_by_confluence INTEGER DEFAULT 0,
                    signals_rejected_by_execution INTEGER DEFAULT 0,
                    signals_rejected_by_news INTEGER DEFAULT 0,
                    signals_rejected_by_limits INTEGER DEFAULT 0,
                    
                    -- Quality Scores
                    avg_consensus_score DECIMAL(5,4),
                    avg_confluence_score DECIMAL(5,4),
                    avg_execution_quality DECIMAL(5,4),
                    avg_final_confidence DECIMAL(5,4),
                    avg_explainability_score DECIMAL(5,4),
                    
                    -- Performance Metrics
                    avg_processing_time_ms INTEGER,
                    p95_processing_time_ms INTEGER,
                    p99_processing_time_ms INTEGER,
                    max_processing_time_ms INTEGER,
                    min_processing_time_ms INTEGER,
                    
                    -- Gate Performance
                    consensus_gate_success_rate DECIMAL(5,4),
                    confluence_gate_success_rate DECIMAL(5,4),
                    execution_gate_success_rate DECIMAL(5,4),
                    news_gate_success_rate DECIMAL(5,4),
                    limits_gate_success_rate DECIMAL(5,4),
                    
                    -- System Health
                    system_health_score DECIMAL(5,4),
                    data_quality_score DECIMAL(5,4),
                    model_health_score DECIMAL(5,4),
                    api_health_score DECIMAL(5,4),
                    
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            logger.info("✅ Created sde_performance_analytics table")
            
            # 5. Create Performance Indexes
            indexes = [
                ("idx_sde_explainability_signal_id", "sde_explainability_payload", "signal_id"),
                ("idx_sde_explainability_symbol_time", "sde_explainability_payload", "symbol, timestamp"),
                
                ("idx_sde_advanced_validation_signal_id", "sde_advanced_validation", "signal_id"),
                ("idx_sde_advanced_validation_symbol_time", "sde_advanced_validation", "symbol, timestamp"),
                ("idx_sde_advanced_validation_sanity", "sde_advanced_validation", "data_sanity_passed"),
                
                ("idx_sde_lifecycle_signal_id", "sde_signal_lifecycle", "signal_id"),
                ("idx_sde_lifecycle_symbol_time", "sde_signal_lifecycle", "symbol, timestamp"),
                ("idx_sde_lifecycle_state", "sde_signal_lifecycle", "current_state"),
                
                ("idx_sde_analytics_date", "sde_performance_analytics", "analytics_date"),
                ("idx_sde_analytics_symbol", "sde_performance_analytics", "symbol"),
                ("idx_sde_analytics_timeframe", "sde_performance_analytics", "timeframe")
            ]
            
            for index_name, table_name, columns in indexes:
                try:
                    await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns})")
                    logger.info(f"✅ Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create index {index_name}: {e}")
            
            # 6. Insert Enhanced Integration Configurations
            enhanced_configs = [
                {
                    'config_name': 'sde_explainability_default',
                    'config_type': 'explainability',
                    'config_data': {
                        'enable_feature_importance': True,
                        'enable_historical_analogs': True,
                        'enable_natural_language': True,
                        'enable_expected_utility': True,
                        'max_historical_analogs': 10,
                        'min_analog_similarity': 0.7,
                        'feature_importance_threshold': 0.05,
                        'reasoning_summary_length': 200
                    },
                    'description': 'Default SDE explainability configuration'
                },
                {
                    'config_name': 'sde_advanced_validation_default',
                    'config_type': 'advanced_validation',
                    'config_data': {
                        'enable_data_sanity': True,
                        'enable_spread_validation': True,
                        'enable_volatility_regime': True,
                        'enable_liquidity_check': True,
                        'enable_microstructure': True,
                        'max_spread_atr_ratio': 0.12,
                        'min_atr_percentile': 25.0,
                        'max_atr_percentile': 75.0,
                        'min_volume_profile_health': 0.8,
                        'min_orderbook_depth': 1000.0
                    },
                    'description': 'Default SDE advanced validation configuration'
                },
                {
                    'config_name': 'sde_lifecycle_default',
                    'config_type': 'lifecycle',
                    'config_data': {
                        'enable_lifecycle_tracking': True,
                        'enable_performance_tracking': True,
                        'enable_state_history': True,
                        'signal_expiry_minutes': 60,
                        'max_processing_time_ms': 100,
                        'enable_auto_cleanup': True,
                        'cleanup_interval_hours': 24
                    },
                    'description': 'Default SDE lifecycle configuration'
                },
                {
                    'config_name': 'sde_analytics_default',
                    'config_type': 'analytics',
                    'config_data': {
                        'analytics_collection_interval': 300,  # 5 minutes
                        'enable_real_time_analytics': True,
                        'enable_historical_analytics': True,
                        'enable_performance_tracking': True,
                        'enable_health_monitoring': True,
                        'alert_threshold_processing_time': 100,
                        'alert_threshold_rejection_rate': 0.3,
                        'alert_threshold_confidence_drop': 0.1
                    },
                    'description': 'Default SDE analytics configuration'
                }
            ]
            
            for config in enhanced_configs:
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
                    logger.info(f"✅ Inserted enhanced config: {config['config_name']}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to insert config {config['config_name']}: {e}")
            
            logger.info("🎉 SDE Integration Phase 2 Migration Completed Successfully!")
            
        await pool.close()
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(run_migration())
