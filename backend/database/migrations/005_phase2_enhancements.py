#!/usr/bin/env python3
"""
Migration: Add Phase 2 Advanced Analytics Columns
Add new columns for Phase 2 enhancements to existing tables
"""

import asyncio
import logging
import os
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for psql authentication
os.environ['PGPASSWORD'] = 'Emon_@17711'

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def add_phase2_columns():
    """Add Phase 2 columns to existing tables"""
    
    # Add Phase 2 columns to enhanced_market_intelligence table
    add_enhanced_market_intelligence_columns = """
    ALTER TABLE enhanced_market_intelligence 
    ADD COLUMN IF NOT EXISTS rolling_beta_btc_eth NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS rolling_beta_btc_altcoins NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS lead_lag_analysis JSONB,
    ADD COLUMN IF NOT EXISTS correlation_breakdown_alerts JSONB,
    ADD COLUMN IF NOT EXISTS optimal_timing_signals JSONB,
    ADD COLUMN IF NOT EXISTS monte_carlo_scenarios JSONB,
    ADD COLUMN IF NOT EXISTS confidence_bands JSONB,
    ADD COLUMN IF NOT EXISTS feature_importance_scores JSONB,
    ADD COLUMN IF NOT EXISTS ensemble_model_weights JSONB,
    ADD COLUMN IF NOT EXISTS prediction_horizons JSONB;
    """
    
    # Add Phase 2 columns to correlation_analysis table
    add_correlation_analysis_columns = """
    ALTER TABLE correlation_analysis 
    ADD COLUMN IF NOT EXISTS cross_market_correlations JSONB,
    ADD COLUMN IF NOT EXISTS beta_regime VARCHAR(50),
    ADD COLUMN IF NOT EXISTS lead_lag_confidence NUMERIC(4,3);
    """
    
    # Add Phase 2 columns to predictive_market_regime table
    add_predictive_regime_columns = """
    ALTER TABLE predictive_market_regime 
    ADD COLUMN IF NOT EXISTS xgboost_prediction NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS catboost_prediction NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS ensemble_prediction NUMERIC(6,4),
    ADD COLUMN IF NOT EXISTS prediction_confidence NUMERIC(4,3),
    ADD COLUMN IF NOT EXISTS model_performance_metrics JSONB;
    """
    
    try:
        # Connect to database
        conn = await asyncpg.connect(**DB_CONFIG)
        logger.info("✅ Connected to database successfully")
        
        # Add columns to enhanced_market_intelligence table
        logger.info("Adding Phase 2 columns to enhanced_market_intelligence table...")
        await conn.execute(add_enhanced_market_intelligence_columns)
        logger.info("✅ Phase 2 columns added to enhanced_market_intelligence table")
        
        # Add columns to correlation_analysis table
        logger.info("Adding Phase 2 columns to correlation_analysis table...")
        await conn.execute(add_correlation_analysis_columns)
        logger.info("✅ Phase 2 columns added to correlation_analysis table")
        
        # Add columns to predictive_market_regime table
        logger.info("Adding Phase 2 columns to predictive_market_regime table...")
        await conn.execute(add_predictive_regime_columns)
        logger.info("✅ Phase 2 columns added to predictive_market_regime table")
        
        await conn.close()
        logger.info("✅ Phase 2 columns migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error adding Phase 2 columns: {e}")
        raise

async def main():
    """Main migration function"""
    logger.info("🚀 Starting Phase 2 Columns Migration...")
    await add_phase2_columns()
    logger.info("✅ Phase 2 Columns Migration completed!")

if __name__ == "__main__":
    asyncio.run(main())
