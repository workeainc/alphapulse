#!/usr/bin/env python3
"""
Database Migration Script for Priority 4: Advanced Signal Validation

Creates tables for tracking:
1. Advanced signal validation metrics and quality scores
2. False positive analysis and pattern failure rates
3. Market regime filtering performance
4. Adaptive threshold management
5. Signal validation performance tracking
"""

import asyncio
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Priority4DatabaseMigration:
    """Database migration for Priority 4 Advanced Signal Validation"""
    
    def __init__(self, connection_string: str = None):
        """Initialize migration with database connection"""
        if connection_string is None:
            # Default connection string - use the correct TimescaleDB credentials
            self.connection_string = (
                "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
            )
        else:
            self.connection_string = connection_string
        
        logger.info("🚀 Priority 4 Database Migration initialized")
    
    async def run_migration(self):
        """Run the complete Priority 4 migration"""
        logger.info("Starting Priority 4 Advanced Signal Validation migration...")
        
        try:
            # Create all tables
            await self._create_advanced_signal_validation_table()
            await self._create_signal_quality_metrics_table()
            await self._create_false_positive_analysis_table()
            await self._create_market_regime_filtering_table()
            await self._create_adaptive_thresholds_table()
            await self._create_validation_performance_table()
            
            # Apply TimescaleDB features
            await self._apply_timescaledb_features()
            
            logger.info("✅ Priority 4 migration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Priority 4 migration failed: {e}")
            raise
    
    async def _create_advanced_signal_validation_table(self):
        """Create table for advanced signal validation results"""
        query = """
        CREATE TABLE IF NOT EXISTS priority4_advanced_signal_validation (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            signal_id VARCHAR(100),
            signal_type VARCHAR(20) NOT NULL,
            strategy_name VARCHAR(100),
            validation_result VARCHAR(30) NOT NULL,
            overall_quality_score DECIMAL(5,4),
            confidence_score DECIMAL(5,4),
            volume_confirmation_score DECIMAL(5,4),
            pattern_strength_score DECIMAL(5,4),
            regime_alignment_score DECIMAL(5,4),
            timeframe_confluence_score DECIMAL(5,4),
            risk_score DECIMAL(5,4),
            volatility_adjustment DECIMAL(6,5),
            quality_level VARCHAR(20),
            warnings JSONB,
            recommendations JSONB,
            validation_details JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Advanced signal validation table created")
        except Exception as e:
            logger.error(f"❌ Failed to create advanced signal validation table: {e}")
            raise
    
    async def _create_signal_quality_metrics_table(self):
        """Create table for detailed signal quality metrics"""
        query = """
        CREATE TABLE IF NOT EXISTS priority4_signal_quality_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            signal_id VARCHAR(100),
            quality_breakdown JSONB,
            feature_importance JSONB,
            technical_indicators JSONB,
            market_conditions JSONB,
            volume_analysis JSONB,
            pattern_analysis JSONB,
            regime_analysis JSONB,
            timeframe_analysis JSONB,
            risk_analysis JSONB,
            volatility_analysis JSONB,
            quality_calibration JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Signal quality metrics table created")
        except Exception as e:
            logger.error(f"❌ Failed to create signal quality metrics table: {e}")
            raise
    
    async def _create_false_positive_analysis_table(self):
        """Create table for false positive analysis results"""
        query = """
        CREATE TABLE IF NOT EXISTS priority4_false_positive_analysis (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            analysis_period VARCHAR(20),
            false_positive_rate DECIMAL(5,4),
            pattern_failure_rate JSONB,
            regime_failure_rate JSONB,
            time_based_failure_rate JSONB,
            confidence_correlation DECIMAL(5,4),
            volume_correlation DECIMAL(5,4),
            pattern_correlation DECIMAL(5,4),
            regime_correlation DECIMAL(5,4),
            recommendations JSONB,
            risk_adjustments JSONB,
            threshold_adjustments JSONB,
            performance_impact JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ False positive analysis table created")
        except Exception as e:
            logger.error(f"❌ Failed to create false positive analysis table: {e}")
            raise
    
    async def _create_market_regime_filtering_table(self):
        """Create table for market regime filtering performance"""
        query = """
        CREATE TABLE IF NOT EXISTS priority4_market_regime_filtering (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            regime_type VARCHAR(20) NOT NULL,
            regime_confidence DECIMAL(5,4),
            signal_count INTEGER,
            approved_signals INTEGER,
            rejected_signals INTEGER,
            avg_quality_score DECIMAL(5,4),
            regime_alignment_score DECIMAL(5,4),
            filtering_performance DECIMAL(5,4),
            regime_specific_metrics JSONB,
            adaptation_parameters JSONB,
            performance_trend JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Market regime filtering table created")
        except Exception as e:
            logger.error(f"❌ Failed to create market regime filtering table: {e}")
            raise
    
    async def _create_adaptive_thresholds_table(self):
        """Create table for adaptive threshold management"""
        query = """
        CREATE TABLE IF NOT EXISTS priority4_adaptive_thresholds (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            threshold_type VARCHAR(50) NOT NULL,
            old_value DECIMAL(5,4),
            new_value DECIMAL(5,4),
            adjustment_reason VARCHAR(200),
            performance_impact DECIMAL(5,4),
            false_positive_rate_before DECIMAL(5,4),
            false_positive_rate_after DECIMAL(5,4),
            signal_count_before INTEGER,
            signal_count_after INTEGER,
            quality_score_before DECIMAL(5,4),
            quality_score_after DECIMAL(5,4),
            adjustment_metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Adaptive thresholds table created")
        except Exception as e:
            logger.error(f"❌ Failed to create adaptive thresholds table: {e}")
            raise
    
    async def _create_validation_performance_table(self):
        """Create table for validation performance tracking"""
        query = """
        CREATE TABLE IF NOT EXISTS priority4_validation_performance (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            period_type VARCHAR(20) NOT NULL,
            total_signals INTEGER,
            approved_signals INTEGER,
            rejected_signals INTEGER,
            conditional_approvals INTEGER,
            avg_quality_score DECIMAL(5,4),
            false_positive_rate DECIMAL(5,4),
            true_positive_rate DECIMAL(5,4),
            precision_score DECIMAL(5,4),
            recall_score DECIMAL(5,4),
            f1_score DECIMAL(5,4),
            processing_time_avg DECIMAL(10,3),
            memory_usage_avg DECIMAL(8,3),
            cpu_usage_avg DECIMAL(5,2),
            performance_metrics JSONB,
            system_health JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Validation performance table created")
        except Exception as e:
            logger.error(f"❌ Failed to create validation performance table: {e}")
            raise
    
    async def _apply_timescaledb_features(self):
        """Apply TimescaleDB specific features to tables"""
        tables = [
            'priority4_advanced_signal_validation',
            'priority4_signal_quality_metrics',
            'priority4_false_positive_analysis',
            'priority4_market_regime_filtering',
            'priority4_adaptive_thresholds',
            'priority4_validation_performance'
        ]
        
        for table in tables:
            try:
                # Convert to hypertable
                hypertable_query = f"SELECT create_hypertable('{table}', 'timestamp', if_not_exists => TRUE);"
                
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor() as cur:
                        cur.execute(hypertable_query)
                        conn.commit()
                        logger.info(f"✅ Converted {table} to hypertable")
                
                # Apply compression policy
                compression_query = f"""
                SELECT add_compression_policy('{table}', INTERVAL '7 days', if_not_exists => TRUE);
                """
                
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor() as cur:
                        cur.execute(compression_query)
                        conn.commit()
                        logger.info(f"✅ Applied compression policy to {table}")
                
                # Apply retention policy
                retention_query = f"""
                SELECT add_retention_policy('{table}', INTERVAL '90 days', if_not_exists => TRUE);
                """
                
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor() as cur:
                        cur.execute(retention_query)
                        conn.commit()
                        logger.info(f"✅ Applied retention policy to {table}")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not apply TimescaleDB features to {table}: {e}")
                # Continue with other tables even if one fails
    
    async def insert_sample_data(self):
        """Insert sample data for testing"""
        logger.info("Inserting sample data for Priority 4 tables...")
        
        try:
            # Sample data for advanced signal validation
            validation_data = {
                'symbol': 'BTCUSDT',
                'signal_type': 'buy',
                'strategy_name': 'trend_following',
                'validation_result': 'approved',
                'overall_quality_score': 0.85,
                'confidence_score': 0.78,
                'volume_confirmation_score': 0.82,
                'pattern_strength_score': 0.79,
                'regime_alignment_score': 0.88,
                'timeframe_confluence_score': 0.75,
                'risk_score': 0.35,
                'volatility_adjustment': 0.02,
                'quality_level': 'high',
                'warnings': json.dumps(['Minor volume confirmation warning']),
                'recommendations': json.dumps(['Consider full position size', 'Monitor volume trends']),
                'validation_details': json.dumps({
                    'feature_summary': {'total_features': 45, 'priority2_features': 12},
                    'quality_breakdown': {'confidence_weight': 0.30, 'volume_weight': 0.20}
                })
            }
            
            query = """
            INSERT INTO priority4_advanced_signal_validation 
            (symbol, signal_type, strategy_name, validation_result, overall_quality_score,
             confidence_score, volume_confirmation_score, pattern_strength_score,
             regime_alignment_score, timeframe_confluence_score, risk_score,
             volatility_adjustment, quality_level, warnings, recommendations, validation_details)
            VALUES (%(symbol)s, %(signal_type)s, %(strategy_name)s, %(validation_result)s,
                    %(overall_quality_score)s, %(confidence_score)s, %(volume_confirmation_score)s,
                    %(pattern_strength_score)s, %(regime_alignment_score)s, %(timeframe_confluence_score)s,
                    %(risk_score)s, %(volatility_adjustment)s, %(quality_level)s,
                    %(warnings)s, %(recommendations)s, %(validation_details)s);
            """
            
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, validation_data)
                    conn.commit()
                    logger.info("✅ Sample data inserted successfully")
                    
        except Exception as e:
            logger.error(f"❌ Failed to insert sample data: {e}")
            raise

async def main():
    """Main migration execution"""
    migration = Priority4DatabaseMigration()
    
    try:
        await migration.run_migration()
        await migration.insert_sample_data()
        logger.info("🎉 Priority 4 migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Priority 4 migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
