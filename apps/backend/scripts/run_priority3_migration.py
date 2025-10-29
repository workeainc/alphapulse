#!/usr/bin/env python3
"""
Database Migration Script for Priority 3: Enhanced Model Accuracy

Creates tables for tracking:
1. Enhanced pattern model performance
2. Market regime model metrics
3. Probability calibration results
4. Ensemble model performance
5. ONNX optimization metrics
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

class Priority3DatabaseMigration:
    """Database migration for Priority 3 Enhanced Model Accuracy"""
    
    def __init__(self, connection_string: str = None):
        """Initialize migration with database connection"""
        if connection_string is None:
            # Default connection string - adjust as needed
            self.connection_string = (
                "postgresql://alphapulse:alphapulse@localhost:5432/alphapulse"
            )
        else:
            self.connection_string = connection_string
        
        logger.info("🚀 Priority 3 Database Migration initialized")
    
    async def run_migration(self):
        """Run the complete Priority 3 migration"""
        logger.info("Starting Priority 3 Enhanced Model Accuracy migration...")
        
        try:
            # Create all tables
            await self._create_enhanced_pattern_models_table()
            await self._create_enhanced_regime_models_table()
            await self._create_probability_calibration_table()
            await self._create_ensemble_performance_table()
            await self._create_onnx_optimization_metrics_table()
            await self._create_model_metadata_table()
            
            # Apply TimescaleDB features
            await self._apply_timescaledb_features()
            
            logger.info("✅ Priority 3 migration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Priority 3 migration failed: {e}")
            raise
    
    async def _create_enhanced_pattern_models_table(self):
        """Create table for enhanced pattern model performance"""
        query = """
        CREATE TABLE IF NOT EXISTS priority3_enhanced_pattern_models (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            pattern_type VARCHAR(20) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            calibration_method VARCHAR(20) NOT NULL,
            roc_auc DECIMAL(5,4),
            brier_score DECIMAL(6,5),
            calibration_error DECIMAL(6,5),
            training_samples INTEGER,
            validation_samples INTEGER,
            training_time_seconds DECIMAL(10,3),
            model_size_mb DECIMAL(8,3),
            feature_importance JSONB,
            hyperparameters JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Enhanced pattern models table created")
        except Exception as e:
            logger.error(f"❌ Failed to create enhanced pattern models table: {e}")
            raise
    
    async def _create_enhanced_regime_models_table(self):
        """Create table for enhanced regime model performance"""
        query = """
        CREATE TABLE IF NOT EXISTS priority3_enhanced_regime_models (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            regime_type VARCHAR(20) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            cv_scores JSONB,
            avg_cv_score DECIMAL(5,4),
            roc_auc DECIMAL(5,4),
            brier_score DECIMAL(6,5),
            calibration_error DECIMAL(6,5),
            training_samples INTEGER,
            validation_samples INTEGER,
            training_time_seconds DECIMAL(10,3),
            model_size_mb DECIMAL(8,3),
            feature_importance JSONB,
            regime_specific_metrics JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Enhanced regime models table created")
        except Exception as e:
            logger.error(f"❌ Failed to create enhanced regime models table: {e}")
            raise
    
    async def _create_probability_calibration_table(self):
        """Create table for probability calibration results"""
        query = """
        CREATE TABLE IF NOT EXISTS priority3_probability_calibration (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            calibration_method VARCHAR(20) NOT NULL,
            original_auc DECIMAL(5,4),
            calibrated_auc DECIMAL(5,4),
            original_brier_score DECIMAL(6,5),
            calibrated_brier_score DECIMAL(6,5),
            calibration_error DECIMAL(6,5),
            reliability_diagram JSONB,
            calibration_parameters JSONB,
            validation_samples INTEGER,
            calibration_time_seconds DECIMAL(10,3),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Probability calibration table created")
        except Exception as e:
            logger.error(f"❌ Failed to create probability calibration table: {e}")
            raise
    
    async def _create_ensemble_performance_table(self):
        """Create table for ensemble model performance"""
        query = """
        CREATE TABLE IF NOT EXISTS priority3_ensemble_performance (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            ensemble_name VARCHAR(100) NOT NULL,
            base_models_count INTEGER,
            pattern_models_count INTEGER,
            regime_models_count INTEGER,
            meta_learner_type VARCHAR(50),
            ensemble_auc DECIMAL(5,4),
            ensemble_brier_score DECIMAL(6,5),
            confidence_score DECIMAL(5,4),
            model_agreement DECIMAL(5,4),
            prediction_variance DECIMAL(6,5),
            base_model_weights JSONB,
            meta_features_count INTEGER,
            training_samples INTEGER,
            training_time_seconds DECIMAL(10,3),
            ensemble_size_mb DECIMAL(8,3),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Ensemble performance table created")
        except Exception as e:
            logger.error(f"❌ Failed to create ensemble performance table: {e}")
            raise
    
    async def _create_onnx_optimization_metrics_table(self):
        """Create table for ONNX optimization metrics"""
        query = """
        CREATE TABLE IF NOT EXISTS priority3_onnx_optimization_metrics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            original_model_size_mb DECIMAL(8,3),
            onnx_model_size_mb DECIMAL(8,3),
            compression_ratio DECIMAL(5,4),
            optimization_level VARCHAR(20),
            execution_provider VARCHAR(20),
            inference_latency_ms DECIMAL(8,3),
            memory_usage_mb DECIMAL(8,3),
            cpu_usage_percent DECIMAL(5,2),
            gpu_usage_percent DECIMAL(5,2),
            optimization_time_seconds DECIMAL(10,3),
            optimization_parameters JSONB,
            performance_improvement DECIMAL(5,2),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ ONNX optimization metrics table created")
        except Exception as e:
            logger.error(f"❌ Failed to create ONNX optimization metrics table: {e}")
            raise
    
    async def _create_model_metadata_table(self):
        """Create table for model metadata and versioning"""
        query = """
        CREATE TABLE IF NOT EXISTS priority3_model_metadata (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            symbol VARCHAR(20) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            model_version VARCHAR(20),
            model_path VARCHAR(500),
            onnx_path VARCHAR(500),
            model_hash VARCHAR(64),
            training_data_hash VARCHAR(64),
            feature_columns JSONB,
            target_column VARCHAR(50),
            data_preprocessing_steps JSONB,
            model_architecture JSONB,
            training_parameters JSONB,
            validation_metrics JSONB,
            deployment_status VARCHAR(20) DEFAULT 'training',
            deployment_timestamp TIMESTAMPTZ,
            model_performance_summary JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    conn.commit()
                    logger.info("✅ Model metadata table created")
        except Exception as e:
            logger.error(f"❌ Failed to create model metadata table: {e}")
            raise
    
    async def _apply_timescaledb_features(self):
        """Apply TimescaleDB specific features to tables"""
        tables = [
            'priority3_enhanced_pattern_models',
            'priority3_enhanced_regime_models',
            'priority3_probability_calibration',
            'priority3_ensemble_performance',
            'priority3_onnx_optimization_metrics',
            'priority3_model_metadata'
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
        logger.info("Inserting sample data for Priority 3 tables...")
        
        try:
            # Sample data for enhanced pattern models
            pattern_data = {
                'symbol': 'BTCUSDT',
                'pattern_type': 'reversal',
                'model_name': 'lightgbm_enhanced',
                'calibration_method': 'isotonic',
                'roc_auc': 0.785,
                'brier_score': 0.234,
                'calibration_error': 0.045,
                'training_samples': 800,
                'validation_samples': 200,
                'training_time_seconds': 45.2,
                'model_size_mb': 12.5,
                'feature_importance': json.dumps({'rsi': 0.15, 'volume_ratio': 0.12}),
                'hyperparameters': json.dumps({'n_estimators': 200, 'learning_rate': 0.05})
            }
            
            query = """
            INSERT INTO priority3_enhanced_pattern_models 
            (symbol, pattern_type, model_name, calibration_method, roc_auc, brier_score, 
             calibration_error, training_samples, validation_samples, training_time_seconds, 
             model_size_mb, feature_importance, hyperparameters)
            VALUES (%(symbol)s, %(pattern_type)s, %(model_name)s, %(calibration_method)s, 
                    %(roc_auc)s, %(brier_score)s, %(calibration_error)s, %(training_samples)s, 
                    %(validation_samples)s, %(training_time_seconds)s, %(model_size_mb)s, 
                    %(feature_importance)s, %(hyperparameters)s);
            """
            
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, pattern_data)
                    conn.commit()
                    logger.info("✅ Sample data inserted successfully")
                    
        except Exception as e:
            logger.error(f"❌ Failed to insert sample data: {e}")
            raise

async def main():
    """Main migration execution"""
    migration = Priority3DatabaseMigration()
    
    try:
        await migration.run_migration()
        await migration.insert_sample_data()
        logger.info("🎉 Priority 3 migration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Priority 3 migration failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
