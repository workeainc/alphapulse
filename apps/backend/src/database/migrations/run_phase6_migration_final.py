#!/usr/bin/env python3
"""
Phase 6: Advanced ML Model Integration - Final Migration Script
Handles database connection issues gracefully and provides clear error messages
"""

import asyncio
import asyncpg
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase6MigrationRunner:
    """Phase 6 Database Migration Runner with Error Handling"""
    
    def __init__(self):
        # Fix connection string format for asyncpg
        self.connection_params = {
            'host': 'localhost',
            'port': 5432,
            'user': 'alpha_emon',
            'password': 'Emon_@17711',
            'database': 'alphapulse'
        }
        self.conn = None
        self.migration_results = {}
    
    async def test_database_connection(self):
        """Test database connection before running migration"""
        try:
            logger.info("🔄 Testing database connection...")
            self.conn = await asyncpg.connect(**self.connection_params)
            logger.info("✅ Database connection successful")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.error("💡 Please ensure:")
            logger.error("   - PostgreSQL server is running")
            logger.error("   - Database 'alphapulse' exists")
            logger.error("   - User 'alpha_emon' has proper permissions")
            logger.error("   - Connection parameters are correct")
            return False
    
    async def run_migration(self):
        """Run the complete Phase 6 migration"""
        try:
            logger.info("🚀 Starting Phase 6: Advanced ML Model Integration Migration")
            logger.info("=" * 60)
            
            # Test connection first
            if not await self.test_database_connection():
                logger.error("❌ Cannot proceed without database connection")
                return False
            
            # Run all migration steps
            await self.create_ml_tables()
            await self.enhance_signals_table()
            await self.create_indexes()
            await self.insert_default_data()
            await self.verify_migration()
            
            logger.info("✅ Phase 6 migration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
        finally:
            if self.conn:
                await self.conn.close()
                logger.info("✅ Database connection closed")
    
    async def create_ml_tables(self):
        """Create all ML-related tables"""
        logger.info("📋 Creating ML-related tables...")
        
        # 1. ML Model Performance Tracking Table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_performance (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                accuracy FLOAT,
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                auc_score FLOAT,
                latency_ms FLOAT,
                throughput_per_sec FLOAT,
                memory_usage_mb FLOAT,
                gpu_usage_percent FLOAT,
                prediction_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                drift_score FLOAT,
                health_score FLOAT,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created ml_model_performance table")
        
        # 2. Model Health Monitoring Table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_health_monitoring (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                health_status VARCHAR(20) DEFAULT 'healthy',
                overall_health_score FLOAT,
                feature_drift_score FLOAT,
                concept_drift_score FLOAT,
                performance_drift_score FLOAT,
                data_quality_score FLOAT,
                model_stability_score FLOAT,
                alert_level VARCHAR(20) DEFAULT 'none',
                alert_message TEXT,
                recommendations JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created model_health_monitoring table")
        
        # 3. Advanced ML Integration Results Table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS advanced_ml_integration_results (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                catboost_prediction FLOAT,
                catboost_confidence FLOAT,
                drift_detection_score FLOAT,
                chart_pattern_score FLOAT,
                candlestick_pattern_score FLOAT,
                volume_analysis_score FLOAT,
                ensemble_prediction FLOAT,
                ensemble_confidence FLOAT,
                ml_health_score FLOAT,
                processing_time_ms FLOAT,
                model_versions JSONB,
                feature_importance JSONB,
                prediction_explanations JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created advanced_ml_integration_results table")
        
        # 4. ML Model Registry Table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_model_registry (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL UNIQUE,
                model_type VARCHAR(50) NOT NULL,
                model_path VARCHAR(500) NOT NULL,
                onnx_path VARCHAR(500),
                version VARCHAR(20) NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                accuracy FLOAT,
                training_date TIMESTAMPTZ,
                last_updated TIMESTAMPTZ DEFAULT NOW(),
                model_size_mb FLOAT,
                input_features JSONB,
                output_classes JSONB,
                hyperparameters JSONB,
                performance_metrics JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created ml_model_registry table")
        
        # 5. Model Training History Table
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_training_history (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(100) NOT NULL,
                training_run_id VARCHAR(100) NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                training_status VARCHAR(20) DEFAULT 'completed',
                training_duration_seconds FLOAT,
                training_samples INTEGER,
                validation_samples INTEGER,
                test_samples INTEGER,
                initial_accuracy FLOAT,
                final_accuracy FLOAT,
                accuracy_improvement FLOAT,
                loss_history JSONB,
                metrics_history JSONB,
                hyperparameters JSONB,
                feature_importance JSONB,
                training_logs TEXT,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        logger.info("✅ Created model_training_history table")
    
    async def enhance_signals_table(self):
        """Add ML-related columns to signals table"""
        logger.info("🔧 Enhancing signals table with ML columns...")
        
        try:
            await self.conn.execute("""
                ALTER TABLE signals 
                ADD COLUMN IF NOT EXISTS ml_model_confidence FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS ml_health_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS catboost_prediction FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS drift_detection_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS chart_pattern_ml_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS candlestick_ml_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS volume_ml_score FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS ml_processing_time_ms FLOAT DEFAULT 0.0,
                ADD COLUMN IF NOT EXISTS ml_model_versions JSONB,
                ADD COLUMN IF NOT EXISTS ml_prediction_explanations JSONB
            """)
            logger.info("✅ Added ML columns to signals table")
        except Exception as e:
            logger.warning(f"⚠️ Could not add ML columns to signals table: {e}")
    
    async def create_indexes(self):
        """Create performance indexes"""
        logger.info("📊 Creating performance indexes...")
        
        indexes = [
            ("idx_ml_model_performance_model_timestamp", 
             "ml_model_performance(model_name, timestamp DESC)"),
            ("idx_ml_model_performance_symbol_timeframe", 
             "ml_model_performance(symbol, timeframe, timestamp DESC)"),
            ("idx_model_health_monitoring_model_timestamp", 
             "model_health_monitoring(model_name, timestamp DESC)"),
            ("idx_advanced_ml_integration_results_signal_id", 
             "advanced_ml_integration_results(signal_id)"),
            ("idx_advanced_ml_integration_results_symbol_timeframe", 
             "advanced_ml_integration_results(symbol, timeframe, timestamp DESC)"),
            ("idx_ml_model_registry_model_type_status", 
             "ml_model_registry(model_type, status)"),
            ("idx_model_training_history_model_timestamp", 
             "model_training_history(model_name, timestamp DESC)")
        ]
        
        for index_name, index_def in indexes:
            try:
                await self.conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {index_def}")
                logger.info(f"✅ Created index: {index_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not create index {index_name}: {e}")
    
    async def insert_default_data(self):
        """Insert default ML model registry entries"""
        logger.info("📝 Inserting default ML model registry entries...")
        
        try:
            await self.conn.execute("""
                INSERT INTO ml_model_registry (model_name, model_type, model_path, onnx_path, version, status, metadata)
                VALUES 
                ('catboost_nightly_incremental', 'catboost', 'models/catboost_nightly_incremental_20250814_151525.model', 'models/onnx/catboost_nightly_incremental.onnx', '1.0.0', 'active', '{"description": "Nightly incremental CatBoost model", "training_frequency": "daily"}'),
                ('xgboost_weekly_quick', 'xgboost', 'models/xgboost_weekly_quick_20250814_151525.model', 'models/onnx/xgboost_weekly_quick.onnx', '1.0.0', 'active', '{"description": "Weekly quick XGBoost model", "training_frequency": "weekly"}'),
                ('lightgbm_monthly_full', 'lightgbm', 'models/lightgbm_monthly_full_20250814_151525.model', 'models/onnx/lightgbm_monthly_full.onnx', '1.0.0', 'active', '{"description": "Monthly full LightGBM model", "training_frequency": "monthly"}'),
                ('pattern_detection_model', 'onnx', 'models/pattern_detection_model.pkl', 'models/onnx/pattern_detection_model.onnx', '1.0.0', 'active', '{"description": "Pattern detection model", "type": "classification"}'),
                ('regime_classification_model', 'onnx', 'models/regime_classification_model.pkl', 'models/onnx/regime_classification_model.onnx', '1.0.0', 'active', '{"description": "Market regime classification model", "type": "classification"}'),
                ('volume_analysis_model', 'onnx', 'models/volume_analysis_model.pkl', 'models/onnx/volume_analysis_model.onnx', '1.0.0', 'active', '{"description": "Volume analysis model", "type": "regression"}'),
                ('sentiment_analysis_model', 'onnx', 'models/sentiment_analysis_model.pkl', 'models/onnx/sentiment_analysis_model.onnx', '1.0.0', 'active', '{"description": "Sentiment analysis model", "type": "classification"}'),
                ('technical_analysis_model', 'onnx', 'models/technical_analysis_model.pkl', 'models/onnx/technical_analysis_model.onnx', '1.0.0', 'active', '{"description": "Technical analysis model", "type": "classification"}')
                ON CONFLICT (model_name) DO NOTHING
            """)
            logger.info("✅ Inserted default ML model registry entries")
        except Exception as e:
            logger.warning(f"⚠️ Could not insert default data: {e}")
    
    async def verify_migration(self):
        """Verify that all tables and columns were created successfully"""
        logger.info("🔍 Verifying migration...")
        
        # Check tables
        tables_to_check = [
            'ml_model_performance',
            'model_health_monitoring', 
            'advanced_ml_integration_results',
            'ml_model_registry',
            'model_training_history'
        ]
        
        existing_tables = []
        missing_tables = []
        
        for table in tables_to_check:
            exists = await self.conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = '{table}'
                )
            """)
            
            if exists:
                existing_tables.append(table)
                logger.info(f"✅ Verified table: {table}")
            else:
                missing_tables.append(table)
                logger.error(f"❌ Missing table: {table}")
        
        # Check ML columns in signals table
        ml_columns = [
            'ml_model_confidence',
            'ml_health_score', 
            'catboost_prediction',
            'drift_detection_score',
            'chart_pattern_ml_score',
            'candlestick_ml_score',
            'volume_ml_score'
        ]
        
        existing_columns = []
        missing_columns = []
        
        for column in ml_columns:
            exists = await self.conn.fetchval(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'signals' 
                    AND column_name = '{column}'
                )
            """)
            
            if exists:
                existing_columns.append(column)
                logger.info(f"✅ Verified column: {column}")
            else:
                missing_columns.append(column)
                logger.error(f"❌ Missing column: {column}")
        
        # Store results
        self.migration_results = {
            'existing_tables': existing_tables,
            'missing_tables': missing_tables,
            'existing_columns': existing_columns,
            'missing_columns': missing_columns,
            'success': len(missing_tables) == 0 and len(missing_columns) == 0
        }
        
        if self.migration_results['success']:
            logger.info("🎉 Migration verification successful!")
        else:
            logger.warning("⚠️ Migration verification has issues")
    
    def print_summary(self):
        """Print migration summary"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Phase 6 Migration Summary")
        logger.info("=" * 60)
        
        if self.migration_results.get('success', False):
            logger.info("✅ Migration Status: SUCCESS")
        else:
            logger.info("⚠️ Migration Status: PARTIAL SUCCESS")
        
        logger.info(f"Tables Created: {len(self.migration_results.get('existing_tables', []))}/5")
        logger.info(f"Columns Added: {len(self.migration_results.get('existing_columns', []))}/7")
        
        if self.migration_results.get('missing_tables'):
            logger.info(f"Missing Tables: {self.migration_results['missing_tables']}")
        
        if self.migration_results.get('missing_columns'):
            logger.info(f"Missing Columns: {self.migration_results['missing_columns']}")
        
        logger.info("=" * 60)

async def main():
    """Main migration function"""
    runner = Phase6MigrationRunner()
    
    try:
        success = await runner.run_migration()
        runner.print_summary()
        
        if success:
            logger.info("🎉 Phase 6 migration completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Phase 6 migration failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
