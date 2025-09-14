"""Phase 5B: Enhanced Ensemble & Meta-Learner Migration"""

import asyncio
import logging
import asyncpg

logger = logging.getLogger(__name__)

DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def run_phase5b_migration():
    """Run Phase 5B ensemble and meta-learner migration"""
    
    logger.info("🚀 Starting Phase 5B: Enhanced Ensemble & Meta-Learner Migration...")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        logger.info("✅ Connected to database")
        
        # Create ensemble models table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_models (
                model_id SERIAL,
                model_type VARCHAR(50) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                model_path VARCHAR(255) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                training_timestamp TIMESTAMPTZ NOT NULL,
                model_metadata JSONB,
                performance_metrics JSONB,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (model_id, created_at)
            );
        """)
        
        # Create regime performance tracking table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_model_performance (
                performance_id SERIAL,
                model_type VARCHAR(50) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                regime_type VARCHAR(50) NOT NULL,
                auc_score NUMERIC(5,4),
                accuracy_score NUMERIC(5,4),
                precision_score NUMERIC(5,4),
                recall_score NUMERIC(5,4),
                f1_score NUMERIC(5,4),
                sample_count INTEGER,
                confidence_score NUMERIC(5,4),
                evaluation_timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (performance_id, created_at)
            );
        """)
        
        # Create meta-learner configuration table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS meta_learner_config (
                config_id SERIAL,
                meta_learner_type VARCHAR(50) NOT NULL,
                regime_weights JSONB NOT NULL,
                model_selection_strategy VARCHAR(50) NOT NULL,
                top_k_models INTEGER DEFAULT 3,
                confidence_threshold NUMERIC(5,4) DEFAULT 0.7,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (config_id, created_at)
            );
        """)
        
        # Create ensemble predictions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_predictions (
                prediction_id SERIAL,
                ensemble_prediction NUMERIC(5,4) NOT NULL,
                confidence_score NUMERIC(5,4) NOT NULL,
                regime_type VARCHAR(50) NOT NULL,
                regime_confidence NUMERIC(5,4) NOT NULL,
                selected_models JSONB NOT NULL,
                individual_predictions JSONB NOT NULL,
                model_weights JSONB NOT NULL,
                meta_learner_score NUMERIC(5,4) NOT NULL,
                prediction_timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (prediction_id, created_at)
            );
        """)
        
        # Create model training history table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_training_history (
                history_id SERIAL,
                model_type VARCHAR(50) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                training_start_time TIMESTAMPTZ NOT NULL,
                training_end_time TIMESTAMPTZ,
                training_duration_seconds INTEGER,
                sample_count INTEGER,
                feature_count INTEGER,
                hyperparameters JSONB,
                training_metrics JSONB,
                status VARCHAR(20) DEFAULT 'running',
                error_message TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (history_id, created_at)
            );
        """)
        
        # Create hypertables
        try:
            await conn.execute("SELECT create_hypertable('ensemble_models', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            await conn.execute("SELECT create_hypertable('regime_model_performance', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            await conn.execute("SELECT create_hypertable('meta_learner_config', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            await conn.execute("SELECT create_hypertable('ensemble_predictions', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            await conn.execute("SELECT create_hypertable('model_training_history', 'created_at', chunk_time_interval => INTERVAL '1 day');")
            logger.info("✅ Created hypertables")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable creation failed: {e}")
        
        # Create indexes for performance
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_ensemble_models_type ON ensemble_models (model_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_ensemble_models_active ON ensemble_models (is_active);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_performance_model ON regime_model_performance (model_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_regime_performance_regime ON regime_model_performance (regime_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_meta_learner_active ON meta_learner_config (is_active);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_ensemble_predictions_regime ON ensemble_predictions (regime_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_history_status ON model_training_history (status);")
        
        logger.info("✅ Created indexes")
        
        # Insert default meta-learner configuration
        default_config = {
            'meta_learner_type': 'regime_aware_logistic',
            'regime_weights': {
                'bull_trending': {
                    'xgboost': 0.25, 'lightgbm': 0.25, 'gradient_boosting': 0.2,
                    'random_forest': 0.1, 'transformer': 0.1, 'lstm': 0.1
                },
                'bear_trending': {
                    'xgboost': 0.2, 'lightgbm': 0.2, 'gradient_boosting': 0.2,
                    'random_forest': 0.15, 'transformer': 0.15, 'lstm': 0.1
                },
                'sideways': {
                    'xgboost': 0.15, 'lightgbm': 0.15, 'gradient_boosting': 0.15,
                    'random_forest': 0.2, 'transformer': 0.2, 'lstm': 0.15
                },
                'high_volatility': {
                    'xgboost': 0.1, 'lightgbm': 0.1, 'gradient_boosting': 0.1,
                    'random_forest': 0.25, 'transformer': 0.25, 'lstm': 0.2
                },
                'low_volatility': {
                    'xgboost': 0.3, 'lightgbm': 0.3, 'gradient_boosting': 0.2,
                    'random_forest': 0.1, 'transformer': 0.05, 'lstm': 0.05
                },
                'crash': {
                    'xgboost': 0.05, 'lightgbm': 0.05, 'gradient_boosting': 0.1,
                    'random_forest': 0.3, 'transformer': 0.3, 'lstm': 0.2
                }
            },
            'model_selection_strategy': 'top_k_performance',
            'top_k_models': 3,
            'confidence_threshold': 0.7
        }
        
        await conn.execute("""
            INSERT INTO meta_learner_config (
                meta_learner_type, regime_weights, model_selection_strategy,
                top_k_models, confidence_threshold
            ) VALUES (
                $1, $2, $3, $4, $5
            )
        """, default_config['meta_learner_type'], 
            json.dumps(default_config['regime_weights']),
            default_config['model_selection_strategy'],
            default_config['top_k_models'],
            default_config['confidence_threshold']
        )
        
        logger.info("✅ Inserted default meta-learner configuration")
        
        # Create SQL functions for analytics
        await conn.execute("""
            CREATE OR REPLACE FUNCTION get_best_models_per_regime(
                p_regime_type VARCHAR(50),
                p_limit INTEGER DEFAULT 3
            ) RETURNS TABLE (
                model_type VARCHAR(50),
                auc_score NUMERIC(5,4),
                accuracy_score NUMERIC(5,4),
                confidence_score NUMERIC(5,4)
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    rmp.model_type,
                    rmp.auc_score,
                    rmp.accuracy_score,
                    rmp.confidence_score
                FROM regime_model_performance rmp
                WHERE rmp.regime_type = p_regime_type
                ORDER BY rmp.auc_score DESC
                LIMIT p_limit;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        await conn.execute("""
            CREATE OR REPLACE FUNCTION get_ensemble_performance_summary(
                p_days_back INTEGER DEFAULT 30
            ) RETURNS TABLE (
                regime_type VARCHAR(50),
                avg_confidence NUMERIC(5,4),
                avg_meta_learner_score NUMERIC(5,4),
                prediction_count BIGINT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    ep.regime_type,
                    AVG(ep.confidence_score) as avg_confidence,
                    AVG(ep.meta_learner_score) as avg_meta_learner_score,
                    COUNT(*) as prediction_count
                FROM ensemble_predictions ep
                WHERE ep.created_at >= NOW() - INTERVAL '1 day' * p_days_back
                GROUP BY ep.regime_type
                ORDER BY avg_confidence DESC;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        logger.info("✅ Created SQL functions")
        
        await conn.close()
        logger.info("✅ Phase 5B: Enhanced Ensemble & Meta-Learner Migration completed!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise

async def verify_migration():
    """Verify the migration was successful"""
    
    logger.info("🔍 Verifying Phase 5B migration...")
    
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        
        # Check tables exist
        tables = [
            'ensemble_models',
            'regime_model_performance', 
            'meta_learner_config',
            'ensemble_predictions',
            'model_training_history'
        ]
        
        for table in tables:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"✅ Table {table}: {result} rows")
        
        # Check functions exist
        functions = [
            'get_best_models_per_regime',
            'get_ensemble_performance_summary'
        ]
        
        for func in functions:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM pg_proc WHERE proname = '{func}'")
            logger.info(f"✅ Function {func}: {result} found")
        
        # Test function calls
        try:
            await conn.fetch("SELECT * FROM get_best_models_per_regime('sideways', 2)")
            logger.info("✅ get_best_models_per_regime function working")
        except Exception as e:
            logger.warning(f"⚠️ get_best_models_per_regime test failed: {e}")
        
        try:
            await conn.fetch("SELECT * FROM get_ensemble_performance_summary(7)")
            logger.info("✅ get_ensemble_performance_summary function working")
        except Exception as e:
            logger.warning(f"⚠️ get_ensemble_performance_summary test failed: {e}")
        
        await conn.close()
        logger.info("✅ Phase 5B migration verification completed!")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        raise

if __name__ == "__main__":
    import json
    asyncio.run(run_phase5b_migration())
    asyncio.run(verify_migration())
