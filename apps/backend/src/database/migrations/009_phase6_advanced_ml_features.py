"""
Phase 6: Advanced ML Features Database Migration

This migration adds tables for:
- Hyperparameter optimization results
- Model interpretability and SHAP values
- ML experiment tracking
- Advanced feature engineering metadata
- Transformer model configurations
- Ensemble model metadata
"""

import asyncio
import asyncpg
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

async def run_migration():
    """Run the Phase 6 advanced ML features migration"""
    try:
        # Database connection
        conn = await asyncpg.connect(
            host='localhost',
            port=5432,
            user='postgres',
            password='Emon_@17711',
            database='alphapulse'
        )
        
        logger.info("🚀 Starting Phase 6: Advanced ML Features Migration")
        
        # 1. Hyperparameter Optimization Results Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS hyperparameter_optimization (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(100) NOT NULL,
                optimization_id VARCHAR(100) NOT NULL,
                trial_number INTEGER NOT NULL,
                hyperparameters JSONB NOT NULL,
                objective_value FLOAT NOT NULL,
                objective_name VARCHAR(50) NOT NULL,
                optimization_status VARCHAR(20) NOT NULL,
                training_duration_seconds FLOAT,
                validation_metrics JSONB,
                best_trial BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create index for optimization queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hyperopt_model_name 
            ON hyperparameter_optimization(model_name)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hyperopt_timestamp 
            ON hyperparameter_optimization(timestamp)
        """)
        
        # 2. Model Interpretability Table (SHAP, LIME, ELI5)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_interpretability (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(100) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                prediction_id VARCHAR(100) NOT NULL,
                feature_name VARCHAR(100) NOT NULL,
                feature_value FLOAT NOT NULL,
                shap_value FLOAT,
                lime_value FLOAT,
                eli5_value FLOAT,
                feature_importance_rank INTEGER,
                interpretation_type VARCHAR(20) NOT NULL, -- 'shap', 'lime', 'eli5'
                sample_id VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for interpretability queries
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interpretability_model 
            ON model_interpretability(model_name, model_version)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_interpretability_prediction 
            ON model_interpretability(prediction_id)
        """)
        
        # 3. ML Experiment Tracking Table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_experiments (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                experiment_id VARCHAR(100) UNIQUE NOT NULL,
                experiment_name VARCHAR(200) NOT NULL,
                experiment_type VARCHAR(50) NOT NULL, -- 'hyperopt', 'ensemble', 'transformer', 'feature_engineering'
                status VARCHAR(20) NOT NULL, -- 'running', 'completed', 'failed'
                config JSONB NOT NULL,
                metrics JSONB,
                artifacts JSONB, -- Model paths, logs, etc.
                parent_experiment_id VARCHAR(100),
                tags JSONB,
                created_by VARCHAR(100),
                started_at TIMESTAMPTZ DEFAULT NOW(),
                completed_at TIMESTAMPTZ,
                duration_seconds FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for experiment tracking
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_type 
            ON ml_experiments(experiment_type)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiments_status 
            ON ml_experiments(status)
        """)
        
        # 4. Advanced Feature Engineering Metadata
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS advanced_feature_engineering (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                feature_name VARCHAR(100) NOT NULL,
                feature_type VARCHAR(50) NOT NULL, -- 'technical', 'sentiment', 'flow', 'correlation', 'custom'
                feature_category VARCHAR(50) NOT NULL, -- 'price', 'volume', 'sentiment', 'flow', 'correlation'
                feature_description TEXT,
                feature_formula TEXT,
                parameters JSONB,
                importance_score FLOAT,
                correlation_with_target FLOAT,
                feature_drift_score FLOAT,
                is_active BOOLEAN DEFAULT TRUE,
                created_by VARCHAR(100),
                version VARCHAR(20),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for feature engineering
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_engineering_type 
            ON advanced_feature_engineering(feature_type)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_engineering_active 
            ON advanced_feature_engineering(is_active)
        """)
        
        # 5. Transformer Model Configurations
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS transformer_models (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                model_name VARCHAR(100) NOT NULL,
                model_type VARCHAR(50) NOT NULL, -- 'lstm', 'gru', 'transformer', 'bert', 'gpt'
                model_config JSONB NOT NULL,
                tokenizer_config JSONB,
                training_config JSONB,
                model_size_mb FLOAT,
                parameters_count BIGINT,
                max_sequence_length INTEGER,
                vocabulary_size INTEGER,
                embedding_dimension INTEGER,
                num_layers INTEGER,
                num_heads INTEGER,
                dropout_rate FLOAT,
                learning_rate FLOAT,
                batch_size INTEGER,
                epochs_trained INTEGER,
                is_fine_tuned BOOLEAN DEFAULT FALSE,
                base_model VARCHAR(100),
                fine_tuning_config JSONB,
                performance_metrics JSONB,
                model_path VARCHAR(500),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for transformer models
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transformer_models_type 
            ON transformer_models(model_type)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_transformer_models_active 
            ON transformer_models(is_active)
        """)
        
        # 6. Ensemble Model Metadata
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ensemble_models (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                ensemble_name VARCHAR(100) NOT NULL,
                ensemble_type VARCHAR(50) NOT NULL, -- 'voting', 'stacking', 'blending', 'bagging'
                base_models JSONB NOT NULL, -- List of base model names and weights
                ensemble_config JSONB NOT NULL,
                weighting_strategy VARCHAR(50), -- 'equal', 'performance', 'dynamic'
                base_model_weights JSONB,
                ensemble_performance JSONB,
                is_active BOOLEAN DEFAULT TRUE,
                created_by VARCHAR(100),
                version VARCHAR(20),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for ensemble models
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ensemble_models_type 
            ON ensemble_models(ensemble_type)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ensemble_models_active 
            ON ensemble_models(is_active)
        """)
        
        # 7. Feature Selection and Importance Tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_selection_history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                selection_id VARCHAR(100) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                selection_method VARCHAR(50) NOT NULL, -- 'mutual_info', 'chi2', 'f_regression', 'recursive', 'lasso'
                selected_features JSONB NOT NULL,
                feature_scores JSONB,
                selection_threshold FLOAT,
                total_features INTEGER,
                selected_count INTEGER,
                performance_impact JSONB,
                is_active BOOLEAN DEFAULT TRUE,
                created_by VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for feature selection
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_selection_model 
            ON feature_selection_history(model_name)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feature_selection_method 
            ON feature_selection_history(selection_method)
        """)
        
        # 8. Model Performance Comparison
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance_comparison (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                comparison_id VARCHAR(100) NOT NULL,
                model_names JSONB NOT NULL, -- List of model names being compared
                comparison_metrics JSONB NOT NULL,
                test_period_start TIMESTAMPTZ,
                test_period_end TIMESTAMPTZ,
                dataset_size INTEGER,
                comparison_method VARCHAR(50), -- 'backtest', 'cross_validation', 'holdout'
                winner_model VARCHAR(100),
                statistical_significance FLOAT,
                confidence_interval JSONB,
                created_by VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Create indexes for performance comparison
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_comparison_winner 
            ON model_performance_comparison(winner_model)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance_comparison_timestamp 
            ON model_performance_comparison(timestamp)
        """)
        
        # Insert default configurations
        await _insert_default_configurations(conn)
        
        logger.info("✅ Phase 6: Advanced ML Features Migration completed successfully")
        
        # Close connection
        await conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 6 migration: {e}")
        raise

async def _insert_default_configurations(conn):
    """Insert default configurations for Phase 6 features"""
    
    # Default hyperparameter optimization config
    await conn.execute("""
        INSERT INTO ml_experiments (
            experiment_id, experiment_name, experiment_type, status, config, created_by
        ) VALUES (
            'default_hyperopt',
            'Default Hyperparameter Optimization',
            'hyperopt',
            'completed',
            $1,
            'system'
        ) ON CONFLICT (experiment_id) DO NOTHING
    """, json.dumps({
        'optimizer': 'optuna',
        'n_trials': 100,
        'timeout': 3600,
        'objective': 'minimize_rmse'
    }))
    
    # Default transformer model config
    await conn.execute("""
        INSERT INTO transformer_models (
            model_name, model_type, model_config, is_active
        ) VALUES (
            'default_lstm',
            'lstm',
            $1,
            TRUE
        ) ON CONFLICT DO NOTHING
    """, json.dumps({
        'layers': [64, 32],
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'return_sequences': False
    }))
    
    # Default ensemble config
    await conn.execute("""
        INSERT INTO ensemble_models (
            ensemble_name, ensemble_type, base_models, ensemble_config, weighting_strategy
        ) VALUES (
            'default_ensemble',
            'voting',
            $1,
            $2,
            'performance'
        ) ON CONFLICT DO NOTHING
    """, 
    json.dumps(['lightgbm', 'xgboost', 'catboost']),
    json.dumps({
        'voting_method': 'soft',
        'weights': [0.4, 0.3, 0.3]
    }))

if __name__ == "__main__":
    asyncio.run(run_migration())
