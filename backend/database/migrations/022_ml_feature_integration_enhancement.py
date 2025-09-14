#!/usr/bin/env python3
"""
Migration 022: ML Feature Integration Enhancement
Enhances existing tables with ML features and creates new ML-specific tables
"""

import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# revision identifiers
revision = '022'
down_revision = '021'
branch_labels = None
depends_on = None

def upgrade():
    """
    Upgrade database schema to support ML feature integration
    """
    logger.info("Starting ML Feature Integration Enhancement migration...")

    # 1. Enhance existing order_book_snapshots table with ML features
    logger.info("Enhancing order_book_snapshots table with ML features...")
    op.add_column('order_book_snapshots', sa.Column('ml_features', postgresql.JSONB, nullable=True))
    op.add_column('order_book_snapshots', sa.Column('feature_timestamp', sa.DateTime, nullable=True))
    op.add_column('order_book_snapshots', sa.Column('feature_hash', sa.String(64), nullable=True))
    
    # 2. Enhance existing liquidation_events table with prediction metadata
    logger.info("Enhancing liquidation_events table with prediction metadata...")
    op.add_column('liquidation_events', sa.Column('prediction_probability', sa.DECIMAL(5, 4), nullable=True))
    op.add_column('liquidation_events', sa.Column('prediction_confidence', sa.DECIMAL(5, 4), nullable=True))
    op.add_column('liquidation_events', sa.Column('prediction_model_version', sa.String(50), nullable=True))
    op.add_column('liquidation_events', sa.Column('prediction_features', postgresql.JSONB, nullable=True))
    
    # 3. Enhance existing comprehensive_analysis table with ML predictions
    logger.info("Enhancing comprehensive_analysis table with ML predictions...")
    op.add_column('comprehensive_analysis', sa.Column('ml_predictions', postgresql.JSONB, nullable=True))
    op.add_column('comprehensive_analysis', sa.Column('ensemble_confidence', sa.DECIMAL(5, 4), nullable=True))
    op.add_column('comprehensive_analysis', sa.Column('model_ensemble_weights', postgresql.JSONB, nullable=True))
    op.add_column('comprehensive_analysis', sa.Column('prediction_accuracy_score', sa.DECIMAL(5, 4), nullable=True))
    
    # 4. Create ML Model Versions table
    logger.info("Creating ml_model_versions table...")
    op.create_table('ml_model_versions',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('version', sa.Integer, nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),  # 'lightgbm', 'lstm', 'transformer'
        sa.Column('status', sa.String(20), default='training'),  # 'training', 'active', 'deprecated', 'failed'
        sa.Column('accuracy_score', sa.DECIMAL(5, 4), nullable=True),
        sa.Column('validation_score', sa.DECIMAL(5, 4), nullable=True),
        sa.Column('feature_importance', postgresql.JSONB, nullable=True),
        sa.Column('hyperparameters', postgresql.JSONB, nullable=True),
        sa.Column('training_data_hash', sa.String(64), nullable=True),
        sa.Column('model_file_path', sa.String(500), nullable=True),
        sa.Column('training_started', sa.DateTime, nullable=True),
        sa.Column('training_completed', sa.DateTime, nullable=True),
        sa.Column('deployed_at', sa.DateTime, nullable=True),
        sa.Column('created_by', sa.String(100), default='auto_trainer'),
        sa.Column('deployment_environment', sa.String(50), default='production'),
        sa.Column('model_metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime, default=sa.func.current_timestamp()),
        sa.Column('updated_at', sa.DateTime, default=sa.func.current_timestamp(), onupdate=sa.func.current_timestamp())
    )
    
    # 5. Create ML Training Jobs table
    logger.info("Creating ml_training_jobs table...")
    op.create_table('ml_training_jobs',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('job_id', sa.String(100), unique=True, nullable=False),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), default='pending'),  # 'pending', 'running', 'completed', 'failed'
        sa.Column('training_config', postgresql.JSONB, nullable=True),
        sa.Column('data_range_start', sa.DateTime, nullable=True),
        sa.Column('data_range_end', sa.DateTime, nullable=True),
        sa.Column('training_samples', sa.Integer, nullable=True),
        sa.Column('validation_samples', sa.Integer, nullable=True),
        sa.Column('feature_count', sa.Integer, nullable=True),
        sa.Column('training_duration_seconds', sa.Integer, nullable=True),
        sa.Column('final_accuracy', sa.DECIMAL(5, 4), nullable=True),
        sa.Column('validation_accuracy', sa.DECIMAL(5, 4), nullable=True),
        sa.Column('training_logs', sa.Text, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime, default=sa.func.current_timestamp()),
        sa.Column('started_at', sa.DateTime, nullable=True),
        sa.Column('completed_at', sa.DateTime, nullable=True)
    )
    
    # 6. Create ML Feature Store table
    logger.info("Creating ml_feature_store table...")
    op.create_table('ml_feature_store',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('feature_name', sa.String(100), nullable=False),
        sa.Column('feature_value', sa.DECIMAL(20, 8), nullable=True),
        sa.Column('feature_metadata', postgresql.JSONB, nullable=True),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('feature_group', sa.String(50), nullable=True),  # 'order_book', 'liquidation', 'market_depth'
        sa.Column('data_source', sa.String(50), nullable=True),    # 'binance', 'bybit', 'okx'
        sa.Column('feature_quality_score', sa.DECIMAL(5, 4), nullable=True),
        sa.Column('is_training_data', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.current_timestamp()),
        sa.PrimaryKeyConstraint('timestamp', 'id')  # Composite primary key for TimescaleDB
    )
    
    # 7. Create ML Drift Monitoring table
    logger.info("Creating ml_drift_monitoring table...")
    op.create_table('ml_drift_monitoring',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('model_version', sa.Integer, nullable=False),
        sa.Column('drift_type', sa.String(20), nullable=False),  # 'data_drift', 'concept_drift', 'performance_drift'
        sa.Column('drift_score', sa.DECIMAL(5, 4), nullable=False),
        sa.Column('drift_threshold', sa.DECIMAL(5, 4), nullable=False),
        sa.Column('drift_detected', sa.Boolean, default=False),
        sa.Column('feature_drifts', postgresql.JSONB, nullable=True),
        sa.Column('monitoring_window_hours', sa.Integer, default=24),
        sa.Column('samples_analyzed', sa.Integer, nullable=True),
        sa.Column('drift_metadata', postgresql.JSONB, nullable=True),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.current_timestamp()),
        sa.PrimaryKeyConstraint('timestamp', 'id')  # Composite primary key for TimescaleDB
    )
    
    # 8. Create ML Performance Metrics table
    logger.info("Creating ml_performance_metrics table...")
    op.create_table('ml_performance_metrics',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('model_name', sa.String(100), nullable=False),
        sa.Column('model_version', sa.Integer, nullable=False),
        sa.Column('metric_name', sa.String(50), nullable=False),  # 'accuracy', 'precision', 'recall', 'f1', 'auc'
        sa.Column('metric_value', sa.DECIMAL(10, 6), nullable=False),
        sa.Column('evaluation_type', sa.String(20), nullable=False),  # 'training', 'validation', 'production'
        sa.Column('data_split', sa.String(20), nullable=True),    # 'train', 'validation', 'test'
        sa.Column('evaluation_samples', sa.Integer, nullable=True),
        sa.Column('confidence_interval_lower', sa.DECIMAL(10, 6), nullable=True),
        sa.Column('confidence_interval_upper', sa.DECIMAL(10, 6), nullable=True),
        sa.Column('evaluation_metadata', postgresql.JSONB, nullable=True),
        sa.Column('timestamp', sa.DateTime, nullable=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.current_timestamp()),
        sa.PrimaryKeyConstraint('timestamp', 'id')  # Composite primary key for TimescaleDB
    )
    
    # 9. Create indexes for performance optimization
    logger.info("Creating performance indexes...")
    
    # ML Model Versions indexes
    op.create_index('idx_ml_model_versions_name_version', 'ml_model_versions', ['model_name', 'version'])
    op.create_index('idx_ml_model_versions_status', 'ml_model_versions', ['status'])
    op.create_index('idx_ml_model_versions_type', 'ml_model_versions', ['model_type'])
    
    # ML Training Jobs indexes
    op.create_index('idx_ml_training_jobs_status', 'ml_training_jobs', ['status'])
    op.create_index('idx_ml_training_jobs_model', 'ml_training_jobs', ['model_name'])
    op.create_index('idx_ml_training_jobs_created', 'ml_training_jobs', ['created_at'])
    
    # ML Feature Store indexes
    op.create_index('idx_ml_feature_store_symbol_timestamp', 'ml_feature_store', ['symbol', 'timestamp'])
    op.create_index('idx_ml_feature_store_feature_name', 'ml_feature_store', ['feature_name'])
    op.create_index('idx_ml_feature_store_group', 'ml_feature_store', ['feature_group'])
    
    # ML Drift Monitoring indexes
    op.create_index('idx_ml_drift_monitoring_model_timestamp', 'ml_drift_monitoring', ['model_name', 'timestamp'])
    op.create_index('idx_ml_drift_monitoring_detected', 'ml_drift_monitoring', ['drift_detected'])
    
    # ML Performance Metrics indexes
    op.create_index('idx_ml_performance_metrics_model_timestamp', 'ml_performance_metrics', ['model_name', 'timestamp'])
    op.create_index('idx_ml_performance_metrics_type', 'ml_performance_metrics', ['evaluation_type'])
    
    # Enhanced table indexes
    op.create_index('idx_order_book_ml_timestamp', 'order_book_snapshots', ['feature_timestamp'])
    op.create_index('idx_liquidation_prediction', 'liquidation_events', ['prediction_probability'])
    op.create_index('idx_comprehensive_ml_confidence', 'comprehensive_analysis', ['ensemble_confidence'])
    
    # 10. Create TimescaleDB hypertables for new time-series tables
    logger.info("Creating TimescaleDB hypertables...")
    try:
        # Create hypertables for new time-series tables
        op.execute("SELECT create_hypertable('ml_feature_store', 'timestamp', if_not_exists => TRUE)")
        op.execute("SELECT create_hypertable('ml_drift_monitoring', 'timestamp', if_not_exists => TRUE)")
        op.execute("SELECT create_hypertable('ml_performance_metrics', 'timestamp', if_not_exists => TRUE)")
        logger.info("TimescaleDB hypertables created successfully")
    except Exception as e:
        logger.warning(f"TimescaleDB hypertable creation failed (normal if TimescaleDB not available): {e}")
    
    # 11. Create unique constraints
    logger.info("Creating unique constraints...")
    op.create_unique_constraint('uq_ml_model_versions_name_version', 'ml_model_versions', ['model_name', 'version'])
    op.create_unique_constraint('uq_ml_training_jobs_job_id', 'ml_training_jobs', ['job_id'])
    
    logger.info("ML Feature Integration Enhancement migration completed successfully!")


def downgrade():
    """
    Downgrade database schema (remove ML feature integration)
    """
    logger.info("Starting ML Feature Integration Enhancement downgrade...")
    
    # Drop unique constraints
    op.drop_constraint('uq_ml_training_jobs_job_id', 'ml_training_jobs')
    op.drop_constraint('uq_ml_model_versions_name_version', 'ml_model_versions')
    
    # Drop indexes
    op.drop_index('idx_comprehensive_ml_confidence')
    op.drop_index('idx_liquidation_prediction')
    op.drop_index('idx_order_book_ml_timestamp')
    op.drop_index('idx_ml_performance_metrics_type')
    op.drop_index('idx_ml_performance_metrics_model_timestamp')
    op.drop_index('idx_ml_drift_monitoring_detected')
    op.drop_index('idx_ml_drift_monitoring_model_timestamp')
    op.drop_index('idx_ml_feature_store_group')
    op.drop_index('idx_ml_feature_store_feature_name')
    op.drop_index('idx_ml_feature_store_symbol_timestamp')
    op.drop_index('idx_ml_training_jobs_created')
    op.drop_index('idx_ml_training_jobs_model')
    op.drop_index('idx_ml_training_jobs_status')
    op.drop_index('idx_ml_model_versions_type')
    op.drop_index('idx_ml_model_versions_status')
    op.drop_index('idx_ml_model_versions_name_version')
    
    # Drop new tables
    op.drop_table('ml_performance_metrics')
    op.drop_table('ml_drift_monitoring')
    op.drop_table('ml_feature_store')
    op.drop_table('ml_training_jobs')
    op.drop_table('ml_model_versions')
    
    # Remove columns from existing tables
    op.drop_column('comprehensive_analysis', 'prediction_accuracy_score')
    op.drop_column('comprehensive_analysis', 'model_ensemble_weights')
    op.drop_column('comprehensive_analysis', 'ensemble_confidence')
    op.drop_column('comprehensive_analysis', 'ml_predictions')
    
    op.drop_column('liquidation_events', 'prediction_features')
    op.drop_column('liquidation_events', 'prediction_model_version')
    op.drop_column('liquidation_events', 'prediction_confidence')
    op.drop_column('liquidation_events', 'prediction_probability')
    
    op.drop_column('order_book_snapshots', 'feature_hash')
    op.drop_column('order_book_snapshots', 'feature_timestamp')
    op.drop_column('order_book_snapshots', 'ml_features')
    
    logger.info("ML Feature Integration Enhancement downgrade completed!")
