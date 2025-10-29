"""Phase 4B: ML Retraining & Self-Learning Framework

Revision ID: 011_phase4b_ml_retraining
Revises: 010_phase4a_enhancements
Create Date: 2024-01-20 10:00:00.000000

Description:
Phase 4B implementation for ML retraining and self-learning framework:
1. Pattern performance history tracking
2. Model versioning and rollback capabilities
3. Retraining event logging
4. Drift metrics monitoring
5. Regime-specific performance tracking
6. Online learning capabilities
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic
revision = '011_phase4b_ml_retraining'
down_revision = '010_phase4a_enhancements'
branch_labels = None
depends_on = None

def upgrade():
    """Create Phase 4B tables for ML retraining and self-learning"""
    
    # Create pattern performance history table
    op.create_table(
        'pattern_performance_history',
        sa.Column('history_id', sa.BigInteger(), nullable=False),
        sa.Column('pattern_id', sa.String(50), nullable=False),
        sa.Column('symbol', sa.String(20), nullable=False),
        sa.Column('timeframe', sa.String(10), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=False),
        sa.Column('detection_timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('outcome_timestamp', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('outcome_type', sa.String(20), nullable=True),  # success, failure, partial
        sa.Column('profit_loss', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('profit_loss_pct', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('holding_period_hours', sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('max_profit', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('market_regime', sa.String(20), nullable=True),  # bull, bear, sideways, crash
        sa.Column('confidence_at_detection', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('volume_confirmation', sa.Boolean(), nullable=True),
        sa.Column('noise_filter_passed', sa.Boolean(), nullable=True),
        sa.Column('validation_passed', sa.Boolean(), nullable=True),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('detection_method', sa.String(20), nullable=True),  # talib, ml, hybrid
        sa.Column('performance_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('history_id')
    )
    
    # Create model versions table
    op.create_table(
        'model_versions',
        sa.Column('version_id', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),  # pattern_detector, quality_scorer, ensemble
        sa.Column('version_name', sa.String(100), nullable=False),
        sa.Column('model_path', sa.String(500), nullable=False),
        sa.Column('training_data_start', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('training_data_end', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('training_samples', sa.BigInteger(), nullable=True),
        sa.Column('accuracy_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('precision_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('recall_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('f1_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('auc_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('calibration_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('drift_score', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_production', sa.Boolean(), nullable=False, default=False),
        sa.Column('deployment_timestamp', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('rollback_timestamp', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('parent_version', sa.String(50), nullable=True),
        sa.Column('retraining_trigger', sa.String(50), nullable=True),  # scheduled, drift, performance, manual
        sa.Column('model_metadata', JSONB, nullable=True),
        sa.Column('hyperparameters', JSONB, nullable=True),
        sa.Column('feature_importance', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('version_id')
    )
    
    # Create retraining events table
    op.create_table(
        'retraining_events',
        sa.Column('event_id', sa.String(50), nullable=False),
        sa.Column('event_type', sa.String(50), nullable=False),  # scheduled, drift_triggered, performance_triggered, manual
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('trigger_source', sa.String(100), nullable=True),  # drift_monitor, performance_tracker, scheduler
        sa.Column('trigger_metadata', JSONB, nullable=True),
        sa.Column('status', sa.String(20), nullable=False),  # pending, running, completed, failed, cancelled
        sa.Column('start_timestamp', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('end_timestamp', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('old_version', sa.String(50), nullable=True),
        sa.Column('new_version', sa.String(50), nullable=True),
        sa.Column('training_samples_used', sa.BigInteger(), nullable=True),
        sa.Column('accuracy_improvement', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('drift_reduction', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('event_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('event_id')
    )
    
    # Create drift metrics table
    op.create_table(
        'drift_metrics',
        sa.Column('drift_id', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('drift_type', sa.String(30), nullable=False),  # feature, concept, latency, combined
        sa.Column('drift_score', sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column('drift_threshold', sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column('drift_severity', sa.String(20), nullable=False),  # low, medium, high, critical
        sa.Column('detection_timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('window_start', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('window_end', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('samples_analyzed', sa.BigInteger(), nullable=True),
        sa.Column('baseline_metrics', JSONB, nullable=True),
        sa.Column('current_metrics', JSONB, nullable=True),
        sa.Column('drift_metadata', JSONB, nullable=True),
        sa.Column('retraining_triggered', sa.Boolean(), nullable=False, default=False),
        sa.Column('retraining_event_id', sa.String(50), nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('drift_id')
    )
    
    # Create regime performance table
    op.create_table(
        'regime_performance',
        sa.Column('regime_id', sa.String(50), nullable=False),
        sa.Column('regime_type', sa.String(20), nullable=False),  # bull, bear, sideways, crash
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('pattern_type', sa.String(50), nullable=True),
        sa.Column('timeframe', sa.String(10), nullable=True),
        sa.Column('regime_start', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('regime_end', TIMESTAMP(timezone=True), nullable=True),
        sa.Column('total_signals', sa.BigInteger(), nullable=False, default=0),
        sa.Column('successful_signals', sa.BigInteger(), nullable=False, default=0),
        sa.Column('failed_signals', sa.BigInteger(), nullable=False, default=0),
        sa.Column('success_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('avg_profit_loss', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('avg_profit_loss_pct', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('total_profit_loss', sa.Numeric(precision=15, scale=4), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('profit_factor', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('win_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('avg_win', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('avg_loss', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('regime_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('regime_id')
    )
    
    # Create online learning updates table
    op.create_table(
        'online_learning_updates',
        sa.Column('update_id', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('update_type', sa.String(30), nullable=False),  # incremental, batch, reinforcement
        sa.Column('update_timestamp', TIMESTAMP(timezone=True), nullable=False),
        sa.Column('samples_processed', sa.BigInteger(), nullable=False),
        sa.Column('learning_rate', sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column('accuracy_before', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('accuracy_after', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('loss_before', sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column('loss_after', sa.Numeric(precision=8, scale=6), nullable=True),
        sa.Column('update_metadata', JSONB, nullable=True),
        sa.Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('update_id')
    )
    
    # Convert to TimescaleDB hypertables
    op.execute("SELECT create_hypertable('pattern_performance_history', 'detection_timestamp', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('drift_metrics', 'detection_timestamp', if_not_exists => TRUE)")
    op.execute("SELECT create_hypertable('online_learning_updates', 'update_timestamp', if_not_exists => TRUE)")
    
    # Create optimized indexes for ultra-fast queries
    
    # Pattern performance history indexes
    op.create_index('idx_performance_history_pattern_id', 
                   'pattern_performance_history', ['pattern_id'])
    op.create_index('idx_performance_history_symbol_timeframe', 
                   'pattern_performance_history', ['symbol', 'timeframe'])
    op.create_index('idx_performance_history_pattern_type', 
                   'pattern_performance_history', ['pattern_type'])
    op.create_index('idx_performance_history_outcome_type', 
                   'pattern_performance_history', ['outcome_type'])
    op.create_index('idx_performance_history_market_regime', 
                   'pattern_performance_history', ['market_regime'])
    op.create_index('idx_performance_history_model_version', 
                   'pattern_performance_history', ['model_version'])
    
    # Partial indexes for successful patterns
    op.execute("""
        CREATE INDEX idx_performance_history_successful 
        ON pattern_performance_history (symbol, pattern_type, detection_timestamp DESC) 
        WHERE outcome_type = 'success'
    """)
    
    op.execute("""
        CREATE INDEX idx_performance_history_profitable 
        ON pattern_performance_history (symbol, pattern_type, detection_timestamp DESC) 
        WHERE profit_loss > 0
    """)
    
    # Model versions indexes
    op.create_index('idx_model_versions_model_type', 
                   'model_versions', ['model_type'])
    op.create_index('idx_model_versions_active', 
                   'model_versions', ['is_active'])
    op.create_index('idx_model_versions_production', 
                   'model_versions', ['is_production'])
    op.create_index('idx_model_versions_accuracy', 
                   'model_versions', ['accuracy_score'])
    op.create_index('idx_model_versions_deployment', 
                   'model_versions', ['deployment_timestamp'])
    
    # Retraining events indexes
    op.create_index('idx_retraining_events_type', 
                   'retraining_events', ['event_type'])
    op.create_index('idx_retraining_events_status', 
                   'retraining_events', ['status'])
    op.create_index('idx_retraining_events_model_type', 
                   'retraining_events', ['model_type'])
    op.create_index('idx_retraining_events_timestamp', 
                   'retraining_events', ['start_timestamp'])
    
    # Drift metrics indexes
    op.create_index('idx_drift_metrics_model_type', 
                   'drift_metrics', ['model_type'])
    op.create_index('idx_drift_metrics_drift_type', 
                   'drift_metrics', ['drift_type'])
    op.create_index('idx_drift_metrics_severity', 
                   'drift_metrics', ['drift_severity'])
    op.create_index('idx_drift_metrics_score', 
                   'drift_metrics', ['drift_score'])
    op.create_index('idx_drift_metrics_triggered', 
                   'drift_metrics', ['retraining_triggered'])
    
    # Regime performance indexes
    op.create_index('idx_regime_performance_regime_type', 
                   'regime_performance', ['regime_type'])
    op.create_index('idx_regime_performance_model_type', 
                   'regime_performance', ['model_type'])
    op.create_index('idx_regime_performance_pattern_type', 
                   'regime_performance', ['pattern_type'])
    op.create_index('idx_regime_performance_success_rate', 
                   'regime_performance', ['success_rate'])
    
    # Online learning updates indexes
    op.create_index('idx_online_learning_model_type', 
                   'online_learning_updates', ['model_type'])
    op.create_index('idx_online_learning_update_type', 
                   'online_learning_updates', ['update_type'])
    op.create_index('idx_online_learning_timestamp', 
                   'online_learning_updates', ['update_timestamp'])
    
    # GIN indexes for JSONB columns
    op.execute("CREATE INDEX idx_performance_metadata_gin ON pattern_performance_history USING GIN (performance_metadata)")
    op.execute("CREATE INDEX idx_model_metadata_gin ON model_versions USING GIN (model_metadata)")
    op.execute("CREATE INDEX idx_hyperparameters_gin ON model_versions USING GIN (hyperparameters)")
    op.execute("CREATE INDEX idx_feature_importance_gin ON model_versions USING GIN (feature_importance)")
    op.execute("CREATE INDEX idx_trigger_metadata_gin ON retraining_events USING GIN (trigger_metadata)")
    op.execute("CREATE INDEX idx_event_metadata_gin ON retraining_events USING GIN (event_metadata)")
    op.execute("CREATE INDEX idx_baseline_metrics_gin ON drift_metrics USING GIN (baseline_metrics)")
    op.execute("CREATE INDEX idx_current_metrics_gin ON drift_metrics USING GIN (current_metrics)")
    op.execute("CREATE INDEX idx_drift_metadata_gin ON drift_metrics USING GIN (drift_metadata)")
    op.execute("CREATE INDEX idx_regime_metadata_gin ON regime_performance USING GIN (regime_metadata)")
    op.execute("CREATE INDEX idx_update_metadata_gin ON online_learning_updates USING GIN (update_metadata)")
    
    # BRIN indexes for time-based queries
    op.execute("CREATE INDEX idx_performance_history_time_brin ON pattern_performance_history USING BRIN (detection_timestamp)")
    op.execute("CREATE INDEX idx_drift_metrics_time_brin ON drift_metrics USING BRIN (detection_timestamp)")
    op.execute("CREATE INDEX idx_online_learning_time_brin ON online_learning_updates USING BRIN (update_timestamp)")
    
    # Add compression policies
    op.execute("SELECT add_compression_policy('pattern_performance_history', INTERVAL '7 days')")
    op.execute("SELECT add_compression_policy('drift_metrics', INTERVAL '7 days')")
    op.execute("SELECT add_compression_policy('online_learning_updates', INTERVAL '7 days')")
    
    # Add retention policies
    op.execute("SELECT add_retention_policy('pattern_performance_history', INTERVAL '2 years')")
    op.execute("SELECT add_retention_policy('drift_metrics', INTERVAL '1 year')")
    op.execute("SELECT add_retention_policy('online_learning_updates', INTERVAL '1 year')")
    op.execute("SELECT add_retention_policy('retraining_events', INTERVAL '1 year')")
    op.execute("SELECT add_retention_policy('regime_performance', INTERVAL '2 years')")
    
    # Create performance tracking functions
    op.execute("""
        CREATE OR REPLACE FUNCTION calculate_pattern_performance_stats(
            p_symbol VARCHAR,
            p_pattern_type VARCHAR,
            p_timeframe VARCHAR,
            p_start_time TIMESTAMPTZ,
            p_end_time TIMESTAMPTZ
        ) RETURNS JSONB AS $$
        DECLARE
            result JSONB;
        BEGIN
            SELECT jsonb_build_object(
                'total_signals', COUNT(*),
                'successful_signals', COUNT(*) FILTER (WHERE outcome_type = 'success'),
                'failed_signals', COUNT(*) FILTER (WHERE outcome_type = 'failure'),
                'success_rate', ROUND(COUNT(*) FILTER (WHERE outcome_type = 'success')::NUMERIC / COUNT(*), 4),
                'avg_profit_loss', ROUND(AVG(profit_loss), 4),
                'avg_profit_loss_pct', ROUND(AVG(profit_loss_pct), 4),
                'total_profit_loss', ROUND(SUM(profit_loss), 4),
                'max_drawdown', ROUND(MIN(profit_loss), 4),
                'avg_holding_period', ROUND(AVG(holding_period_hours), 2)
            ) INTO result
            FROM pattern_performance_history
            WHERE symbol = p_symbol
                AND pattern_type = p_pattern_type
                AND timeframe = p_timeframe
                AND detection_timestamp BETWEEN p_start_time AND p_end_time;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Create drift detection function
    op.execute("""
        CREATE OR REPLACE FUNCTION detect_model_drift(
            p_model_type VARCHAR,
            p_model_version VARCHAR,
            p_window_hours INTEGER DEFAULT 24
        ) RETURNS JSONB AS $$
        DECLARE
            result JSONB;
            baseline_metrics JSONB;
            current_metrics JSONB;
        BEGIN
            -- Get baseline metrics (last 7 days)
            SELECT jsonb_build_object(
                'avg_accuracy', ROUND(AVG(accuracy_score), 4),
                'avg_drift_score', ROUND(AVG(drift_score), 4),
                'total_detections', COUNT(*)
            ) INTO baseline_metrics
            FROM drift_metrics
            WHERE model_type = p_model_type
                AND model_version = p_model_version
                AND detection_timestamp >= NOW() - INTERVAL '7 days';
            
            -- Get current metrics (last window_hours)
            SELECT jsonb_build_object(
                'avg_accuracy', ROUND(AVG(accuracy_score), 4),
                'avg_drift_score', ROUND(AVG(drift_score), 4),
                'total_detections', COUNT(*)
            ) INTO current_metrics
            FROM drift_metrics
            WHERE model_type = p_model_type
                AND model_version = p_model_version
                AND detection_timestamp >= NOW() - INTERVAL '1 hour' * p_window_hours;
            
            -- Calculate drift
            SELECT jsonb_build_object(
                'model_type', p_model_type,
                'model_version', p_model_version,
                'baseline_metrics', baseline_metrics,
                'current_metrics', current_metrics,
                'drift_detected', (current_metrics->>'avg_drift_score')::NUMERIC > 0.6,
                'drift_score', (current_metrics->>'avg_drift_score')::NUMERIC,
                'detection_timestamp', NOW()
            ) INTO result;
            
            RETURN result;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    logger.info("✅ Phase 4B database migration completed successfully")

def downgrade():
    """Remove Phase 4B tables"""
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS calculate_pattern_performance_stats(VARCHAR, VARCHAR, VARCHAR, TIMESTAMPTZ, TIMESTAMPTZ);")
    op.execute("DROP FUNCTION IF EXISTS detect_model_drift(VARCHAR, VARCHAR, INTEGER);")
    
    # Remove compression and retention policies
    op.execute("SELECT remove_compression_policy('pattern_performance_history')")
    op.execute("SELECT remove_compression_policy('drift_metrics')")
    op.execute("SELECT remove_compression_policy('online_learning_updates')")
    op.execute("SELECT remove_retention_policy('pattern_performance_history')")
    op.execute("SELECT remove_retention_policy('drift_metrics')")
    op.execute("SELECT remove_retention_policy('online_learning_updates')")
    op.execute("SELECT remove_retention_policy('retraining_events')")
    op.execute("SELECT remove_retention_policy('regime_performance')")
    
    # Drop tables
    op.drop_table('online_learning_updates')
    op.drop_table('regime_performance')
    op.drop_table('drift_metrics')
    op.drop_table('retraining_events')
    op.drop_table('model_versions')
    op.drop_table('pattern_performance_history')
    
    logger.info("✅ Phase 4B database migration rolled back successfully")
