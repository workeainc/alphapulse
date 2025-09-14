"""
Migration: Phase 4A Enhancements - Multi-Timeframe & Confidence Calibration
Adds multi-timeframe integration and confidence calibration fields to existing tables
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
import logging

logger = logging.getLogger(__name__)

# revision identifiers
revision = '010_phase4a_enhancements'
down_revision = '009_enhanced_pattern_detection'
depends_on = None

def upgrade():
    """
    Upgrade: Add Phase 4A enhancement fields to existing tables
    """
    
    # Add Phase 4A fields to enhanced_candlestick_patterns table
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('timeframe_hierarchy', JSONB, nullable=True))
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('calibrated_confidence', sa.Numeric(precision=4, scale=3), nullable=True))
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('market_regime', sa.String(20), nullable=True))
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('order_flow_metadata', JSONB, nullable=True))
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('explanation_factors', JSONB, nullable=True))
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('multi_timeframe_alignment', sa.Numeric(precision=4, scale=3), nullable=True))
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('timeframe_confirmation_count', sa.Integer(), nullable=True, default=0))
    op.add_column('enhanced_candlestick_patterns', 
                  sa.Column('calibration_confidence_interval', JSONB, nullable=True))
    
    # Add Phase 4A fields to pattern_ml_models table
    op.add_column('pattern_ml_models', 
                  sa.Column('model_type', sa.String(50), nullable=True, default='gradient_boosting'))
    op.add_column('pattern_ml_models', 
                  sa.Column('ensemble_weight', sa.Numeric(precision=3, scale=2), nullable=True, default=1.0))
    op.add_column('pattern_ml_models', 
                  sa.Column('calibration_metrics', JSONB, nullable=True))
    op.add_column('pattern_ml_models', 
                  sa.Column('multi_timeframe_features', JSONB, nullable=True))
    op.add_column('pattern_ml_models', 
                  sa.Column('sequence_model_config', JSONB, nullable=True))
    
    # Add Phase 4A fields to pattern_correlations table
    op.add_column('pattern_correlations', 
                  sa.Column('regime_analysis', JSONB, nullable=True))
    op.add_column('pattern_correlations', 
                  sa.Column('multi_timeframe_correlation', JSONB, nullable=True))
    op.add_column('pattern_correlations', 
                  sa.Column('regime_specific_strength', sa.Numeric(precision=4, scale=3), nullable=True))
    op.add_column('pattern_correlations', 
                  sa.Column('timeframe_hierarchy_impact', JSONB, nullable=True))
    
    # Add Phase 4A fields to pattern_validations table
    op.add_column('pattern_validations', 
                  sa.Column('multi_timeframe_validation', JSONB, nullable=True))
    op.add_column('pattern_validations', 
                  sa.Column('regime_validation_score', sa.Numeric(precision=4, scale=3), nullable=True))
    op.add_column('pattern_validations', 
                  sa.Column('calibration_validation', JSONB, nullable=True))
    
    # Create new indexes for Phase 4A fields
    op.create_index('idx_enhanced_patterns_timeframe_hierarchy', 
                   'enhanced_candlestick_patterns', ['timeframe_hierarchy'])
    op.create_index('idx_enhanced_patterns_calibrated_confidence', 
                   'enhanced_candlestick_patterns', ['calibrated_confidence'])
    op.create_index('idx_enhanced_patterns_market_regime', 
                   'enhanced_candlestick_patterns', ['market_regime'])
    op.create_index('idx_enhanced_patterns_multi_timeframe_alignment', 
                   'enhanced_candlestick_patterns', ['multi_timeframe_alignment'])
    op.create_index('idx_enhanced_patterns_timeframe_confirmation_count', 
                   'enhanced_candlestick_patterns', ['timeframe_confirmation_count'])
    
    op.create_index('idx_ml_models_model_type', 
                   'pattern_ml_models', ['model_type'])
    op.create_index('idx_ml_models_ensemble_weight', 
                   'pattern_ml_models', ['ensemble_weight'])
    
    op.create_index('idx_correlations_regime_analysis', 
                   'pattern_correlations', ['regime_analysis'])
    op.create_index('idx_correlations_regime_specific_strength', 
                   'pattern_correlations', ['regime_specific_strength'])
    
    # Create partial indexes for high-quality signals
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enhanced_patterns_high_calibrated_confidence 
        ON enhanced_candlestick_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE calibrated_confidence >= 0.8
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_enhanced_patterns_strong_multi_timeframe 
        ON enhanced_candlestick_patterns (symbol, pattern_name, timestamp DESC) 
        WHERE multi_timeframe_alignment >= 0.7 AND timeframe_confirmation_count >= 2
    """)
    
    # Create GIN indexes for JSONB fields
    op.execute("CREATE INDEX idx_enhanced_patterns_timeframe_hierarchy_gin ON enhanced_candlestick_patterns USING GIN (timeframe_hierarchy)")
    op.execute("CREATE INDEX idx_enhanced_patterns_order_flow_metadata_gin ON enhanced_candlestick_patterns USING GIN (order_flow_metadata)")
    op.execute("CREATE INDEX idx_enhanced_patterns_explanation_factors_gin ON enhanced_candlestick_patterns USING GIN (explanation_factors)")
    op.execute("CREATE INDEX idx_enhanced_patterns_calibration_confidence_interval_gin ON enhanced_candlestick_patterns USING GIN (calibration_confidence_interval)")
    
    op.execute("CREATE INDEX idx_ml_models_calibration_metrics_gin ON pattern_ml_models USING GIN (calibration_metrics)")
    op.execute("CREATE INDEX idx_ml_models_multi_timeframe_features_gin ON pattern_ml_models USING GIN (multi_timeframe_features)")
    op.execute("CREATE INDEX idx_ml_models_sequence_model_config_gin ON pattern_ml_models USING GIN (sequence_model_config)")
    
    op.execute("CREATE INDEX idx_correlations_regime_analysis_gin ON pattern_correlations USING GIN (regime_analysis)")
    op.execute("CREATE INDEX idx_correlations_multi_timeframe_correlation_gin ON pattern_correlations USING GIN (multi_timeframe_correlation)")
    op.execute("CREATE INDEX idx_correlations_timeframe_hierarchy_impact_gin ON pattern_correlations USING GIN (timeframe_hierarchy_impact)")
    
    op.execute("CREATE INDEX idx_validations_multi_timeframe_validation_gin ON pattern_validations USING GIN (multi_timeframe_validation)")
    op.execute("CREATE INDEX idx_validations_calibration_validation_gin ON pattern_validations USING GIN (calibration_validation)")
    
    logger.info("✅ Phase 4A database enhancements completed successfully")

def downgrade():
    """
    Downgrade: Remove Phase 4A enhancement fields
    """
    
    # Drop GIN indexes
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_timeframe_hierarchy_gin")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_order_flow_metadata_gin")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_explanation_factors_gin")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_calibration_confidence_interval_gin")
    op.execute("DROP INDEX IF EXISTS idx_ml_models_calibration_metrics_gin")
    op.execute("DROP INDEX IF EXISTS idx_ml_models_multi_timeframe_features_gin")
    op.execute("DROP INDEX IF EXISTS idx_ml_models_sequence_model_config_gin")
    op.execute("DROP INDEX IF EXISTS idx_correlations_regime_analysis_gin")
    op.execute("DROP INDEX IF EXISTS idx_correlations_multi_timeframe_correlation_gin")
    op.execute("DROP INDEX IF EXISTS idx_correlations_timeframe_hierarchy_impact_gin")
    op.execute("DROP INDEX IF EXISTS idx_validations_multi_timeframe_validation_gin")
    op.execute("DROP INDEX IF EXISTS idx_validations_calibration_validation_gin")
    
    # Drop partial indexes
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_high_calibrated_confidence")
    op.execute("DROP INDEX IF EXISTS idx_enhanced_patterns_strong_multi_timeframe")
    
    # Drop regular indexes
    op.drop_index('idx_enhanced_patterns_timeframe_hierarchy')
    op.drop_index('idx_enhanced_patterns_calibrated_confidence')
    op.drop_index('idx_enhanced_patterns_market_regime')
    op.drop_index('idx_enhanced_patterns_multi_timeframe_alignment')
    op.drop_index('idx_enhanced_patterns_timeframe_confirmation_count')
    op.drop_index('idx_ml_models_model_type')
    op.drop_index('idx_ml_models_ensemble_weight')
    op.drop_index('idx_correlations_regime_analysis')
    op.drop_index('idx_correlations_regime_specific_strength')
    
    # Drop columns from enhanced_candlestick_patterns
    op.drop_column('enhanced_candlestick_patterns', 'timeframe_hierarchy')
    op.drop_column('enhanced_candlestick_patterns', 'calibrated_confidence')
    op.drop_column('enhanced_candlestick_patterns', 'market_regime')
    op.drop_column('enhanced_candlestick_patterns', 'order_flow_metadata')
    op.drop_column('enhanced_candlestick_patterns', 'explanation_factors')
    op.drop_column('enhanced_candlestick_patterns', 'multi_timeframe_alignment')
    op.drop_column('enhanced_candlestick_patterns', 'timeframe_confirmation_count')
    op.drop_column('enhanced_candlestick_patterns', 'calibration_confidence_interval')
    
    # Drop columns from pattern_ml_models
    op.drop_column('pattern_ml_models', 'model_type')
    op.drop_column('pattern_ml_models', 'ensemble_weight')
    op.drop_column('pattern_ml_models', 'calibration_metrics')
    op.drop_column('pattern_ml_models', 'multi_timeframe_features')
    op.drop_column('pattern_ml_models', 'sequence_model_config')
    
    # Drop columns from pattern_correlations
    op.drop_column('pattern_correlations', 'regime_analysis')
    op.drop_column('pattern_correlations', 'multi_timeframe_correlation')
    op.drop_column('pattern_correlations', 'regime_specific_strength')
    op.drop_column('pattern_correlations', 'timeframe_hierarchy_impact')
    
    # Drop columns from pattern_validations
    op.drop_column('pattern_validations', 'multi_timeframe_validation')
    op.drop_column('pattern_validations', 'regime_validation_score')
    op.drop_column('pattern_validations', 'calibration_validation')
