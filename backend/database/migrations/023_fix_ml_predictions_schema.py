#!/usr/bin/env python3
"""
Migration 023: Fix ML Predictions Schema
Adds missing symbol column and other necessary columns to ml_predictions table
"""

import logging
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# revision identifiers
revision = '023'
down_revision = '022'
branch_labels = None
depends_on = None

def upgrade():
    """
    Upgrade database schema to fix ml_predictions table
    """
    logger.info("Starting ML Predictions Schema Fix migration...")

    # 1. Add missing columns to ml_predictions table
    logger.info("Adding missing columns to ml_predictions table...")
    
    # Add symbol column (required for multi-asset support)
    op.add_column('ml_predictions', sa.Column('symbol', sa.String(20), nullable=True))
    
    # Add prediction_value column (for ensemble predictions)
    op.add_column('ml_predictions', sa.Column('prediction_value', sa.DECIMAL(10, 6), nullable=True))
    
    # Add feature_vector column (for storing feature data)
    op.add_column('ml_predictions', sa.Column('feature_vector', postgresql.JSONB, nullable=True))
    
    # Add ensemble_weights column (for storing ensemble model weights)
    op.add_column('ml_predictions', sa.Column('ensemble_weights', postgresql.JSONB, nullable=True))
    
    # Add market_regime column (for storing market regime information)
    op.add_column('ml_predictions', sa.Column('market_regime', sa.String(50), nullable=True))
    
    # Add prediction_horizon column (for storing prediction time horizon)
    op.add_column('ml_predictions', sa.Column('prediction_horizon', sa.Integer, nullable=True))
    
    # Add risk_level column (for storing risk assessment)
    op.add_column('ml_predictions', sa.Column('risk_level', sa.String(20), nullable=True))
    
    # 2. Add missing columns to comprehensive_analysis table
    logger.info("Adding missing columns to comprehensive_analysis table...")
    
    # Add liquidity_score and volatility_score columns
    op.add_column('comprehensive_analysis', sa.Column('liquidity_score', sa.DECIMAL(5, 4), nullable=True))
    op.add_column('comprehensive_analysis', sa.Column('volatility_score', sa.DECIMAL(5, 4), nullable=True))
    
    # 3. Create indexes for performance optimization
    logger.info("Creating performance indexes...")
    
    # Indexes for ml_predictions
    op.create_index('idx_ml_predictions_symbol_timestamp', 'ml_predictions', ['symbol', 'timestamp'])
    op.create_index('idx_ml_predictions_model_type_symbol', 'ml_predictions', ['model_type', 'symbol'])
    op.create_index('idx_ml_predictions_confidence_symbol', 'ml_predictions', ['confidence_score', 'symbol'])
    op.create_index('idx_ml_predictions_market_regime', 'ml_predictions', ['market_regime'])
    
    # Indexes for comprehensive_analysis
    op.create_index('idx_comprehensive_analysis_liquidity_score', 'comprehensive_analysis', ['liquidity_score'])
    op.create_index('idx_comprehensive_analysis_volatility_score', 'comprehensive_analysis', ['volatility_score'])
    
    # 4. Update existing records to have default values
    logger.info("Updating existing records with default values...")
    
    # Update ml_predictions records to have default symbol
    op.execute("UPDATE ml_predictions SET symbol = 'BTC/USDT' WHERE symbol IS NULL")
    
    # Update comprehensive_analysis records to have default scores
    op.execute("UPDATE comprehensive_analysis SET liquidity_score = 0.5 WHERE liquidity_score IS NULL")
    op.execute("UPDATE comprehensive_analysis SET volatility_score = 0.5 WHERE volatility_score IS NULL")
    
    # 5. Make symbol column NOT NULL after setting default values
    logger.info("Making symbol column NOT NULL...")
    op.alter_column('ml_predictions', 'symbol', nullable=False)
    
    logger.info("ML Predictions Schema Fix migration completed successfully!")

def downgrade():
    """
    Downgrade database schema
    """
    logger.info("Starting ML Predictions Schema Fix downgrade...")
    
    # Remove indexes
    op.drop_index('idx_ml_predictions_symbol_timestamp', 'ml_predictions')
    op.drop_index('idx_ml_predictions_model_type_symbol', 'ml_predictions')
    op.drop_index('idx_ml_predictions_confidence_symbol', 'ml_predictions')
    op.drop_index('idx_ml_predictions_market_regime', 'ml_predictions')
    op.drop_index('idx_comprehensive_analysis_liquidity_score', 'comprehensive_analysis')
    op.drop_index('idx_comprehensive_analysis_volatility_score', 'comprehensive_analysis')
    
    # Remove columns from ml_predictions
    op.drop_column('ml_predictions', 'symbol')
    op.drop_column('ml_predictions', 'prediction_value')
    op.drop_column('ml_predictions', 'feature_vector')
    op.drop_column('ml_predictions', 'ensemble_weights')
    op.drop_column('ml_predictions', 'market_regime')
    op.drop_column('ml_predictions', 'prediction_horizon')
    op.drop_column('ml_predictions', 'risk_level')
    
    # Remove columns from comprehensive_analysis
    op.drop_column('comprehensive_analysis', 'liquidity_score')
    op.drop_column('comprehensive_analysis', 'volatility_score')
    
    logger.info("ML Predictions Schema Fix downgrade completed!")
