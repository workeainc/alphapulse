#!/usr/bin/env python3
"""
Script to run the Priority 2 Advanced Feature Engineering tables migration
"""

import asyncio
import logging
from sqlalchemy import text
from ..src.database.connection import TimescaleDBConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_priority2_feature_engineering_tables():
    """Create the Priority 2 Advanced Feature Engineering tracking tables"""
    
    logger.info("🚀 Creating Priority 2 Advanced Feature Engineering tables...")
    
    try:
        # Initialize database connection
        db_connection = TimescaleDBConnection()
        db_connection.initialize()
        
        async with db_connection.get_async_session() as session:
            
            # Create priority2_feature_extraction_metrics table
            create_extraction_metrics_sql = """
            CREATE TABLE IF NOT EXISTS priority2_feature_extraction_metrics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                extraction_type VARCHAR(50) NOT NULL,  -- 'sliding_windows', 'pca', 'advanced_indicators'
                original_shape_rows INTEGER NOT NULL,
                original_shape_cols INTEGER NOT NULL,
                final_shape_rows INTEGER NOT NULL,
                final_shape_cols INTEGER NOT NULL,
                features_removed INTEGER NOT NULL,
                extraction_time_seconds FLOAT NOT NULL,
                cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
                cache_source VARCHAR(20),  -- 'redis', 'local', 'none'
                
                -- Sliding window metrics
                sliding_window_features_count INTEGER,
                overlapping_window_features_count INTEGER,
                adaptive_window_features_count INTEGER,
                
                -- PCA metrics
                pca_variant_used VARCHAR(50),  -- 'standard', 'incremental', 'kernel_rbf'
                pca_explained_variance_ratio FLOAT,
                pca_n_components INTEGER,
                
                -- Advanced indicators metrics
                advanced_indicators_count INTEGER,
                divergence_features_count INTEGER,
                regime_features_count INTEGER,
                breakout_features_count INTEGER,
                
                -- Performance metrics
                memory_usage_mb FLOAT,
                cpu_usage_percent FLOAT,
                processing_efficiency FLOAT,  -- features per second
                
                -- Quality metrics
                nan_values_count INTEGER DEFAULT 0,
                infinite_values_count INTEGER DEFAULT 0,
                data_quality_score FLOAT,
                
                -- Metadata
                extraction_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                feature_engineering_version VARCHAR(20),
                configuration_hash VARCHAR(64),
                notes TEXT
            );
            """
            
            await session.execute(text(create_extraction_metrics_sql))
            logger.info("✅ Created priority2_feature_extraction_metrics table")
            
            # Create priority2_feature_cache_metrics table
            create_cache_metrics_sql = """
            CREATE TABLE IF NOT EXISTS priority2_feature_cache_metrics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                cache_key VARCHAR(100) NOT NULL,
                cache_source VARCHAR(20) NOT NULL,  -- 'redis', 'local'
                cache_hit BOOLEAN NOT NULL,
                cache_miss BOOLEAN NOT NULL,
                cache_size_mb FLOAT,
                cache_retrieval_time_ms FLOAT,
                cache_storage_time_ms FLOAT,
                cache_expiration_time TIMESTAMPTZ,
                cache_age_minutes INTEGER,
                
                -- Cache performance
                cache_hit_rate FLOAT,
                cache_efficiency_score FLOAT,
                
                -- Metadata
                cache_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                cache_operation VARCHAR(20),  -- 'read', 'write', 'delete'
                error_message TEXT
            );
            """
            
            await session.execute(text(create_cache_metrics_sql))
            logger.info("✅ Created priority2_feature_cache_metrics table")
            
            # Create priority2_sliding_window_metrics table
            create_sliding_window_metrics_sql = """
            CREATE TABLE IF NOT EXISTS priority2_sliding_window_metrics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                window_size INTEGER NOT NULL,
                window_type VARCHAR(30) NOT NULL,  -- 'price', 'volume', 'overlapping', 'adaptive'
                features_generated INTEGER NOT NULL,
                computation_time_ms FLOAT NOT NULL,
                memory_usage_mb FLOAT,
                
                -- Statistical features
                mean_features_count INTEGER,
                std_features_count INTEGER,
                min_max_features_count INTEGER,
                percentile_features_count INTEGER,
                momentum_features_count INTEGER,
                volatility_features_count INTEGER,
                
                -- Performance metrics
                window_creation_efficiency FLOAT,  -- windows per second
                feature_density FLOAT,  -- features per window
                
                -- Quality metrics
                data_coverage_percent FLOAT,
                feature_correlation_avg FLOAT,
                
                -- Metadata
                extraction_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                window_configuration JSONB,
                notes TEXT
            );
            """
            
            await session.execute(text(create_sliding_window_metrics_sql))
            logger.info("✅ Created priority2_sliding_window_metrics table")
            
            # Create priority2_pca_metrics table
            create_pca_metrics_sql = """
            CREATE TABLE IF NOT EXISTS priority2_pca_metrics (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                pca_variant VARCHAR(50) NOT NULL,  -- 'standard', 'incremental', 'kernel_rbf'
                original_features_count INTEGER NOT NULL,
                reduced_features_count INTEGER NOT NULL,
                compression_ratio FLOAT NOT NULL,  -- original/reduced
                explained_variance_ratio FLOAT,
                cumulative_explained_variance FLOAT,
                
                -- Performance metrics
                pca_fit_time_ms FLOAT,
                pca_transform_time_ms FLOAT,
                total_pca_time_ms FLOAT,
                pca_efficiency FLOAT,  -- features per second
                
                -- Quality metrics
                reconstruction_error FLOAT,
                information_loss_percent FLOAT,
                feature_importance_scores JSONB,
                
                -- Configuration
                n_components INTEGER,
                random_state INTEGER,
                kernel_type VARCHAR(20),  -- for kernel PCA
                
                -- Metadata
                extraction_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                pca_configuration JSONB,
                notes TEXT
            );
            """
            
            await session.execute(text(create_pca_metrics_sql))
            logger.info("✅ Created priority2_pca_metrics table")
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS ix_priority2_extraction_metrics_symbol ON priority2_feature_extraction_metrics(symbol);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_extraction_metrics_date ON priority2_feature_extraction_metrics(extraction_date);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_extraction_metrics_type ON priority2_feature_extraction_metrics(extraction_type);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_extraction_metrics_cache_hit ON priority2_feature_extraction_metrics(cache_hit);",
                
                "CREATE INDEX IF NOT EXISTS ix_priority2_cache_metrics_symbol ON priority2_feature_cache_metrics(symbol);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_cache_metrics_date ON priority2_feature_cache_metrics(cache_date);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_cache_metrics_hit ON priority2_feature_cache_metrics(cache_hit);",
                
                "CREATE INDEX IF NOT EXISTS ix_priority2_sliding_window_symbol ON priority2_sliding_window_metrics(symbol);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_sliding_window_size ON priority2_sliding_window_metrics(window_size);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_sliding_window_type ON priority2_sliding_window_metrics(window_type);",
                
                "CREATE INDEX IF NOT EXISTS ix_priority2_pca_symbol ON priority2_pca_metrics(symbol);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_pca_variant ON priority2_pca_metrics(pca_variant);",
                "CREATE INDEX IF NOT EXISTS ix_priority2_pca_date ON priority2_pca_metrics(extraction_date);"
            ]
            
            for index_sql in indexes:
                await session.execute(text(index_sql))
            logger.info("✅ Created performance indexes")
            
            # Convert to TimescaleDB hypertables
            try:
                await session.execute(text("SELECT create_hypertable('priority2_feature_extraction_metrics', 'extraction_date', if_not_exists => TRUE);"))
                await session.execute(text("SELECT create_hypertable('priority2_feature_cache_metrics', 'cache_date', if_not_exists => TRUE);"))
                await session.execute(text("SELECT create_hypertable('priority2_sliding_window_metrics', 'extraction_date', if_not_exists => TRUE);"))
                await session.execute(text("SELECT create_hypertable('priority2_pca_metrics', 'extraction_date', if_not_exists => TRUE);"))
                logger.info("✅ Converted to TimescaleDB hypertables")
            except Exception as e:
                logger.warning(f"Could not convert to hypertables (TimescaleDB might not be available): {e}")
            
            # Set compression policies
            try:
                await session.execute(text("""
                    ALTER TABLE priority2_feature_extraction_metrics SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,extraction_type',
                        timescaledb.compress_orderby = 'extraction_date DESC'
                    );
                """))
                
                await session.execute(text("""
                    ALTER TABLE priority2_feature_cache_metrics SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,cache_source',
                        timescaledb.compress_orderby = 'cache_date DESC'
                    );
                """))
                
                await session.execute(text("""
                    ALTER TABLE priority2_sliding_window_metrics SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,window_type',
                        timescaledb.compress_orderby = 'extraction_date DESC'
                    );
                """))
                
                await session.execute(text("""
                    ALTER TABLE priority2_pca_metrics SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,pca_variant',
                        timescaledb.compress_orderby = 'extraction_date DESC'
                    );
                """))
                logger.info("✅ Set compression policies")
            except Exception as e:
                logger.warning(f"Could not set compression policies: {e}")
            
            # Add compression and retention policies
            try:
                await session.execute(text("SELECT add_compression_policy('priority2_feature_extraction_metrics', INTERVAL '7 days');"))
                await session.execute(text("SELECT add_compression_policy('priority2_feature_cache_metrics', INTERVAL '7 days');"))
                await session.execute(text("SELECT add_compression_policy('priority2_sliding_window_metrics', INTERVAL '7 days');"))
                await session.execute(text("SELECT add_compression_policy('priority2_pca_metrics', INTERVAL '7 days');"))
                
                await session.execute(text("SELECT add_retention_policy('priority2_feature_extraction_metrics', INTERVAL '1 year');"))
                await session.execute(text("SELECT add_retention_policy('priority2_feature_cache_metrics', INTERVAL '1 year');"))
                await session.execute(text("SELECT add_retention_policy('priority2_sliding_window_metrics', INTERVAL '1 year');"))
                await session.execute(text("SELECT add_retention_policy('priority2_pca_metrics', INTERVAL '1 year');"))
                logger.info("✅ Added compression and retention policies")
            except Exception as e:
                logger.warning(f"Could not add compression/retention policies: {e}")
            
            await session.commit()
            logger.info("🎉 Priority 2 Advanced Feature Engineering tables created successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error creating Priority 2 tables: {e}")
        raise

async def main():
    """Main function"""
    await create_priority2_feature_engineering_tables()

if __name__ == "__main__":
    asyncio.run(main())
