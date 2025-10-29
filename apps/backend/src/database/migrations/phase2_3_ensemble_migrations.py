#!/usr/bin/env python3
"""
Phase 2.3: Ensemble Model Integration Database Migrations
Adds multi-model ensemble voting capabilities to the enhanced_signals table
"""
import asyncio
import logging
from src.app.core.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_phase2_3_migrations():
    """Run Phase 2.3 Ensemble Model Integration migrations"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize({
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'username': 'alpha_emon',
            'password': 'Emon_@17711'
        })
        
        logger.info("🔧 Applying Phase 2.3: Ensemble Model Integration migrations...")
        
        async with db_manager.get_connection() as conn:
            # Add Phase 2.3 Ensemble Model columns
            logger.info("🤖 Adding Phase 2.3: Ensemble Model columns...")
            
            await conn.execute("""
                ALTER TABLE enhanced_signals 
                ADD COLUMN IF NOT EXISTS ensemble_analysis JSONB,
                ADD COLUMN IF NOT EXISTS ensemble_voting_method VARCHAR(50),
                ADD COLUMN IF NOT EXISTS ensemble_model_weights JSONB,
                ADD COLUMN IF NOT EXISTS ensemble_individual_predictions JSONB,
                ADD COLUMN IF NOT EXISTS ensemble_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS ensemble_diversity_score FLOAT,
                ADD COLUMN IF NOT EXISTS ensemble_agreement_ratio FLOAT,
                ADD COLUMN IF NOT EXISTS ensemble_bias VARCHAR(20),
                ADD COLUMN IF NOT EXISTS ensemble_model_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS ensemble_performance_score FLOAT,
                ADD COLUMN IF NOT EXISTS ensemble_last_updated TIMESTAMPTZ
            """)
            
            logger.info("✅ Added Phase 2.3 ensemble columns")
            
            # Create ensemble-specific indexes
            logger.info("📊 Creating ensemble indexes...")
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_ensemble_confidence 
                ON enhanced_signals (ensemble_confidence)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_ensemble_voting_method 
                ON enhanced_signals (ensemble_voting_method)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_ensemble_bias 
                ON enhanced_signals (ensemble_bias)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_ensemble_diversity_score 
                ON enhanced_signals (ensemble_diversity_score)
            """)
            
            logger.info("✅ Created ensemble indexes")
            
            # Create ensemble enhanced signals view
            logger.info("👁️ Creating ensemble enhanced signals view...")
            
            await conn.execute("""
                CREATE OR REPLACE VIEW ensemble_enhanced_signals AS
                SELECT * FROM enhanced_signals
                WHERE ensemble_analysis IS NOT NULL
                  AND ensemble_confidence >= 0.6
                  AND ensemble_model_count >= 3
                  AND confidence >= 0.6
                ORDER BY ensemble_confidence DESC, ensemble_diversity_score DESC
            """)
            
            logger.info("✅ Created ensemble_enhanced_signals view")
            
            # Create ensemble quality function
            logger.info("⚙️ Creating ensemble quality function...")
            
            await conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_ensemble_enhanced_quality(
                    p_confidence FLOAT,
                    p_ensemble_confidence FLOAT,
                    p_ensemble_diversity_score FLOAT,
                    p_ensemble_agreement_ratio FLOAT
                ) RETURNS FLOAT AS $$
                BEGIN
                    RETURN (
                        p_confidence * 0.3 +
                        COALESCE(p_ensemble_confidence, 0.0) * 0.3 +
                        COALESCE(p_ensemble_diversity_score, 0.0) * 0.2 +
                        COALESCE(p_ensemble_agreement_ratio, 0.0) * 0.2
                    );
                END;
                $$ LANGUAGE plpgsql
            """)
            
            logger.info("✅ Created calculate_ensemble_enhanced_quality function")
            
            # Create ensemble performance tracking function
            await conn.execute("""
                CREATE OR REPLACE FUNCTION update_ensemble_performance(
                    p_signal_id VARCHAR(50),
                    p_ensemble_performance_score FLOAT
                ) RETURNS VOID AS $$
                BEGIN
                    UPDATE enhanced_signals 
                    SET ensemble_performance_score = p_ensemble_performance_score,
                        ensemble_last_updated = NOW()
                    WHERE id = p_signal_id;
                END;
                $$ LANGUAGE plpgsql
            """)
            
            logger.info("✅ Created update_ensemble_performance function")
            
            # Verify migrations
            logger.info("🔍 Verifying Phase 2.3 migrations...")
            
            # Check ensemble columns
            result = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'ensemble_%'
                ORDER BY column_name
            """)
            
            ensemble_columns = [row['column_name'] for row in result]
            expected_ensemble_columns = [
                'ensemble_analysis', 'ensemble_voting_method', 'ensemble_model_weights',
                'ensemble_individual_predictions', 'ensemble_confidence', 'ensemble_diversity_score',
                'ensemble_agreement_ratio', 'ensemble_bias', 'ensemble_model_count',
                'ensemble_performance_score', 'ensemble_last_updated'
            ]
            
            missing_columns = set(expected_ensemble_columns) - set(ensemble_columns)
            if missing_columns:
                logger.error(f"❌ Missing ensemble columns: {missing_columns}")
                return False
            
            logger.info(f"✅ All {len(ensemble_columns)} ensemble columns verified")
            
            # Check view
            result = await conn.fetch("""
                SELECT viewname FROM pg_views 
                WHERE viewname = 'ensemble_enhanced_signals'
            """)
            
            if result:
                logger.info("✅ ensemble_enhanced_signals view verified")
            else:
                logger.error("❌ ensemble_enhanced_signals view missing")
                return False
            
            # Check functions
            result = await conn.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN ('calculate_ensemble_enhanced_quality', 'update_ensemble_performance')
                ORDER BY proname
            """)
            
            functions = [row['proname'] for row in result]
            if len(functions) == 2:
                logger.info("✅ All ensemble functions verified")
            else:
                logger.error(f"❌ Missing functions. Found: {functions}")
                return False
        
        logger.info("🎉 Phase 2.3 Ensemble Model Integration migrations completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 2.3 migration failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(run_phase2_3_migrations())
