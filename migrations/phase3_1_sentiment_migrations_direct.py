#!/usr/bin/env python3
"""
Phase 3.1: Sentiment Analysis Integration Database Migrations (Direct Connection)
Adds comprehensive sentiment analysis columns to enhanced_signals table
"""

import asyncio
import logging
import asyncpg
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase3_1SentimentMigrationsDirect:
    def __init__(self):
        self.conn = None
    
    async def initialize_database(self):
        """Initialize database connection"""
        try:
            self.conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                database='alphapulse',
                user='alpha_emon',
                password='Emon_@17711'
            )
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    async def create_sentiment_enhanced_signals_migration(self):
        """Add sentiment analysis columns to enhanced_signals table"""
        logger.info("🔧 Creating Phase 3.1: Sentiment Analysis Migration...")
        
        try:
            # Add sentiment analysis columns to enhanced_signals table
            await self.conn.execute("""
                ALTER TABLE enhanced_signals 
                ADD COLUMN IF NOT EXISTS sentiment_analysis JSONB,
                ADD COLUMN IF NOT EXISTS sentiment_score FLOAT,
                ADD COLUMN IF NOT EXISTS sentiment_label VARCHAR(20),
                ADD COLUMN IF NOT EXISTS sentiment_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS sentiment_sources JSONB,
                ADD COLUMN IF NOT EXISTS twitter_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS reddit_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS news_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS telegram_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS discord_sentiment FLOAT,
                ADD COLUMN IF NOT EXISTS sentiment_trend VARCHAR(20),
                ADD COLUMN IF NOT EXISTS sentiment_volatility FLOAT,
                ADD COLUMN IF NOT EXISTS sentiment_momentum FLOAT,
                ADD COLUMN IF NOT EXISTS sentiment_correlation FLOAT,
                ADD COLUMN IF NOT EXISTS sentiment_last_updated TIMESTAMPTZ
            """)
            
            logger.info("✅ Added sentiment analysis columns to enhanced_signals table")
            
            # Create indexes for sentiment analysis
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_sentiment_score 
                ON enhanced_signals(sentiment_score DESC);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_sentiment_label 
                ON enhanced_signals(sentiment_label);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_sentiment_confidence 
                ON enhanced_signals(sentiment_confidence DESC);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_sentiment_analysis 
                ON enhanced_signals USING GIN (sentiment_analysis);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_sentiment_sources 
                ON enhanced_signals USING GIN (sentiment_sources);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_sentiment_trend 
                ON enhanced_signals(sentiment_trend);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_sentiment_last_updated 
                ON enhanced_signals(sentiment_last_updated DESC);
            """)
            
            logger.info("✅ Created sentiment analysis indexes")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment migration: {e}")
            return False
    
    async def create_sentiment_enhanced_signals_view(self):
        """Create sentiment-enhanced signals view"""
        logger.info("🔧 Creating sentiment-enhanced signals view...")
        
        try:
            await self.conn.execute("""
                CREATE OR REPLACE VIEW sentiment_enhanced_signals AS
                SELECT 
                    *,
                    CASE 
                        WHEN sentiment_score > 0.1 THEN 'positive'
                        WHEN sentiment_score < -0.1 THEN 'negative'
                        ELSE 'neutral'
                    END as sentiment_bias,
                    CASE 
                        WHEN sentiment_confidence >= 0.8 THEN 'high'
                        WHEN sentiment_confidence >= 0.6 THEN 'medium'
                        ELSE 'low'
                    END as sentiment_confidence_level,
                    COALESCE(sentiment_score * sentiment_confidence, 0) as sentiment_weighted_score
                FROM enhanced_signals
                WHERE sentiment_analysis IS NOT NULL
                ORDER BY created_at DESC;
            """)
            
            logger.info("✅ Created sentiment-enhanced signals view")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment view: {e}")
            return False
    
    async def create_sentiment_functions(self):
        """Create sentiment analysis functions"""
        logger.info("🔧 Creating sentiment analysis functions...")
        
        try:
            # Function to calculate sentiment-enhanced quality
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_sentiment_enhanced_quality(
                    base_quality FLOAT,
                    sentiment_score FLOAT,
                    sentiment_confidence FLOAT
                ) RETURNS FLOAT AS $$
                BEGIN
                    -- Combine base quality with sentiment factors
                    RETURN LEAST(1.0, base_quality + (ABS(sentiment_score) * sentiment_confidence * 0.2));
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to update sentiment performance
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION update_sentiment_performance(
                    signal_id BIGINT,
                    sentiment_accuracy FLOAT
                ) RETURNS VOID AS $$
                BEGIN
                    UPDATE enhanced_signals 
                    SET sentiment_correlation = sentiment_accuracy
                    WHERE id = signal_id;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Created sentiment analysis functions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment functions: {e}")
            return False
    
    async def create_sentiment_triggers(self):
        """Create sentiment analysis triggers"""
        logger.info("🔧 Creating sentiment analysis triggers...")
        
        try:
            # Trigger function to update sentiment-enhanced quality
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION trigger_update_sentiment_enhanced_quality()
                RETURNS TRIGGER AS $$
                BEGIN
                    -- Update sentiment-enhanced quality when sentiment data changes
                    IF NEW.sentiment_score IS NOT NULL AND NEW.sentiment_confidence IS NOT NULL THEN
                        NEW.signal_quality_score = calculate_sentiment_enhanced_quality(
                            COALESCE(NEW.signal_quality_score, 0.5),
                            NEW.sentiment_score,
                            NEW.sentiment_confidence
                        );
                    END IF;
                    
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Create trigger
            await self.conn.execute("""
                DROP TRIGGER IF EXISTS trigger_update_sentiment_enhanced_quality ON enhanced_signals;
                
                CREATE TRIGGER trigger_update_sentiment_enhanced_quality
                BEFORE UPDATE ON enhanced_signals
                FOR EACH ROW
                EXECUTE FUNCTION trigger_update_sentiment_enhanced_quality();
            """)
            
            logger.info("✅ Created sentiment analysis triggers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment triggers: {e}")
            return False
    
    async def verify_migration(self):
        """Verify that migration was successful"""
        logger.info("🔍 Verifying Phase 3.1 migration...")
        
        try:
            # Check for sentiment columns
            result = await self.conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND column_name LIKE 'sentiment_%'
                ORDER BY column_name
            """)
            
            sentiment_columns = [row['column_name'] for row in result]
            expected_columns = [
                'sentiment_analysis', 'sentiment_score', 'sentiment_label', 
                'sentiment_confidence', 'sentiment_sources', 'twitter_sentiment',
                'reddit_sentiment', 'news_sentiment', 'telegram_sentiment',
                'discord_sentiment', 'sentiment_trend', 'sentiment_volatility',
                'sentiment_momentum', 'sentiment_correlation', 'sentiment_last_updated'
            ]
            
            missing_columns = [col for col in expected_columns if col not in sentiment_columns]
            
            if missing_columns:
                logger.error(f"❌ Missing sentiment columns: {missing_columns}")
                return False
            
            logger.info(f"✅ All {len(sentiment_columns)} sentiment columns present")
            
            # Check for view
            view_exists = await self.conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_views 
                    WHERE viewname = 'sentiment_enhanced_signals'
                )
            """)
            
            if view_exists:
                logger.info("✅ sentiment_enhanced_signals view exists")
            else:
                logger.warning("⚠️ sentiment_enhanced_signals view missing")
            
            # Check for functions
            functions = await self.conn.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN ('calculate_sentiment_enhanced_quality', 'update_sentiment_performance')
            """)
            
            logger.info(f"✅ Found {len(functions)} sentiment functions")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying migration: {e}")
            return False
    
    async def run_migrations(self):
        """Run all Phase 3.1 migrations"""
        logger.info("🚀 Starting Phase 3.1: Sentiment Analysis Integration Migrations...")
        
        try:
            # Initialize database
            if not await self.initialize_database():
                return False
            
            # Run migrations
            migrations = [
                self.create_sentiment_enhanced_signals_migration(),
                self.create_sentiment_enhanced_signals_view(),
                self.create_sentiment_functions(),
                self.create_sentiment_triggers()
            ]
            
            for migration in migrations:
                if not await migration:
                    logger.error("❌ Migration failed")
                    return False
            
            # Verify migration
            if not await self.verify_migration():
                logger.error("❌ Migration verification failed")
                return False
            
            logger.info("🎉 Phase 3.1: Sentiment Analysis Integration migrations completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration process failed: {e}")
            return False
        finally:
            if self.conn:
                await self.conn.close()

async def main():
    """Main migration function"""
    migrations = Phase3_1SentimentMigrationsDirect()
    success = await migrations.run_migrations()
    
    if success:
        logger.info("🚀 Phase 3.1: Sentiment Analysis Integration migrations completed!")
        return 0
    else:
        logger.error("💥 Phase 3.1: Sentiment Analysis Integration migrations failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
