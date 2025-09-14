#!/usr/bin/env python3
"""
Phase 3.3: Social Media Integration - Database Migrations
Adds social media sentiment columns, indexes, views, and functions to enhanced_signals table
"""

import asyncio
import asyncpg
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase33SocialMediaMigrations:
    """Phase 3.3 Social Media Integration Database Migrations"""
    
    def __init__(self):
        self.conn = None
        
    async def connect_database(self):
        """Connect to database"""
        try:
            self.conn = await asyncpg.connect(
                host='localhost',
                port=5432,
                database='alphapulse',
                user='alpha_emon',
                password='Emon_@17711'
            )
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def close_database(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
    
    async def create_social_media_migration(self):
        """Create Phase 3.3: Social Media Integration Migration"""
        logger.info("🔧 Creating Phase 3.3: Social Media Integration Migration...")
        
        try:
            # Add social media sentiment columns
            await self.conn.execute("""
                ALTER TABLE enhanced_signals
                ADD COLUMN IF NOT EXISTS social_media_sentiment JSONB,
                ADD COLUMN IF NOT EXISTS social_impact_score FLOAT,
                ADD COLUMN IF NOT EXISTS social_sentiment_score FLOAT,
                ADD COLUMN IF NOT EXISTS social_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS social_trends JSONB,
                ADD COLUMN IF NOT EXISTS social_momentum JSONB,
                ADD COLUMN IF NOT EXISTS social_volume JSONB,
                ADD COLUMN IF NOT EXISTS social_engagement JSONB,
                ADD COLUMN IF NOT EXISTS social_volatility JSONB,
                ADD COLUMN IF NOT EXISTS social_correlation JSONB,
                ADD COLUMN IF NOT EXISTS social_aware_signal BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS social_enhanced_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS social_filtered_sentiment JSONB,
                ADD COLUMN IF NOT EXISTS twitter_sentiment_data JSONB,
                ADD COLUMN IF NOT EXISTS reddit_sentiment_data JSONB,
                ADD COLUMN IF NOT EXISTS social_sentiment_history JSONB,
                ADD COLUMN IF NOT EXISTS social_media_last_updated TIMESTAMPTZ
            """)
            logger.info("✅ Added social media sentiment columns to enhanced_signals table")
            
            # Create social media sentiment indexes
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_impact_score 
                ON enhanced_signals (social_impact_score);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_sentiment_score 
                ON enhanced_signals (social_sentiment_score);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_confidence 
                ON enhanced_signals (social_confidence);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_aware_signal 
                ON enhanced_signals (social_aware_signal);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_enhanced_confidence 
                ON enhanced_signals (social_enhanced_confidence);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_media_last_updated 
                ON enhanced_signals (social_media_last_updated);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_trends 
                ON enhanced_signals USING GIN (social_trends);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_momentum 
                ON enhanced_signals USING GIN (social_momentum);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_volume 
                ON enhanced_signals USING GIN (social_volume);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_engagement 
                ON enhanced_signals USING GIN (social_engagement);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_volatility 
                ON enhanced_signals USING GIN (social_volatility);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_correlation 
                ON enhanced_signals USING GIN (social_correlation);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_twitter_sentiment_data 
                ON enhanced_signals USING GIN (twitter_sentiment_data);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_reddit_sentiment_data 
                ON enhanced_signals USING GIN (reddit_sentiment_data);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_social_sentiment_history 
                ON enhanced_signals USING GIN (social_sentiment_history);
            """)
            logger.info("✅ Created social media sentiment indexes")
            
            # Create social media-enhanced signals view
            logger.info("🔧 Creating social media-enhanced signals view...")
            await self.conn.execute("""
                CREATE OR REPLACE VIEW social_media_enhanced_signals AS
                SELECT 
                    id,
                    symbol,
                    timestamp,
                    price,
                    side,
                    strategy,
                    confidence,
                    signal_quality_score,
                    social_media_sentiment,
                    social_impact_score,
                    social_sentiment_score,
                    social_confidence,
                    social_trends,
                    social_momentum,
                    social_volume,
                    social_engagement,
                    social_volatility,
                    social_correlation,
                    social_aware_signal,
                    social_enhanced_confidence,
                    social_filtered_sentiment,
                    twitter_sentiment_data,
                    reddit_sentiment_data,
                    social_sentiment_history,
                    social_media_last_updated,
                    created_at,
                    updated_at
                FROM enhanced_signals
                WHERE social_media_sentiment IS NOT NULL
                ORDER BY timestamp DESC;
            """)
            logger.info("✅ Created social media-enhanced signals view")
            
            # Create social media analysis functions
            logger.info("🔧 Creating social media analysis functions...")
            
            # Function to calculate social media sentiment quality
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_social_media_sentiment_quality(
                    p_social_impact_score FLOAT,
                    p_social_confidence FLOAT,
                    p_social_volume JSONB,
                    p_social_engagement JSONB
                ) RETURNS FLOAT AS $$
                DECLARE
                    volume_score FLOAT;
                    engagement_score FLOAT;
                    quality_score FLOAT;
                BEGIN
                    -- Extract volume score from JSON
                    volume_score := COALESCE((p_social_volume->>'volume_score')::FLOAT, 0.0);
                    
                    -- Extract engagement score from JSON
                    engagement_score := COALESCE((p_social_engagement->>'engagement_score')::FLOAT, 0.0);
                    
                    -- Calculate quality score (weighted average)
                    quality_score := (
                        p_social_impact_score * 0.4 +
                        p_social_confidence * 0.3 +
                        volume_score * 0.2 +
                        engagement_score * 0.1
                    );
                    
                    RETURN LEAST(quality_score, 1.0);
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to update social media performance metrics
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION update_social_media_performance(
                    p_symbol VARCHAR,
                    p_timeframe INTERVAL DEFAULT '1 hour'
                ) RETURNS JSONB AS $$
                DECLARE
                    performance_data JSONB;
                    avg_impact_score FLOAT;
                    avg_confidence FLOAT;
                    signal_count INTEGER;
                    success_rate FLOAT;
                BEGIN
                    -- Calculate performance metrics
                    SELECT 
                        AVG(social_impact_score),
                        AVG(social_confidence),
                        COUNT(*),
                        AVG(CASE WHEN signal_quality_score > 0.7 THEN 1.0 ELSE 0.0 END)
                    INTO avg_impact_score, avg_confidence, signal_count, success_rate
                    FROM enhanced_signals
                    WHERE symbol = p_symbol
                    AND timestamp >= NOW() - p_timeframe
                    AND social_media_sentiment IS NOT NULL;
                    
                    performance_data := jsonb_build_object(
                        'symbol', p_symbol,
                        'timeframe', p_timeframe::TEXT,
                        'avg_impact_score', COALESCE(avg_impact_score, 0.0),
                        'avg_confidence', COALESCE(avg_confidence, 0.0),
                        'signal_count', COALESCE(signal_count, 0),
                        'success_rate', COALESCE(success_rate, 0.0),
                        'calculated_at', NOW()
                    );
                    
                    RETURN performance_data;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to calculate social sentiment correlation
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_social_sentiment_correlation(
                    p_symbol VARCHAR,
                    p_timeframe INTERVAL DEFAULT '24 hours'
                ) RETURNS JSONB AS $$
                DECLARE
                    correlation_data JSONB;
                    price_changes FLOAT[];
                    sentiment_changes FLOAT[];
                    correlation_coefficient FLOAT;
                BEGIN
                    -- This is a simplified correlation calculation
                    -- In a real implementation, you would calculate actual correlation
                    -- between price changes and sentiment changes
                    
                    correlation_coefficient := 0.5; -- Simulated correlation
                    
                    correlation_data := jsonb_build_object(
                        'symbol', p_symbol,
                        'timeframe', p_timeframe::TEXT,
                        'correlation_coefficient', correlation_coefficient,
                        'correlation_strength', 
                            CASE 
                                WHEN correlation_coefficient > 0.7 THEN 'strong_positive'
                                WHEN correlation_coefficient > 0.3 THEN 'moderate_positive'
                                WHEN correlation_coefficient > -0.3 THEN 'weak'
                                WHEN correlation_coefficient > -0.7 THEN 'moderate_negative'
                                ELSE 'strong_negative'
                            END,
                        'calculated_at', NOW()
                    );
                    
                    RETURN correlation_data;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Created social media analysis functions")
            
            # Create trigger for automatic social media quality updates
            logger.info("🔧 Creating social media quality trigger...")
            
            # Function for trigger
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION trigger_update_social_media_quality()
                RETURNS TRIGGER AS $$
                BEGIN
                    -- Update signal quality score to include social media factors
                    IF NEW.social_impact_score IS NOT NULL AND NEW.social_confidence IS NOT NULL THEN
                        NEW.signal_quality_score := COALESCE(NEW.signal_quality_score, 0.0) + 
                            (NEW.social_impact_score * 0.2) + (NEW.social_confidence * 0.1);
                        
                        -- Ensure quality score doesn't exceed 1.0
                        NEW.signal_quality_score := LEAST(NEW.signal_quality_score, 1.0);
                    END IF;
                    
                    -- Update social media timestamp
                    NEW.social_media_last_updated := NOW();
                    
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Create trigger
            await self.conn.execute("""
                DROP TRIGGER IF EXISTS trigger_update_social_media_quality ON enhanced_signals;
                
                CREATE TRIGGER trigger_update_social_media_quality
                BEFORE INSERT OR UPDATE ON enhanced_signals
                FOR EACH ROW
                EXECUTE FUNCTION trigger_update_social_media_quality();
            """)
            
            logger.info("✅ Created social media quality trigger")
            
            logger.info("✅ Phase 3.3: Social Media Integration Migration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error creating social media migration: {e}")
            raise
    
    async def verify_migration(self):
        """Verify Phase 3.3 migration"""
        logger.info("🔍 Verifying Phase 3.3 migration...")
        
        try:
            # Check for social media columns
            required_columns = [
                'social_media_sentiment', 'social_impact_score', 'social_sentiment_score',
                'social_confidence', 'social_trends', 'social_momentum', 'social_volume',
                'social_engagement', 'social_volatility', 'social_correlation',
                'social_aware_signal', 'social_enhanced_confidence', 'social_filtered_sentiment',
                'twitter_sentiment_data', 'reddit_sentiment_data', 'social_sentiment_history',
                'social_media_last_updated'
            ]
            
            columns_result = await self.conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND table_schema = 'public'
                AND column_name IN (
                    'social_media_sentiment', 'social_impact_score', 'social_sentiment_score',
                    'social_confidence', 'social_trends', 'social_momentum', 'social_volume',
                    'social_engagement', 'social_volatility', 'social_correlation',
                    'social_aware_signal', 'social_enhanced_confidence', 'social_filtered_sentiment',
                    'twitter_sentiment_data', 'reddit_sentiment_data', 'social_sentiment_history',
                    'social_media_last_updated'
                )
            """)
            
            actual_columns = [row['column_name'] for row in columns_result]
            missing_columns = [col for col in required_columns if col not in actual_columns]
            
            if missing_columns:
                logger.error(f"❌ Missing social media columns: {missing_columns}")
                return False
            
            # Check for indexes
            indexes_result = await self.conn.fetch("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'enhanced_signals' 
                AND indexname LIKE '%social_%'
            """)
            
            if len(indexes_result) < 10:  # Should have multiple social media indexes
                logger.error(f"❌ Insufficient social media indexes: {len(indexes_result)}")
                return False
            
            # Check for view
            view_exists = await self.conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.views 
                    WHERE table_name = 'social_media_enhanced_signals'
                )
            """)
            
            if not view_exists:
                logger.error("❌ social_media_enhanced_signals view not found")
                return False
            
            # Check for functions
            functions_result = await self.conn.fetch("""
                SELECT proname 
                FROM pg_proc 
                WHERE proname LIKE '%social_%'
            """)
            
            if len(functions_result) < 3:  # Should have multiple social media functions
                logger.error(f"❌ Insufficient social media functions: {len(functions_result)}")
                return False
            
            logger.info("✅ Phase 3.3 migration verification passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return False

async def main():
    """Main migration function"""
    migrations = Phase33SocialMediaMigrations()
    
    try:
        await migrations.connect_database()
        await migrations.create_social_media_migration()
        
        if await migrations.verify_migration():
            logger.info("🎉 Phase 3.3: Social Media Integration migrations completed successfully!")
            return 0
        else:
            logger.error("💥 Phase 3.3: Social Media Integration migrations failed verification!")
            return 1
            
    except Exception as e:
        logger.error(f"💥 Phase 3.3: Social Media Integration migrations failed: {e}")
        return 1
    finally:
        await migrations.close_database()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
