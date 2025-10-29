#!/usr/bin/env python3
"""
Phase 3.2: News Event Integration Database Migrations
Adds comprehensive news event analysis columns to enhanced_signals table
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

class Phase3_2NewsEventsMigrations:
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
    
    async def create_news_events_enhanced_signals_migration(self):
        """Add news event analysis columns to enhanced_signals table"""
        logger.info("🔧 Creating Phase 3.2: News Event Integration Migration...")
        
        try:
            # Add news event analysis columns to enhanced_signals table
            await self.conn.execute("""
                ALTER TABLE enhanced_signals 
                ADD COLUMN IF NOT EXISTS news_events JSONB,
                ADD COLUMN IF NOT EXISTS event_impact_score FLOAT,
                ADD COLUMN IF NOT EXISTS event_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS high_impact_events INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS medium_impact_events INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS low_impact_events INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS event_categories JSONB,
                ADD COLUMN IF NOT EXISTS news_aware_signal BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS event_filtered_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS event_enhanced_confidence FLOAT,
                ADD COLUMN IF NOT EXISTS event_filtered_sentiment JSONB,
                ADD COLUMN IF NOT EXISTS event_keywords JSONB,
                ADD COLUMN IF NOT EXISTS event_relevance_score FLOAT,
                ADD COLUMN IF NOT EXISTS event_sentiment_analysis JSONB,
                ADD COLUMN IF NOT EXISTS news_events_last_updated TIMESTAMPTZ
            """)
            
            logger.info("✅ Added news event analysis columns to enhanced_signals table")
            
            # Create indexes for news event analysis
            await self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_event_impact_score 
                ON enhanced_signals(event_impact_score DESC);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_event_count 
                ON enhanced_signals(event_count DESC);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_high_impact_events 
                ON enhanced_signals(high_impact_events DESC);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_news_aware_signal 
                ON enhanced_signals(news_aware_signal);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_event_filtered_confidence 
                ON enhanced_signals(event_filtered_confidence DESC);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_event_enhanced_confidence 
                ON enhanced_signals(event_enhanced_confidence DESC);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_news_events 
                ON enhanced_signals USING GIN (news_events);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_event_categories 
                ON enhanced_signals USING GIN (event_categories);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_event_keywords 
                ON enhanced_signals USING GIN (event_keywords);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_event_sentiment_analysis 
                ON enhanced_signals USING GIN (event_sentiment_analysis);
                
                CREATE INDEX IF NOT EXISTS idx_enhanced_signals_news_events_last_updated 
                ON enhanced_signals(news_events_last_updated DESC);
            """)
            
            logger.info("✅ Created news event analysis indexes")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating news events migration: {e}")
            return False
    
    async def create_news_events_enhanced_signals_view(self):
        """Create news events-enhanced signals view"""
        logger.info("🔧 Creating news events-enhanced signals view...")
        
        try:
            await self.conn.execute("""
                CREATE OR REPLACE VIEW news_events_enhanced_signals AS
                SELECT 
                    *,
                    CASE 
                        WHEN event_impact_score > 0.7 THEN 'very_high'
                        WHEN event_impact_score > 0.5 THEN 'high'
                        WHEN event_impact_score > 0.3 THEN 'medium'
                        WHEN event_impact_score > 0.1 THEN 'low'
                        ELSE 'none'
                    END as event_impact_level,
                    CASE 
                        WHEN event_filtered_confidence >= 0.8 THEN 'very_high'
                        WHEN event_filtered_confidence >= 0.6 THEN 'high'
                        WHEN event_filtered_confidence >= 0.4 THEN 'medium'
                        ELSE 'low'
                    END as event_confidence_level,
                    COALESCE(event_impact_score * event_filtered_confidence, 0) as event_weighted_score,
                    CASE 
                        WHEN high_impact_events > 0 THEN 'high_impact'
                        WHEN medium_impact_events > 0 THEN 'medium_impact'
                        WHEN low_impact_events > 0 THEN 'low_impact'
                        ELSE 'no_events'
                    END as event_priority_level
                FROM enhanced_signals
                WHERE news_events IS NOT NULL
                ORDER BY created_at DESC;
            """)
            
            logger.info("✅ Created news events-enhanced signals view")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating news events view: {e}")
            return False
    
    async def create_news_events_functions(self):
        """Create news event analysis functions"""
        logger.info("🔧 Creating news event analysis functions...")
        
        try:
            # Function to calculate news event-enhanced quality
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_news_events_enhanced_quality(
                    base_quality FLOAT,
                    event_impact_score FLOAT,
                    event_filtered_confidence FLOAT,
                    news_aware_signal BOOLEAN
                ) RETURNS FLOAT AS $$
                BEGIN
                    -- Combine base quality with news event factors
                    DECLARE
                        event_boost FLOAT;
                    BEGIN
                        IF news_aware_signal THEN
                            event_boost = (event_impact_score * event_filtered_confidence * 0.3);
                        ELSE
                            event_boost = 0.0;
                        END IF;
                        
                        RETURN LEAST(1.0, base_quality + event_boost);
                    END;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to update news event performance
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION update_news_events_performance(
                    signal_id BIGINT,
                    event_accuracy FLOAT,
                    event_impact_accuracy FLOAT
                ) RETURNS VOID AS $$
                BEGIN
                    UPDATE enhanced_signals 
                    SET 
                        event_relevance_score = event_accuracy,
                        event_impact_score = event_impact_accuracy
                    WHERE id = signal_id;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Function to calculate event sentiment correlation
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION calculate_event_sentiment_correlation(
                    sentiment_score FLOAT,
                    event_sentiment_score FLOAT
                ) RETURNS FLOAT AS $$
                BEGIN
                    -- Calculate correlation between sentiment and event sentiment
                    IF sentiment_score IS NULL OR event_sentiment_score IS NULL THEN
                        RETURN 0.0;
                    END IF;
                    
                    -- Simple correlation calculation
                    RETURN LEAST(1.0, ABS(sentiment_score - event_sentiment_score));
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Created news event analysis functions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating news events functions: {e}")
            return False
    
    async def create_news_events_triggers(self):
        """Create news event analysis triggers"""
        logger.info("🔧 Creating news event analysis triggers...")
        
        try:
            # Trigger function to update news events-enhanced quality
            await self.conn.execute("""
                CREATE OR REPLACE FUNCTION trigger_update_news_events_enhanced_quality()
                RETURNS TRIGGER AS $$
                BEGIN
                    -- Update news events-enhanced quality when event data changes
                    IF NEW.event_impact_score IS NOT NULL AND NEW.event_filtered_confidence IS NOT NULL THEN
                        NEW.signal_quality_score = calculate_news_events_enhanced_quality(
                            COALESCE(NEW.signal_quality_score, 0.5),
                            NEW.event_impact_score,
                            NEW.event_filtered_confidence,
                            COALESCE(NEW.news_aware_signal, FALSE)
                        );
                    END IF;
                    
                    -- Update event enhanced confidence
                    IF NEW.event_impact_score IS NOT NULL AND NEW.sentiment_confidence IS NOT NULL THEN
                        IF NEW.event_impact_score > 0.5 THEN
                            NEW.event_enhanced_confidence = LEAST(1.0, NEW.sentiment_confidence + (NEW.event_impact_score * 0.3));
                        ELSE
                            NEW.event_enhanced_confidence = NEW.sentiment_confidence + (NEW.event_impact_score * 0.1);
                        END IF;
                    END IF;
                    
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            # Create trigger
            await self.conn.execute("""
                DROP TRIGGER IF EXISTS trigger_update_news_events_enhanced_quality ON enhanced_signals;
                
                CREATE TRIGGER trigger_update_news_events_enhanced_quality
                BEFORE UPDATE ON enhanced_signals
                FOR EACH ROW
                EXECUTE FUNCTION trigger_update_news_events_enhanced_quality();
            """)
            
            logger.info("✅ Created news event analysis triggers")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating news events triggers: {e}")
            return False
    
    async def verify_migration(self):
        """Verify that migration was successful"""
        logger.info("🔍 Verifying Phase 3.2 migration...")
        
        try:
            # Check for news event columns
            result = await self.conn.fetch("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_signals' 
                AND (column_name LIKE 'event_%' OR column_name LIKE 'news_events%')
                ORDER BY column_name
            """)
            
            news_event_columns = [row['column_name'] for row in result]
            expected_columns = [
                'news_events', 'event_impact_score', 'event_count', 'high_impact_events',
                'medium_impact_events', 'low_impact_events', 'event_categories',
                'news_aware_signal', 'event_filtered_confidence', 'event_enhanced_confidence',
                'event_filtered_sentiment', 'event_keywords', 'event_relevance_score',
                'event_sentiment_analysis', 'news_events_last_updated'
            ]
            
            missing_columns = [col for col in expected_columns if col not in news_event_columns]
            
            if missing_columns:
                logger.error(f"❌ Missing news event columns: {missing_columns}")
                return False
            
            logger.info(f"✅ All {len(news_event_columns)} news event columns present")
            
            # Check for view
            view_exists = await self.conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_views 
                    WHERE viewname = 'news_events_enhanced_signals'
                )
            """)
            
            if view_exists:
                logger.info("✅ news_events_enhanced_signals view exists")
            else:
                logger.warning("⚠️ news_events_enhanced_signals view missing")
            
            # Check for functions
            functions = await self.conn.fetch("""
                SELECT proname FROM pg_proc 
                WHERE proname IN (
                    'calculate_news_events_enhanced_quality', 
                    'update_news_events_performance',
                    'calculate_event_sentiment_correlation'
                )
            """)
            
            logger.info(f"✅ Found {len(functions)} news event functions")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error verifying migration: {e}")
            return False
    
    async def run_migrations(self):
        """Run all Phase 3.2 migrations"""
        logger.info("🚀 Starting Phase 3.2: News Event Integration Migrations...")
        
        try:
            # Initialize database
            if not await self.initialize_database():
                return False
            
            # Run migrations
            migrations = [
                self.create_news_events_enhanced_signals_migration(),
                self.create_news_events_enhanced_signals_view(),
                self.create_news_events_functions(),
                self.create_news_events_triggers()
            ]
            
            for migration in migrations:
                if not await migration:
                    logger.error("❌ Migration failed")
                    return False
            
            # Verify migration
            if not await self.verify_migration():
                logger.error("❌ Migration verification failed")
                return False
            
            logger.info("🎉 Phase 3.2: News Event Integration migrations completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration process failed: {e}")
            return False
        finally:
            if self.conn:
                await self.conn.close()

async def main():
    """Main migration function"""
    migrations = Phase3_2NewsEventsMigrations()
    success = await migrations.run_migrations()
    
    if success:
        logger.info("🚀 Phase 3.2: News Event Integration migrations completed!")
        return 0
    else:
        logger.error("💥 Phase 3.2: News Event Integration migrations failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
