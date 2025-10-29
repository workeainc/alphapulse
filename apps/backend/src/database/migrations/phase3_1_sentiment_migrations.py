#!/usr/bin/env python3
"""
Phase 3.1: Sentiment Analysis Integration Database Migrations
Adds comprehensive sentiment analysis columns to enhanced_signals table
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.app.core.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase3_1SentimentMigrations:
    def __init__(self):
        self.db_manager = None
        self.db_connection = None
    
    async def initialize_database(self):
        """Initialize database connection"""
        try:
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            self.db_connection = await self.db_manager.get_connection()
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
            await self.db_connection.execute("""
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
            await self.db_connection.execute("""
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
    
    async def create_sentiment_enhanced_view(self):
        """Create view for sentiment-enhanced signals"""
        logger.info("🔧 Creating sentiment-enhanced signals view...")
        
        try:
            await self.db_connection.execute("""
                CREATE OR REPLACE VIEW sentiment_enhanced_signals AS
                SELECT 
                    id,
                    symbol,
                    side,
                    confidence,
                    sentiment_score,
                    sentiment_label,
                    sentiment_confidence,
                    sentiment_trend,
                    sentiment_volatility,
                    sentiment_momentum,
                    sentiment_correlation,
                    timestamp,
                    sentiment_last_updated,
                    metadata,
                    sentiment_analysis,
                    sentiment_sources,
                    twitter_sentiment,
                    reddit_sentiment,
                    news_sentiment,
                    telegram_sentiment,
                    discord_sentiment
                FROM enhanced_signals
                WHERE sentiment_score IS NOT NULL
                AND sentiment_confidence >= 0.6
                AND ABS(sentiment_score) >= 0.2
                ORDER BY sentiment_confidence DESC, sentiment_score DESC, timestamp DESC;
            """)
            
            logger.info("✅ Created sentiment_enhanced_signals view")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment view: {e}")
            return False
    
    async def create_sentiment_quality_function(self):
        """Create function to calculate sentiment-enhanced signal quality"""
        logger.info("🔧 Creating sentiment quality calculation function...")
        
        try:
            await self.db_connection.execute("""
                CREATE OR REPLACE FUNCTION calculate_sentiment_enhanced_quality(
                    base_confidence FLOAT,
                    sentiment_score FLOAT,
                    sentiment_confidence FLOAT,
                    sentiment_correlation FLOAT DEFAULT 0.0
                ) RETURNS FLOAT AS $$
                DECLARE
                    sentiment_weight FLOAT := 0.3;
                    correlation_weight FLOAT := 0.1;
                    enhanced_confidence FLOAT;
                BEGIN
                    -- Base calculation
                    enhanced_confidence := base_confidence;
                    
                    -- Add sentiment boost if sentiment is strong and confident
                    IF sentiment_confidence >= 0.7 AND ABS(sentiment_score) >= 0.3 THEN
                        enhanced_confidence := enhanced_confidence + (sentiment_weight * sentiment_confidence * ABS(sentiment_score));
                    END IF;
                    
                    -- Add correlation boost if sentiment correlates with price
                    IF sentiment_correlation > 0.5 THEN
                        enhanced_confidence := enhanced_confidence + (correlation_weight * sentiment_correlation);
                    END IF;
                    
                    -- Ensure confidence doesn't exceed 1.0
                    enhanced_confidence := LEAST(enhanced_confidence, 1.0);
                    
                    RETURN enhanced_confidence;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Created sentiment quality calculation function")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment quality function: {e}")
            return False
    
    async def create_sentiment_update_function(self):
        """Create function to update sentiment performance metrics"""
        logger.info("🔧 Creating sentiment update function...")
        
        try:
            await self.db_connection.execute("""
                CREATE OR REPLACE FUNCTION update_sentiment_performance(
                    symbol_param VARCHAR,
                    sentiment_score_param FLOAT,
                    price_change_param FLOAT,
                    timeframe_param VARCHAR DEFAULT '1h'
                ) RETURNS VOID AS $$
                BEGIN
                    -- Insert sentiment performance record
                    INSERT INTO sentiment_correlation (
                        symbol,
                        timestamp,
                        timeframe,
                        sentiment_score,
                        price_change,
                        sentiment_price_correlation,
                        sentiment_volatility_correlation,
                        sentiment_predictive_power
                    ) VALUES (
                        symbol_param,
                        NOW(),
                        timeframe_param,
                        sentiment_score_param,
                        price_change_param,
                        CASE 
                            WHEN ABS(sentiment_score_param) > 0.3 AND ABS(price_change_param) > 0.01 
                            THEN LEAST(ABS(sentiment_score_param * price_change_param), 1.0)
                            ELSE 0.0
                        END,
                        CASE 
                            WHEN ABS(sentiment_score_param) > 0.5 
                            THEN LEAST(ABS(sentiment_score_param), 1.0)
                            ELSE 0.0
                        END,
                        CASE 
                            WHEN ABS(sentiment_score_param) > 0.4 AND ABS(price_change_param) > 0.02 
                            THEN LEAST(ABS(sentiment_score_param * price_change_param * 2), 1.0)
                            ELSE 0.0
                        END
                    );
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Created sentiment update function")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment update function: {e}")
            return False
    
    async def create_sentiment_trigger(self):
        """Create trigger to automatically update sentiment-enhanced quality"""
        logger.info("🔧 Creating sentiment quality trigger...")
        
        try:
            await self.db_connection.execute("""
                DROP TRIGGER IF EXISTS trigger_update_sentiment_enhanced_quality ON enhanced_signals;
                
                CREATE TRIGGER trigger_update_sentiment_enhanced_quality
                BEFORE INSERT OR UPDATE ON enhanced_signals
                FOR EACH ROW
                EXECUTE FUNCTION update_sentiment_enhanced_quality();
            """)
            
            # Create the trigger function
            await self.db_connection.execute("""
                CREATE OR REPLACE FUNCTION update_sentiment_enhanced_quality() RETURNS TRIGGER AS $$
                BEGIN
                    -- Update sentiment-enhanced quality if sentiment data is available
                    IF NEW.sentiment_score IS NOT NULL AND NEW.sentiment_confidence IS NOT NULL THEN
                        NEW.confidence := calculate_sentiment_enhanced_quality(
                            NEW.confidence,
                            NEW.sentiment_score,
                            NEW.sentiment_confidence,
                            COALESCE(NEW.sentiment_correlation, 0.0)
                        );
                    END IF;
                    
                    -- Update sentiment last updated timestamp
                    NEW.sentiment_last_updated := NOW();
                    
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("✅ Created sentiment quality trigger")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating sentiment trigger: {e}")
            return False
    
    async def run_all_migrations(self):
        """Run all Phase 3.1 sentiment migrations"""
        logger.info("🚀 Starting Phase 3.1: Sentiment Analysis Integration Migrations...")
        
        # Initialize database
        if not await self.initialize_database():
            return False
        
        migrations = [
            ("Sentiment Enhanced Signals Migration", self.create_sentiment_enhanced_signals_migration),
            ("Sentiment Enhanced View", self.create_sentiment_enhanced_view),
            ("Sentiment Quality Function", self.create_sentiment_quality_function),
            ("Sentiment Update Function", self.create_sentiment_update_function),
            ("Sentiment Quality Trigger", self.create_sentiment_trigger)
        ]
        
        results = {}
        
        for migration_name, migration_func in migrations:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔧 Running: {migration_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await migration_func()
                results[migration_name] = result
                
                if result:
                    logger.info(f"✅ {migration_name}: SUCCESS")
                else:
                    logger.error(f"❌ {migration_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {migration_name}: ERROR - {e}")
                results[migration_name] = False
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📊 Phase 3.1: Migration Results Summary")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for migration_name, result in results.items():
            status = "✅ SUCCESS" if result else "❌ FAILED"
            logger.info(f"{migration_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} migrations successful")
        
        if passed == total:
            logger.info("🎉 Phase 3.1: All sentiment migrations completed successfully!")
        else:
            logger.error(f"⚠️ Phase 3.1: {total - passed} migrations failed. Please check the errors above.")
        
        return passed == total

async def main():
    """Main migration function"""
    migrator = Phase3_1SentimentMigrations()
    success = await migrator.run_all_migrations()
    
    if success:
        logger.info("🎯 Phase 3.1: Sentiment Analysis Integration migrations are ready!")
        sys.exit(0)
    else:
        logger.error("💥 Phase 3.1: Sentiment Analysis Integration migrations have issues that need to be resolved.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
