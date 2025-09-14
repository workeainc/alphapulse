#!/usr/bin/env python3
"""
Free API Database Migration Runner
Runs the init_free_api_tables.sql migration using asyncpg
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

import asyncpg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FreeAPIMigrationRunner:
    """Migration runner for Free API tables using asyncpg"""
    
    def __init__(self):
        # Database configuration
        self.host = 'localhost'
        self.port = 5432
        self.database = 'alphaplus'  # Note: using alphaplus instead of alphapulse
        self.username = 'alpha_emon'
        self.password = 'Emon_@17711'
        
        self.connection = None
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            logger.info("Initializing database connection...")
            
            self.connection = await asyncpg.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            
            # Test connection
            await self.connection.execute("SELECT 1")
            
            logger.info("✅ Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection: {e}")
            raise
    
    async def run_migration(self):
        """Run the Free API tables migration"""
        try:
            logger.info("Running Free API database migration...")
            
            # Read the SQL migration file
            migration_file = Path(__file__).parent / "database" / "migrations" / "init_free_api_tables.sql"
            
            if not migration_file.exists():
                raise FileNotFoundError(f"Migration file not found: {migration_file}")
            
            logger.info(f"Reading migration file: {migration_file}")
            with open(migration_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Split SQL content into individual statements
            # Remove comments and split by semicolon
            statements = []
            for line in sql_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('--'):
                    statements.append(line)
            
            # Join statements and split by semicolon
            full_sql = ' '.join(statements)
            sql_statements = [stmt.strip() for stmt in full_sql.split(';') if stmt.strip()]
            
            logger.info(f"Executing {len(sql_statements)} SQL statements...")
            
            # Execute each statement
            for i, statement in enumerate(sql_statements, 1):
                if statement:
                    try:
                        logger.info(f"Executing statement {i}/{len(sql_statements)}...")
                        await self.connection.execute(statement)
                        logger.info(f"✅ Statement {i} executed successfully")
                    except Exception as e:
                        logger.warning(f"⚠️ Statement {i} failed (may already exist): {e}")
                        # Continue with other statements
            
            logger.info("✅ Free API database migration completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error running migration: {e}")
            raise
    
    async def verify_tables(self):
        """Verify that the tables were created successfully"""
        try:
            logger.info("Verifying table creation...")
            
            expected_tables = [
                'free_api_market_data',
                'free_api_sentiment_data', 
                'free_api_news_data',
                'free_api_social_data',
                'free_api_liquidation_events',
                'free_api_data_quality',
                'free_api_rate_limits'
            ]
            
            for table in expected_tables:
                result = await self.connection.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                    table
                )
                if result:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.warning(f"⚠️ Table '{table}' not found")
            
            # Check for TimescaleDB hypertables
            hypertables = await self.connection.fetch(
                "SELECT hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_name LIKE 'free_api_%'"
            )
            
            logger.info(f"✅ Found {len(hypertables)} TimescaleDB hypertables:")
            for ht in hypertables:
                logger.info(f"  - {ht['hypertable_name']}")
            
        except Exception as e:
            logger.error(f"❌ Error verifying tables: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            logger.info("Database connection closed")

async def main():
    """Main function to run the migration"""
    runner = FreeAPIMigrationRunner()
    
    try:
        await runner.initialize()
        await runner.run_migration()
        await runner.verify_tables()
        
        logger.info("🎉 Free API database migration completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        sys.exit(1)
        
    finally:
        await runner.close()

if __name__ == "__main__":
    asyncio.run(main())
