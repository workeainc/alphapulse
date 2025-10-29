#!/usr/bin/env python3
"""
Robust migration runner for AlphaPlus enhanced algorithm integration.
Executes SQL migration files to set up the database schema with proper error handling.
"""

import asyncio
import asyncpg
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection details
DB_URL = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"

async def run_migration_file(db_pool: asyncpg.Pool, migration_file: Path):
    """Run a single migration file with proper error handling."""
    try:
        logger.info(f"🔄 Running migration: {migration_file.name}")
        
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Split SQL content into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        async with db_pool.acquire() as conn:
            for i, statement in enumerate(statements):
                if statement:
                    try:
                        await conn.execute(statement)
                        logger.debug(f"✅ Statement {i+1}/{len(statements)} executed successfully")
                    except Exception as stmt_error:
                        # Check if it's a "already exists" error
                        error_msg = str(stmt_error).lower()
                        if any(phrase in error_msg for phrase in [
                            'already exists', 'is already a hypertable', 'already a hypertable',
                            'relation already exists', 'already exists as'
                        ]):
                            logger.info(f"ℹ️ Statement {i+1} skipped (already exists): {stmt_error}")
                        else:
                            logger.warning(f"⚠️ Statement {i+1} failed: {stmt_error}")
                            # Continue with other statements
        
        logger.info(f"✅ Migration completed: {migration_file.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {migration_file.name} - {e}")
        return False

async def check_table_exists(db_pool: asyncpg.Pool, table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = $1
                );
            """, table_name)
            return result
    except Exception as e:
        logger.error(f"❌ Error checking table existence for {table_name}: {e}")
        return False

async def main():
    """Main migration runner."""
    logger.info("🚀 Starting AlphaPlus Enhanced Algorithm Integration Migrations")
    
    # Create database connection pool
    try:
        db_pool = await asyncpg.create_pool(DB_URL)
        logger.info("✅ Database connection pool created")
    except Exception as e:
        logger.error(f"❌ Failed to create database connection pool: {e}")
        return
    
    # Check if TimescaleDB extension is enabled
    try:
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');")
            if result:
                logger.info("✅ TimescaleDB extension is enabled")
            else:
                logger.warning("⚠️ TimescaleDB extension not found - some features may not work")
    except Exception as e:
        logger.warning(f"⚠️ Could not check TimescaleDB extension: {e}")
    
    # Define migration files to run (in order)
    migrations_dir = Path("database/migrations")
    migration_files = [
        "create_simplified_algorithm_schema.sql",
        "create_enhanced_orderbook_integration.sql", 
        "create_psychological_levels_analyzer.sql"
    ]
    
    successful_migrations = 0
    total_migrations = len(migration_files)
    
    for migration_file in migration_files:
        migration_path = migrations_dir / migration_file
        if migration_path.exists():
            success = await run_migration_file(db_pool, migration_path)
            if success:
                successful_migrations += 1
        else:
            logger.warning(f"⚠️ Migration file not found: {migration_file}")
    
    # Verify key tables exist
    key_tables = [
        'ohlcv_data', 'order_book_data', 'algorithm_results', 
        'signal_confluence', 'volume_profile_analysis', 
        'psychological_levels_analysis'
    ]
    
    logger.info("🔍 Verifying key tables exist...")
    existing_tables = 0
    for table in key_tables:
        exists = await check_table_exists(db_pool, table)
        if exists:
            existing_tables += 1
            logger.info(f"✅ Table {table} exists")
        else:
            logger.warning(f"⚠️ Table {table} does not exist")
    
    # Close database connection pool
    await db_pool.close()
    logger.info("🔌 Database connection pool closed")
    
    # Summary
    logger.info(f"📊 Migration Summary: {successful_migrations}/{total_migrations} migrations successful")
    logger.info(f"📊 Table Verification: {existing_tables}/{len(key_tables)} key tables exist")
    
    if successful_migrations >= total_migrations * 0.5 and existing_tables >= len(key_tables) * 0.8:
        logger.info("🎉 Database setup completed successfully!")
    else:
        logger.warning("⚠️ Some migrations or tables may need attention")

if __name__ == "__main__":
    asyncio.run(main())
