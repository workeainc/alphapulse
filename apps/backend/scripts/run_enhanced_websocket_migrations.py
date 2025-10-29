#!/usr/bin/env python3
"""
Enhanced WebSocket Migration Runner
Comprehensive migration script for the enhanced WebSocket system
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/migrations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedWebSocketMigrationRunner:
    """Comprehensive migration runner for enhanced WebSocket system"""
    
    def __init__(self):
        self.migrations = [
            {
                'name': 'Data Versioning Tables',
                'file': 'database/migrations/001_create_data_versioning_tables.py',
                'description': 'Create core data versioning tables with TimescaleDB optimizations'
            },
            {
                'name': 'Enhanced WebSocket Tables',
                'file': 'database/migrations/002_enhanced_websocket_tables.py',
                'description': 'Create enhanced WebSocket system tables and optimizations'
            }
        ]
        self.results = {}
    
    async def check_database_connection(self) -> bool:
        """Check if database is accessible"""
        logger.info("Checking database connection...")
        
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text
            
            # Use the same connection string as in migrations
            DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
            engine = create_async_engine(DATABASE_URL)
            
            async with engine.begin() as conn:
                result = await conn.execute(text("SELECT version()"))
                version = result.scalar()
                logger.info(f"Database connected: {version.split(',')[0]}")
                
                # Check if TimescaleDB extension is available
                result = await conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb'"))
                if result.fetchone():
                    logger.info("TimescaleDB extension is available")
                else:
                    logger.warning("TimescaleDB extension not found - some features may not work")
                
                return True
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Make sure TimescaleDB is running and accessible")
            return False
        finally:
            if 'engine' in locals():
                await engine.dispose()
    
    async def check_timescaledb_extension(self) -> bool:
        """Check if TimescaleDB extension is properly installed"""
        logger.info("Checking TimescaleDB extension...")
        
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text
            
            DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
            engine = create_async_engine(DATABASE_URL)
            
            async with engine.begin() as conn:
                # Check if TimescaleDB extension exists
                result = await conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'timescaledb'"))
                if not result.fetchone():
                    logger.info("Installing TimescaleDB extension...")
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                    logger.info("TimescaleDB extension installed")
                else:
                    logger.info("TimescaleDB extension already installed")
                
                # Check TimescaleDB version
                result = await conn.execute(text("SELECT default_version, installed_version FROM pg_available_extensions WHERE name = 'timescaledb'"))
                ext_info = result.fetchone()
                if ext_info:
                    logger.info(f"TimescaleDB version: {ext_info.installed_version}")
                
                return True
                
        except Exception as e:
            logger.error(f"TimescaleDB extension check failed: {e}")
            return False
        finally:
            if 'engine' in locals():
                await engine.dispose()
    
    async def run_migration(self, migration: Dict[str, str]) -> bool:
        """Run a single migration"""
        logger.info(f"Running migration: {migration['name']}")
        logger.info(f"Description: {migration['description']}")
        
        try:
            migration_file = backend_dir / migration['file']
            
            if not migration_file.exists():
                logger.error(f"Migration file not found: {migration_file}")
                return False
            
            # Import and run the migration
            import importlib.util
            spec = importlib.util.spec_from_file_location("migration", migration_file)
            migration_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(migration_module)
            
            # Run the migration
            if hasattr(migration_module, 'main'):
                await migration_module.main()
                logger.info(f"Migration completed: {migration['name']}")
                return True
            else:
                logger.error(f"Migration module has no main function: {migration['name']}")
                return False
                
        except Exception as e:
            logger.error(f"Migration failed: {migration['name']} - {e}")
            return False
    
    async def verify_migration_results(self) -> bool:
        """Verify that all migrations were successful"""
        logger.info("Verifying migration results...")
        
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text
            
            DATABASE_URL = "postgresql+asyncpg://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
            engine = create_async_engine(DATABASE_URL)
            
            async with engine.begin() as conn:
                # Check if all required tables exist
                required_tables = [
                    'signals', 'candles', 'retrain_queue',
                    'websocket_connections', 'websocket_performance', 'redis_pubsub_events'
                ]
                
                result = await conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_name = ANY(:tables)
                    ORDER BY table_name
                """), {'tables': required_tables})
                
                existing_tables = [row[0] for row in result.fetchall()]
                
                logger.info("Existing tables:")
                for table in existing_tables:
                    logger.info(f"   - {table}")
                
                missing_tables = set(required_tables) - set(existing_tables)
                if missing_tables:
                    logger.warning(f"Missing tables: {missing_tables}")
                    return False
                
                # Check hypertables
                result = await conn.execute(text("""
                    SELECT hypertable_name
                    FROM timescaledb_information.hypertables
                    ORDER BY hypertable_name
                """))
                
                hypertables = [row[0] for row in result.fetchall()]
                logger.info("TimescaleDB hypertables:")
                for ht in hypertables:
                    logger.info(f"   - {ht}")
                
                # Check signals table structure
                result = await conn.execute(text("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'signals'
                    ORDER BY ordinal_position
                """))
                
                logger.info("Signals table structure:")
                for row in result.fetchall():
                    logger.info(f"   - {row[0]}: {row[1]}")
                
                return True
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
        finally:
            if 'engine' in locals():
                await engine.dispose()
    
    async def create_directories(self):
        """Create necessary directories"""
        logger.info("Creating directories...")
        
        directories = [
            'logs',
            'data',
            'cache',
            'reports',
            'migrations'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        logger.info("Directories created")
    
    async def run_all_migrations(self) -> bool:
        """Run all migrations in sequence"""
        logger.info("Starting Enhanced WebSocket migrations...")
        
        try:
            # Create directories
            await self.create_directories()
            
            # Check database connection
            if not await self.check_database_connection():
                return False
            
            # Check TimescaleDB extension
            if not await self.check_timescaledb_extension():
                return False
            
            # Run each migration
            for migration in self.migrations:
                success = await self.run_migration(migration)
                self.results[migration['name']] = success
                
                if not success:
                    logger.error(f"Migration failed: {migration['name']}")
                    return False
                
                # Small delay between migrations
                await asyncio.sleep(1)
            
            # Verify results
            if not await self.verify_migration_results():
                logger.error("Migration verification failed")
                return False
            
            logger.info("All migrations completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Migration runner failed: {e}")
            return False
    
    def print_summary(self):
        """Print migration summary"""
        print("\n" + "="*60)
        print("ENHANCED WEBSOCKET MIGRATION SUMMARY")
        print("="*60)
        
        total_migrations = len(self.migrations)
        successful_migrations = sum(1 for success in self.results.values() if success)
        
        print(f"Total migrations: {total_migrations}")
        print(f"Successful: {successful_migrations}")
        print(f"Failed: {total_migrations - successful_migrations}")
        print()
        
        for migration in self.migrations:
            status = "SUCCESS" if self.results.get(migration['name'], False) else "FAILED"
            print(f"{status} - {migration['name']}")
            print(f"   {migration['description']}")
            print()
        
        if successful_migrations == total_migrations:
            print("All migrations completed successfully!")
            print("Enhanced WebSocket system is ready to use!")
        else:
            print("Some migrations failed. Please check the logs and try again.")
        
        print("="*60)

async def main():
    """Main migration runner function"""
    runner = EnhancedWebSocketMigrationRunner()
    
    try:
        success = await runner.run_all_migrations()
        runner.print_summary()
        
        if not success:
            logger.error("Migration runner failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Enhanced WebSocket Migration Runner")
    print("=" * 50)
    print("This script will run all necessary database migrations")
    print("for the enhanced WebSocket system.")
    print()
    
    # Run the migration runner
    asyncio.run(main())
