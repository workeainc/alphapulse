#!/usr/bin/env python3
"""
Migration Runner for AlphaPlus Algorithm Integration
Runs all database migrations for the enhanced algorithm implementations
"""

import asyncio
import asyncpg
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any
import subprocess

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Handles database migrations for AlphaPlus"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.logger = logger
        self.connection = None
        
        # Define migration files
        self.migrations = [
            {
                'name': 'Integrated Algorithm Schema',
                'file': 'backend/database/migrations/create_integrated_algorithm_schema.sql',
                'description': 'Creates integrated tables for enhanced algorithm implementations'
            }
        ]
        
        # Migration tracking
        self.migration_results = []
        
        logger.info("🔧 Migration Runner initialized")
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.connection = await asyncpg.connect(self.db_url)
            self.logger.info("✅ Database connection established")
            
            # Create migration tracking table if it doesn't exist
            await self._create_migration_tracking_table()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database connection: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.connection:
            await self.connection.close()
            self.logger.info("🔌 Database connection closed")
    
    async def _create_migration_tracking_table(self):
        """Create migration tracking table"""
        try:
            query = """
                CREATE TABLE IF NOT EXISTS migration_history (
                    id SERIAL PRIMARY KEY,
                    migration_name VARCHAR(255) NOT NULL,
                    migration_file VARCHAR(500) NOT NULL,
                    executed_at TIMESTAMPTZ DEFAULT NOW(),
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    execution_time_seconds NUMERIC(10,3)
                );
            """
            
            await self.connection.execute(query)
            self.logger.info("✅ Migration tracking table created/verified")
            
        except Exception as e:
            self.logger.error(f"❌ Error creating migration tracking table: {e}")
            raise
    
    async def check_migration_status(self, migration_name: str) -> bool:
        """Check if a migration has already been executed"""
        try:
            query = """
                SELECT COUNT(*) FROM migration_history 
                WHERE migration_name = $1 AND success = TRUE
            """
            
            count = await self.connection.fetchval(query, migration_name)
            return count > 0
            
        except Exception as e:
            self.logger.error(f"❌ Error checking migration status: {e}")
            return False
    
    async def record_migration_result(self, migration_name: str, migration_file: str, 
                                     success: bool, execution_time: float, error_message: str = None):
        """Record migration execution result"""
        try:
            query = """
                INSERT INTO migration_history 
                (migration_name, migration_file, success, error_message, execution_time_seconds)
                VALUES ($1, $2, $3, $4, $5)
            """
            
            await self.connection.execute(query, migration_name, migration_file, 
                                        success, error_message, execution_time)
            
        except Exception as e:
            self.logger.error(f"❌ Error recording migration result: {e}")
    
    async def run_migration_file(self, migration_file: str) -> Dict[str, Any]:
        """Run a single migration file"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"📄 Reading migration file: {migration_file}")
            
            # Check if file exists
            if not os.path.exists(migration_file):
                raise FileNotFoundError(f"Migration file not found: {migration_file}")
            
            # Read migration file
            with open(migration_file, 'r', encoding='utf-8') as f:
                migration_sql = f.read()
            
            if not migration_sql.strip():
                raise ValueError("Migration file is empty")
            
            self.logger.info(f"🚀 Executing migration: {migration_file}")
            
            # Execute migration
            await self.connection.execute(migration_sql)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"✅ Migration completed successfully in {execution_time:.2f}s")
            
            return {
                'success': True,
                'execution_time': execution_time,
                'error_message': None
            }
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            error_message = str(e)
            self.logger.error(f"❌ Migration failed: {error_message}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'error_message': error_message
            }
    
    async def run_all_migrations(self, force: bool = False) -> Dict[str, Any]:
        """Run all migrations"""
        self.logger.info("🚀 Starting migration process...")
        
        total_start_time = datetime.now()
        results = {
            'total_migrations': len(self.migrations),
            'successful_migrations': 0,
            'failed_migrations': 0,
            'skipped_migrations': 0,
            'total_execution_time': 0.0,
            'migration_details': []
        }
        
        for migration in self.migrations:
            migration_name = migration['name']
            migration_file = migration['file']
            
            self.logger.info(f"\n📋 Processing: {migration_name}")
            self.logger.info(f"   File: {migration_file}")
            self.logger.info(f"   Description: {migration['description']}")
            
            # Check if migration already executed
            if not force and await self.check_migration_status(migration_name):
                self.logger.info(f"⏭️  Skipping {migration_name} (already executed)")
                results['skipped_migrations'] += 1
                results['migration_details'].append({
                    'name': migration_name,
                    'file': migration_file,
                    'status': 'skipped',
                    'execution_time': 0.0,
                    'error_message': None
                })
                continue
            
            # Run migration
            migration_result = await self.run_migration_file(migration_file)
            
            # Record result
            await self.record_migration_result(
                migration_name,
                migration_file,
                migration_result['success'],
                migration_result['execution_time'],
                migration_result['error_message']
            )
            
            # Update results
            if migration_result['success']:
                results['successful_migrations'] += 1
                status = 'success'
            else:
                results['failed_migrations'] += 1
                status = 'failed'
            
            results['total_execution_time'] += migration_result['execution_time']
            results['migration_details'].append({
                'name': migration_name,
                'file': migration_file,
                'status': status,
                'execution_time': migration_result['execution_time'],
                'error_message': migration_result['error_message']
            })
        
        total_end_time = datetime.now()
        total_time = (total_end_time - total_start_time).total_seconds()
        
        # Print summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 MIGRATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Migrations: {results['total_migrations']}")
        self.logger.info(f"Successful: {results['successful_migrations']}")
        self.logger.info(f"Failed: {results['failed_migrations']}")
        self.logger.info(f"Skipped: {results['skipped_migrations']}")
        self.logger.info(f"Total Time: {total_time:.2f}s")
        
        # Print detailed results
        self.logger.info("\n📋 DETAILED RESULTS")
        self.logger.info("-" * 60)
        
        for detail in results['migration_details']:
            status_icon = "✅" if detail['status'] == 'success' else "❌" if detail['status'] == 'failed' else "⏭️"
            self.logger.info(f"{status_icon} {detail['name']} ({detail['execution_time']:.2f}s)")
            
            if detail['status'] == 'failed' and detail['error_message']:
                self.logger.info(f"   Error: {detail['error_message']}")
        
        return results
    
    async def rollback_migration(self, migration_name: str) -> bool:
        """Rollback a specific migration (manual process)"""
        try:
            self.logger.warning(f"⚠️  Rollback requested for: {migration_name}")
            self.logger.warning("⚠️  Manual rollback required - please review migration file and execute rollback SQL manually")
            
            # Mark migration as failed in tracking table
            query = """
                UPDATE migration_history 
                SET success = FALSE, error_message = 'Rolled back manually'
                WHERE migration_name = $1
            """
            
            await self.connection.execute(query, migration_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error during rollback: {e}")
            return False
    
    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get migration execution history"""
        try:
            query = """
                SELECT migration_name, migration_file, executed_at, success, 
                       error_message, execution_time_seconds
                FROM migration_history
                ORDER BY executed_at DESC
            """
            
            rows = await self.connection.fetch(query)
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            self.logger.error(f"❌ Error getting migration history: {e}")
            return []
    
    async def verify_database_schema(self) -> Dict[str, Any]:
        """Verify that all required tables exist"""
        try:
            required_tables = [
                'ohlcv_data',
                'order_book_data',
                'volume_profile_analysis',
                'order_book_levels',
                'market_microstructure',
                'psychological_levels_analysis',
                'psychological_levels',
                'psychological_level_interactions',
                'algorithm_results',
                'signal_confluence',
                'algorithm_performance'
            ]
            
            existing_tables = []
            missing_tables = []
            
            for table in required_tables:
                query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = $1
                    )
                """
                
                exists = await self.connection.fetchval(query, table)
                
                if exists:
                    existing_tables.append(table)
                else:
                    missing_tables.append(table)
            
            return {
                'total_required': len(required_tables),
                'existing_tables': existing_tables,
                'missing_tables': missing_tables,
                'all_tables_exist': len(missing_tables) == 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error verifying database schema: {e}")
            return {
                'total_required': 0,
                'existing_tables': [],
                'missing_tables': [],
                'all_tables_exist': False,
                'error': str(e)
            }

async def main():
    """Main migration runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaPlus Database Migration Runner')
    parser.add_argument('--force', action='store_true', help='Force re-run all migrations')
    parser.add_argument('--verify', action='store_true', help='Verify database schema only')
    parser.add_argument('--history', action='store_true', help='Show migration history')
    parser.add_argument('--rollback', help='Rollback specific migration by name')
    
    args = parser.parse_args()
    
    runner = MigrationRunner()
    
    try:
        await runner.initialize()
        
        if args.verify:
            # Verify database schema
            print("🔍 Verifying database schema...")
            schema_status = await runner.verify_database_schema()
            
            print(f"\n📊 Schema Verification Results:")
            print(f"Required Tables: {schema_status['total_required']}")
            print(f"Existing Tables: {len(schema_status['existing_tables'])}")
            print(f"Missing Tables: {len(schema_status['missing_tables'])}")
            
            if schema_status['missing_tables']:
                print(f"\n❌ Missing Tables:")
                for table in schema_status['missing_tables']:
                    print(f"  - {table}")
            else:
                print(f"\n✅ All required tables exist!")
            
        elif args.history:
            # Show migration history
            print("📋 Migration History:")
            history = await runner.get_migration_history()
            
            for record in history:
                status = "✅" if record['success'] else "❌"
                print(f"{status} {record['migration_name']} - {record['executed_at']}")
                if not record['success'] and record['error_message']:
                    print(f"   Error: {record['error_message']}")
        
        elif args.rollback:
            # Rollback specific migration
            success = await runner.rollback_migration(args.rollback)
            if success:
                print(f"✅ Rollback initiated for: {args.rollback}")
            else:
                print(f"❌ Rollback failed for: {args.rollback}")
        
        else:
            # Run all migrations
            results = await runner.run_all_migrations(force=args.force)
            
            if results['failed_migrations'] > 0:
                print(f"\n❌ {results['failed_migrations']} migrations failed!")
                sys.exit(1)
            else:
                print(f"\n✅ All migrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration runner error: {e}")
        sys.exit(1)
    
    finally:
        await runner.close()

if __name__ == "__main__":
    asyncio.run(main())
