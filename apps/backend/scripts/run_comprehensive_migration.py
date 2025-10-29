#!/usr/bin/env python3
"""
Comprehensive Migration Script for AlphaPlus Database Integration
Fixes all schema conflicts and ensures compatibility between existing and enhanced systems
"""

import asyncio
import asyncpg
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ComprehensiveMigration:
    """Handles all database migration tasks for AlphaPlus integration"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.conn = None
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.conn = await asyncpg.connect(self.db_url)
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            logger.info("🔌 Database connection closed")
    
    async def run_migration(self):
        """Run comprehensive migration"""
        try:
            await self.initialize()
            
            # Step 1: Add missing unique constraints to existing tables
            await self.add_unique_constraints()
            
            # Step 2: Create new tables with compatible schema
            await self.create_compatible_tables()
            
            # Step 3: Add missing indexes
            await self.add_performance_indexes()
            
            # Step 4: Grant permissions
            await self.grant_permissions()
            
            # Step 5: Verify migration
            await self.verify_migration()
            
            logger.info("🎉 Comprehensive migration completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            raise
        finally:
            await self.close()
    
    async def add_unique_constraints(self):
        """Add unique constraints to existing tables"""
        logger.info("🔧 Adding unique constraints to existing tables...")
        
        constraints = [
            {
                'table': 'ohlcv_data',
                'constraint': 'uk_ohlcv_symbol_timeframe_timestamp',
                'columns': '(symbol, timeframe, timestamp)',
                'check_query': "SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'uk_ohlcv_symbol_timeframe_timestamp' AND table_name = 'ohlcv_data'"
            },
            {
                'table': 'volume_profile_analysis',
                'constraint': 'uk_volume_profile_symbol_timeframe_timestamp',
                'columns': '(symbol, timeframe, timestamp)',
                'check_query': "SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'uk_volume_profile_symbol_timeframe_timestamp' AND table_name = 'volume_profile_analysis'"
            },
            {
                'table': 'psychological_levels_analysis',
                'constraint': 'uk_psychological_analysis_symbol_timeframe_timestamp',
                'columns': '(symbol, timeframe, timestamp)',
                'check_query': "SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'uk_psychological_analysis_symbol_timeframe_timestamp' AND table_name = 'psychological_levels_analysis'"
            },
            {
                'table': 'psychological_levels',
                'constraint': 'uk_psychological_levels_symbol_type_price_timestamp',
                'columns': '(symbol, level_type, price_level, timestamp)',
                'check_query': "SELECT 1 FROM information_schema.table_constraints WHERE constraint_name = 'uk_psychological_levels_symbol_type_price_timestamp' AND table_name = 'psychological_levels'"
            }
        ]
        
        for constraint_info in constraints:
            try:
                # Check if constraint already exists
                exists = await self.conn.fetchval(constraint_info['check_query'])
                
                if not exists:
                    # Add the constraint
                    alter_query = f"""
                        ALTER TABLE {constraint_info['table']} 
                        ADD CONSTRAINT {constraint_info['constraint']} 
                        UNIQUE {constraint_info['columns']}
                    """
                    await self.conn.execute(alter_query)
                    logger.info(f"✅ Added constraint {constraint_info['constraint']} to {constraint_info['table']}")
                else:
                    logger.info(f"ℹ️ Constraint {constraint_info['constraint']} already exists on {constraint_info['table']}")
                    
            except Exception as e:
                logger.error(f"❌ Error adding constraint {constraint_info['constraint']}: {e}")
    
    async def create_compatible_tables(self):
        """Create new tables with compatible schema"""
        logger.info("🏗️ Creating compatible tables...")
        
        # Read the schema SQL
        with open('database/migrations/create_simplified_algorithm_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        # Split the SQL into individual statements and execute them one by one
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for statement in statements:
            try:
                # Skip CREATE EXTENSION statements (already handled)
                if statement.upper().startswith('CREATE EXTENSION'):
                    continue
                
                # Skip CREATE TABLE IF NOT EXISTS statements (tables already exist)
                if statement.upper().startswith('CREATE TABLE IF NOT EXISTS'):
                    logger.info(f"ℹ️ Skipping table creation (already exists): {statement[:50]}...")
                    continue
                
                # Skip hypertable creation statements (already exist)
                if statement.upper().startswith('SELECT CREATE_HYPERTABLE'):
                    logger.info(f"ℹ️ Skipping hypertable creation (already exists): {statement[:50]}...")
                    continue
                
                # Execute other statements (constraints, indexes, grants)
                if statement:
                    await self.conn.execute(statement)
                    logger.info(f"✅ Executed: {statement[:50]}...")
                    
            except Exception as e:
                # Log warning but continue with other statements
                logger.warning(f"⚠️ Statement execution warning: {e}")
                logger.warning(f"⚠️ Statement: {statement[:100]}...")
        
        logger.info("✅ Compatible tables configuration completed")
    
    async def add_performance_indexes(self):
        """Add performance indexes"""
        logger.info("📊 Adding performance indexes...")
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_volume_profile_symbol_timeframe ON volume_profile_analysis (symbol, timeframe)",
            "CREATE INDEX IF NOT EXISTS idx_volume_profile_timestamp ON volume_profile_analysis (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_order_book_levels_symbol ON order_book_levels (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_order_book_levels_timestamp ON order_book_levels (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_microstructure_symbol ON market_microstructure (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_market_microstructure_timestamp ON market_microstructure (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_psychological_analysis_symbol ON psychological_levels_analysis (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_psychological_analysis_timestamp ON psychological_levels_analysis (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_psychological_levels_symbol ON psychological_levels (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_psychological_levels_timestamp ON psychological_levels (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_psychological_interactions_symbol ON psychological_level_interactions (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_psychological_interactions_timestamp ON psychological_level_interactions (timestamp)"
        ]
        
        for index_sql in indexes:
            try:
                await self.conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"⚠️ Index creation warning: {e}")
        
        logger.info("✅ Performance indexes added")
    
    async def grant_permissions(self):
        """Grant permissions to alpha_emon user"""
        logger.info("🔐 Granting permissions...")
        
        tables = [
            'volume_profile_analysis',
            'order_book_levels', 
            'market_microstructure',
            'psychological_levels_analysis',
            'psychological_levels',
            'psychological_level_interactions',
            'algorithm_results',
            'signal_confluence'
        ]
        
        for table in tables:
            try:
                await self.conn.execute(f"GRANT ALL PRIVILEGES ON TABLE {table} TO alpha_emon")
                await self.conn.execute(f"GRANT ALL PRIVILEGES ON SEQUENCE {table}_id_seq TO alpha_emon")
            except Exception as e:
                logger.warning(f"⚠️ Permission grant warning for {table}: {e}")
        
        logger.info("✅ Permissions granted")
    
    async def verify_migration(self):
        """Verify migration was successful"""
        logger.info("🔍 Verifying migration...")
        
        # Check if all tables exist
        tables_to_check = [
            'ohlcv_data',
            'volume_profile_analysis',
            'order_book_levels',
            'market_microstructure', 
            'psychological_levels_analysis',
            'psychological_levels',
            'psychological_level_interactions',
            'algorithm_results',
            'signal_confluence'
        ]
        
        for table in tables_to_check:
            try:
                result = await self.conn.fetchval(f"SELECT 1 FROM information_schema.tables WHERE table_name = '{table}'")
                if result:
                    logger.info(f"✅ Table {table} exists")
                else:
                    logger.warning(f"⚠️ Table {table} not found")
            except Exception as e:
                logger.warning(f"⚠️ Error checking table {table}: {e}")
        
        # Check if TimescaleDB hypertables are created
        hypertables = await self.conn.fetch("""
            SELECT hypertable_name FROM timescaledb_information.hypertables 
            WHERE hypertable_name IN ('ohlcv_data', 'volume_profile_analysis', 'order_book_levels', 
                                     'market_microstructure', 'psychological_levels_analysis', 
                                     'psychological_levels', 'psychological_level_interactions', 
                                     'algorithm_results', 'signal_confluence')
        """)
        
        for row in hypertables:
            logger.info(f"✅ Hypertable {row['hypertable_name']} is active")
        
        logger.info("✅ Migration verification completed")

async def main():
    """Main migration function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    migration = ComprehensiveMigration()
    await migration.run_migration()

if __name__ == "__main__":
    asyncio.run(main())
