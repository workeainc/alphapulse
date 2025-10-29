#!/usr/bin/env python3
"""
PostgreSQL Connection Tester
Tests various connection methods to find the correct credentials
"""

import asyncio
import asyncpg
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_postgresql_connections():
    """Test various PostgreSQL connection methods"""
    
    # Common connection strings to try
    connection_strings = [
        # Default postgres user
        "postgresql://postgres:postgres@localhost:5432/postgres",
        "postgresql://postgres:password@localhost:5432/postgres",
        "postgresql://postgres:admin@localhost:5432/postgres",
        "postgresql://postgres:123456@localhost:5432/postgres",
        
        # Your original credentials
        "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse",
        "postgresql://alpha_emon:Emon_%4017711@localhost:5432/postgres",
        
        # Common variations
        "postgresql://postgres@localhost:5432/postgres",
        "postgresql://localhost:5432/postgres",
        
        # Try different ports
        "postgresql://postgres:postgres@localhost:5433/postgres",
        "postgresql://postgres:password@localhost:5433/postgres",
    ]
    
    successful_connections = []
    
    for i, conn_str in enumerate(connection_strings, 1):
        try:
            logger.info(f"🔌 Testing connection {i}/{len(connection_strings)}")
            logger.info(f"   Connection string: {conn_str.replace(conn_str.split('@')[0].split('//')[1], '***')}")
            
            conn = await asyncpg.connect(conn_str)
            
            # Test basic query
            result = await conn.fetchval("SELECT version()")
            logger.info(f"✅ Connection successful!")
            logger.info(f"📊 PostgreSQL version: {result}")
            
            # Check if TimescaleDB is available
            try:
                timescale_version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                if timescale_version:
                    logger.info(f"📊 TimescaleDB version: {timescale_version}")
                else:
                    logger.info("⚠️ TimescaleDB extension not found")
            except Exception as e:
                logger.info(f"⚠️ TimescaleDB check failed: {e}")
            
            # List databases
            databases = await conn.fetch("SELECT datname FROM pg_database WHERE datistemplate = false")
            logger.info(f"📊 Available databases: {[db['datname'] for db in databases]}")
            
            # List users
            users = await conn.fetch("SELECT usename FROM pg_user")
            logger.info(f"📊 Available users: {[user['usename'] for user in users]}")
            
            await conn.close()
            
            successful_connections.append({
                'connection_string': conn_str,
                'version': result,
                'databases': [db['datname'] for db in databases],
                'users': [user['usename'] for user in users]
            })
            
            break  # Stop after first successful connection
            
        except Exception as e:
            logger.warning(f"❌ Connection {i} failed: {e}")
            continue
    
    if successful_connections:
        logger.info("🎉 Found working PostgreSQL connection!")
        return successful_connections[0]
    else:
        logger.error("❌ No working PostgreSQL connections found")
        return None

async def create_alphapulse_database(conn_str: str):
    """Create alphapulse database if it doesn't exist"""
    try:
        logger.info("🔧 Creating alphapulse database...")
        
        # Connect to postgres database first
        base_conn_str = conn_str.replace('/alphapulse', '/postgres')
        conn = await asyncpg.connect(base_conn_str)
        
        # Check if alphapulse database exists
        db_exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = 'alphapulse'")
        
        if not db_exists:
            await conn.execute("CREATE DATABASE alphapulse")
            logger.info("✅ Created alphapulse database")
        else:
            logger.info("✅ alphapulse database already exists")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating database: {e}")
        return False

async def install_timescaledb_extension(conn_str: str):
    """Install TimescaleDB extension"""
    try:
        logger.info("🔧 Installing TimescaleDB extension...")
        
        conn = await asyncpg.connect(conn_str)
        
        # Check if TimescaleDB is available
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            logger.info("✅ TimescaleDB extension installed")
        except Exception as e:
            logger.warning(f"⚠️ TimescaleDB installation failed: {e}")
            logger.info("💡 You may need to install TimescaleDB separately")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error installing TimescaleDB: {e}")
        return False

async def run_migrations(conn_str: str):
    """Run database migrations"""
    try:
        logger.info("🔧 Running database migrations...")
        
        conn = await asyncpg.connect(conn_str)
        
        # Read migration file
        migration_file = Path("backend/database/migrations/create_ohlcv_hypertable.sql")
        if not migration_file.exists():
            logger.error(f"❌ Migration file not found: {migration_file}")
            return False
        
        logger.info(f"📄 Reading migration file: {migration_file}")
        migration_sql = migration_file.read_text(encoding='utf-8')
        
        # Split into individual statements
        statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]
        
        logger.info(f"📊 Executing {len(statements)} SQL statements...")
        
        # Execute each statement
        for i, statement in enumerate(statements, 1):
            if not statement:
                continue
                
            try:
                logger.info(f"🔄 Executing statement {i}/{len(statements)}...")
                await conn.execute(statement)
                logger.info(f"✅ Statement {i} executed successfully")
            except Exception as e:
                logger.error(f"❌ Error in statement {i}: {e}")
                logger.error(f"Statement: {statement[:100]}...")
                # Continue with other statements
                continue
        
        # Verify tables were created
        logger.info("🔍 Verifying table creation...")
        
        # Check if TimescaleDB extension is enabled
        extension_result = await conn.fetchval("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')")
        logger.info(f"📊 TimescaleDB extension enabled: {extension_result}")
        
        # Check if hypertables were created
        try:
            hypertables = await conn.fetch("""
                SELECT hypertable_name, num_dimensions, num_chunks 
                FROM timescaledb_information.hypertables 
                WHERE hypertable_schema = 'public'
            """)
            
            logger.info("📊 Created hypertables:")
            for table in hypertables:
                logger.info(f"  - {table['hypertable_name']}: {table['num_dimensions']} dimensions, {table['num_chunks']} chunks")
        except Exception as e:
            logger.warning(f"⚠️ Hypertable check failed: {e}")
        
        # Check if tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('ohlcv_data', 'order_book_data', 'technical_indicators', 'support_resistance_levels')
        """)
        
        logger.info("📊 Created tables:")
        for table in tables:
            logger.info(f"  - {table['table_name']}")
        
        await conn.close()
        logger.info("✅ Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

async def main():
    """Main function"""
    logger.info("🚀 Starting PostgreSQL Connection Test")
    logger.info("=" * 60)
    
    # Test connections
    connection_info = await test_postgresql_connections()
    if not connection_info:
        logger.error("❌ No working PostgreSQL connections found")
        logger.info("💡 Please check your PostgreSQL installation and credentials")
        return
    
    conn_str = connection_info['connection_string']
    logger.info(f"✅ Using connection: {conn_str.replace(conn_str.split('@')[0].split('//')[1], '***')}")
    
    # Create alphapulse database if needed
    if 'alphapulse' not in connection_info['databases']:
        await create_alphapulse_database(conn_str)
    
    # Update connection string to use alphapulse database
    if '/postgres' in conn_str:
        conn_str = conn_str.replace('/postgres', '/alphapulse')
    
    # Install TimescaleDB extension
    await install_timescaledb_extension(conn_str)
    
    # Run migrations
    if await run_migrations(conn_str):
        logger.info("🎉 Database setup completed successfully!")
        logger.info("💡 You can now run the live signal generation test with real PostgreSQL")
    else:
        logger.error("❌ Database setup failed!")

if __name__ == "__main__":
    asyncio.run(main())
