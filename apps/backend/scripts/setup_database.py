"""
Setup Database Schema for Live Signals
Executes the SQL schema creation using Python
"""

import asyncio
import asyncpg

DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def setup_database():
    """Create database schema"""
    
    print("=" * 70)
    print("AlphaPulse Database Setup")
    print("=" * 70)
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        print(f"Connected to database: {DB_CONFIG['database']}")
    except Exception as e:
        print(f"\nERROR: Could not connect to database")
        print(f"Details: {e}")
        print(f"\nMake sure PostgreSQL/TimescaleDB is running on port {DB_CONFIG['port']}")
        print("Or update DB_CONFIG with correct credentials")
        return False
    
    try:
        # Read SQL file
        with open("scripts/create_live_signals_schema.sql", "r") as f:
            sql_script = f.read()
        
        print("\nExecuting schema creation...")
        
        # Split and execute each statement
        statements = [s.strip() for s in sql_script.split(';') if s.strip()]
        
        for i, statement in enumerate(statements):
            if not statement or statement.startswith('--'):
                continue
            
            try:
                await conn.execute(statement)
                if i % 5 == 0:
                    print(f"  Executed {i}/{len(statements)} statements...")
            except Exception as e:
                # Some statements might fail if objects already exist
                if 'already exists' not in str(e) and 'does not exist' not in str(e):
                    print(f"  Warning: {e}")
        
        print(f"\nSchema creation complete!")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public' 
              AND tablename IN ('live_signals', 'signal_history', 'signal_lifecycle', 'current_market_prices')
            ORDER BY tablename
        """)
        
        print(f"\nVerifying tables:")
        for table in tables:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table['tablename']}")
            print(f"  ✓ {table['tablename']}: {count} rows")
        
        print(f"\n{'=' * 70}")
        print("Database setup successful!")
        print(f"{'=' * 70}")
        return True
        
    except Exception as e:
        print(f"\nERROR during schema creation: {e}")
        return False
    finally:
        await conn.close()

if __name__ == "__main__":
    success = asyncio.run(setup_database())
    if success:
        print("\nNext step: Run import_historical_signals.py to load backtest data")
    else:
        print("\nFix database connection issues before proceeding")
