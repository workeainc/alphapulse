"""
Create Database Tables Directly via Python
Simpler approach without complex SQL triggers
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

async def create_tables():
    """Create all required tables"""
    
    print("=" * 70)
    print("Creating Database Tables")
    print("=" * 70)
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        print(f"Connected to: {DB_CONFIG['database']}")
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    try:
        # 1. Live Signals Table
        print("\nCreating live_signals table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS live_signals (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                entry_price DECIMAL(20, 8) NOT NULL,
                current_price DECIMAL(20, 8) NOT NULL,
                stop_loss DECIMAL(20, 8) NOT NULL,
                take_profit DECIMAL(20, 8) NOT NULL,
                confidence DECIMAL(5, 4) NOT NULL,
                quality_score DECIMAL(5, 4) NOT NULL,
                pattern_type VARCHAR(100),
                entry_proximity_pct DECIMAL(10, 6) NOT NULL,
                entry_proximity_status VARCHAR(20) NOT NULL,
                sde_consensus JSONB NOT NULL,
                mtf_analysis JSONB NOT NULL,
                agreeing_heads INT NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                invalidation_reason VARCHAR(200)
            )
        """)
        print("✓ live_signals created")
        
        # 2. Signal History Table
        print("Creating signal_history table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) UNIQUE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                entry_price DECIMAL(20, 8) NOT NULL,
                stop_loss DECIMAL(20, 8) NOT NULL,
                take_profit DECIMAL(20, 8) NOT NULL,
                confidence DECIMAL(5, 4) NOT NULL,
                quality_score DECIMAL(5, 4),
                pattern_type VARCHAR(100),
                sde_consensus JSONB,
                mtf_analysis JSONB,
                agreeing_heads INT,
                rsi DECIMAL(10, 4),
                macd DECIMAL(10, 6),
                volume_ratio DECIMAL(10, 4),
                indicators JSONB,
                outcome VARCHAR(20) DEFAULT 'pending',
                source VARCHAR(50) NOT NULL,
                lifecycle_status VARCHAR(20) DEFAULT 'completed',
                signal_timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        print("✓ signal_history created")
        
        # 3. Signal Lifecycle Table
        print("Creating signal_lifecycle table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_lifecycle (
                id SERIAL PRIMARY KEY,
                signal_id VARCHAR(100) NOT NULL,
                from_status VARCHAR(20) NOT NULL,
                to_status VARCHAR(20) NOT NULL,
                reason VARCHAR(200),
                current_price DECIMAL(20, 8),
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        print("✓ signal_lifecycle created")
        
        # 4. Current Market Prices Table
        print("Creating current_market_prices table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS current_market_prices (
                symbol VARCHAR(20) PRIMARY KEY,
                price DECIMAL(20, 8) NOT NULL,
                volume_24h DECIMAL(30, 8),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        print("✓ current_market_prices created")
        
        # Create indexes
        print("\nCreating indexes...")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_live_signals_symbol ON live_signals(symbol)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_live_signals_status ON live_signals(status)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_history_symbol ON signal_history(symbol)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_history_timestamp ON signal_history(signal_timestamp DESC)")
        print("✓ Indexes created")
        
        # Verify
        print("\nVerifying tables:")
        for table in ['live_signals', 'signal_history', 'signal_lifecycle', 'current_market_prices']:
            count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            print(f"  ✓ {table}: {count} rows")
        
        print(f"\n{'=' * 70}")
        print("Database tables created successfully!")
        print(f"{'=' * 70}")
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return False
    finally:
        await conn.close()

if __name__ == "__main__":
    success = asyncio.run(create_tables())
    if success:
        print("\nNext: python scripts/import_historical_signals.py")

