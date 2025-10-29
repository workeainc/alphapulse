"""
Import Historical Backtest Signals into Database
Populates signal_history table with 1,259 backtest signals for ML training
"""

import asyncio
import json
import uuid
from datetime import datetime
import asyncpg
from typing import List, Dict

# Database connection details from ENV.md
DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def import_historical_signals():
    """Import all historical signals from JSON to database"""
    
    print("=" * 70)
    print("AlphaPulse Historical Signal Import")
    print("=" * 70)
    
    # Load JSON file
    try:
        with open("historical_signals.json", "r") as f:
            signals = json.load(f)
        print(f"Loaded {len(signals)} signals from JSON file")
    except FileNotFoundError:
        print("ERROR: historical_signals.json not found!")
        print("Run backtest_data_generator.py first")
        return
    
    # Connect to database
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        print(f"Connected to database: {DB_CONFIG['database']}")
    except Exception as e:
        print(f"ERROR connecting to database: {e}")
        print(f"Make sure PostgreSQL/TimescaleDB is running on port {DB_CONFIG['port']}")
        return
    
    try:
        # Check if schema exists
        schema_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'signal_history'
            );
        """)
        
        if not schema_exists:
            print("\nERROR: signal_history table doesn't exist!")
            print("Run create_live_signals_schema.sql first:")
            print("  psql -U alpha_emon -d alphapulse -f apps/backend/scripts/create_live_signals_schema.sql")
            return
        
        print("\nStarting import...")
        imported_count = 0
        skipped_count = 0
        
        for signal in signals:
            try:
                signal_id = f"HIST_{uuid.uuid4().hex[:8].upper()}"
                
                # Prepare SDE consensus (simplified for backtest data)
                sde_consensus = {
                    "agreeing_heads": 7,  # Assume good consensus for historical
                    "consensus_score": signal['confidence'],
                    "final_confidence": signal['confidence']
                }
                
                # Prepare MTF analysis (simplified)
                mtf_analysis = {
                    "base_confidence": signal['confidence'],
                    "mtf_boost": 0.0,
                    "final_confidence": signal['confidence'],
                    "alignment_status": "unknown"
                }
                
                # Prepare indicators
                indicators = {
                    "rsi": signal.get('rsi'),
                    "macd": signal.get('macd'),
                    "volume_ratio": signal.get('volume_ratio')
                }
                
                # Insert into database
                await conn.execute("""
                    INSERT INTO signal_history (
                        signal_id, symbol, timeframe, direction,
                        entry_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        rsi, macd, volume_ratio, indicators,
                        outcome, source, lifecycle_status,
                        signal_timestamp, created_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17,
                        $18, $19, $20, $21, $22
                    )
                    ON CONFLICT (signal_id) DO NOTHING
                """,
                signal_id,
                signal['symbol'],
                signal['timeframe'],
                signal['direction'],
                signal['entry_price'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['confidence'],
                signal['confidence'],  # quality_score = confidence for historical
                signal['pattern_type'],
                json.dumps(sde_consensus),
                json.dumps(mtf_analysis),
                7,  # agreeing_heads
                signal.get('rsi'),
                signal.get('macd'),
                signal.get('volume_ratio'),
                json.dumps(indicators),
                'pending',  # outcome
                'backtest',  # source
                'completed',  # lifecycle_status
                datetime.fromisoformat(signal['timestamp'].replace('Z', '+00:00')),
                datetime.now()
                )
                
                imported_count += 1
                
                if imported_count % 100 == 0:
                    print(f"  Imported {imported_count}/{len(signals)}...")
                    
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 5:
                    print(f"  Warning: Skipped signal - {e}")
        
        print(f"\n{'=' * 70}")
        print(f"Import Complete!")
        print(f"{'=' * 70}")
        print(f"Successfully imported: {imported_count} signals")
        print(f"Skipped (duplicates/errors): {skipped_count} signals")
        
        # Verify import
        total_in_db = await conn.fetchval("SELECT COUNT(*) FROM signal_history WHERE source = 'backtest'")
        print(f"Total backtest signals in database: {total_in_db}")
        
        # Show distribution
        print("\nSignal Distribution by Symbol:")
        distribution = await conn.fetch("""
            SELECT symbol, COUNT(*) as count
            FROM signal_history
            WHERE source = 'backtest'
            GROUP BY symbol
            ORDER BY count DESC
        """)
        
        for row in distribution:
            print(f"  {row['symbol']}: {row['count']}")
        
        print(f"\n{'=' * 70}")
        print("Ready for real-time signal generation!")
        print(f"{'=' * 70}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(import_historical_signals())

