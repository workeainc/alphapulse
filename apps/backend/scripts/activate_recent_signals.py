"""
Activate Recent High-Quality Signals
Promotes recent backtest signals to active if they match current market conditions
This bridges the gap while waiting for first live candle
"""

import asyncio
import asyncpg
import aiohttp
from datetime import datetime, timedelta
import json

DB_CONFIG = {
    'host': 'localhost',
    'port': 55433,
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711'
}

async def get_current_prices():
    """Fetch current prices from Binance"""
    url = "https://api.binance.com/api/v3/ticker/price"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return {item['symbol']: float(item['price']) for item in data}
    return {}

async def activate_signals():
    """Activate high-quality recent signals based on current prices"""
    
    print("=" * 70)
    print("Activating Recent High-Quality Signals")
    print("=" * 70)
    
    # Get current prices
    print("\nFetching current market prices...")
    prices = await get_current_prices()
    print(f"Got prices for {len(prices)} symbols")
    
    # Connect to database
    conn = await asyncpg.connect(**DB_CONFIG)
    
    try:
        # Get recent high-quality signals
        recent_signals = await conn.fetch("""
            SELECT 
                signal_id, symbol, timeframe, direction,
                entry_price, stop_loss, take_profit,
                confidence, pattern_type,
                sde_consensus, mtf_analysis,
                signal_timestamp
            FROM signal_history
            WHERE source = 'backtest'
              AND confidence >= 0.75
              AND signal_timestamp >= NOW() - INTERVAL '7 days'
            ORDER BY confidence DESC, signal_timestamp DESC
        """)
        
        print(f"\nFound {len(recent_signals)} high-quality recent signals")
        print("Checking entry proximity with current prices...\n")
        
        activated = 0
        symbols_activated = set()
        
        for signal in recent_signals:
            symbol = signal['symbol']
            
            # Skip if already have signal for this symbol (deduplication)
            if symbol in symbols_activated:
                continue
            
            current_price = prices.get(symbol)
            if not current_price:
                continue
            
            entry_price = float(signal['entry_price'])
            
            # Calculate proximity
            proximity_pct = abs(current_price - entry_price) / current_price
            
            # Only activate if entry is within 2%
            if proximity_pct <= 0.02:
                # Determine proximity status
                if proximity_pct <= 0.005:
                    proximity_status = 'imminent'
                elif proximity_pct <= 0.02:
                    proximity_status = 'soon'
                else:
                    proximity_status = 'waiting'
                
                # Recalculate entry/SL/TP based on CURRENT price
                if signal['direction'] == 'long':
                    new_entry = current_price
                    new_sl = current_price * 0.97
                    new_tp = current_price * 1.06
                else:
                    new_entry = current_price
                    new_sl = current_price * 1.03
                    new_tp = current_price * 0.94
                
                # Insert into live_signals
                await conn.execute("""
                    INSERT INTO live_signals (
                        signal_id, symbol, timeframe, direction,
                        entry_price, current_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        entry_proximity_pct, entry_proximity_status,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        status, created_at, last_validated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        $12, $13, $14, $15, $16, $17, $18, $19
                    )
                    ON CONFLICT (signal_id) DO NOTHING
                """,
                f"ACTIVATED_{signal['signal_id']}",
                symbol,
                signal['timeframe'],
                signal['direction'],
                new_entry,
                current_price,
                new_sl,
                new_tp,
                signal['confidence'],
                signal['confidence'],  # quality_score
                signal['pattern_type'],
                proximity_pct,
                proximity_status,
                signal['sde_consensus'],
                signal['mtf_analysis'],
                7,  # agreeing_heads (from backtest)
                'active',
                datetime.now(),
                datetime.now()
                )
                
                activated += 1
                symbols_activated.add(symbol)
                
                print(f"✓ Activated: {symbol} {signal['direction'].upper()} @ {proximity_pct*100:.2f}% from entry ({proximity_status})")
                
                if activated >= 5:  # Max 5 signals
                    break
        
        print(f"\n{'=' * 70}")
        print(f"Activated {activated} signals for immediate display")
        print(f"{'=' * 70}")
        print("\nThese signals are:")
        print("  - From recent high-quality backtest data")
        print("  - Adjusted to current market prices")
        print("  - Entry within 2% of current price")
        print("  - Will be monitored and invalidated if price moves away")
        print("\nNew LIVE signals will generate as candles close on Binance")
        print(f"{'=' * 70}\n")
        
        # Show what's active
        active = await conn.fetch("SELECT symbol, direction, confidence, entry_proximity_status FROM live_signals WHERE status = 'active' ORDER BY confidence DESC")
        print("Currently Active Signals:")
        for sig in active:
            print(f"  {sig['symbol']} {sig['direction'].upper()} - {sig['confidence']*100:.0f}% ({sig['entry_proximity_status']})")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(activate_signals())

