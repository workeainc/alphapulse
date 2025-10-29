"""
Signal Lifecycle Manager
Tracks signal states and validates entry proximity
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import asyncpg

logger = logging.getLogger(__name__)

class SignalLifecycleManager:
    """Manage signal lifecycle and entry proximity validation"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.db_pool = None
        self.is_running = False
        self.current_prices = {}
    
    async def initialize(self):
        """Initialize database connection"""
        self.db_pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)
        logger.info("Signal Lifecycle Manager initialized")
    
    async def close(self):
        """Close database connection"""
        self.is_running = False
        if self.db_pool:
            await self.db_pool.close()
    
    def update_current_price(self, symbol: str, price: float):
        """Update current price for a symbol"""
        self.current_prices[symbol] = price
    
    async def start_monitoring(self):
        """Start monitoring signal lifecycle"""
        self.is_running = True
        logger.info("Starting signal lifecycle monitoring...")
        
        while self.is_running:
            try:
                await self._validate_all_signals()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in lifecycle monitoring: {e}")
                await asyncio.sleep(5)
    
    async def _validate_all_signals(self):
        """Validate all active and pending signals"""
        
        if not self.db_pool:
            return
        
        async with self.db_pool.acquire() as conn:
            # Get all non-final signals
            signals = await conn.fetch("""
                SELECT signal_id, symbol, direction, entry_price, stop_loss,
                       status, created_at
                FROM live_signals
                WHERE status IN ('pending', 'active')
            """)
            
            for signal in signals:
                symbol = signal['symbol']
                current_price = self.current_prices.get(symbol)
                
                if not current_price:
                    continue  # No price data yet
                
                # Calculate entry proximity
                entry_price = float(signal['entry_price'])
                proximity_pct = abs(current_price - entry_price) / current_price
                
                # Determine new status
                new_status = signal['status']
                new_proximity_status = 'waiting'
                invalidation_reason = None
                
                # Check if entry hit
                if signal['direction'] == 'long' and current_price <= entry_price:
                    new_status = 'filled'
                    invalidation_reason = 'Entry price hit'
                elif signal['direction'] == 'short' and current_price >= entry_price:
                    new_status = 'filled'
                    invalidation_reason = 'Entry price hit'
                
                # Check if stop loss hit before entry
                elif signal['direction'] == 'long' and current_price <= float(signal['stop_loss']):
                    new_status = 'invalid'
                    invalidation_reason = 'Stop loss hit before entry'
                elif signal['direction'] == 'short' and current_price >= float(signal['stop_loss']):
                    new_status = 'invalid'
                    invalidation_reason = 'Stop loss hit before entry'
                
                # Check entry proximity
                elif proximity_pct <= 0.005:  # Within 0.5%
                    new_proximity_status = 'imminent'
                    new_status = 'active'
                elif proximity_pct <= 0.02:  # Within 2%
                    new_proximity_status = 'soon'
                    new_status = 'active'
                elif proximity_pct > 0.02:  # More than 2% away
                    new_proximity_status = 'waiting'
                    # If it was active and moved away, mark invalid
                    if signal['status'] == 'active':
                        new_status = 'invalid'
                        invalidation_reason = 'Price moved >2% from entry'
                
                # Check timeout (30 minutes)
                age = datetime.now() - signal['created_at']
                if age > timedelta(minutes=30) and new_status in ('pending', 'active'):
                    new_status = 'expired'
                    invalidation_reason = 'Entry window timeout (30 min)'
                
                # Update if status changed
                if new_status != signal['status']:
                    await conn.execute("""
                        UPDATE live_signals
                        SET status = $1,
                            current_price = $2,
                            entry_proximity_pct = $3,
                            entry_proximity_status = $4,
                            last_validated_at = $5,
                            invalidated_at = CASE WHEN $1 IN ('invalid', 'expired', 'filled') THEN $5 ELSE invalidated_at END,
                            invalidation_reason = $6
                        WHERE signal_id = $7
                    """,
                    new_status,
                    current_price,
                    proximity_pct,
                    new_proximity_status,
                    datetime.now(),
                    invalidation_reason,
                    signal['signal_id']
                    )
                    
                    logger.info(f"{signal['symbol']}: {signal['status']} → {new_status} ({invalidation_reason or 'proximity check'})")
                else:
                    # Just update proximity
                    await conn.execute("""
                        UPDATE live_signals
                        SET current_price = $1,
                            entry_proximity_pct = $2,
                            entry_proximity_status = $3,
                            last_validated_at = $4
                        WHERE signal_id = $5
                    """,
                    current_price,
                    proximity_pct,
                    new_proximity_status,
                    datetime.now(),
                    signal['signal_id']
                    )
    
    async def get_active_signals(self) -> List[Dict]:
        """Get all active signals (ready for display)"""
        
        if not self.db_pool:
            return []
        
        async with self.db_pool.acquire() as conn:
            signals = await conn.fetch("""
                SELECT signal_id, symbol, timeframe, direction,
                       entry_price, current_price, stop_loss, take_profit,
                       confidence, quality_score, pattern_type,
                       entry_proximity_pct, entry_proximity_status,
                       sde_consensus, mtf_analysis, agreeing_heads,
                       status, created_at
                FROM live_signals
                WHERE status = 'active'
                  AND entry_proximity_status IN ('imminent', 'soon')
                ORDER BY quality_score DESC, confidence DESC
                LIMIT 5
            """)
            
            return [dict(row) for row in signals]
    
    async def cleanup_old_signals(self):
        """Remove old filled/invalid/expired signals"""
        
        if not self.db_pool:
            return
        
        async with self.db_pool.acquire() as conn:
            # Keep filled signals for 1 hour, invalid/expired for 5 minutes
            await conn.execute("""
                DELETE FROM live_signals
                WHERE (status = 'filled' AND invalidated_at < NOW() - INTERVAL '1 hour')
                   OR (status IN ('invalid', 'expired') AND invalidated_at < NOW() - INTERVAL '5 minutes')
            """)

