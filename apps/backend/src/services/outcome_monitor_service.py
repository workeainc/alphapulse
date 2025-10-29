#!/usr/bin/env python3
"""
Outcome Monitor Service
Continuously monitors active signals and detects TP/SL hits
Triggers learning when outcomes occur
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import asyncpg
import ccxt
from decimal import Decimal

logger = logging.getLogger(__name__)

class OutcomeMonitorService:
    """
    Monitors active signals and automatically detects when they hit TP/SL
    Triggers learning coordinator when outcomes are detected
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange, learning_coordinator):
        self.db_pool = db_pool
        self.exchange = exchange
        self.learning_coordinator = learning_coordinator
        
        # Configuration
        self.check_interval = 60  # Check every 60 seconds
        self.max_signal_age_hours = 72  # Auto-expire after 72 hours
        self.price_tolerance = 0.001  # 0.1% tolerance for TP/SL detection
        
        # State tracking
        self.is_running = False
        self.last_check_time = None
        self.monitored_signals = {}  # signal_id -> last_price
        
        # Statistics
        self.stats = {
            'checks_performed': 0,
            'tp_hits_detected': 0,
            'sl_hits_detected': 0,
            'time_exits': 0,
            'errors': 0,
            'signals_monitored': 0
        }
        
        logger.info("🔍 Outcome Monitor Service initialized")
    
    async def monitor_active_signals(self):
        """
        Main monitoring loop - runs continuously
        """
        self.is_running = True
        logger.info("🚀 Starting outcome monitoring loop...")
        
        while self.is_running:
            try:
                await self._check_all_active_signals()
                self.stats['checks_performed'] += 1
                self.last_check_time = datetime.now(timezone.utc)
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_active_signals(self):
        """
        Check all active signals for TP/SL hits
        """
        try:
            # Get all active signals from database
            active_signals = await self._get_active_signals()
            
            if not active_signals:
                logger.debug("No active signals to monitor")
                return
            
            self.stats['signals_monitored'] = len(active_signals)
            
            # Get current prices for all symbols
            symbols = list(set([sig['symbol'] for sig in active_signals]))
            current_prices = await self._get_current_prices(symbols)
            
            # Check each signal
            for signal in active_signals:
                await self._check_signal_outcome(signal, current_prices)
                
        except Exception as e:
            logger.error(f"❌ Error checking active signals: {e}")
            raise
    
    async def _get_active_signals(self) -> List[Dict]:
        """
        Retrieve all active signals from database
        """
        try:
            query = """
                SELECT 
                    signal_id, symbol, direction, timeframe,
                    entry_price, current_price, stop_loss, take_profit,
                    confidence, quality_score, pattern_type,
                    sde_consensus, agreeing_heads, indicators,
                    market_regime, created_at
                FROM live_signals
                WHERE status = 'active'
                ORDER BY created_at DESC
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query)
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error fetching active signals: {e}")
            return []
    
    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices from Binance for all symbols
        """
        prices = {}
        try:
            for symbol in symbols:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    prices[symbol] = float(ticker['last'])
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch price for {symbol}: {e}")
                    # Use fallback or skip
                    continue
            
            return prices
            
        except Exception as e:
            logger.error(f"❌ Error fetching prices: {e}")
            return {}
    
    async def _check_signal_outcome(self, signal: Dict, current_prices: Dict[str, float]):
        """
        Check if a signal has hit TP, SL, or should time out
        """
        try:
            symbol = signal['symbol']
            current_price = current_prices.get(symbol)
            
            if not current_price:
                logger.warning(f"⚠️ No price available for {symbol}")
                return
            
            signal_id = signal['signal_id']
            direction = signal['direction']
            entry_price = float(signal['entry_price'])
            take_profit = float(signal['take_profit'])
            stop_loss = float(signal['stop_loss'])
            
            # Check for TP hit
            if self._check_tp_hit(direction, current_price, take_profit):
                logger.info(f"✅ TP HIT detected: {signal_id} ({symbol} {direction})")
                await self._handle_tp_hit(signal, current_price)
                self.stats['tp_hits_detected'] += 1
                return
            
            # Check for SL hit
            if self._check_sl_hit(direction, current_price, stop_loss):
                logger.info(f"🛑 SL HIT detected: {signal_id} ({symbol} {direction})")
                await self._handle_sl_hit(signal, current_price)
                self.stats['sl_hits_detected'] += 1
                return
            
            # Check for time exit
            if self._check_time_exit(signal):
                logger.info(f"⏰ TIME EXIT: {signal_id} ({symbol} {direction})")
                await self._handle_time_exit(signal, current_price)
                self.stats['time_exits'] += 1
                return
            
            # Update current price in database
            await self._update_signal_current_price(signal_id, current_price)
            
        except Exception as e:
            logger.error(f"❌ Error checking signal {signal.get('signal_id')}: {e}")
    
    def _check_tp_hit(self, direction: str, current_price: float, take_profit: float) -> bool:
        """
        Check if take profit has been hit
        """
        if direction.lower() == 'long':
            # For LONG: current_price >= take_profit
            return current_price >= take_profit * (1 - self.price_tolerance)
        else:
            # For SHORT: current_price <= take_profit
            return current_price <= take_profit * (1 + self.price_tolerance)
    
    def _check_sl_hit(self, direction: str, current_price: float, stop_loss: float) -> bool:
        """
        Check if stop loss has been hit
        """
        if direction.lower() == 'long':
            # For LONG: current_price <= stop_loss
            return current_price <= stop_loss * (1 + self.price_tolerance)
        else:
            # For SHORT: current_price >= stop_loss
            return current_price >= stop_loss * (1 - self.price_tolerance)
    
    def _check_time_exit(self, signal: Dict) -> bool:
        """
        Check if signal should be closed due to time
        """
        created_at = signal['created_at']
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        age = datetime.now(timezone.utc) - created_at
        return age > timedelta(hours=self.max_signal_age_hours)
    
    async def _handle_tp_hit(self, signal: Dict, exit_price: float):
        """
        Handle take profit hit - UPDATE database and TRIGGER learning
        """
        try:
            signal_id = signal['signal_id']
            entry_price = float(signal['entry_price'])
            direction = signal['direction']
            
            # Calculate profit/loss
            if direction.lower() == 'long':
                profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100
            
            # Update database
            async with self.db_pool.acquire() as conn:
                # Update live_signals
                await conn.execute("""
                    UPDATE live_signals
                    SET status = 'filled',
                        current_price = $1,
                        last_validated_at = NOW()
                    WHERE signal_id = $2
                """, exit_price, signal_id)
                
                # Insert into signal_history
                await conn.execute("""
                    INSERT INTO signal_history (
                        signal_id, symbol, timeframe, direction,
                        entry_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        rsi, macd, volume_ratio, indicators,
                        outcome, actual_entry_price, actual_exit_price, profit_loss_pct,
                        source, lifecycle_status, signal_timestamp, created_at, completed_at
                    )
                    SELECT 
                        signal_id, symbol, timeframe, direction,
                        entry_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        NULL as rsi, NULL as macd, NULL as volume_ratio, indicators,
                        'win', entry_price, $1, $2,
                        'live', 'completed', created_at, created_at, NOW()
                    FROM live_signals
                    WHERE signal_id = $3
                    ON CONFLICT (signal_id) DO NOTHING
                """, exit_price, profit_loss_pct, signal_id)
            
            # Prepare outcome data for learning
            outcome_data = {
                'signal_id': signal_id,
                'symbol': signal['symbol'],
                'direction': direction,
                'outcome_type': 'TP_HIT',
                'exit_price': exit_price,
                'profit_loss_pct': profit_loss_pct,
                'exit_timestamp': datetime.now(timezone.utc),
                'sde_consensus': signal['sde_consensus'],
                'agreeing_heads': signal['agreeing_heads'],
                'confidence': float(signal['confidence']),
                'quality_score': float(signal['quality_score']),
                'pattern_type': signal['pattern_type'],
                'indicators': signal.get('indicators'),
                'market_regime': signal.get('market_regime')
            }
            
            # Trigger learning! (THIS IS THE KEY)
            if self.learning_coordinator:
                await self.learning_coordinator.process_signal_outcome(signal_id, outcome_data)
            
            logger.info(f"✅ TP HIT processed: {signal_id}, P/L: {profit_loss_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error handling TP hit for {signal_id}: {e}")
    
    async def _handle_sl_hit(self, signal: Dict, exit_price: float):
        """
        Handle stop loss hit - UPDATE database and TRIGGER learning
        """
        try:
            signal_id = signal['signal_id']
            entry_price = float(signal['entry_price'])
            direction = signal['direction']
            
            # Calculate profit/loss (negative)
            if direction.lower() == 'long':
                profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100
            
            # Update database
            async with self.db_pool.acquire() as conn:
                # Update live_signals
                await conn.execute("""
                    UPDATE live_signals
                    SET status = 'filled',
                        current_price = $1,
                        last_validated_at = NOW()
                    WHERE signal_id = $2
                """, exit_price, signal_id)
                
                # Insert into signal_history
                await conn.execute("""
                    INSERT INTO signal_history (
                        signal_id, symbol, timeframe, direction,
                        entry_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        rsi, macd, volume_ratio, indicators,
                        outcome, actual_entry_price, actual_exit_price, profit_loss_pct,
                        source, lifecycle_status, signal_timestamp, created_at, completed_at
                    )
                    SELECT 
                        signal_id, symbol, timeframe, direction,
                        entry_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        NULL as rsi, NULL as macd, NULL as volume_ratio, indicators,
                        'loss', entry_price, $1, $2,
                        'live', 'completed', created_at, created_at, NOW()
                    FROM live_signals
                    WHERE signal_id = $3
                    ON CONFLICT (signal_id) DO NOTHING
                """, exit_price, profit_loss_pct, signal_id)
            
            # Prepare outcome data for learning
            outcome_data = {
                'signal_id': signal_id,
                'symbol': signal['symbol'],
                'direction': direction,
                'outcome_type': 'SL_HIT',
                'exit_price': exit_price,
                'profit_loss_pct': profit_loss_pct,
                'exit_timestamp': datetime.now(timezone.utc),
                'sde_consensus': signal['sde_consensus'],
                'agreeing_heads': signal['agreeing_heads'],
                'confidence': float(signal['confidence']),
                'quality_score': float(signal['quality_score']),
                'pattern_type': signal['pattern_type'],
                'indicators': signal.get('indicators'),
                'market_regime': signal.get('market_regime')
            }
            
            # Trigger learning!
            if self.learning_coordinator:
                await self.learning_coordinator.process_signal_outcome(signal_id, outcome_data)
            
            logger.info(f"🛑 SL HIT processed: {signal_id}, P/L: {profit_loss_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error handling SL hit for {signal_id}: {e}")
    
    async def _handle_time_exit(self, signal: Dict, exit_price: float):
        """
        Handle time-based exit - UPDATE database and TRIGGER learning
        """
        try:
            signal_id = signal['signal_id']
            entry_price = float(signal['entry_price'])
            direction = signal['direction']
            
            # Calculate profit/loss at time exit
            if direction.lower() == 'long':
                profit_loss_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                profit_loss_pct = ((entry_price - exit_price) / entry_price) * 100
            
            # Determine outcome based on P/L
            if profit_loss_pct > 0:
                outcome = 'win'  # Positive but didn't hit TP
            elif profit_loss_pct < -1:
                outcome = 'loss'  # Negative
            else:
                outcome = 'breakeven'  # Near breakeven
            
            # Update database
            async with self.db_pool.acquire() as conn:
                # Update live_signals
                await conn.execute("""
                    UPDATE live_signals
                    SET status = 'expired',
                        current_price = $1,
                        invalidation_reason = 'time_exit',
                        last_validated_at = NOW()
                    WHERE signal_id = $2
                """, exit_price, signal_id)
                
                # Insert into signal_history
                await conn.execute("""
                    INSERT INTO signal_history (
                        signal_id, symbol, timeframe, direction,
                        entry_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        rsi, macd, volume_ratio, indicators,
                        outcome, actual_entry_price, actual_exit_price, profit_loss_pct,
                        source, lifecycle_status, signal_timestamp, created_at, completed_at
                    )
                    SELECT 
                        signal_id, symbol, timeframe, direction,
                        entry_price, stop_loss, take_profit,
                        confidence, quality_score, pattern_type,
                        sde_consensus, mtf_analysis, agreeing_heads,
                        NULL as rsi, NULL as macd, NULL as volume_ratio, indicators,
                        $1, entry_price, $2, $3,
                        'live', 'completed', created_at, created_at, NOW()
                    FROM live_signals
                    WHERE signal_id = $4
                    ON CONFLICT (signal_id) DO NOTHING
                """, outcome, exit_price, profit_loss_pct, signal_id)
            
            # Prepare outcome data for learning
            outcome_data = {
                'signal_id': signal_id,
                'symbol': signal['symbol'],
                'direction': direction,
                'outcome_type': 'TIME_EXIT',
                'exit_price': exit_price,
                'profit_loss_pct': profit_loss_pct,
                'exit_timestamp': datetime.now(timezone.utc),
                'sde_consensus': signal['sde_consensus'],
                'agreeing_heads': signal['agreeing_heads'],
                'confidence': float(signal['confidence']),
                'quality_score': float(signal['quality_score']),
                'pattern_type': signal['pattern_type'],
                'indicators': signal.get('indicators'),
                'market_regime': signal.get('market_regime')
            }
            
            # Trigger learning!
            if self.learning_coordinator:
                await self.learning_coordinator.process_signal_outcome(signal_id, outcome_data)
            
            logger.info(f"⏰ TIME EXIT processed: {signal_id}, P/L: {profit_loss_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error handling time exit for {signal_id}: {e}")
    
    async def _update_signal_current_price(self, signal_id: str, current_price: float):
        """
        Update current price for active signal (for tracking)
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE live_signals
                    SET current_price = $1,
                        last_validated_at = NOW()
                    WHERE signal_id = $2 AND status = 'active'
                """, current_price, signal_id)
        except Exception as e:
            logger.debug(f"Could not update current price for {signal_id}: {e}")
    
    def stop(self):
        """
        Stop the monitoring loop
        """
        self.is_running = False
        logger.info("🛑 Outcome monitor stopped")
    
    def get_stats(self) -> Dict:
        """
        Get monitoring statistics
        """
        return {
            **self.stats,
            'is_running': self.is_running,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'check_interval_seconds': self.check_interval
        }

