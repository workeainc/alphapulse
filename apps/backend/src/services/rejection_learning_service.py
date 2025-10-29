#!/usr/bin/env python3
"""
Rejection Learning Service
Learns from rejected signals - "the road not taken"
Implements counterfactual learning for complete decision coverage
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import asyncpg
import ccxt

logger = logging.getLogger(__name__)

class RejectionLearningService:
    """
    Tracks rejected signals and learns from what WOULD have happened
    
    Purpose: Learn from ALL decisions, not just the 2% that become signals
    - 98% of scans are rejected
    - These rejections contain valuable learning data
    - Did we reject a winner? (missed opportunity - bad rejection)
    - Did we reject a loser? (good rejection - correct decision)
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange, learning_coordinator):
        self.db_pool = db_pool
        self.exchange = exchange
        self.learning_coordinator = learning_coordinator
        
        # Configuration
        self.monitoring_duration_hours = 48  # Monitor rejected signals for 48 hours
        self.check_interval = 300  # Check every 5 minutes (300 seconds)
        self.min_profit_for_opportunity = 1.0  # 1% profit to count as missed opportunity
        
        # State
        self.is_running = False
        self.monitored_rejections = {}
        
        # Statistics
        self.stats = {
            'rejections_tracked': 0,
            'missed_opportunities': 0,
            'good_rejections': 0,
            'neutral_rejections': 0,
            'learning_from_rejections': 0,
            'checks_performed': 0
        }
        
        logger.info("🚫 Rejection Learning Service initialized")
    
    async def track_rejection(
        self,
        symbol: str,
        timeframe: str,
        signal_candidate: Optional[Dict],
        consensus_data: Optional[Dict],
        rejection_reason: str,
        rejection_stage: str
    ):
        """
        Track a rejected signal for counterfactual learning
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            signal_candidate: Signal data if available (may be None for no_consensus)
            consensus_data: Head votes and consensus data
            rejection_reason: Why it was rejected
            rejection_stage: Which stage rejected it
        """
        try:
            shadow_id = f"SHADOW_{uuid.uuid4().hex[:12].upper()}"
            
            # Extract data from signal_candidate or consensus
            if signal_candidate:
                direction = signal_candidate.get('direction', 'unknown')
                entry_price = signal_candidate.get('entry_price')
                take_profit = signal_candidate.get('take_profit')
                stop_loss = signal_candidate.get('stop_loss')
                sde_consensus = signal_candidate.get('sde_consensus', {})
                indicators = signal_candidate.get('indicators')
                market_regime = signal_candidate.get('regime')
            elif consensus_data:
                # For no_consensus rejections, we still have head votes
                direction = consensus_data.get('direction', 'none')
                entry_price = None  # No price if no consensus
                take_profit = None
                stop_loss = None
                sde_consensus = consensus_data
                indicators = None
                market_regime = None
            else:
                logger.warning(f"No signal or consensus data for rejection tracking")
                return None
            
            # Calculate simulated TP/SL if we have entry price
            if entry_price and not take_profit:
                # Use default 2% TP and 1% SL
                if direction == 'long':
                    take_profit = entry_price * 1.02
                    stop_loss = entry_price * 0.99
                elif direction == 'short':
                    take_profit = entry_price * 0.98
                    stop_loss = entry_price * 1.01
            
            monitor_until = datetime.now(timezone.utc) + timedelta(hours=self.monitoring_duration_hours)
            
            # Store rejection in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rejected_signals (
                        shadow_id, symbol, timeframe, direction,
                        simulated_entry_price, simulated_take_profit, simulated_stop_loss,
                        rejection_reason, rejection_stage,
                        sde_consensus, head_votes, consensus_score, agreeing_heads,
                        indicators, market_regime,
                        simulated_entry_time, monitor_until
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                shadow_id, symbol, timeframe, direction,
                entry_price, take_profit, stop_loss,
                rejection_reason, rejection_stage,
                json.dumps(sde_consensus) if sde_consensus else None,
                json.dumps(sde_consensus.get('heads', {})) if sde_consensus else None,
                sde_consensus.get('confidence') if sde_consensus else None,
                sde_consensus.get('agreeing_heads', 0) if sde_consensus else 0,
                json.dumps(indicators) if indicators else None,
                market_regime,
                datetime.now(timezone.utc),
                monitor_until
                )
            
            self.stats['rejections_tracked'] += 1
            logger.debug(f"📝 Tracked rejection: {shadow_id} ({symbol} {rejection_reason})")
            
            return shadow_id
            
        except Exception as e:
            logger.error(f"❌ Error tracking rejection: {e}")
            return None
    
    async def monitor_shadow_signals(self):
        """
        Main monitoring loop - check rejected signals to see what would have happened
        """
        self.is_running = True
        logger.info("🔍 Starting shadow signal monitoring loop...")
        
        while self.is_running:
            try:
                await self._check_all_shadow_signals()
                self.stats['checks_performed'] += 1
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in shadow monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_shadow_signals(self):
        """
        Check all monitored shadow signals
        """
        try:
            # Get all shadow signals being monitored
            async with self.db_pool.acquire() as conn:
                shadows = await conn.fetch("""
                    SELECT 
                        shadow_id, symbol, direction,
                        simulated_entry_price, simulated_take_profit, simulated_stop_loss,
                        sde_consensus, rejection_reason,
                        simulated_entry_time, monitor_until
                    FROM rejected_signals
                    WHERE monitoring_status = 'monitoring'
                    AND monitor_until > NOW()
                    ORDER BY created_at ASC
                    LIMIT 100
                """)
                
                if not shadows:
                    return
                
                logger.debug(f"🔍 Monitoring {len(shadows)} shadow signals")
                
                # Get current prices
                symbols = list(set([s['symbol'] for s in shadows]))
                current_prices = await self._get_current_prices(symbols)
                
                # Check each shadow signal
                for shadow in shadows:
                    await self._check_shadow_outcome(shadow, current_prices)
                
        except Exception as e:
            logger.error(f"❌ Error checking shadow signals: {e}")
    
    async def _check_shadow_outcome(self, shadow: Dict, current_prices: Dict[str, float]):
        """
        Check if rejected signal would have hit TP or SL
        """
        try:
            symbol = shadow['symbol']
            current_price = current_prices.get(symbol)
            
            if not current_price:
                return
            
            shadow_id = shadow['shadow_id']
            direction = shadow['direction']
            entry_price = float(shadow['simulated_entry_price']) if shadow['simulated_entry_price'] else None
            take_profit = float(shadow['simulated_take_profit']) if shadow['simulated_take_profit'] else None
            stop_loss = float(shadow['simulated_stop_loss']) if shadow['simulated_stop_loss'] else None
            
            if not entry_price or not take_profit or not stop_loss:
                # Can't monitor without prices
                return
            
            # Check if would have hit TP
            if self._would_hit_tp(direction, current_price, take_profit):
                await self._handle_missed_opportunity(shadow, current_price)
                self.stats['missed_opportunities'] += 1
                return
            
            # Check if would have hit SL
            if self._would_hit_sl(direction, current_price, stop_loss):
                await self._handle_good_rejection(shadow, current_price)
                self.stats['good_rejections'] += 1
                return
            
            # Check if monitoring period expired
            if datetime.now(timezone.utc) >= shadow['monitor_until']:
                await self._handle_neutral_rejection(shadow, current_price)
                self.stats['neutral_rejections'] += 1
                return
                
        except Exception as e:
            logger.error(f"❌ Error checking shadow {shadow.get('shadow_id')}: {e}")
    
    def _would_hit_tp(self, direction: str, current_price: float, take_profit: float) -> bool:
        """Check if price would have hit take profit"""
        if direction == 'long':
            return current_price >= take_profit
        else:
            return current_price <= take_profit
    
    def _would_hit_sl(self, direction: str, current_price: float, stop_loss: float) -> bool:
        """Check if price would have hit stop loss"""
        if direction == 'long':
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss
    
    async def _handle_missed_opportunity(self, shadow: Dict, exit_price: float):
        """
        Rejected signal would have won - MISSED OPPORTUNITY
        Learn: Should have generated the signal!
        """
        try:
            shadow_id = shadow['shadow_id']
            entry_price = float(shadow['simulated_entry_price'])
            direction = shadow['direction']
            
            # Calculate what profit would have been
            if direction == 'long':
                profit_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - exit_price) / entry_price) * 100
            
            # Update database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE rejected_signals
                    SET monitoring_status = 'completed',
                        simulated_outcome = 'would_tp',
                        simulated_exit_price = $1,
                        simulated_exit_time = NOW(),
                        simulated_profit_pct = $2,
                        learning_outcome = 'missed_opportunity',
                        completed_at = NOW()
                    WHERE shadow_id = $3
                """, exit_price, profit_pct, shadow_id)
            
            # Prepare data for learning
            rejection_outcome_data = {
                'shadow_id': shadow_id,
                'symbol': shadow['symbol'],
                'direction': direction,
                'outcome_type': 'MISSED_OPPORTUNITY',
                'simulated_profit_pct': profit_pct,
                'sde_consensus': shadow['sde_consensus'],
                'rejection_reason': shadow['rejection_reason']
            }
            
            # Trigger learning from rejection!
            if self.learning_coordinator:
                await self.learning_coordinator.process_rejection_outcome(
                    shadow_id, 
                    rejection_outcome_data
                )
            
            logger.info(f"⚠️ MISSED OPPORTUNITY: {shadow_id} ({shadow['symbol']} {direction}) "
                       f"would have gained {profit_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error handling missed opportunity: {e}")
    
    async def _handle_good_rejection(self, shadow: Dict, exit_price: float):
        """
        Rejected signal would have lost - GOOD REJECTION
        Learn: Rejection was correct!
        """
        try:
            shadow_id = shadow['shadow_id']
            entry_price = float(shadow['simulated_entry_price'])
            direction = shadow['direction']
            
            # Calculate what loss would have been
            if direction == 'long':
                profit_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - exit_price) / entry_price) * 100
            
            # Update database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE rejected_signals
                    SET monitoring_status = 'completed',
                        simulated_outcome = 'would_sl',
                        simulated_exit_price = $1,
                        simulated_exit_time = NOW(),
                        simulated_profit_pct = $2,
                        learning_outcome = 'good_rejection',
                        completed_at = NOW()
                    WHERE shadow_id = $3
                """, exit_price, profit_pct, shadow_id)
            
            # Prepare data for learning
            rejection_outcome_data = {
                'shadow_id': shadow_id,
                'symbol': shadow['symbol'],
                'direction': direction,
                'outcome_type': 'GOOD_REJECTION',
                'simulated_profit_pct': profit_pct,
                'sde_consensus': shadow['sde_consensus'],
                'rejection_reason': shadow['rejection_reason']
            }
            
            # Trigger learning from rejection!
            if self.learning_coordinator:
                await self.learning_coordinator.process_rejection_outcome(
                    shadow_id,
                    rejection_outcome_data
                )
            
            logger.info(f"✅ GOOD REJECTION: {shadow_id} ({shadow['symbol']} {direction}) "
                       f"avoided loss of {abs(profit_pct):.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Error handling good rejection: {e}")
    
    async def _handle_neutral_rejection(self, shadow: Dict, current_price: float):
        """
        Monitoring period expired without hitting TP/SL - NEUTRAL
        """
        try:
            shadow_id = shadow['shadow_id']
            entry_price = float(shadow['simulated_entry_price'])
            direction = shadow['direction']
            
            # Calculate where it ended up
            if direction == 'long':
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Update database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE rejected_signals
                    SET monitoring_status = 'completed',
                        simulated_outcome = 'would_neutral',
                        simulated_exit_price = $1,
                        simulated_exit_time = NOW(),
                        simulated_profit_pct = $2,
                        learning_outcome = 'neutral',
                        completed_at = NOW()
                    WHERE shadow_id = $3
                """, current_price, profit_pct, shadow_id)
            
            self.stats['neutral_rejections'] += 1
            logger.debug(f"⏸️ NEUTRAL: {shadow_id} expired without clear outcome ({profit_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"❌ Error handling neutral rejection: {e}")
    
    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for shadow monitoring"""
        prices = {}
        try:
            for symbol in symbols:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    prices[symbol] = float(ticker['last'])
                except Exception as e:
                    logger.debug(f"Could not fetch price for {symbol}: {e}")
                    continue
            return prices
        except Exception as e:
            logger.error(f"❌ Error fetching prices: {e}")
            return {}
    
    def stop(self):
        """Stop shadow monitoring"""
        self.is_running = False
        logger.info("🛑 Shadow signal monitoring stopped")
    
    def get_stats(self) -> Dict:
        """Get rejection learning statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'monitoring_duration_hours': self.monitoring_duration_hours,
            'missed_opportunity_rate': (
                self.stats['missed_opportunities'] / 
                (self.stats['missed_opportunities'] + self.stats['good_rejections'])
                if (self.stats['missed_opportunities'] + self.stats['good_rejections']) > 0 else 0
            )
        }

