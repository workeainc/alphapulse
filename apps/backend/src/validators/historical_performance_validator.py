"""
Historical Performance Validator
Validates new signals against historical performance from database
REJECTS signals with poor historical win rates
"""

import logging
import numpy as np
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)

class HistoricalPerformanceValidator:
    """
    Validates signals against historical performance
    Learns from YOUR 1,259 backtest signals in database
    """
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.min_win_rate = 0.60  # Need 60%+ historical win rate
        self.min_avg_profit = 3.0  # Need 3%+ average profit
        self.min_sample_size = 5  # Need at least 5 similar signals
        
        logger.info("Historical Performance Validator initialized")
    
    async def validate_signal(self, signal_candidate: Dict) -> Tuple[bool, str]:
        """
        Validate signal against historical similar signals
        
        Returns:
            (is_valid, reason)
        """
        
        try:
            # Query similar historical signals from YOUR database
            similar_signals = await self._query_similar_signals(signal_candidate)
            
            if len(similar_signals) < self.min_sample_size:
                # Not enough historical data - allow signal (give it a chance)
                logger.debug(f"{signal_candidate['symbol']}: Insufficient historical data ({len(similar_signals)} signals)")
                return True, f"Insufficient historical data ({len(similar_signals)} signals)"
            
            # Calculate win rate
            wins = len([s for s in similar_signals if s['outcome'] == 'win'])
            losses = len([s for s in similar_signals if s['outcome'] == 'loss'])
            total = wins + losses
            
            if total == 0:
                return True, "No completed historical signals"
            
            win_rate = wins / total
            
            # === QUALITY GATE: Historical Win Rate ===
            if win_rate < self.min_win_rate:
                logger.info(f"{signal_candidate['symbol']}: REJECTED - Historical win rate too low: {win_rate:.1%} (need {self.min_win_rate:.1%})")
                return False, f"Historical win rate only {win_rate:.1%} (need {self.min_win_rate:.1%})"
            
            # Calculate average profit of wins
            winning_signals = [s for s in similar_signals if s['outcome'] == 'win' and s['profit_loss_pct']]
            
            if winning_signals:
                avg_profit = np.mean([s['profit_loss_pct'] for s in winning_signals])
                
                # === QUALITY GATE: Average Profitability ===
                if avg_profit < self.min_avg_profit:
                    logger.info(f"{signal_candidate['symbol']}: REJECTED - Low historical profitability: {avg_profit:.1f}%")
                    return False, f"Low historical profitability: {avg_profit:.1f}% (need {self.min_avg_profit}%)"
            
            # === PASSED ALL GATES ===
            logger.info(f"{signal_candidate['symbol']}: ✅ Historical validation PASSED - "
                       f"Win rate: {win_rate:.1%}, Sample size: {total}")
            
            return True, f"Historical validation passed: {win_rate:.1%} win rate from {total} similar signals"
            
        except Exception as e:
            logger.error(f"Error in historical validation: {e}")
            return True, f"Validation error: {e}"  # Allow on error
    
    async def _query_similar_signals(self, signal_candidate: Dict) -> List[Dict]:
        """Query similar historical signals from database"""
        
        async with self.db_pool.acquire() as conn:
            # Find similar signals:
            # - Same symbol
            # - Same direction
            # - Similar pattern type
            # - Similar confidence range (±10%)
            
            similar = await conn.fetch("""
                SELECT 
                    signal_id,
                    outcome,
                    profit_loss_pct,
                    confidence,
                    pattern_type
                FROM signal_history
                WHERE symbol = $1
                  AND direction = $2
                  AND pattern_type SIMILAR TO $3
                  AND ABS(confidence - $4) < 0.10
                  AND outcome IN ('win', 'loss', 'breakeven')
                ORDER BY signal_timestamp DESC
                LIMIT 20
            """,
            signal_candidate['symbol'],
            signal_candidate['direction'],
            f"%{signal_candidate.get('pattern_type', 'unknown').split('_')[0]}%",  # Pattern prefix
            signal_candidate['confidence']
            )
            
            return [dict(s) for s in similar]

