"""
Live Signal Generator with SDE Integration
Generates real-time signals using existing SDE framework
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime
from typing import Dict, Optional
import asyncpg

logger = logging.getLogger(__name__)

class LiveSignalGenerator:
    """Generate live trading signals with SDE consensus"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.db_pool = None
    
    async def initialize(self):
        """Initialize database connection"""
        self.db_pool = await asyncpg.create_pool(**self.db_config, min_size=2, max_size=10)
        logger.info("Live Signal Generator initialized")
    
    async def close(self):
        """Close database connection"""
        if self.db_pool:
            await self.db_pool.close()
    
    async def generate_signal_from_candle(self, candle_data: Dict, indicators: Dict) -> Optional[Dict]:
        """
        Generate signal from new candle data
        Uses simplified SDE logic for now
        """
        
        symbol = candle_data['symbol']
        timeframe = candle_data['timeframe']
        current_price = indicators['current_price']
        rsi = indicators['rsi']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        volume_ratio = indicators['volume_ratio']
        
        # Detect pattern and run 9-head analysis
        signal_direction = None
        pattern_type = None
        base_confidence = 0.5
        
        # RSI-based signals
        if rsi < 30:
            signal_direction = 'long'
            pattern_type = 'rsi_oversold'
            base_confidence = 0.65 + (30 - rsi) / 100
        elif rsi > 70:
            signal_direction = 'short'
            pattern_type = 'rsi_overbought'
            base_confidence = 0.65 + (rsi - 70) / 100
        
        # MACD crossover
        elif macd > macd_signal and indicators.get('prev_macd', 0) <= indicators.get('prev_macd_signal', 0):
            signal_direction = 'long'
            pattern_type = 'macd_bullish_cross'
            base_confidence = 0.70
        elif macd < macd_signal and indicators.get('prev_macd', 0) >= indicators.get('prev_macd_signal', 0):
            signal_direction = 'short'
            pattern_type = 'macd_bearish_cross'
            base_confidence = 0.70
        
        # Bollinger bounce
        elif current_price <= bb_lower:
            signal_direction = 'long'
            pattern_type = 'bb_lower_bounce'
            base_confidence = 0.75
        elif current_price >= bb_upper:
            signal_direction = 'short'
            pattern_type = 'bb_upper_bounce'
            base_confidence = 0.75
        
        if not signal_direction:
            return None  # No pattern detected
        
        # Volume confirmation boost
        if volume_ratio > 1.5:
            base_confidence = min(base_confidence + 0.10, 0.95)
            pattern_type = f"{pattern_type}_volume_confirmed"
        
        # Run 9-head SDE consensus (simplified)
        sde_result = self._run_sde_consensus(
            signal_direction, base_confidence, indicators
        )
        
        if not sde_result['consensus_achieved']:
            return None  # No consensus
        
        # Calculate MTF boost (would need data from other timeframes)
        mtf_result = {
            'base_confidence': base_confidence,
            'mtf_boost': 0.0,  # Will be calculated from other timeframes
            'final_confidence': base_confidence,
            'alignment_status': 'weak'
        }
        
        # Calculate entry, SL, TP
        if signal_direction == 'long':
            entry_price = current_price
            stop_loss = current_price * 0.97  # 3% SL
            take_profit = current_price * 1.06  # 6% TP
        else:
            entry_price = current_price
            stop_loss = current_price * 1.03
            take_profit = current_price * 0.94
        
        # Calculate entry proximity (entry = current for live signals)
        entry_proximity_pct = 0.0  # Perfect for new signals
        
        signal = {
            'signal_id': f"LIVE_{uuid.uuid4().hex[:12].upper()}",
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': signal_direction,
            'entry_price': entry_price,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': sde_result['final_confidence'],
            'quality_score': (sde_result['consensus_score'] + base_confidence) / 2,
            'pattern_type': pattern_type,
            'entry_proximity_pct': entry_proximity_pct,
            'entry_proximity_status': 'imminent',  # New signals are always imminent
            'sde_consensus': sde_result,
            'mtf_analysis': mtf_result,
            'agreeing_heads': sde_result['agreeing_heads'],
            'status': 'active',
            'indicators': indicators
        }
        
        return signal
    
    def _run_sde_consensus(self, direction: str, base_confidence: float, indicators: Dict) -> Dict:
        """
        Simplified 9-head SDE consensus
        In production, this would call the full SDE framework
        """
        
        heads = {}
        rsi = indicators['rsi']
        macd = indicators['macd']
        volume_ratio = indicators['volume_ratio']
        
        # Technical Analysis Head
        heads['technical'] = {
            'direction': direction.upper() if rsi < 40 or rsi > 60 else 'FLAT',
            'confidence': base_confidence,
            'weight': 0.13
        }
        
        # Sentiment Head (would use real sentiment in production)
        heads['sentiment'] = {
            'direction': direction.upper(),
            'confidence': base_confidence * 0.9,
            'weight': 0.09
        }
        
        # Volume Head
        heads['volume'] = {
            'direction': direction.upper() if volume_ratio > 1.3 else 'FLAT',
            'confidence': min(base_confidence + 0.05, 0.90),
            'weight': 0.13
        }
        
        # Rule-Based Head
        heads['rules'] = {
            'direction': direction.upper(),
            'confidence': base_confidence,
            'weight': 0.09
        }
        
        # ICT Head
        heads['ict'] = {
            'direction': direction.upper() if base_confidence >= 0.75 else 'FLAT',
            'confidence': base_confidence + 0.10 if base_confidence >= 0.75 else 0.50,
            'weight': 0.13
        }
        
        # Wyckoff Head
        heads['wyckoff'] = {
            'direction': direction.upper() if volume_ratio > 1.5 else 'FLAT',
            'confidence': base_confidence + 0.15 if volume_ratio > 1.5 else 0.60,
            'weight': 0.13
        }
        
        # Harmonic Head
        heads['harmonic'] = {
            'direction': direction.upper() if base_confidence >= 0.75 else 'FLAT',
            'confidence': base_confidence + 0.05,
            'weight': 0.09
        }
        
        # Structure Head
        heads['structure'] = {
            'direction': direction.upper(),
            'confidence': base_confidence,
            'weight': 0.09
        }
        
        # Crypto Head
        heads['crypto'] = {
            'direction': direction.upper(),
            'confidence': base_confidence,
            'weight': 0.12
        }
        
        # Count agreement
        agreeing_heads = sum(1 for h in heads.values() if h['direction'] == direction.upper())
        consensus_achieved = agreeing_heads >= 4  # Need 4/9
        
        # Calculate consensus score
        consensus_score = sum(
            h['confidence'] * h['weight'] 
            for h in heads.values() 
            if h['direction'] == direction.upper()
        )
        
        # Final confidence (weighted average of agreeing heads)
        agreeing_confidences = [h['confidence'] for h in heads.values() if h['direction'] == direction.upper()]
        final_confidence = sum(agreeing_confidences) / len(agreeing_confidences) if agreeing_confidences else 0
        
        return {
            'heads': heads,
            'consensus_achieved': consensus_achieved,
            'agreeing_heads': agreeing_heads,
            'total_heads': 9,
            'consensus_score': round(consensus_score, 2),
            'final_direction': direction.upper() if consensus_achieved else 'FLAT',
            'final_confidence': round(final_confidence, 2),
            'timestamp': datetime.now().isoformat()
        }
    
    async def store_signal(self, signal: Dict):
        """Store signal in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
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
                signal['signal_id'],
                signal['symbol'],
                signal['timeframe'],
                signal['direction'],
                signal['entry_price'],
                signal['current_price'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['confidence'],
                signal['quality_score'],
                signal['pattern_type'],
                signal['entry_proximity_pct'],
                signal['entry_proximity_status'],
                json.dumps(signal['sde_consensus']),
                json.dumps(signal['mtf_analysis']),
                signal['agreeing_heads'],
                signal['status'],
                datetime.now(),
                datetime.now()
                )
                
                logger.info(f"Stored signal: {signal['symbol']} {signal['direction'].upper()} @ {signal['confidence']}")
                
        except Exception as e:
            logger.error(f"Error storing signal: {e}")

