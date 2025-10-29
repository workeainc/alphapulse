#!/usr/bin/env python3
"""
SDE Framework Database Integration for AlphaPlus
Integrates SDE model heads with TimescaleDB for signal generation and storage
"""

import asyncio
import logging
import asyncpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import json

from .model_heads import ModelHeadsManager, ModelHeadResult, SignalDirection
from .consensus_manager import ConsensusManager

logger = logging.getLogger(__name__)

@dataclass
class SignalGenerationRequest:
    """Request for signal generation"""
    symbol: str
    timeframe: str
    market_data: Dict[str, Any]
    analysis_results: Dict[str, Any]
    timestamp: datetime

@dataclass
class SignalGenerationResult:
    """Result of signal generation"""
    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    confidence: float
    strength: float
    model_head_results: List[ModelHeadResult]
    consensus_result: Dict[str, Any]
    technical_indicators: Dict[str, float]
    market_conditions: Dict[str, Any]
    timestamp: datetime

class SDEDatabaseIntegration:
    """SDE Framework integrated with TimescaleDB for signal generation"""
    
    def __init__(self, db_url: str = None):
        """
        Initialize SDE-Database integration
        
        Args:
            db_url: TimescaleDB connection URL
        """
        self.db_url = db_url or "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.db_pool = None
        
        # Initialize SDE components
        self.model_heads_manager = ModelHeadsManager()
        self.consensus_manager = ConsensusManager()
        
        # Performance tracking
        self.stats = {
            'signals_generated': 0,
            'signals_stored': 0,
            'consensus_reached': 0,
            'consensus_failed': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            logger.info("✅ SDE-Database integration initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SDE-Database integration: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 SDE-Database integration closed")
    
    async def generate_signal(self, request: SignalGenerationRequest) -> Optional[SignalGenerationResult]:
        """
        Generate trading signal using SDE framework and store in database
        
        Args:
            request: Signal generation request
            
        Returns:
            SignalGenerationResult if successful, None otherwise
        """
        try:
            logger.info(f"🎯 Generating signal for {request.symbol} {request.timeframe}")
            
            # Step 1: Run all model heads
            logger.info(f"Running model heads with market_data: {request.market_data}")
            logger.info(f"Running model heads with analysis_results: {request.analysis_results}")
            
            model_head_results = await self.model_heads_manager.analyze_all_heads(
                request.market_data, 
                request.analysis_results
            )
            
            logger.info(f"Model head results: {model_head_results}")
            
            if not model_head_results:
                logger.warning("No model head results available")
                return None
            
            # Step 2: Run consensus mechanism
            consensus_result = await self.consensus_manager.check_consensus(model_head_results)
            
            if not consensus_result.consensus_achieved:
                logger.info("No consensus reached among model heads")
                self.stats['consensus_failed'] += 1
                return None
            
            # Step 3: Get technical indicators from database
            technical_indicators = await self._get_technical_indicators(
                request.symbol, 
                request.timeframe
            )
            
            # Step 4: Analyze market conditions
            market_conditions = await self._analyze_market_conditions(
                request.symbol, 
                request.timeframe
            )
            
            # Step 5: Create signal result
            signal_id = f"{request.symbol}_{request.timeframe}_{request.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            result = SignalGenerationResult(
                signal_id=signal_id,
                symbol=request.symbol,
                timeframe=request.timeframe,
                direction=consensus_result.consensus_direction.value if consensus_result.consensus_direction else 'flat',
                confidence=consensus_result.consensus_score,
                strength=consensus_result.consensus_score,
                model_head_results=model_head_results,
                consensus_result=consensus_result,
                technical_indicators=technical_indicators,
                market_conditions=market_conditions,
                timestamp=request.timestamp
            )
            
            # Step 6: Store signal in database
            await self._store_signal(result)
            
            # Update stats
            self.stats['signals_generated'] += 1
            self.stats['signals_stored'] += 1
            self.stats['consensus_reached'] += 1
            
            logger.info(f"✅ Signal generated: {signal_id} - {result.direction} (conf: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {e}")
            self.stats['errors'] += 1
            return None
    
    async def _get_technical_indicators(self, symbol: str, timeframe: str) -> Dict[str, float]:
        """Get latest technical indicators from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get latest indicators
                rows = await conn.fetch("""
                    SELECT indicator_name, indicator_value
                    FROM technical_indicators
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '1 hour'
                    ORDER BY timestamp DESC
                """, symbol, timeframe)
                
                indicators = {}
                for row in rows:
                    indicators[row['indicator_name']] = float(row['indicator_value'])
                
                return indicators
                
        except Exception as e:
            logger.error(f"❌ Error getting technical indicators: {e}")
            return {}
    
    async def _analyze_market_conditions(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze current market conditions"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get latest OHLCV data
                ohlcv_data = await conn.fetch("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_data
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp DESC
                    LIMIT 20
                """, symbol, timeframe)
                
                if not ohlcv_data:
                    return {'status': 'insufficient_data'}
                
                # Convert to DataFrame
                df = pd.DataFrame([dict(row) for row in ohlcv_data])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Calculate market conditions
                current_price = df['close'].iloc[-1]
                price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100 if len(df) > 1 else 0
                volume_avg = df['volume'].mean()
                volume_ratio = df['volume'].iloc[-1] / volume_avg if volume_avg > 0 else 1
                
                # Volatility calculation
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(24) if len(returns) > 1 else 0  # Daily volatility
                
                # Trend analysis
                sma_short = df['close'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else current_price
                sma_long = df['close'].rolling(window=10).mean().iloc[-1] if len(df) >= 10 else current_price
                trend_direction = 'bullish' if sma_short > sma_long else 'bearish'
                
                return {
                    'current_price': float(current_price),
                    'price_change_pct': float(price_change),
                    'volume_ratio': float(volume_ratio),
                    'volatility': float(volatility),
                    'trend_direction': trend_direction,
                    'sma_short': float(sma_short),
                    'sma_long': float(sma_long),
                    'data_points': len(df)
                }
                
        except Exception as e:
            logger.error(f"❌ Error analyzing market conditions: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _store_signal(self, result: SignalGenerationResult):
        """Store generated signal in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store in signals table
                await conn.execute("""
                    INSERT INTO signals (
                        signal_id, symbol, timeframe, direction, confidence, entry_price,
                        pattern_type, volume_confirmation, trend_alignment, market_regime,
                        indicators, validation_metrics, timestamp, outcome
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                result.signal_id,
                result.symbol,
                result.timeframe,
                result.direction,
                result.confidence,
                result.market_conditions.get('current_price', 0),
                'sde_consensus',
                result.market_conditions.get('volume_ratio', 1) > 1.5,
                result.market_conditions.get('trend_direction') == result.direction,
                result.market_conditions.get('trend_direction', 'unknown'),
                json.dumps(result.technical_indicators),
                json.dumps({
                    'strength': result.strength,
                    'consensus_details': result.consensus_result,
                    'model_head_count': len(result.model_head_results),
                    'market_conditions': result.market_conditions
                }),
                result.timestamp,
                'pending'
                )
                
                # Store model head results
                for head_result in result.model_head_results:
                    await conn.execute("""
                        INSERT INTO ml_predictions (
                            timestamp, symbol, model_name, model_type, prediction, confidence,
                            features_used, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    result.timestamp,
                    result.symbol,
                    head_result.head_type.value,
                    'model_head',
                    head_result.probability,
                    head_result.confidence,
                    json.dumps(head_result.features_used),
                    result.timestamp
                    )
                
                logger.info(f"✅ Signal stored in database: {result.signal_id}")
                
        except Exception as e:
            logger.error(f"❌ Error storing signal: {e}")
            raise
    
    async def get_recent_signals(self, symbol: str, timeframe: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent signals from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT signal_id, symbol, timeframe, direction, confidence, entry_price,
                           pattern_type, volume_confirmation, trend_alignment, market_regime,
                           indicators, validation_metrics, timestamp, outcome
                    FROM signals
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp DESC
                """ % hours, symbol, timeframe)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return []
    
    async def get_signal_performance(self, symbol: str, timeframe: str, days: int = 30) -> Dict[str, Any]:
        """Get signal performance metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get signal statistics
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN outcome = 'success' THEN 1 END) as successful_signals,
                        COUNT(CASE WHEN outcome = 'failure' THEN 1 END) as failed_signals,
                        AVG(confidence) as avg_confidence,
                        AVG(CASE WHEN outcome = 'success' THEN confidence END) as avg_success_confidence,
                        AVG(CASE WHEN outcome = 'failure' THEN confidence END) as avg_failure_confidence
                    FROM signals
                    WHERE symbol = $1 AND timeframe = $2
                    AND timestamp >= NOW() - INTERVAL '%s days'
                """ % days, symbol, timeframe)
                
                if not stats or stats['total_signals'] == 0:
                    return {'status': 'no_data'}
                
                success_rate = (stats['successful_signals'] / stats['total_signals']) * 100
                
                return {
                    'total_signals': stats['total_signals'],
                    'successful_signals': stats['successful_signals'],
                    'failed_signals': stats['failed_signals'],
                    'success_rate': float(success_rate),
                    'avg_confidence': float(stats['avg_confidence']) if stats['avg_confidence'] else 0,
                    'avg_success_confidence': float(stats['avg_success_confidence']) if stats['avg_success_confidence'] else 0,
                    'avg_failure_confidence': float(stats['avg_failure_confidence']) if stats['avg_failure_confidence'] else 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting signal performance: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get SDE-Database integration statistics"""
        return {
            'signals_generated': self.stats['signals_generated'],
            'signals_stored': self.stats['signals_stored'],
            'consensus_reached': self.stats['consensus_reached'],
            'consensus_failed': self.stats['consensus_failed'],
            'errors': self.stats['errors'],
            'consensus_rate': (self.stats['consensus_reached'] / max(self.stats['consensus_reached'] + self.stats['consensus_failed'], 1)) * 100
        }

# Example usage
async def main():
    """Example usage of SDE-Database integration"""
    integration = SDEDatabaseIntegration()
    
    try:
        await integration.initialize()
        
        # Create sample signal generation request
        request = SignalGenerationRequest(
            symbol="BTCUSDT",
            timeframe="1m",
            market_data={
                'current_price': 45000.0,
                'indicators': {
                    'sma_20': 44800.0,
                    'sma_50': 44500.0,
                    'rsi_14': 35.2,
                    'macd': 0.85
                }
            },
            analysis_results={
                'sentiment_analysis': {
                    'overall_sentiment': 0.3,
                    'confidence': 0.8
                }
            },
            timestamp=datetime.now(timezone.utc)
        )
        
        # Generate signal
        result = await integration.generate_signal(request)
        
        if result:
            print(f"Signal generated: {result.signal_id}")
            print(f"Direction: {result.direction}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Strength: {result.strength:.3f}")
        
        # Get recent signals
        recent_signals = await integration.get_recent_signals("BTCUSDT", "1m", 24)
        print(f"Recent signals: {len(recent_signals)}")
        
        # Get performance
        performance = await integration.get_signal_performance("BTCUSDT", "1m", 30)
        print(f"Performance: {performance}")
        
        # Get stats
        stats = integration.get_integration_stats()
        print(f"Integration stats: {stats}")
        
    finally:
        await integration.close()

if __name__ == "__main__":
    asyncio.run(main())
