"""
Enhanced Indicators Integration Service for AlphaPlus
Seamless integration of enhanced indicators with existing system
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from src.core.enhanced_indicators_engine import EnhancedIndicatorsEngine, EnhancedIndicatorValues
from src.core.indicators_engine import TechnicalIndicators, IndicatorValues

logger = logging.getLogger(__name__)

class EnhancedIndicatorsIntegration:
    """
    Enhanced Indicators Integration Service
    Provides seamless integration between enhanced and legacy indicator systems
    """
    
    def __init__(self, 
                 db_session: AsyncSession,
                 redis_client: Optional[redis.Redis] = None,
                 enable_enhanced: bool = True):
        """Initialize enhanced indicators integration"""
        self.db_session = db_session
        self.redis_client = redis_client
        self.enable_enhanced = enable_enhanced
        
        # Initialize both engines
        self.enhanced_engine = EnhancedIndicatorsEngine(redis_client, db_session)
        self.legacy_engine = TechnicalIndicators()
        
        # Performance tracking
        self.stats = {
            'enhanced_calculations': 0,
            'legacy_calculations': 0,
            'cache_hits': 0,
            'avg_enhanced_time_ms': 0.0,
            'avg_legacy_time_ms': 0.0,
            'total_requests': 0
        }
        
        # Configuration
        self.config = {
            'use_enhanced_by_default': enable_enhanced,
            'fallback_to_legacy': True,
            'cache_ttl_seconds': 300,  # 5 minutes
            'performance_threshold_ms': 50,  # Switch to legacy if enhanced > 50ms
            'batch_size': 100
        }
        
        logger.info(f"Enhanced Indicators Integration initialized (enhanced: {enable_enhanced})")
    
    async def calculate_indicators(self, 
                                 df: pd.DataFrame,
                                 symbol: str,
                                 timeframe: str,
                                 force_legacy: bool = False) -> Union[EnhancedIndicatorValues, IndicatorValues]:
        """
        Calculate indicators using the best available engine
        Returns enhanced indicators if available, falls back to legacy
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Check cache first
            cached_result = await self._get_cached_indicators(symbol, timeframe)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
            
            # Use enhanced engine if enabled and not forced to legacy
            if self.enable_enhanced and not force_legacy:
                try:
                    result = await self._calculate_enhanced_indicators(df, symbol, timeframe)
                    calculation_time = (time.time() - start_time) * 1000
                    self._update_enhanced_stats(calculation_time)
                    
                    # Cache the result
                    await self._cache_indicators(result, symbol, timeframe)
                    
                    return result
                    
                except Exception as e:
                    logger.warning(f"Enhanced calculation failed, falling back to legacy: {e}")
                    if self.config['fallback_to_legacy']:
                        return await self._calculate_legacy_indicators(df, symbol, timeframe)
                    else:
                        raise
            
            # Use legacy engine
            result = await self._calculate_legacy_indicators(df, symbol, timeframe)
            calculation_time = (time.time() - start_time) * 1000
            self._update_legacy_stats(calculation_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            raise
    
    async def _calculate_enhanced_indicators(self, 
                                           df: pd.DataFrame,
                                           symbol: str,
                                           timeframe: str) -> EnhancedIndicatorValues:
        """Calculate indicators using enhanced engine"""
        return await self.enhanced_engine.calculate_all_indicators(df, symbol, timeframe)
    
    async def _calculate_legacy_indicators(self, 
                                         df: pd.DataFrame,
                                         symbol: str,
                                         timeframe: str) -> IndicatorValues:
        """Calculate indicators using legacy engine"""
        # Convert DataFrame to list format expected by legacy engine
        close_prices = df['close'].tolist()
        
        if len(close_prices) == 0:
            raise ValueError("No data available for indicator calculation")
        
        # Get the latest OHLCV values
        latest_row = df.iloc[-1]
        
        return self.legacy_engine.calculate_all_indicators(
            open_price=latest_row['open'],
            high=latest_row['high'],
            low=latest_row['low'],
            close=latest_row['close'],
            volume=latest_row['volume'],
            close_prices=close_prices
        )
    
    async def _get_cached_indicators(self, symbol: str, timeframe: str) -> Optional[Union[EnhancedIndicatorValues, IndicatorValues]]:
        """Get indicators from cache"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"indicators:{symbol}:{timeframe}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                # Try to parse as enhanced indicators first
                try:
                    # This is a simplified cache implementation
                    # In a real system, you'd serialize/deserialize properly
                    return None  # For now, return None to force recalculation
                except:
                    return None
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_indicators(self, 
                              indicators: Union[EnhancedIndicatorValues, IndicatorValues],
                              symbol: str,
                              timeframe: str):
        """Cache indicators in Redis"""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"indicators:{symbol}:{timeframe}"
            # In a real implementation, you'd serialize the indicators properly
            # For now, we'll skip caching to avoid serialization complexity
            pass
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def get_indicators_from_timescaledb(self,
                                            symbol: str,
                                            timeframe: str,
                                            hours_back: int = 24,
                                            use_aggregates: bool = True) -> pd.DataFrame:
        """
        Get indicators from TimescaleDB with optional continuous aggregates
        """
        try:
            if use_aggregates:
                # Use continuous aggregates for better performance
                query = self._build_aggregate_query(symbol, timeframe, hours_back)
            else:
                # Use raw data
                query = self._build_raw_query(symbol, timeframe, hours_back)
            
            result = await self.db_session.execute(text(query))
            rows = result.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=result.keys())
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get indicators from TimescaleDB: {e}")
            return pd.DataFrame()
    
    def _build_aggregate_query(self, symbol: str, timeframe: str, hours_back: int) -> str:
        """Build query using continuous aggregates"""
        time_bucket = self._get_appropriate_time_bucket(hours_back)
        
        return f"""
            SELECT 
                bucket as timestamp,
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume,
                avg_rsi as rsi,
                avg_macd as macd,
                avg_macd_signal as macd_signal,
                avg_bollinger_upper as bb_upper,
                avg_bollinger_middle as bb_middle,
                avg_bollinger_lower as bb_lower,
                avg_atr as atr,
                avg_market_sentiment as market_sentiment,
                data_points
            FROM enhanced_market_data_{time_bucket}
            WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND bucket >= NOW() - INTERVAL '{hours_back} hours'
            ORDER BY bucket DESC
        """
    
    def _build_raw_query(self, symbol: str, timeframe: str, hours_back: int) -> str:
        """Build query using raw data"""
        return f"""
            SELECT 
                timestamp,
                symbol,
                timeframe,
                open,
                high,
                low,
                close,
                volume,
                rsi,
                macd,
                macd_signal,
                bollinger_upper as bb_upper,
                bollinger_middle as bb_middle,
                bollinger_lower as bb_lower,
                atr,
                market_sentiment
            FROM enhanced_market_data
            WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND timestamp >= NOW() - INTERVAL '{hours_back} hours'
            ORDER BY timestamp DESC
        """
    
    def _get_appropriate_time_bucket(self, hours_back: int) -> str:
        """Get appropriate time bucket based on time range"""
        if hours_back <= 1:
            return "5m"
        elif hours_back <= 4:
            return "15m"
        elif hours_back <= 24:
            return "1h"
        elif hours_back <= 168:  # 7 days
            return "4h"
        else:
            return "1d"
    
    async def get_analysis_summary(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis summary using continuous aggregates
        """
        try:
            # Get RSI analysis
            rsi_analysis = await self._get_rsi_analysis(symbol, timeframe)
            
            # Get MACD analysis
            macd_analysis = await self._get_macd_analysis(symbol, timeframe)
            
            # Get Bollinger Bands analysis
            bb_analysis = await self._get_bollinger_analysis(symbol, timeframe)
            
            # Get market sentiment analysis
            sentiment_analysis = await self._get_sentiment_analysis(symbol, timeframe)
            
            # Get volatility analysis
            volatility_analysis = await self._get_volatility_analysis(symbol, timeframe)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now(timezone.utc),
                'rsi_analysis': rsi_analysis,
                'macd_analysis': macd_analysis,
                'bollinger_analysis': bb_analysis,
                'sentiment_analysis': sentiment_analysis,
                'volatility_analysis': volatility_analysis,
                'overall_signal': self._calculate_overall_signal(
                    rsi_analysis, macd_analysis, bb_analysis, 
                    sentiment_analysis, volatility_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis summary: {e}")
            return {}
    
    async def _get_rsi_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get RSI trend analysis"""
        query = f"""
            SELECT 
                avg_rsi,
                min_rsi,
                max_rsi,
                rsi_volatility,
                rsi_regime,
                data_points
            FROM rsi_trend_analysis
            WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND bucket >= NOW() - INTERVAL '24 hours'
            ORDER BY bucket DESC
            LIMIT 1
        """
        
        result = await self.db_session.execute(text(query))
        row = result.fetchone()
        
        if row:
            return {
                'current_rsi': float(row.avg_rsi) if row.avg_rsi else 50.0,
                'min_rsi': float(row.min_rsi) if row.min_rsi else 0.0,
                'max_rsi': float(row.max_rsi) if row.max_rsi else 100.0,
                'volatility': float(row.rsi_volatility) if row.rsi_volatility else 0.0,
                'regime': row.rsi_regime or 'neutral',
                'data_points': int(row.data_points) if row.data_points else 0
            }
        
        return {'current_rsi': 50.0, 'regime': 'neutral'}
    
    async def _get_macd_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get MACD signal analysis"""
        query = f"""
            SELECT 
                avg_macd,
                avg_macd_signal,
                avg_macd_histogram,
                macd_signal,
                data_points
            FROM macd_signal_analysis
            WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND bucket >= NOW() - INTERVAL '24 hours'
            ORDER BY bucket DESC
            LIMIT 1
        """
        
        result = await self.db_session.execute(text(query))
        row = result.fetchone()
        
        if row:
            return {
                'macd': float(row.avg_macd) if row.avg_macd else 0.0,
                'signal': float(row.avg_macd_signal) if row.avg_macd_signal else 0.0,
                'histogram': float(row.avg_macd_histogram) if row.avg_macd_histogram else 0.0,
                'signal_direction': row.macd_signal or 'neutral',
                'data_points': int(row.data_points) if row.data_points else 0
            }
        
        return {'macd': 0.0, 'signal': 0.0, 'signal_direction': 'neutral'}
    
    async def _get_bollinger_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get Bollinger Bands analysis"""
        query = f"""
            SELECT 
                avg_bb_upper,
                avg_bb_middle,
                avg_bb_lower,
                avg_bb_width,
                avg_bb_position,
                bb_position,
                data_points
            FROM bollinger_bands_analysis
            WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND bucket >= NOW() - INTERVAL '24 hours'
            ORDER BY bucket DESC
            LIMIT 1
        """
        
        result = await self.db_session.execute(text(query))
        row = result.fetchone()
        
        if row:
            return {
                'upper': float(row.avg_bb_upper) if row.avg_bb_upper else 0.0,
                'middle': float(row.avg_bb_middle) if row.avg_bb_middle else 0.0,
                'lower': float(row.avg_bb_lower) if row.avg_bb_lower else 0.0,
                'width': float(row.avg_bb_width) if row.avg_bb_width else 0.0,
                'position': float(row.avg_bb_position) if row.avg_bb_position else 0.5,
                'position_type': row.bb_position or 'within_bands',
                'data_points': int(row.data_points) if row.data_points else 0
            }
        
        return {'position_type': 'within_bands', 'position': 0.5}
    
    async def _get_sentiment_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get market sentiment analysis"""
        query = f"""
            SELECT 
                avg_sentiment,
                min_sentiment,
                max_sentiment,
                sentiment_volatility,
                sentiment_regime,
                data_points
            FROM market_sentiment_analysis
            WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND bucket >= NOW() - INTERVAL '24 hours'
            ORDER BY bucket DESC
            LIMIT 1
        """
        
        result = await self.db_session.execute(text(query))
        row = result.fetchone()
        
        if row:
            return {
                'sentiment': float(row.avg_sentiment) if row.avg_sentiment else 0.5,
                'min_sentiment': float(row.min_sentiment) if row.min_sentiment else 0.0,
                'max_sentiment': float(row.max_sentiment) if row.max_sentiment else 1.0,
                'volatility': float(row.sentiment_volatility) if row.sentiment_volatility else 0.0,
                'regime': row.sentiment_regime or 'neutral',
                'data_points': int(row.data_points) if row.data_points else 0
            }
        
        return {'sentiment': 0.5, 'regime': 'neutral'}
    
    async def _get_volatility_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get volatility analysis"""
        query = f"""
            SELECT 
                avg_volatility,
                min_volatility,
                max_volatility,
                avg_atr,
                volatility_regime,
                data_points
            FROM volatility_analysis
            WHERE symbol = '{symbol}'
                AND timeframe = '{timeframe}'
                AND bucket >= NOW() - INTERVAL '24 hours'
            ORDER BY bucket DESC
            LIMIT 1
        """
        
        result = await self.db_session.execute(text(query))
        row = result.fetchone()
        
        if row:
            return {
                'volatility': float(row.avg_volatility) if row.avg_volatility else 0.0,
                'min_volatility': float(row.min_volatility) if row.min_volatility else 0.0,
                'max_volatility': float(row.max_volatility) if row.max_volatility else 0.0,
                'atr': float(row.avg_atr) if row.avg_atr else 0.0,
                'regime': row.volatility_regime or 'normal',
                'data_points': int(row.data_points) if row.data_points else 0
            }
        
        return {'volatility': 0.0, 'regime': 'normal'}
    
    def _calculate_overall_signal(self, 
                                rsi_analysis: Dict,
                                macd_analysis: Dict,
                                bb_analysis: Dict,
                                sentiment_analysis: Dict,
                                volatility_analysis: Dict) -> Dict[str, Any]:
        """Calculate overall trading signal based on all analyses"""
        
        # RSI signal
        rsi_signal = 'neutral'
        if rsi_analysis.get('current_rsi', 50) > 70:
            rsi_signal = 'sell'
        elif rsi_analysis.get('current_rsi', 50) < 30:
            rsi_signal = 'buy'
        
        # MACD signal
        macd_signal = macd_analysis.get('signal_direction', 'neutral')
        
        # Bollinger Bands signal
        bb_signal = 'neutral'
        bb_position = bb_analysis.get('position_type', 'within_bands')
        if bb_position == 'above_upper':
            bb_signal = 'sell'
        elif bb_position == 'below_lower':
            bb_signal = 'buy'
        
        # Sentiment signal
        sentiment_signal = 'neutral'
        sentiment_regime = sentiment_analysis.get('regime', 'neutral')
        if sentiment_regime == 'bullish':
            sentiment_signal = 'buy'
        elif sentiment_regime == 'bearish':
            sentiment_signal = 'sell'
        
        # Calculate signal strength
        signals = [rsi_signal, macd_signal, bb_signal, sentiment_signal]
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        
        if buy_signals > sell_signals:
            overall_signal = 'buy'
            strength = buy_signals / len(signals)
        elif sell_signals > buy_signals:
            overall_signal = 'sell'
            strength = sell_signals / len(signals)
        else:
            overall_signal = 'neutral'
            strength = 0.5
        
        return {
            'signal': overall_signal,
            'strength': strength,
            'confidence': self._calculate_confidence(
                rsi_analysis, macd_analysis, bb_analysis, 
                sentiment_analysis, volatility_analysis
            ),
            'components': {
                'rsi': rsi_signal,
                'macd': macd_signal,
                'bollinger_bands': bb_signal,
                'sentiment': sentiment_signal
            }
        }
    
    def _calculate_confidence(self, 
                            rsi_analysis: Dict,
                            macd_analysis: Dict,
                            bb_analysis: Dict,
                            sentiment_analysis: Dict,
                            volatility_analysis: Dict) -> float:
        """Calculate confidence level based on data quality and consistency"""
        
        # Base confidence on data points
        total_data_points = sum([
            rsi_analysis.get('data_points', 0),
            macd_analysis.get('data_points', 0),
            bb_analysis.get('data_points', 0),
            sentiment_analysis.get('data_points', 0),
            volatility_analysis.get('data_points', 0)
        ])
        
        # Normalize to 0-1 range
        if total_data_points > 0:
            data_confidence = min(total_data_points / 100, 1.0)
        else:
            data_confidence = 0.0
        
        # Volatility confidence (lower volatility = higher confidence)
        volatility = volatility_analysis.get('volatility', 0.0)
        volatility_confidence = max(0, 1 - (volatility * 10))
        
        # Overall confidence
        confidence = (data_confidence + volatility_confidence) / 2
        
        return min(confidence, 1.0)
    
    def _update_enhanced_stats(self, calculation_time_ms: float):
        """Update enhanced engine statistics"""
        self.stats['enhanced_calculations'] += 1
        
        # Update average calculation time
        total_time = self.stats['avg_enhanced_time_ms'] * (self.stats['enhanced_calculations'] - 1)
        self.stats['avg_enhanced_time_ms'] = (total_time + calculation_time_ms) / self.stats['enhanced_calculations']
    
    def _update_legacy_stats(self, calculation_time_ms: float):
        """Update legacy engine statistics"""
        self.stats['legacy_calculations'] += 1
        
        # Update average calculation time
        total_time = self.stats['avg_legacy_time_ms'] * (self.stats['legacy_calculations'] - 1)
        self.stats['avg_legacy_time_ms'] = (total_time + calculation_time_ms) / self.stats['legacy_calculations']
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_calculations = self.stats['enhanced_calculations'] + self.stats['legacy_calculations']
        
        return {
            **self.stats,
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['total_requests'], 1),
            'enhanced_usage_rate': self.stats['enhanced_calculations'] / max(total_calculations, 1),
            'legacy_usage_rate': self.stats['legacy_calculations'] / max(total_calculations, 1),
            'enhanced_engine_stats': self.enhanced_engine.get_performance_stats()
        }
    
    async def toggle_enhanced_mode(self, enable: bool):
        """Toggle enhanced mode on/off"""
        self.enable_enhanced = enable
        logger.info(f"Enhanced mode {'enabled' if enable else 'disabled'}")
    
    async def update_config(self, config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(config)
        logger.info(f"Configuration updated: {config}")
