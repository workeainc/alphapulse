"""
Real Data Integration Service
Connects sophisticated single-pair interface to TimescaleDB and AI/ML models
Phase 6: Real Data Integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import asyncpg
import numpy as np
from dataclasses import dataclass
import json

from database.connection import TimescaleDBConnection
from services.free_api_manager import FreeAPIManager
from services.news_sentiment_service import NewsSentimentService

logger = logging.getLogger(__name__)

@dataclass
class RealMarketData:
    """Real market data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    volume_change_24h: float
    fear_greed_index: int
    data_quality_score: float

@dataclass
class RealSentimentData:
    """Real sentiment data structure"""
    symbol: str
    timestamp: datetime
    sentiment_score: float
    sentiment_label: str
    confidence: float
    volume: int
    source: str
    data_quality_score: float

@dataclass
class RealTechnicalIndicators:
    """Real technical indicators structure"""
    symbol: str
    timestamp: datetime
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    volume_sma: float
    data_quality_score: float

class RealDataIntegrationService:
    """Service for integrating real data from TimescaleDB and external APIs"""
    
    def __init__(self):
        self.db_connection = TimescaleDBConnection()
        self.free_api_manager = FreeAPIManager()
        self.news_service = NewsSentimentService()
        self.logger = logger
        
        # Cache for performance
        self._market_data_cache: Dict[str, RealMarketData] = {}
        self._sentiment_cache: Dict[str, List[RealSentimentData]] = {}
        self._technical_cache: Dict[str, RealTechnicalIndicators] = {}
        self._cache_ttl = 30  # seconds
        
    async def get_real_market_data(self, symbol: str, timeframe: str = "1h") -> Optional[RealMarketData]:
        """Get real market data from TimescaleDB"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._market_data_cache:
                cached_data = self._market_data_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_data.timestamp).seconds < self._cache_ttl:
                    return cached_data
            
            # Get data from TimescaleDB
            query = """
            SELECT 
                symbol,
                timestamp,
                price,
                volume_24h,
                market_cap,
                price_change_24h,
                volume_change_24h,
                fear_greed_index,
                data_quality_score
            FROM free_api_market_data 
            WHERE symbol = $1 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            async with self.db_connection.get_connection() as conn:
                row = await conn.fetchrow(query, symbol.upper())
                
                if row:
                    market_data = RealMarketData(
                        symbol=row['symbol'],
                        timestamp=row['timestamp'],
                        price=float(row['price']),
                        volume_24h=float(row['volume_24h']) if row['volume_24h'] else 0.0,
                        market_cap=float(row['market_cap']) if row['market_cap'] else 0.0,
                        price_change_24h=float(row['price_change_24h']) if row['price_change_24h'] else 0.0,
                        volume_change_24h=float(row['volume_change_24h']) if row['volume_change_24h'] else 0.0,
                        fear_greed_index=int(row['fear_greed_index']) if row['fear_greed_index'] else 50,
                        data_quality_score=float(row['data_quality_score'])
                    )
                    
                    # Cache the data
                    self._market_data_cache[cache_key] = market_data
                    return market_data
                    
        except Exception as e:
            self.logger.error(f"Error getting real market data for {symbol}: {e}")
            
        return None
    
    async def get_real_sentiment_data(self, symbol: str, hours_back: int = 24) -> List[RealSentimentData]:
        """Get real sentiment data from TimescaleDB"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{hours_back}"
            if cache_key in self._sentiment_cache:
                cached_data = self._sentiment_cache[cache_key]
                if cached_data and (datetime.now(timezone.utc) - cached_data[0].timestamp).seconds < self._cache_ttl:
                    return cached_data
            
            # Get data from TimescaleDB
            query = """
            SELECT 
                symbol,
                timestamp,
                sentiment_score,
                sentiment_label,
                confidence,
                volume,
                source,
                data_quality_score
            FROM free_api_sentiment_data 
            WHERE symbol = $1 
            AND timestamp >= NOW() - INTERVAL '%s hours'
            ORDER BY timestamp DESC
            """ % hours_back
            
            async with self.db_connection.get_connection() as conn:
                rows = await conn.fetch(query, symbol.upper())
                
                sentiment_data = []
                for row in rows:
                    sentiment_data.append(RealSentimentData(
                        symbol=row['symbol'],
                        timestamp=row['timestamp'],
                        sentiment_score=float(row['sentiment_score']),
                        sentiment_label=row['sentiment_label'],
                        confidence=float(row['confidence']),
                        volume=int(row['volume']) if row['volume'] else 0,
                        source=row['source'],
                        data_quality_score=float(row['data_quality_score'])
                    ))
                
                # Cache the data
                self._sentiment_cache[cache_key] = sentiment_data
                return sentiment_data
                
        except Exception as e:
            self.logger.error(f"Error getting real sentiment data for {symbol}: {e}")
            
        return []
    
    async def get_real_technical_indicators(self, symbol: str, timeframe: str = "1h") -> Optional[RealTechnicalIndicators]:
        """Get real technical indicators from TimescaleDB"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self._technical_cache:
                cached_data = self._technical_cache[cache_key]
                if (datetime.now(timezone.utc) - cached_data.timestamp).seconds < self._cache_ttl:
                    return cached_data
            
            # Get historical price data for technical analysis
            query = """
            SELECT 
                timestamp,
                price,
                volume_24h
            FROM free_api_market_data 
            WHERE symbol = $1 
            ORDER BY timestamp DESC 
            LIMIT 100
            """
            
            async with self.db_connection.get_connection() as conn:
                rows = await conn.fetch(query, symbol.upper())
                
                if len(rows) < 20:  # Need at least 20 data points
                    return None
                
                # Convert to arrays for technical analysis
                prices = [float(row['price']) for row in reversed(rows)]
                volumes = [float(row['volume_24h']) if row['volume_24h'] else 0.0 for row in reversed(rows)]
                timestamps = [row['timestamp'] for row in reversed(rows)]
                
                # Calculate technical indicators
                technical_data = self._calculate_technical_indicators(prices, volumes, timestamps[-1])
                
                if technical_data:
                    # Cache the data
                    self._technical_cache[cache_key] = technical_data
                    return technical_data
                    
        except Exception as e:
            self.logger.error(f"Error getting real technical indicators for {symbol}: {e}")
            
        return None
    
    def _calculate_technical_indicators(self, prices: List[float], volumes: List[float], timestamp: datetime) -> Optional[RealTechnicalIndicators]:
        """Calculate technical indicators from price data"""
        try:
            if len(prices) < 20:
                return None
            
            # Convert to numpy arrays for easier calculation
            price_array = np.array(prices)
            volume_array = np.array(volumes)
            
            # RSI calculation
            rsi = self._calculate_rsi(price_array, 14)
            
            # MACD calculation
            macd, macd_signal, macd_histogram = self._calculate_macd(price_array, 12, 26, 9)
            
            # Moving averages
            sma_20 = self._calculate_sma(price_array, 20)
            sma_50 = self._calculate_sma(price_array, 50)
            ema_12 = self._calculate_ema(price_array, 12)
            ema_26 = self._calculate_ema(price_array, 26)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(price_array, 20, 2)
            
            # Volume SMA
            volume_sma = self._calculate_sma(volume_array, 20)
            
            return RealTechnicalIndicators(
                symbol="",  # Will be set by caller
                timestamp=timestamp,
                rsi=rsi[-1] if len(rsi) > 0 else 50.0,
                macd=macd[-1] if len(macd) > 0 else 0.0,
                macd_signal=macd_signal[-1] if len(macd_signal) > 0 else 0.0,
                macd_histogram=macd_histogram[-1] if len(macd_histogram) > 0 else 0.0,
                sma_20=sma_20[-1] if len(sma_20) > 0 else price_array[-1],
                sma_50=sma_50[-1] if len(sma_50) > 0 else price_array[-1],
                ema_12=ema_12[-1] if len(ema_12) > 0 else price_array[-1],
                ema_26=ema_26[-1] if len(ema_26) > 0 else price_array[-1],
                bollinger_upper=bb_upper[-1] if len(bb_upper) > 0 else price_array[-1],
                bollinger_lower=bb_lower[-1] if len(bb_lower) > 0 else price_array[-1],
                bollinger_middle=bb_middle[-1] if len(bb_middle) > 0 else price_array[-1],
                volume_sma=volume_sma[-1] if len(volume_sma) > 0 else volume_array[-1],
                data_quality_score=1.0
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
            avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
            
            rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return np.array([50.0])
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        try:
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            macd = ema_fast - ema_slow
            macd_signal = self._calculate_ema(macd, signal)
            macd_histogram = macd - macd_signal
            
            return macd, macd_signal, macd_histogram
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        try:
            return np.convolve(prices, np.ones(period)/period, mode='valid')
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return np.array([0.0])
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        try:
            alpha = 2 / (period + 1)
            ema = np.zeros_like(prices)
            ema[0] = prices[0]
            
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
            return ema
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return np.array([0.0])
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        try:
            sma = self._calculate_sma(prices, period)
            
            # Calculate standard deviation
            std = np.zeros_like(sma)
            for i in range(len(sma)):
                start_idx = max(0, i - period + 1)
                std[i] = np.std(prices[start_idx:i+period])
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return upper, sma, lower
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return np.array([0.0]), np.array([0.0]), np.array([0.0])
    
    async def calculate_real_confidence(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Calculate real confidence based on actual data"""
        try:
            # Get real data
            market_data = await self.get_real_market_data(symbol, timeframe)
            sentiment_data = await self.get_real_sentiment_data(symbol, 24)
            technical_data = await self.get_real_technical_indicators(symbol, timeframe)
            
            if not market_data:
                return self._get_fallback_confidence(symbol)
            
            # Calculate technical confidence
            technical_confidence = 0.5
            if technical_data:
                # RSI-based confidence
                rsi_score = 1.0 - abs(technical_data.rsi - 50) / 50
                
                # MACD-based confidence
                macd_score = 1.0 if abs(technical_data.macd) > abs(technical_data.macd_signal) else 0.7
                
                # Moving average alignment
                ma_score = 1.0 if technical_data.sma_20 > technical_data.sma_50 else 0.6
                
                technical_confidence = (rsi_score + macd_score + ma_score) / 3
            
            # Calculate sentiment confidence
            sentiment_confidence = 0.5
            if sentiment_data:
                avg_sentiment = np.mean([s.sentiment_score for s in sentiment_data])
                avg_confidence = np.mean([s.confidence for s in sentiment_data])
                volume_factor = min(len(sentiment_data) / 10, 1.0)  # More data = higher confidence
                
                sentiment_confidence = (abs(avg_sentiment) * avg_confidence * volume_factor)
            
            # Calculate volume confidence
            volume_confidence = 0.5
            if market_data.volume_24h > 0:
                # Higher volume = higher confidence
                volume_factor = min(market_data.volume_24h / 1000000000, 1.0)  # Normalize to 1B
                volume_confidence = 0.5 + (volume_factor * 0.5)
            
            # Overall confidence (weighted average)
            overall_confidence = (
                technical_confidence * 0.4 +
                sentiment_confidence * 0.3 +
                volume_confidence * 0.3
            )
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "current_confidence": overall_confidence,
                "technical_confidence": technical_confidence,
                "sentiment_confidence": sentiment_confidence,
                "volume_confidence": volume_confidence,
                "is_building": overall_confidence < 0.85,
                "threshold_reached": overall_confidence >= 0.85,
                "confidence_factors": {
                    "technical": technical_confidence,
                    "sentiment": sentiment_confidence,
                    "volume": volume_confidence,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_quality": market_data.data_quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating real confidence for {symbol}: {e}")
            return self._get_fallback_confidence(symbol)
    
    def _get_fallback_confidence(self, symbol: str) -> Dict[str, Any]:
        """Fallback confidence when real data is not available"""
        return {
            "symbol": symbol,
            "timeframe": "1h",
            "current_confidence": 0.5,
            "technical_confidence": 0.5,
            "sentiment_confidence": 0.5,
            "volume_confidence": 0.5,
            "is_building": True,
            "threshold_reached": False,
            "confidence_factors": {
                "technical": 0.5,
                "sentiment": 0.5,
                "volume": 0.5,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_quality": 0.0
        }
    
    async def get_real_analysis_data(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Get real analysis data for FA, TA, and Sentiment"""
        try:
            # Get real data
            market_data = await self.get_real_market_data(symbol, timeframe)
            sentiment_data = await self.get_real_sentiment_data(symbol, 24)
            technical_data = await self.get_real_technical_indicators(symbol, timeframe)
            
            # Fundamental Analysis
            fundamental = {
                "market_regime": "Bullish" if market_data.price_change_24h > 0 else "Bearish" if market_data.price_change_24h < 0 else "Neutral",
                "news_impact": abs(market_data.price_change_24h) if market_data.price_change_24h else 0.0,
                "macro_factors": "Positive" if market_data.fear_greed_index > 50 else "Negative" if market_data.fear_greed_index < 50 else "Neutral",
                "confidence": market_data.data_quality_score,
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            
            # Technical Analysis
            technical = {
                "rsi": technical_data.rsi if technical_data else 50.0,
                "macd": "Bullish" if technical_data and technical_data.macd > technical_data.macd_signal else "Bearish" if technical_data else "Neutral",
                "pattern": "Ascending Triangle" if technical_data and technical_data.sma_20 > technical_data.sma_50 else "Descending Triangle" if technical_data else "No Pattern",
                "confidence": technical_data.data_quality_score if technical_data else 0.5,
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            
            # Sentiment Analysis
            sentiment = {
                "social_sentiment": np.mean([s.sentiment_score for s in sentiment_data]) * 100 if sentiment_data else 0.0,
                "fear_greed": "Greed" if market_data.fear_greed_index > 70 else "Fear" if market_data.fear_greed_index < 30 else "Neutral",
                "volume": len(sentiment_data) if sentiment_data else 0,
                "confidence": np.mean([s.confidence for s in sentiment_data]) if sentiment_data else 0.5,
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            
            return {
                "pair": symbol,
                "timeframe": timeframe,
                "analysis": {
                    "fundamental": fundamental,
                    "technical": technical,
                    "sentiment": sentiment
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting real analysis data for {symbol}: {e}")
            return self._get_fallback_analysis(symbol, timeframe)
    
    def _get_fallback_analysis(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Fallback analysis when real data is not available"""
        return {
            "pair": symbol,
            "timeframe": timeframe,
            "analysis": {
                "fundamental": {
                    "market_regime": "Neutral",
                    "news_impact": 0.0,
                    "macro_factors": "Neutral",
                    "confidence": 0.0,
                    "last_update": datetime.now(timezone.utc).isoformat()
                },
                "technical": {
                    "rsi": 50.0,
                    "macd": "Neutral",
                    "pattern": "No Pattern",
                    "confidence": 0.0,
                    "last_update": datetime.now(timezone.utc).isoformat()
                },
                "sentiment": {
                    "social_sentiment": 0.0,
                    "fear_greed": "Neutral",
                    "volume": 0,
                    "confidence": 0.0,
                    "last_update": datetime.now(timezone.utc).isoformat()
                }
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global instance
real_data_service = RealDataIntegrationService()
