#!/usr/bin/env python3
"""
Multi-Symbol Correlation Detector
BTC dominance + correlated alt checks for pattern confirmation
"""

import numpy as np
import pandas as pd
import logging
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import talib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CorrelationResult:
    """Result from correlation analysis"""
    symbol: str
    correlation_score: float
    btc_dominance: float
    market_sentiment: str
    confidence_adjustment: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class MultiSymbolPatternResult:
    """Result from multi-symbol pattern detection"""
    primary_symbol: str
    pattern_name: str
    confidence: float
    correlation_boost: float
    btc_confirmation: bool
    alt_confirmation: bool
    market_regime: str
    timestamp: datetime
    price_level: float
    metadata: Dict[str, Any]

class CorrelationCalculator:
    """Calculate correlations between symbols"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.correlation_cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=5)  # Cache for 5 minutes
        
    def calculate_correlation(self, symbol1_prices: np.ndarray, 
                            symbol2_prices: np.ndarray) -> float:
        """Calculate correlation between two price series"""
        if len(symbol1_prices) != len(symbol2_prices) or len(symbol1_prices) < 10:
            return 0.0
        
        # Calculate returns
        returns1 = np.diff(symbol1_prices) / symbol1_prices[:-1]
        returns2 = np.diff(symbol2_prices) / symbol2_prices[:-1]
        
        # Calculate correlation
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def calculate_btc_dominance(self, btc_price: float, total_market_cap: float) -> float:
        """Calculate BTC dominance percentage"""
        if total_market_cap <= 0:
            return 0.0
        
        # Simplified BTC dominance calculation
        # In real implementation, you'd get this from market data APIs
        btc_market_cap = btc_price * 19_000_000  # Approximate BTC supply
        dominance = (btc_market_cap / total_market_cap) * 100
        
        return min(max(dominance, 0.0), 100.0)
    
    def get_cached_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get cached correlation value"""
        cache_key = f"{symbol1}_{symbol2}"
        
        if cache_key in self.correlation_cache:
            if datetime.now(timezone.utc) < self.cache_expiry[cache_key]:
                return self.correlation_cache[cache_key]
            else:
                # Remove expired cache
                del self.correlation_cache[cache_key]
                del self.cache_expiry[cache_key]
        
        return None
    
    def cache_correlation(self, symbol1: str, symbol2: str, correlation: float):
        """Cache correlation value"""
        cache_key = f"{symbol1}_{symbol2}"
        self.correlation_cache[cache_key] = correlation
        self.cache_expiry[cache_key] = datetime.now(timezone.utc) + self.cache_duration

class MarketSentimentAnalyzer:
    """Analyze market sentiment based on multiple factors"""
    
    def __init__(self):
        self.sentiment_weights = {
            'btc_dominance': 0.4,
            'correlation_strength': 0.3,
            'volume_ratio': 0.2,
            'volatility': 0.1
        }
    
    def analyze_sentiment(self, btc_dominance: float, correlation_score: float,
                         volume_ratio: float, volatility: float) -> str:
        """Analyze overall market sentiment"""
        # Calculate weighted sentiment score
        sentiment_score = (
            self.sentiment_weights['btc_dominance'] * self._normalize_btc_dominance(btc_dominance) +
            self.sentiment_weights['correlation_strength'] * abs(correlation_score) +
            self.sentiment_weights['volume_ratio'] * min(volume_ratio, 2.0) / 2.0 +
            self.sentiment_weights['volatility'] * min(volatility, 0.1) / 0.1
        )
        
        # Classify sentiment
        if sentiment_score > 0.7:
            return "bullish"
        elif sentiment_score > 0.4:
            return "neutral"
        else:
            return "bearish"
    
    def _normalize_btc_dominance(self, dominance: float) -> float:
        """Normalize BTC dominance to 0-1 scale"""
        # BTC dominance typically ranges from 30% to 70%
        # Higher dominance often indicates bearish sentiment (flight to safety)
        normalized = (dominance - 30) / 40  # Normalize to 0-1
        return 1.0 - normalized  # Invert so higher = more bullish

class MultiSymbolCorrelationDetector:
    """Multi-symbol correlation detector for pattern confirmation"""
    
    def __init__(self, btc_symbol: str = "BTC/USDT"):
        self.btc_symbol = btc_symbol
        self.correlation_calc = CorrelationCalculator()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        
        # Correlation thresholds
        self.high_correlation_threshold = 0.7
        self.medium_correlation_threshold = 0.4
        
        # Market regime thresholds
        self.btc_dominance_bullish = 45.0  # Below this is bullish for alts
        self.btc_dominance_bearish = 55.0  # Above this is bearish for alts
        
        # Confidence adjustment factors
        self.correlation_boost_factor = 0.2
        self.btc_confirmation_boost = 0.15
        self.alt_confirmation_boost = 0.1
        
        logger.info(f"🚀 Multi-Symbol Correlation Detector initialized with BTC symbol: {btc_symbol}")
    
    async def analyze_correlation(self, symbol: str, symbol_data: Dict,
                                btc_data: Dict, market_data: Dict) -> CorrelationResult:
        """Analyze correlation between symbol and BTC"""
        try:
            # Extract price data
            symbol_prices = np.array([c['close'] for c in symbol_data.get('candles', [])])
            btc_prices = np.array([c['close'] for c in btc_data.get('candles', [])])
            
            if len(symbol_prices) < 10 or len(btc_prices) < 10:
                return self._create_default_correlation_result(symbol)
            
            # Calculate correlation
            correlation = self.correlation_calc.calculate_correlation(symbol_prices, btc_prices)
            
            # Cache correlation
            self.correlation_calc.cache_correlation(symbol, self.btc_symbol, correlation)
            
            # Calculate BTC dominance
            btc_price = btc_prices[-1] if len(btc_prices) > 0 else 50000.0
            total_market_cap = market_data.get('total_market_cap', 2_000_000_000_000)  # 2T default
            btc_dominance = self.correlation_calc.calculate_btc_dominance(btc_price, total_market_cap)
            
            # Analyze market sentiment
            volume_ratio = market_data.get('volume_ratio', 1.0)
            volatility = market_data.get('volatility', 0.05)
            market_sentiment = self.sentiment_analyzer.analyze_sentiment(
                btc_dominance, correlation, volume_ratio, volatility
            )
            
            # Calculate confidence adjustment
            confidence_adjustment = self._calculate_confidence_adjustment(
                correlation, btc_dominance, market_sentiment
            )
            
            return CorrelationResult(
                symbol=symbol,
                correlation_score=correlation,
                btc_dominance=btc_dominance,
                market_sentiment=market_sentiment,
                confidence_adjustment=confidence_adjustment,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    'volume_ratio': volume_ratio,
                    'volatility': volatility,
                    'price_level': float(symbol_prices[-1]) if len(symbol_prices) > 0 else 0.0
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing correlation for {symbol}: {e}")
            return self._create_default_correlation_result(symbol)
    
    def _create_default_correlation_result(self, symbol: str) -> CorrelationResult:
        """Create default correlation result when analysis fails"""
        return CorrelationResult(
            symbol=symbol,
            correlation_score=0.0,
            btc_dominance=50.0,
            market_sentiment="neutral",
            confidence_adjustment=0.0,
            timestamp=datetime.now(timezone.utc),
            metadata={'error': 'Analysis failed'}
        )
    
    def _calculate_confidence_adjustment(self, correlation: float, 
                                       btc_dominance: float, 
                                       market_sentiment: str) -> float:
        """Calculate confidence adjustment based on correlation and market conditions"""
        adjustment = 0.0
        
        # Correlation-based adjustment
        if abs(correlation) > self.high_correlation_threshold:
            adjustment += self.correlation_boost_factor
        elif abs(correlation) > self.medium_correlation_threshold:
            adjustment += self.correlation_boost_factor * 0.5
        
        # BTC dominance adjustment
        if btc_dominance < self.btc_dominance_bullish:
            adjustment += self.btc_confirmation_boost  # Bullish for alts
        elif btc_dominance > self.btc_dominance_bearish:
            adjustment -= self.btc_confirmation_boost  # Bearish for alts
        
        # Market sentiment adjustment
        if market_sentiment == "bullish":
            adjustment += self.alt_confirmation_boost
        elif market_sentiment == "bearish":
            adjustment -= self.alt_confirmation_boost
        
        return max(min(adjustment, 0.3), -0.3)  # Cap at ±30%
    
    async def detect_multi_symbol_patterns(self, primary_symbol: str,
                                         primary_patterns: List[Dict],
                                         all_symbols_data: Dict[str, Dict],
                                         market_data: Dict) -> List[MultiSymbolPatternResult]:
        """Detect patterns with multi-symbol correlation analysis"""
        results = []
        
        # Get BTC data
        btc_data = all_symbols_data.get(self.btc_symbol, {})
        
        # Analyze correlation for primary symbol
        correlation_result = await self.analyze_correlation(
            primary_symbol, all_symbols_data.get(primary_symbol, {}), btc_data, market_data
        )
        
        # Process each pattern
        for pattern in primary_patterns:
            # Determine market regime
            market_regime = self._determine_market_regime(
                correlation_result.btc_dominance, correlation_result.market_sentiment
            )
            
            # Check BTC confirmation
            btc_confirmation = self._check_btc_confirmation(
                pattern, btc_data, correlation_result
            )
            
            # Check alt confirmation
            alt_confirmation = self._check_alt_confirmation(
                pattern, all_symbols_data, primary_symbol, correlation_result
            )
            
            # Calculate correlation boost
            correlation_boost = self._calculate_correlation_boost(
                pattern, correlation_result, btc_confirmation, alt_confirmation
            )
            
            # Create result
            result = MultiSymbolPatternResult(
                primary_symbol=primary_symbol,
                pattern_name=pattern.get('pattern_name', 'unknown'),
                confidence=pattern.get('confidence', 0.0) + correlation_boost,
                correlation_boost=correlation_boost,
                btc_confirmation=btc_confirmation,
                alt_confirmation=alt_confirmation,
                market_regime=market_regime,
                timestamp=datetime.now(timezone.utc),
                price_level=pattern.get('price_level', 0.0),
                metadata={
                    'correlation_score': correlation_result.correlation_score,
                    'btc_dominance': correlation_result.btc_dominance,
                    'market_sentiment': correlation_result.market_sentiment,
                    'original_confidence': pattern.get('confidence', 0.0)
                }
            )
            
            results.append(result)
        
        return results
    
    def _determine_market_regime(self, btc_dominance: float, sentiment: str) -> str:
        """Determine current market regime"""
        if btc_dominance > self.btc_dominance_bearish:
            return "btc_dominant"
        elif btc_dominance < self.btc_dominance_bullish:
            return "alt_season"
        else:
            return "neutral"
    
    def _check_btc_confirmation(self, pattern: Dict, btc_data: Dict, 
                               correlation_result: CorrelationResult) -> bool:
        """Check if BTC confirms the pattern"""
        try:
            btc_candles = btc_data.get('candles', [])
            if len(btc_candles) < 5:
                return False
            
            # Check if BTC shows similar pattern or trend
            pattern_name = pattern.get('pattern_name', '').lower()
            pattern_direction = pattern.get('direction', 'neutral')
            
            # Simple trend check for BTC
            recent_btc_prices = [c['close'] for c in btc_candles[-5:]]
            btc_trend = 'bullish' if recent_btc_prices[-1] > recent_btc_prices[0] else 'bearish'
            
            # Check if BTC trend aligns with pattern direction
            if pattern_direction == 'bullish' and btc_trend == 'bullish':
                return True
            elif pattern_direction == 'bearish' and btc_trend == 'bearish':
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking BTC confirmation: {e}")
            return False
    
    def _check_alt_confirmation(self, pattern: Dict, all_symbols_data: Dict,
                               primary_symbol: str, correlation_result: CorrelationResult) -> bool:
        """Check if other alt coins confirm the pattern"""
        try:
            # Check top correlated alt coins
            correlated_alts = self._get_correlated_alts(primary_symbol, all_symbols_data)
            
            confirmations = 0
            total_checks = 0
            
            for alt_symbol in correlated_alts[:3]:  # Check top 3 correlated alts
                alt_data = all_symbols_data.get(alt_symbol, {})
                if self._check_alt_pattern_confirmation(pattern, alt_data):
                    confirmations += 1
                total_checks += 1
            
            # Require at least 50% confirmation
            return (confirmations / total_checks) >= 0.5 if total_checks > 0 else False
            
        except Exception as e:
            logger.error(f"Error checking alt confirmation: {e}")
            return False
    
    def _get_correlated_alts(self, primary_symbol: str, all_symbols_data: Dict) -> List[str]:
        """Get list of alt coins correlated with primary symbol"""
        # In real implementation, you'd have a correlation matrix
        # For now, return common alt coins
        common_alts = ['ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT']
        return [alt for alt in common_alts if alt in all_symbols_data and alt != primary_symbol]
    
    def _check_alt_pattern_confirmation(self, pattern: Dict, alt_data: Dict) -> bool:
        """Check if alt coin shows similar pattern"""
        try:
            alt_candles = alt_data.get('candles', [])
            if len(alt_candles) < 5:
                return False
            
            # Simple trend check
            recent_prices = [c['close'] for c in alt_candles[-5:]]
            alt_trend = 'bullish' if recent_prices[-1] > recent_prices[0] else 'bearish'
            
            pattern_direction = pattern.get('direction', 'neutral')
            
            return alt_trend == pattern_direction
            
        except Exception as e:
            logger.error(f"Error checking alt pattern confirmation: {e}")
            return False
    
    def _calculate_correlation_boost(self, pattern: Dict, correlation_result: CorrelationResult,
                                   btc_confirmation: bool, alt_confirmation: bool) -> float:
        """Calculate confidence boost from correlation analysis"""
        boost = correlation_result.confidence_adjustment
        
        # Add confirmation bonuses
        if btc_confirmation:
            boost += self.btc_confirmation_boost
        
        if alt_confirmation:
            boost += self.alt_confirmation_boost
        
        # Cap the boost
        return max(min(boost, 0.5), -0.3)  # Max +50%, min -30%
    
    def get_correlation_stats(self) -> Dict[str, Any]:
        """Get correlation analysis statistics"""
        return {
            'cache_size': len(self.correlation_calc.correlation_cache),
            'high_correlation_threshold': self.high_correlation_threshold,
            'medium_correlation_threshold': self.medium_correlation_threshold,
            'btc_dominance_bullish': self.btc_dominance_bullish,
            'btc_dominance_bearish': self.btc_dominance_bearish,
            'correlation_boost_factor': self.correlation_boost_factor,
            'btc_confirmation_boost': self.btc_confirmation_boost,
            'alt_confirmation_boost': self.alt_confirmation_boost
        }
