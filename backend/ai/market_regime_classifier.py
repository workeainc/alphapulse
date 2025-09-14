#!/usr/bin/env python3
"""
Market Regime Classifier for Advanced Pattern Recognition
Classifies market conditions into different regimes for adaptive pattern detection
"""

import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

def safe_float(value, default=0.0):
    """Safely convert value to float, handling NaN and None"""
    try:
        if pd.isna(value) or value is None:
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)

def safe_json_dumps(data):
    """Safely convert dictionary to JSON, handling NaN values"""
    try:
        # Convert all NaN values to None before JSON serialization
        def convert_nan(obj):
            if isinstance(obj, dict):
                return {k: convert_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan(v) for v in obj]
            elif pd.isna(obj) if hasattr(pd, 'isna') else obj != obj:  # Check for NaN
                return None
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if not pd.isna(obj) else None
            else:
                return obj
        
        cleaned_data = convert_nan(data)
        return json.dumps(cleaned_data)
    except Exception as e:
        logger.warning(f"Failed to serialize data to JSON: {e}")
        return json.dumps({})

class MarketRegimeClassifier:
    """Classifier for identifying market regimes"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.initialized = False
        
        # Regime classification parameters
        self.regime_thresholds = {
            'trending': {
                'min_trend_strength': 0.7,
                'min_momentum_score': 0.6,
                'max_volatility': 0.03
            },
            'sideways': {
                'max_trend_strength': 0.3,
                'max_momentum_score': 0.4,
                'max_volatility': 0.02
            },
            'volatile': {
                'min_volatility': 0.025,
                'max_trend_strength': 0.5
            },
            'consolidation': {
                'max_trend_strength': 0.4,
                'max_volatility': 0.015,
                'min_volume_profile': 'normal'
            }
        }
        
        logger.info("🔧 Market Regime Classifier initialized")
    
    async def initialize(self):
        """Initialize the market regime classifier"""
        try:
            logger.info("🔧 Initializing Market Regime Classifier...")
            
            # Connect to database
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            
            self.initialized = True
            logger.info("✅ Market Regime Classifier ready")
            
        except Exception as e:
            logger.error(f"❌ Market Regime Classifier initialization failed: {e}")
            raise
    
    async def classify_market_regime(self, market_data: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Classify the current market regime
        
        Args:
            market_data: Market data DataFrame with OHLCV
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            Market regime classification results
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            if len(market_data) < 10:
                return {
                    'regime_type': 'unknown',
                    'regime_confidence': 0.0,
                    'trend_strength': 0.0,
                    'volatility_level': 0.0,
                    'volume_profile': 'unknown',
                    'momentum_score': 0.0,
                    'support_resistance_proximity': 0.0,
                    'market_microstructure': {},
                    'regime_features': {},
                    'reason': 'Insufficient data for classification'
                }
            
            # Calculate market features
            features = await self._calculate_market_features(market_data)
            
            # Classify regime
            regime_result = await self._classify_regime(features)
            
            # Store classification
            await self._store_regime_classification(symbol, timeframe, regime_result, features)
            
            return regime_result
            
        except Exception as e:
            logger.error(f"❌ Market regime classification failed: {e}")
            return {
                'regime_type': 'unknown',
                'regime_confidence': 0.0,
                'error': str(e)
            }
    
    async def _calculate_market_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market features for regime classification"""
        try:
            features = {}
            
            # Calculate trend strength using linear regression
            trend_strength = await self._calculate_trend_strength(market_data)
            features['trend_strength'] = float(trend_strength)
            
            # Calculate volatility level
            volatility_level = await self._calculate_volatility_level(market_data)
            features['volatility_level'] = float(volatility_level)
            
            # Calculate volume profile
            features['volume_profile'] = await self._calculate_volume_profile(market_data)
            
            # Calculate momentum score
            momentum_score = await self._calculate_momentum_score(market_data)
            features['momentum_score'] = float(momentum_score)
            
            # Calculate support/resistance proximity
            support_resistance_proximity = await self._calculate_support_resistance_proximity(market_data)
            features['support_resistance_proximity'] = float(support_resistance_proximity)
            
            # Calculate market microstructure
            features['market_microstructure'] = await self._calculate_market_microstructure(market_data)
            
            # Calculate additional technical indicators
            features['regime_features'] = await self._calculate_regime_features(market_data)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature calculation failed: {e}")
            return {}
    
    async def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength using linear regression"""
        try:
            # Use closing prices for trend calculation
            prices = market_data['close'].values
            x = np.arange(len(prices))
            
            # Linear regression
            slope, intercept = np.polyfit(x, prices, 1)
            
            # Calculate R-squared (trend strength)
            y_pred = slope * x + intercept
            ss_res = np.sum((prices - y_pred) ** 2)
            ss_tot = np.sum((prices - np.mean(prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Normalize to 0-1 range
            trend_strength = min(r_squared, 1.0)
            
            return trend_strength
            
        except Exception as e:
            logger.error(f"❌ Trend strength calculation failed: {e}")
            return 0.0
    
    async def _calculate_volatility_level(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility level using ATR"""
        try:
            # Calculate ATR
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            current_price = market_data['close'].iloc[-1]
            volatility_level = atr / current_price if current_price > 0 else 0.0
            
            return volatility_level
            
        except Exception as e:
            logger.error(f"❌ Volatility calculation failed: {e}")
            return 0.0
    
    async def _calculate_volume_profile(self, market_data: pd.DataFrame) -> str:
        """Calculate volume profile"""
        try:
            recent_volume = market_data['volume'].tail(min(20, len(market_data))).mean()
            historical_volume = market_data['volume'].tail(min(100, len(market_data))).mean()
            
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
            
            if volume_ratio > 1.5:
                return 'high'
            elif volume_ratio < 0.7:
                return 'low'
            elif volume_ratio > 2.0:
                return 'spike'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"❌ Volume profile calculation failed: {e}")
            return 'normal'
    
    async def _calculate_momentum_score(self, market_data: pd.DataFrame) -> float:
        """Calculate momentum score using RSI and MACD"""
        try:
            # Calculate RSI
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Calculate MACD
            ema12 = market_data['close'].ewm(span=12).mean()
            ema26 = market_data['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_histogram = macd - signal
            
            # Normalize momentum indicators
            rsi_score = (current_rsi - 50) / 50  # -1 to 1
            macd_score = macd_histogram.iloc[-1] / market_data['close'].iloc[-1] if market_data['close'].iloc[-1] > 0 else 0
            
            # Combine scores
            momentum_score = (abs(rsi_score) + abs(macd_score)) / 2
            momentum_score = min(momentum_score, 1.0)
            
            # Handle NaN values
            if pd.isna(momentum_score):
                momentum_score = 0.0
            
            return float(momentum_score)
            
        except Exception as e:
            logger.error(f"❌ Momentum calculation failed: {e}")
            return 0.0
    
    async def _calculate_support_resistance_proximity(self, market_data: pd.DataFrame) -> float:
        """Calculate proximity to support/resistance levels"""
        try:
            current_price = market_data['close'].iloc[-1]
            
            # Find recent highs and lows
            recent_highs = market_data['high'].tail(20).nlargest(3)
            recent_lows = market_data['low'].tail(20).nsmallest(3)
            
            # Calculate distances
            high_distances = [(high - current_price) / current_price for high in recent_highs]
            low_distances = [(current_price - low) / current_price for low in recent_lows]
            
            # Find minimum distance
            min_high_distance = min(high_distances) if high_distances else 1.0
            min_low_distance = min(low_distances) if low_distances else 1.0
            min_distance = min(min_high_distance, min_low_distance)
            
            # Convert to proximity score (closer = higher score)
            proximity = max(0, 1.0 - min_distance)
            
            # Handle NaN values
            if pd.isna(proximity):
                proximity = 0.0
            
            return float(proximity)
            
        except Exception as e:
            logger.error(f"❌ Support/resistance proximity calculation failed: {e}")
            return 0.0
    
    async def _calculate_market_microstructure(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market microstructure features"""
        try:
            # Simplified microstructure calculation
            # In a real implementation, you would use actual bid/ask data
            
            current_price = market_data['close'].iloc[-1]
            
            # Simulate spread (0.1% of price)
            spread = current_price * 0.001
            spread_ratio = spread / current_price
            
            # Simulate depth (order book depth)
            depth_score = 0.8  # Placeholder
            
            microstructure = {
                'spread_ratio': spread_ratio,
                'depth_score': depth_score,
                'liquidity_score': 0.9,  # Placeholder
                'market_impact': 0.1  # Placeholder
            }
            
            return microstructure
            
        except Exception as e:
            logger.error(f"❌ Market microstructure calculation failed: {e}")
            return {}
    
    async def _calculate_regime_features(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate additional features for regime classification"""
        try:
            features = {}
            
            # Bollinger Bands
            sma = market_data['close'].rolling(window=20).mean()
            std = market_data['close'].rolling(window=20).std()
            bb_upper = sma + (std * 2)
            bb_lower = sma - (std * 2)
            
            current_price = market_data['close'].iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
            
            # Handle NaN values
            if pd.isna(bb_position):
                bb_position = 0.5
            
            features['bollinger_position'] = float(bb_position)
            
            # Price range
            recent_range = (market_data['high'].tail(20).max() - market_data['low'].tail(20).min()) / current_price
            features['price_range'] = float(recent_range) if not pd.isna(recent_range) else 0.0
            
            # Volume trend
            volume_trend = market_data['volume'].tail(10).pct_change().mean()
            features['volume_trend'] = float(volume_trend) if not pd.isna(volume_trend) else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Regime features calculation failed: {e}")
            return {}
    
    async def _classify_regime(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify market regime based on features"""
        try:
            trend_strength = features.get('trend_strength', 0.0)
            volatility_level = features.get('volatility_level', 0.0)
            volume_profile = features.get('volume_profile', 'normal')
            momentum_score = features.get('momentum_score', 0.0)
            
            regime_scores = {}
            
            # Score each regime
            regime_scores['trending'] = await self._score_trending_regime(features)
            regime_scores['sideways'] = await self._score_sideways_regime(features)
            regime_scores['volatile'] = await self._score_volatile_regime(features)
            regime_scores['consolidation'] = await self._score_consolidation_regime(features)
            
            # Find best regime
            best_regime = max(regime_scores, key=regime_scores.get)
            regime_confidence = regime_scores[best_regime]
            
            # Handle NaN values
            if pd.isna(regime_confidence):
                regime_confidence = 0.0
            
            return {
                'regime_type': best_regime,
                'regime_confidence': regime_confidence,
                'trend_strength': trend_strength,
                'volatility_level': volatility_level,
                'volume_profile': volume_profile,
                'momentum_score': momentum_score,
                'support_resistance_proximity': features.get('support_resistance_proximity', 0.0),
                'market_microstructure': features.get('market_microstructure', {}),
                'regime_features': features.get('regime_features', {}),
                'regime_scores': regime_scores,
                'reason': f'Classified as {best_regime} with {regime_confidence:.3f} confidence'
            }
            
        except Exception as e:
            logger.error(f"❌ Regime classification failed: {e}")
            return {
                'regime_type': 'unknown',
                'regime_confidence': 0.0,
                'reason': f'Classification error: {e}'
            }
    
    async def _score_trending_regime(self, features: Dict[str, Any]) -> float:
        """Score trending regime"""
        try:
            thresholds = self.regime_thresholds['trending']
            
            trend_strength = features.get('trend_strength', 0.0)
            momentum_score = features.get('momentum_score', 0.0)
            volatility_level = features.get('volatility_level', 0.0)
            
            # Calculate scores
            trend_score = min(trend_strength / thresholds['min_trend_strength'], 1.0)
            momentum_score_norm = min(momentum_score / thresholds['min_momentum_score'], 1.0)
            volatility_score = max(0, 1.0 - (volatility_level / thresholds['max_volatility']))
            
            # Combine scores
            overall_score = (trend_score * 0.5 + momentum_score_norm * 0.3 + volatility_score * 0.2)
            
            return overall_score
            
        except Exception as e:
            logger.error(f"❌ Trending regime scoring failed: {e}")
            return 0.0
    
    async def _score_sideways_regime(self, features: Dict[str, Any]) -> float:
        """Score sideways regime"""
        try:
            thresholds = self.regime_thresholds['sideways']
            
            trend_strength = features.get('trend_strength', 0.0)
            momentum_score = features.get('momentum_score', 0.0)
            volatility_level = features.get('volatility_level', 0.0)
            
            # Calculate scores (lower is better for sideways)
            trend_score = max(0, 1.0 - (trend_strength / thresholds['max_trend_strength']))
            momentum_score_norm = max(0, 1.0 - (momentum_score / thresholds['max_momentum_score']))
            volatility_score = max(0, 1.0 - (volatility_level / thresholds['max_volatility']))
            
            # Combine scores
            overall_score = (trend_score * 0.4 + momentum_score_norm * 0.3 + volatility_score * 0.3)
            
            return overall_score
            
        except Exception as e:
            logger.error(f"❌ Sideways regime scoring failed: {e}")
            return 0.0
    
    async def _score_volatile_regime(self, features: Dict[str, Any]) -> float:
        """Score volatile regime"""
        try:
            thresholds = self.regime_thresholds['volatile']
            
            volatility_level = features.get('volatility_level', 0.0)
            trend_strength = features.get('trend_strength', 0.0)
            
            # Calculate scores
            volatility_score = min(volatility_level / thresholds['min_volatility'], 1.0)
            trend_score = max(0, 1.0 - (trend_strength / thresholds['max_trend_strength']))
            
            # Combine scores
            overall_score = (volatility_score * 0.7 + trend_score * 0.3)
            
            return overall_score
            
        except Exception as e:
            logger.error(f"❌ Volatile regime scoring failed: {e}")
            return 0.0
    
    async def _score_consolidation_regime(self, features: Dict[str, Any]) -> float:
        """Score consolidation regime"""
        try:
            thresholds = self.regime_thresholds['consolidation']
            
            trend_strength = features.get('trend_strength', 0.0)
            volatility_level = features.get('volatility_level', 0.0)
            volume_profile = features.get('volume_profile', 'normal')
            
            # Calculate scores
            trend_score = max(0, 1.0 - (trend_strength / thresholds['max_trend_strength']))
            volatility_score = max(0, 1.0 - (volatility_level / thresholds['max_volatility']))
            volume_score = 1.0 if volume_profile == 'normal' else 0.5
            
            # Combine scores
            overall_score = (trend_score * 0.4 + volatility_score * 0.4 + volume_score * 0.2)
            
            return overall_score
            
        except Exception as e:
            logger.error(f"❌ Consolidation regime scoring failed: {e}")
            return 0.0
    
    async def _store_regime_classification(self, symbol: str, timeframe: str, regime_result: Dict[str, Any], features: Dict[str, Any]):
        """Store regime classification in database"""
        try:
            regime_id = f"regime_{symbol}_{timeframe}_{int(datetime.now().timestamp())}"
            
            self.cursor.execute("""
                INSERT INTO market_regime_classification (
                    timestamp, regime_id, symbol, timeframe, regime_type, regime_confidence,
                    trend_strength, volatility_level, volume_profile, momentum_score,
                    support_resistance_proximity, market_microstructure, regime_features
                ) VALUES (
                    NOW(), %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s::jsonb, %s::jsonb
                )
            """, (
                regime_id,
                symbol,
                timeframe,
                regime_result['regime_type'],
                safe_float(regime_result['regime_confidence']),
                safe_float(regime_result['trend_strength']),
                safe_float(regime_result['volatility_level']),
                regime_result['volume_profile'],
                safe_float(regime_result['momentum_score']),
                safe_float(regime_result['support_resistance_proximity']),
                safe_json_dumps(regime_result['market_microstructure']),
                safe_json_dumps(regime_result['regime_features'])
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to store regime classification: {e}")
            self.conn.rollback()
    
    async def get_recent_regime(self, symbol: str, timeframe: str, hours: int = 24) -> Optional[Dict[str, Any]]:
        """Get the most recent regime classification"""
        try:
            self.cursor.execute("""
                SELECT regime_type, regime_confidence, trend_strength, volatility_level,
                       volume_profile, momentum_score, support_resistance_proximity,
                       market_microstructure, regime_features, created_at
                FROM market_regime_classification
                WHERE symbol = %s AND timeframe = %s
                AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol, timeframe, hours))
            
            result = self.cursor.fetchone()
            
            if result:
                return {
                    'regime_type': result[0],
                    'regime_confidence': result[1],
                    'trend_strength': result[2],
                    'volatility_level': result[3],
                    'volume_profile': result[4],
                    'momentum_score': result[5],
                    'support_resistance_proximity': result[6],
                    'market_microstructure': result[7],
                    'regime_features': result[8],
                    'created_at': result[9]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent regime: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.info("✅ Market Regime Classifier cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Example usage
async def test_market_regime_classifier():
    """Test the market regime classifier"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',
        'user': 'postgres',
        'password': 'Emon_@17711'
    }
    
    # Create market regime classifier
    classifier = MarketRegimeClassifier(db_config)
    
    try:
        # Initialize classifier
        await classifier.initialize()
        
        # Sample market data (trending market)
        market_data = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900],
            'high': [50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900, 51000],
            'low': [49900, 50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800],
            'close': [50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900, 51000],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Test classification
        regime_result = await classifier.classify_market_regime(market_data, 'BTCUSDT', '1h')
        
        print(f"🎯 Market Regime Classification Results:")
        print(f"   Regime Type: {regime_result['regime_type']}")
        print(f"   Confidence: {regime_result['regime_confidence']:.3f}")
        print(f"   Trend Strength: {regime_result['trend_strength']:.3f}")
        print(f"   Volatility Level: {regime_result['volatility_level']:.4f}")
        print(f"   Volume Profile: {regime_result['volume_profile']}")
        print(f"   Momentum Score: {regime_result['momentum_score']:.3f}")
        print(f"   Reason: {regime_result['reason']}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    
    finally:
        await classifier.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_market_regime_classifier())
