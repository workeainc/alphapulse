#!/usr/bin/env python3
"""
Market Regime Detection Service
Identifies market regimes and adjusts thresholds accordingly
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class RegimeType(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class VolatilityRegime(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LiquidityRegime(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RegimeFeatures:
    """Market regime features"""
    volatility: float
    trend_strength: float
    volume_consistency: float
    price_range: float
    liquidity_score: float
    regime_confidence: float

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: RegimeType
    volatility_regime: VolatilityRegime
    liquidity_regime: LiquidityRegime
    regime_confidence: float
    regime_features: RegimeFeatures
    metadata: Dict[str, Any]

@dataclass
class RegimeThresholds:
    """Regime-specific thresholds"""
    volume_spike_threshold: float
    breakout_threshold: float
    anomaly_threshold: float
    confidence_level: float

class MarketRegimeDetectionService:
    """Service for detecting market regimes and adjusting thresholds"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Regime detection parameters
        self.regime_params = {
            'volatility_window': 20,  # Periods for volatility calculation
            'trend_window': 50,       # Periods for trend calculation
            'volume_window': 20,      # Periods for volume analysis
            'liquidity_window': 10,   # Periods for liquidity calculation
            'regime_confidence_threshold': 0.7
        }
        
        # Default thresholds for different regimes
        self.default_thresholds = {
            RegimeType.TRENDING: {
                'volume_spike_threshold': 2.0,
                'breakout_threshold': 1.5,
                'anomaly_threshold': 0.6
            },
            RegimeType.RANGING: {
                'volume_spike_threshold': 3.0,
                'breakout_threshold': 2.5,
                'anomaly_threshold': 0.8
            },
            RegimeType.HIGH_VOLATILITY: {
                'volume_spike_threshold': 2.5,
                'breakout_threshold': 2.0,
                'anomaly_threshold': 0.7
            },
            RegimeType.LOW_VOLATILITY: {
                'volume_spike_threshold': 4.0,
                'breakout_threshold': 3.0,
                'anomaly_threshold': 0.9
            }
        }
        
        self.logger.info("📊 Market Regime Detection Service initialized")
    
    async def detect_market_regime(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]) -> MarketRegime:
        """Detect current market regime for a symbol"""
        try:
            if not ohlcv_data or len(ohlcv_data) < max(self.regime_params.values()):
                return self._get_default_regime()
            
            df = pd.DataFrame(ohlcv_data)
            
            # Calculate regime features
            regime_features = await self._calculate_regime_features(df)
            
            # Classify regime type
            regime_type = self._classify_regime_type(regime_features)
            
            # Classify volatility regime
            volatility_regime = self._classify_volatility_regime(regime_features.volatility)
            
            # Classify liquidity regime
            liquidity_regime = self._classify_liquidity_regime(regime_features.liquidity_score)
            
            # Calculate overall confidence
            regime_confidence = self._calculate_regime_confidence(regime_features, regime_type)
            
            # Create market regime object
            market_regime = MarketRegime(
                regime_type=regime_type,
                volatility_regime=volatility_regime,
                liquidity_regime=liquidity_regime,
                regime_confidence=regime_confidence,
                regime_features=regime_features,
                metadata={
                    'detection_timestamp': datetime.now().isoformat(),
                    'data_points': len(ohlcv_data),
                    'analysis_window': max(self.regime_params.values())
                }
            )
            
            # Store regime in database
            await self._store_market_regime(symbol, timeframe, market_regime)
            
            # Update regime-specific thresholds
            await self._update_regime_thresholds(symbol, market_regime)
            
            self.logger.info(f"📊 Detected {regime_type.value} regime for {symbol} with confidence {regime_confidence:.2f}")
            
            return market_regime
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting market regime: {e}")
            return self._get_default_regime()
    
    async def _calculate_regime_features(self, df: pd.DataFrame) -> RegimeFeatures:
        """Calculate market regime features"""
        try:
            # Calculate volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(window=self.regime_params['volatility_window']).std().iloc[-1]
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(df)
            
            # Calculate volume consistency
            volume_consistency = self._calculate_volume_consistency(df)
            
            # Calculate price range
            price_range = (df['high'].max() - df['low'].min()) / df['close'].mean()
            
            # Calculate liquidity score
            liquidity_score = self._calculate_liquidity_score(df)
            
            # Calculate regime confidence
            regime_confidence = self._calculate_feature_confidence(df)
            
            return RegimeFeatures(
                volatility=float(volatility),
                trend_strength=float(trend_strength),
                volume_consistency=float(volume_consistency),
                price_range=float(price_range),
                liquidity_score=float(liquidity_score),
                regime_confidence=float(regime_confidence)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime features: {e}")
            return RegimeFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using linear regression"""
        try:
            # Use linear regression on close prices
            x = np.arange(len(df))
            y = df['close'].values
            
            # Calculate linear regression
            slope, intercept = np.polyfit(x, y, 1)
            
            # Calculate R-squared (trend strength)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return float(r_squared)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_volume_consistency(self, df: pd.DataFrame) -> float:
        """Calculate volume consistency"""
        try:
            # Calculate volume coefficient of variation
            volume_cv = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 1.0
            
            # Convert to consistency score (lower CV = higher consistency)
            consistency = 1.0 / (1.0 + volume_cv)
            
            return float(consistency)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating volume consistency: {e}")
            return 0.0
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and price impact"""
        try:
            # Calculate average volume
            avg_volume = df['volume'].mean()
            
            # Calculate price impact (high volume with low price change = high liquidity)
            price_changes = df['close'].pct_change().abs()
            avg_price_impact = price_changes.mean()
            
            # Normalize volume and price impact
            volume_score = min(avg_volume / 1000, 1.0)  # Normalize to 0-1
            impact_score = max(1.0 - avg_price_impact, 0.0)  # Lower impact = higher score
            
            # Combine scores
            liquidity_score = (volume_score + impact_score) / 2
            
            return float(liquidity_score)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating liquidity score: {e}")
            return 0.0
    
    def _calculate_feature_confidence(self, df: pd.DataFrame) -> float:
        """Calculate confidence in regime features"""
        try:
            # Calculate confidence based on data quality and consistency
            confidence_factors = []
            
            # Data completeness
            completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            confidence_factors.append(completeness)
            
            # Data consistency (low variance in key metrics)
            volume_cv = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 1.0
            volume_consistency = 1.0 / (1.0 + volume_cv)
            confidence_factors.append(volume_consistency)
            
            # Price stability
            price_volatility = df['close'].pct_change().std()
            price_stability = max(1.0 - price_volatility, 0.0)
            confidence_factors.append(price_stability)
            
            # Average confidence
            avg_confidence = np.mean(confidence_factors)
            
            return float(avg_confidence)
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating feature confidence: {e}")
            return 0.5
    
    def _classify_regime_type(self, features: RegimeFeatures) -> RegimeType:
        """Classify market regime type based on features"""
        try:
            # High volatility regime
            if features.volatility > 0.03:  # 3% volatility threshold
                return RegimeType.HIGH_VOLATILITY
            
            # Low volatility regime
            if features.volatility < 0.01:  # 1% volatility threshold
                return RegimeType.LOW_VOLATILITY
            
            # Trending vs ranging based on trend strength
            if features.trend_strength > 0.7:  # Strong trend
                return RegimeType.TRENDING
            else:
                return RegimeType.RANGING
            
        except Exception as e:
            self.logger.error(f"❌ Error classifying regime type: {e}")
            return RegimeType.RANGING
    
    def _classify_volatility_regime(self, volatility: float) -> VolatilityRegime:
        """Classify volatility regime"""
        try:
            if volatility > 0.03:
                return VolatilityRegime.HIGH
            elif volatility > 0.015:
                return VolatilityRegime.MEDIUM
            else:
                return VolatilityRegime.LOW
                
        except Exception as e:
            self.logger.error(f"❌ Error classifying volatility regime: {e}")
            return VolatilityRegime.MEDIUM
    
    def _classify_liquidity_regime(self, liquidity_score: float) -> LiquidityRegime:
        """Classify liquidity regime"""
        try:
            if liquidity_score > 0.7:
                return LiquidityRegime.HIGH
            elif liquidity_score > 0.4:
                return LiquidityRegime.MEDIUM
            else:
                return LiquidityRegime.LOW
                
        except Exception as e:
            self.logger.error(f"❌ Error classifying liquidity regime: {e}")
            return LiquidityRegime.MEDIUM
    
    def _calculate_regime_confidence(self, features: RegimeFeatures, regime_type: RegimeType) -> float:
        """Calculate confidence in regime classification"""
        try:
            # Base confidence from feature confidence
            base_confidence = features.regime_confidence
            
            # Adjust based on regime type characteristics
            if regime_type == RegimeType.TRENDING:
                # High confidence if trend is strong
                trend_confidence = min(features.trend_strength * 1.2, 1.0)
                base_confidence = (base_confidence + trend_confidence) / 2
                
            elif regime_type == RegimeType.RANGING:
                # High confidence if low volatility and consistent volume
                range_confidence = (1.0 - features.volatility * 10) * features.volume_consistency
                base_confidence = (base_confidence + range_confidence) / 2
                
            elif regime_type == RegimeType.HIGH_VOLATILITY:
                # High confidence if volatility is clearly high
                vol_confidence = min(features.volatility * 20, 1.0)
                base_confidence = (base_confidence + vol_confidence) / 2
                
            elif regime_type == RegimeType.LOW_VOLATILITY:
                # High confidence if volatility is clearly low
                vol_confidence = 1.0 - min(features.volatility * 50, 1.0)
                base_confidence = (base_confidence + vol_confidence) / 2
            
            return float(max(min(base_confidence, 1.0), 0.0))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime confidence: {e}")
            return 0.5
    
    async def get_regime_thresholds(self, symbol: str, regime_type: RegimeType) -> RegimeThresholds:
        """Get regime-specific thresholds for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT threshold_value, confidence_level 
                    FROM regime_thresholds 
                    WHERE symbol = $1 AND regime_type = $2
                    ORDER BY last_updated DESC LIMIT 1
                """, symbol, regime_type.value)
                
                if row:
                    return RegimeThresholds(
                        volume_spike_threshold=float(row['threshold_value']),
                        breakout_threshold=float(row['threshold_value']) * 1.2,
                        anomaly_threshold=float(row['threshold_value']) * 0.8,
                        confidence_level=float(row['confidence_level'])
                    )
                else:
                    # Return default thresholds
                    defaults = self.default_thresholds[regime_type]
                    return RegimeThresholds(
                        volume_spike_threshold=defaults['volume_spike_threshold'],
                        breakout_threshold=defaults['breakout_threshold'],
                        anomaly_threshold=defaults['anomaly_threshold'],
                        confidence_level=0.7
                    )
                    
        except Exception as e:
            self.logger.error(f"❌ Error getting regime thresholds: {e}")
            # Return default thresholds
            defaults = self.default_thresholds[regime_type]
            return RegimeThresholds(
                volume_spike_threshold=defaults['volume_spike_threshold'],
                breakout_threshold=defaults['breakout_threshold'],
                anomaly_threshold=defaults['anomaly_threshold'],
                confidence_level=0.7
            )
    
    async def _store_market_regime(self, symbol: str, timeframe: str, market_regime: MarketRegime):
        """Store market regime in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_regimes (
                        symbol, timeframe, timestamp, regime_type, volatility_regime,
                        liquidity_regime, regime_confidence, regime_features, regime_metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, symbol, timeframe, datetime.now(), market_regime.regime_type.value,
                     market_regime.volatility_regime.value, market_regime.liquidity_regime.value,
                     market_regime.regime_confidence, json.dumps({
                         'volatility': market_regime.regime_features.volatility,
                         'trend_strength': market_regime.regime_features.trend_strength,
                         'volume_consistency': market_regime.regime_features.volume_consistency,
                         'price_range': market_regime.regime_features.price_range,
                         'liquidity_score': market_regime.regime_features.liquidity_score,
                         'regime_confidence': market_regime.regime_features.regime_confidence
                     }), json.dumps(market_regime.metadata))
                
        except Exception as e:
            self.logger.error(f"❌ Error storing market regime: {e}")
    
    async def _update_regime_thresholds(self, symbol: str, market_regime: MarketRegime):
        """Update regime-specific thresholds"""
        try:
            # Get default thresholds for the regime
            defaults = self.default_thresholds[market_regime.regime_type]
            
            # Adjust thresholds based on regime confidence
            confidence_multiplier = market_regime.regime_confidence
            
            # Update volume spike threshold
            volume_threshold = defaults['volume_spike_threshold'] * (1.0 + (1.0 - confidence_multiplier))
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO regime_thresholds (
                        symbol, regime_type, threshold_type, threshold_value, confidence_level, sample_size
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (symbol, regime_type, threshold_type) 
                    DO UPDATE SET 
                        threshold_value = $4,
                        confidence_level = $5,
                        last_updated = NOW()
                """, symbol, market_regime.regime_type.value, 'volume_spike', 
                     volume_threshold, market_regime.regime_confidence, 100)
                
        except Exception as e:
            self.logger.error(f"❌ Error updating regime thresholds: {e}")
    
    def _get_default_regime(self) -> MarketRegime:
        """Get default market regime when detection fails"""
        return MarketRegime(
            regime_type=RegimeType.RANGING,
            volatility_regime=VolatilityRegime.MEDIUM,
            liquidity_regime=LiquidityRegime.MEDIUM,
            regime_confidence=0.5,
            regime_features=RegimeFeatures(0.02, 0.5, 0.5, 0.02, 0.5, 0.5),
            metadata={'default': True}
        )
    
    async def get_current_regime(self, symbol: str, timeframe: str) -> Optional[MarketRegime]:
        """Get current market regime from database"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM market_regimes 
                    WHERE symbol = $1 AND timeframe = $2 
                    ORDER BY timestamp DESC LIMIT 1
                """, symbol, timeframe)
                
                if row:
                    # Reconstruct MarketRegime object
                    regime_features = RegimeFeatures(
                        volatility=row['regime_features']['volatility'],
                        trend_strength=row['regime_features']['trend_strength'],
                        volume_consistency=row['regime_features']['volume_consistency'],
                        price_range=row['regime_features']['price_range'],
                        liquidity_score=row['regime_features']['liquidity_score'],
                        regime_confidence=row['regime_features']['regime_confidence']
                    )
                    
                    return MarketRegime(
                        regime_type=RegimeType(row['regime_type']),
                        volatility_regime=VolatilityRegime(row['volatility_regime']),
                        liquidity_regime=LiquidityRegime(row['liquidity_regime']),
                        regime_confidence=float(row['regime_confidence']),
                        regime_features=regime_features,
                        metadata=row['regime_metadata']
                    )
                
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error getting current regime: {e}")
            return None
    
    async def get_regime_history(self, symbol: str, timeframe: str, hours: int = 24) -> List[Dict]:
        """Get market regime history"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM market_regimes 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 hour' * $3
                    ORDER BY timestamp DESC
                """, symbol, timeframe, hours)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error getting regime history: {e}")
            return []
    
    async def get_regime_statistics(self, symbol: str, timeframe: str, days: int = 7) -> Dict:
        """Get regime statistics for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT regime_type, COUNT(*) as count, AVG(regime_confidence) as avg_confidence
                    FROM market_regimes 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 day' * $3
                    GROUP BY regime_type
                """, symbol, timeframe, days)
                
                stats = {
                    'total_regimes': 0,
                    'regime_distribution': {},
                    'avg_confidence': 0.0
                }
                
                total_count = 0
                total_confidence = 0.0
                
                for row in rows:
                    regime_type = row['regime_type']
                    count = row['count']
                    confidence = row['avg_confidence']
                    
                    stats['regime_distribution'][regime_type] = {
                        'count': count,
                        'percentage': 0.0,
                        'avg_confidence': float(confidence) if confidence else 0.0
                    }
                    
                    total_count += count
                    total_confidence += count * (confidence or 0.0)
                
                stats['total_regimes'] = total_count
                if total_count > 0:
                    stats['avg_confidence'] = total_confidence / total_count
                    
                    # Calculate percentages
                    for regime_data in stats['regime_distribution'].values():
                        regime_data['percentage'] = (regime_data['count'] / total_count) * 100
                
                return stats
                
        except Exception as e:
            self.logger.error(f"❌ Error getting regime statistics: {e}")
            return {'total_regimes': 0, 'regime_distribution': {}, 'avg_confidence': 0.0}
