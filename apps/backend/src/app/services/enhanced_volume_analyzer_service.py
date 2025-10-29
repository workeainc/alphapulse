#!/usr/bin/env python3
"""
Enhanced Volume Analyzer Service for AlphaPlus
Advanced volume analysis with real-time pattern detection and TimescaleDB integration
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
import asyncpg

logger = logging.getLogger(__name__)

class VolumePatternType(Enum):
    """Types of volume patterns"""
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DIVERGENCE = "volume_divergence"
    VOLUME_CLIMAX = "volume_climax"
    VOLUME_DRY_UP = "volume_dry_up"
    VOLUME_ABSORPTION = "volume_absorption"
    VOLUME_BREAKOUT = "volume_breakout"
    VOLUME_NODE = "volume_node"

@dataclass
class VolumeAnalysisResult:
    """Volume analysis result with comprehensive metrics"""
    symbol: str
    timeframe: str
    timestamp: datetime
    volume_ratio: float
    volume_trend: str
    volume_positioning_score: float
    order_book_imbalance: float
    buy_volume_ratio: float
    sell_volume_ratio: float
    volume_breakout: bool
    # Advanced metrics
    vwap: Optional[float] = None
    cumulative_volume_delta: Optional[float] = None
    relative_volume: Optional[float] = None
    volume_weighted_price: Optional[float] = None
    volume_flow_imbalance: Optional[float] = None
    # Delta profile metrics
    delta_profile: Optional[Dict] = None
    support_levels: Optional[List[float]] = None
    resistance_levels: Optional[List[float]] = None
    volume_nodes: Optional[List[Dict]] = None
    # Liquidity metrics
    liquidity_score: Optional[float] = None
    spoofing_detected: Optional[bool] = None
    whale_activity: Optional[bool] = None
    # Pattern detection
    volume_pattern_type: Optional[str] = None
    volume_pattern_strength: Optional[str] = None
    volume_pattern_confidence: Optional[float] = None
    volume_analysis: Optional[str] = None
    volume_context: Optional[Dict] = None

class EnhancedVolumeAnalyzerService:
    """Enhanced volume analyzer service with real-time pattern detection"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.volume_spike_thresholds = {
            'weak': 1.5,
            'moderate': 2.0,
            'strong': 3.0,
            'very_strong': 5.0
        }
        
        self.volume_breakout_threshold = 2.0
        
        # Advanced configuration
        self.context_aware_thresholds = {
            'high_volatility': 1.8,  # Lower threshold for volatile markets
            'low_volatility': 2.5,   # Higher threshold for stable markets
            'high_liquidity': 1.5,   # Lower threshold for liquid assets
            'low_liquidity': 3.0     # Higher threshold for illiquid assets
        }
        
        self.multi_timeframe_weights = {
            '1m': 0.3,
            '5m': 0.4,
            '15m': 0.2,
            '1h': 0.1
        }
        
        # Phase 6: Real-time streaming configuration
        self.streaming_config = {
            'enable_real_time_streaming': True,
            'streaming_interval_ms': 100,  # 100ms intervals for real-time updates
            'batch_size': 10,  # Process 10 updates at once for efficiency
            'enable_compression': True,  # Enable TimescaleDB compression
            'compression_after_hours': 24,  # Compress data after 24 hours
            'enable_continuous_aggregates': True  # Enable real-time aggregates
        }
        
        # Phase 7: ML integration configuration
        self.ml_config = {
            'enable_ml_predictions': True,
            'ml_prediction_threshold': 0.7,  # Minimum confidence for ML predictions
            'ml_feature_generation': True,  # Generate ML features
            'ml_model_cache_size': 10,  # Number of models to cache
            'ml_prediction_interval': 60  # Seconds between ML predictions
        }
        
        # Phase 8: Advanced ML Features configuration
        self.phase8_config = {
            'enable_anomaly_detection': True,
            'enable_reinforcement_learning': True,
            'enable_trading_integration': True,
            'anomaly_confidence_threshold': 0.7,
            'rl_agent_confidence_threshold': 0.6,
            'trading_signal_threshold': 0.7,
            'anomaly_detection_interval': 30,  # seconds
            'rl_episode_length': 100,
            'trading_optimization_interval': 60  # seconds
        }
        
        # Symbol-specific threshold calibration (Phase 6 enhancement)
        self.symbol_thresholds = {
            'BTCUSDT': {'base_threshold': 2.0, 'volatility_multiplier': 1.0, 'liquidity_multiplier': 1.0},
            'ETHUSDT': {'base_threshold': 2.2, 'volatility_multiplier': 1.1, 'liquidity_multiplier': 0.95},
            'ADAUSDT': {'base_threshold': 1.6, 'volatility_multiplier': 0.8, 'liquidity_multiplier': 1.2},
            'SOLUSDT': {'base_threshold': 1.8, 'volatility_multiplier': 0.9, 'liquidity_multiplier': 1.1},
            'DOTUSDT': {'base_threshold': 1.7, 'volatility_multiplier': 0.85, 'liquidity_multiplier': 1.15}
        }
        self.logger.info("🔍 Enhanced Volume Analyzer Service initialized with Phase 8 Advanced ML Features")
    
    async def analyze_volume(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]) -> VolumeAnalysisResult:
        """Perform comprehensive volume analysis for a symbol"""
        try:
            if not ohlcv_data or len(ohlcv_data) < 20:
                self.logger.warning(f"Insufficient data for volume analysis: {symbol}")
                return self._create_default_analysis(symbol, timeframe)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate volume metrics
            volume_ratio = await self._calculate_volume_ratio(df)
            volume_trend = await self._determine_volume_trend(df)
            volume_positioning_score = await self._calculate_volume_positioning_score(df)
            
            # Calculate order book metrics
            order_book_imbalance = await self._calculate_order_book_imbalance(df)
            buy_volume_ratio, sell_volume_ratio = await self._calculate_volume_ratios(df)
            
            # Calculate advanced volume metrics
            vwap = await self._calculate_vwap(df)
            cumulative_volume_delta = await self._calculate_cumulative_volume_delta(df)
            relative_volume = await self._calculate_relative_volume(df)
            volume_weighted_price = await self._calculate_volume_weighted_price(df)
            volume_flow_imbalance = await self._calculate_volume_flow_imbalance(df)
             
            # Calculate delta profile and volume nodes
            delta_profile = await self._calculate_delta_profile(df, symbol, timeframe)
            support_levels, resistance_levels = await self._identify_support_resistance_levels(df)
            volume_nodes = await self._detect_volume_nodes(df, symbol, timeframe)
            
            # Calculate liquidity metrics
            liquidity_score = await self._calculate_liquidity_score(df)
            spoofing_detected = await self._detect_spoofing(df)
            whale_activity = await self._detect_whale_activity(df)
            
            # Detect volume breakouts
            volume_breakout = await self._detect_volume_breakout(df, volume_ratio)
            
            # Detect volume patterns
            volume_pattern = await self._detect_volume_patterns(df, symbol, timeframe)
            
            # Generate analysis text
            volume_analysis = await self._generate_volume_analysis(
                volume_ratio, volume_trend, order_book_imbalance, 
                volume_positioning_score, buy_volume_ratio, sell_volume_ratio, 
                volume_breakout, volume_pattern
            )
            
            # Create volume context
            volume_context = {
                'volume_metrics': {
                    'volume_ratio': volume_ratio,
                    'volume_trend': volume_trend,
                    'volume_positioning_score': volume_positioning_score
                },
                'advanced_metrics': {
                    'vwap': vwap,
                    'cumulative_volume_delta': cumulative_volume_delta,
                    'relative_volume': relative_volume,
                    'volume_weighted_price': volume_weighted_price,
                    'volume_flow_imbalance': volume_flow_imbalance
                },
                'delta_profile_metrics': {
                    'delta_profile': delta_profile,
                    'support_levels': support_levels,
                    'resistance_levels': resistance_levels,
                    'volume_nodes': volume_nodes
                },
                'liquidity_metrics': {
                    'liquidity_score': liquidity_score,
                    'spoofing_detected': spoofing_detected,
                    'whale_activity': whale_activity
                },
                'order_book_metrics': {
                    'order_book_imbalance': order_book_imbalance,
                    'buy_volume_ratio': buy_volume_ratio,
                    'sell_volume_ratio': sell_volume_ratio
                },
                'pattern_metrics': {
                    'volume_breakout': volume_breakout,
                    'pattern_type': volume_pattern['pattern_type'] if volume_pattern else None,
                    'pattern_strength': volume_pattern['pattern_strength'] if volume_pattern else None
                },
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Create result
            result = VolumeAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=df['timestamp'].iloc[-1],
                volume_ratio=volume_ratio,
                volume_trend=volume_trend,
                volume_positioning_score=volume_positioning_score,
                order_book_imbalance=order_book_imbalance,
                buy_volume_ratio=buy_volume_ratio,
                sell_volume_ratio=sell_volume_ratio,
                volume_breakout=volume_breakout,
                # Advanced metrics
                vwap=vwap,
                cumulative_volume_delta=cumulative_volume_delta,
                relative_volume=relative_volume,
                volume_weighted_price=volume_weighted_price,
                volume_flow_imbalance=volume_flow_imbalance,
                # Delta profile metrics
                delta_profile=delta_profile,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                volume_nodes=volume_nodes,
                # Liquidity metrics
                liquidity_score=liquidity_score,
                spoofing_detected=spoofing_detected,
                whale_activity=whale_activity,
                # Pattern detection
                volume_pattern_type=volume_pattern['pattern_type'] if volume_pattern else None,
                volume_pattern_strength=volume_pattern['pattern_strength'] if volume_pattern else None,
                volume_pattern_confidence=volume_pattern['pattern_confidence'] if volume_pattern else None,
                volume_analysis=volume_analysis,
                volume_context=volume_context
            )
            
            # Store in database
            await self._store_volume_analysis(result)
            
            # Phase 6: Enable TimescaleDB compression and create ML dataset
            if self.streaming_config['enable_compression']:
                await self._enable_timescaledb_compression(symbol, timeframe)
            
            # Create ML training dataset entry
            await self._create_ml_training_dataset(symbol, timeframe, result)
            
            # Phase 7: ML integration
            if self.ml_config['enable_ml_predictions']:
                await self._generate_ml_features_and_predict(symbol, timeframe, ohlcv_data)
            
            # Phase 8: Advanced ML Features integration
            if self.phase8_config['enable_anomaly_detection'] or self.phase8_config['enable_reinforcement_learning'] or self.phase8_config['enable_trading_integration']:
                await self._integrate_phase8_features(symbol, timeframe, ohlcv_data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in volume analysis for {symbol}: {e}")
            return self._create_default_analysis(symbol, timeframe)
    
    async def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volume ratio compared to average"""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            
            if avg_volume == 0:
                return 1.0
            
            return float(current_volume / avg_volume)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume ratio: {e}")
            return 1.0
    
    async def _determine_volume_trend(self, df: pd.DataFrame) -> str:
        """Determine volume trend direction"""
        try:
            volumes = df['volume'].tail(10).values
            x = np.arange(len(volumes))
            
            if len(volumes) < 2:
                return "stable"
            
            slope = np.polyfit(x, volumes, 1)[0]
            
            if slope > 0.05 * np.mean(volumes):
                return "increasing"
            elif slope < -0.05 * np.mean(volumes):
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            self.logger.error(f"Error determining volume trend: {e}")
            return "stable"
    
    async def _calculate_volume_positioning_score(self, df: pd.DataFrame) -> float:
        """Calculate volume positioning score (0-1)"""
        try:
            volume_ratio = await self._calculate_volume_ratio(df)
            volume_trend = await self._determine_volume_trend(df)
            
            base_score = min(volume_ratio / 3.0, 1.0)
            trend_adjustment = 0.1 if volume_trend == "increasing" else -0.1 if volume_trend == "decreasing" else 0
            
            recent_volumes = df['volume'].tail(5).values
            volume_consistency = 1.0 - (np.std(recent_volumes) / np.mean(recent_volumes))
            consistency_bonus = max(0, volume_consistency * 0.2)
            
            final_score = base_score + trend_adjustment + consistency_bonus
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating volume positioning score: {e}")
            return 0.5
    
    async def _calculate_order_book_imbalance(self, df: pd.DataFrame) -> float:
        """Calculate order book imbalance (-1 to 1)"""
        try:
            price_change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            volume_ratio = await self._calculate_volume_ratio(df)
            
            imbalance = price_change * volume_ratio
            return max(-1.0, min(1.0, imbalance))
            
        except Exception as e:
            self.logger.error(f"Error calculating order book imbalance: {e}")
            return 0.0
    
    async def _calculate_volume_ratios(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate buy/sell volume ratios"""
        try:
            price_change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            
            if price_change > 0:
                buy_ratio = 0.6 + (price_change * 0.4)
                sell_ratio = 1.0 - buy_ratio
            else:
                sell_ratio = 0.6 + (abs(price_change) * 0.4)
                buy_ratio = 1.0 - sell_ratio
            
            return float(buy_ratio), float(sell_ratio)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume ratios: {e}")
            return 0.5, 0.5
    
    async def _detect_volume_breakout(self, df: pd.DataFrame, volume_ratio: float) -> bool:
        """Detect volume breakouts with enhanced criteria"""
        try:
            # Basic volume threshold check
            volume_breakout = volume_ratio >= self.volume_breakout_threshold
            
            if not volume_breakout:
                return False
            
            # Enhanced criteria for volume-weighted breakouts
            sustained_breakout = await self._check_sustained_volume_breakout(df, volume_ratio)
            cvd_confirmation = await self._check_cvd_breakout_confirmation(df)
            
            return volume_breakout and sustained_breakout and cvd_confirmation
            
        except Exception as e:
            self.logger.error(f"Error detecting volume breakout: {e}")
            return False
    
    async def _check_sustained_volume_breakout(self, df: pd.DataFrame, current_volume_ratio: float) -> bool:
        """Check if volume breakout is sustained over multiple periods"""
        try:
            if len(df) < 5:
                return True  # Not enough data to check sustainability
            
            # Check last 3 periods for sustained volume
            recent_volumes = df['volume'].tail(3)
            recent_avg_volume = df['volume'].tail(20).mean()
            
            sustained_count = 0
            for volume in recent_volumes:
                if volume >= recent_avg_volume * 1.5:  # 1.5x threshold for sustainability
                    sustained_count += 1
            
            # At least 2 out of 3 recent periods should have elevated volume
            return sustained_count >= 2
            
        except Exception as e:
            self.logger.error(f"Error checking sustained volume breakout: {e}")
            return True
    
    async def _check_cvd_breakout_confirmation(self, df: pd.DataFrame) -> bool:
        """Check if CVD confirms the breakout direction"""
        try:
            if len(df) < 10:
                return True  # Not enough data for CVD confirmation
            
            # Calculate recent CVD trend
            recent_cvd = []
            for i in range(len(df) - 5, len(df)):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:
                    recent_cvd.append(candle['volume'])  # Buy volume
                else:
                    recent_cvd.append(-candle['volume'])  # Sell volume
            
            cvd_sum = sum(recent_cvd)
            
            # Check if CVD aligns with price movement
            price_change = df['close'].iloc[-1] - df['close'].iloc[-5]
            
            if price_change > 0 and cvd_sum > 0:  # Bullish breakout with buying pressure
                return True
            elif price_change < 0 and cvd_sum < 0:  # Bearish breakout with selling pressure
                return True
            else:
                # Mixed signals - require higher confidence
                return abs(cvd_sum) > df['volume'].tail(10).mean() * 2
                
        except Exception as e:
            self.logger.error(f"Error checking CVD breakout confirmation: {e}")
            return True
    
    async def _detect_volume_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Detect advanced volume patterns with context-aware analysis"""
        try:
            volume_ratio = await self._calculate_volume_ratio(df)
            price_change = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
            
            # Calculate market context
            volatility = await self._calculate_volatility(df)
            liquidity_score = await self._calculate_liquidity_score(df)
            
            # Context-aware threshold adjustment
            adjusted_threshold = await self._get_context_aware_threshold(volatility, liquidity_score, symbol)
            
            # Enhanced Volume Spike Detection
            if volume_ratio >= adjusted_threshold:
                strength = await self._determine_pattern_strength(volume_ratio, adjusted_threshold)
                confidence = await self._calculate_pattern_confidence(volume_ratio, volatility, liquidity_score)
                
                return {
                    'pattern_type': VolumePatternType.VOLUME_SPIKE.value,
                    'pattern_strength': strength,
                    'pattern_confidence': confidence,
                    'pattern_description': f"Context-aware volume spike: {volume_ratio:.2f}x average (threshold: {adjusted_threshold:.2f})"
                }
            
            # Volume Climax Detection (high volume + low price change)
            if volume_ratio >= 2.0 and abs(price_change) < 0.005:
                return {
                    'pattern_type': VolumePatternType.VOLUME_CLIMAX.value,
                    'pattern_strength': 'moderate',
                    'pattern_confidence': 0.7,
                    'pattern_description': f"Volume climax: {volume_ratio:.2f}x volume with minimal price change"
                }
            
            # Volume Dry-up Detection (low volume + low volatility)
            if volume_ratio < 0.5 and volatility < 0.01:
                return {
                    'pattern_type': VolumePatternType.VOLUME_DRY_UP.value,
                    'pattern_strength': 'weak',
                    'pattern_confidence': 0.6,
                    'pattern_description': f"Volume dry-up: {volume_ratio:.2f}x volume with low volatility"
                }
            
            # Volume Absorption Detection
            absorption_result = await self._detect_volume_absorption(df)
            if absorption_result:
                return absorption_result
            
            # Volume Divergence Detection
            divergence = await self._detect_volume_divergence(df)
            if divergence:
                return divergence
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting volume patterns: {e}")
            return None
    
    async def _generate_volume_analysis(self, volume_ratio: float, volume_trend: str, 
                                      order_book_imbalance: float, volume_positioning_score: float,
                                      buy_volume_ratio: float, sell_volume_ratio: float,
                                      volume_breakout: bool, volume_pattern) -> str:
        """Generate human-readable volume analysis with advanced context"""
        try:
            analysis_parts = []
            
            # Volume analysis
            if volume_ratio > 2.0:
                analysis_parts.append(f"High volume: {volume_ratio:.2f}x average")
            elif volume_ratio < 0.5:
                analysis_parts.append(f"Low volume: {volume_ratio:.2f}x average")
            else:
                analysis_parts.append(f"Normal volume: {volume_ratio:.2f}x average")
            
            analysis_parts.append(f"Trend: {volume_trend}")
            
            # Order book analysis
            if abs(order_book_imbalance) > 0.3:
                side = "buying" if order_book_imbalance > 0 else "selling"
                analysis_parts.append(f"Strong {side} pressure")
            
            # Volume breakout
            if volume_breakout:
                analysis_parts.append("Volume breakout confirmed")
            
            # Advanced pattern analysis
            if volume_pattern:
                pattern_type = volume_pattern['pattern_type']
                pattern_strength = volume_pattern.get('pattern_strength', 'unknown')
                pattern_confidence = volume_pattern.get('pattern_confidence', 0)
                
                analysis_parts.append(f"Pattern: {pattern_type} ({pattern_strength}, {pattern_confidence:.1%} confidence)")
                
                # Add pattern-specific insights
                if pattern_type == VolumePatternType.VOLUME_CLIMAX.value:
                    analysis_parts.append("Volume climax - potential exhaustion")
                elif pattern_type == VolumePatternType.VOLUME_DRY_UP.value:
                    analysis_parts.append("Volume dry-up - consolidation phase")
                elif pattern_type == VolumePatternType.VOLUME_DIVERGENCE.value:
                    analysis_parts.append("Volume divergence - trend weakening")
            
            return " | ".join(analysis_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating volume analysis: {e}")
            return "Volume analysis unavailable"
    
    async def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price (VWAP)"""
        try:
            # Calculate typical price (HLC/3)
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP
            vwap = (df['typical_price'] * df['volume']).sum() / df['volume'].sum()
            return float(vwap)
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return float(df['close'].iloc[-1]) if not df.empty else 0.0
    
    async def _calculate_cumulative_volume_delta(self, df: pd.DataFrame) -> float:
        """Calculate Cumulative Volume Delta (CVD) - buy vs sell imbalance"""
        try:
            cvd = 0.0
            
            for i in range(len(df)):
                close_price = df['close'].iloc[i]
                open_price = df['open'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Determine if this is buy or sell volume based on price movement
                if close_price > open_price:
                    # Price went up, likely more buying
                    cvd += volume
                elif close_price < open_price:
                    # Price went down, likely more selling
                    cvd -= volume
                else:
                    # No price change, neutral
                    pass
            
            return float(cvd)
            
        except Exception as e:
            self.logger.error(f"Error calculating CVD: {e}")
            return 0.0
    
    async def _calculate_relative_volume(self, df: pd.DataFrame) -> float:
        """Calculate Relative Volume (RVOL) - current volume vs same time average"""
        try:
            current_volume = df['volume'].iloc[-1]
            current_time = df['timestamp'].iloc[-1]
            
            # Get historical data for same time of day (last 7 days)
            # For now, use simple average of recent volumes
            historical_volumes = df['volume'].tail(20).values  # Last 20 periods
            
            if len(historical_volumes) == 0:
                return 1.0
            
            avg_volume = np.mean(historical_volumes)
            
            if avg_volume == 0:
                return 1.0
            
            return float(current_volume / avg_volume)
            
        except Exception as e:
            self.logger.error(f"Error calculating RVOL: {e}")
            return 1.0
    
    async def _calculate_volume_weighted_price(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Price for current period"""
        try:
            if df.empty:
                return 0.0
            
            # Use typical price for current period
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_close = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            typical_price = (current_high + current_low + current_close) / 3
            return float(typical_price)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume weighted price: {e}")
            return float(df['close'].iloc[-1]) if not df.empty else 0.0
    
    async def _calculate_volume_flow_imbalance(self, df: pd.DataFrame) -> float:
        """Calculate Volume Flow Imbalance - advanced order flow analysis"""
        try:
            if len(df) < 2:
                return 0.0
            
            # Calculate volume flow based on price movement and volume
            flow_imbalance = 0.0
            
            for i in range(1, len(df)):
                prev_close = df['close'].iloc[i-1]
                curr_close = df['close'].iloc[i]
                volume = df['volume'].iloc[i]
                
                # Calculate price change percentage
                price_change_pct = (curr_close - prev_close) / prev_close
                
                # Volume flow imbalance: positive for buying pressure, negative for selling
                if price_change_pct > 0:
                    flow_imbalance += volume * price_change_pct
                else:
                    flow_imbalance += volume * price_change_pct
            
            # Normalize by total volume
            total_volume = df['volume'].sum()
            if total_volume > 0:
                return float(flow_imbalance / total_volume)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating volume flow imbalance: {e}")
            return 0.0
    
    async def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate price volatility"""
        try:
            if len(df) < 10:
                return 0.0
            
            # Calculate price changes
            price_changes = df['close'].pct_change().dropna()
            
            # Calculate volatility as standard deviation
            volatility = price_changes.std()
            return float(volatility)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    async def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and price stability"""
        try:
            if len(df) < 10:
                return 0.5
            
            # Average volume
            avg_volume = df['volume'].mean()
            
            # Volume consistency
            volume_std = df['volume'].std()
            volume_consistency = 1.0 - (volume_std / avg_volume) if avg_volume > 0 else 0.0
            
            # Price stability
            price_changes = df['close'].pct_change().dropna()
            price_stability = 1.0 - price_changes.std()
            
            # Combined liquidity score
            liquidity_score = (volume_consistency + price_stability) / 2.0
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 0.5
    
    async def _get_context_aware_threshold(self, volatility: float, liquidity_score: float, symbol: str = None) -> float:
        """Get context-aware volume threshold with enhanced symbol-specific adjustments (Phase 6)"""
        try:
            base_threshold = self.volume_spike_thresholds['moderate']  # 2.0
            
            # Get enhanced symbol-specific threshold with volatility and liquidity inputs
            if symbol:
                base_threshold = await self._get_symbol_specific_threshold(symbol, base_threshold, volatility, liquidity_score)
            
            # Additional market condition adjustments
            market_adjustment = 0.0
            
            # Volatility adjustment (additional to symbol-specific)
            if volatility > 0.03:  # High volatility
                market_adjustment -= 0.1
            elif volatility < 0.01:  # Low volatility
                market_adjustment += 0.1
            
            # Liquidity adjustment (additional to symbol-specific)
            if liquidity_score > 0.7:  # High liquidity
                market_adjustment -= 0.1
            elif liquidity_score < 0.3:  # Low liquidity
                market_adjustment += 0.2
            
            adjusted_threshold = base_threshold + market_adjustment
            return max(1.0, adjusted_threshold)  # Minimum threshold of 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating context-aware threshold: {e}")
            return self.volume_spike_thresholds['moderate']
    
    async def _determine_pattern_strength(self, volume_ratio: float, threshold: float) -> str:
        """Determine pattern strength based on volume ratio and threshold"""
        try:
            ratio_above_threshold = volume_ratio / threshold
            
            if ratio_above_threshold >= 2.0:
                return 'very_strong'
            elif ratio_above_threshold >= 1.5:
                return 'strong'
            elif ratio_above_threshold >= 1.2:
                return 'moderate'
            else:
                return 'weak'
                
        except Exception as e:
            self.logger.error(f"Error determining pattern strength: {e}")
            return 'moderate'
    
    async def _calculate_pattern_confidence(self, volume_ratio: float, volatility: float, liquidity_score: float) -> float:
        """Calculate pattern confidence based on multiple factors"""
        try:
            # Base confidence from volume ratio
            base_confidence = min(volume_ratio / 5.0, 1.0)
            
            # Volatility adjustment (lower volatility = higher confidence)
            volatility_adjustment = max(0, (0.02 - volatility) * 10)  # Bonus for low volatility
            
            # Liquidity adjustment (higher liquidity = higher confidence)
            liquidity_adjustment = liquidity_score * 0.2
            
            # Volume consistency bonus
            volume_consistency_bonus = 0.1 if volume_ratio > 2.0 else 0.0
            
            total_confidence = base_confidence + volatility_adjustment + liquidity_adjustment + volume_consistency_bonus
            return max(0.0, min(1.0, total_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {e}")
            return 0.5
    
    async def _detect_volume_divergence(self, df: pd.DataFrame):
        """Detect volume-price divergence patterns"""
        try:
            if len(df) < 20:
                return None
            
            # Calculate price and volume trends
            price_trend = await self._calculate_price_trend(df)
            volume_trend = await self._calculate_volume_trend_numeric(df)
            
            # Detect bullish divergence (price down, volume up)
            if price_trend < -0.01 and volume_trend > 0.1:
                return {
                    'pattern_type': VolumePatternType.VOLUME_DIVERGENCE.value,
                    'pattern_strength': 'moderate',
                    'pattern_confidence': 0.65,
                    'pattern_description': f"Bullish volume divergence: price down {abs(price_trend):.2%}, volume up {volume_trend:.2%}"
                }
            
            # Detect bearish divergence (price up, volume down)
            elif price_trend > 0.01 and volume_trend < -0.1:
                return {
                    'pattern_type': VolumePatternType.VOLUME_DIVERGENCE.value,
                    'pattern_strength': 'moderate',
                    'pattern_confidence': 0.65,
                    'pattern_description': f"Bearish volume divergence: price up {price_trend:.2%}, volume down {abs(volume_trend):.2%}"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting volume divergence: {e}")
            return None
    
    async def _calculate_price_trend(self, df: pd.DataFrame) -> float:
        """Calculate price trend as percentage change"""
        try:
            if len(df) < 10:
                return 0.0
            
            start_price = df['close'].iloc[-10]
            end_price = df['close'].iloc[-1]
            
            return (end_price - start_price) / start_price
            
        except Exception as e:
            self.logger.error(f"Error calculating price trend: {e}")
            return 0.0
    
    async def _calculate_volume_trend_numeric(self, df: pd.DataFrame) -> float:
        """Calculate volume trend as percentage change"""
        try:
            if len(df) < 10:
                return 0.0
            
            start_volume = df['volume'].iloc[-10]
            end_volume = df['volume'].iloc[-1]
            
            return (end_volume - start_volume) / start_volume
            
        except Exception as e:
            self.logger.error(f"Error calculating volume trend numeric: {e}")
            return 0.0
    
    async def _calculate_delta_profile(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Calculate delta profile for footprint charts"""
        try:
            if len(df) < 20:
                return {}
            
            # Calculate price bands (0.1% intervals)
            current_price = df['close'].iloc[-1]
            price_range = current_price * 0.002  # 0.2% range
            band_size = price_range / 20  # 20 bands
            
            delta_profile = {}
            
            for i in range(20):
                band_start = current_price - price_range/2 + i * band_size
                band_end = band_start + band_size
                
                # Find candles in this price band
                mask = (df['low'] <= band_end) & (df['high'] >= band_start)
                band_data = df[mask]
                
                if len(band_data) > 0:
                    total_volume = band_data['volume'].sum()
                    buy_volume = band_data[band_data['close'] > band_data['open']]['volume'].sum()
                    sell_volume = total_volume - buy_volume
                    delta_imbalance = buy_volume - sell_volume
                    volume_density = total_volume / len(band_data) if len(band_data) > 0 else 0
                    
                    delta_profile[f"{band_start:.2f}-{band_end:.2f}"] = {
                        'price_level': (band_start + band_end) / 2,
                        'volume_at_level': total_volume,
                        'buy_volume': buy_volume,
                        'sell_volume': sell_volume,
                        'delta_imbalance': delta_imbalance,
                        'volume_density': volume_density
                    }
            
            return delta_profile
            
        except Exception as e:
            self.logger.error(f"Error calculating delta profile: {e}")
            return {}
    
    async def _identify_support_resistance_levels(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels from volume data"""
        try:
            if len(df) < 20:
                return [], []
            
            # Find price levels with high volume
            volume_threshold = df['volume'].quantile(0.8)
            high_volume_candles = df[df['volume'] > volume_threshold]
            
            support_levels = []
            resistance_levels = []
            
            for _, candle in high_volume_candles.iterrows():
                price_level = candle['close']
                
                # Determine if it's support or resistance based on price action
                if candle['close'] > candle['open']:  # Bullish candle
                    if price_level not in resistance_levels:
                        resistance_levels.append(price_level)
                else:  # Bearish candle
                    if price_level not in support_levels:
                        support_levels.append(price_level)
            
            # Sort and return top levels
            support_levels = sorted(support_levels, reverse=True)[:5]
            resistance_levels = sorted(resistance_levels)[:5]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            self.logger.error(f"Error identifying support/resistance levels: {e}")
            return [], []
    
    async def _detect_volume_nodes(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict]:
        """Detect high-volume price nodes"""
        try:
            if len(df) < 20:
                return []
            
            # Find price levels with sustained high volume
            volume_threshold = df['volume'].quantile(0.9)
            high_volume_candles = df[df['volume'] > volume_threshold]
            
            nodes = []
            current_price = df['close'].iloc[-1]
            
            for _, candle in high_volume_candles.iterrows():
                price_level = candle['close']
                volume_cluster = candle['volume']
                
                # Calculate node strength based on volume and price proximity
                price_distance = abs(current_price - price_level) / current_price
                volume_strength = min(volume_cluster / volume_threshold, 3.0)
                node_strength = volume_strength * (1 - price_distance)
                
                if node_strength > 0.5:  # Only significant nodes
                    node_type = 'support' if candle['close'] < candle['open'] else 'resistance'
                    
                    nodes.append({
                        'price_level': price_level,
                        'node_type': node_type,
                        'volume_cluster_size': volume_cluster,
                        'node_strength': node_strength,
                        'distance_from_current': price_distance
                    })
            
            # Sort by strength and return top nodes
            nodes.sort(key=lambda x: x['node_strength'], reverse=True)
            return nodes[:5]
            
        except Exception as e:
            self.logger.error(f"Error detecting volume nodes: {e}")
            return []
    
    async def _detect_spoofing(self, df: pd.DataFrame) -> bool:
        """Detect potential spoofing activity"""
        try:
            if len(df) < 10:
                return False
            
            # Look for unusual volume patterns that might indicate spoofing
            recent_volumes = df['volume'].tail(5)
            volume_mean = recent_volumes.mean()
            volume_std = recent_volumes.std()
            
            # Check for sudden volume spikes followed by quick reversals
            for i in range(len(recent_volumes) - 1):
                if (recent_volumes.iloc[i] > volume_mean + 2 * volume_std and 
                    recent_volumes.iloc[i+1] < volume_mean - volume_std):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting spoofing: {e}")
            return False
    
    async def _detect_whale_activity(self, df: pd.DataFrame) -> bool:
        """Detect potential whale activity"""
        try:
            if len(df) < 10:
                return False
            
            # Look for large volume transactions
            volume_threshold = df['volume'].quantile(0.95)
            recent_volume = df['volume'].iloc[-1]
            
            # Check if recent volume is significantly higher than normal
            if recent_volume > volume_threshold * 1.5:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting whale activity: {e}")
            return False
    
    async def _detect_volume_absorption(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect volume absorption patterns - price stalling with continued volume flow"""
        try:
            if len(df) < 15:
                return None
            
            # Get recent price and volume data
            recent_prices = df['close'].tail(10)
            recent_volumes = df['volume'].tail(10)
            recent_cvd = []
            
            # Calculate recent CVD
            for i in range(len(df) - 10, len(df)):
                candle = df.iloc[i]
                if candle['close'] > candle['open']:
                    recent_cvd.append(candle['volume'])  # Buy volume
                else:
                    recent_cvd.append(-candle['volume'])  # Sell volume
            
            # Check for price stalling (low price movement)
            price_range = recent_prices.max() - recent_prices.min()
            avg_price = recent_prices.mean()
            price_stall_threshold = avg_price * 0.01  # 1% range
            
            price_stalling = price_range <= price_stall_threshold
            
            # Check for continued volume flow
            avg_volume = df['volume'].tail(30).mean()
            recent_volume_elevated = recent_volumes.mean() > avg_volume * 1.2
            
            # Check for consistent CVD direction
            cvd_sum = sum(recent_cvd)
            cvd_consistency = abs(cvd_sum) / sum(abs(x) for x in recent_cvd) if recent_cvd else 0
            
            if price_stalling and recent_volume_elevated and cvd_consistency > 0.6:
                absorption_type = "bullish" if cvd_sum > 0 else "bearish"
                strength = "strong" if cvd_consistency > 0.8 else "moderate"
                confidence = min(0.9, 0.5 + cvd_consistency * 0.4)
                
                return {
                    'pattern_type': VolumePatternType.VOLUME_ABSORPTION.value,
                    'pattern_strength': strength,
                    'pattern_confidence': confidence,
                    'pattern_description': f"Volume absorption ({absorption_type}): Price stalling with {cvd_consistency:.1%} CVD consistency"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting volume absorption: {e}")
            return None
    
    async def _calculate_session_vwap(self, df: pd.DataFrame, session_start: Optional[datetime] = None) -> float:
        """Calculate session-based VWAP"""
        try:
            if session_start:
                # Filter data from session start
                session_data = df[df['timestamp'] >= session_start]
            else:
                # Use daily session (last 24 hours of data)
                session_data = df.tail(1440) if len(df) > 1440 else df  # Assuming 1-minute data
            
            if len(session_data) == 0:
                return float(df['close'].iloc[-1])
            
            session_data['typical_price'] = (session_data['high'] + session_data['low'] + session_data['close']) / 3
            session_vwap = (session_data['typical_price'] * session_data['volume']).sum() / session_data['volume'].sum()
            
            return float(session_vwap)
            
        except Exception as e:
            self.logger.error(f"Error calculating session VWAP: {e}")
            return float(df['close'].iloc[-1])
    
    async def _get_symbol_specific_threshold(self, symbol: str, base_threshold: float, volatility: float = None, liquidity_score: float = None) -> float:
        """Get symbol-specific threshold with dynamic calibration (Phase 6 enhancement)"""
        try:
            # Get symbol configuration
            symbol_config = self.symbol_thresholds.get(symbol, {
                'base_threshold': base_threshold,
                'volatility_multiplier': 1.0,
                'liquidity_multiplier': 1.0
            })
            
            # Start with symbol-specific base threshold
            adjusted_threshold = symbol_config['base_threshold']
            
            # Apply volatility adjustment if available
            if volatility is not None:
                volatility_adjustment = symbol_config['volatility_multiplier']
                if volatility > 0.03:  # High volatility
                    adjusted_threshold *= volatility_adjustment * 0.9  # Lower threshold for high volatility
                elif volatility < 0.01:  # Low volatility
                    adjusted_threshold *= volatility_adjustment * 1.1  # Higher threshold for low volatility
            
            # Apply liquidity adjustment if available
            if liquidity_score is not None:
                liquidity_adjustment = symbol_config['liquidity_multiplier']
                if liquidity_score > 0.7:  # High liquidity
                    adjusted_threshold *= liquidity_adjustment * 0.9  # Lower threshold for high liquidity
                elif liquidity_score < 0.3:  # Low liquidity
                    adjusted_threshold *= liquidity_adjustment * 1.2  # Higher threshold for low liquidity
            
            # Ensure minimum threshold
            return max(1.0, adjusted_threshold)
            
        except Exception as e:
            self.logger.error(f"Error getting symbol-specific threshold: {e}")
            return base_threshold
    
    async def _check_multi_timeframe_confirmation(self, symbol: str, pattern_type: str) -> bool:
        """Check for multi-timeframe volume confirmation (placeholder for future implementation)"""
        try:
            # This is a placeholder for multi-timeframe analysis
            # In a full implementation, this would analyze volume patterns across multiple timeframes
            # For now, return True to not block patterns
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking multi-timeframe confirmation: {e}")
            return True
    
    async def _enable_timescaledb_compression(self, symbol: str, timeframe: str):
        """Enable TimescaleDB compression for performance optimization (Phase 6)"""
        try:
            if not self.streaming_config['enable_compression']:
                return
            
            async with self.db_pool.acquire() as conn:
                # Enable compression on volume_analysis table
                await conn.execute("""
                    ALTER TABLE volume_analysis SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,timeframe',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """)
                
                # Set compression policy
                await conn.execute(f"""
                    SELECT add_compression_policy('volume_analysis', INTERVAL '{self.streaming_config["compression_after_hours"]} hours');
                """)
                
                self.logger.info(f"✅ TimescaleDB compression enabled for {symbol} {timeframe}")
                
        except Exception as e:
            self.logger.error(f"Error enabling TimescaleDB compression: {e}")
    
    async def _create_ml_training_dataset(self, symbol: str, timeframe: str, result: VolumeAnalysisResult):
        """Create ML training dataset entry (Phase 6)"""
        try:
            # Store comprehensive data for ML training
            ml_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': result.timestamp.isoformat(),
                'features': {
                    'volume_ratio': result.volume_ratio,
                    'volume_trend': result.volume_trend,
                    'volume_positioning_score': result.volume_positioning_score,
                    'order_book_imbalance': result.order_book_imbalance,
                    'vwap': result.vwap,
                    'cumulative_volume_delta': result.cumulative_volume_delta,
                    'relative_volume': result.relative_volume,
                    'volume_flow_imbalance': result.volume_flow_imbalance,
                    'liquidity_score': result.liquidity_score,
                    'volatility': await self._calculate_volatility(pd.DataFrame([])),  # Placeholder
                    'pattern_type': result.volume_pattern_type,
                    'pattern_strength': result.volume_pattern_strength,
                    'pattern_confidence': result.volume_pattern_confidence
                },
                'targets': {
                    'volume_breakout': result.volume_breakout,
                    'price_change_1h': 0.0,  # Placeholder for future implementation
                    'price_change_4h': 0.0,  # Placeholder for future implementation
                    'successful_pattern': False  # Placeholder for future implementation
                },
                'metadata': {
                    'analysis_version': 'phase6',
                    'symbol_threshold': await self._get_symbol_specific_threshold(symbol, 2.0),
                    'market_conditions': 'normal'  # Placeholder
                }
            }
            
            # Store in database for ML training
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO volume_analysis_ml_dataset (
                        symbol, timeframe, timestamp, features, targets, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, symbol, timeframe, result.timestamp, ml_data['features'], 
                     ml_data['targets'], ml_data['metadata'])
                
        except Exception as e:
            self.logger.error(f"Error creating ML training dataset: {e}")
    
    async def _store_volume_analysis(self, result: VolumeAnalysisResult):
        """Store volume analysis in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO volume_analysis (
                        symbol, timeframe, timestamp, volume_ratio, volume_trend,
                        volume_positioning_score, order_book_imbalance, buy_volume_ratio,
                        sell_volume_ratio, volume_breakout, vwap, cumulative_volume_delta,
                        relative_volume, volume_weighted_price, volume_flow_imbalance,
                        volume_pattern_type, volume_pattern_strength, volume_pattern_confidence,
                        volume_analysis, volume_context
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                """, result.symbol, result.timeframe, result.timestamp, result.volume_ratio,
                     result.volume_trend, result.volume_positioning_score, result.order_book_imbalance,
                     result.buy_volume_ratio, result.sell_volume_ratio, result.volume_breakout,
                     result.vwap, result.cumulative_volume_delta, result.relative_volume,
                     result.volume_weighted_price, result.volume_flow_imbalance,
                     result.volume_pattern_type, result.volume_pattern_strength,
                     result.volume_pattern_confidence, result.volume_analysis, result.volume_context)
                
        except Exception as e:
            self.logger.error(f"Error storing volume analysis: {e}")
    
    def _create_default_analysis(self, symbol: str, timeframe: str) -> VolumeAnalysisResult:
        """Create default volume analysis when insufficient data"""
        return VolumeAnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(timezone.utc),
            volume_ratio=1.0,
            volume_trend="stable",
            volume_positioning_score=0.5,
            order_book_imbalance=0.0,
            buy_volume_ratio=0.5,
            sell_volume_ratio=0.5,
            volume_breakout=False,
            # Advanced metrics defaults
            vwap=0.0,
            cumulative_volume_delta=0.0,
            relative_volume=1.0,
            volume_weighted_price=0.0,
            volume_flow_imbalance=0.0,
            # Delta profile metrics defaults
            delta_profile={},
            support_levels=[],
            resistance_levels=[],
            volume_nodes=[],
            # Liquidity metrics defaults
            liquidity_score=0.5,
            spoofing_detected=False,
            whale_activity=False,
            volume_analysis="Insufficient data for analysis"
        )
    
    async def _generate_ml_features_and_predict(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]):
        """Generate ML features and make predictions (Phase 7)"""
        try:
            # Import ML services
            from .ml_feature_engineering_service import MLFeatureEngineeringService
            from .ml_prediction_service import MLPredictionService
            
            # Initialize ML services
            feature_service = MLFeatureEngineeringService(self.db_pool)
            prediction_service = MLPredictionService(self.db_pool)
            
            # Generate comprehensive ML features
            ml_features = await feature_service.generate_comprehensive_features(symbol, timeframe, ohlcv_data)
            if ml_features:
                # Store ML features
                await feature_service.store_ml_features(symbol, timeframe, datetime.now(), ml_features)
                
                # Make ML prediction
                prediction = await prediction_service.predict(symbol, timeframe, ohlcv_data)
                if prediction:
                    self.logger.info(f"🤖 ML Prediction for {symbol}: {prediction.prediction_type.value} = {prediction.prediction_value:.4f} (confidence: {prediction.confidence_score:.2f})")
                    
                    # Store prediction in volume context for integration
                    if not hasattr(self, '_volume_context'):
                        self._volume_context = {}
                    
                    self._volume_context['ml_prediction'] = {
                        'prediction_type': prediction.prediction_type.value,
                        'prediction_value': prediction.prediction_value,
                        'confidence_score': prediction.confidence_score,
                        'model_version': prediction.model_version,
                        'feature_contributions': prediction.feature_contributions
                    }
                    
        except Exception as e:
            self.logger.error(f"❌ Error in ML features and prediction: {e}")

    async def _integrate_phase8_features(self, symbol: str, timeframe: str, ohlcv_data: List[Dict], volume_result: VolumeAnalysisResult):
        """Integrate Phase 8 Advanced ML Features (anomaly detection, RL, trading integration)"""
        try:
            # Import Phase 8 services
            from .anomaly_detection_service import AnomalyDetectionService
            from .reinforcement_learning_service import ReinforcementLearningService
            from .trading_system_integration_service import TradingSystemIntegrationService
            
            # Initialize Phase 8 services
            anomaly_service = AnomalyDetectionService(self.db_pool)
            rl_service = ReinforcementLearningService(self.db_pool)
            trading_service = TradingSystemIntegrationService(self.db_pool)
            
            # 1. Anomaly Detection
            anomalies = []
            if self.phase8_config['enable_anomaly_detection']:
                anomalies = await anomaly_service.detect_anomalies(symbol, timeframe, ohlcv_data)
                if anomalies:
                    self.logger.info(f"🔍 Detected {len(anomalies)} anomalies for {symbol}")
                    
                    # Store anomaly info in volume context
                    if not hasattr(self, '_volume_context'):
                        self._volume_context = {}
                    
                    self._volume_context['anomalies'] = [
                        {
                            'anomaly_type': anomaly.anomaly_type.value,
                            'anomaly_score': anomaly.anomaly_score,
                            'confidence_score': anomaly.confidence_score,
                            'severity_level': anomaly.severity_level.value,
                            'detection_method': anomaly.detection_method.value
                        }
                        for anomaly in anomalies
                    ]
            
            # 2. Reinforcement Learning
            rl_action = None
            if self.phase8_config['enable_reinforcement_learning']:
                # Initialize RL agent if not exists
                agent_id = f"{symbol}_{timeframe}_agent"
                if not hasattr(self, '_rl_agents'):
                    self._rl_agents = {}
                
                if agent_id not in self._rl_agents:
                    self._rl_agents[agent_id] = await rl_service.initialize_agent(symbol, timeframe)
                
                if self._rl_agents[agent_id]:
                    # Get state features
                    state_features = await rl_service.get_state_features(symbol, timeframe, ohlcv_data)
                    
                    # Choose action
                    action = await rl_service.choose_action(self._rl_agents[agent_id], state_features)
                    
                    # Calculate reward (simplified - would need actual price movement)
                    current_price = ohlcv_data[-1]['close'] if ohlcv_data else 0
                    next_price = current_price  # In real implementation, this would be the next price
                    reward = await rl_service.calculate_reward(action, current_price, next_price)
                    
                    # Update agent
                    await rl_service.update_agent(self._rl_agents[agent_id], action, reward, state_features)
                    
                    rl_action = {
                        'action_type': action.action_type.value,
                        'confidence': action.confidence,
                        'current_price': current_price,
                        'agent_id': self._rl_agents[agent_id]
                    }
                    
                    self.logger.info(f"🤖 RL Agent {agent_id} chose {action.action_type.value} with confidence {action.confidence:.2f}")
            
            # 3. Trading System Integration
            if self.phase8_config['enable_trading_integration']:
                # Prepare volume analysis data for trading signals
                volume_analysis_data = {
                    'volume_ratio': volume_result.volume_ratio,
                    'volume_positioning_score': volume_result.volume_positioning_score,
                    'volume_pattern_type': volume_result.volume_pattern_type,
                    'volume_breakout': volume_result.volume_breakout,
                    'close': ohlcv_data[-1]['close'] if ohlcv_data else 0
                }
                
                # Get ML prediction if available
                ml_prediction = None
                if hasattr(self, '_volume_context') and 'ml_prediction' in self._volume_context:
                    ml_prediction = self._volume_context['ml_prediction']
                    ml_prediction['current_price'] = ohlcv_data[-1]['close'] if ohlcv_data else 0
                
                # Generate trading signals
                trading_signals = await trading_service.generate_trading_signals(
                    symbol, timeframe, volume_analysis_data, ml_prediction, rl_action, 
                    [{'anomaly_type': a.anomaly_type.value, 'anomaly_score': a.anomaly_score, 'severity_level': a.severity_level.value} for a in anomalies] if anomalies else None
                )
                
                if trading_signals:
                    self.logger.info(f"🎯 Generated {len(trading_signals)} trading signals for {symbol}")
                    
                    # Position optimization
                    current_position = {
                        'position_size': 0.0,
                        'stop_loss_price': 0.0,
                        'take_profit_price': 0.0,
                        'current_price': ohlcv_data[-1]['close'] if ohlcv_data else 0
                    }
                    
                    # Calculate volatility for optimization
                    if len(ohlcv_data) > 20:
                        prices = [d['close'] for d in ohlcv_data[-20:]]
                        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                        volatility = np.std(returns) if returns else 0.02
                    else:
                        volatility = 0.02
                    
                    position_optimization = await trading_service.optimize_position_parameters(
                        symbol, timeframe, current_position, trading_signals, volume_analysis_data, volatility
                    )
                    
                    if position_optimization:
                        self.logger.info(f"🔧 Position optimization for {symbol}: recommended size {position_optimization.recommended_position_size:.4f}")
                    
                    # Generate alerts
                    alerts = await trading_service.generate_alerts(
                        symbol, timeframe, trading_signals, 
                        [{'anomaly_type': a.anomaly_type.value, 'anomaly_score': a.anomaly_score, 'severity_level': a.severity_level.value} for a in anomalies] if anomalies else None,
                        volume_analysis_data
                    )
                    
                    if alerts:
                        self.logger.info(f"🚨 Generated {len(alerts)} alerts for {symbol}")
                        
                        # Log high priority alerts
                        high_priority_alerts = [a for a in alerts if a.priority_level.value in ['high', 'critical']]
                        for alert in high_priority_alerts:
                            self.logger.warning(f"🚨 {alert.priority_level.value.upper()} ALERT: {alert.alert_message}")
            
            self.logger.info(f"✅ Phase 8 features integrated for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error integrating Phase 8 features: {e}")
