#!/usr/bin/env python3
"""
Candlestick Pattern Detector for AlphaPulse
Identifies common candlestick patterns for trading signals
Enhanced with noise filtering and adaptive learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from dataclasses import dataclass
import asyncio

# Try to import TA-Lib, fallback to basic implementations if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TA-Lib successfully imported for pattern detection")
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available, using basic pattern detection implementations")

# Import noise filtering and adaptive learning components
try:
    from ..src.ai.noise_filter_engine import NoiseFilterEngine
    from ..src.ai.market_regime_classifier import MarketRegimeClassifier
    from ..src.ai.adaptive_learning_engine import AdaptiveLearningEngine
    ADVANCED_FEATURES_AVAILABLE = True
    logger.info("Advanced pattern features (noise filtering, adaptive learning) available")
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.warning("Advanced pattern features not available, using basic pattern detection")

@dataclass
class PatternSignal:
    """Represents a detected candlestick pattern signal"""
    pattern: str
    index: int
    strength: float
    type: str  # 'bullish', 'bearish', or 'neutral'
    confidence: float
    timestamp: Optional[str] = None
    additional_info: Optional[Dict] = None

class CandlestickPatternDetector:
    """Detects candlestick patterns using TA-Lib or fallback implementations
    Enhanced with noise filtering and adaptive learning"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        """Initialize pattern detector with optional advanced features"""
        # Database configuration for advanced features
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'user': 'postgres',
            'password': 'Emon_@17711'
        }
        
        # Initialize advanced components if available
        self.noise_filter = None
        self.market_regime_classifier = None
        self.adaptive_learning = None
        self.advanced_features_enabled = False
        
        if ADVANCED_FEATURES_AVAILABLE:
            self.advanced_features_enabled = True
            logger.info("Advanced pattern features enabled")
        
        # Initialize advanced components
        if self.advanced_features_enabled:
            self._initialize_advanced_components()
        # Define pattern functions with TA-Lib
        if TALIB_AVAILABLE:
            self.patterns = {
                'hammer': talib.CDLHAMMER,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'engulfing': talib.CDLENGULFING,
                'doji': talib.CDLDOJI,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR,
                'three_white_soldiers': talib.CDL3WHITESOLDIERS,
                'three_black_crows': talib.CDL3BLACKCROWS,
                'hanging_man': talib.CDLHANGINGMAN,
                'inverted_hammer': talib.CDLINVERTEDHAMMER,
                'spinning_top': talib.CDLSPINNINGTOP,
                'marubozu': talib.CDLMARUBOZU,
                'tristar': talib.CDLTRISTAR,
                'three_inside_up': talib.CDL3INSIDE,
                'three_inside_down': talib.CDL3INSIDE,
                'three_outside_up': talib.CDL3OUTSIDE,
                'three_outside_down': talib.CDL3OUTSIDE,
                'breakaway': talib.CDLBREAKAWAY,
                'dark_cloud_cover': talib.CDLDARKCLOUDCOVER,
                'dragonfly_doji': talib.CDLDRAGONFLYDOJI,
                'gravestone_doji': talib.CDLGRAVESTONEDOJI,
                'harami': talib.CDLHARAMI,
                'harami_cross': talib.CDLHARAMICROSS,
                'high_wave': talib.CDLHIGHWAVE,
                'identical_three_crows': talib.CDLIDENTICAL3CROWS,
                'kicking': talib.CDLKICKING,
                'ladder_bottom': talib.CDLLADDERBOTTOM,
                'long_legged_doji': talib.CDLLONGLEGGEDDOJI,
                'long_line': talib.CDLLONGLINE,
                'meeting_lines': talib.CDLSEPARATINGLINES,  # Using similar pattern as fallback
                'on_neck': talib.CDLONNECK,
                'piercing': talib.CDLPIERCING,
                'rising_falling_three_methods': talib.CDLRISEFALL3METHODS,
                'separating_lines': talib.CDLSEPARATINGLINES,
                'short_line': talib.CDLSHORTLINE,
                'takuri': talib.CDLTAKURI,
                'thrusting': talib.CDLTHRUSTING,
                'unique_three_rivers': talib.CDLUNIQUE3RIVER,
                'upside_gap_two_crows': talib.CDLUPSIDEGAP2CROWS
            }
        else:
            # Basic patterns without TA-Lib
            self.patterns = {
                'hammer': self._detect_hammer_basic,
                'shooting_star': self._detect_shooting_star_basic,
                'engulfing': self._detect_engulfing_basic,
                'doji': self._detect_doji_basic,
                'morning_star': self._detect_morning_star_basic,
                'evening_star': self._detect_evening_star_basic,
                'three_white_soldiers': self._detect_three_white_soldiers_basic,
                'three_black_crows': self._detect_three_black_crows_basic
            }
        
        # Pattern metadata
        self.pattern_metadata = {
            'hammer': {'type': 'bullish', 'reliability': 0.7, 'description': 'Potential reversal pattern'},
            'shooting_star': {'type': 'bearish', 'reliability': 0.7, 'description': 'Potential reversal pattern'},
            'engulfing': {'type': 'both', 'reliability': 0.8, 'description': 'Strong reversal signal'},
            'doji': {'type': 'neutral', 'reliability': 0.6, 'description': 'Indecision in market'},
            'morning_star': {'type': 'bullish', 'reliability': 0.8, 'description': 'Strong bullish reversal'},
            'evening_star': {'type': 'bearish', 'reliability': 0.8, 'description': 'Strong bearish reversal'},
            'three_white_soldiers': {'type': 'bullish', 'reliability': 0.7, 'description': 'Bullish continuation'},
            'three_black_crows': {'type': 'bearish', 'reliability': 0.7, 'description': 'Bearish continuation'}
        }
        
        logger.info(f"Pattern detector initialized with {len(self.patterns)} patterns. TA-Lib: {TALIB_AVAILABLE}")
    
    async def _initialize_advanced_components(self):
        """Initialize advanced pattern detection components"""
        try:
            if not self.advanced_features_enabled:
                return
            
            # Initialize noise filter engine
            self.noise_filter = NoiseFilterEngine(self.db_config)
            await self.noise_filter.initialize()
            
            # Initialize market regime classifier
            self.market_regime_classifier = MarketRegimeClassifier(self.db_config)
            await self.market_regime_classifier.initialize()
            
            # Initialize adaptive learning engine
            self.adaptive_learning = AdaptiveLearningEngine(self.db_config)
            await self.adaptive_learning.initialize()
            
            logger.info("✅ Advanced pattern detection components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize advanced components: {e}")
            self.advanced_features_enabled = False
    
    def detect_patterns(self, opens: np.ndarray, highs: np.ndarray, 
                       lows: np.ndarray, closes: np.ndarray, 
                       volumes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Detect all candlestick patterns using TA-Lib or fallback implementations
        
        Args:
            opens: Array of opening prices
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            volumes: Optional array of volumes for volume confirmation
            
        Returns:
            Dictionary with pattern names as keys and detection results as values
        """
        if len(opens) < 5:
            logger.warning("Insufficient data for pattern detection (minimum 5 candles required)")
            return {}
        
        results = {}
        
        if TALIB_AVAILABLE:
            # Use TA-Lib for pattern detection
            # Convert to float64 for TA-Lib compatibility
            opens_f64 = opens.astype(np.float64)
            highs_f64 = highs.astype(np.float64)
            lows_f64 = lows.astype(np.float64)
            closes_f64 = closes.astype(np.float64)
            
            for pattern_name, pattern_func in self.patterns.items():
                try:
                    result = pattern_func(opens_f64, highs_f64, lows_f64, closes_f64)
                    results[pattern_name] = result
                except Exception as e:
                    logger.error(f"Error detecting {pattern_name} with TA-Lib: {e}")
                    results[pattern_name] = np.zeros_like(closes)
        else:
            # Use basic implementations
            for pattern_name, pattern_func in self.patterns.items():
                try:
                    result = pattern_func(opens, highs, lows, closes, volumes)
                    results[pattern_name] = result
                except Exception as e:
                    logger.error(f"Error detecting {pattern_name} with basic implementation: {e}")
                    results[pattern_name] = np.zeros_like(closes)
        
        logger.info(f"Detected patterns: {list(results.keys())}")
        return results
    
    def get_pattern_signals(self, pattern_results: Dict[str, np.ndarray], 
                           timestamps: Optional[List] = None) -> List[PatternSignal]:
        """
        Convert pattern results to trading signals
        
        Args:
            pattern_results: Dictionary of pattern detection results
            timestamps: Optional list of timestamps for the signals
            
        Returns:
            List of PatternSignal objects
        """
        signals = []
        
        for pattern_name, values in pattern_results.items():
            if pattern_name not in self.pattern_metadata:
                continue
                
            metadata = self.pattern_metadata[pattern_name]
            
            # Find where patterns occur (non-zero values)
            pattern_indices = np.where(values != 0)[0]
            
            for idx in pattern_indices:
                # Determine signal type based on pattern and value
                if metadata['type'] == 'both':
                    signal_type = 'bullish' if values[idx] > 0 else 'bearish'
                else:
                    signal_type = metadata['type']
                
                # Calculate confidence based on pattern reliability and volume confirmation
                confidence = metadata['reliability']
                
                # Create pattern signal
                signal = PatternSignal(
                    pattern=pattern_name,
                    index=idx,
                    strength=abs(values[idx]) if TALIB_AVAILABLE else 1.0,
                    type=signal_type,
                    confidence=confidence,
                    timestamp=timestamps[idx] if timestamps and idx < len(timestamps) else None,
                    additional_info={
                        'description': metadata['description'],
                        'reliability': metadata['reliability']
                    }
                )
                signals.append(signal)
        
        # Sort signals by confidence (highest first)
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Generated {len(signals)} pattern signals")
        return signals
    
    def detect_patterns_from_dataframe(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        Detect patterns from a pandas DataFrame with OHLCV data
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            List of PatternSignal objects
        """
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return []
        
        # Extract OHLCV data
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        timestamps = df.index.tolist() if hasattr(df.index, 'tolist') else None
        
        # Detect patterns
        pattern_results = self.detect_patterns(opens, highs, lows, closes, volumes)
        
        # Generate signals
        signals = self.get_pattern_signals(pattern_results, timestamps)
        
        return signals
    
    async def detect_patterns_enhanced(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[PatternSignal]:
        """
        Enhanced pattern detection with noise filtering and adaptive learning
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            List of enhanced PatternSignal objects
        """
        try:
            if not self.advanced_features_enabled:
                logger.warning("Advanced features not enabled, using basic pattern detection")
                return self.detect_patterns_from_dataframe(df)
            
            # Ensure advanced components are initialized
            if not self.noise_filter or not self.market_regime_classifier or not self.adaptive_learning:
                await self._initialize_advanced_components()
            
            # Step 1: Classify market regime
            regime_result = await self.market_regime_classifier.classify_market_regime(df, symbol, timeframe)
            
            # Step 2: Detect basic patterns
            basic_signals = self.detect_patterns_from_dataframe(df)
            
            # Step 3: Apply noise filtering and adaptive learning
            enhanced_signals = []
            
            for signal in basic_signals:
                # Create pattern data for filtering
                pattern_data = {
                    'pattern_id': f"{symbol}_{signal.pattern}_{signal.index}",
                    'tracking_id': f"track_{symbol}_{signal.pattern}_{int(pd.Timestamp.now().timestamp())}",
                    'symbol': symbol,
                    'pattern_name': signal.pattern,
                    'timeframe': timeframe,
                    'pattern_confidence': signal.confidence,
                    'market_regime': regime_result['regime_type'],
                    'confidence': signal.confidence
                }
                
                # Apply noise filtering
                passed_filter, filter_results = await self.noise_filter.filter_pattern(pattern_data, df)
                
                if passed_filter:
                    # Apply adaptive learning for confidence adjustment
                    adaptive_confidence = await self.adaptive_learning.get_adaptive_confidence(
                        pattern_data, signal.confidence
                    )
                    
                    # Create enhanced signal
                    enhanced_signal = PatternSignal(
                        pattern=signal.pattern,
                        index=signal.index,
                        strength=signal.strength,
                        type=signal.type,
                        confidence=adaptive_confidence,
                        timestamp=signal.timestamp,
                        additional_info={
                            **signal.additional_info,
                            'market_regime': regime_result['regime_type'],
                            'regime_confidence': regime_result['regime_confidence'],
                            'noise_filter_score': filter_results['overall_score'],
                            'noise_level': filter_results['noise_level'],
                            'filter_reasons': filter_results['filter_reasons'],
                            'adaptive_confidence': adaptive_confidence,
                            'base_confidence': signal.confidence,
                            'tracking_id': pattern_data['tracking_id']
                        }
                    )
                    
                    enhanced_signals.append(enhanced_signal)
                    
                    # Store pattern for performance tracking
                    await self._store_pattern_for_tracking(pattern_data, filter_results, regime_result)
                else:
                    logger.info(f"Pattern {signal.pattern} filtered out: {filter_results['filter_reasons']}")
            
            # Sort by enhanced confidence
            enhanced_signals.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"Enhanced pattern detection: {len(enhanced_signals)} patterns passed filtering out of {len(basic_signals)} detected")
            return enhanced_signals
            
        except Exception as e:
            logger.error(f"❌ Enhanced pattern detection failed: {e}")
            # Fallback to basic detection
            return self.detect_patterns_from_dataframe(df)
    
    async def _store_pattern_for_tracking(self, pattern_data: Dict[str, Any], 
                                        filter_results: Dict[str, Any], 
                                        regime_result: Dict[str, Any]):
        """Store pattern data for performance tracking"""
        try:
            if not self.adaptive_learning:
                return
            
            # This will be used by the adaptive learning engine to track outcomes
            # The actual storage is handled by the noise filter engine
            logger.debug(f"Pattern stored for tracking: {pattern_data['pattern_id']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store pattern for tracking: {e}")
    
    def get_pattern_summary(self, signals: List[PatternSignal]) -> Dict:
        """
        Get summary statistics for detected patterns
        
        Args:
            signals: List of PatternSignal objects
            
        Returns:
            Dictionary with pattern summary statistics
        """
        if not signals:
            return {}
        
        summary = {
            'total_signals': len(signals),
            'bullish_signals': len([s for s in signals if s.type == 'bullish']),
            'bearish_signals': len([s for s in signals if s.type == 'bearish']),
            'neutral_signals': len([s for s in signals if s.type == 'neutral']),
            'patterns_detected': {},
            'average_confidence': np.mean([s.confidence for s in signals]),
            'strongest_patterns': []
        }
        
        # Count patterns by type
        for signal in signals:
            if signal.pattern not in summary['patterns_detected']:
                summary['patterns_detected'][signal.pattern] = 0
            summary['patterns_detected'][signal.pattern] += 1
        
        # Get strongest patterns (highest confidence)
        strong_signals = sorted(signals, key=lambda x: x.confidence, reverse=True)[:5]
        summary['strongest_patterns'] = [
            {
                'pattern': s.pattern,
                'type': s.type,
                'confidence': s.confidence,
                'index': s.index
            }
            for s in strong_signals
        ]
        
        return summary
    
    def detect_single_pattern(self, df: pd.DataFrame, pattern_name: str) -> Optional[Dict]:
        """
        Detect a single specific pattern from DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            pattern_name: Name of the pattern to detect
            
        Returns:
            Dictionary with detection result or None if not detected
        """
        if pattern_name not in self.patterns:
            logger.warning(f"Pattern '{pattern_name}' not supported")
            return None
        
        # Extract OHLCV data
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        if len(opens) < 5:
            return None
        
        try:
            if TALIB_AVAILABLE:
                # Use TA-Lib
                pattern_func = self.patterns[pattern_name]
                result = pattern_func(opens, highs, lows, closes)
                
                # Check if pattern detected at the last candle
                if result[-1] != 0:
                    metadata = self.pattern_metadata.get(pattern_name, {})
                    signal_type = metadata.get('type', 'neutral')
                    
                    if signal_type == 'both':
                        signal_type = 'bullish' if result[-1] > 0 else 'bearish'
                    
                    return {
                        'detected': True,
                        'type': signal_type,
                        'strength': abs(result[-1]),
                        'metadata': metadata
                    }
            else:
                # Use basic implementation
                pattern_func = self.patterns[pattern_name]
                result = pattern_func(opens, highs, lows, closes, volumes)
                
                # Check if pattern detected at the last candle
                if result[-1] != 0:
                    metadata = self.pattern_metadata.get(pattern_name, {})
                    signal_type = metadata.get('type', 'neutral')
                    
                    if signal_type == 'both':
                        signal_type = 'bullish' if result[-1] > 0 else 'bearish'
                    
                    return {
                        'detected': True,
                        'type': signal_type,
                        'strength': 1.0,
                        'metadata': metadata
                    }
                    
        except Exception as e:
            logger.error(f"Error detecting {pattern_name}: {e}")
        
        return None
    
    # Basic pattern detection implementations (fallback when TA-Lib is not available)
    
    def _detect_hammer_basic(self, opens: np.ndarray, highs: np.ndarray, 
                            lows: np.ndarray, closes: np.ndarray, 
                            volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic hammer pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(1, len(opens)):
            body = abs(closes[i] - opens[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            
            # Hammer criteria: small body, long lower shadow, small upper shadow
            if (lower_shadow > 2 * body and 
                upper_shadow < body and 
                body > 0):
                result[i] = 1  # Bullish hammer
        
        return result
    
    def _detect_shooting_star_basic(self, opens: np.ndarray, highs: np.ndarray, 
                                   lows: np.ndarray, closes: np.ndarray, 
                                   volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic shooting star pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(1, len(opens)):
            body = abs(closes[i] - opens[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            
            # Shooting star criteria: small body, long upper shadow, small lower shadow
            if (upper_shadow > 2 * body and 
                lower_shadow < body and 
                body > 0):
                result[i] = -1  # Bearish shooting star
        
        return result
    
    def _detect_engulfing_basic(self, opens: np.ndarray, highs: np.ndarray, 
                               lows: np.ndarray, closes: np.ndarray, 
                               volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic engulfing pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(1, len(opens)):
            prev_body = abs(closes[i-1] - opens[i-1])
            curr_body = abs(closes[i] - opens[i])
            
            # Bullish engulfing: current green candle completely engulfs previous red candle
            if (closes[i] > opens[i] and  # Current candle is green
                closes[i-1] < opens[i-1] and  # Previous candle is red
                opens[i] < closes[i-1] and  # Current open below previous close
                closes[i] > opens[i-1] and  # Current close above previous open
                curr_body > prev_body):  # Current body larger than previous
                result[i] = 1  # Bullish engulfing
            
            # Bearish engulfing: current red candle completely engulfs previous green candle
            elif (closes[i] < opens[i] and  # Current candle is red
                  closes[i-1] > opens[i-1] and  # Previous candle is green
                  opens[i] > closes[i-1] and  # Current open above previous close
                  closes[i] < opens[i-1] and  # Current close below previous open
                  curr_body > prev_body):  # Current body larger than previous
                result[i] = -1  # Bearish engulfing
        
        return result
    
    def _detect_doji_basic(self, opens: np.ndarray, highs: np.ndarray, 
                          lows: np.ndarray, closes: np.ndarray, 
                          volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic doji pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(len(opens)):
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]
            
            # Doji criteria: very small body relative to total range
            if total_range > 0 and body / total_range < 0.1:
                result[i] = 1  # Neutral doji
        
        return result
    
    def _detect_morning_star_basic(self, opens: np.ndarray, highs: np.ndarray, 
                                  lows: np.ndarray, closes: np.ndarray, 
                                  volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic morning star pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(2, len(opens)):
            # Morning star: red candle, small body (doji-like), green candle
            if (closes[i-2] < opens[i-2] and  # First candle is red
                abs(closes[i-1] - opens[i-1]) < abs(highs[i-1] - lows[i-1]) * 0.3 and  # Middle candle small
                closes[i] > opens[i] and  # Third candle is green
                closes[i] > (opens[i-2] + closes[i-2]) / 2):  # Third candle closes above midpoint of first
                result[i] = 1  # Bullish morning star
        
        return result
    
    def _detect_evening_star_basic(self, opens: np.ndarray, highs: np.ndarray, 
                                  lows: np.ndarray, closes: np.ndarray, 
                                  volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic evening star pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(2, len(opens)):
            # Evening star: green candle, small body (doji-like), red candle
            if (closes[i-2] > opens[i-2] and  # First candle is green
                abs(closes[i-1] - opens[i-1]) < abs(highs[i-1] - lows[i-1]) * 0.3 and  # Middle candle small
                closes[i] < opens[i] and  # Third candle is red
                closes[i] < (opens[i-2] + closes[i-2]) / 2):  # Third candle closes below midpoint of first
                result[i] = -1  # Bearish evening star
        
        return result
    
    def _detect_three_white_soldiers_basic(self, opens: np.ndarray, highs: np.ndarray, 
                                          lows: np.ndarray, closes: np.ndarray, 
                                          volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic three white soldiers pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(2, len(opens)):
            # Three white soldiers: three consecutive green candles with higher closes
            if (closes[i-2] > opens[i-2] and  # First candle is green
                closes[i-1] > opens[i-1] and  # Second candle is green
                closes[i] > opens[i] and  # Third candle is green
                closes[i-1] > closes[i-2] and  # Second close higher than first
                closes[i] > closes[i-1]):  # Third close higher than second
                result[i] = 1  # Bullish three white soldiers
        
        return result
    
    def _detect_three_black_crows_basic(self, opens: np.ndarray, highs: np.ndarray, 
                                       lows: np.ndarray, closes: np.ndarray, 
                                       volumes: Optional[np.ndarray] = None) -> np.ndarray:
        """Basic three black crows pattern detection"""
        result = np.zeros_like(opens)
        
        for i in range(2, len(opens)):
            # Three black crows: three consecutive red candles with lower closes
            if (closes[i-2] < opens[i-2] and  # First candle is red
                closes[i-1] < opens[i-1] and  # Second candle is red
                closes[i] < opens[i] and  # Third candle is red
                closes[i-1] < closes[i-2] and  # Second close lower than first
                closes[i] < closes[i-1]):  # Third close lower than second
                result[i] = -1  # Bearish three black crows
        
        return result
    
    async def cleanup(self):
        """Cleanup advanced components"""
        try:
            if self.noise_filter:
                await self.noise_filter.cleanup()
            if self.market_regime_classifier:
                await self.market_regime_classifier.cleanup()
            if self.adaptive_learning:
                await self.adaptive_learning.cleanup()
            logger.info("✅ Pattern detector cleanup completed")
        except Exception as e:
            logger.error(f"❌ Pattern detector cleanup failed: {e}")

# Example usage
if __name__ == "__main__":
    # Test the pattern detector
    import numpy as np
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n = 100
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    highs = closes + np.random.rand(n) * 2
    lows = closes - np.random.rand(n) * 2
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = np.random.randint(1000, 10000, n)
    
    # Create some pattern-like data
    # Add a hammer pattern
    opens[50] = closes[50] - 0.5
    lows[50] = closes[50] - 2.0
    highs[50] = closes[50] + 0.1
    
    # Add a doji pattern
    opens[75] = closes[75] + 0.05
    highs[75] = closes[75] + 1.0
    lows[75] = closes[75] - 1.0
    
    # Initialize pattern detector
    detector = CandlestickPatternDetector()
    
    # Detect patterns
    pattern_results = detector.detect_patterns(opens, highs, lows, closes, volumes)
    
    # Generate signals
    signals = detector.get_pattern_signals(pattern_results)
    
    # Get summary
    summary = detector.get_pattern_summary(signals)
    
    print("Pattern Detection Results:")
    print(f"Total signals: {summary['total_signals']}")
    print(f"Bullish signals: {summary['bullish_signals']}")
    print(f"Bearish signals: {summary['bearish_signals']}")
    print(f"Patterns detected: {summary['patterns_detected']}")
    
    if signals:
        print("\nStrongest patterns:")
        for pattern in summary['strongest_patterns'][:3]:
            print(f"  {pattern['pattern']}: {pattern['type']} (confidence: {pattern['confidence']:.2f})")
