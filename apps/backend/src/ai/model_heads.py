"""
AI Model Heads Implementation for AlphaPlus SDE Framework
Simplified implementation of the 4 model heads for decision making
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class SignalDirection(Enum):
    """Signal direction enumeration"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class ModelHead(Enum):
    """Model head types"""
    HEAD_A = "head_a"  # Technical Analysis
    HEAD_B = "head_b"  # Sentiment Analysis
    HEAD_C = "head_c"  # Volume/Orderbook Analysis
    HEAD_D = "head_d"  # Rule-based Analysis
    ICT_CONCEPTS = "ict_concepts"  # ICT Concepts Analysis
    WYCKOFF = "wyckoff"  # Wyckoff Methodology
    HARMONIC = "harmonic"  # Harmonic Patterns
    MARKET_STRUCTURE = "market_structure"  # Enhanced Market Structure
    CRYPTO_METRICS = "crypto_metrics"  # Crypto-Specific Metrics

@dataclass
class ModelHeadResult:
    """Result from a single model head"""
    head_type: ModelHead
    direction: SignalDirection
    probability: float
    confidence: float
    features_used: List[str]
    reasoning: str

class TechnicalAnalysisHead:
    """Head A: Technical Analysis Model - Enhanced with 50+ Indicator Aggregation"""
    
    def __init__(self):
        self.name = "Technical Analysis Head"
        self.features = ['trend_score', 'momentum_score', 'volatility_score', 'technical_score']
        self.aggregator = None  # Lazy initialization
    
    async def _initialize_aggregator(self):
        """Lazy initialization of Technical Indicator Aggregator"""
        if self.aggregator is None:
            try:
                from .indicator_aggregator import TechnicalIndicatorAggregator
                self.aggregator = TechnicalIndicatorAggregator()
                logger.info("✅ Technical Indicator Aggregator initialized in Head A")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Technical Indicator Aggregator: {e}")
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform comprehensive technical analysis using 50+ indicators"""
        try:
            # Initialize aggregator if needed
            await self._initialize_aggregator()
            
            if self.aggregator is None:
                # Fallback to basic analysis
                return await self._basic_analysis(market_data, analysis_results)
            
            # Get dataframe
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 50:
                return await self._basic_analysis(market_data, analysis_results)
            
            # Get indicators
            indicators = market_data.get('indicators', {})
            
            # Aggregate all technical indicators
            agg_result = await self.aggregator.aggregate_technical_signals(df, indicators)
            
            # Convert technical score to direction
            if agg_result.direction == "bullish":
                direction = SignalDirection.LONG
                probability = agg_result.technical_score
            elif agg_result.direction == "bearish":
                direction = SignalDirection.SHORT
                probability = 1.0 - agg_result.technical_score
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
            
            # Use aggregated confidence
            confidence = agg_result.confidence
            
            # Enhanced reasoning
            reasoning = f"Technical Aggregation: {agg_result.reasoning}"
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_A,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=agg_result.contributing_indicators or self.features,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Technical analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HEAD_A,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )
    
    async def _basic_analysis(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Fallback basic analysis if aggregator unavailable"""
        try:
            indicators = market_data.get('indicators', {})
            current_price = market_data.get('current_price', 0)
            
            # Get technical indicators
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            rsi = indicators.get('rsi_14', 50)
            macd = indicators.get('macd', 0)
            
            # Basic analysis logic
            bullish_signals = 0
            bearish_signals = 0
            reasoning = ["Fallback basic analysis"]
            
            if sma_20 > sma_50:
                bullish_signals += 1
                reasoning.append("SMA20>SMA50")
            else:
                bearish_signals += 1
            
            if rsi < 30:
                bullish_signals += 1
            elif rsi > 70:
                bearish_signals += 1
            
            if macd > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                direction = SignalDirection.LONG
                probability = min(0.9, 0.5 + (bullish_signals - bearish_signals) * 0.15)
            elif bearish_signals > bullish_signals:
                direction = SignalDirection.SHORT
                probability = min(0.9, 0.5 + (bearish_signals - bullish_signals) * 0.15)
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
            
            confidence = min(0.95, 0.6 + abs(bullish_signals - bearish_signals) * 0.1)
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_A,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=['sma_20', 'sma_50', 'rsi_14', 'macd'],
                reasoning="; ".join(reasoning)
            )
        except Exception as e:
            logger.error(f"Basic technical analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HEAD_A,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class SentimentAnalysisHead:
    """Head B: Sentiment Analysis Model"""
    
    def __init__(self):
        self.name = "Sentiment Analysis Head"
        self.features = ['news_sentiment', 'social_sentiment', 'market_sentiment']
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform sentiment analysis"""
        try:
            sentiment_data = analysis_results.get('sentiment_analysis', {})
            overall_sentiment = sentiment_data.get('overall_sentiment', 0.0)
            sentiment_confidence = sentiment_data.get('confidence', 0.5)
            
            # Sentiment analysis logic
            reasoning = []
            
            if overall_sentiment > 0.2:
                direction = SignalDirection.LONG
                probability = min(0.9, 0.5 + overall_sentiment * 0.4)
                reasoning.append(f"Positive sentiment: {overall_sentiment:.3f}")
            elif overall_sentiment < -0.2:
                direction = SignalDirection.SHORT
                probability = min(0.9, 0.5 + abs(overall_sentiment) * 0.4)
                reasoning.append(f"Negative sentiment: {overall_sentiment:.3f}")
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
                reasoning.append(f"Neutral sentiment: {overall_sentiment:.3f}")
            
            confidence = sentiment_confidence
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_B,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=self.features,
                reasoning="; ".join(reasoning)
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HEAD_B,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class VolumeAnalysisHead:
    """Head C: Volume/Orderbook Analysis Model - Enhanced with Multi-Indicator Aggregation + CVD Divergences"""
    
    def __init__(self):
        self.name = "Volume Analysis Head"
        self.features = ['cvd', 'cvd_divergences', 'obv', 'vwap', 'chaikin_mf', 'ad_line', 'smart_money_flow', 'volume_profile']
        self.aggregator = None  # Lazy initialization
        self.cvd_analyzer = None  # Lazy initialization
    
    async def _initialize_aggregator(self):
        """Lazy initialization of Volume Indicator Aggregator and CVD Analyzer"""
        if self.aggregator is None:
            try:
                from .volume_aggregator import VolumeIndicatorAggregator
                self.aggregator = VolumeIndicatorAggregator()
                logger.info("✅ Volume Indicator Aggregator initialized in Head C")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Volume Indicator Aggregator: {e}")
        
        # Initialize CVD Analyzer for divergence detection
        if self.cvd_analyzer is None:
            try:
                from ..strategies.cvd_analyzer import CVDAnalyzer
                self.cvd_analyzer = CVDAnalyzer()
                logger.info("✅ CVD Analyzer initialized in Head C for divergence detection")
            except Exception as e:
                logger.error(f"❌ Failed to initialize CVD Analyzer: {e}")
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform comprehensive volume analysis using multiple volume indicators + CVD divergence detection"""
        try:
            # Initialize aggregator and CVD analyzer if needed
            await self._initialize_aggregator()
            
            if self.aggregator is None:
                # Fallback to basic analysis
                return await self._basic_volume_analysis(market_data, analysis_results)
            
            # Get dataframe
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 20:
                return await self._basic_volume_analysis(market_data, analysis_results)
            
            # Get indicators and orderbook data
            indicators = market_data.get('indicators', {})
            orderbook_data = analysis_results.get('orderbook', None)
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # Aggregate all volume indicators
            agg_result = await self.aggregator.aggregate_volume_signals(df, indicators, orderbook_data)
            
            # === CVD DIVERGENCE ANALYSIS (User's Key Requirement) ===
            cvd_divergence_boost = 0.0
            cvd_reasoning = []
            
            if self.cvd_analyzer and len(df) >= 50:
                try:
                    # Analyze CVD divergences
                    cvd_analysis = await self.cvd_analyzer.analyze_cvd(df, symbol, timeframe='1h')
                    
                    if cvd_analysis.overall_confidence > 0.6:
                        # Check for bullish/bearish divergences
                        bullish_divs = [d for d in cvd_analysis.divergences if 'bullish' in d.divergence_type.value]
                        bearish_divs = [d for d in cvd_analysis.divergences if 'bearish' in d.divergence_type.value]
                        
                        if bullish_divs:
                            cvd_divergence_boost = +0.15  # Boost confidence by 15% for bullish divergence
                            cvd_reasoning.append(f"CVD BULLISH divergence detected (conf: {bullish_divs[0].confidence:.2f}) - Big money accumulating")
                        elif bearish_divs:
                            cvd_divergence_boost = -0.15  # Reduce confidence by 15% for bearish divergence
                            cvd_reasoning.append(f"CVD BEARISH divergence detected (conf: {bearish_divs[0].confidence:.2f}) - Big money distributing")
                        
                        # Check CVD trend
                        if cvd_analysis.cvd_trend == 'bullish':
                            cvd_reasoning.append(f"CVD trend: BULLISH (accumulation phase)")
                        elif cvd_analysis.cvd_trend == 'bearish':
                            cvd_reasoning.append(f"CVD trend: BEARISH (distribution phase)")
                        
                        logger.info(f"✅ CVD divergence analysis complete for {symbol}: {len(bullish_divs)} bullish, {len(bearish_divs)} bearish")
                except Exception as e:
                    logger.debug(f"CVD analysis skipped: {e}")
            
            # Convert smart money flow to direction
            if agg_result.smart_money_flow == "accumulating":
                direction = SignalDirection.LONG
                probability = agg_result.volume_score
            elif agg_result.smart_money_flow == "distributing":
                direction = SignalDirection.SHORT
                probability = 1.0 - agg_result.volume_score
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
            
            # Apply CVD divergence boost/penalty
            confidence = agg_result.confidence + cvd_divergence_boost
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
            # Enhanced reasoning with CVD divergences
            reasoning_parts = [f"Volume Aggregation: {agg_result.reasoning}"]
            if cvd_reasoning:
                reasoning_parts.extend(cvd_reasoning)
            reasoning = "; ".join(reasoning_parts)
            
            # Add CVD divergences to features used
            features_used = agg_result.contributing_indicators or self.features
            if cvd_reasoning:
                features_used = ['cvd_divergences'] + features_used
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_C,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=features_used,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Volume analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HEAD_C,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )
    
    async def _basic_volume_analysis(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Fallback basic volume analysis"""
        try:
            volume_data = analysis_results.get('volume_analysis', {})
            volume_trend = volume_data.get('volume_trend', 'stable')
            volume_strength = volume_data.get('volume_strength', 'normal')
            
            reasoning = ["Fallback basic volume analysis"]
            bullish_signals = 0
            bearish_signals = 0
            
            if volume_trend == 'increasing':
                bullish_signals += 1
                reasoning.append("Volume increasing")
            elif volume_trend == 'decreasing':
                bearish_signals += 1
            
            if volume_strength == 'strong':
                bullish_signals += 1
                reasoning.append("Strong volume")
            elif volume_strength == 'weak':
                bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                direction = SignalDirection.LONG
                probability = min(0.9, 0.5 + (bullish_signals - bearish_signals) * 0.2)
            elif bearish_signals > bullish_signals:
                direction = SignalDirection.SHORT
                probability = min(0.9, 0.5 + (bearish_signals - bullish_signals) * 0.2)
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
            
            confidence = min(0.95, 0.6 + abs(bullish_signals - bearish_signals) * 0.15)
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_C,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=['volume', 'volume_trend'],
                reasoning="; ".join(reasoning)
            )
        except Exception as e:
            logger.error(f"Basic volume analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HEAD_C,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class RuleBasedHead:
    """Head D: Rule-based Analysis Model - Enhanced with 60+ Pattern Detection
    
    Scans for:
    - 60+ candlestick patterns (TA-Lib)
    - Chart patterns (head & shoulders, double tops, triangles, flags)
    - Support/resistance levels
    - Volume confirmation
    
    Confidence is high when patterns are textbook-perfect and volume-confirmed.
    """
    
    def __init__(self):
        self.name = "Rule-based Analysis Head"
        self.features = ['candlestick_patterns', 'chart_patterns', 'support_resistance', 'volume_confirmation']
        
        # Lazy initialization of pattern detectors
        self.candlestick_detector = None
        self.chart_pattern_detector = None
        self.sr_analyzer = None
        self._initialized = False
    
    async def _initialize_detectors(self):
        """Lazy initialization of pattern detection components"""
        if not self._initialized:
            try:
                from ..strategies.pattern_detector import CandlestickPatternDetector
                from ..services.advanced_pattern_recognition import AdvancedPatternRecognition
                
                self.candlestick_detector = CandlestickPatternDetector()
                self.chart_pattern_detector = AdvancedPatternRecognition()
                self._initialized = True
                logger.info("✅ Rule-based pattern detectors initialized in Head D")
            except Exception as e:
                logger.error(f"❌ Failed to initialize pattern detectors: {e}")
                self._initialized = False
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform comprehensive rule-based analysis with 60+ patterns"""
        try:
            # Initialize detectors if needed
            await self._initialize_detectors()
            
            # If initialization failed, fallback to basic analysis
            if not self._initialized or self.candlestick_detector is None:
                return await self._basic_analysis(market_data, analysis_results)
            
            # Get dataframe
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 30:
                return await self._basic_analysis(market_data, analysis_results)
            
            current_price = market_data.get('current_price', 0)
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            # STEP 1: Detect 60+ candlestick patterns using TA-Lib
            candlestick_patterns = await self._detect_candlestick_patterns(df)
            
            # STEP 2: Detect chart patterns (head & shoulders, double tops, triangles, flags)
            chart_patterns = await self._detect_chart_patterns(df)
            
            # STEP 3: Find support/resistance levels
            support_levels, resistance_levels = self._find_support_resistance(df, current_price)
            
            # STEP 4: Check if patterns are at key S/R levels
            patterns_at_key_levels = self._check_patterns_at_sr(
                candlestick_patterns,
                chart_patterns,
                support_levels,
                resistance_levels,
                current_price
            )
            
            # STEP 5: Confirm patterns with volume
            volume_confirmed_patterns = self._confirm_with_volume(
                patterns_at_key_levels,
                df
            )
            
            # STEP 6: Calculate final signal
            direction, probability, confidence, reasoning, features = self._calculate_signal(
                volume_confirmed_patterns,
                candlestick_patterns,
                chart_patterns,
                support_levels,
                resistance_levels,
                current_price
            )
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_D,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=features,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Rule-based analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HEAD_D,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )
    
    async def _detect_candlestick_patterns(self, df) -> List[Dict[str, Any]]:
        """Detect 60+ candlestick patterns using TA-Lib"""
        try:
            import numpy as np
            
            patterns_detected = []
            
            if len(df) < 3:
                return patterns_detected
            
            # Convert to numpy arrays for TA-Lib
            opens = np.array(df['open'].values, dtype=float)
            highs = np.array(df['high'].values, dtype=float)
            lows = np.array(df['low'].values, dtype=float)
            closes = np.array(df['close'].values, dtype=float)
            
            # Use the pattern detector to detect patterns
            all_patterns = await self.candlestick_detector.detect_all_patterns(
                opens, highs, lows, closes
            )
            
            # Get the most recent patterns (last 5 candles)
            for pattern_name, pattern_values in all_patterns.items():
                if len(pattern_values) > 0:
                    # Check last 5 candles for patterns
                    recent_signals = pattern_values[-5:]
                    for i, signal in enumerate(recent_signals):
                        if signal != 0:
                            pattern_type = 'bullish' if signal > 0 else 'bearish'
                            patterns_detected.append({
                                'name': pattern_name,
                                'type': pattern_type,
                                'strength': abs(signal) / 100.0,  # TA-Lib returns -100, 0, or 100
                                'confidence': 0.70 + (abs(signal) / 100.0) * 0.20,  # 0.70-0.90
                                'index': len(df) - len(recent_signals) + i
                            })
            
            return patterns_detected
            
        except Exception as e:
            logger.warning(f"Candlestick pattern detection failed: {e}")
            return []
    
    async def _detect_chart_patterns(self, df) -> List[Dict[str, Any]]:
        """Detect chart patterns (H&S, double tops, triangles, flags)"""
        try:
            patterns_detected = []
            
            if len(df) < 30:
                return patterns_detected
            
            # Use chart pattern detector
            current_idx = len(df) - 1
            
            # Check for major chart patterns
            pattern_checks = [
                ('head_shoulders', self.chart_pattern_detector.detect_head_shoulders),
                ('inverse_head_shoulders', self.chart_pattern_detector.detect_inverse_head_shoulders),
                ('double_top', self.chart_pattern_detector.detect_double_top),
                ('double_bottom', self.chart_pattern_detector.detect_double_bottom),
                ('triangle_ascending', self.chart_pattern_detector.detect_triangle_ascending),
                ('triangle_descending', self.chart_pattern_detector.detect_triangle_descending),
                ('flag_bullish', self.chart_pattern_detector.detect_bullish_flag),
                ('flag_bearish', self.chart_pattern_detector.detect_bearish_flag),
            ]
            
            for pattern_name, detect_func in pattern_checks:
                try:
                    if detect_func(df, current_idx):
                        # Determine pattern type
                        if 'bearish' in pattern_name or pattern_name in ['head_shoulders', 'double_top', 'triangle_descending']:
                            pattern_type = 'bearish'
                        else:
                            pattern_type = 'bullish'
                        
                        patterns_detected.append({
                            'name': pattern_name,
                            'type': pattern_type,
                            'strength': 0.85,  # Chart patterns are typically strong
                            'confidence': 0.80,  # High confidence for chart patterns
                            'index': current_idx
                        })
                except Exception as e:
                    logger.debug(f"Pattern check {pattern_name} failed: {e}")
            
            return patterns_detected
            
        except Exception as e:
            logger.warning(f"Chart pattern detection failed: {e}")
            return []
    
    def _find_support_resistance(self, df, current_price: float) -> tuple:
        """Find support and resistance levels"""
        try:
            support_levels = []
            resistance_levels = []
            
            if len(df) < 10:
                return support_levels, resistance_levels
            
            # Use the advanced pattern recognition S/R finder
            support_levels = self.chart_pattern_detector.find_support_levels(df)
            resistance_levels = self.chart_pattern_detector.find_resistance_levels(df)
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.warning(f"S/R detection failed: {e}")
            return [], []
    
    def _check_patterns_at_sr(self, candlestick_patterns, chart_patterns, 
                              support_levels, resistance_levels, current_price: float) -> List[Dict[str, Any]]:
        """Check if detected patterns occur at key support/resistance levels"""
        patterns_at_key_levels = []
        
        # Tolerance for S/R level (1% of price)
        tolerance = current_price * 0.01
        
        all_patterns = candlestick_patterns + chart_patterns
        
        for pattern in all_patterns:
            at_support = any(abs(current_price - level) < tolerance for level in support_levels)
            at_resistance = any(abs(current_price - level) < tolerance for level in resistance_levels)
            
            if at_support or at_resistance:
                pattern['at_key_level'] = True
                pattern['level_type'] = 'support' if at_support else 'resistance'
                patterns_at_key_levels.append(pattern)
        
        return patterns_at_key_levels
    
    def _confirm_with_volume(self, patterns, df) -> List[Dict[str, Any]]:
        """Confirm patterns with volume analysis"""
        confirmed_patterns = []
        
        try:
            if 'volume' not in df.columns or len(df) < 20:
                # Can't confirm without volume, return patterns as-is with lower confidence
                for pattern in patterns:
                    pattern['volume_confirmed'] = False
                    pattern['confidence'] *= 0.9  # Reduce confidence slightly
                return patterns
            
            # Calculate average volume
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            for pattern in patterns:
                # Volume confirmation: current volume > 1.5x average
                if volume_ratio >= 1.5:
                    pattern['volume_confirmed'] = True
                    pattern['volume_ratio'] = volume_ratio
                    confirmed_patterns.append(pattern)
                elif volume_ratio >= 1.2:
                    # Weak volume confirmation
                    pattern['volume_confirmed'] = 'weak'
                    pattern['volume_ratio'] = volume_ratio
                    pattern['confidence'] *= 0.95
                    confirmed_patterns.append(pattern)
                else:
                    # No volume confirmation - reduce confidence
                    pattern['volume_confirmed'] = False
                    pattern['volume_ratio'] = volume_ratio
                    pattern['confidence'] *= 0.85
                    confirmed_patterns.append(pattern)
            
            return confirmed_patterns
            
        except Exception as e:
            logger.warning(f"Volume confirmation failed: {e}")
            return patterns
    
    def _calculate_signal(self, confirmed_patterns, all_candlestick, all_chart,
                         support_levels, resistance_levels, current_price: float):
        """Calculate final signal based on confirmed patterns
        
        Confidence is high when patterns are textbook-perfect and volume-confirmed.
        """
        bullish_signals = 0
        bearish_signals = 0
        confidence_scores = []
        reasoning_parts = []
        features_used = []
        
        # Analyze confirmed patterns (at key S/R with volume)
        for pattern in confirmed_patterns:
            if pattern['type'] == 'bullish':
                bullish_signals += 1
                vol_status = "vol confirmed" if pattern.get('volume_confirmed') == True else "weak vol"
                reasoning_parts.append(
                    f"Bullish {pattern['name']} at {pattern.get('level_type', 'key level')} ({vol_status})"
                )
                confidence_scores.append(pattern['confidence'])
                features_used.append(pattern['name'])
            elif pattern['type'] == 'bearish':
                bearish_signals += 1
                vol_status = "vol confirmed" if pattern.get('volume_confirmed') == True else "weak vol"
                reasoning_parts.append(
                    f"Bearish {pattern['name']} at {pattern.get('level_type', 'key level')} ({vol_status})"
                )
                confidence_scores.append(pattern['confidence'])
                features_used.append(pattern['name'])
        
        # Add S/R context to reasoning
        if support_levels:
            closest_support = min(support_levels, key=lambda x: abs(current_price - x))
            distance_to_support = ((current_price - closest_support) / current_price) * 100
            if abs(distance_to_support) < 2:
                reasoning_parts.append(f"Price at support ${closest_support:,.2f}")
        
        if resistance_levels:
            closest_resistance = min(resistance_levels, key=lambda x: abs(current_price - x))
            distance_to_resistance = ((closest_resistance - current_price) / current_price) * 100
            if abs(distance_to_resistance) < 2:
                reasoning_parts.append(f"Price at resistance ${closest_resistance:,.2f}")
        
        # Determine direction
        if bullish_signals > bearish_signals and bullish_signals >= 1:
            direction = SignalDirection.LONG
            probability = min(0.95, 0.60 + (bullish_signals * 0.08))
        elif bearish_signals > bullish_signals and bearish_signals >= 1:
            direction = SignalDirection.SHORT
            probability = min(0.95, 0.60 + (bearish_signals * 0.08))
        else:
            direction = SignalDirection.FLAT
            probability = 0.5
        
        # Calculate confidence based on pattern quality (textbook-perfect + volume-confirmed)
        if len(confirmed_patterns) >= 3 and len(confidence_scores) >= 3:
            # Multiple strong patterns = high confidence
            avg_pattern_quality = sum(confidence_scores) / len(confidence_scores)
            textbook_patterns = sum(1 for p in confirmed_patterns if p.get('volume_confirmed') == True)
            
            if textbook_patterns >= 2:
                # "Textbook-perfect and volume-confirmed" per requirements
                confidence = min(0.95, avg_pattern_quality + 0.05)
            else:
                confidence = avg_pattern_quality
        elif len(confirmed_patterns) >= 1:
            # At least one confirmed pattern
            avg_pattern_quality = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.70
            confidence = min(0.90, avg_pattern_quality)
        else:
            # No strong patterns detected
            confidence = 0.60
            reasoning_parts = ["No strong classical patterns detected at key levels"]
        
        # Build reasoning
        if reasoning_parts:
            reasoning = "; ".join(reasoning_parts[:5])  # Limit to top 5 reasons
        else:
            reasoning = f"Analyzed {len(all_candlestick)} candlestick + {len(all_chart)} chart patterns; no strong setups"
        
        # Add summary
        reasoning = f"[{len(confirmed_patterns)} patterns confirmed] " + reasoning
        
        return direction, probability, confidence, reasoning, features_used if features_used else self.features
    
    async def _basic_analysis(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Fallback basic analysis when pattern detectors unavailable"""
        try:
            technical_data = analysis_results.get('technical_analysis', {})
            trend = technical_data.get('trend', 'neutral')
            strength = technical_data.get('strength', 'normal')
            
            reasoning = ["Fallback: Pattern detectors unavailable"]
            
            if trend == 'bullish' and strength == 'strong':
                direction = SignalDirection.LONG
                probability = 0.65
                confidence = 0.65
                reasoning.append("Bullish trend with strength")
            elif trend == 'bearish' and strength == 'strong':
                direction = SignalDirection.SHORT
                probability = 0.65
                confidence = 0.65
                reasoning.append("Bearish trend with strength")
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
                confidence = 0.60
                reasoning.append("No clear pattern setup")
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_D,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=self.features,
                reasoning="; ".join(reasoning)
            )
            
        except Exception as e:
            logger.error(f"Basic analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HEAD_D,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class ICTConceptsHead:
    """ICT (Inner Circle Trader) Concepts Analysis Head"""
    
    def __init__(self):
        self.name = "ICT Concepts Head"
        self.features = ['ote_zones', 'bpr', 'judas_swings', 'liquidity_sweeps', 'fair_value_gaps', 'kill_zones']
        
        # Initialize ICT engine lazily
        self.ict_engine = None
        self.session_context_manager = None
    
    async def _initialize_engines(self):
        """Lazy initialization of ICT engines"""
        if self.ict_engine is None:
            try:
                from ..strategies.ict_concepts_engine import ICTConceptsEngine
                from ..strategies.session_context_manager import SessionContextManager
                
                self.ict_engine = ICTConceptsEngine()
                self.session_context_manager = SessionContextManager()
                logger.info("✅ ICT Concepts Engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize ICT engines: {e}")
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform ICT concepts analysis"""
        try:
            # Initialize engines if needed
            await self._initialize_engines()
            
            if self.ict_engine is None:
                return ModelHeadResult(
                    head_type=ModelHead.ICT_CONCEPTS,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="ICT engine not available"
                )
            
            # Get market data DataFrame
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 20:
                return ModelHeadResult(
                    head_type=ModelHead.ICT_CONCEPTS,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Insufficient data for ICT analysis"
                )
            
            symbol = market_data.get('symbol', 'UNKNOWN')
            timeframe = market_data.get('timeframe', '1h')
            
            # Perform ICT analysis
            ict_analysis = await self.ict_engine.analyze_ict_concepts(df, symbol, timeframe)
            
            # Get session context for kill zone filtering
            session_context = self.session_context_manager.get_session_context()
            
            # Analyze ICT signals
            bullish_signals = 0
            bearish_signals = 0
            reasoning = []
            total_confidence = 0.0
            signal_count = 0
            
            # OTE zone analysis
            for ote in ict_analysis.ote_zones:
                if ote.is_price_in_zone:
                    if ote.zone_type == 'bullish':
                        bullish_signals += 1
                        reasoning.append(f"Price in bullish OTE zone (conf: {ote.confidence:.2f})")
                    else:
                        bearish_signals += 1
                        reasoning.append(f"Price in bearish OTE zone (conf: {ote.confidence:.2f})")
                    total_confidence += ote.confidence
                    signal_count += 1
            
            # BPR analysis
            for bpr in ict_analysis.balanced_price_ranges:
                if bpr.is_near_equilibrium:
                    reasoning.append(f"Price near BPR equilibrium (conf: {bpr.confidence:.2f})")
                    total_confidence += bpr.confidence * 0.5  # Lower weight for BPR
                    signal_count += 1
            
            # Judas swing analysis
            for judas in ict_analysis.judas_swings:
                if judas.swing_type == 'bullish':
                    bullish_signals += 1
                    reasoning.append(f"Bullish Judas swing detected (strength: {judas.reversal_strength:.2f})")
                else:
                    bearish_signals += 1
                    reasoning.append(f"Bearish Judas swing detected (strength: {judas.reversal_strength:.2f})")
                total_confidence += judas.confidence
                signal_count += 1
            
            # Liquidity sweep analysis (CRITICAL - highest priority ICT signal)
            for sweep in ict_analysis.liquidity_sweeps:
                vol_status = " + volume spike" if sweep.volume_spike else ""
                if sweep.sweep_type == 'bullish':
                    bullish_signals += 2  # Sweeps are weighted heavily
                    reasoning.append(f"Bullish liquidity sweep at ${sweep.swept_level:,.2f}{vol_status} (conf: {sweep.confidence:.2f})")
                else:
                    bearish_signals += 2
                    reasoning.append(f"Bearish liquidity sweep at ${sweep.swept_level:,.2f}{vol_status} (conf: {sweep.confidence:.2f})")
                # Liquidity sweeps get extra weight (1.2x)
                total_confidence += sweep.confidence * 1.2
                signal_count += 1
            
            # Kill zone bonus
            if session_context.is_high_probability_time:
                reasoning.append(f"Active kill zone: {session_context.active_kill_zone.value}")
                total_confidence *= session_context.probability_multiplier
            
            # Determine direction and confidence
            if bullish_signals > bearish_signals and bullish_signals > 0:
                direction = SignalDirection.LONG
                probability = min(0.95, 0.6 + (bullish_signals * 0.1))
            elif bearish_signals > bullish_signals and bearish_signals > 0:
                direction = SignalDirection.SHORT
                probability = min(0.95, 0.6 + (bearish_signals * 0.1))
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
            
            # Calculate final confidence
            if signal_count > 0:
                confidence = min(0.95, total_confidence / signal_count)
            else:
                confidence = 0.0
            
            # Apply kill zone multiplier to confidence
            confidence *= session_context.probability_multiplier
            confidence = min(0.95, confidence)
            
            reasoning_text = "; ".join(reasoning) if reasoning else "No significant ICT patterns detected"
            
            return ModelHeadResult(
                head_type=ModelHead.ICT_CONCEPTS,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=self.features,
                reasoning=reasoning_text
            )
            
        except Exception as e:
            logger.error(f"ICT concepts analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.ICT_CONCEPTS,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class WyckoffHead:
    """Wyckoff Methodology Analysis Head"""
    
    def __init__(self):
        self.name = "Wyckoff Head"
        self.features = ['wyckoff_phase', 'spring', 'utad', 'sos', 'sow', 'composite_operator']
        
        # Initialize Wyckoff engine lazily
        self.wyckoff_engine = None
    
    async def _initialize_engine(self):
        """Lazy initialization of Wyckoff engine"""
        if self.wyckoff_engine is None:
            try:
                from ..strategies.wyckoff_analysis_engine import WyckoffAnalysisEngine
                self.wyckoff_engine = WyckoffAnalysisEngine()
                logger.info("✅ Wyckoff Analysis Engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Wyckoff engine: {e}")
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform Wyckoff methodology analysis"""
        try:
            # Initialize engine if needed
            await self._initialize_engine()
            
            if self.wyckoff_engine is None:
                return ModelHeadResult(
                    head_type=ModelHead.WYCKOFF,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Wyckoff engine not available"
                )
            
            # Get market data DataFrame
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 50:
                return ModelHeadResult(
                    head_type=ModelHead.WYCKOFF,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Insufficient data for Wyckoff analysis"
                )
            
            symbol = market_data.get('symbol', 'UNKNOWN')
            timeframe = market_data.get('timeframe', '1h')
            
            # Perform Wyckoff analysis
            wyckoff_analysis = await self.wyckoff_engine.analyze_wyckoff(df, symbol, timeframe)
            
            # Import phase types for comparison
            from ..strategies.wyckoff_analysis_engine import WyckoffPhase, WyckoffSchematic
            
            # Analyze signals
            reasoning = []
            direction = SignalDirection.FLAT
            probability = 0.5
            confidence = wyckoff_analysis.overall_confidence
            factors = []
            indicators_dict = {}
            score_breakdown = {}
            
            # High-confidence phases (Spring, UTAD) - HIGHEST confidence signals
            if wyckoff_analysis.current_phase == WyckoffPhase.SPRING:
                direction = SignalDirection.LONG
                probability = 0.9
                confidence = 0.90  # Fixed 0.90 - "rarely fails" per requirements
                reasoning.append("Spring detected - final shakeout before rally (HIGH CONFIDENCE)")
                factors.append("Price broke below support then reversed")
                factors.append("Low volume on breakdown (weak selling)")
                factors.append("Quick recovery = institutional trap")
                score_breakdown['spring_pattern'] = 0.90
                
            elif wyckoff_analysis.current_phase == WyckoffPhase.UTAD:
                direction = SignalDirection.SHORT
                probability = 0.9
                confidence = 0.90  # Fixed 0.90 - "rarely fails" per requirements
                reasoning.append("UTAD detected - final pump before dump (HIGH CONFIDENCE)")
                factors.append("Price broke above resistance then reversed")
                factors.append("Volume climax then weakness")
                factors.append("Quick rejection = institutional trap")
                score_breakdown['utad_pattern'] = 0.90
                
            elif wyckoff_analysis.current_phase == WyckoffPhase.SOS:
                direction = SignalDirection.LONG
                probability = 0.75
                confidence = 0.75  # Medium-high confidence
                reasoning.append("Sign of Strength - price advances on increasing volume")
                factors.append("Strong bullish breakout")
                factors.append("High volume confirmation")
                score_breakdown['sos_pattern'] = 0.75
                
            elif wyckoff_analysis.current_phase == WyckoffPhase.SOW:
                direction = SignalDirection.SHORT
                probability = 0.75
                confidence = 0.75  # Medium-high confidence
                reasoning.append("Sign of Weakness - price declines on increasing volume")
                factors.append("Strong bearish breakdown")
                factors.append("High volume confirmation")
                score_breakdown['sow_pattern'] = 0.75
            
            # Composite operator signals
            composite_conf = wyckoff_analysis.composite_operator.confidence
            if wyckoff_analysis.composite_operator.is_accumulating:
                if direction == SignalDirection.FLAT:
                    direction = SignalDirection.LONG
                    probability = 0.7
                    confidence = 0.70  # Moderate confidence for phase ID
                reasoning.append(f"Smart money accumulation (institutional footprint: {wyckoff_analysis.composite_operator.institutional_footprint:.2f})")
                factors.append("Composite operator accumulating")
                # Boost confidence if we already have a signal
                if confidence >= 0.75:
                    confidence = min(0.95, confidence * 1.05)  # Small boost
                score_breakdown['composite_operator'] = composite_conf
                
            elif wyckoff_analysis.composite_operator.is_distributing:
                if direction == SignalDirection.FLAT:
                    direction = SignalDirection.SHORT
                    probability = 0.7
                    confidence = 0.70  # Moderate confidence for phase ID
                reasoning.append(f"Smart money distribution (institutional footprint: {wyckoff_analysis.composite_operator.institutional_footprint:.2f})")
                factors.append("Composite operator distributing")
                # Boost confidence if we already have a signal
                if confidence >= 0.75:
                    confidence = min(0.95, confidence * 1.05)
                score_breakdown['composite_operator'] = composite_conf
            
            # Schematic context
            if wyckoff_analysis.current_schematic == WyckoffSchematic.ACCUMULATION:
                reasoning.append(f"Accumulation phase: {wyckoff_analysis.current_phase.value}")
                indicators_dict['schematic'] = 'accumulation'
            elif wyckoff_analysis.current_schematic == WyckoffSchematic.DISTRIBUTION:
                reasoning.append(f"Distribution phase: {wyckoff_analysis.current_phase.value}")
                indicators_dict['schematic'] = 'distribution'
            
            # Add current phase to indicators
            indicators_dict['current_phase'] = wyckoff_analysis.current_phase.value
            indicators_dict['institutional_footprint'] = wyckoff_analysis.composite_operator.institutional_footprint
            indicators_dict['absorption_detected'] = wyckoff_analysis.composite_operator.absorption_detected
            indicators_dict['effort_vs_result'] = wyckoff_analysis.composite_operator.effort_vs_result_score
            
            # Add detection thresholds for transparency
            indicators_dict['thresholds'] = {
                'spring_volume_max': '< 0.8x avg',
                'utad_volume_climax': '> 1.5x avg',
                'sos_volume_min': '> 1.5x avg',
                'sow_volume_min': '> 1.5x avg',
                'high_confidence': '0.90 (Spring/UTAD)',
                'medium_confidence': '0.75 (SOS/SOW)',
                'phase_confidence': '0.65-0.75'
            }
            
            # Add event details to factors
            for event in wyckoff_analysis.wyckoff_events[-3:]:  # Last 3 events
                factors.append(f"{event.event_type.value.replace('_', ' ').title()} at ${event.price:,.2f} (conf: {event.confidence:.2f})")
            
            # Add overall to score breakdown
            score_breakdown['overall'] = confidence
            
            confidence = min(0.95, confidence)
            reasoning_text = "; ".join(reasoning) if reasoning else "No significant Wyckoff patterns detected"
            
            # Store enhanced data for frontend (will be passed through if signal generation supports it)
            result_data = {
                'head_type': ModelHead.WYCKOFF,
                'direction': direction,
                'probability': probability,
                'confidence': confidence,
                'features_used': self.features,
                'reasoning': reasoning_text,
                'indicators': indicators_dict,  # For frontend display
                'factors': factors,  # For frontend display
                'score_breakdown': score_breakdown  # For frontend display
            }
            
            return ModelHeadResult(
                head_type=ModelHead.WYCKOFF,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=self.features,
                reasoning=reasoning_text
            )
            
        except Exception as e:
            logger.error(f"Wyckoff analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.WYCKOFF,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class HarmonicPatternsHead:
    """Harmonic Patterns Analysis Head"""
    
    def __init__(self):
        self.name = "Harmonic Patterns Head"
        self.features = ['gartley', 'butterfly', 'bat', 'crab', 'abcd']
        self.harmonic_engine = None
    
    async def _initialize_engine(self):
        """Lazy initialization of Harmonic engine"""
        if self.harmonic_engine is None:
            try:
                from ..strategies.harmonic_patterns_engine import HarmonicPatternsEngine
                self.harmonic_engine = HarmonicPatternsEngine()
                logger.info("✅ Harmonic Patterns Engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Harmonic engine: {e}")
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform harmonic patterns analysis"""
        try:
            await self._initialize_engine()
            
            if self.harmonic_engine is None:
                return ModelHeadResult(
                    head_type=ModelHead.HARMONIC,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Harmonic engine not available"
                )
            
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 50:
                return ModelHeadResult(
                    head_type=ModelHead.HARMONIC,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Insufficient data for harmonic analysis"
                )
            
            symbol = market_data.get('symbol', 'UNKNOWN')
            timeframe = market_data.get('timeframe', '1h')
            
            harmonic_analysis = await self.harmonic_engine.analyze_harmonic_patterns(df, symbol, timeframe)
            
            # Import pattern types
            from ..strategies.harmonic_patterns_engine import PatternDirection
            
            reasoning = []
            direction = SignalDirection.FLAT
            probability = 0.5
            confidence = harmonic_analysis.overall_confidence
            
            # Analyze active patterns
            bullish_patterns = [p for p in harmonic_analysis.active_patterns if p.direction == PatternDirection.BULLISH]
            bearish_patterns = [p for p in harmonic_analysis.active_patterns if p.direction == PatternDirection.BEARISH]
            
            if bullish_patterns:
                direction = SignalDirection.LONG
                probability = min(0.9, 0.7 + len(bullish_patterns) * 0.1)
                for p in bullish_patterns:
                    reasoning.append(f"{p.pattern_type.value.title()} bullish completion (precision: {p.ratio_precision:.2f})")
            elif bearish_patterns:
                direction = SignalDirection.SHORT
                probability = min(0.9, 0.7 + len(bearish_patterns) * 0.1)
                for p in bearish_patterns:
                    reasoning.append(f"{p.pattern_type.value.title()} bearish completion (precision: {p.ratio_precision:.2f})")
            
            reasoning_text = "; ".join(reasoning) if reasoning else "No active harmonic patterns"
            
            return ModelHeadResult(
                head_type=ModelHead.HARMONIC,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=self.features,
                reasoning=reasoning_text
            )
            
        except Exception as e:
            logger.error(f"Harmonic patterns analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.HARMONIC,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class EnhancedMarketStructureHead:
    """Enhanced Market Structure Analysis Head"""
    
    def __init__(self):
        self.name = "Enhanced Market Structure Head"
        self.features = ['mtf_alignment', 'premium_discount', 'mitigation_blocks', 'breaker_blocks']
        self.structure_engine = None
    
    async def _initialize_engine(self):
        """Lazy initialization of Structure engine"""
        if self.structure_engine is None:
            try:
                from ..strategies.enhanced_market_structure_engine import EnhancedMarketStructureEngine
                self.structure_engine = EnhancedMarketStructureEngine()
                logger.info("✅ Enhanced Market Structure Engine initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Structure engine: {e}")
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform enhanced market structure analysis"""
        try:
            await self._initialize_engine()
            
            if self.structure_engine is None:
                return ModelHeadResult(
                    head_type=ModelHead.MARKET_STRUCTURE,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Market structure engine not available"
                )
            
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 50:
                return ModelHeadResult(
                    head_type=ModelHead.MARKET_STRUCTURE,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Insufficient data for market structure analysis"
                )
            
            symbol = market_data.get('symbol', 'UNKNOWN')
            timeframe = market_data.get('timeframe', '1h')
            
            # Get multi-timeframe data if available
            multi_tf_data = analysis_results.get('multi_timeframe_data')
            
            structure_analysis = await self.structure_engine.analyze_enhanced_structure(
                df, symbol, timeframe, multi_tf_data
            )
            
            # Import enums
            from ..strategies.enhanced_market_structure_engine import PriceZone
            
            reasoning = []
            direction = SignalDirection.FLAT
            probability = 0.5
            confidence = 0.60  # Base confidence for conflicted timeframes
            factors = []
            indicators_dict = {}
            score_breakdown = {}
            
            # Get premium/discount zone first
            pd_zone = structure_analysis.premium_discount
            
            # Multi-timeframe alignment analysis
            mtf = structure_analysis.mtf_alignment
            
            # Calculate confidence based on MTF alignment (per requirements)
            if mtf.aligned and mtf.alignment_score >= 0.75:
                # Multi-timeframe aligned → HIGH confidence (0.85-0.90)
                base_confidence = 0.85 + ((mtf.alignment_score - 0.75) / 0.25) * 0.05  # 0.85-0.90
                
                if mtf.alignment_direction == 'bullish':
                    direction = SignalDirection.LONG
                    probability = min(0.9, 0.7 + mtf.alignment_score * 0.2)
                    reasoning.append(f"MTF bullish alignment ({int(mtf.alignment_score * 100)}% of timeframes)")
                    factors.append(f"{len([s for s in mtf.structures.values() if s.trend == 'bullish'])}/{len(mtf.structures)} timeframes bullish")
                    
                elif mtf.alignment_direction == 'bearish':
                    direction = SignalDirection.SHORT
                    probability = min(0.9, 0.7 + mtf.alignment_score * 0.2)
                    reasoning.append(f"MTF bearish alignment ({int(mtf.alignment_score * 100)}% of timeframes)")
                    factors.append(f"{len([s for s in mtf.structures.values() if s.trend == 'bearish'])}/{len(mtf.structures)} timeframes bearish")
                
                confidence = base_confidence
                score_breakdown['mtf_alignment'] = mtf.alignment_score
                
            else:
                # Timeframes conflicted → LOW confidence (0.60-0.70)
                confidence = 0.60 + (mtf.alignment_score * 0.10)  # 0.60-0.70
                reasoning.append(f"MTF conflicted (alignment: {mtf.alignment_score:.2f})")
                score_breakdown['mtf_alignment'] = mtf.alignment_score
            
            # Perfect setup detection: MTF aligned + correct zone
            if (mtf.aligned and mtf.alignment_direction == 'bullish' and 
                pd_zone.current_zone == PriceZone.DISCOUNT):
                # PERFECT BULLISH SETUP
                confidence = 0.90  # FIXED at 0.90 for perfect setup
                reasoning.insert(0, "PERFECT SETUP: All TFs bullish + discount zone")
                factors.append("Price in lower 50% of range (good to buy)")
                factors.append("All timeframes showing bullish structure")
                score_breakdown['perfect_setup'] = 0.90
                
            elif (mtf.aligned and mtf.alignment_direction == 'bearish' and 
                  pd_zone.current_zone == PriceZone.PREMIUM):
                # PERFECT BEARISH SETUP
                confidence = 0.90  # FIXED at 0.90 for perfect setup
                reasoning.insert(0, "PERFECT SETUP: All TFs bearish + premium zone")
                factors.append("Price in upper 50% of range (good to sell)")
                factors.append("All timeframes showing bearish structure")
                score_breakdown['perfect_setup'] = 0.90
                
            elif direction != SignalDirection.FLAT:
                # Good setup but not perfect (aligned but wrong zone or vice versa)
                if pd_zone.current_zone == PriceZone.DISCOUNT and direction == SignalDirection.LONG:
                    reasoning.append("Price in discount zone (buy zone)")
                    factors.append(f"Discount zone: {pd_zone.metadata.get('price_percentage', 0):.1f}% of range")
                    confidence = min(0.88, confidence * 1.03)  # Small boost
                elif pd_zone.current_zone == PriceZone.PREMIUM and direction == SignalDirection.SHORT:
                    reasoning.append("Price in premium zone (sell zone)")
                    factors.append(f"Premium zone: {pd_zone.metadata.get('price_percentage', 0):.1f}% of range")
                    confidence = min(0.88, confidence * 1.03)
                elif pd_zone.current_zone == PriceZone.EQUILIBRIUM:
                    reasoning.append("Price at equilibrium (decision point)")
                    factors.append("Price at 50% equilibrium level")
            
            # Unmitigated order blocks
            unmitigated = [b for b in structure_analysis.mitigation_blocks if not b.is_mitigated]
            if unmitigated:
                reasoning.append(f"{len(unmitigated)} unmitigated order block(s)")
                for block in unmitigated[:2]:  # Show top 2
                    factors.append(f"Unmitigated {block.block_type} block at ${block.low:,.2f}-${block.high:,.2f}")
                confidence = min(0.92, confidence + len(unmitigated) * 0.02)
                score_breakdown['order_blocks'] = min(1.0, len(unmitigated) * 0.25)
            
            # Breaker blocks (only set direction if FLAT, don't override confidence)
            if structure_analysis.breaker_blocks:
                for breaker in structure_analysis.breaker_blocks[:1]:
                    if breaker.breaker_type == 'bullish':
                        if direction == SignalDirection.FLAT:
                            direction = SignalDirection.LONG
                            probability = 0.75
                            confidence = max(confidence, 0.75)  # Don't reduce confidence
                        reasoning.append("Breaker block flipped to support")
                        factors.append(f"Breaker block polarity flip at ${breaker.low:,.2f}")
                    elif breaker.breaker_type == 'bearish':
                        if direction == SignalDirection.FLAT:
                            direction = SignalDirection.SHORT
                            probability = 0.75
                            confidence = max(confidence, 0.75)  # Don't reduce confidence
                        reasoning.append("Breaker block flipped to resistance")
                        factors.append(f"Breaker block polarity flip at ${breaker.high:,.2f}")
                    
                    score_breakdown['breaker_blocks'] = 0.75
            
            # Build indicators dict for frontend display
            indicators_dict = {
                'timeframes_analyzed': mtf.timeframes,
                'aligned_count': sum(1 for s in mtf.structures.values() if s.trend == mtf.alignment_direction) if mtf.aligned else 0,
                'total_timeframes': len(mtf.timeframes),
                'alignment_score': mtf.alignment_score,
                'alignment_direction': mtf.alignment_direction,
                'current_zone': pd_zone.current_zone.value,
                'zone_percentage': pd_zone.metadata.get('price_percentage', 50),
                'equilibrium_level': pd_zone.equilibrium,
                'range_high': pd_zone.range_high,
                'range_low': pd_zone.range_low,
                'unmitigated_blocks': len(unmitigated),
                'breaker_blocks': len(structure_analysis.breaker_blocks),
                'thresholds': {
                    'mtf_alignment_min': '75% (3/4 TFs)',
                    'perfect_setup_confidence': '0.90',
                    'high_confidence': '0.85-0.90 (aligned)',
                    'low_confidence': '0.60-0.70 (conflicted)',
                    'discount_zone': '0-50% of range',
                    'premium_zone': '50-100% of range',
                    'equilibrium': '45-55%'
                }
            }
            
            # Add timeframe breakdown to factors
            for tf, structure in list(mtf.structures.items())[:5]:  # Show up to 5 TFs
                factors.append(f"{tf}: {structure.trend.title()} structure")
            
            # Add overall score
            score_breakdown['zone_quality'] = 0.80 if pd_zone.current_zone != PriceZone.EQUILIBRIUM else 0.50
            score_breakdown['overall'] = confidence
            
            confidence = min(0.95, confidence)
            reasoning_text = "; ".join(reasoning) if reasoning else "No significant structure patterns"
            
            return ModelHeadResult(
                head_type=ModelHead.MARKET_STRUCTURE,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=self.features,
                reasoning=reasoning_text
            )
            
        except Exception as e:
            logger.error(f"Enhanced market structure analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.MARKET_STRUCTURE,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class CryptoMetricsHead:
    """Crypto-Specific Metrics Analysis Head"""
    
    def __init__(self):
        self.name = "Crypto Metrics Head"
        self.features = [
            'cvd', 'alt_season_index', 'long_short_ratio', 'perpetual_premium',
            'liquidation_risk', 'taker_flow', 'exchange_reserves'
        ]
        
        # Initialize crypto analyzers lazily
        self.cvd_analyzer = None
        self.alt_season_index = None
        self.exchange_metrics = None
        self.derivatives_analyzer = None
        self.taker_flow_analyzer = None
        self.reserves_tracker = None
    
    async def _initialize_analyzers(self):
        """Lazy initialization of crypto analyzers"""
        if self.cvd_analyzer is None:
            try:
                from ..strategies.cvd_analyzer import CVDAnalyzer
                from ..strategies.altcoin_season_index import AltcoinSeasonIndex
                from ..data.exchange_metrics_collector import ExchangeMetricsCollector
                from ..strategies.derivatives_analyzer import DerivativesAnalyzer
                from ..strategies.taker_flow_analyzer import TakerFlowAnalyzer
                from ..data.exchange_reserves_tracker import ExchangeReservesTracker
                
                self.cvd_analyzer = CVDAnalyzer()
                self.alt_season_index = AltcoinSeasonIndex()
                self.exchange_metrics = ExchangeMetricsCollector()
                self.derivatives_analyzer = DerivativesAnalyzer()
                self.taker_flow_analyzer = TakerFlowAnalyzer()
                self.reserves_tracker = ExchangeReservesTracker()
                
                logger.info("✅ Crypto Metrics analyzers initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize crypto analyzers: {e}")
    
    async def analyze(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> ModelHeadResult:
        """Perform crypto-specific metrics analysis"""
        try:
            # Initialize analyzers if needed
            await self._initialize_analyzers()
            
            if self.cvd_analyzer is None:
                return ModelHeadResult(
                    head_type=ModelHead.CRYPTO_METRICS,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Crypto metrics engines not available"
                )
            
            df = analysis_results.get('dataframe')
            if df is None or len(df) < 20:
                return ModelHeadResult(
                    head_type=ModelHead.CRYPTO_METRICS,
                    direction=SignalDirection.FLAT,
                    probability=0.5,
                    confidence=0.0,
                    features_used=[],
                    reasoning="Insufficient data for crypto metrics analysis"
                )
            
            symbol = market_data.get('symbol', 'UNKNOWN')
            timeframe = market_data.get('timeframe', '1h')
            
            # Aggregate scores
            bullish_signals = 0
            bearish_signals = 0
            reasoning = []
            total_confidence = 0.0
            signal_count = 0
            
            # 1. CVD Analysis
            try:
                cvd_analysis = await self.cvd_analyzer.analyze_cvd(df, symbol, timeframe)
                
                if cvd_analysis.overall_confidence > 0.6:
                    # Check for bullish divergences
                    bullish_divs = [d for d in cvd_analysis.divergences if 'bullish' in d.divergence_type.value]
                    bearish_divs = [d for d in cvd_analysis.divergences if 'bearish' in d.divergence_type.value]
                    
                    if bullish_divs:
                        bullish_signals += 1
                        div_conf = max(0.85, bullish_divs[0].confidence)  # Ensure 0.85+ for strong divergences
                        reasoning.append(f"CVD bullish divergence (conf: {div_conf:.2f})")
                        total_confidence += div_conf
                        signal_count += 1
                    elif bearish_divs:
                        bearish_signals += 1
                        div_conf = max(0.85, bearish_divs[0].confidence)  # Ensure 0.85+ for strong divergences
                        reasoning.append(f"CVD bearish divergence (conf: {div_conf:.2f})")
                        total_confidence += div_conf
                        signal_count += 1
                    
                    # CVD trend
                    if cvd_analysis.cvd_trend == 'bullish':
                        bullish_signals += 1
                        reasoning.append("CVD trend bullish")
                    elif cvd_analysis.cvd_trend == 'bearish':
                        bearish_signals += 1
                        reasoning.append("CVD trend bearish")
            except Exception as e:
                logger.debug(f"CVD analysis skipped: {e}")
            
            # 2. Altcoin Season Index
            try:
                alt_season = await self.alt_season_index.calculate_index()
                
                if alt_season.index_value > 75:
                    # Alt season - bullish for alts
                    if 'BTC' not in symbol:  # If trading an altcoin
                        bullish_signals += 1
                        reasoning.append(f"Alt season active ({alt_season.index_value:.0f})")
                        total_confidence += 0.8
                        signal_count += 1
                elif alt_season.index_value < 25:
                    # BTC season - bearish for alts
                    if 'BTC' not in symbol:
                        bearish_signals += 1
                        reasoning.append(f"BTC season active ({alt_season.index_value:.0f})")
                        total_confidence += 0.8
                        signal_count += 1
            except Exception as e:
                logger.debug(f"Alt season analysis skipped: {e}")
            
            # 3. Long/Short Ratio
            try:
                exchange_metrics = await self.exchange_metrics.analyze_exchange_metrics(symbol)
                
                from ..data.exchange_metrics_collector import SentimentExtreme
                
                if exchange_metrics.overall_sentiment == SentimentExtreme.EXTREME_LONG:
                    bearish_signals += 1  # Contrarian
                    reasoning.append(f"Extreme long positioning (>{3.0:.1f} ratio) - contrarian bearish (conf: 0.85)")
                    total_confidence += 0.85  # Fixed to 0.85 per requirements
                    signal_count += 1
                elif exchange_metrics.overall_sentiment == SentimentExtreme.EXTREME_SHORT:
                    bullish_signals += 1  # Contrarian
                    reasoning.append(f"Extreme short positioning (<{0.33:.2f} ratio) - contrarian bullish (conf: 0.85)")
                    total_confidence += 0.85  # Fixed to 0.85 per requirements
                    signal_count += 1
            except Exception as e:
                logger.debug(f"Exchange metrics analysis skipped: {e}")
            
            # 4. Perpetual Premium
            try:
                current_price = df['close'].iloc[-1]
                derivatives_analysis = await self.derivatives_analyzer.analyze_derivatives(
                    symbol, current_price
                )
                
                if derivatives_analysis.overall_signal:
                    if derivatives_analysis.overall_signal == 'bullish':
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                    
                    reasoning.append(f"Derivatives {derivatives_analysis.overall_signal}")
                    total_confidence += derivatives_analysis.overall_confidence
                    signal_count += 1
            except Exception as e:
                logger.debug(f"Derivatives analysis skipped: {e}")
            
            # 5. Taker Flow
            try:
                taker_analysis = await self.taker_flow_analyzer.analyze_taker_flow(df, symbol, timeframe)
                
                from ..strategies.taker_flow_analyzer import TakerSentiment
                
                if taker_analysis.taker_sentiment == TakerSentiment.STRONG_BUY_PRESSURE:
                    bullish_signals += 1
                    reasoning.append("Strong taker buy pressure")
                    total_confidence += taker_analysis.overall_confidence
                    signal_count += 1
                elif taker_analysis.taker_sentiment == TakerSentiment.STRONG_SELL_PRESSURE:
                    bearish_signals += 1
                    reasoning.append("Strong taker sell pressure")
                    total_confidence += taker_analysis.overall_confidence
                    signal_count += 1
            except Exception as e:
                logger.debug(f"Taker flow analysis skipped: {e}")
            
            # 6. Exchange Reserves (CRITICAL - was missing)
            try:
                reserves_analysis = await self.reserves_tracker.analyze_reserves(symbol)
                
                from ..data.exchange_reserves_tracker import ReserveLevel
                
                if reserves_analysis.reserve_level == ReserveLevel.MULTI_YEAR_LOW:
                    bullish_signals += 1
                    reasoning.append(f"Exchange reserves at multi-year lows - supply shock risk (conf: 0.85)")
                    total_confidence += 0.85  # High confidence per requirements
                    signal_count += 1
                elif reserves_analysis.overall_trend.value == 'sharp_outflow':
                    bullish_signals += 1
                    reasoning.append("Sharp exchange outflow - accumulation signal")
                    total_confidence += 0.75
                    signal_count += 1
                elif reserves_analysis.overall_trend.value == 'sharp_inflow':
                    bearish_signals += 1
                    reasoning.append("Sharp exchange inflow - distribution signal")
                    total_confidence += 0.70
                    signal_count += 1
            except Exception as e:
                logger.debug(f"Exchange reserves analysis skipped: {e}")
            
            # Determine final direction and confidence
            if bullish_signals > bearish_signals and bullish_signals > 0:
                direction = SignalDirection.LONG
                probability = min(0.90, 0.65 + (bullish_signals * 0.08))
            elif bearish_signals > bullish_signals and bearish_signals > 0:
                direction = SignalDirection.SHORT
                probability = min(0.90, 0.65 + (bearish_signals * 0.08))
            else:
                direction = SignalDirection.FLAT
                probability = 0.5
            
            # Calculate final confidence based on signal count (per requirements)
            if signal_count >= 5:
                # 5+ crypto signals agree strongly → High confidence (0.85+)
                avg_confidence = total_confidence / signal_count
                confidence = 0.85 + (avg_confidence - 0.80) * 0.25  # 0.85-0.90
                confidence = min(0.90, confidence)
            elif signal_count >= 3:
                # 3+ crypto signals agree → Medium-high confidence (0.80+)
                avg_confidence = total_confidence / signal_count
                confidence = 0.80 + (avg_confidence - 0.75) * 0.20  # 0.80-0.85
                confidence = min(0.85, confidence)
            elif signal_count > 0:
                # 1-2 signals
                avg_confidence = total_confidence / signal_count
                confidence = min(0.80, avg_confidence)
            else:
                confidence = 0.0
            
            # Build detailed response for frontend
            indicators_dict = {
                'signals_detected': signal_count,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'signal_strength': 'strong' if signal_count >= 5 else 'moderate' if signal_count >= 3 else 'weak',
                'thresholds': {
                    'long_short_extreme': '> 3.0 or < 0.33',
                    'perpetual_premium_extreme': '> 0.5% or < -0.3%',
                    'alt_season_threshold': '> 75 or < 25',
                    'taker_flow_threshold': '> 60%',
                    'high_confidence': '0.85+ (5+ signals)',
                    'medium_confidence': '0.80+ (3-4 signals)',
                    'low_confidence': '< 0.80 (1-2 signals)'
                }
            }
            
            factors = []
            if signal_count >= 5:
                factors.append(f"{signal_count} crypto signals aligned (VERY STRONG)")
            elif signal_count >= 3:
                factors.append(f"{signal_count} crypto signals aligned (STRONG)")
            
            # Add details from each signal for frontend display
            for reason in reasoning[:6]:  # Top 6 signals
                factors.append(reason)
            
            score_breakdown = {
                'signal_count': signal_count,
                'avg_signal_confidence': total_confidence / signal_count if signal_count > 0 else 0.0,
                'aggregation_boost': 0.85 if signal_count >= 5 else 0.80 if signal_count >= 3 else 0.70,
                'overall': confidence
            }
            
            reasoning_text = "; ".join(reasoning) if reasoning else "No significant crypto metric signals"
            
            return ModelHeadResult(
                head_type=ModelHead.CRYPTO_METRICS,
                direction=direction,
                probability=probability,
                confidence=confidence,
                features_used=self.features,
                reasoning=reasoning_text
            )
            
        except Exception as e:
            logger.error(f"Crypto metrics analysis error: {e}")
            return ModelHeadResult(
                head_type=ModelHead.CRYPTO_METRICS,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.0,
                features_used=[],
                reasoning=f"Error: {str(e)}"
            )

class ModelHeadsManager:
    """Manager for all AI model heads"""
    
    def __init__(self):
        self.heads = {
            ModelHead.HEAD_A: TechnicalAnalysisHead(),
            ModelHead.HEAD_B: SentimentAnalysisHead(),
            ModelHead.HEAD_C: VolumeAnalysisHead(),
            ModelHead.HEAD_D: RuleBasedHead(),
            ModelHead.ICT_CONCEPTS: ICTConceptsHead(),
            ModelHead.WYCKOFF: WyckoffHead(),
            ModelHead.HARMONIC: HarmonicPatternsHead(),
            ModelHead.MARKET_STRUCTURE: EnhancedMarketStructureHead(),
            ModelHead.CRYPTO_METRICS: CryptoMetricsHead()
        }
    
    async def analyze_all_heads(self, market_data: Dict[str, Any], analysis_results: Dict[str, Any]) -> List[ModelHeadResult]:
        """Run analysis on all model heads"""
        try:
            results = []
            
            for head_type, head in self.heads.items():
                try:
                    result = await head.analyze(market_data, analysis_results)
                    results.append(result)
                    logger.debug(f"Head {head_type.value}: {result.direction.value} (prob: {result.probability:.3f}, conf: {result.confidence:.3f})")
                except Exception as e:
                    logger.error(f"Error in head {head_type.value}: {e}")
                    # Add fallback result
                    results.append(ModelHeadResult(
                        head_type=head_type,
                        direction=SignalDirection.FLAT,
                        probability=0.5,
                        confidence=0.0,
                        features_used=[],
                        reasoning=f"Error: {str(e)}"
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Model heads analysis error: {e}")
            return []
