"""
Intelligent Analysis Engine for AlphaPulse
Integrates existing pattern detection with enhanced data collection for intelligent signal generation
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncpg
import ccxt
from dataclasses import dataclass
import uuid

# Import existing pattern detection components
from strategies.advanced_pattern_detector import AdvancedPatternDetector, PatternResult, PatternType, PatternStrength
from strategies.ml_pattern_detector import MLPatternDetector, MLPatternSignal
from strategies.volume_enhanced_pattern_detector import VolumeEnhancedPatternDetector, VolumeEnhancedPatternSignal

# Import new enhanced data collection components
from ..data_collection.market_intelligence_collector import MarketIntelligenceCollector, MarketIntelligenceData
from ..data_collection.volume_positioning_analyzer import VolumePositioningAnalyzer, VolumeAnalysis
from ..data_collection.enhanced_data_collection_manager import EnhancedDataCollectionManager

logger = logging.getLogger(__name__)

@dataclass
class IntelligentAnalysisResult:
    """Result of intelligent analysis"""
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Pattern Analysis
    pattern_confidence: float
    pattern_type: str
    pattern_strength: str
    
    # Technical Analysis
    technical_confidence: float
    rsi_value: float
    macd_signal: str
    bollinger_position: str
    support_level: float
    resistance_level: float
    
    # Sentiment Analysis
    sentiment_confidence: float
    news_sentiment: float
    social_sentiment: float
    market_sentiment: float
    
    # Volume Analysis
    volume_confidence: float
    volume_ratio: float
    volume_positioning: str
    order_book_imbalance: float
    
    # Market Regime Analysis
    market_regime_confidence: float
    market_regime: str
    volatility_level: str
    trend_direction: str
    
    # Overall Assessment
    overall_confidence: float
    risk_reward_ratio: float
    safe_entry_detected: bool
    
    # Entry/Exit Levels (only if safe entry detected)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None  # 50% of position
    take_profit_2: Optional[float] = None  # 25% of position
    take_profit_3: Optional[float] = None  # 15% of position
    take_profit_4: Optional[float] = None  # 10% of position
    position_size_percentage: Optional[float] = None
    
    # Analysis Details
    analysis_reasoning: str = ""
    no_safe_entry_reasons: List[str] = None
    signal_direction: str = "neutral"  # 'long', 'short', 'neutral'
    signal_strength: str = "weak"  # 'weak', 'moderate', 'strong', 'very_strong'

class IntelligentAnalysisEngine:
    """
    Intelligent Analysis Engine
    Integrates existing pattern detection with enhanced data collection
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange):
        self.db_pool = db_pool
        self.exchange = exchange
        
        # Initialize existing pattern detection components
        self.advanced_pattern_detector = AdvancedPatternDetector()
        self.ml_pattern_detector = MLPatternDetector()
        self.volume_enhanced_detector = VolumeEnhancedPatternDetector()
        
        # Initialize new enhanced data collection components
        self.market_intelligence_collector = MarketIntelligenceCollector(db_pool)
        self.volume_analyzer = VolumePositioningAnalyzer(db_pool, exchange)
        self.data_collection_manager = EnhancedDataCollectionManager(db_pool, exchange)
        
        # Analysis configuration
        self.confidence_threshold = 0.85  # 85% confidence threshold
        self.min_risk_reward_ratio = 2.0  # Minimum 2:1 risk/reward
        self.max_volatility_threshold = 0.05  # Maximum 5% volatility
        self.min_volume_ratio = 1.5  # Minimum 1.5x average volume
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info("Intelligent Analysis Engine initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🔄 Initializing Intelligent Analysis Engine...")
            
            # Initialize pattern detectors
            await self.advanced_pattern_detector.initialize()
            
            # Initialize data collection
            await self.data_collection_manager.start_collection()
            
            logger.info("✅ Intelligent Analysis Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Intelligent Analysis Engine: {e}")
            raise
    
    async def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> IntelligentAnalysisResult:
        """Perform comprehensive intelligent analysis for a symbol"""
        try:
            logger.info(f"🔄 Performing intelligent analysis for {symbol} {timeframe}")
            
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if (datetime.now() - cached_result['timestamp']).seconds < self.cache_duration:
                    logger.info(f"✅ Using cached analysis for {symbol}")
                    return cached_result['result']
            
            # Get current market data
            current_price = await self._get_current_price(symbol)
            if not current_price:
                raise ValueError(f"Could not get current price for {symbol}")
            
            # Perform analysis components
            pattern_analysis = await self._analyze_patterns(symbol, timeframe)
            technical_analysis = await self._analyze_technical_indicators(symbol, timeframe)
            sentiment_analysis = await self._analyze_sentiment(symbol)
            volume_analysis = await self._analyze_volume_positioning(symbol)
            market_regime_analysis = await self._analyze_market_regime()
            
            # Overall assessment
            overall_assessment = await self._perform_overall_assessment(
                pattern_analysis, technical_analysis, sentiment_analysis, 
                volume_analysis, market_regime_analysis, current_price
            )
            
            # Create analysis result
            analysis_result = IntelligentAnalysisResult(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.utcnow(),
                pattern_confidence=pattern_analysis['confidence'],
                pattern_type=pattern_analysis['type'],
                pattern_strength=pattern_analysis['strength'],
                technical_confidence=technical_analysis['confidence'],
                rsi_value=technical_analysis['rsi'],
                macd_signal=technical_analysis['macd_signal'],
                bollinger_position=technical_analysis['bollinger_position'],
                support_level=technical_analysis['support'],
                resistance_level=technical_analysis['resistance'],
                sentiment_confidence=sentiment_analysis['confidence'],
                news_sentiment=sentiment_analysis['news_sentiment'],
                social_sentiment=sentiment_analysis['social_sentiment'],
                market_sentiment=sentiment_analysis['market_sentiment'],
                volume_confidence=volume_analysis['confidence'],
                volume_ratio=volume_analysis['volume_ratio'],
                volume_positioning=volume_analysis['positioning'],
                order_book_imbalance=volume_analysis['order_book_imbalance'],
                market_regime_confidence=market_regime_analysis['confidence'],
                market_regime=market_regime_analysis['regime'],
                volatility_level=market_regime_analysis['volatility_level'],
                trend_direction=market_regime_analysis['trend_direction'],
                overall_confidence=overall_assessment['confidence'],
                risk_reward_ratio=overall_assessment['risk_reward_ratio'],
                safe_entry_detected=overall_assessment['safe_entry_detected'],
                entry_price=overall_assessment.get('entry_price'),
                stop_loss=overall_assessment.get('stop_loss'),
                take_profit_1=overall_assessment.get('take_profit_1'),
                take_profit_2=overall_assessment.get('take_profit_2'),
                take_profit_3=overall_assessment.get('take_profit_3'),
                take_profit_4=overall_assessment.get('take_profit_4'),
                position_size_percentage=overall_assessment.get('position_size_percentage'),
                analysis_reasoning=overall_assessment['reasoning'],
                no_safe_entry_reasons=overall_assessment.get('no_safe_entry_reasons', []),
                signal_direction=overall_assessment['signal_direction'],
                signal_strength=overall_assessment['signal_strength']
            )
            
            # Cache the result
            self.analysis_cache[cache_key] = {
                'result': analysis_result,
                'timestamp': datetime.now()
            }
            
            # Store in database
            await self._store_analysis_result(analysis_result)
            
            logger.info(f"✅ Intelligent analysis completed for {symbol}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return await self._create_default_analysis_result(symbol, timeframe)
    
    async def _analyze_patterns(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze patterns using existing pattern detection components"""
        try:
            # Get candlestick data
            candlestick_data = await self._get_candlestick_data(symbol, timeframe, 100)
            if not candlestick_data:
                return self._get_default_pattern_analysis()
            
            # Convert to DataFrame
            df = pd.DataFrame(candlestick_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 1. Advanced Pattern Detection
            advanced_patterns = await self.advanced_pattern_detector.detect_patterns(symbol, timeframe, 50)
            
            # 2. ML Pattern Detection
            ml_patterns = self.ml_pattern_detector.detect_patterns_ml(df)
            
            # 3. Volume-Enhanced Pattern Detection
            volume_patterns = self.volume_enhanced_detector.detect_patterns_with_volume(df)
            
            # Combine and score patterns
            combined_patterns = await self._combine_pattern_analysis(
                advanced_patterns, ml_patterns, volume_patterns
            )
            
            return combined_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return self._get_default_pattern_analysis()
    
    async def _analyze_technical_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze technical indicators"""
        try:
            # Get candlestick data
            candlestick_data = await self._get_candlestick_data(symbol, timeframe, 50)
            if not candlestick_data:
                return self._get_default_technical_analysis()
            
            df = pd.DataFrame(candlestick_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(df['close'], 14)
            macd_signal = self._calculate_macd_signal(df['close'])
            bollinger_position = self._calculate_bollinger_position(df['close'])
            support_resistance = self._calculate_support_resistance(df)
            
            # Calculate confidence based on indicator alignment
            # Get the latest RSI value as a float
            rsi_value = float(rsi.iloc[-1]) if not rsi.empty else 50.0
            confidence = self._calculate_technical_confidence(rsi_value, macd_signal, bollinger_position)
            
            return {
                'confidence': confidence,
                'rsi': float(rsi.iloc[-1]) if not rsi.empty else 50.0,
                'macd_signal': macd_signal,
                'bollinger_position': bollinger_position,
                'support': support_resistance['support'],
                'resistance': support_resistance['resistance']
            }
            
        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {e}")
            return self._get_default_technical_analysis()
    
    async def _analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze market sentiment"""
        try:
            # Get latest market intelligence
            market_intelligence = await self.market_intelligence_collector.get_latest_market_intelligence()
            
            if market_intelligence:
                return {
                    'confidence': 0.7,  # Base confidence for sentiment analysis
                    'news_sentiment': market_intelligence.news_sentiment_score,
                    'social_sentiment': 0.5,  # Placeholder
                    'market_sentiment': market_intelligence.market_sentiment_score
                }
            else:
                return self._get_default_sentiment_analysis()
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._get_default_sentiment_analysis()
    
    async def _analyze_volume_positioning(self, symbol: str) -> Dict[str, Any]:
        """Analyze volume positioning"""
        try:
            # Get latest volume analysis
            volume_analysis = await self.volume_analyzer.analyze_volume_positioning(symbol, '1h')
            
            if volume_analysis:
                return {
                    'confidence': volume_analysis.volume_positioning_score,
                    'volume_ratio': volume_analysis.volume_ratio,
                    'positioning': volume_analysis.volume_trend,
                    'order_book_imbalance': volume_analysis.order_book_imbalance
                }
            else:
                return self._get_default_volume_analysis()
                
        except Exception as e:
            logger.error(f"Error analyzing volume positioning: {e}")
            return self._get_default_volume_analysis()
    
    async def _analyze_market_regime(self) -> Dict[str, Any]:
        """Analyze market regime"""
        try:
            # Get latest market intelligence
            market_intelligence = await self.market_intelligence_collector.get_latest_market_intelligence()
            
            if market_intelligence:
                return {
                    'confidence': 0.8,  # Base confidence for market regime
                    'regime': market_intelligence.market_regime,
                    'volatility_level': self._classify_volatility(market_intelligence.volatility_index),
                    'trend_direction': self._classify_trend(market_intelligence.trend_strength)
                }
            else:
                return self._get_default_market_regime_analysis()
                
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return self._get_default_market_regime_analysis()
    
    async def _perform_overall_assessment(self, pattern_analysis: Dict, technical_analysis: Dict,
                                        sentiment_analysis: Dict, volume_analysis: Dict,
                                        market_regime_analysis: Dict, current_price: float) -> Dict[str, Any]:
        """Perform overall assessment and determine if safe entry exists"""
        try:
            # Calculate weighted confidence score
            weights = {
                'pattern': 0.25,
                'technical': 0.25,
                'sentiment': 0.15,
                'volume': 0.20,
                'market_regime': 0.15
            }
            
            overall_confidence = (
                pattern_analysis['confidence'] * weights['pattern'] +
                technical_analysis['confidence'] * weights['technical'] +
                sentiment_analysis['confidence'] * weights['sentiment'] +
                volume_analysis['confidence'] * weights['volume'] +
                market_regime_analysis['confidence'] * weights['market_regime']
            )
            
            # Determine signal direction
            signal_direction = self._determine_signal_direction(
                pattern_analysis, technical_analysis, sentiment_analysis
            )
            
            # Check if safe entry conditions are met
            safe_entry_detected, no_safe_entry_reasons = await self._check_safe_entry_conditions(
                overall_confidence, pattern_analysis, technical_analysis, 
                volume_analysis, market_regime_analysis, current_price
            )
            
            # Calculate risk/reward ratio if safe entry detected
            risk_reward_ratio = 0.0
            entry_price = None
            stop_loss = None
            take_profits = [None, None, None, None]
            position_size = None
            
            if safe_entry_detected:
                entry_price, stop_loss, take_profits, risk_reward_ratio = await self._calculate_entry_levels(
                    signal_direction, current_price, pattern_analysis, technical_analysis
                )
                position_size = self._calculate_position_size(risk_reward_ratio, overall_confidence)
            
            # Generate reasoning
            reasoning = self._generate_analysis_reasoning(
                pattern_analysis, technical_analysis, sentiment_analysis,
                volume_analysis, market_regime_analysis, safe_entry_detected
            )
            
            # Determine signal strength
            signal_strength = self._determine_signal_strength(overall_confidence)
            
            return {
                'confidence': overall_confidence,
                'risk_reward_ratio': risk_reward_ratio,
                'safe_entry_detected': safe_entry_detected,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit_1': take_profits[0],
                'take_profit_2': take_profits[1],
                'take_profit_3': take_profits[2],
                'take_profit_4': take_profits[3],
                'position_size_percentage': position_size,
                'reasoning': reasoning,
                'no_safe_entry_reasons': no_safe_entry_reasons,
                'signal_direction': signal_direction,
                'signal_strength': signal_strength
            }
            
        except Exception as e:
            logger.error(f"Error performing overall assessment: {e}")
            return self._get_default_overall_assessment()
    
    async def _check_safe_entry_conditions(self, overall_confidence: float, pattern_analysis: Dict,
                                         technical_analysis: Dict, volume_analysis: Dict,
                                         market_regime_analysis: Dict, current_price: float) -> Tuple[bool, List[str]]:
        """Check if safe entry conditions are met"""
        reasons = []
        
        # 1. Confidence threshold check
        if overall_confidence < self.confidence_threshold:
            reasons.append(f"Low confidence ({overall_confidence:.1%} < {self.confidence_threshold:.1%})")
        
        # 2. Volume confirmation check
        if volume_analysis['volume_ratio'] < self.min_volume_ratio:
            reasons.append(f"Insufficient volume (ratio: {volume_analysis['volume_ratio']:.2f} < {self.min_volume_ratio})")
        
        # 3. Volatility check
        if market_regime_analysis['volatility_level'] == 'high':
            reasons.append("High volatility market conditions")
        
        # 4. Market regime check
        if market_regime_analysis['regime'] == 'volatile':
            reasons.append("Volatile market regime")
        
        # 5. Pattern strength check
        if pattern_analysis['strength'] == 'weak':
            reasons.append("Weak pattern formation")
        
        # 6. Technical indicator alignment check
        if technical_analysis['confidence'] < 0.6:
            reasons.append("Poor technical indicator alignment")
        
        safe_entry = len(reasons) == 0
        return safe_entry, reasons
    
    async def _calculate_entry_levels(self, signal_direction: str, current_price: float,
                                    pattern_analysis: Dict, technical_analysis: Dict) -> Tuple[float, float, List[float], float]:
        """Calculate entry, stop loss, and take profit levels"""
        try:
            # Calculate ATR for volatility-based levels
            atr = await self._calculate_atr(current_price, 14)
            
            if signal_direction == 'long':
                entry_price = current_price
                stop_loss = current_price - (atr * 2.0)  # 2 ATR below entry
                take_profit_1 = current_price + (atr * 3.0)  # 3 ATR above entry
                take_profit_2 = current_price + (atr * 4.5)  # 4.5 ATR above entry
                take_profit_3 = current_price + (atr * 6.0)  # 6 ATR above entry
                take_profit_4 = current_price + (atr * 8.0)  # 8 ATR above entry
                
            elif signal_direction == 'short':
                entry_price = current_price
                stop_loss = current_price + (atr * 2.0)  # 2 ATR above entry
                take_profit_1 = current_price - (atr * 3.0)  # 3 ATR below entry
                take_profit_2 = current_price - (atr * 4.5)  # 4.5 ATR below entry
                take_profit_3 = current_price - (atr * 6.0)  # 6 ATR below entry
                take_profit_4 = current_price - (atr * 8.0)  # 8 ATR below entry
                
            else:
                # Neutral - no entry levels
                return current_price, None, [None, None, None, None], 0.0
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit_1 - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0.0
            
            return entry_price, stop_loss, [take_profit_1, take_profit_2, take_profit_3, take_profit_4], risk_reward_ratio
            
        except Exception as e:
            logger.error(f"Error calculating entry levels: {e}")
            return current_price, None, [None, None, None, None], 0.0
    
    def _calculate_position_size(self, risk_reward_ratio: float, confidence: float) -> float:
        """Calculate position size percentage based on risk/reward and confidence"""
        try:
            # Base position size
            base_size = 2.0  # 2% base position
            
            # Adjust based on risk/reward ratio
            if risk_reward_ratio >= 3.0:
                size_multiplier = 1.5
            elif risk_reward_ratio >= 2.0:
                size_multiplier = 1.0
            else:
                size_multiplier = 0.5
            
            # Adjust based on confidence
            if confidence >= 0.9:
                confidence_multiplier = 1.5
            elif confidence >= 0.85:
                confidence_multiplier = 1.0
            else:
                confidence_multiplier = 0.5
            
            position_size = base_size * size_multiplier * confidence_multiplier
            
            # Cap at 5% maximum position size
            return min(position_size, 5.0)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 2.0
    
    # Helper methods for technical analysis
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series([50.0] * len(prices))
    
    async def _calculate_rsi_async(self, df: pd.DataFrame, period: int = 14) -> List[float]:
        """Calculate RSI values for charting"""
        try:
            prices = df['close']
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50.0).tolist()
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return [50.0] * len(df)
    
    async def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> List[Dict]:
        """Calculate MACD values for charting"""
        try:
            prices = df['close']
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return [
                {
                    'macd': float(macd_line.iloc[i]),
                    'signal': float(signal_line.iloc[i]),
                    'histogram': float(histogram.iloc[i])
                }
                for i in range(len(df))
            ]
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return [{'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}] * len(df)
    
    async def _calculate_sma(self, df: pd.DataFrame, period: int = 20) -> List[float]:
        """Calculate Simple Moving Average for charting"""
        try:
            prices = df['close']
            sma = prices.rolling(window=period).mean()
            return sma.fillna(prices.iloc[0]).tolist()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return [df['close'].iloc[0]] * len(df)
    
    async def _calculate_ema(self, df: pd.DataFrame, period: int = 12) -> List[float]:
        """Calculate Exponential Moving Average for charting"""
        try:
            prices = df['close']
            ema = prices.ewm(span=period).mean()
            return ema.fillna(prices.iloc[0]).tolist()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return [df['close'].iloc[0]] * len(df)
    
    async def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> List[Dict]:
        """Calculate Bollinger Bands for charting"""
        try:
            prices = df['close']
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return [
                {
                    'upper': float(upper_band.iloc[i]),
                    'middle': float(sma.iloc[i]),
                    'lower': float(lower_band.iloc[i])
                }
                for i in range(len(df))
            ]
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return [{'upper': 0.0, 'middle': 0.0, 'lower': 0.0}] * len(df)
    
    def _calculate_macd_signal(self, prices: pd.Series) -> str:
        """Calculate MACD signal"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                return "bullish_crossover"
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                return "bearish_crossover"
            elif macd.iloc[-1] > signal.iloc[-1]:
                return "bullish"
            else:
                return "bearish"
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return "neutral"
    
    def _calculate_bollinger_position(self, prices: pd.Series) -> str:
        """Calculate Bollinger Bands position"""
        try:
            sma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            current_price = prices.iloc[-1]
            if current_price > upper_band.iloc[-1]:
                return "above_upper"
            elif current_price < lower_band.iloc[-1]:
                return "below_lower"
            else:
                return "within_bands"
        except Exception as e:
            logger.error(f"Error calculating Bollinger position: {e}")
            return "within_bands"
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            recent_highs = df['high'].tail(20).max()
            recent_lows = df['low'].tail(20).min()
            return {'support': float(recent_lows), 'resistance': float(recent_highs)}
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {'support': 0.0, 'resistance': 0.0}
    
    async def _calculate_atr(self, current_price: float, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            # Simplified ATR calculation
            return current_price * 0.02  # 2% of current price as approximation
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return current_price * 0.02
    
    # Helper methods for data retrieval
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # CCXT fetch_ticker is synchronous in our setup
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def _get_candlestick_data(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Get candlestick data"""
        try:
            # CCXT fetch_ohlcv is synchronous in our setup
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [
                {
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5]
                }
                for candle in ohlcv
            ]
        except Exception as e:
            logger.error(f"Error getting candlestick data: {e}")
            return []
    
    # Helper methods for classification
    def _classify_volatility(self, volatility_index: float) -> str:
        """Classify volatility level"""
        if volatility_index > 0.05:
            return "high"
        elif volatility_index > 0.025:
            return "medium"
        else:
            return "low"
    
    def _classify_trend(self, trend_strength: float) -> str:
        """Classify trend direction"""
        if trend_strength > 0.6:
            return "strong"
        elif trend_strength > 0.3:
            return "moderate"
        else:
            return "weak"
    
    # Helper methods for signal determination
    def _determine_signal_direction(self, pattern_analysis: Dict, technical_analysis: Dict, sentiment_analysis: Dict) -> str:
        """Determine signal direction"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Pattern analysis
        if 'bullish' in pattern_analysis['type'].lower():
            bullish_signals += 1
        elif 'bearish' in pattern_analysis['type'].lower():
            bearish_signals += 1
        
        # Technical analysis
        if technical_analysis['macd_signal'] == 'bullish':
            bullish_signals += 1
        elif technical_analysis['macd_signal'] == 'bearish':
            bearish_signals += 1
        
        # Sentiment analysis
        if sentiment_analysis['market_sentiment'] > 0.6:
            bullish_signals += 1
        elif sentiment_analysis['market_sentiment'] < 0.4:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return "long"
        elif bearish_signals > bullish_signals:
            return "short"
        else:
            return "neutral"
    
    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength"""
        if confidence >= 0.9:
            return "very_strong"
        elif confidence >= 0.85:
            return "strong"
        elif confidence >= 0.75:
            return "moderate"
        else:
            return "weak"
    
    # Helper methods for pattern combination
    async def _combine_pattern_analysis(self, advanced_patterns: List[PatternResult], 
                                      ml_patterns: List[MLPatternSignal],
                                      volume_patterns: List[VolumeEnhancedPatternSignal]) -> Dict[str, Any]:
        """Combine different pattern analysis results"""
        try:
            # Get the strongest pattern from each detector
            best_advanced = max(advanced_patterns, key=lambda x: x.confidence) if advanced_patterns else None
            best_ml = max(ml_patterns, key=lambda x: x.confidence) if ml_patterns else None
            best_volume = max(volume_patterns, key=lambda x: x.base_confidence) if volume_patterns else None
            
            # Combine confidences with weights
            weights = {'advanced': 0.4, 'ml': 0.35, 'volume': 0.25}
            combined_confidence = 0.0
            pattern_type = "none"
            strength = "weak"
            
            if best_advanced:
                combined_confidence += best_advanced.confidence * weights['advanced']
                pattern_type = best_advanced.pattern_type.value
                strength = best_advanced.strength.value
            
            if best_ml:
                combined_confidence += best_ml.confidence * weights['ml']
            
            if best_volume:
                combined_confidence += best_volume.base_confidence * weights['volume']
            
            return {
                'confidence': combined_confidence,
                'type': pattern_type,
                'strength': strength
            }
            
        except Exception as e:
            logger.error(f"Error combining pattern analysis: {e}")
            return self._get_default_pattern_analysis()
    
    # Helper methods for reasoning generation
    def _generate_analysis_reasoning(self, pattern_analysis: Dict, technical_analysis: Dict,
                                   sentiment_analysis: Dict, volume_analysis: Dict,
                                   market_regime_analysis: Dict, safe_entry_detected: bool) -> str:
        """Generate human-readable analysis reasoning"""
        try:
            reasoning_parts = []
            
            # Pattern reasoning
            if pattern_analysis['confidence'] > 0.7:
                reasoning_parts.append(f"Strong {pattern_analysis['type']} pattern detected")
            
            # Technical reasoning
            if technical_analysis['confidence'] > 0.7:
                reasoning_parts.append(f"Technical indicators align {technical_analysis['macd_signal']}")
            
            # Volume reasoning
            if volume_analysis['volume_ratio'] > 1.5:
                reasoning_parts.append("High volume confirms pattern")
            
            # Market regime reasoning
            reasoning_parts.append(f"Market in {market_regime_analysis['regime']} regime")
            
            # Safe entry reasoning
            if safe_entry_detected:
                reasoning_parts.append("Safe entry conditions met")
            else:
                reasoning_parts.append("Safe entry conditions not met")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Analysis reasoning unavailable"
    
    # Helper methods for confidence calculations
    def _calculate_technical_confidence(self, rsi: float, macd_signal: str, bollinger_position: str) -> float:
        """Calculate technical analysis confidence"""
        try:
            confidence = 0.5  # Base confidence
            
            # RSI contribution
            if 30 <= rsi <= 70:
                confidence += 0.1
            elif 20 <= rsi <= 80:
                confidence += 0.05
            
            # MACD contribution
            if macd_signal in ['bullish_crossover', 'bearish_crossover']:
                confidence += 0.2
            elif macd_signal in ['bullish', 'bearish']:
                confidence += 0.1
            
            # Bollinger Bands contribution
            if bollinger_position == 'within_bands':
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating technical confidence: {e}")
            return 0.5
    
    # Database storage methods
    async def _store_analysis_result(self, analysis_result: IntelligentAnalysisResult) -> bool:
        """Store analysis result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO comprehensive_analysis (
                        symbol, timeframe, timestamp, pattern_confidence, pattern_type, pattern_strength,
                        technical_confidence, rsi_value, macd_signal, bollinger_position, support_level, resistance_level,
                        sentiment_confidence, news_sentiment, social_sentiment, market_sentiment,
                        volume_confidence, volume_ratio, volume_positioning, order_book_imbalance,
                        market_regime_confidence, market_regime, volatility_level, trend_direction,
                        overall_confidence, risk_reward_ratio, safe_entry_detected,
                        entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, take_profit_4, position_size_percentage,
                        analysis_reasoning, no_safe_entry_reasons, signal_direction, signal_strength
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38)
                """, 
                analysis_result.symbol, analysis_result.timeframe, analysis_result.timestamp,
                analysis_result.pattern_confidence, analysis_result.pattern_type, analysis_result.pattern_strength,
                analysis_result.technical_confidence, analysis_result.rsi_value, analysis_result.macd_signal, analysis_result.bollinger_position,
                analysis_result.support_level, analysis_result.resistance_level,
                analysis_result.sentiment_confidence, analysis_result.news_sentiment, analysis_result.social_sentiment, analysis_result.market_sentiment,
                analysis_result.volume_confidence, analysis_result.volume_ratio, analysis_result.volume_positioning, analysis_result.order_book_imbalance,
                analysis_result.market_regime_confidence, analysis_result.market_regime, analysis_result.volatility_level, analysis_result.trend_direction,
                analysis_result.overall_confidence, analysis_result.risk_reward_ratio, analysis_result.safe_entry_detected,
                analysis_result.entry_price, analysis_result.stop_loss, analysis_result.take_profit_1, analysis_result.take_profit_2,
                analysis_result.take_profit_3, analysis_result.take_profit_4, analysis_result.position_size_percentage,
                analysis_result.analysis_reasoning, analysis_result.no_safe_entry_reasons, analysis_result.signal_direction, analysis_result.signal_strength
                )
            
            logger.info(f"✅ Analysis result stored for {analysis_result.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing analysis result: {e}")
            return False
    
    # Default analysis methods
    def _get_default_pattern_analysis(self) -> Dict[str, Any]:
        return {'confidence': 0.5, 'type': 'none', 'strength': 'weak'}
    
    def _get_default_technical_analysis(self) -> Dict[str, Any]:
        return {'confidence': 0.5, 'rsi': 50.0, 'macd_signal': 'neutral', 'bollinger_position': 'within_bands', 'support': 0.0, 'resistance': 0.0}
    
    def _get_default_sentiment_analysis(self) -> Dict[str, Any]:
        return {'confidence': 0.5, 'news_sentiment': 0.5, 'social_sentiment': 0.5, 'market_sentiment': 0.5}
    
    def _get_default_volume_analysis(self) -> Dict[str, Any]:
        return {'confidence': 0.5, 'volume_ratio': 1.0, 'positioning': 'stable', 'order_book_imbalance': 0.0}
    
    def _get_default_market_regime_analysis(self) -> Dict[str, Any]:
        return {'confidence': 0.5, 'regime': 'sideways', 'volatility_level': 'medium', 'trend_direction': 'weak'}
    
    def _get_default_overall_assessment(self) -> Dict[str, Any]:
        return {
            'confidence': 0.5, 'risk_reward_ratio': 0.0, 'safe_entry_detected': False,
            'reasoning': 'Analysis unavailable', 'signal_direction': 'neutral', 'signal_strength': 'weak'
        }
    
    async def _create_default_analysis_result(self, symbol: str, timeframe: str) -> IntelligentAnalysisResult:
        """Create default analysis result when analysis fails"""
        return IntelligentAnalysisResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.utcnow(),
            pattern_confidence=0.5,
            pattern_type="none",
            pattern_strength="weak",
            technical_confidence=0.5,
            rsi_value=50.0,
            macd_signal="neutral",
            bollinger_position="within_bands",
            support_level=0.0,
            resistance_level=0.0,
            sentiment_confidence=0.5,
            news_sentiment=0.5,
            social_sentiment=0.5,
            market_sentiment=0.5,
            volume_confidence=0.5,
            volume_ratio=1.0,
            volume_positioning="stable",
            order_book_imbalance=0.0,
            market_regime_confidence=0.5,
            market_regime="sideways",
            volatility_level="medium",
            trend_direction="weak",
            overall_confidence=0.5,
            risk_reward_ratio=0.0,
            safe_entry_detected=False,
            analysis_reasoning="Analysis failed - using default values",
            no_safe_entry_reasons=["Analysis unavailable"],
            signal_direction="neutral",
            signal_strength="weak"
        )
    
    async def close(self):
        """Close the analysis engine"""
        try:
            await self.advanced_pattern_detector.close()
            await self.data_collection_manager.stop_collection()
            logger.info("Intelligent Analysis Engine closed")
        except Exception as e:
            logger.error(f"Error closing analysis engine: {e}")

# Example usage
async def main():
    """Example usage of Intelligent Analysis Engine"""
    # Initialize database pool
    db_pool = await asyncpg.create_pool(
        host='postgres',
        port=5432,
        database='alphapulse',
        user='alpha_emon',
        password='Emon_@17711',
        min_size=5,
        max_size=20
    )
    
    # Initialize exchange
    exchange = ccxt.binance({
        'sandbox': False,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    })
    
    # Create analysis engine
    engine = IntelligentAnalysisEngine(db_pool, exchange)
    
    try:
        # Initialize
        await engine.initialize()
        
        # Analyze a symbol
        result = await engine.analyze_symbol('BTC/USDT', '1h')
        
        print(f"Analysis Result for {result.symbol}:")
        print(f"Overall Confidence: {result.overall_confidence:.1%}")
        print(f"Safe Entry Detected: {result.safe_entry_detected}")
        print(f"Signal Direction: {result.signal_direction}")
        print(f"Signal Strength: {result.signal_strength}")
        print(f"Risk/Reward Ratio: {result.risk_reward_ratio:.2f}")
        print(f"Analysis Reasoning: {result.analysis_reasoning}")
        
        if not result.safe_entry_detected:
            print(f"No Safe Entry Reasons: {', '.join(result.no_safe_entry_reasons)}")
        
    finally:
        await engine.close()
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
