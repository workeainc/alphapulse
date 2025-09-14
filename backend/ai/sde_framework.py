"""
Single-Decision Engine (SDE) Framework for AlphaPlus
Core framework for unified signal generation with calibrated confidence
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import asyncpg

# Import divergence analyzer
from .divergence_analyzer import AdvancedDivergenceAnalyzer, DivergenceAnalysis

logger = logging.getLogger(__name__)

# Import ML components for enhanced model heads
try:
    from .onnx_inference import ONNXInferenceEngine
    from .advanced_feature_engineering import AdvancedFeatureEngineering
    from .market_regime_classifier import MarketRegimeClassifier
    from .enhanced_sentiment_analysis import EnhancedSentimentAnalyzer
    from .advanced_model_fusion import AdvancedModelFusion, ModelPrediction
    from .advanced_calibration_system import AdvancedCalibrationSystem, CalibrationResult
    from .advanced_signal_quality_validator import AdvancedSignalQualityValidator, SignalQualityMetrics
    from .production_monitoring_system import ProductionMonitoringSystem
    ML_COMPONENTS_AVAILABLE = True
except ImportError:
    ML_COMPONENTS_AVAILABLE = False
    logger.warning("ML components not available for enhanced model heads")
    # Fallback definitions for missing classes
    @dataclass
    class CalibrationResult:
        """Fallback CalibrationResult when advanced_calibration_system is not available"""
        calibrated_probability: float
        calibration_method: str
        confidence_interval: Tuple[float, float]
        reliability_score: float
        method_performance: Dict[str, float]
    
    @dataclass
    class SignalQualityMetrics:
        """Fallback SignalQualityMetrics when advanced_signal_quality_validator is not available"""
        overall_score: float
        confidence_score: float
        reliability_score: float
        validation_passed: bool
        issues: List[str]

class SignalDirection(Enum):
    """Signal direction enumeration"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class ModelHead(Enum):
    """Model head types"""
    HEAD_A = "head_a"  # CatBoost
    HEAD_B = "head_b"  # Logistic
    HEAD_C = "head_c"  # Orderbook
    HEAD_D = "head_d"  # Rule-based

@dataclass
class ModelHeadResult:
    """Result from a single model head"""
    head_type: ModelHead
    direction: SignalDirection
    probability: float
    confidence: float
    features_used: List[str]
    reasoning: str

@dataclass
class ConsensusResult:
    """Model consensus result"""
    consensus_achieved: bool
    consensus_direction: Optional[SignalDirection]
    consensus_score: float
    agreeing_heads: List[ModelHead]
    disagreeing_heads: List[ModelHead]
    confidence_threshold: float

@dataclass
class ConfluenceScore:
    """Confluence score components"""
    zone_score: float  # 0-10
    htf_bias: float    # 0-10
    trigger_quality: float  # 0-10
    fvg_confluence: float   # 0-10
    ob_alignment: float     # 0-10
    total_score: float      # 0-10

@dataclass
class ExecutionQuality:
    """Execution quality metrics"""
    spread_ok: bool
    impact_cost: float
    volatility_regime: str
    liquidity_score: float
    execution_score: float  # 0-10

@dataclass
class NewsBlackout:
    """News blackout information"""
    blackout_active: bool
    event_type: Optional[str]
    event_impact: Optional[str]
    blackout_until: Optional[datetime]
    reason: str

@dataclass
class SignalLimits:
    """Signal limit information"""
    account_limit_reached: bool
    symbol_limit_reached: bool
    max_account_signals: int
    max_symbol_signals: int
    current_account_signals: int
    current_symbol_signals: int
    
    @property
    def allowed(self) -> bool:
        """Check if signal is allowed based on limits"""
        return not (self.account_limit_reached or self.symbol_limit_reached)

@dataclass
class TPStructure:
    """Take profit structure"""
    tp1_price: float
    tp2_price: float
    tp3_price: float
    tp4_price: float
    tp1_rr: float
    tp2_rr: float
    tp3_rr: float
    tp4_rr: float
    partial_exit_sizes: List[float]
    
    @property
    def tp_levels(self) -> List[float]:
        """Get all TP levels as a list"""
        return [self.tp1_price, self.tp2_price, self.tp3_price, self.tp4_price]

@dataclass
class SDEOutput:
    """Final SDE output"""
    direction: SignalDirection
    confidence: float
    stop_loss: float
    tp_structure: TPStructure
    confluence_score: float
    execution_quality: float
    divergence_analysis: Optional[DivergenceAnalysis]
    reasoning: List[str]
    risk_reward: float
    position_size: float

class SDEFramework:
    """Single-Decision Engine Framework"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        # Initialize divergence analyzer
        self.divergence_analyzer = AdvancedDivergenceAnalyzer(db_pool)
        
        # Initialize ML components for enhanced model heads
        if ML_COMPONENTS_AVAILABLE:
            try:
                self.onnx_inference = ONNXInferenceEngine()
                logger.info("✅ ONNX Inference Engine initialized")
            except Exception as e:
                logger.warning(f"⚠️ ONNX Inference Engine not available: {e}")
                self.onnx_inference = None
            
            try:
                self.feature_engineering = AdvancedFeatureEngineering({'db_pool': db_pool})
                logger.info("✅ Advanced Feature Engineering initialized")
            except Exception as e:
                logger.warning(f"⚠️ Advanced Feature Engineering not available: {e}")
                self.feature_engineering = None
            
            try:
                self.market_regime_classifier = MarketRegimeClassifier(db_pool)
                logger.info("✅ Market Regime Classifier initialized")
            except Exception as e:
                logger.warning(f"⚠️ Market Regime Classifier not available: {e}")
                self.market_regime_classifier = None
            
            try:
                # Try to initialize with None for redis_client for now
                self.sentiment_analyzer = EnhancedSentimentAnalyzer(db_pool, None)
                logger.info("✅ Enhanced Sentiment Analyzer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced Sentiment Analyzer not available: {e}")
                self.sentiment_analyzer = None
            
            try:
                self.advanced_model_fusion = AdvancedModelFusion(db_pool)
                logger.info("✅ Advanced Model Fusion created")
            except Exception as e:
                logger.warning(f"⚠️ Advanced Model Fusion not available: {e}")
                self.advanced_model_fusion = None
            
            try:
                self.advanced_calibration = AdvancedCalibrationSystem(db_pool)
                logger.info("✅ Advanced Calibration System created")
            except Exception as e:
                logger.warning(f"⚠️ Advanced Calibration System not available: {e}")
                self.advanced_calibration = None
            
            try:
                self.signal_quality_validator = AdvancedSignalQualityValidator(db_pool)
                logger.info("✅ Advanced Signal Quality Validator created")
            except Exception as e:
                logger.warning(f"⚠️ Advanced Signal Quality Validator not available: {e}")
                self.signal_quality_validator = None
            
            try:
                self.production_monitoring = ProductionMonitoringSystem(db_pool)
                logger.info("✅ Production Monitoring System created")
            except Exception as e:
                logger.warning(f"⚠️ Production Monitoring System not available: {e}")
                self.production_monitoring = None
        else:
            self.onnx_inference = None
            self.feature_engineering = None
            self.market_regime_classifier = None
            self.sentiment_analyzer = None
            self.advanced_calibration = None
            self.signal_quality_validator = None
            self.production_monitoring = None
        
        # Model head configurations
        self.model_head_config = {
            'head_a': {
                'model_name': 'catboost_technical',
                'features': ['rsi', 'macd', 'volume', 'bollinger', 'ema', 'atr'],
                'confidence_threshold': 0.70,
                'weight': 0.4
            },
            'head_b': {
                'model_name': 'logistic_sentiment',
                'features': ['sentiment_score', 'news_impact', 'social_sentiment', 'fear_greed'],
                'confidence_threshold': 0.65,
                'weight': 0.2
            },
            'head_c': {
                'model_name': 'tree_orderflow',
                'features': ['volume_delta', 'orderbook_imbalance', 'liquidity_score', 'spread'],
                'confidence_threshold': 0.60,
                'weight': 0.2
            },
            'head_d': {
                'model_name': 'rule_based',
                'features': ['zone_score', 'structure_score', 'pattern_score', 'confluence'],
                'confidence_threshold': 0.75,
                'weight': 0.2
            }
        }
        
        # Consensus configuration
        self.consensus_config = {
            'min_agreeing_heads': 3,
            'min_probability': 0.70,
            'confidence_threshold': 0.85
        }
        
        # Confluence configuration
        self.confluence_config = {
            'min_total_score': 8.0,
            'zone_weight': 0.25,
            'htf_weight': 0.20,
            'trigger_weight': 0.25,
            'fvg_weight': 0.15,
            'ob_weight': 0.15
        }
        
        # Execution quality configuration
        self.execution_config = {
            'max_spread_atr_ratio': 0.12,
            'max_impact_atr_ratio': 0.15,
            'min_liquidity_score': 0.7
        }
        
        # News blackout configuration
        self.blackout_config = {
            'blackout_window_minutes': 15,
            'min_event_impact': 'medium'
        }
        
        # Signal limits configuration
        self.limits_config = {
            'max_account_signals': 3,
            'max_symbol_signals': 1,
            'max_system_signals': 10
        }
        
        logger.info("🚀 SDE Framework initialized with divergence analyzer")
    
    async def _initialize_phase8_calibration(self):
        """Initialize Phase 8 advanced calibration components"""
        try:
            if self.advanced_calibration:
                # Initialize advanced calibration system
                await self.advanced_calibration.initialize()
                
                # Test Phase 8 calibration capabilities
                logger.info("🧪 Testing Phase 8 advanced calibration...")
                
                # Test calibration with mock data
                test_features = {
                    'rsi': 65.0,
                    'volume_ratio': 1.3,
                    'sentiment_score': 0.7,
                    'market_regime': 'bullish'
                }
                
                calibration_result = await self.advanced_calibration.calibrate_probability(
                    0.75, test_features, 'BTCUSDT', '1h', 'bullish'
                )
                
                if calibration_result:
                    improvement = calibration_result.calibrated_probability - 0.75
                    logger.info(f"✅ Calibration test: 0.75 → {calibration_result.calibrated_probability:.3f} "
                              f"(Δ{improvement:+.3f})")
                    logger.info(f"   Method: {calibration_result.calibration_method}")
                    logger.info(f"   Reliability: {calibration_result.reliability_score:.3f}")
                
                # Test dynamic thresholds
                await self._test_dynamic_thresholds()
                
                logger.info("🎉 Phase 8 advanced calibration initialized successfully")
            else:
                logger.warning("⚠️ Advanced calibration not available for Phase 8 initialization")
                
        except Exception as e:
            logger.error(f"❌ Phase 8 initialization failed: {e}")
    
    async def _test_dynamic_thresholds(self):
        """Test dynamic threshold configuration"""
        try:
            async with self.db_pool.acquire() as conn:
                # Test different market regimes
                test_cases = [
                    ('bullish', 'low'),
                    ('bearish', 'medium'),
                    ('volatile', 'high')
                ]
                
                for regime, volatility in test_cases:
                    threshold = await conn.fetchrow("""
                        SELECT min_confidence_threshold, min_consensus_heads, min_probability_threshold
                        FROM sde_dynamic_thresholds
                        WHERE market_regime = $1 AND volatility_level = $2 AND is_active = TRUE
                    """, regime, volatility)
                    
                    if threshold:
                        logger.info(f"✅ {regime} {volatility}: "
                                  f"Confidence={threshold['min_confidence_threshold']:.3f}, "
                                  f"Heads={threshold['min_consensus_heads']}")
                
        except Exception as e:
            logger.error(f"❌ Dynamic threshold test failed: {e}")
    
    async def _initialize_phase6_features(self):
        """Initialize Phase 6 advanced feature engineering components"""
        try:
            if self.feature_engineering:
                # Initialize advanced feature engineering
                await self.feature_engineering.initialize()
                
                # Test Phase 6 feature creation
                logger.info("🧪 Testing Phase 6 feature creation...")
                
                # Test multi-timeframe features
                mtf_result = await self.feature_engineering.create_multitimeframe_features('BTCUSDT', '1h')
                if 'error' not in mtf_result:
                    logger.info(f"✅ Multi-timeframe features: {mtf_result.get('feature_count', 0)} features")
                
                # Test market regime features
                regime_result = await self.feature_engineering.create_market_regime_features('BTCUSDT', '1h')
                if 'error' not in regime_result:
                    logger.info(f"✅ Market regime features: {regime_result.get('feature_count', 0)} features")
                
                # Test news sentiment features
                sentiment_result = await self.feature_engineering.create_news_sentiment_features('BTCUSDT')
                if 'error' not in sentiment_result:
                    logger.info(f"✅ News sentiment features: {sentiment_result.get('feature_count', 0)} features")
                
                # Test volume profile features
                volume_result = await self.feature_engineering.create_volume_profile_features('BTCUSDT', '1h')
                if 'error' not in volume_result:
                    logger.info(f"✅ Volume profile features: {volume_result.get('feature_count', 0)} features")
                
                logger.info("🎉 Phase 6 advanced features initialized successfully")
            else:
                logger.warning("⚠️ Feature engineering not available for Phase 6 initialization")
                
        except Exception as e:
            logger.error(f"❌ Phase 6 initialization failed: {e}")
    
    async def _initialize_phase7_fusion(self):
        """Initialize Phase 7 advanced model fusion components"""
        try:
            if self.advanced_model_fusion:
                # Initialize advanced model fusion
                await self.advanced_model_fusion.initialize()
                
                # Test Phase 7 fusion capabilities
                logger.info("🧪 Testing Phase 7 model fusion...")
                
                # Create mock predictions for testing
                mock_predictions = [
                    ModelPrediction(
                        model_name='catboost',
                        prediction=0.75,
                        confidence=0.85,
                        features={'rsi': 0.6, 'macd': 0.7},
                        timestamp=datetime.now()
                    ),
                    ModelPrediction(
                        model_name='logistic',
                        prediction=0.68,
                        confidence=0.78,
                        features={'sentiment': 0.5, 'volume': 0.8},
                        timestamp=datetime.now()
                    ),
                    ModelPrediction(
                        model_name='decision_tree',
                        prediction=0.72,
                        confidence=0.82,
                        features={'pattern': 0.7, 'structure': 0.6},
                        timestamp=datetime.now()
                    ),
                    ModelPrediction(
                        model_name='rule_based',
                        prediction=0.80,
                        confidence=0.90,
                        features={'zone': 0.8, 'confluence': 0.7},
                        timestamp=datetime.now()
                    )
                ]
                
                # Test ensemble fusion
                ensemble_result = await self.advanced_model_fusion.fuse_predictions(
                    mock_predictions, 'BTCUSDT', '1h'
                )
                
                if ensemble_result:
                    logger.info(f"✅ Ensemble fusion: {ensemble_result.signal_direction} "
                              f"confidence={ensemble_result.confidence_score:.3f}")
                
                # Test performance tracking
                await self.advanced_model_fusion.update_model_performance(
                    'catboost', 'BTCUSDT', '1h', 0.75, 1.0, datetime.now()
                )
                
                logger.info("🎉 Phase 7 advanced model fusion initialized successfully")
            else:
                logger.warning("⚠️ Advanced model fusion not available for Phase 7 initialization")
                
        except Exception as e:
            logger.error(f"❌ Phase 7 initialization failed: {e}")
    
    async def check_model_consensus(self, model_results: List[ModelHeadResult]) -> ConsensusResult:
        """Check if model heads reach consensus"""
        try:
            if len(model_results) < 4:
                logger.warning("Insufficient model heads for consensus check")
                return ConsensusResult(
                    consensus_achieved=False,
                    consensus_direction=None,
                    consensus_score=0.0,
                    agreeing_heads=[],
                    disagreeing_heads=[],
                    confidence_threshold=self.consensus_config['confidence_threshold']
                )
            
            # Count agreeing heads
            agreeing_heads = []
            disagreeing_heads = []
            
            for result in model_results:
                if (result.probability >= self.consensus_config['min_probability'] and 
                    result.confidence >= self.consensus_config['confidence_threshold']):
                    agreeing_heads.append(result.head_type)
                else:
                    disagreeing_heads.append(result.head_type)
            
            # Check consensus
            consensus_achieved = len(agreeing_heads) >= self.consensus_config['min_agreeing_heads']
            
            if consensus_achieved:
                # Check direction agreement
                directions = [r.direction for r in model_results if r.head_type in agreeing_heads]
                unique_directions = set(directions)
                
                if len(unique_directions) == 1:
                    consensus_direction = list(unique_directions)[0]
                    consensus_score = np.mean([r.probability for r in model_results if r.head_type in agreeing_heads])
                else:
                    consensus_achieved = False
                    consensus_direction = None
                    consensus_score = 0.0
            else:
                consensus_direction = None
                consensus_score = 0.0
            
            # Store consensus tracking
            await self._store_consensus_tracking(model_results, consensus_achieved, consensus_score)
            
            return ConsensusResult(
                consensus_achieved=consensus_achieved,
                consensus_direction=consensus_direction,
                consensus_score=consensus_score,
                agreeing_heads=agreeing_heads,
                disagreeing_heads=disagreeing_heads,
                confidence_threshold=self.consensus_config['confidence_threshold']
            )
            
        except Exception as e:
            logger.error(f"❌ Model consensus check failed: {e}")
            return ConsensusResult(
                consensus_achieved=False,
                consensus_direction=None,
                consensus_score=0.0,
                agreeing_heads=[],
                disagreeing_heads=[],
                confidence_threshold=self.consensus_config['confidence_threshold']
            )
    
    async def calculate_confluence_score(self, df: pd.DataFrame, symbol: str, timeframe: str) -> ConfluenceScore:
        """Calculate confluence score across multiple components"""
        try:
            # Calculate zone score
            zone_score = await self._calculate_zone_score(df, symbol, timeframe)
            
            # Calculate HTF bias
            htf_bias = await self._calculate_htf_bias(df, symbol, timeframe)
            
            # Calculate trigger quality
            trigger_quality = await self._calculate_trigger_quality(df, symbol, timeframe)
            
            # Calculate FVG confluence
            fvg_confluence = await self._calculate_fvg_confluence(df, symbol, timeframe)
            
            # Calculate orderbook alignment
            ob_alignment = await self._calculate_ob_alignment(df, symbol, timeframe)
            
            # Calculate weighted total score
            total_score = (
                zone_score * self.confluence_config['zone_weight'] +
                htf_bias * self.confluence_config['htf_weight'] +
                trigger_quality * self.confluence_config['trigger_weight'] +
                fvg_confluence * self.confluence_config['fvg_weight'] +
                ob_alignment * self.confluence_config['ob_weight']
            )
            
            return ConfluenceScore(
                zone_score=zone_score,
                htf_bias=htf_bias,
                trigger_quality=trigger_quality,
                fvg_confluence=fvg_confluence,
                ob_alignment=ob_alignment,
                total_score=total_score
            )
            
        except Exception as e:
            logger.error(f"❌ Confluence score calculation failed: {e}")
            return ConfluenceScore(
                zone_score=0.0,
                htf_bias=0.0,
                trigger_quality=0.0,
                fvg_confluence=0.0,
                ob_alignment=0.0,
                total_score=0.0
            )
    
    async def check_execution_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> ExecutionQuality:
        """Check execution quality metrics"""
        try:
            # Calculate ATR for spread comparison
            atr = self._calculate_atr(df, 14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0.001
            
            # Check spread
            spread = await self._get_current_spread(symbol)
            spread_ok = spread <= (current_atr * self.execution_config['max_spread_atr_ratio'])
            
            # Calculate impact cost
            impact_cost = await self._calculate_impact_cost(symbol, timeframe)
            
            # Determine volatility regime
            volatility_regime = self._determine_volatility_regime(df)
            
            # Calculate liquidity score
            liquidity_score = await self._calculate_liquidity_score(symbol, timeframe)
            
            # Calculate execution score
            execution_score = self._calculate_execution_score(
                spread_ok, impact_cost, volatility_regime, liquidity_score, current_atr
            )
            
            return ExecutionQuality(
                spread_ok=spread_ok,
                impact_cost=impact_cost,
                volatility_regime=volatility_regime,
                liquidity_score=liquidity_score,
                execution_score=execution_score
            )
            
        except Exception as e:
            logger.error(f"❌ Execution quality check failed: {e}")
            return ExecutionQuality(
                spread_ok=False,
                impact_cost=0.0,
                volatility_regime="unknown",
                liquidity_score=0.0,
                execution_score=0.0
            )
    
    async def assess_execution_quality(self, market_data: Dict[str, Any]) -> ExecutionQuality:
        """Assess execution quality from market data"""
        try:
            # Extract market data components
            spread = market_data.get('spread', 0.0001)
            volatility = market_data.get('volatility', 0.3)
            liquidity = market_data.get('liquidity', 0.8)
            news_impact = market_data.get('news_impact', 0.1)
            
            # Determine spread quality
            spread_ok = spread <= 0.0002  # 2 pips max
            
            # Calculate impact cost based on volatility and liquidity
            impact_cost = (volatility * 0.5) + (1 - liquidity) * 0.3
            
            # Determine volatility regime
            if volatility < 0.2:
                volatility_regime = "low"
            elif volatility < 0.5:
                volatility_regime = "medium"
            else:
                volatility_regime = "high"
            
            # Calculate liquidity score
            liquidity_score = max(0.0, min(1.0, liquidity))
            
            # Calculate execution score (0-10)
            execution_score = 10.0
            if not spread_ok:
                execution_score -= 3.0
            if impact_cost > 0.5:
                execution_score -= 2.0
            if volatility_regime == "high":
                execution_score -= 1.0
            if liquidity_score < 0.5:
                execution_score -= 2.0
            if news_impact > 0.3:
                execution_score -= 1.0
            
            execution_score = max(0.0, min(10.0, execution_score))
            
            return ExecutionQuality(
                spread_ok=spread_ok,
                impact_cost=impact_cost,
                volatility_regime=volatility_regime,
                liquidity_score=liquidity_score,
                execution_score=execution_score
            )
            
        except Exception as e:
            logger.error(f"❌ Execution quality assessment failed: {e}")
            return ExecutionQuality(
                spread_ok=False,
                impact_cost=0.0,
                volatility_regime="unknown",
                liquidity_score=0.0,
                execution_score=0.0
            )
    
    async def check_news_blackout(self, symbol: str, current_time: datetime = None) -> NewsBlackout:
        """Check for news blackout conditions"""
        try:
            if current_time is None:
                current_time = datetime.now()
            
            async with self.db_pool.acquire() as conn:
                # Check for upcoming events within blackout window
                blackout_window = self.blackout_config.get('blackout_window_minutes', 15)
                future_blackout = await conn.fetch(f"""
                    SELECT event_type, event_impact, event_title, start_time
                    FROM sde_news_blackout
                    WHERE symbol = $1 AND blackout_active = true
                        AND start_time BETWEEN $2 AND $2 + INTERVAL '{blackout_window} minutes'
                    ORDER BY event_impact DESC
                    LIMIT 1
                """, symbol, current_time)
                
                if future_blackout:
                    record = future_blackout[0]
                    return NewsBlackout(
                        blackout_active=True,
                        event_type=record['event_type'],
                        event_impact=record['event_impact'],
                        blackout_until=record['start_time'] + timedelta(minutes=blackout_window),
                        reason=f"Upcoming {record['event_impact']} event: {record['event_title']}"
                    )
                
                return NewsBlackout(
                    blackout_active=False,
                    event_type=None,
                    event_impact=None,
                    blackout_until=None,
                    reason="No upcoming events"
                )
                
        except Exception as e:
            logger.error(f"❌ News blackout check failed: {e}")
            return NewsBlackout(
                blackout_active=False,
                event_type=None,
                event_impact=None,
                blackout_until=None,
                reason=f"Error checking blackout: {e}"
            )
    
    async def check_signal_limits(self, account_id: str, symbol: str, timeframe: str = None) -> SignalLimits:
        """Check signal limits"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check account-level signals
                account_signals = await conn.fetchval("""
                    SELECT COUNT(*) FROM signals
                    WHERE account_id = $1 AND outcome = 'pending'
                """, account_id)
                
                # Check symbol-level signals
                symbol_signals = await conn.fetchval("""
                    SELECT COUNT(*) FROM signals
                    WHERE symbol = $1 AND outcome = 'pending'
                """, symbol)
                
                return SignalLimits(
                    account_limit_reached=account_signals >= self.limits_config['max_account_signals'],
                    symbol_limit_reached=symbol_signals >= self.limits_config['max_symbol_signals'],
                    max_account_signals=self.limits_config['max_account_signals'],
                    max_symbol_signals=self.limits_config['max_symbol_signals'],
                    current_account_signals=account_signals,
                    current_symbol_signals=symbol_signals
                )
                
        except Exception as e:
            logger.error(f"❌ Signal limits check failed: {e}")
            return SignalLimits(
                account_limit_reached=False,
                symbol_limit_reached=False,
                max_account_signals=self.limits_config['max_account_signals'],
                max_symbol_signals=self.limits_config['max_symbol_signals'],
                current_account_signals=0,
                current_symbol_signals=0
            )
    
    async def calculate_tp_structure(self, entry_price: float, stop_loss: float, 
                                   direction: SignalDirection, df: pd.DataFrame) -> TPStructure:
        """Calculate four-tier take profit structure"""
        try:
            atr = self._calculate_atr(df, 14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0.001
            
            if direction == SignalDirection.LONG:
                # Long position TP structure
                tp1_price = entry_price + (current_atr * 0.5)  # 0.5R
                tp2_price = entry_price + (current_atr * 1.0)  # 1.0R
                tp3_price = entry_price + (current_atr * 2.0)  # 2.0R
                tp4_price = entry_price + (current_atr * 4.0)  # 4.0R
            else:
                # Short position TP structure
                tp1_price = entry_price - (current_atr * 0.5)  # 0.5R
                tp2_price = entry_price - (current_atr * 1.0)  # 1.0R
                tp3_price = entry_price - (current_atr * 2.0)  # 2.0R
                tp4_price = entry_price - (current_atr * 4.0)  # 4.0R
            
            # Calculate risk-reward ratios
            risk = abs(entry_price - stop_loss)
            tp1_rr = abs(tp1_price - entry_price) / risk
            tp2_rr = abs(tp2_price - entry_price) / risk
            tp3_rr = abs(tp3_price - entry_price) / risk
            tp4_rr = abs(tp4_price - entry_price) / risk
            
            return TPStructure(
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                tp3_price=tp3_price,
                tp4_price=tp4_price,
                tp1_rr=tp1_rr,
                tp2_rr=tp2_rr,
                tp3_rr=tp3_rr,
                tp4_rr=tp4_rr,
                partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]  # 25% each
            )
            
        except Exception as e:
            logger.error(f"❌ TP structure calculation failed: {e}")
            return TPStructure(
                tp1_price=entry_price,
                tp2_price=entry_price,
                tp3_price=entry_price,
                tp4_price=entry_price,
                tp1_rr=0.0,
                tp2_rr=0.0,
                tp3_rr=0.0,
                tp4_rr=0.0,
                partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]
            )
    
    async def analyze_divergences(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[DivergenceAnalysis]:
        """Analyze divergences using the divergence analyzer"""
        try:
            return await self.divergence_analyzer.analyze_divergences(df, symbol, timeframe)
        except Exception as e:
            logger.error(f"❌ Divergence analysis failed: {e}")
            return None
    
    async def generate_sde_output(self, model_results: List[ModelHeadResult], df: pd.DataFrame,
                                symbol: str, timeframe: str, account_id: str) -> SDEOutput:
        """Generate final SDE output"""
        try:
            # Check model consensus
            consensus = await self.check_model_consensus(model_results)
            
            if not consensus.consensus_achieved:
                return SDEOutput(
                    direction=SignalDirection.FLAT,
                    confidence=0.0,
                    stop_loss=0.0,
                    tp_structure=TPStructure(
                        tp1_price=0.0, tp2_price=0.0, tp3_price=0.0, tp4_price=0.0,
                        tp1_rr=0.0, tp2_rr=0.0, tp3_rr=0.0, tp4_rr=0.0,
                        partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]
                    ),
                    confluence_score=0.0,
                    execution_quality=0.0,
                    divergence_analysis=None,
                    reasoning=["No model consensus achieved"],
                    risk_reward=0.0,
                    position_size=0.0
                )
            
            # Calculate confluence score
            confluence = await self.calculate_confluence_score(df, symbol, timeframe)
            
            if confluence.total_score < self.confluence_config['min_total_score']:
                return SDEOutput(
                    direction=SignalDirection.FLAT,
                    confidence=0.0,
                    stop_loss=0.0,
                    tp_structure=TPStructure(
                        tp1_price=0.0, tp2_price=0.0, tp3_price=0.0, tp4_price=0.0,
                        tp1_rr=0.0, tp2_rr=0.0, tp3_rr=0.0, tp4_rr=0.0,
                        partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]
                    ),
                    confluence_score=confluence.total_score,
                    execution_quality=0.0,
                    divergence_analysis=None,
                    reasoning=[f"Insufficient confluence score: {confluence.total_score}"],
                    risk_reward=0.0,
                    position_size=0.0
                )
            
            # Check execution quality
            execution = await self.check_execution_quality(df, symbol, timeframe)
            
            if not execution.spread_ok:
                return SDEOutput(
                    direction=SignalDirection.FLAT,
                    confidence=0.0,
                    stop_loss=0.0,
                    tp_structure=TPStructure(
                        tp1_price=0.0, tp2_price=0.0, tp3_price=0.0, tp4_price=0.0,
                        tp1_rr=0.0, tp2_rr=0.0, tp3_rr=0.0, tp4_rr=0.0,
                        partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]
                    ),
                    confluence_score=confluence.total_score,
                    execution_quality=execution.execution_score,
                    divergence_analysis=None,
                    reasoning=["Spread too wide for execution"],
                    risk_reward=0.0,
                    position_size=0.0
                )
            
            # Check news blackout
            blackout = await self.check_news_blackout(symbol)
            
            if blackout.blackout_active:
                return SDEOutput(
                    direction=SignalDirection.FLAT,
                    confidence=0.0,
                    stop_loss=0.0,
                    tp_structure=TPStructure(
                        tp1_price=0.0, tp2_price=0.0, tp3_price=0.0, tp4_price=0.0,
                        tp1_rr=0.0, tp2_rr=0.0, tp3_rr=0.0, tp4_rr=0.0,
                        partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]
                    ),
                    confluence_score=confluence.total_score,
                    execution_quality=execution.execution_score,
                    divergence_analysis=None,
                    reasoning=[f"News blackout active: {blackout.reason}"],
                    risk_reward=0.0,
                    position_size=0.0
                )
            
            # Check signal limits
            limits = await self.check_signal_limits(symbol, account_id)
            
            if limits.account_limit_reached or limits.symbol_limit_reached:
                return SDEOutput(
                    direction=SignalDirection.FLAT,
                    confidence=0.0,
                    stop_loss=0.0,
                    tp_structure=TPStructure(
                        tp1_price=0.0, tp2_price=0.0, tp3_price=0.0, tp4_price=0.0,
                        tp1_rr=0.0, tp2_rr=0.0, tp3_rr=0.0, tp4_rr=0.0,
                        partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]
                    ),
                    confluence_score=confluence.total_score,
                    execution_quality=execution.execution_score,
                    divergence_analysis=None,
                    reasoning=["Signal limits reached"],
                    risk_reward=0.0,
                    position_size=0.0
                )
            
            # Analyze divergences
            divergence_analysis = await self.analyze_divergences(df, symbol, timeframe)
            
            # Calculate entry price and stop loss
            current_price = df['close'].iloc[-1]
            atr = self._calculate_atr(df, 14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0.001
            
            if consensus.consensus_direction == SignalDirection.LONG:
                entry_price = current_price
                stop_loss = current_price - (current_atr * 1.5)
            else:
                entry_price = current_price
                stop_loss = current_price + (current_atr * 1.5)
            
            # Calculate TP structure
            tp_structure = await self.calculate_tp_structure(entry_price, stop_loss, consensus.consensus_direction, df)
            
            # Calculate risk-reward
            risk = abs(entry_price - stop_loss)
            reward = abs(tp_structure.tp2_price - entry_price)  # Use TP2 as primary target
            risk_reward = reward / risk if risk > 0 else 0.0
            
            # Calculate position size (simplified)
            position_size = 0.1  # 10% of account
            
            # Generate reasoning
            reasoning = [
                f"Model consensus: {len(consensus.agreeing_heads)}/4 heads agree",
                f"Confluence score: {confluence.total_score:.2f}/10",
                f"Execution quality: {execution.execution_score:.2f}/10",
                f"Risk-reward: {risk_reward:.2f}"
            ]
            
            if divergence_analysis and divergence_analysis.overall_confidence > 0.5:
                reasoning.append(f"Divergence detected: {divergence_analysis.overall_confidence:.2f} confidence")
            
            return SDEOutput(
                direction=consensus.consensus_direction,
                confidence=consensus.consensus_score,
                stop_loss=stop_loss,
                tp_structure=tp_structure,
                confluence_score=confluence.total_score,
                execution_quality=execution.execution_score,
                divergence_analysis=divergence_analysis,
                reasoning=reasoning,
                risk_reward=risk_reward,
                position_size=position_size
            )
            
        except Exception as e:
            logger.error(f"❌ SDE output generation failed: {e}")
            return SDEOutput(
                direction=SignalDirection.FLAT,
                confidence=0.0,
                stop_loss=0.0,
                tp_structure=TPStructure(
                    tp1_price=0.0, tp2_price=0.0, tp3_price=0.0, tp4_price=0.0,
                    tp1_rr=0.0, tp2_rr=0.0, tp3_rr=0.0, tp4_rr=0.0,
                    partial_exit_sizes=[0.25, 0.25, 0.25, 0.25]
                ),
                confluence_score=0.0,
                execution_quality=0.0,
                divergence_analysis=None,
                reasoning=[f"Error generating SDE output: {e}"],
                risk_reward=0.0,
                position_size=0.0
            )
    
    # Helper methods
    async def _store_consensus_tracking(self, model_results: List[ModelHeadResult],
                                      consensus_achieved: bool, consensus_score: float) -> None:
        """Store consensus tracking data"""
        try:
            async with self.db_pool.acquire() as conn:
                # Calculate agreeing heads count
                agreeing_heads = sum(1 for result in model_results if result.probability >= 0.6)
                
                # Determine consensus direction
                directions = [result.direction.value for result in model_results if result.probability >= 0.6]
                consensus_direction = max(set(directions), key=directions.count) if directions else 'flat'
                
                await conn.execute("""
                    INSERT INTO sde_model_consensus_tracking (
                        symbol, timeframe, head_a_probability, head_a_confidence, head_a_direction,
                        head_b_probability, head_b_confidence, head_b_direction,
                        head_c_probability, head_c_confidence, head_c_direction,
                        head_d_probability, head_d_confidence, head_d_direction,
                        consensus_achieved, consensus_probability, consensus_direction,
                        agreeing_heads_count, min_agreeing_heads, min_head_probability,
                        timestamp, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)
                """, 
                'EURUSD', '1h',  # Default values for now
                model_results[0].probability if len(model_results) > 0 else 0.0,
                model_results[0].confidence if len(model_results) > 0 else 0.0,
                model_results[0].direction.value if len(model_results) > 0 else 'flat',
                model_results[1].probability if len(model_results) > 1 else 0.0,
                model_results[1].confidence if len(model_results) > 1 else 0.0,
                model_results[1].direction.value if len(model_results) > 1 else 'flat',
                model_results[2].probability if len(model_results) > 2 else 0.0,
                model_results[2].confidence if len(model_results) > 2 else 0.0,
                model_results[2].direction.value if len(model_results) > 2 else 'flat',
                model_results[3].probability if len(model_results) > 3 else 0.0,
                model_results[3].confidence if len(model_results) > 3 else 0.0,
                model_results[3].direction.value if len(model_results) > 3 else 'flat',
                consensus_achieved, consensus_score, consensus_direction,
                agreeing_heads, 2, 0.6,  # min_agreeing_heads=2, min_head_probability=0.6
                datetime.now(), datetime.now())
                
        except Exception as e:
            logger.error(f"❌ Failed to store consensus tracking: {e}")
    
    async def create_enhanced_model_head_results(self, 
                                               analysis_results: Dict[str, Any],
                                               market_data: Dict[str, Any],
                                               symbol: str,
                                               timeframe: str) -> List[ModelHeadResult]:
        """Create enhanced model head results with real ML models"""
        try:
            model_results = []
            
            # Head A: Technical ML (CatBoost)
            head_a = await self._create_head_a_technical(analysis_results, market_data, symbol, timeframe)
            model_results.append(head_a)
            
            # Head B: Sentiment ML (Logistic)
            head_b = await self._create_head_b_sentiment(analysis_results, market_data, symbol, timeframe)
            model_results.append(head_b)
            
            # Head C: Order Flow ML (Tree)
            head_c = await self._create_head_c_orderflow(analysis_results, market_data, symbol, timeframe)
            model_results.append(head_c)
            
            # Head D: Rule-based (Deterministic)
            head_d = await self._create_head_d_rulebased(analysis_results, market_data, symbol, timeframe)
            model_results.append(head_d)
            
            logger.info(f"✅ Created {len(model_results)} enhanced model head results for {symbol}")
            return model_results
            
        except Exception as e:
            logger.error(f"❌ Failed to create enhanced model head results: {e}")
            return await self._create_fallback_model_heads(analysis_results)
    
    async def _create_head_a_technical(self, 
                                     analysis_results: Dict[str, Any],
                                     market_data: Dict[str, Any],
                                     symbol: str,
                                     timeframe: str) -> ModelHeadResult:
        """Create Head A: Technical ML model result"""
        try:
            config = self.model_head_config['head_a']
            
            # Extract technical features
            features = {
                'rsi': analysis_results.get('rsi', 50.0),
                'macd': analysis_results.get('macd_signal', 0.0),
                'volume': analysis_results.get('volume_ratio', 1.0),
                'bollinger': analysis_results.get('bollinger_position', 0.0),
                'ema': analysis_results.get('ema_trend', 0.0),
                'atr': analysis_results.get('atr_percentile', 50.0)
            }
            
            # Use ONNX inference if available
            if self.onnx_inference and self.feature_engineering:
                try:
                    # Create feature vector - use existing method or fallback
                    if hasattr(self.feature_engineering, 'create_technical_features'):
                        feature_vector = await self.feature_engineering.create_technical_features(
                            features, symbol, timeframe
                        )
                    else:
                        # Fallback: create simple feature vector
                        feature_vector = np.array([
                            features['rsi'], features['macd'], features['volume'],
                            features['bollinger'], features['ema'], features['atr']
                        ]).reshape(1, -1)
                    
                    # Get ONNX prediction
                    prediction = await self.onnx_inference.predict(
                        config['model_name'], feature_vector
                    )
                    
                    probability = prediction.get('probability', 0.5)
                    direction = self._convert_probability_to_direction(probability)
                    
                except Exception as e:
                    logger.warning(f"ONNX inference failed for Head A: {e}")
                    # Fallback to analysis results
                    technical_confidence = analysis_results.get('technical_confidence', 0.5)
                    technical_direction = analysis_results.get('technical_direction', 'neutral')
                    
                    probability = technical_confidence
                    direction = SignalDirection.LONG if technical_direction == 'long' else \
                               SignalDirection.SHORT if technical_direction == 'short' else SignalDirection.FLAT
            else:
                # Fallback to analysis results
                technical_confidence = analysis_results.get('technical_confidence', 0.5)
                technical_direction = analysis_results.get('technical_direction', 'neutral')
                
                probability = technical_confidence
                direction = SignalDirection.LONG if technical_direction == 'long' else \
                           SignalDirection.SHORT if technical_direction == 'short' else SignalDirection.FLAT
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_A,
                direction=direction,
                probability=probability,
                confidence=probability,
                features_used=config['features'],
                reasoning=f"Technical ML model ({config['model_name']}) prediction"
            )
            
        except Exception as e:
            logger.error(f"❌ Head A creation failed: {e}")
            return self._create_default_head(ModelHead.HEAD_A, "Technical analysis failed")
    
    async def _create_head_b_sentiment(self, 
                                     analysis_results: Dict[str, Any],
                                     market_data: Dict[str, Any],
                                     symbol: str,
                                     timeframe: str) -> ModelHeadResult:
        """Create Head B: Sentiment ML model result"""
        try:
            config = self.model_head_config['head_b']
            
            # Extract sentiment features
            features = {
                'sentiment_score': analysis_results.get('sentiment_score', 0.5),
                'news_impact': analysis_results.get('news_impact', 0.0),
                'social_sentiment': analysis_results.get('social_sentiment', 0.5),
                'fear_greed': analysis_results.get('fear_greed_index', 50.0)
            }
            
            # Use sentiment analyzer if available
            if self.sentiment_analyzer:
                try:
                    if hasattr(self.sentiment_analyzer, 'analyze_sentiment'):
                        sentiment_result = await self.sentiment_analyzer.analyze_sentiment(
                            symbol, timeframe, features
                        )
                    else:
                        # Fallback: use basic sentiment analysis
                        sentiment_result = {'sentiment_confidence': features['sentiment_score']}
                    
                    probability = sentiment_result.get('sentiment_confidence', 0.5)
                    direction = self._convert_probability_to_direction(probability)
                    
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for Head B: {e}")
                    # Fallback to analysis results
                    sentiment_score = features['sentiment_score']
                    probability = abs(sentiment_score - 0.5) * 2
                    direction = SignalDirection.LONG if sentiment_score > 0.6 else \
                               SignalDirection.SHORT if sentiment_score < 0.4 else SignalDirection.FLAT
            else:
                # Fallback to analysis results
                sentiment_score = features['sentiment_score']
                probability = abs(sentiment_score - 0.5) * 2
                direction = SignalDirection.LONG if sentiment_score > 0.6 else \
                           SignalDirection.SHORT if sentiment_score < 0.4 else SignalDirection.FLAT
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_B,
                direction=direction,
                probability=probability,
                confidence=probability,
                features_used=config['features'],
                reasoning=f"Sentiment ML model ({config['model_name']}) prediction"
            )
            
        except Exception as e:
            logger.error(f"❌ Head B creation failed: {e}")
            return self._create_default_head(ModelHead.HEAD_B, "Sentiment analysis failed")
    
    async def _create_head_c_orderflow(self, 
                                     analysis_results: Dict[str, Any],
                                     market_data: Dict[str, Any],
                                     symbol: str,
                                     timeframe: str) -> ModelHeadResult:
        """Create Head C: Order Flow ML model result"""
        try:
            config = self.model_head_config['head_c']
            
            # Extract order flow features
            features = {
                'volume_delta': analysis_results.get('volume_delta', 0.0),
                'orderbook_imbalance': analysis_results.get('orderbook_imbalance', 0.0),
                'liquidity_score': analysis_results.get('liquidity_score', 0.5),
                'spread': analysis_results.get('spread_atr_ratio', 0.1)
            }
            
            # Use ONNX inference if available
            if self.onnx_inference and self.feature_engineering:
                try:
                    # Create order flow feature vector - use existing method or fallback
                    if hasattr(self.feature_engineering, 'create_orderflow_features'):
                        feature_vector = await self.feature_engineering.create_orderflow_features(
                            features, symbol, timeframe
                        )
                    else:
                        # Fallback: create simple feature vector
                        feature_vector = np.array([
                            features['volume_delta'], features['orderbook_imbalance'],
                            features['liquidity_score'], features['spread']
                        ]).reshape(1, -1)
                    
                    # Get ONNX prediction
                    prediction = await self.onnx_inference.predict(
                        config['model_name'], feature_vector
                    )
                    
                    probability = prediction.get('probability', 0.5)
                    direction = self._convert_probability_to_direction(probability)
                    
                except Exception as e:
                    logger.warning(f"ONNX inference failed for Head C: {e}")
                    # Fallback to volume analysis
                    volume_confidence = analysis_results.get('volume_confidence', 0.5)
                    volume_direction = analysis_results.get('volume_direction', 'neutral')
                    
                    probability = volume_confidence
                    direction = SignalDirection.LONG if volume_direction == 'long' else \
                               SignalDirection.SHORT if volume_direction == 'short' else SignalDirection.FLAT
            else:
                # Fallback to volume analysis
                volume_confidence = analysis_results.get('volume_confidence', 0.5)
                volume_direction = analysis_results.get('volume_direction', 'neutral')
                
                probability = volume_confidence
                direction = SignalDirection.LONG if volume_direction == 'long' else \
                           SignalDirection.SHORT if volume_direction == 'short' else SignalDirection.FLAT
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_C,
                direction=direction,
                probability=probability,
                confidence=probability,
                features_used=config['features'],
                reasoning=f"Order Flow ML model ({config['model_name']}) prediction"
            )
            
        except Exception as e:
            logger.error(f"❌ Head C creation failed: {e}")
            return self._create_default_head(ModelHead.HEAD_C, "Order flow analysis failed")
    
    async def _create_head_d_rulebased(self, 
                                     analysis_results: Dict[str, Any],
                                     market_data: Dict[str, Any],
                                     symbol: str,
                                     timeframe: str) -> ModelHeadResult:
        """Create Head D: Rule-based deterministic result"""
        try:
            config = self.model_head_config['head_d']
            
            # Extract rule-based features
            features = {
                'zone_score': analysis_results.get('zone_score', 0.0),
                'structure_score': analysis_results.get('structure_score', 0.0),
                'pattern_score': analysis_results.get('pattern_score', 0.0),
                'confluence': analysis_results.get('confluence_score', 0.0)
            }
            
            # Calculate rule-based probability
            total_score = sum(features.values()) / len(features)
            probability = min(total_score / 10.0, 1.0)  # Normalize to 0-1
            
            # Determine direction based on scores
            if probability > 0.6:
                direction = SignalDirection.LONG
            elif probability < 0.4:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.FLAT
            
            return ModelHeadResult(
                head_type=ModelHead.HEAD_D,
                direction=direction,
                probability=probability,
                confidence=probability,
                features_used=config['features'],
                reasoning=f"Rule-based deterministic analysis (score: {total_score:.2f}/10)"
            )
            
        except Exception as e:
            logger.error(f"❌ Head D creation failed: {e}")
            return self._create_default_head(ModelHead.HEAD_D, "Rule-based analysis failed")
    
    def _convert_probability_to_direction(self, probability: float) -> SignalDirection:
        """Convert probability to signal direction"""
        if probability > 0.6:
            return SignalDirection.LONG
        elif probability < 0.4:
            return SignalDirection.SHORT
        else:
            return SignalDirection.FLAT
    
    def _create_default_head(self, head_type: ModelHead, reason: str) -> ModelHeadResult:
        """Create default model head result"""
        return ModelHeadResult(
            head_type=head_type,
            direction=SignalDirection.FLAT,
            probability=0.5,
            confidence=0.5,
            features_used=['default'],
            reasoning=reason
        )
    
    async def _create_fallback_model_heads(self, analysis_results: Dict[str, Any]) -> List[ModelHeadResult]:
        """Create fallback model heads when enhanced creation fails"""
        return [
            self._create_default_head(ModelHead.HEAD_A, "Technical analysis"),
            self._create_default_head(ModelHead.HEAD_B, "Sentiment analysis"),
            self._create_default_head(ModelHead.HEAD_C, "Volume analysis"),
            self._create_default_head(ModelHead.HEAD_D, "Market regime analysis")
        ]
    
    async def _calculate_zone_score(self, df: pd.DataFrame, symbol: str, timeframe: str) -> float:
        """Calculate zone score (placeholder)"""
        try:
            # Simplified zone score calculation
            atr = self._calculate_atr(df, 14)
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0.001
            
            # Check if price is near support/resistance
            current_price = df['close'].iloc[-1]
            high_20 = df['high'].rolling(20).max().iloc[-1]
            low_20 = df['low'].rolling(20).min().iloc[-1]
            
            # Calculate distance to levels
            distance_to_high = (high_20 - current_price) / current_atr
            distance_to_low = (current_price - low_20) / current_atr
            
            # Score based on proximity to levels
            if distance_to_high < 1.0 or distance_to_low < 1.0:
                return 8.0
            elif distance_to_high < 2.0 or distance_to_low < 2.0:
                return 6.0
            else:
                return 4.0
                
        except Exception as e:
            logger.error(f"❌ Zone score calculation failed: {e}")
            return 5.0
    
    async def _calculate_htf_bias(self, df: pd.DataFrame, symbol: str, timeframe: str) -> float:
        """Calculate higher timeframe bias (placeholder)"""
        try:
            # Simplified HTF bias calculation
            ema_20 = df['close'].ewm(span=20).mean()
            ema_50 = df['close'].ewm(span=50).mean()
            
            current_price = df['close'].iloc[-1]
            ema_20_current = ema_20.iloc[-1]
            ema_50_current = ema_50.iloc[-1]
            
            # Bullish bias
            if current_price > ema_20_current > ema_50_current:
                return 8.0
            # Bearish bias
            elif current_price < ema_20_current < ema_50_current:
                return 2.0
            # Neutral
            else:
                return 5.0
                
        except Exception as e:
            logger.error(f"❌ HTF bias calculation failed: {e}")
            return 5.0
    
    async def _calculate_trigger_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> float:
        """Calculate trigger quality (placeholder)"""
        try:
            # Simplified trigger quality calculation
            rsi = self._calculate_rsi(df['close'], 14)
            current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
            
            # Score based on RSI extremes
            if current_rsi < 30 or current_rsi > 70:
                return 8.0
            elif current_rsi < 40 or current_rsi > 60:
                return 6.0
            else:
                return 4.0
                
        except Exception as e:
            logger.error(f"❌ Trigger quality calculation failed: {e}")
            return 5.0
    
    async def _calculate_fvg_confluence(self, df: pd.DataFrame, symbol: str, timeframe: str) -> float:
        """Calculate FVG confluence (placeholder)"""
        try:
            # Simplified FVG confluence calculation
            # Check for gaps in recent price action
            gaps = []
            for i in range(1, len(df)):
                gap = df['low'].iloc[i] - df['high'].iloc[i-1]
                if gap > 0:
                    gaps.append(gap)
            
            if gaps:
                return 7.0
            else:
                return 5.0
                
        except Exception as e:
            logger.error(f"❌ FVG confluence calculation failed: {e}")
            return 5.0
    
    async def _calculate_ob_alignment(self, df: pd.DataFrame, symbol: str, timeframe: str) -> float:
        """Calculate orderbook alignment (placeholder)"""
        try:
            # Simplified OB alignment calculation
            # Use volume as proxy for orderbook activity
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            if volume_ratio > 1.5:
                return 8.0
            elif volume_ratio > 1.0:
                return 6.0
            else:
                return 4.0
                
        except Exception as e:
            logger.error(f"❌ OB alignment calculation failed: {e}")
            return 5.0
    
    async def _get_current_spread(self, symbol: str) -> float:
        """Get current spread (placeholder)"""
        try:
            # Simplified spread calculation
            return 0.001  # 0.1% spread
        except Exception as e:
            logger.error(f"❌ Spread calculation failed: {e}")
            return 0.01
    
    async def _calculate_impact_cost(self, symbol: str, timeframe: str) -> float:
        """Calculate impact cost (placeholder)"""
        try:
            # Simplified impact cost calculation
            return 0.0005  # 0.05% impact
        except Exception as e:
            logger.error(f"❌ Impact cost calculation failed: {e}")
            return 0.001
    
    def _determine_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine volatility regime"""
        try:
            atr = self._calculate_atr(df, 14)
            atr_20 = atr.rolling(20).mean()
            
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0.001
            avg_atr = atr_20.iloc[-1] if len(atr_20) > 0 else 0.001
            
            if current_atr > avg_atr * 1.5:
                return "high_volatility"
            elif current_atr < avg_atr * 0.5:
                return "low_volatility"
            else:
                return "normal_volatility"
                
        except Exception as e:
            logger.error(f"❌ Volatility regime determination failed: {e}")
            return "unknown"
    
    async def _calculate_liquidity_score(self, symbol: str, timeframe: str) -> float:
        """Calculate liquidity score (placeholder)"""
        try:
            # Simplified liquidity score calculation
            return 0.8  # 80% liquidity
        except Exception as e:
            logger.error(f"❌ Liquidity score calculation failed: {e}")
            return 0.5
    
    def _calculate_execution_score(self, spread_ok: bool, impact_cost: float, 
                                 volatility_regime: str, liquidity_score: float, atr: float) -> float:
        """Calculate execution score"""
        try:
            score = 5.0  # Base score
            
            if spread_ok:
                score += 2.0
            
            if impact_cost < (atr * 0.1):
                score += 1.0
            
            if volatility_regime == "normal_volatility":
                score += 1.0
            
            if liquidity_score > 0.7:
                score += 1.0
            
            return min(score, 10.0)
            
        except Exception as e:
            logger.error(f"❌ Execution score calculation failed: {e}")
            return 5.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"❌ ATR calculation failed: {e}")
            return pd.Series([0.001] * len(df), index=df.index)
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"❌ RSI calculation failed: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    async def get_dynamic_threshold(self, market_regime: str, volatility_level: str) -> Dict[str, Any]:
        """Get dynamic threshold configuration for market regime and volatility"""
        try:
            async with self.db_pool.acquire() as conn:
                threshold = await conn.fetchrow("""
                    SELECT min_confidence_threshold, min_consensus_heads, min_probability_threshold,
                           calibration_weight_isotonic, calibration_weight_platt, 
                           calibration_weight_temperature, calibration_weight_ensemble
                    FROM sde_dynamic_thresholds
                    WHERE market_regime = $1 AND volatility_level = $2 AND is_active = TRUE
                """, market_regime, volatility_level)
                
                if threshold:
                    return {
                        'min_confidence_threshold': float(threshold['min_confidence_threshold']),
                        'min_consensus_heads': int(threshold['min_consensus_heads']),
                        'min_probability_threshold': float(threshold['min_probability_threshold']),
                        'calibration_weights': {
                            'isotonic': float(threshold['calibration_weight_isotonic']),
                            'platt': float(threshold['calibration_weight_platt']),
                            'temperature': float(threshold['calibration_weight_temperature']),
                            'ensemble': float(threshold['calibration_weight_ensemble'])
                        }
                    }
                else:
                    # Return default thresholds if not found
                    return {
                        'min_confidence_threshold': 0.85,
                        'min_consensus_heads': 3,
                        'min_probability_threshold': 0.70,
                        'calibration_weights': {
                            'isotonic': 0.4,
                            'platt': 0.3,
                            'temperature': 0.2,
                            'ensemble': 0.1
                        }
                    }
                    
        except Exception as e:
            logger.error(f"❌ Failed to get dynamic threshold: {e}")
            return {
                'min_confidence_threshold': 0.85,
                'min_consensus_heads': 3,
                'min_probability_threshold': 0.70,
                'calibration_weights': {
                    'isotonic': 0.4,
                    'platt': 0.3,
                    'temperature': 0.2,
                    'ensemble': 0.1
                }
            }
    
    async def apply_advanced_calibration(self, 
                                       raw_probability: float,
                                       features: Dict[str, float],
                                       symbol: str,
                                       timeframe: str,
                                       market_regime: str) -> CalibrationResult:
        """Apply advanced calibration to raw probability"""
        try:
            if self.advanced_calibration:
                return await self.advanced_calibration.calibrate_probability(
                    raw_probability, features, symbol, timeframe, market_regime
                )
            else:
                # Fallback to basic calibration
                return CalibrationResult(
                    calibrated_probability=raw_probability,
                    calibration_method='fallback',
                    confidence_interval=(raw_probability - 0.1, raw_probability + 0.1),
                    reliability_score=0.5,
                    method_performance={'fallback': 0.5}
                )
                
        except Exception as e:
            logger.error(f"❌ Advanced calibration failed: {e}")
            return CalibrationResult(
                calibrated_probability=raw_probability,
                calibration_method='error',
                confidence_interval=(raw_probability - 0.1, raw_probability + 0.1),
                reliability_score=0.5,
                method_performance={'error': 0.5}
            )
    
    async def get_consensus_optimization(self, market_regime: str, volatility_level: str) -> Dict[str, Any]:
        """Get consensus optimization configuration"""
        try:
            async with self.db_pool.acquire() as conn:
                consensus = await conn.fetchrow("""
                    SELECT min_agreeing_heads, min_probability_threshold,
                           consensus_weight_head_a, consensus_weight_head_b,
                           consensus_weight_head_c, consensus_weight_head_d,
                           direction_agreement_required, confidence_threshold
                    FROM sde_consensus_optimization
                    WHERE market_regime = $1 AND volatility_level = $2 AND is_active = TRUE
                """, market_regime, volatility_level)
                
                if consensus:
                    return {
                        'min_agreeing_heads': int(consensus['min_agreeing_heads']),
                        'min_probability_threshold': float(consensus['min_probability_threshold']),
                        'head_weights': {
                            'head_a': float(consensus['consensus_weight_head_a']),
                            'head_b': float(consensus['consensus_weight_head_b']),
                            'head_c': float(consensus['consensus_weight_head_c']),
                            'head_d': float(consensus['consensus_weight_head_d'])
                        },
                        'direction_agreement_required': bool(consensus['direction_agreement_required']),
                        'confidence_threshold': float(consensus['confidence_threshold'])
                    }
                else:
                    # Return default consensus configuration
                    return {
                        'min_agreeing_heads': 3,
                        'min_probability_threshold': 0.70,
                        'head_weights': {
                            'head_a': 0.4,
                            'head_b': 0.2,
                            'head_c': 0.2,
                            'head_d': 0.2
                        },
                        'direction_agreement_required': True,
                        'confidence_threshold': 0.85
                    }
                    
        except Exception as e:
            logger.error(f"❌ Failed to get consensus optimization: {e}")
            return {
                'min_agreeing_heads': 3,
                'min_probability_threshold': 0.70,
                'head_weights': {
                    'head_a': 0.4,
                    'head_b': 0.2,
                    'head_c': 0.2,
                    'head_d': 0.2
                },
                'direction_agreement_required': True,
                'confidence_threshold': 0.85
            }
    
    # Phase 9: Advanced Signal Quality Validation Methods
    
    async def validate_signal_quality(self, 
                                    signal_data: Dict[str, Any],
                                    market_data: Dict[str, Any],
                                    historical_data: pd.DataFrame) -> SignalQualityMetrics:
        """Validate signal quality using advanced validation system"""
        try:
            if self.signal_quality_validator:
                return await self.signal_quality_validator.validate_signal_quality(
                    signal_data, market_data, historical_data
                )
            else:
                # Fallback validation
                logger.warning("⚠️ Signal quality validator not available, using fallback")
                return SignalQualityMetrics(
                    confidence_score=signal_data.get('confidence', 0.0),
                    overall_quality_score=signal_data.get('confidence', 0.0),
                    validation_passed=signal_data.get('confidence', 0.0) >= 0.85,
                    rejection_reasons=[] if signal_data.get('confidence', 0.0) >= 0.85 else ["Fallback validation failed"]
                )
                
        except Exception as e:
            logger.error(f"❌ Signal quality validation failed: {e}")
            return SignalQualityMetrics(
                validation_passed=False,
                rejection_reasons=[f"Validation error: {str(e)}"]
            )
    
    async def get_signal_quality_performance(self, symbol: str, timeframe: str, days: int = 7) -> Dict[str, Any]:
        """Get signal quality performance summary"""
        try:
            if self.signal_quality_validator:
                return await self.signal_quality_validator.get_performance_summary()
            else:
                return {
                    'performance_metrics': {
                        'total_validated': 0,
                        'total_passed': 0,
                        'total_rejected': 0,
                        'avg_quality_score': 0.0,
                        'avg_processing_time_ms': 0.0
                    },
                    'current_thresholds': {
                        'min_confidence': 0.85,
                        'min_quality_score': 0.70,
                        'min_volume_confirmation': 0.60,
                        'min_trend_strength': 0.65,
                        'max_volatility': 0.80
                    },
                    'quality_distribution': {},
                    'recent_performance': {}
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get signal quality performance: {e}")
            return {}
    
    async def get_false_positive_analysis(self, symbol: str, timeframe: str, days: int = 7) -> Dict[str, Any]:
        """Get false positive analysis for signal validation"""
        try:
            if self.signal_quality_validator:
                analysis = await self.signal_quality_validator.get_false_positive_analysis(symbol, timeframe, days)
                return {
                    'total_signals': analysis.total_signals,
                    'rejected_signals': analysis.rejected_signals,
                    'false_positives': analysis.false_positives,
                    'accuracy': analysis.accuracy,
                    'precision': analysis.precision,
                    'recall': analysis.recall,
                    'f1_score': analysis.f1_score,
                    'analysis_period_days': analysis.analysis_period.days
                }
            else:
                return {
                    'total_signals': 0,
                    'rejected_signals': 0,
                    'false_positives': 0,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'analysis_period_days': days
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get false positive analysis: {e}")
            return {}
    
    async def _initialize_phase9_quality_validation(self):
        """Initialize Phase 9 signal quality validation components"""
        try:
            logger.info("🚀 Initializing Phase 9: Advanced Signal Quality Validation...")
            
            # Test signal quality validator
            if self.signal_quality_validator:
                # Create mock data for testing
                mock_signal_data = {
                    'symbol': 'BTCUSDT',
                    'timeframe': '1h',
                    'signal_id': 'test_signal_001',
                    'confidence': 0.85,
                    'model_agreement': 0.80,
                    'feature_importance': 0.75,
                    'historical_accuracy': 0.82
                }
                
                mock_market_data = {
                    'current_price': 45000.0,
                    'volume_24h': 1000000000,
                    'volatility_24h': 0.025
                }
                
                # Create mock historical data
                import numpy as np
                dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
                mock_historical = pd.DataFrame({
                    'timestamp': dates,
                    'open': np.random.normal(45000, 1000, 100),
                    'high': np.random.normal(45500, 1000, 100),
                    'low': np.random.normal(44500, 1000, 100),
                    'close': np.random.normal(45000, 1000, 100),
                    'volume': np.random.normal(1000000, 200000, 100)
                })
                
                # Test validation
                quality_metrics = await self.validate_signal_quality(
                    mock_signal_data, mock_market_data, mock_historical
                )
                
                logger.info(f"✅ Signal Quality Validation Test:")
                logger.info(f"   Overall Quality Score: {quality_metrics.overall_quality_score:.4f}")
                logger.info(f"   Quality Level: {quality_metrics.quality_level.value}")
                logger.info(f"   Validation Passed: {quality_metrics.validation_passed}")
                logger.info(f"   Rejection Reasons: {quality_metrics.rejection_reasons}")
                
                # Test performance summary
                performance = await self.get_signal_quality_performance('BTCUSDT', '1h')
                logger.info(f"✅ Performance Summary: {performance.get('performance_metrics', {})}")
                
                # Test false positive analysis
                fp_analysis = await self.get_false_positive_analysis('BTCUSDT', '1h')
                logger.info(f"✅ False Positive Analysis: {fp_analysis}")
                
            else:
                logger.warning("⚠️ Signal quality validator not available for Phase 9 testing")
            
            logger.info("✅ Phase 9: Advanced Signal Quality Validation initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Phase 9 initialization failed: {e}")
            raise
    
    async def _initialize_phase10_production_monitoring(self):
        """Initialize Phase 10 production monitoring components"""
        try:
            logger.info("🚀 Initializing Phase 10: Production Monitoring & Deployment System...")
            
            # Test production monitoring system
            if self.production_monitoring:
                # Start monitoring
                await self.production_monitoring.start_monitoring()
                
                # Wait a moment for initial metrics collection
                await asyncio.sleep(5)
                
                # Get system status
                system_status = await self.production_monitoring.get_system_status()
                
                logger.info(f"✅ Production Monitoring System Status:")
                logger.info(f"   Monitoring Running: {system_status.get('monitoring_status', {}).get('is_running', False)}")
                logger.info(f"   Services Monitored: {len(system_status.get('service_status', {}))}")
                logger.info(f"   Active Alerts: {system_status.get('performance_summary', {}).get('active_alerts', 0)}")
                logger.info(f"   Metrics Collected: {system_status.get('performance_summary', {}).get('metrics_collected', 0)}")
                
                # Test service health checks
                if system_status.get('service_status'):
                    for service_name, service_info in system_status['service_status'].items():
                        logger.info(f"   {service_name}: {service_info['status']} ({service_info['response_time_ms']:.1f}ms)")
                
                # Stop monitoring for now (will be started by main application)
                await self.production_monitoring.stop_monitoring()
                
            else:
                logger.warning("⚠️ Production monitoring system not available for Phase 10 testing")
            
            logger.info("✅ Phase 10: Production Monitoring & Deployment System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Phase 10 initialization failed: {e}")
            raise