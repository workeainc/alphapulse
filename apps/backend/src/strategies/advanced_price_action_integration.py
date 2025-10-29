"""
Advanced Price Action Integration Engine
Integrates sophisticated price action models with signal generator for enhanced accuracy
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import asyncpg

# Import existing sophisticated price action components
try:
    from .demand_supply_zone_analyzer import DemandSupplyZoneAnalyzer
    from .dynamic_support_resistance_analyzer import DynamicSupportResistanceAnalyzer
    from .market_structure_analyzer import MarketStructureAnalyzer
    from .advanced_pattern_detector import AdvancedPatternDetector
    PRICE_ACTION_AVAILABLE = True
except ImportError:
    PRICE_ACTION_AVAILABLE = False
    logging.warning("Price action components not available")

# Import ML components
try:
    from ..src.ai.onnx_inference import ONNXInferenceEngine
    from ..src.ai.feature_drift_detector import FeatureDriftDetector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML components not available")

logger = logging.getLogger(__name__)

@dataclass
class PriceActionAnalysis:
    """Comprehensive price action analysis result"""
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Support & Resistance Analysis
    support_resistance_score: float = 0.0
    support_resistance_confidence: float = 0.0
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    support_strength: float = 0.0
    resistance_strength: float = 0.0
    
    # Market Structure Analysis (HH, HL, LH, LL)
    market_structure_score: float = 0.0
    market_structure_confidence: float = 0.0
    structure_type: Optional[str] = None  # 'HH', 'HL', 'LH', 'LL', 'breakout', 'breakdown'
    trend_alignment: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    momentum_score: float = 0.0
    
    # Demand & Supply Zones
    demand_supply_score: float = 0.0
    demand_supply_confidence: float = 0.0
    zone_type: Optional[str] = None  # 'demand', 'supply', 'equilibrium'
    zone_strength: Optional[str] = None  # 'weak', 'moderate', 'strong', 'very_strong'
    breakout_probability: float = 0.0
    hold_probability: float = 0.0
    
    # ML-Enhanced Pattern Analysis
    pattern_ml_score: float = 0.0
    pattern_ml_confidence: float = 0.0
    pattern_type: Optional[str] = None
    pattern_probability: float = 0.0
    
    # Combined Metrics
    combined_price_action_score: float = 0.0
    price_action_confidence: float = 0.0
    
    # Context Data
    support_resistance_context: Dict[str, Any] = field(default_factory=dict)
    market_structure_context: Dict[str, Any] = field(default_factory=dict)
    demand_supply_context: Dict[str, Any] = field(default_factory=dict)
    pattern_ml_context: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Tracking
    processing_time_ms: float = 0.0
    component_latencies: Dict[str, float] = field(default_factory=dict)

@dataclass
class EnhancedSignal:
    """Enhanced signal with price action integration"""
    original_signal_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Original Signal Metrics
    original_confidence: float
    original_risk_reward: float
    original_entry_price: Optional[float]
    original_stop_loss: Optional[float]
    original_take_profit: Optional[float]
    
    # Enhanced Metrics
    enhanced_confidence: float
    enhanced_risk_reward: float
    enhanced_entry_price: Optional[float]
    enhanced_stop_loss: Optional[float]
    enhanced_take_profit: Optional[float]
    
    # Price Action Integration
    price_action_analysis: PriceActionAnalysis
    
    # Enhancement Factors
    confidence_improvement: float = 0.0
    risk_reward_improvement: float = 0.0
    entry_optimization: float = 0.0
    
    # Reasoning
    enhancement_reasons: List[str] = field(default_factory=list)
    price_action_reasons: List[str] = field(default_factory=list)

class AdvancedPriceActionIntegration:
    """
    Advanced Price Action Integration Engine
    Integrates sophisticated price action models with signal generator
    """
    
    def __init__(self, db_pool: asyncpg.Pool = None):
        self.db_pool = db_pool
        self.logger = logger
        
        # Price Action Components
        self.demand_supply_analyzer = None
        self.support_resistance_analyzer = None
        self.market_structure_analyzer = None
        self.pattern_detector = None
        
        # ML Components
        self.onnx_engine = None
        self.drift_detector = None
        
        # Configuration
        self.config = self._load_configuration()
        
        # Performance Tracking
        self.stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'avg_processing_time_ms': 0.0,
            'enhancement_success_rate': 0.0
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("Advanced Price Action Integration Engine initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load price action integration configuration"""
        return {
            'support_resistance_weight': 0.25,
            'market_structure_weight': 0.25,
            'demand_supply_weight': 0.25,
            'pattern_ml_weight': 0.25,
            'min_combined_score': 0.75,
            'min_confidence_threshold': 0.8,
            'enhancement_factor': 1.2,
            'risk_reward_improvement': 0.1,
            'ml_prediction_required': True,
            'market_context_required': True
        }
    
    async def initialize(self):
        """Initialize price action components"""
        try:
            if PRICE_ACTION_AVAILABLE:
                # Initialize demand & supply analyzer
                self.demand_supply_analyzer = DemandSupplyZoneAnalyzer()
                
                # Initialize support & resistance analyzer
                self.support_resistance_analyzer = DynamicSupportResistanceAnalyzer()
                
                # Initialize market structure analyzer
                self.market_structure_analyzer = MarketStructureAnalyzer()
                
                # Initialize pattern detector
                self.pattern_detector = AdvancedPatternDetector()
                
                self.logger.info("✅ Price action components initialized")
            
            if ML_AVAILABLE and self.db_pool:
                # Initialize ML components
                self.onnx_engine = ONNXInferenceEngine(db_pool=self.db_pool)
                self.drift_detector = FeatureDriftDetector(db_pool=self.db_pool)
                
                self.logger.info("✅ ML components initialized")
            
            # Load configuration from database
            await self._load_database_config()
            
            self.logger.info("✅ Advanced Price Action Integration Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize price action integration: {e}")
            raise
    
    async def _load_database_config(self):
        """Load configuration from database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                config_row = await conn.fetchrow("""
                    SELECT config_data FROM price_action_config 
                    WHERE config_name = 'price_action_integration_default' AND is_active = true
                """)
                
                if config_row:
                    db_config = config_row['config_data']
                    # Parse JSON if it's a string
                    if isinstance(db_config, str):
                        import json
                        db_config = json.loads(db_config)
                    self.config.update(db_config)
                    self.logger.info("✅ Loaded configuration from database")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load database config: {e}")
    
    async def analyze_price_action(self, 
                                 symbol: str, 
                                 timeframe: str, 
                                 market_data: Dict[str, Any]) -> PriceActionAnalysis:
        """
        Perform comprehensive price action analysis
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            market_data: Market data including OHLCV, indicators, etc.
            
        Returns:
            PriceActionAnalysis with comprehensive results
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            analysis = PriceActionAnalysis(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now()
            )
            
            # Parallel analysis tasks
            tasks = []
            
            # Support & Resistance Analysis
            if self.support_resistance_analyzer:
                tasks.append(self._analyze_support_resistance(symbol, timeframe, market_data))
            
            # Market Structure Analysis
            if self.market_structure_analyzer:
                tasks.append(self._analyze_market_structure(symbol, timeframe, market_data))
            
            # Demand & Supply Analysis
            if self.demand_supply_analyzer:
                tasks.append(self._analyze_demand_supply(symbol, timeframe, market_data))
            
            # ML-Enhanced Pattern Analysis
            if self.pattern_detector and ML_AVAILABLE:
                tasks.append(self._analyze_pattern_ml(symbol, timeframe, market_data))
            
            # Execute all analyses in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"❌ Analysis task {i} failed: {result}")
                        continue
                    
                    if result and hasattr(result, 'support_resistance_score'):
                        # Support & Resistance result
                        analysis.support_resistance_score = result.support_resistance_score
                        analysis.support_resistance_confidence = result.support_resistance_confidence
                        analysis.nearest_support = result.nearest_support
                        analysis.nearest_resistance = result.nearest_resistance
                        analysis.support_strength = result.support_strength
                        analysis.resistance_strength = result.resistance_strength
                        analysis.support_resistance_context = result.context
                        
                    elif result and hasattr(result, 'market_structure_score'):
                        # Market Structure result
                        analysis.market_structure_score = result.market_structure_score
                        analysis.market_structure_confidence = result.market_structure_confidence
                        analysis.structure_type = result.structure_type
                        analysis.trend_alignment = result.trend_alignment
                        analysis.momentum_score = result.momentum_score
                        analysis.market_structure_context = result.context
                        
                    elif result and hasattr(result, 'demand_supply_score'):
                        # Demand & Supply result
                        analysis.demand_supply_score = result.demand_supply_score
                        analysis.demand_supply_confidence = result.demand_supply_confidence
                        analysis.zone_type = result.zone_type
                        analysis.zone_strength = result.zone_strength
                        analysis.breakout_probability = result.breakout_probability
                        analysis.hold_probability = result.hold_probability
                        analysis.demand_supply_context = result.context
                        
                    elif result and hasattr(result, 'pattern_ml_score'):
                        # Pattern ML result
                        analysis.pattern_ml_score = result.pattern_ml_score
                        analysis.pattern_ml_confidence = result.pattern_ml_confidence
                        analysis.pattern_type = result.pattern_type
                        analysis.pattern_probability = result.pattern_probability
                        analysis.pattern_ml_context = result.context
            
            # Calculate combined scores
            analysis.combined_price_action_score = self._calculate_combined_score(analysis)
            analysis.price_action_confidence = self._calculate_confidence(analysis)
            
            # Calculate processing time
            analysis.processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update statistics
            self.stats['total_analyses'] += 1
            self.stats['successful_analyses'] += 1
            self.stats['avg_processing_time_ms'] = (
                (self.stats['avg_processing_time_ms'] * (self.stats['total_analyses'] - 1) + 
                 analysis.processing_time_ms) / self.stats['total_analyses']
            )
            
            # Store analysis in database
            await self._store_analysis(analysis)
            
            self.logger.info(f"✅ Price action analysis completed for {symbol} {timeframe} "
                           f"(score: {analysis.combined_price_action_score:.3f}, "
                           f"confidence: {analysis.price_action_confidence:.3f})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Price action analysis failed: {e}")
            self.stats['failed_analyses'] += 1
            raise
    
    async def _analyze_support_resistance(self, symbol: str, timeframe: str, market_data: Dict[str, Any]):
        """Analyze support and resistance levels"""
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv', [])
            if not ohlcv_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Analyze support and resistance
            result = await self.support_resistance_analyzer.analyze_support_resistance(symbol, timeframe, ohlcv_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Support resistance analysis failed: {e}")
            return None
    
    async def _analyze_market_structure(self, symbol: str, timeframe: str, market_data: Dict[str, Any]):
        """Analyze market structure (HH, HL, LH, LL)"""
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv', [])
            if not ohlcv_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Analyze market structure
            result = await self.market_structure_analyzer.analyze_market_structure(symbol, timeframe, ohlcv_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Market structure analysis failed: {e}")
            return None
    
    async def _analyze_demand_supply(self, symbol: str, timeframe: str, market_data: Dict[str, Any]):
        """Analyze demand and supply zones"""
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv', [])
            if not ohlcv_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Analyze demand and supply zones
            result = await self.demand_supply_analyzer.analyze_demand_supply_zones(symbol, timeframe, df)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Demand supply analysis failed: {e}")
            return None
    
    async def _analyze_pattern_ml(self, symbol: str, timeframe: str, market_data: Dict[str, Any]):
        """Analyze patterns with ML enhancement"""
        try:
            # Extract OHLCV data
            ohlcv_data = market_data.get('ohlcv', [])
            if not ohlcv_data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Analyze patterns with ML
            result = await self.pattern_detector.analyze_patterns_ml(symbol, timeframe, df)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Pattern ML analysis failed: {e}")
            return None
    
    def _calculate_combined_score(self, analysis: PriceActionAnalysis) -> float:
        """Calculate combined price action score"""
        weights = {
            'support_resistance': self.config['support_resistance_weight'],
            'market_structure': self.config['market_structure_weight'],
            'demand_supply': self.config['demand_supply_weight'],
            'pattern_ml': self.config['pattern_ml_weight']
        }
        
        scores = {
            'support_resistance': analysis.support_resistance_score,
            'market_structure': analysis.market_structure_score,
            'demand_supply': analysis.demand_supply_score,
            'pattern_ml': analysis.pattern_ml_score
        }
        
        combined_score = sum(scores[key] * weights[key] for key in weights)
        return min(1.0, max(0.0, combined_score))
    
    def _calculate_confidence(self, analysis: PriceActionAnalysis) -> float:
        """Calculate overall confidence score"""
        confidences = [
            analysis.support_resistance_confidence,
            analysis.market_structure_confidence,
            analysis.demand_supply_confidence,
            analysis.pattern_ml_confidence
        ]
        
        # Remove None values
        confidences = [c for c in confidences if c is not None]
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    async def enhance_signal(self, 
                           signal_id: str,
                           symbol: str, 
                           timeframe: str, 
                           original_confidence: float,
                           original_risk_reward: float,
                           original_entry_price: Optional[float],
                           original_stop_loss: Optional[float],
                           original_take_profit: Optional[float],
                           market_data: Dict[str, Any]) -> EnhancedSignal:
        """
        Enhance signal with price action analysis
        
        Args:
            signal_id: Original signal ID
            symbol: Trading symbol
            timeframe: Timeframe
            original_confidence: Original signal confidence
            original_risk_reward: Original risk/reward ratio
            original_entry_price: Original entry price
            original_stop_loss: Original stop loss
            original_take_profit: Original take profit
            market_data: Market data for analysis
            
        Returns:
            EnhancedSignal with improved metrics
        """
        try:
            # Perform price action analysis
            price_action_analysis = await self.analyze_price_action(symbol, timeframe, market_data)
            
            # Create enhanced signal
            enhanced_signal = EnhancedSignal(
                original_signal_id=signal_id,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                
                # Original metrics
                original_confidence=original_confidence,
                original_risk_reward=original_risk_reward,
                original_entry_price=original_entry_price,
                original_stop_loss=original_stop_loss,
                original_take_profit=original_take_profit,
                
                # Enhanced metrics
                enhanced_confidence=0.0,
                enhanced_risk_reward=0.0,
                enhanced_entry_price=original_entry_price,
                enhanced_stop_loss=original_stop_loss,
                enhanced_take_profit=original_take_profit,
                
                # Price action analysis
                price_action_analysis=price_action_analysis
            )
            
            # Apply enhancements
            enhanced_signal = self._apply_enhancements(enhanced_signal)
            
            # Store enhanced signal
            await self._store_enhanced_signal(enhanced_signal)
            
            self.logger.info(f"✅ Signal enhanced for {symbol} {timeframe} "
                           f"(confidence: {original_confidence:.3f} → {enhanced_signal.enhanced_confidence:.3f}, "
                           f"RR: {original_risk_reward:.2f} → {enhanced_signal.enhanced_risk_reward:.2f})")
            
            return enhanced_signal
            
        except Exception as e:
            self.logger.error(f"❌ Signal enhancement failed: {e}")
            raise
    
    def _apply_enhancements(self, enhanced_signal: EnhancedSignal) -> EnhancedSignal:
        """Apply price action enhancements to signal"""
        analysis = enhanced_signal.price_action_analysis
        
        # Check if price action analysis is strong enough
        if (analysis.combined_price_action_score < self.config['min_combined_score'] or
            analysis.price_action_confidence < self.config['min_confidence_threshold']):
            
            # Price action analysis is weak, keep original metrics
            enhanced_signal.enhanced_confidence = enhanced_signal.original_confidence
            enhanced_signal.enhanced_risk_reward = enhanced_signal.original_risk_reward
            enhanced_signal.enhancement_reasons.append("Price action analysis below threshold")
            return enhanced_signal
        
        # Apply confidence enhancement
        confidence_boost = analysis.combined_price_action_score * self.config['enhancement_factor']
        enhanced_signal.enhanced_confidence = min(1.0, enhanced_signal.original_confidence + confidence_boost)
        enhanced_signal.confidence_improvement = enhanced_signal.enhanced_confidence - enhanced_signal.original_confidence
        
        # Apply risk/reward enhancement
        rr_boost = self.config['risk_reward_improvement']
        enhanced_signal.enhanced_risk_reward = enhanced_signal.original_risk_reward + rr_boost
        enhanced_signal.risk_reward_improvement = rr_boost
        
        # Optimize entry/exit levels based on price action
        enhanced_signal = self._optimize_entry_exit_levels(enhanced_signal)
        
        # Add reasoning
        enhanced_signal.enhancement_reasons.extend([
            f"Price action score: {analysis.combined_price_action_score:.3f}",
            f"Support/Resistance: {analysis.support_resistance_score:.3f}",
            f"Market Structure: {analysis.market_structure_score:.3f}",
            f"Demand/Supply: {analysis.demand_supply_score:.3f}",
            f"Pattern ML: {analysis.pattern_ml_score:.3f}"
        ])
        
        return enhanced_signal
    
    def _optimize_entry_exit_levels(self, enhanced_signal: EnhancedSignal) -> EnhancedSignal:
        """Optimize entry and exit levels based on price action analysis"""
        analysis = enhanced_signal.price_action_analysis
        
        # Optimize entry price based on support/resistance
        if analysis.nearest_support and analysis.nearest_resistance:
            current_price = enhanced_signal.original_entry_price or 0
            
            # If bullish signal, optimize entry near support
            if enhanced_signal.original_confidence > 0.5:  # Assuming this indicates bullish
                if analysis.support_strength > 0.7:
                    enhanced_signal.enhanced_entry_price = analysis.nearest_support
                    enhanced_signal.entry_optimization = abs(current_price - analysis.nearest_support)
                    enhanced_signal.price_action_reasons.append(f"Entry optimized to strong support: {analysis.nearest_support}")
            
            # If bearish signal, optimize entry near resistance
            else:
                if analysis.resistance_strength > 0.7:
                    enhanced_signal.enhanced_entry_price = analysis.nearest_resistance
                    enhanced_signal.entry_optimization = abs(current_price - analysis.nearest_resistance)
                    enhanced_signal.price_action_reasons.append(f"Entry optimized to strong resistance: {analysis.nearest_resistance}")
        
        # Optimize stop loss based on market structure
        if analysis.structure_type in ['LL', 'breakdown'] and analysis.momentum_score > 0.7:
            # Strong downward momentum, tighten stop loss
            if enhanced_signal.original_stop_loss:
                enhanced_signal.enhanced_stop_loss = enhanced_signal.original_stop_loss * 0.95
                enhanced_signal.price_action_reasons.append("Stop loss tightened due to strong downward momentum")
        
        elif analysis.structure_type in ['HH', 'breakout'] and analysis.momentum_score > 0.7:
            # Strong upward momentum, widen stop loss
            if enhanced_signal.original_stop_loss:
                enhanced_signal.enhanced_stop_loss = enhanced_signal.original_stop_loss * 1.05
                enhanced_signal.price_action_reasons.append("Stop loss widened due to strong upward momentum")
        
        return enhanced_signal
    
    async def _store_analysis(self, analysis: PriceActionAnalysis):
        """Store price action analysis in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO price_action_ml_predictions (
                        symbol, timeframe, timestamp, prediction_type, prediction_probability,
                        confidence_score, feature_vector, model_output, market_context
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, 
                analysis.symbol, analysis.timeframe, analysis.timestamp,
                'comprehensive_analysis', analysis.combined_price_action_score,
                analysis.price_action_confidence,
                json.dumps(analysis.support_resistance_context),
                json.dumps({
                    'support_resistance_score': analysis.support_resistance_score,
                    'market_structure_score': analysis.market_structure_score,
                    'demand_supply_score': analysis.demand_supply_score,
                    'pattern_ml_score': analysis.pattern_ml_score
                }),
                json.dumps({
                    'support_resistance_context': analysis.support_resistance_context,
                    'market_structure_context': analysis.market_structure_context,
                    'demand_supply_context': analysis.demand_supply_context,
                    'pattern_ml_context': analysis.pattern_ml_context
                })
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store analysis: {e}")
    
    async def _store_enhanced_signal(self, enhanced_signal: EnhancedSignal):
        """Store enhanced signal in database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO price_action_signal_integration (
                        signal_id, symbol, timeframe, timestamp,
                        support_resistance_score, market_structure_score, demand_supply_score, pattern_ml_score,
                        combined_price_action_score, price_action_confidence,
                        enhanced_confidence_score, enhanced_risk_reward_ratio,
                        enhanced_entry_price, enhanced_stop_loss, enhanced_take_profit,
                        support_resistance_context, market_structure_context, demand_supply_context, pattern_ml_context
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                """,
                enhanced_signal.original_signal_id, enhanced_signal.symbol, enhanced_signal.timeframe, enhanced_signal.timestamp,
                enhanced_signal.price_action_analysis.support_resistance_score,
                enhanced_signal.price_action_analysis.market_structure_score,
                enhanced_signal.price_action_analysis.demand_supply_score,
                enhanced_signal.price_action_analysis.pattern_ml_score,
                enhanced_signal.price_action_analysis.combined_price_action_score,
                enhanced_signal.price_action_analysis.price_action_confidence,
                enhanced_signal.enhanced_confidence, enhanced_signal.enhanced_risk_reward,
                enhanced_signal.enhanced_entry_price, enhanced_signal.enhanced_stop_loss, enhanced_signal.enhanced_take_profit,
                json.dumps(enhanced_signal.price_action_analysis.support_resistance_context),
                json.dumps(enhanced_signal.price_action_analysis.market_structure_context),
                json.dumps(enhanced_signal.price_action_analysis.demand_supply_context),
                json.dumps(enhanced_signal.price_action_analysis.pattern_ml_context)
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store enhanced signal: {e}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_analyses': self.stats['total_analyses'],
            'successful_analyses': self.stats['successful_analyses'],
            'failed_analyses': self.stats['failed_analyses'],
            'success_rate': (self.stats['successful_analyses'] / max(1, self.stats['total_analyses'])) * 100,
            'avg_processing_time_ms': self.stats['avg_processing_time_ms'],
            'enhancement_success_rate': self.stats['enhancement_success_rate']
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            
            self.logger.info("✅ Advanced Price Action Integration Engine cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
