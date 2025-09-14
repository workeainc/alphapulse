"""
AI-Driven Threshold Manager for AlphaPulse
Integrates Reinforcement Learning, LLM, and Enhanced Regime Detection for optimal threshold management
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import json

# Import Phase 3 components
from .threshold_env import ThresholdEnv
from .llm_threshold_predictor import LLMThresholdPredictor, MarketContext, ThresholdPrediction
from .enhanced_regime_detection import EnhancedRegimeDetector, RegimeClassification, RegimeThresholds

# RL imports
try:
    from stable_baselines3 import PPO, DDPG
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ThresholdDecision:
    """Final threshold decision combining all AI components"""
    volume_threshold: float
    trend_threshold: float
    confidence_threshold: float
    decision_confidence: float
    primary_method: str  # 'regime', 'rl', 'llm', 'ensemble'
    regime: str
    reasoning: str
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThresholdPerformance:
    """Threshold performance tracking"""
    timestamp: datetime
    thresholds: ThresholdDecision
    signal_confidence: float
    signal_passed: bool
    actual_outcome: bool
    performance_score: float

class AIDrivenThresholdManager:
    """
    Main AI-driven threshold manager that orchestrates RL, LLM, and regime detection.
    Implements tiered decision logic: regime-based → RL/LLM → fallback.
    """
    
    def __init__(self,
                 enable_rl: bool = True,
                 enable_llm: bool = True,
                 enable_regime_detection: bool = True,
                 decision_interval: int = 60,  # 1 minute
                 performance_window: int = 1000,
                 ensemble_weights: Dict[str, float] = None):
        
        self.enable_rl = enable_rl and RL_AVAILABLE
        self.enable_llm = enable_llm
        self.enable_regime_detection = enable_regime_detection
        self.decision_interval = decision_interval
        self.performance_window = performance_window
        
        # Ensemble weights for combining predictions
        self.ensemble_weights = ensemble_weights or {
            'regime': 0.4,
            'rl': 0.3,
            'llm': 0.3
        }
        
        # Initialize components
        self.regime_detector = EnhancedRegimeDetector() if self.enable_regime_detection else None
        self.llm_predictor = LLMThresholdPredictor() if self.enable_llm else None
        self.rl_agent = None
        self.rl_env = None
        
        # Performance tracking
        self.performance_history = deque(maxlen=performance_window)
        self.decision_history = deque(maxlen=100)
        self.last_decision = datetime.now()
        
        # Current thresholds
        self.current_thresholds = ThresholdDecision(
            volume_threshold=500.0,
            trend_threshold=0.5,
            confidence_threshold=0.6,
            decision_confidence=0.5,
            primary_method='fallback',
            regime='normal_trending',
            reasoning='Initial fallback thresholds',
            processing_time=0.0
        )
        
        # Background tasks
        self.background_task = None
        self.rl_training_task = None
        
        logger.info(f"AI-Driven Threshold Manager initialized (RL: {self.enable_rl}, LLM: {self.enable_llm}, Regime: {self.enable_regime_detection})")
    
    async def start(self):
        """Start the AI-driven threshold manager"""
        try:
            # Start regime detector
            if self.regime_detector:
                await self.regime_detector.start()
            
            # Initialize RL agent
            if self.enable_rl:
                await self._initialize_rl_agent()
            
            # Start background tasks
            self.background_task = asyncio.create_task(self._background_decision_loop())
            
            logger.info("AI-Driven Threshold Manager started successfully")
            
        except Exception as e:
            logger.error(f"Error starting AI-Driven Threshold Manager: {e}")
    
    async def stop(self):
        """Stop the AI-driven threshold manager"""
        try:
            # Stop regime detector
            if self.regime_detector:
                await self.regime_detector.stop()
            
            # Stop background tasks
            if self.background_task:
                self.background_task.cancel()
                try:
                    await self.background_task
                except asyncio.CancelledError:
                    pass
            
            if self.rl_training_task:
                self.rl_training_task.cancel()
                try:
                    await self.rl_training_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("AI-Driven Threshold Manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping AI-Driven Threshold Manager: {e}")
    
    async def get_optimal_thresholds(self, 
                                   market_data: Dict[str, Any],
                                   signal_confidence: float = 0.5) -> ThresholdDecision:
        """
        Get optimal thresholds using AI-driven decision making
        
        Args:
            market_data: Current market data
            signal_confidence: Current signal confidence
            
        Returns:
            ThresholdDecision with optimal thresholds
        """
        start_time = time.time()
        
        try:
            # Extract market context
            market_context = self._extract_market_context(market_data)
            
            # Tier 1: Regime-based thresholds
            if self.enable_regime_detection and self.regime_detector:
                regime_decision = await self._get_regime_based_thresholds(market_context)
                if regime_decision.decision_confidence > 0.7:
                    regime_decision.processing_time = time.time() - start_time
                    return regime_decision
            
            # Tier 2: AI ensemble (RL + LLM)
            if self.enable_rl or self.enable_llm:
                ensemble_decision = await self._get_ensemble_thresholds(market_context)
                if ensemble_decision.decision_confidence > 0.6:
                    ensemble_decision.processing_time = time.time() - start_time
                    return ensemble_decision
            
            # Tier 3: Fallback
            fallback_decision = self._get_fallback_thresholds(market_context)
            fallback_decision.processing_time = time.time() - start_time
            return fallback_decision
            
        except Exception as e:
            logger.error(f"Error getting optimal thresholds: {e}")
            return self._get_fallback_thresholds({})
    
    async def _get_regime_based_thresholds(self, market_context: Dict[str, Any]) -> ThresholdDecision:
        """Get thresholds based on regime classification"""
        try:
            # Classify regime
            regime_classification = self.regime_detector.classify_regime(
                prices=market_context.get('prices', []),
                volumes=market_context.get('volumes', []),
                indicators=market_context.get('indicators', {})
            )
            
            # Get regime-specific thresholds
            regime_thresholds = self.regime_detector.get_regime_thresholds(regime_classification.regime)
            
            # Create decision
            decision = ThresholdDecision(
                volume_threshold=regime_thresholds.volume_threshold,
                trend_threshold=regime_thresholds.trend_threshold,
                confidence_threshold=regime_thresholds.confidence_threshold,
                decision_confidence=regime_classification.confidence,
                primary_method='regime',
                regime=regime_classification.regime,
                reasoning=f"Regime-based thresholds for {regime_classification.regime} (confidence: {regime_classification.confidence:.3f})",
                processing_time=time.time() - start_time,
                metadata={
                    'regime_confidence': regime_classification.confidence,
                    'cluster_id': regime_classification.cluster_id,
                    'distance_to_center': regime_classification.distance_to_center
                }
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Regime-based threshold error: {e}")
            raise
    
    async def _get_ensemble_thresholds(self, market_context: Dict[str, Any]) -> ThresholdDecision:
        """Get thresholds using ensemble of RL and LLM"""
        try:
            predictions = []
            weights = []
            
            # Get RL prediction
            if self.enable_rl and self.rl_agent:
                rl_prediction = await self._get_rl_prediction(market_context)
                if rl_prediction:
                    predictions.append(rl_prediction)
                    weights.append(self.ensemble_weights.get('rl', 0.3))
            
            # Get LLM prediction
            if self.enable_llm and self.llm_predictor:
                llm_prediction = await self._get_llm_prediction(market_context)
                if llm_prediction:
                    predictions.append(llm_prediction)
                    weights.append(self.ensemble_weights.get('llm', 0.3))
            
            if not predictions:
                raise ValueError("No AI predictions available")
            
            # Ensemble the predictions
            ensemble_thresholds = self._ensemble_predictions(predictions, weights)
            
            # Calculate ensemble confidence
            confidence = np.mean([pred.prediction_confidence for pred in predictions])
            
            # Create decision
            decision = ThresholdDecision(
                volume_threshold=ensemble_thresholds['volume'],
                trend_threshold=ensemble_thresholds['trend'],
                confidence_threshold=ensemble_thresholds['confidence'],
                decision_confidence=confidence,
                primary_method='ensemble',
                regime=market_context.get('regime', 'unknown'),
                reasoning=f"Ensemble prediction from {len(predictions)} AI models (confidence: {confidence:.3f})",
                processing_time=time.time() - start_time,
                metadata={
                    'num_predictions': len(predictions),
                    'prediction_methods': [pred.model_used for pred in predictions],
                    'individual_confidences': [pred.prediction_confidence for pred in predictions]
                }
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Ensemble threshold error: {e}")
            raise
    
    async def _get_rl_prediction(self, market_context: Dict[str, Any]) -> Optional[ThresholdPrediction]:
        """Get prediction from RL agent"""
        try:
            if self.rl_agent is None or self.rl_env is None:
                return None
            
            # Create state for RL agent
            state = self._create_rl_state(market_context)
            
            # Get action from RL agent
            action, _ = self.rl_agent.predict(state, deterministic=True)
            
            # Convert action to thresholds
            volume_threshold = float(action[0])
            trend_threshold = float(action[1])
            confidence_threshold = float(action[2])
            
            return ThresholdPrediction(
                volume_threshold=volume_threshold,
                trend_threshold=trend_threshold,
                confidence_threshold=confidence_threshold,
                prediction_confidence=0.7,  # Default confidence for RL
                reasoning="Reinforcement learning prediction",
                processing_time=0.001,
                model_used="stable_baselines3_ppo",
                metadata={'action': action.tolist()}
            )
            
        except Exception as e:
            logger.error(f"RL prediction error: {e}")
            return None
    
    async def _get_llm_prediction(self, market_context: Dict[str, Any]) -> Optional[ThresholdPrediction]:
        """Get prediction from LLM"""
        try:
            if self.llm_predictor is None:
                return None
            
            # Create market context for LLM
            llm_context = MarketContext(
                market_state=market_context.get('market_state', 'neutral'),
                volume=market_context.get('volume', 500.0),
                volatility=market_context.get('volatility', 0.02),
                trend_strength=market_context.get('trend_strength', 0.5),
                recent_performance=market_context.get('recent_performance', []),
                current_threshold=market_context.get('current_threshold', 0.6),
                market_regime=market_context.get('regime', 'normal')
            )
            
            # Get LLM prediction
            prediction = await self.llm_predictor.predict_thresholds(llm_context)
            
            return prediction
            
        except Exception as e:
            logger.error(f"LLM prediction error: {e}")
            return None
    
    def _ensemble_predictions(self, predictions: List[ThresholdPrediction], weights: List[float]) -> Dict[str, float]:
        """Ensemble multiple predictions using weighted average"""
        if not predictions:
            return {'volume': 500.0, 'trend': 0.5, 'confidence': 0.6}
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        volume_threshold = sum(pred.volume_threshold * weight for pred, weight in zip(predictions, normalized_weights))
        trend_threshold = sum(pred.trend_threshold * weight for pred, weight in zip(predictions, normalized_weights))
        confidence_threshold = sum(pred.confidence_threshold * weight for pred, weight in zip(predictions, normalized_weights))
        
        return {
            'volume': volume_threshold,
            'trend': trend_threshold,
            'confidence': confidence_threshold
        }
    
    def _get_fallback_thresholds(self, market_context: Dict[str, Any]) -> ThresholdDecision:
        """Get fallback thresholds when AI components are not available"""
        # Simple heuristic based on market context
        base_volume = 500.0
        base_trend = 0.5
        base_confidence = 0.6
        
        # Adjust based on volatility
        volatility = market_context.get('volatility', 0.02)
        if volatility > 0.03:
            base_confidence += 0.1
        elif volatility < 0.015:
            base_confidence -= 0.05
        
        # Adjust based on market state
        market_state = market_context.get('market_state', 'neutral')
        if market_state == 'bullish':
            base_volume += 100
        elif market_state == 'bearish':
            base_volume -= 100
        
        return ThresholdDecision(
            volume_threshold=max(100.0, min(1000.0, base_volume)),
            trend_threshold=max(0.1, min(0.9, base_trend)),
            confidence_threshold=max(0.1, min(0.9, base_confidence)),
            decision_confidence=0.3,
            primary_method='fallback',
            regime=market_context.get('regime', 'normal'),
            reasoning="Fallback heuristic thresholds",
            processing_time=0.001,
            metadata={'method': 'fallback_heuristic'}
        )
    
    def _extract_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market context from raw market data"""
        return {
            'prices': market_data.get('prices', []),
            'volumes': market_data.get('volumes', []),
            'indicators': market_data.get('indicators', {}),
            'volume': market_data.get('volume', 500.0),
            'volatility': market_data.get('volatility', 0.02),
            'trend_strength': market_data.get('trend_strength', 0.5),
            'market_state': market_data.get('market_state', 'neutral'),
            'regime': market_data.get('regime', 'normal'),
            'current_threshold': market_data.get('current_threshold', 0.6),
            'recent_performance': market_data.get('recent_performance', [])
        }
    
    def _create_rl_state(self, market_context: Dict[str, Any]) -> np.ndarray:
        """Create state vector for RL agent"""
        # Create state similar to ThresholdEnv
        recent_performance = market_context.get('recent_performance', [])
        
        state = np.array([
            market_context.get('volume', 500.0),
            np.mean(market_context.get('prices', [300.0])),
            market_context.get('volatility', 0.02),
            market_context.get('trend_strength', 0.5),
            self._regime_to_float(market_context.get('regime', 'normal')),
            np.mean(recent_performance[-10:]) if recent_performance else 0.5,
            np.std(recent_performance[-10:]) if recent_performance else 0.1,
            market_context.get('current_threshold', 0.6),
            0.1,  # False positive rate (placeholder)
            0.5   # True positive rate (placeholder)
        ], dtype=np.float32)
        
        return state
    
    def _regime_to_float(self, regime: str) -> float:
        """Convert regime string to float"""
        regime_map = {
            'low_volatility_ranging': 0.0,
            'normal_trending': 0.5,
            'high_volatility_breakout': 0.8,
            'consolidation': 0.2,
            'extreme_volatility': 1.0
        }
        return regime_map.get(regime, 0.5)
    
    async def _initialize_rl_agent(self):
        """Initialize the RL agent"""
        try:
            if not RL_AVAILABLE:
                logger.warning("Stable-baselines3 not available, RL disabled")
                return
            
            # Create environment
            self.rl_env = ThresholdEnv(reward_type="precision_recall")
            
            # Create PPO agent
            self.rl_agent = PPO(
                "MlpPolicy",
                self.rl_env,
                verbose=0,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
            
            # Start background training
            self.rl_training_task = asyncio.create_task(self._background_rl_training())
            
            logger.info("RL agent initialized successfully")
            
        except Exception as e:
            logger.error(f"RL agent initialization error: {e}")
            self.rl_agent = None
            self.rl_env = None
    
    async def _background_rl_training(self):
        """Background task for RL agent training"""
        while True:
            try:
                await asyncio.sleep(3600)  # Train every hour
                
                if self.rl_agent and self.rl_env:
                    # Train the agent
                    self.rl_agent.learn(total_timesteps=10000)
                    logger.info("RL agent training completed")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RL training error: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _background_decision_loop(self):
        """Background task for periodic threshold decisions"""
        while True:
            try:
                await asyncio.sleep(self.decision_interval)
                
                # Update thresholds based on recent performance
                if len(self.performance_history) > 10:
                    await self._update_thresholds_from_performance()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background decision error: {e}")
                await asyncio.sleep(60)
    
    async def _update_thresholds_from_performance(self):
        """Update thresholds based on recent performance"""
        try:
            recent_performance = list(self.performance_history)[-50:]
            
            # Calculate performance metrics
            success_rate = np.mean([p.performance_score for p in recent_performance])
            
            # Adjust thresholds based on performance
            if success_rate < 0.4:
                # Poor performance - lower thresholds
                self.current_thresholds.confidence_threshold *= 0.95
                self.current_thresholds.volume_threshold *= 0.95
            elif success_rate > 0.8:
                # Good performance - raise thresholds
                self.current_thresholds.confidence_threshold *= 1.05
                self.current_thresholds.volume_threshold *= 1.05
            
            # Ensure bounds
            self.current_thresholds.confidence_threshold = max(0.1, min(0.9, self.current_thresholds.confidence_threshold))
            self.current_thresholds.volume_threshold = max(100.0, min(1000.0, self.current_thresholds.volume_threshold))
            
            logger.info(f"Thresholds updated based on performance (success_rate: {success_rate:.3f})")
            
        except Exception as e:
            logger.error(f"Threshold update error: {e}")
    
    def record_performance(self, 
                          thresholds: ThresholdDecision,
                          signal_confidence: float,
                          signal_passed: bool,
                          actual_outcome: bool):
        """Record performance for threshold optimization"""
        try:
            # Calculate performance score
            if signal_passed and actual_outcome:
                performance_score = 1.0  # True positive
            elif signal_passed and not actual_outcome:
                performance_score = 0.0  # False positive
            elif not signal_passed and actual_outcome:
                performance_score = 0.5  # False negative
            else:
                performance_score = 0.8  # True negative
            
            # Create performance record
            performance = ThresholdPerformance(
                timestamp=datetime.now(),
                thresholds=thresholds,
                signal_confidence=signal_confidence,
                signal_passed=signal_passed,
                actual_outcome=actual_outcome,
                performance_score=performance_score
            )
            
            # Add to history
            self.performance_history.append(performance)
            
            # Update RL environment if available
            if self.rl_env and self.rl_agent:
                self._update_rl_environment(performance)
                
        except Exception as e:
            logger.error(f"Performance recording error: {e}")
    
    def _update_rl_environment(self, performance: ThresholdPerformance):
        """Update RL environment with performance data"""
        try:
            # This would update the RL environment with the performance data
            # For now, we'll just log it
            logger.debug(f"RL environment updated with performance: {performance.performance_score}")
            
        except Exception as e:
            logger.error(f"RL environment update error: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'total_decisions': len(self.decision_history),
            'total_performance_records': len(self.performance_history),
            'current_thresholds': {
                'volume': self.current_thresholds.volume_threshold,
                'trend': self.current_thresholds.trend_threshold,
                'confidence': self.current_thresholds.confidence_threshold
            },
            'last_decision': self.last_decision.isoformat() if self.last_decision else None
        }
        
        # Add component-specific metrics
        if self.regime_detector:
            metrics['regime_detector'] = self.regime_detector.get_performance_metrics()
        
        if self.llm_predictor:
            metrics['llm_predictor'] = self.llm_predictor.get_performance_metrics()
        
        # Add performance statistics
        if self.performance_history:
            recent_performance = list(self.performance_history)[-100:]
            metrics['performance_stats'] = {
                'avg_performance': np.mean([p.performance_score for p in recent_performance]),
                'success_rate': np.mean([p.performance_score > 0.5 for p in recent_performance]),
                'total_signals': len(recent_performance)
            }
        
        return metrics

# Global AI-driven threshold manager instance
ai_driven_threshold_manager = AIDrivenThresholdManager(
    enable_rl=True,
    enable_llm=True,
    enable_regime_detection=True,
    decision_interval=60,
    performance_window=1000,
    ensemble_weights={
        'regime': 0.4,
        'rl': 0.3,
        'llm': 0.3
    }
)
