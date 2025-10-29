"""
Minimum Confidence Threshold with AI-Driven Dynamic Adjustment for AlphaPulse
Phase 2: Intelligent threshold management using reinforcement learning and Bayesian optimization
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import json
import hashlib

# Machine learning imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)

class ThresholdAdjustmentType(Enum):
    """Types of threshold adjustments"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    MARKET_REGIME = "market_regime"
    ENSEMBLE = "ensemble"

@dataclass
class ThresholdState:
    """Current threshold state"""
    base_threshold: float = 0.6
    adjusted_threshold: float = 0.6
    confidence_score: float = 0.0
    market_regime: str = "normal"
    volatility_level: float = 0.5
    adjustment_factor: float = 1.0
    last_adjustment: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThresholdAdjustment:
    """Threshold adjustment result"""
    adjustment_type: ThresholdAdjustmentType
    old_threshold: float
    new_threshold: float
    adjustment_reason: str
    confidence_gain: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MarketRegimeClassifier:
    """Market regime classification using K-Means clustering"""
    
    def __init__(self, n_clusters: int = 3, update_interval: int = 300):
        self.n_clusters = n_clusters
        self.update_interval = update_interval
        self.kmeans = None
        self.scaler = StandardScaler()
        self.regime_labels = ["low_volatility", "normal", "high_volatility"]
        
        # Market data buffer
        self.price_data = deque(maxlen=1000)
        self.volume_data = deque(maxlen=1000)
        self.volatility_data = deque(maxlen=1000)
        
        # Performance tracking
        self.last_update = datetime.now()
        self.classification_count = 0
        
        logger.info(f"Market Regime Classifier initialized with {n_clusters} clusters")
    
    def add_market_data(self, price: float, volume: float, volatility: float):
        """Add market data for regime classification"""
        self.price_data.append(price)
        self.volume_data.append(volume)
        self.volatility_data.append(volatility)
    
    def classify_regime(self) -> Dict[str, Any]:
        """Classify current market regime"""
        if len(self.price_data) < 50:
            return {
                'regime': 'normal',
                'confidence': 0.5,
                'volatility_level': 0.5,
                'features': {}
            }
        
        try:
            # Prepare features
            features = self._extract_features()
            
            if self.kmeans is None or (datetime.now() - self.last_update).seconds > self.update_interval:
                self._update_model(features)
            
            # Classify current regime
            if self.kmeans is not None:
                scaled_features = self.scaler.transform([features])
                cluster = self.kmeans.predict(scaled_features)[0]
                regime = self.regime_labels[cluster] if cluster < len(self.regime_labels) else "normal"
                
                # Calculate confidence based on distance to cluster center
                distance = np.linalg.norm(scaled_features[0] - self.kmeans.cluster_centers_[cluster])
                confidence = max(0.1, 1.0 - distance / 10.0)
                
                self.classification_count += 1
                
                return {
                    'regime': regime,
                    'confidence': confidence,
                    'volatility_level': features[2],  # Volatility feature
                    'features': {
                        'price_momentum': features[0],
                        'volume_trend': features[1],
                        'volatility': features[2]
                    }
                }
            
            return {
                'regime': 'normal',
                'confidence': 0.5,
                'volatility_level': 0.5,
                'features': {}
            }
            
        except Exception as e:
            logger.error(f"Market regime classification error: {e}")
            return {
                'regime': 'normal',
                'confidence': 0.5,
                'volatility_level': 0.5,
                'features': {}
            }
    
    def _extract_features(self) -> List[float]:
        """Extract features for regime classification"""
        prices = list(self.price_data)
        volumes = list(self.volume_data)
        volatilities = list(self.volatility_data)
        
        # Price momentum (rate of change)
        if len(prices) >= 20:
            price_momentum = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0.0
        else:
            price_momentum = 0.0
        
        # Volume trend
        if len(volumes) >= 20:
            volume_trend = np.mean(volumes[-10:]) / np.mean(volumes[-20:-10]) if np.mean(volumes[-20:-10]) > 0 else 1.0
        else:
            volume_trend = 1.0
        
        # Volatility level
        volatility_level = np.mean(volatilities[-20:]) if volatilities else 0.5
        
        return [price_momentum, volume_trend, volatility_level]
    
    def _update_model(self, features: List[float]):
        """Update the K-Means model"""
        try:
            if len(self.price_data) < 100:
                return
            
            # Prepare training data
            all_features = []
            for i in range(0, len(self.price_data) - 20, 10):
                if i + 20 <= len(self.price_data):
                    price_momentum = (self.price_data[i+19] - self.price_data[i]) / self.price_data[i] if self.price_data[i] > 0 else 0.0
                    volume_trend = np.mean(list(self.volume_data)[i+10:i+20]) / np.mean(list(self.volume_data)[i:i+10]) if np.mean(list(self.volume_data)[i:i+10]) > 0 else 1.0
                    volatility = np.mean(list(self.volatility_data)[i:i+20]) if self.volatility_data else 0.5
                    all_features.append([price_momentum, volume_trend, volatility])
            
            if len(all_features) < self.n_clusters:
                return
            
            # Fit model
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(all_features)
            
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            self.kmeans.fit(scaled_features)
            
            self.last_update = datetime.now()
            logger.info(f"Market regime model updated with {len(all_features)} samples")
            
        except Exception as e:
            logger.error(f"Model update error: {e}")

class ReinforcementLearningAgent:
    """Reinforcement learning agent for threshold optimization"""
    
    def __init__(self, 
                learning_rate: float = 0.1,
                discount_factor: float = 0.95,
                epsilon: float = 0.1,
                state_size: int = 10,
                action_size: int = 21):
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-table (state -> action -> value)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Performance tracking
        self.total_rewards = 0
        self.episode_count = 0
        self.successful_adjustments = 0
        
        logger.info("Reinforcement Learning Agent initialized")
    
    def get_action(self, state: Tuple[float, ...]) -> int:
        """Get action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.action_size)
        else:
            # Exploitation: best action
            state_key = self._state_to_key(state)
            actions = self.q_table[state_key]
            if actions:
                return max(actions.keys(), key=lambda k: actions[k])
            else:
                return self.action_size // 2  # Default to middle action
    
    def update_q_value(self, state: Tuple[float, ...], action: int, reward: float, next_state: Tuple[float, ...]):
        """Update Q-value using Q-learning"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for next state
        next_actions = self.q_table[next_state_key]
        max_next_q = max(next_actions.values()) if next_actions else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Store experience
        self.experience_buffer.append((state, action, reward, next_state))
        
        # Update statistics
        self.total_rewards += reward
        if reward > 0:
            self.successful_adjustments += 1
    
    def _state_to_key(self, state: Tuple[float, ...]) -> str:
        """Convert state tuple to string key"""
        return json.dumps([round(x, 3) for x in state])
    
    def get_threshold_adjustment(self, current_threshold: float, market_regime: str, volatility: float, 
                               recent_performance: List[float]) -> float:
        """Get threshold adjustment based on current state"""
        # Create state representation
        state = (
            current_threshold,
            self._regime_to_float(market_regime),
            volatility,
            np.mean(recent_performance[-10:]) if recent_performance else 0.5,
            np.std(recent_performance[-10:]) if recent_performance else 0.1
        )
        
        # Get action
        action = self.get_action(state)
        
        # Convert action to threshold adjustment
        # Action 0-20 maps to -0.1 to +0.1 adjustment
        adjustment = (action - 10) * 0.01
        
        return max(0.1, min(0.9, current_threshold + adjustment))
    
    def _regime_to_float(self, regime: str) -> float:
        """Convert regime string to float"""
        regime_map = {
            'low_volatility': 0.0,
            'normal': 0.5,
            'high_volatility': 1.0
        }
        return regime_map.get(regime, 0.5)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get RL agent performance metrics"""
        return {
            'total_rewards': self.total_rewards,
            'episode_count': self.episode_count,
            'successful_adjustments': self.successful_adjustments,
            'success_rate': self.successful_adjustments / max(1, self.episode_count),
            'avg_reward': self.total_rewards / max(1, self.episode_count),
            'q_table_size': len(self.q_table),
            'experience_buffer_size': len(self.experience_buffer)
        }

class BayesianOptimizer:
    """Bayesian optimization for threshold tuning"""
    
    def __init__(self, study_name: str = "threshold_optimization"):
        self.study_name = study_name
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
        if OPTUNA_AVAILABLE:
            try:
                self.study = optuna.create_study(
                    direction="maximize",
                    study_name=study_name,
                    storage=None  # In-memory storage
                )
                logger.info("Bayesian Optimizer initialized with Optuna")
            except Exception as e:
                logger.error(f"Failed to initialize Optuna study: {e}")
        else:
            logger.warning("Optuna not available, Bayesian optimization disabled")
    
    def suggest_threshold(self, market_regime: str, volatility: float, recent_performance: List[float]) -> float:
        """Suggest optimal threshold using Bayesian optimization"""
        if not OPTUNA_AVAILABLE or self.study is None:
            # Fallback to simple heuristic
            base_threshold = 0.6
            if market_regime == "high_volatility":
                base_threshold += 0.1
            elif market_regime == "low_volatility":
                base_threshold -= 0.05
            
            if volatility > 0.7:
                base_threshold += 0.05
            
            return max(0.1, min(0.9, base_threshold))
        
        try:
            # Create objective function for this context
            def objective(trial):
                threshold = trial.suggest_float("threshold", 0.1, 0.9)
                
                # Simulate performance based on recent data
                if recent_performance:
                    # Simple simulation: higher threshold should filter more signals
                    filtered_signals = [p for p in recent_performance if p >= threshold]
                    if filtered_signals:
                        return np.mean(filtered_signals)
                    else:
                        return 0.0
                else:
                    return 0.5
            
            # Run optimization
            self.study.optimize(objective, n_trials=10)
            
            # Get best threshold
            best_threshold = self.study.best_params.get("threshold", 0.6)
            
            # Store optimization result
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'market_regime': market_regime,
                'volatility': volatility,
                'suggested_threshold': best_threshold,
                'best_value': self.study.best_value
            })
            
            return best_threshold
            
        except Exception as e:
            logger.error(f"Bayesian optimization error: {e}")
            return 0.6
    
    def update_performance(self, threshold: float, actual_performance: float):
        """Update optimization with actual performance"""
        if self.study is not None:
            try:
                # Create a new trial with the actual result
                trial = self.study.ask()
                trial.suggest_float("threshold", threshold, threshold)
                self.study.tell(trial, actual_performance)
                
                logger.debug(f"Updated Bayesian optimizer with performance: {actual_performance}")
            except Exception as e:
                logger.error(f"Failed to update Bayesian optimizer: {e}")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not OPTUNA_AVAILABLE or self.study is None:
            return {'available': False}
        
        return {
            'available': True,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'optimization_history': len(self.optimization_history)
        }

class MinimumConfidenceThreshold:
    """Main minimum confidence threshold manager with AI-driven adjustments"""
    
    def __init__(self, 
                base_threshold: float = 0.6,
                adjustment_interval: int = 300,
                enable_rl: bool = True,
                enable_bayesian: bool = True,
                enable_regime_classification: bool = True):
        
        self.base_threshold = base_threshold
        self.adjustment_interval = adjustment_interval
        self.current_threshold = base_threshold
        
        # Enable/disable components
        self.enable_rl = enable_rl and ML_AVAILABLE
        self.enable_bayesian = enable_bayesian and OPTUNA_AVAILABLE
        self.enable_regime_classification = enable_regime_classification and ML_AVAILABLE
        
        # Initialize components
        self.market_classifier = MarketRegimeClassifier() if self.enable_regime_classification else None
        self.rl_agent = ReinforcementLearningAgent() if self.enable_rl else None
        self.bayesian_optimizer = BayesianOptimizer() if self.enable_bayesian else None
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adjustment_history = deque(maxlen=100)
        self.last_adjustment = datetime.now()
        
        # Statistics
        self.total_signals = 0
        self.passed_signals = 0
        self.failed_signals = 0
        
        logger.info(f"Minimum Confidence Threshold initialized (RL: {self.enable_rl}, Bayesian: {self.enable_bayesian}, Regime: {self.enable_regime_classification})")
    
    async def evaluate_signal(self, signal_confidence: float, market_data: Dict[str, Any] = None) -> bool:
        """Evaluate signal against current threshold"""
        self.total_signals += 1
        
        # Update market data
        if market_data and self.market_classifier:
            self.market_classifier.add_market_data(
                market_data.get('price', 0.0),
                market_data.get('volume', 0.0),
                market_data.get('volatility', 0.5)
            )
        
        # Check against threshold
        passed = signal_confidence >= self.current_threshold
        
        if passed:
            self.passed_signals += 1
        else:
            self.failed_signals += 1
        
        # Store performance
        self.performance_history.append(signal_confidence)
        
        # Periodic threshold adjustment
        if (datetime.now() - self.last_adjustment).seconds >= self.adjustment_interval:
            await self._adjust_threshold()
        
        return passed
    
    async def _adjust_threshold(self):
        """Adjust threshold using AI-driven methods"""
        start_time = time.time()
        old_threshold = self.current_threshold
        
        try:
            # Get market regime classification
            market_regime = "normal"
            volatility = 0.5
            if self.market_classifier:
                regime_info = self.market_classifier.classify_regime()
                market_regime = regime_info['regime']
                volatility = regime_info['volatility_level']
            
            # Get recent performance
            recent_performance = list(self.performance_history)[-100:] if self.performance_history else []
            
            # Collect adjustments from different methods
            adjustments = []
            
            # Reinforcement learning adjustment
            if self.rl_agent and recent_performance:
                rl_threshold = self.rl_agent.get_threshold_adjustment(
                    self.current_threshold, market_regime, volatility, recent_performance
                )
                adjustments.append(('rl', rl_threshold))
            
            # Bayesian optimization adjustment
            if self.bayesian_optimizer:
                bayesian_threshold = self.bayesian_optimizer.suggest_threshold(
                    market_regime, volatility, recent_performance
                )
                adjustments.append(('bayesian', bayesian_threshold))
            
            # Ensemble adjustment (weighted average)
            if adjustments:
                weights = {'rl': 0.4, 'bayesian': 0.6}
                weighted_sum = 0.0
                total_weight = 0.0
                
                for method, threshold in adjustments:
                    weight = weights.get(method, 0.5)
                    weighted_sum += threshold * weight
                    total_weight += weight
                
                if total_weight > 0:
                    new_threshold = weighted_sum / total_weight
                else:
                    new_threshold = self.current_threshold
            else:
                # Fallback to simple heuristic
                new_threshold = self._heuristic_adjustment(market_regime, volatility, recent_performance)
            
            # Apply adjustment with bounds
            self.current_threshold = max(0.1, min(0.9, new_threshold))
            
            # Record adjustment
            adjustment = ThresholdAdjustment(
                adjustment_type=ThresholdAdjustmentType.ENSEMBLE,
                old_threshold=old_threshold,
                new_threshold=self.current_threshold,
                adjustment_reason=f"Market regime: {market_regime}, Volatility: {volatility:.3f}",
                confidence_gain=abs(new_threshold - old_threshold),
                processing_time=time.time() - start_time,
                metadata={
                    'market_regime': market_regime,
                    'volatility': volatility,
                    'adjustments': adjustments
                }
            )
            
            self.adjustment_history.append(adjustment)
            self.last_adjustment = datetime.now()
            
            logger.info(f"Threshold adjusted: {old_threshold:.3f} -> {self.current_threshold:.3f} ({adjustment.adjustment_reason})")
            
        except Exception as e:
            logger.error(f"Threshold adjustment error: {e}")
    
    def _heuristic_adjustment(self, market_regime: str, volatility: float, recent_performance: List[float]) -> float:
        """Simple heuristic threshold adjustment"""
        base_threshold = self.base_threshold
        
        # Market regime adjustments
        if market_regime == "high_volatility":
            base_threshold += 0.1
        elif market_regime == "low_volatility":
            base_threshold -= 0.05
        
        # Volatility adjustments
        if volatility > 0.7:
            base_threshold += 0.05
        elif volatility < 0.3:
            base_threshold -= 0.03
        
        # Performance-based adjustments
        if recent_performance:
            avg_performance = np.mean(recent_performance)
            if avg_performance < 0.4:
                base_threshold -= 0.05  # Lower threshold if performance is poor
            elif avg_performance > 0.8:
                base_threshold += 0.03  # Raise threshold if performance is excellent
        
        return base_threshold
    
    def get_current_state(self) -> ThresholdState:
        """Get current threshold state"""
        market_regime = "normal"
        volatility = 0.5
        
        if self.market_classifier:
            regime_info = self.market_classifier.classify_regime()
            market_regime = regime_info['regime']
            volatility = regime_info['volatility_level']
        
        return ThresholdState(
            base_threshold=self.base_threshold,
            adjusted_threshold=self.current_threshold,
            confidence_score=self.passed_signals / max(1, self.total_signals),
            market_regime=market_regime,
            volatility_level=volatility,
            adjustment_factor=self.current_threshold / self.base_threshold,
            last_adjustment=self.last_adjustment,
            metadata={
                'total_signals': self.total_signals,
                'passed_signals': self.passed_signals,
                'failed_signals': self.failed_signals
            }
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'current_threshold': self.current_threshold,
            'base_threshold': self.base_threshold,
            'total_signals': self.total_signals,
            'passed_signals': self.passed_signals,
            'failed_signals': self.failed_signals,
            'pass_rate': self.passed_signals / max(1, self.total_signals),
            'adjustment_count': len(self.adjustment_history),
            'last_adjustment': self.last_adjustment.isoformat() if self.last_adjustment else None
        }
        
        # Add component-specific metrics
        if self.market_classifier:
            metrics['market_classifier'] = {
                'classification_count': self.market_classifier.classification_count
            }
        
        if self.rl_agent:
            metrics['reinforcement_learning'] = self.rl_agent.get_performance_metrics()
        
        if self.bayesian_optimizer:
            metrics['bayesian_optimization'] = self.bayesian_optimizer.get_optimization_stats()
        
        return metrics

# Global minimum confidence threshold instance
minimum_confidence_threshold = MinimumConfidenceThreshold(
    base_threshold=0.6,
    adjustment_interval=300,
    enable_rl=True,
    enable_bayesian=True,
    enable_regime_classification=True
)
