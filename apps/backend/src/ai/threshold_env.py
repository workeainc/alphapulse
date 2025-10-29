"""
Threshold Environment for Reinforcement Learning
Gym environment for optimizing signal validation thresholds
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    """Market state representation"""
    volume: float
    price: float
    volatility: float
    trend_strength: float
    market_regime: str
    recent_performance: List[float]
    current_threshold: float

@dataclass
class ThresholdAction:
    """Threshold adjustment action"""
    volume_threshold: float
    trend_threshold: float
    confidence_threshold: float

class ThresholdEnv(gym.Env):
    """
    Gym environment for threshold optimization using reinforcement learning.
    
    State: Market data features, filter outputs, and pipeline performance metrics
    Action: Adjust thresholds (volume, trend, confidence) within bounds
    Reward: Balance signal precision and recall, or simulated trading profit
    """
    
    def __init__(self, 
                 data_source: Optional[pd.DataFrame] = None,
                 reward_type: str = "precision_recall",
                 action_bounds: Dict[str, Tuple[float, float]] = None):
        super().__init__()
        
        self.data_source = data_source
        self.reward_type = reward_type
        
        # Action space: [volume_threshold, trend_threshold, confidence_threshold]
        # Default bounds: volume [0, 1000], trend [0, 1], confidence [0.1, 0.9]
        default_bounds = {
            'volume': (0.0, 1000.0),
            'trend': (0.0, 1.0),
            'confidence': (0.1, 0.9)
        }
        if action_bounds:
            default_bounds.update(action_bounds)
        
        self.action_space = spaces.Box(
            low=np.array([default_bounds['volume'][0], 
                         default_bounds['trend'][0], 
                         default_bounds['confidence'][0]]),
            high=np.array([default_bounds['volume'][1], 
                          default_bounds['trend'][1], 
                          default_bounds['confidence'][1]]),
            dtype=np.float32
        )
        
        # Observation space: [volume, price, volatility, trend_strength, 
        #                    market_regime, recent_performance_mean, recent_performance_std,
        #                    current_threshold, false_positive_rate, true_positive_rate]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,), 
            dtype=np.float32
        )
        
        # Environment state
        self.current_step = 0
        self.current_thresholds = ThresholdAction(
            volume_threshold=500.0,
            trend_threshold=0.5,
            confidence_threshold=0.6
        )
        
        # Performance tracking
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        # History for reward calculation
        self.signal_history = []
        self.performance_history = []
        
        logger.info(f"ThresholdEnv initialized with reward_type: {reward_type}")
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_thresholds = ThresholdAction(
            volume_threshold=500.0,
            trend_threshold=0.5,
            confidence_threshold=0.6
        )
        
        # Reset performance tracking
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        self.signal_history = []
        self.performance_history = []
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment
        
        Args:
            action: Array of [volume_threshold, trend_threshold, confidence_threshold]
            
        Returns:
            observation: Current state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        # Update thresholds
        self.current_thresholds = ThresholdAction(
            volume_threshold=float(action[0]),
            trend_threshold=float(action[1]),
            confidence_threshold=float(action[2])
        )
        
        # Get current market data
        if self.data_source is not None and self.current_step < len(self.data_source):
            data_point = self.data_source.iloc[self.current_step]
        else:
            # Generate synthetic data for testing
            data_point = self._generate_synthetic_data()
        
        # Apply filters and get signal
        signal_result = self._apply_filters(data_point, self.current_thresholds)
        
        # Calculate reward
        reward = self._compute_reward(signal_result, data_point)
        
        # Update performance tracking
        self._update_performance_tracking(signal_result, data_point)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = False
        if self.data_source is not None:
            done = self.current_step >= len(self.data_source)
        else:
            # For synthetic data, end after 1000 steps
            done = self.current_step >= 1000
        
        # Get current state
        observation = self._get_state()
        
        # Prepare info
        info = {
            'thresholds': {
                'volume': self.current_thresholds.volume_threshold,
                'trend': self.current_thresholds.trend_threshold,
                'confidence': self.current_thresholds.confidence_threshold
            },
            'signal_result': signal_result,
            'performance': {
                'precision': self._calculate_precision(),
                'recall': self._calculate_recall(),
                'f1_score': self._calculate_f1_score()
            }
        }
        
        return observation, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Get recent performance metrics
        recent_performance = self.performance_history[-20:] if self.performance_history else [0.5]
        
        # Calculate market regime (simplified)
        market_regime = self._calculate_market_regime()
        
        # Create state vector
        state = np.array([
            self._get_current_volume(),      # Current volume
            self._get_current_price(),       # Current price
            self._get_current_volatility(),  # Current volatility
            self._get_current_trend(),       # Current trend strength
            self._regime_to_float(market_regime),  # Market regime
            np.mean(recent_performance),     # Recent performance mean
            np.std(recent_performance) if len(recent_performance) > 1 else 0.1,  # Recent performance std
            self.current_thresholds.confidence_threshold,  # Current threshold
            self._calculate_false_positive_rate(),  # False positive rate
            self._calculate_true_positive_rate()    # True positive rate
        ], dtype=np.float32)
        
        return state
    
    def _apply_filters(self, data_point: pd.Series, thresholds: ThresholdAction) -> Dict[str, Any]:
        """Apply filters to data point using current thresholds"""
        # Extract features
        volume = data_point.get('volume', 0.0)
        price = data_point.get('close', 0.0)
        trend = data_point.get('trend', 0.0)
        confidence = data_point.get('confidence', 0.5)
        
        # Apply filters
        volume_passed = volume > thresholds.volume_threshold
        trend_passed = abs(trend) > thresholds.trend_threshold
        confidence_passed = confidence > thresholds.confidence_threshold
        
        # Overall signal
        signal_passed = volume_passed and trend_passed and confidence_passed
        
        # Calculate overall confidence
        overall_confidence = (confidence + (1.0 if volume_passed else 0.0) + 
                            (1.0 if trend_passed else 0.0)) / 3.0
        
        return {
            'signal_passed': signal_passed,
            'volume_passed': volume_passed,
            'trend_passed': trend_passed,
            'confidence_passed': confidence_passed,
            'overall_confidence': overall_confidence,
            'volume': volume,
            'trend': trend,
            'confidence': confidence,
            'is_valid_signal': data_point.get('is_valid_signal', False)
        }
    
    def _compute_reward(self, signal_result: Dict[str, Any], data_point: pd.Series) -> float:
        """Compute reward based on signal result and actual outcome"""
        if self.reward_type == "precision_recall":
            return self._compute_precision_recall_reward(signal_result, data_point)
        elif self.reward_type == "trading_profit":
            return self._compute_trading_profit_reward(signal_result, data_point)
        else:
            return self._compute_simple_reward(signal_result, data_point)
    
    def _compute_precision_recall_reward(self, signal_result: Dict[str, Any], data_point: pd.Series) -> float:
        """Compute reward based on precision and recall"""
        signal_passed = signal_result['signal_passed']
        is_valid = signal_result['is_valid_signal']
        
        if signal_passed and is_valid:
            # True positive
            reward = 1.0
        elif signal_passed and not is_valid:
            # False positive
            reward = -0.5
        elif not signal_passed and is_valid:
            # False negative
            reward = -0.3
        else:
            # True negative
            reward = 0.1
        
        # Add confidence bonus
        confidence_bonus = signal_result['overall_confidence'] * 0.2
        reward += confidence_bonus
        
        return reward
    
    def _compute_trading_profit_reward(self, signal_result: Dict[str, Any], data_point: pd.Series) -> float:
        """Compute reward based on simulated trading profit"""
        if not signal_result['signal_passed']:
            return 0.0  # No trade, no profit/loss
        
        # Simulate trade outcome
        is_valid = signal_result['is_valid_signal']
        
        if is_valid:
            # Successful trade
            profit = data_point.get('profit', 0.02)  # 2% default profit
            reward = profit
        else:
            # Failed trade
            loss = data_point.get('loss', -0.01)  # 1% default loss
            reward = loss
        
        return reward
    
    def _compute_simple_reward(self, signal_result: Dict[str, Any], data_point: pd.Series) -> float:
        """Simple reward function"""
        if signal_result['signal_passed'] and signal_result['is_valid_signal']:
            return 1.0
        elif signal_result['signal_passed'] and not signal_result['is_valid_signal']:
            return -1.0
        else:
            return 0.0
    
    def _update_performance_tracking(self, signal_result: Dict[str, Any], data_point: pd.Series):
        """Update performance tracking metrics"""
        signal_passed = signal_result['signal_passed']
        is_valid = signal_result['is_valid_signal']
        
        if signal_passed and is_valid:
            self.true_positives += 1
        elif signal_passed and not is_valid:
            self.false_positives += 1
        elif not signal_passed and is_valid:
            self.false_negatives += 1
        else:
            self.true_negatives += 1
        
        # Store performance
        performance = 1.0 if signal_passed and is_valid else 0.0
        self.performance_history.append(performance)
        
        # Keep history manageable
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    def _calculate_precision(self) -> float:
        """Calculate precision"""
        total_positive = self.true_positives + self.false_positives
        return self.true_positives / total_positive if total_positive > 0 else 0.0
    
    def _calculate_recall(self) -> float:
        """Calculate recall"""
        total_actual_positive = self.true_positives + self.false_negatives
        return self.true_positives / total_actual_positive if total_actual_positive > 0 else 0.0
    
    def _calculate_f1_score(self) -> float:
        """Calculate F1 score"""
        precision = self._calculate_precision()
        recall = self._calculate_recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_false_positive_rate(self) -> float:
        """Calculate false positive rate"""
        total_negative = self.false_positives + self.true_negatives
        return self.false_positives / total_negative if total_negative > 0 else 0.0
    
    def _calculate_true_positive_rate(self) -> float:
        """Calculate true positive rate (same as recall)"""
        return self._calculate_recall()
    
    def _generate_synthetic_data(self) -> pd.Series:
        """Generate synthetic market data for testing"""
        # Simulate market data
        volume = np.random.uniform(100, 1000)
        price = np.random.uniform(100, 500)
        volatility = np.random.uniform(0.01, 0.05)
        trend = np.random.uniform(-0.1, 0.1)
        confidence = np.random.uniform(0.3, 0.9)
        
        # Simulate signal validity (60% valid signals)
        is_valid_signal = np.random.random() < 0.6
        
        return pd.Series({
            'volume': volume,
            'close': price,
            'trend': trend,
            'confidence': confidence,
            'is_valid_signal': is_valid_signal,
            'profit': np.random.uniform(0.01, 0.05) if is_valid_signal else 0.0,
            'loss': np.random.uniform(-0.03, -0.01) if not is_valid_signal else 0.0
        })
    
    def _get_current_volume(self) -> float:
        """Get current volume (simplified)"""
        return 500.0  # Default value
    
    def _get_current_price(self) -> float:
        """Get current price (simplified)"""
        return 300.0  # Default value
    
    def _get_current_volatility(self) -> float:
        """Get current volatility (simplified)"""
        return 0.02  # Default value
    
    def _get_current_trend(self) -> float:
        """Get current trend strength (simplified)"""
        return 0.1  # Default value
    
    def _calculate_market_regime(self) -> str:
        """Calculate current market regime (simplified)"""
        regimes = ['low_volatility', 'normal', 'high_volatility']
        return np.random.choice(regimes, p=[0.3, 0.5, 0.2])
    
    def _regime_to_float(self, regime: str) -> float:
        """Convert regime string to float"""
        regime_map = {
            'low_volatility': 0.0,
            'normal': 0.5,
            'high_volatility': 1.0
        }
        return regime_map.get(regime, 0.5)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'total_steps': self.current_step,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'precision': self._calculate_precision(),
            'recall': self._calculate_recall(),
            'f1_score': self._calculate_f1_score(),
            'current_thresholds': {
                'volume': self.current_thresholds.volume_threshold,
                'trend': self.current_thresholds.trend_threshold,
                'confidence': self.current_thresholds.confidence_threshold
            }
        }
