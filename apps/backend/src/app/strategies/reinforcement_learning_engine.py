"""
Reinforcement Learning Engine for AlphaPulse
Comprehensive RL implementation with multiple agents and environments for trading strategy optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import warnings
warnings.filterwarnings('ignore')

# RL imports
try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    logging.warning("Gym not available - using mock environment")

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    STABLE_BASELINES3_AVAILABLE = True
except ImportError:
    STABLE_BASELINES3_AVAILABLE = False
    logging.warning("Stable-baselines3 not available - using mock agents")

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """RL Agent types"""
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"
    CUSTOM = "custom"

class EnvironmentType(Enum):
    """RL Environment types"""
    TRADING = "trading"
    SIGNAL_OPTIMIZATION = "signal_optimization"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO = "portfolio"

@dataclass
class RLState:
    """Reinforcement Learning state representation"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    volatility: float
    trend_strength: float
    market_regime: str
    position_size: float
    current_pnl: float
    risk_metrics: Dict[str, float]
    signal_strength: float
    confidence: float
    market_features: List[float]

@dataclass
class RLAction:
    """Reinforcement Learning action representation"""
    action_type: str  # 'buy', 'sell', 'hold', 'adjust_position'
    position_size: float
    stop_loss: float
    take_profit: float
    confidence_threshold: float
    risk_allocation: float

@dataclass
class RLResult:
    """Reinforcement Learning result"""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool
    info: Dict[str, Any]

if GYM_AVAILABLE:
    class TradingEnvironment(gym.Env):
        """Trading environment for reinforcement learning"""
        
        def __init__(self, 
                     data_source: Optional[pd.DataFrame] = None,
                     initial_balance: float = 10000.0,
                     transaction_cost: float = 0.001,
                     max_position_size: float = 0.5):
            super().__init__()
else:
    class TradingEnvironment:
        """Mock Trading environment for reinforcement learning (gym not available)"""
        
        def __init__(self, 
                     data_source: Optional[pd.DataFrame] = None,
                     initial_balance: float = 10000.0,
                     transaction_cost: float = 0.001,
                     max_position_size: float = 0.5):
            
            self.data_source = data_source
            self.initial_balance = initial_balance
            self.transaction_cost = transaction_cost
            self.max_position_size = max_position_size
            
            # Environment state
            self.current_step = 0
            self.balance = initial_balance
            self.position = 0.0
            self.current_price = 0.0
            self.total_pnl = 0.0
            self.trades_count = 0
            self.winning_trades = 0
            
            # Action space: [action_type, position_size, stop_loss, take_profit, confidence_threshold]
            # action_type: 0=hold, 1=buy, 2=sell
            # position_size: 0.0 to max_position_size
            # stop_loss: 0.01 to 0.1 (1% to 10%)
            # take_profit: 0.01 to 0.2 (1% to 20%)
            # confidence_threshold: 0.5 to 0.95
            if GYM_AVAILABLE:
                self.action_space = spaces.Box(
                    low=np.array([0.0, 0.0, 0.01, 0.01, 0.5]),
                    high=np.array([2.0, max_position_size, 0.1, 0.2, 0.95]),
                    dtype=np.float32
                )
                
                # Observation space: [price, volume, volatility, trend_strength, position, balance, pnl, confidence]
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(8,),
                    dtype=np.float32
                )
            
            logger.info("Trading environment initialized")
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.winning_trades = 0
        
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        
        # Parse action
        action_type = int(action[0])
        position_size = float(action[1])
        stop_loss = float(action[2])
        take_profit = float(action[3])
        confidence_threshold = float(action[4])
        
        # Get current market data
        if self.data_source is not None and self.current_step < len(self.data_source):
            data_point = self.data_source.iloc[self.current_step]
            self.current_price = float(data_point['close'])
            volume = float(data_point['volume'])
            volatility = float(data_point.get('volatility', 0.02))
            trend_strength = float(data_point.get('trend_strength', 0.5))
        else:
            # Generate synthetic data
            self.current_price = 100.0 + np.random.normal(0, 2)
            volume = np.random.uniform(1000, 10000)
            volatility = np.random.uniform(0.01, 0.05)
            trend_strength = np.random.uniform(0.0, 1.0)
        
        # Execute action
        reward = self._execute_action(action_type, position_size, stop_loss, take_profit)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = False
        if self.data_source is not None:
            done = self.current_step >= len(self.data_source)
        else:
            done = self.current_step >= 1000  # 1000 steps for synthetic data
        
        # Get current state
        observation = self._get_state()
        
        # Prepare info
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_pnl': self.total_pnl,
            'trades_count': self.trades_count,
            'win_rate': self.winning_trades / max(self.trades_count, 1),
            'current_price': self.current_price
        }
        
        return observation, reward, done, info
    
    def _execute_action(self, action_type: int, position_size: float, 
                       stop_loss: float, take_profit: float) -> float:
        """Execute trading action and return reward"""
        
        old_pnl = self.total_pnl
        
        if action_type == 1:  # Buy
            if self.position == 0:  # No position
                cost = position_size * self.current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.position = position_size
                    self.balance -= cost
                    self.trades_count += 1
                    logger.debug(f"Buy: {position_size} at {self.current_price}")
        
        elif action_type == 2:  # Sell
            if self.position > 0:  # Has position
                revenue = self.position * self.current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.position = 0.0
                self.trades_count += 1
                logger.debug(f"Sell: {self.position} at {self.current_price}")
        
        # Calculate PnL
        if self.position > 0:
            self.total_pnl = (self.current_price - self._entry_price) * self.position
        
        # Update winning trades
        if self.total_pnl > old_pnl:
            self.winning_trades += 1
        
        # Calculate reward (Sharpe ratio approximation)
        reward = self.total_pnl / max(self.initial_balance, 1)
        
        return reward
    
    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        return np.array([
            self.current_price,
            self._get_volume(),
            self._get_volatility(),
            self._get_trend_strength(),
            self.position,
            self.balance,
            self.total_pnl,
            self._get_confidence()
        ], dtype=np.float32)
    
    def _get_volume(self) -> float:
        """Get current volume"""
        if self.data_source is not None and self.current_step < len(self.data_source):
            return float(self.data_source.iloc[self.current_step]['volume'])
        return np.random.uniform(1000, 10000)
    
    def _get_volatility(self) -> float:
        """Get current volatility"""
        if self.data_source is not None and self.current_step < len(self.data_source):
            return float(self.data_source.iloc[self.current_step].get('volatility', 0.02))
        return np.random.uniform(0.01, 0.05)
    
    def _get_trend_strength(self) -> float:
        """Get current trend strength"""
        if self.data_source is not None and self.current_step < len(self.data_source):
            return float(self.data_source.iloc[self.current_step].get('trend_strength', 0.5))
        return np.random.uniform(0.0, 1.0)
    
    def _get_confidence(self) -> float:
        """Get current confidence"""
        return np.random.uniform(0.5, 0.95)
    
    @property
    def _entry_price(self) -> float:
        """Get entry price for current position"""
        return self.current_price  # Simplified for now

if GYM_AVAILABLE:
    class SignalOptimizationEnvironment(gym.Env):
        """Environment for optimizing signal generation parameters"""
        
        def __init__(self, 
                     signal_history: Optional[List[Dict]] = None,
                     reward_type: str = "precision_recall"):
            super().__init__()
else:
    class SignalOptimizationEnvironment:
        """Mock Environment for optimizing signal generation parameters (gym not available)"""
        
        def __init__(self, 
                     signal_history: Optional[List[Dict]] = None,
                     reward_type: str = "precision_recall"):
            
            self.signal_history = signal_history or []
            self.reward_type = reward_type
            self.current_step = 0
            
            # Action space: [confidence_threshold, volume_threshold, trend_threshold, strength_threshold]
            if GYM_AVAILABLE:
                self.action_space = spaces.Box(
                    low=np.array([0.1, 0.0, 0.0, 0.1]),
                    high=np.array([0.95, 1.0, 1.0, 0.95]),
                    dtype=np.float32
                )
                
                # Observation space: [precision, recall, f1_score, total_signals, successful_signals]
                self.observation_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(5,),
                    dtype=np.float32
                )
            
            logger.info("Signal optimization environment initialized")
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        
        # Parse action
        confidence_threshold = float(action[0])
        volume_threshold = float(action[1])
        trend_threshold = float(action[2])
        strength_threshold = float(action[3])
        
        # Apply thresholds to signals
        filtered_signals = self._apply_thresholds(
            confidence_threshold, volume_threshold, trend_threshold, strength_threshold
        )
        
        # Calculate metrics
        precision, recall, f1_score = self._calculate_metrics(filtered_signals)
        
        # Calculate reward
        if self.reward_type == "precision_recall":
            reward = (precision + recall) / 2  # F1 score
        elif self.reward_type == "precision":
            reward = precision
        else:
            reward = f1_score
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= 100  # 100 optimization steps
        
        # Get current state
        observation = self._get_state()
        
        # Prepare info
        info = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_signals': len(filtered_signals),
            'successful_signals': len([s for s in filtered_signals if s.get('outcome') == 'win'])
        }
        
        return observation, reward, done, info
    
    def _apply_thresholds(self, confidence_threshold: float, volume_threshold: float,
                         trend_threshold: float, strength_threshold: float) -> List[Dict]:
        """Apply thresholds to filter signals"""
        filtered = []
        for signal in self.signal_history:
            if (signal.get('confidence', 0) >= confidence_threshold and
                signal.get('volume', 0) >= volume_threshold and
                signal.get('trend_strength', 0) >= trend_threshold and
                signal.get('strength', 0) >= strength_threshold):
                filtered.append(signal)
        return filtered
    
    def _calculate_metrics(self, signals: List[Dict]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        if not signals:
            return 0.0, 0.0, 0.0
        
        successful = len([s for s in signals if s.get('outcome') == 'win'])
        total = len(signals)
        
        precision = successful / total if total > 0 else 0.0
        recall = successful / len(self.signal_history) if self.signal_history else 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score
    
    def _get_state(self) -> np.ndarray:
        """Get current state"""
        # Use current metrics as state
        filtered_signals = self._apply_thresholds(0.5, 0.0, 0.0, 0.5)  # Default thresholds
        precision, recall, f1_score = self._calculate_metrics(filtered_signals)
        
        return np.array([
            precision,
            recall,
            f1_score,
            len(filtered_signals) / max(len(self.signal_history), 1),
            len([s for s in filtered_signals if s.get('outcome') == 'win']) / max(len(filtered_signals), 1)
        ], dtype=np.float32)

class ReinforcementLearningEngine:
    """Main Reinforcement Learning engine for AlphaPulse"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the RL engine"""
        self.config = config or {}
        self.is_running = False
        
        # RL components
        self.trading_env = None
        self.signal_env = None
        self.trading_agent = None
        self.signal_agent = None
        
        # Performance tracking
        self.training_episodes = 0
        self.total_rewards = 0.0
        self.best_reward = -np.inf
        self.model_performance = {}
        
        # Configuration
        self.use_trading_rl = self.config.get('use_trading_rl', True)
        self.use_signal_optimization = self.config.get('use_signal_optimization', True)
        self.training_frequency = self.config.get('training_frequency', 100)  # episodes
        self.model_save_path = self.config.get('model_save_path', './models/')
        
        logger.info("🚀 Reinforcement Learning Engine initialized")
    
    async def start(self):
        """Start the RL engine"""
        if self.is_running:
            logger.warning("RL engine already running")
            return
        
        self.is_running = True
        
        # Initialize environments
        if self.use_trading_rl:
            await self._initialize_trading_environment()
        
        if self.use_signal_optimization:
            await self._initialize_signal_environment()
        
        # Start background training
        asyncio.create_task(self._background_training())
        
        logger.info("✅ RL engine started successfully")
    
    async def stop(self):
        """Stop the RL engine"""
        self.is_running = False
        logger.info("🛑 RL engine stopped")
    
    async def _initialize_trading_environment(self):
        """Initialize trading environment and agent"""
        try:
            if not GYM_AVAILABLE:
                logger.warning("Gym not available - trading RL disabled")
                return
            
            # Create trading environment
            self.trading_env = TradingEnvironment()
            
            if STABLE_BASELINES3_AVAILABLE:
                # Create PPO agent for trading
                self.trading_agent = PPO(
                    "MlpPolicy",
                    self.trading_env,
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
                logger.info("✅ Trading RL agent initialized (PPO)")
            else:
                logger.warning("Stable-baselines3 not available - using mock trading agent")
                self.trading_agent = MockRLAgent()
                
        except Exception as e:
            logger.error(f"Trading environment initialization error: {e}")
            self.trading_agent = None
            self.trading_env = None
    
    async def _initialize_signal_environment(self):
        """Initialize signal optimization environment and agent"""
        try:
            if not GYM_AVAILABLE:
                logger.warning("Gym not available - signal optimization RL disabled")
                return
            
            # Create signal optimization environment
            self.signal_env = SignalOptimizationEnvironment()
            
            if STABLE_BASELINES3_AVAILABLE:
                # Create A2C agent for signal optimization
                self.signal_agent = A2C(
                    "MlpPolicy",
                    self.signal_env,
                    verbose=0,
                    learning_rate=0.0007,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=1.0,
                    ent_coef=0.01
                )
                logger.info("✅ Signal optimization RL agent initialized (A2C)")
            else:
                logger.warning("Stable-baselines3 not available - using mock signal agent")
                self.signal_agent = MockRLAgent()
                
        except Exception as e:
            logger.error(f"Signal environment initialization error: {e}")
            self.signal_agent = None
            self.signal_env = None
    
    async def _background_training(self):
        """Background training loop"""
        while self.is_running:
            try:
                # Train trading agent
                if self.trading_agent and self.trading_env:
                    await self._train_trading_agent()
                
                # Train signal agent
                if self.signal_agent and self.signal_env:
                    await self._train_signal_agent()
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep between training cycles
                await asyncio.sleep(60)  # Train every minute
                
            except Exception as e:
                logger.error(f"Background training error: {e}")
                await asyncio.sleep(30)
    
    async def _train_trading_agent(self):
        """Train the trading agent"""
        try:
            if not hasattr(self.trading_agent, 'learn'):
                return
            
            # Train for a few episodes
            self.trading_agent.learn(total_timesteps=1000, reset_num_timesteps=False)
            
            # Evaluate performance
            mean_reward = self._evaluate_agent(self.trading_agent, self.trading_env)
            
            # Update best reward
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                await self._save_model(self.trading_agent, 'trading_best')
            
            self.training_episodes += 1
            self.total_rewards += mean_reward
            
            logger.debug(f"Trading agent training - Episode {self.training_episodes}, Mean Reward: {mean_reward:.4f}")
            
        except Exception as e:
            logger.error(f"Trading agent training error: {e}")
    
    async def _train_signal_agent(self):
        """Train the signal optimization agent"""
        try:
            if not hasattr(self.signal_agent, 'learn'):
                return
            
            # Train for a few episodes
            self.signal_agent.learn(total_timesteps=500, reset_num_timesteps=False)
            
            # Evaluate performance
            mean_reward = self._evaluate_agent(self.signal_agent, self.signal_env)
            
            self.training_episodes += 1
            self.total_rewards += mean_reward
            
            logger.debug(f"Signal agent training - Episode {self.training_episodes}, Mean Reward: {mean_reward:.4f}")
            
        except Exception as e:
            logger.error(f"Signal agent training error: {e}")
    
    def _evaluate_agent(self, agent, env, n_eval_episodes: int = 5) -> float:
        """Evaluate agent performance"""
        try:
            if not hasattr(agent, 'predict'):
                return 0.0
            
            rewards = []
            for _ in range(n_eval_episodes):
                obs = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = agent.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    episode_reward += reward
                
                rewards.append(episode_reward)
            
            return np.mean(rewards)
            
        except Exception as e:
            logger.error(f"Agent evaluation error: {e}")
            return 0.0
    
    async def _save_model(self, agent, model_name: str):
        """Save trained model"""
        try:
            if hasattr(agent, 'save'):
                import os
                os.makedirs(self.model_save_path, exist_ok=True)
                agent.save(f"{self.model_save_path}/{model_name}")
                logger.info(f"Model saved: {model_name}")
        except Exception as e:
            logger.error(f"Model save error: {e}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        self.model_performance = {
            'trading_agent': {
                'episodes': self.training_episodes,
                'total_rewards': self.total_rewards,
                'best_reward': self.best_reward,
                'avg_reward': self.total_rewards / max(self.training_episodes, 1)
            },
            'signal_agent': {
                'episodes': self.training_episodes,
                'total_rewards': self.total_rewards,
                'avg_reward': self.total_rewards / max(self.training_episodes, 1)
            }
        }
    
    def get_trading_action(self, state: RLState) -> RLAction:
        """Get trading action from RL agent"""
        try:
            if not self.trading_agent or not hasattr(self.trading_agent, 'predict'):
                return self._get_default_action()
            
            # Convert state to observation
            obs = self._state_to_observation(state)
            
            # Get action from agent
            action, _ = self.trading_agent.predict(obs, deterministic=True)
            
            # Convert action to RLAction
            return self._action_to_rl_action(action)
            
        except Exception as e:
            logger.error(f"Trading action error: {e}")
            return self._get_default_action()
    
    def get_signal_optimization(self, signal_history: List[Dict]) -> Dict[str, float]:
        """Get signal optimization parameters from RL agent"""
        try:
            if not self.signal_agent or not hasattr(self.signal_agent, 'predict'):
                return self._get_default_signal_params()
            
            # Update environment with signal history
            if self.signal_env:
                self.signal_env.signal_history = signal_history
            
            # Get current state
            obs = self.signal_env._get_state()
            
            # Get action from agent
            action, _ = self.signal_agent.predict(obs, deterministic=True)
            
            # Convert action to parameters
            return {
                'confidence_threshold': float(action[0]),
                'volume_threshold': float(action[1]),
                'trend_threshold': float(action[2]),
                'strength_threshold': float(action[3])
            }
            
        except Exception as e:
            logger.error(f"Signal optimization error: {e}")
            return self._get_default_signal_params()
    
    def _state_to_observation(self, state: RLState) -> np.ndarray:
        """Convert RLState to observation array"""
        return np.array([
            state.price,
            state.volume,
            state.volatility,
            state.trend_strength,
            state.position_size,
            state.current_pnl,
            state.signal_strength,
            state.confidence
        ], dtype=np.float32)
    
    def _action_to_rl_action(self, action: np.ndarray) -> RLAction:
        """Convert agent action to RLAction"""
        action_type = int(action[0])
        action_type_str = ['hold', 'buy', 'sell'][action_type]
        
        return RLAction(
            action_type=action_type_str,
            position_size=float(action[1]),
            stop_loss=float(action[2]),
            take_profit=float(action[3]),
            confidence_threshold=float(action[4]),
            risk_allocation=0.1  # Default 10% risk
        )
    
    def _get_default_action(self) -> RLAction:
        """Get default action when RL is not available"""
        return RLAction(
            action_type='hold',
            position_size=0.0,
            stop_loss=0.02,
            take_profit=0.05,
            confidence_threshold=0.7,
            risk_allocation=0.1
        )
    
    def _get_default_signal_params(self) -> Dict[str, float]:
        """Get default signal parameters"""
        return {
            'confidence_threshold': 0.7,
            'volume_threshold': 0.5,
            'trend_threshold': 0.6,
            'strength_threshold': 0.7
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get RL performance summary"""
        return {
            'is_running': self.is_running,
            'training_episodes': self.training_episodes,
            'total_rewards': self.total_rewards,
            'best_reward': self.best_reward,
            'avg_reward': self.total_rewards / max(self.training_episodes, 1),
            'model_performance': self.model_performance,
            'trading_agent_available': self.trading_agent is not None,
            'signal_agent_available': self.signal_agent is not None
        }

class MockRLAgent:
    """Mock RL agent for when stable-baselines3 is not available"""
    
    def __init__(self):
        self.trained = False
    
    def learn(self, total_timesteps: int, reset_num_timesteps: bool = True):
        """Mock learning"""
        self.trained = True
    
    def predict(self, observation, deterministic: bool = True):
        """Mock prediction"""
        # Return random action
        action = np.random.uniform(0, 1, 5)
        return action, None
    
    def save(self, path: str):
        """Mock save"""
        pass
