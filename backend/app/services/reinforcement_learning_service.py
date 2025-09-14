import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncpg
import uuid

# Try to import ML libraries
try:
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class StateType(Enum):
    VOLUME_ANALYSIS = "volume_analysis"
    PRICE_ACTION = "price_action"
    TECHNICAL_INDICATORS = "technical_indicators"
    MARKET_REGIME = "market_regime"

@dataclass
class State:
    symbol: str
    timeframe: str
    timestamp: datetime
    features: Dict
    state_type: StateType

@dataclass
class Action:
    action_type: ActionType
    confidence: float
    metadata: Dict

@dataclass
class Reward:
    reward_value: float
    reward_type: str  # 'pnl', 'sharpe', 'drawdown'
    metadata: Dict

@dataclass
class Episode:
    episode_id: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    total_reward: float
    actions_taken: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float

class QLearningAgent:
    """Simple Q-Learning agent for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, discount_factor: float = 0.95, epsilon: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        self.state_mapping = {}
        self.action_mapping = {
            ActionType.BUY: 0,
            ActionType.SELL: 1,
            ActionType.HOLD: 2,
            ActionType.CLOSE: 3
        }
    
    def get_state_index(self, state_features: Dict) -> int:
        """Convert state features to state index"""
        # Simple hash-based state mapping
        state_key = hash(tuple(sorted(state_features.items())))
        if state_key not in self.state_mapping:
            self.state_mapping[state_key] = len(self.state_mapping)
        return self.state_mapping[state_key]
    
    def choose_action(self, state_features: Dict) -> ActionType:
        """Choose action using epsilon-greedy policy"""
        state_idx = self.get_state_index(state_features)
        
        if np.random.random() < self.epsilon:
            # Exploration: random action
            action_idx = np.random.randint(0, self.action_size)
        else:
            # Exploitation: best action
            action_idx = np.argmax(self.q_table[state_idx])
        
        # Map back to action type
        for action_type, idx in self.action_mapping.items():
            if idx == action_idx:
                return action_type
        
        return ActionType.HOLD  # Default
    
    def update_q_value(self, state_features: Dict, action: ActionType, reward: float, next_state_features: Dict):
        """Update Q-value using Q-learning update rule"""
        state_idx = self.get_state_index(state_features)
        next_state_idx = self.get_state_index(next_state_features)
        action_idx = self.action_mapping[action]
        
        current_q = self.q_table[state_idx, action_idx]
        max_next_q = np.max(self.q_table[next_state_idx])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_idx, action_idx] = new_q
    
    def get_action_confidence(self, state_features: Dict, action: ActionType) -> float:
        """Get confidence score for an action"""
        state_idx = self.get_state_index(state_features)
        action_idx = self.action_mapping[action]
        
        q_value = self.q_table[state_idx, action_idx]
        max_q = np.max(self.q_table[state_idx])
        
        if max_q == 0:
            return 0.5  # Neutral confidence
        
        return min(q_value / max_q, 1.0)

class ReinforcementLearningService:
    """Service for reinforcement learning-based trading decisions"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # RL parameters
        self.episode_length = 100  # Number of steps per episode
        self.min_episode_length = 20
        self.reward_scaling = 100.0
        
        # Initialize agents for different symbols/timeframes
        self.agents = {}
        self.episodes = {}
        
        # State and action tracking
        self.current_states = {}
        self.episode_states = {}
        
        self.logger.info("🤖 Reinforcement Learning Service initialized")
    
    async def initialize_agent(self, symbol: str, timeframe: str) -> str:
        """Initialize a new RL agent for a symbol/timeframe combination"""
        try:
            agent_id = f"{symbol}_{timeframe}_{uuid.uuid4().hex[:8]}"
            
            # Create Q-learning agent
            state_size = 1000  # Approximate state space size
            action_size = 4    # BUY, SELL, HOLD, CLOSE
            
            agent = QLearningAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon=0.1
            )
            
            self.agents[agent_id] = agent
            self.episodes[agent_id] = []
            
            self.logger.info(f"🤖 Initialized RL agent {agent_id} for {symbol} {timeframe}")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing RL agent: {e}")
            return None
    
    async def get_state_features(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]) -> Dict:
        """Extract state features from OHLCV data"""
        try:
            if not ohlcv_data or len(ohlcv_data) < 10:
                return {}
            
            df = pd.DataFrame(ohlcv_data)
            latest = df.iloc[-1]
            
            # Volume features
            volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = latest['volume'] / volume_ma if volume_ma > 0 else 1.0
            
            # Price features
            price_change = (latest['close'] - latest['open']) / latest['open']
            high_low_ratio = (latest['high'] - latest['low']) / latest['close']
            
            # Technical indicators
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            price_vs_ema20 = (latest['close'] - ema_20) / ema_20
            price_vs_ema50 = (latest['close'] - ema_50) / ema_50
            
            # Volatility
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            # Market regime (simplified)
            market_regime = 1 if price_vs_ema20 > 0 and price_vs_ema50 > 0 else -1
            
            state_features = {
                'volume_ratio': min(volume_ratio, 10.0),  # Cap at 10x
                'price_change': max(min(price_change, 0.1), -0.1),  # Cap at ±10%
                'high_low_ratio': min(high_low_ratio, 0.1),  # Cap at 10%
                'price_vs_ema20': max(min(price_vs_ema20, 0.05), -0.05),  # Cap at ±5%
                'price_vs_ema50': max(min(price_vs_ema50, 0.1), -0.1),  # Cap at ±10%
                'volatility': min(volatility * 100, 5.0),  # Cap at 5%
                'market_regime': market_regime
            }
            
            return state_features
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting state features: {e}")
            return {}
    
    async def choose_action(self, agent_id: str, state_features: Dict) -> Action:
        """Choose action using the RL agent"""
        try:
            if agent_id not in self.agents:
                self.logger.error(f"❌ Agent {agent_id} not found")
                return Action(ActionType.HOLD, 0.5, {})
            
            agent = self.agents[agent_id]
            action_type = agent.choose_action(state_features)
            confidence = agent.get_action_confidence(state_features, action_type)
            
            action = Action(
                action_type=action_type,
                confidence=confidence,
                metadata={
                    'agent_id': agent_id,
                    'epsilon': agent.epsilon,
                    'state_features': state_features
                }
            )
            
            # Store current state
            self.current_states[agent_id] = state_features
            
            self.logger.info(f"🤖 Agent {agent_id} chose {action_type.value} with confidence {confidence:.2f}")
            return action
            
        except Exception as e:
            self.logger.error(f"❌ Error choosing action: {e}")
            return Action(ActionType.HOLD, 0.5, {})
    
    async def calculate_reward(self, action: Action, current_price: float, next_price: float, 
                             position_size: float = 1.0) -> Reward:
        """Calculate reward based on action and price movement"""
        try:
            if action.action_type == ActionType.HOLD:
                reward_value = 0.0
                reward_type = "hold"
            elif action.action_type == ActionType.BUY:
                price_change = (next_price - current_price) / current_price
                reward_value = price_change * position_size * self.reward_scaling
                reward_type = "buy_pnl"
            elif action.action_type == ActionType.SELL:
                price_change = (current_price - next_price) / current_price
                reward_value = price_change * position_size * self.reward_scaling
                reward_type = "sell_pnl"
            elif action.action_type == ActionType.CLOSE:
                # Small penalty for closing to encourage holding good positions
                reward_value = -0.1
                reward_type = "close_penalty"
            else:
                reward_value = 0.0
                reward_type = "unknown"
            
            # Add confidence bonus/penalty
            confidence_bonus = (action.confidence - 0.5) * 0.1
            reward_value += confidence_bonus
            
            reward = Reward(
                reward_value=reward_value,
                reward_type=reward_type,
                metadata={
                    'action_type': action.action_type.value,
                    'confidence': action.confidence,
                    'price_change': (next_price - current_price) / current_price,
                    'position_size': position_size
                }
            )
            
            return reward
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating reward: {e}")
            return Reward(0.0, "error", {})
    
    async def update_agent(self, agent_id: str, action: Action, reward: Reward, next_state_features: Dict):
        """Update the RL agent with the experience"""
        try:
            if agent_id not in self.agents:
                self.logger.error(f"❌ Agent {agent_id} not found")
                return
            
            agent = self.agents[agent_id]
            current_state = self.current_states.get(agent_id, {})
            
            # Update Q-values
            agent.update_q_value(current_state, action.action_type, reward.reward_value, next_state_features)
            
            # Store experience for episode tracking
            if agent_id not in self.episode_states:
                self.episode_states[agent_id] = []
            
            self.episode_states[agent_id].append({
                'state': current_state,
                'action': action,
                'reward': reward,
                'next_state': next_state_features,
                'timestamp': datetime.now()
            })
            
            # Check if episode should end
            if len(self.episode_states[agent_id]) >= self.episode_length:
                await self._end_episode(agent_id)
            
            self.logger.info(f"🤖 Updated agent {agent_id} with reward {reward.reward_value:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating agent: {e}")
    
    async def _end_episode(self, agent_id: str):
        """End current episode and calculate performance metrics"""
        try:
            if agent_id not in self.episode_states or not self.episode_states[agent_id]:
                return
            
            episode_data = self.episode_states[agent_id]
            
            # Calculate episode metrics
            total_reward = sum(exp['reward'].reward_value for exp in episode_data)
            actions_taken = len(episode_data)
            
            # Calculate win rate (positive rewards)
            positive_rewards = sum(1 for exp in episode_data if exp['reward'].reward_value > 0)
            win_rate = positive_rewards / actions_taken if actions_taken > 0 else 0.0
            
            # Calculate Sharpe ratio (simplified)
            rewards = [exp['reward'].reward_value for exp in episode_data]
            if len(rewards) > 1:
                sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-8)
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            cumulative_rewards = np.cumsum(rewards)
            running_max = np.maximum.accumulate(cumulative_rewards)
            drawdowns = running_max - cumulative_rewards
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            
            # Create episode
            episode = Episode(
                episode_id=f"episode_{uuid.uuid4().hex[:8]}",
                symbol=episode_data[0]['state'].get('symbol', 'unknown'),
                timeframe=episode_data[0]['state'].get('timeframe', 'unknown'),
                start_time=episode_data[0]['timestamp'],
                end_time=episode_data[-1]['timestamp'],
                total_reward=total_reward,
                actions_taken=actions_taken,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
            # Store episode
            self.episodes[agent_id].append(episode)
            
            # Store episode in database
            await self._store_episode(agent_id, episode, episode_data)
            
            # Reset episode states
            self.episode_states[agent_id] = []
            
            self.logger.info(f"🎬 Episode ended for agent {agent_id}: "
                           f"Reward={total_reward:.2f}, Win Rate={win_rate:.2f}, "
                           f"Sharpe={sharpe_ratio:.2f}, MaxDD={max_drawdown:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error ending episode: {e}")
    
    async def _store_episode(self, agent_id: str, episode: Episode, episode_data: List[Dict]):
        """Store episode data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store episode performance
                await conn.execute("""
                    INSERT INTO rl_policy_performance (
                        agent_id, symbol, timeframe, timestamp, episode_id,
                        total_reward, episode_length, win_rate, sharpe_ratio,
                        max_drawdown, profit_factor, policy_version, training_metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, 
                agent_id, episode.symbol, episode.timeframe, datetime.now(), episode.episode_id,
                episode.total_reward, episode.actions_taken, episode.win_rate, episode.sharpe_ratio,
                episode.max_drawdown, 0.0, "v1.0", {"episode_length": len(episode_data)}
                )
                
                # Store individual states
                for exp in episode_data:
                    await conn.execute("""
                        INSERT INTO rl_agent_states (
                            symbol, timeframe, timestamp, agent_id, state_features,
                            action_taken, reward_received, next_state_features,
                            episode_id, step_number, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    episode.symbol, episode.timeframe, exp['timestamp'], agent_id,
                    exp['state'], exp['action'].action_type.value, exp['reward'].reward_value,
                    exp['next_state'], episode.episode_id, len(self.episode_states[agent_id]),
                    {"confidence": exp['action'].confidence, "reward_type": exp['reward'].reward_type}
                    )
            
            self.logger.info(f"💾 Stored episode {episode.episode_id} for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing episode: {e}")
    
    async def get_agent_performance(self, agent_id: str, episodes: int = 10) -> Dict:
        """Get recent performance metrics for an agent"""
        try:
            if agent_id not in self.episodes:
                return {}
            
            recent_episodes = self.episodes[agent_id][-episodes:]
            
            if not recent_episodes:
                return {}
            
            total_rewards = [ep.total_reward for ep in recent_episodes]
            win_rates = [ep.win_rate for ep in recent_episodes]
            sharpe_ratios = [ep.sharpe_ratio for ep in recent_episodes]
            max_drawdowns = [ep.max_drawdown for ep in recent_episodes]
            
            performance = {
                'agent_id': agent_id,
                'episodes_analyzed': len(recent_episodes),
                'avg_total_reward': np.mean(total_rewards),
                'avg_win_rate': np.mean(win_rates),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'best_episode_reward': max(total_rewards),
                'worst_episode_reward': min(total_rewards),
                'consistency_score': 1.0 - (np.std(total_rewards) / (abs(np.mean(total_rewards)) + 1e-8))
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"❌ Error getting agent performance: {e}")
            return {}
    
    async def optimize_agent_parameters(self, agent_id: str):
        """Optimize agent parameters based on recent performance"""
        try:
            if agent_id not in self.agents:
                return
            
            performance = await self.get_agent_performance(agent_id)
            
            if not performance or performance['episodes_analyzed'] < 5:
                return
            
            agent = self.agents[agent_id]
            
            # Adjust epsilon based on performance
            if performance['avg_win_rate'] > 0.6:
                # Good performance, reduce exploration
                agent.epsilon = max(agent.epsilon * 0.9, 0.05)
            elif performance['avg_win_rate'] < 0.4:
                # Poor performance, increase exploration
                agent.epsilon = min(agent.epsilon * 1.1, 0.3)
            
            # Adjust learning rate based on consistency
            if performance['consistency_score'] < 0.5:
                # Inconsistent performance, reduce learning rate
                agent.learning_rate = max(agent.learning_rate * 0.9, 0.01)
            else:
                # Consistent performance, increase learning rate
                agent.learning_rate = min(agent.learning_rate * 1.1, 0.2)
            
            self.logger.info(f"🔧 Optimized agent {agent_id}: "
                           f"epsilon={agent.epsilon:.3f}, lr={agent.learning_rate:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing agent parameters: {e}")
