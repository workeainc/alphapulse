"""
Feedback Loop System for AlphaPulse
Phase 3: Automated retraining, walk-forward optimization, and Monte Carlo simulations
"""

import asyncio
import logging
import time
import numpy as np
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class RetrainingTrigger(Enum):
    """Types of retraining triggers"""
    SCHEDULED = "scheduled"
    PERFORMANCE_DECAY = "performance_decay"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"

class ModelState(Enum):
    """Model states during retraining"""
    TRAINING = "training"
    VALIDATING = "validating"
    ACTIVE = "active"
    FAILED = "failed"

@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    training_time: float = 0.0
    model_size_mb: float = 0.0

@dataclass
class RetrainingEvent:
    """Retraining event record"""
    id: str
    trigger: RetrainingTrigger
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    training_metrics: Optional[TrainingMetrics] = None
    error_message: str = ""

class ReinforcementLearningTrainer:
    """Reinforcement learning trainer with experience replay"""
    
    def __init__(self, learning_rate: float = 0.001, experience_buffer_size: int = 10000):
        self.learning_rate = learning_rate
        self.experience_buffer = deque(maxlen=experience_buffer_size)
        self.q_network = {}
        self.training_episodes = 0
        self.total_rewards = 0.0
        
        logger.info("Reinforcement Learning Trainer initialized")
    
    def add_experience(self, state: Tuple[float, ...], action: int, reward: float, next_state: Tuple[float, ...]):
        """Add experience to replay buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': datetime.now()
        }
        self.experience_buffer.append(experience)
    
    async def train_step(self) -> Dict[str, Any]:
        """Perform one training step"""
        if len(self.experience_buffer) < 32:
            return {'status': 'insufficient_data'}
        
        # Sample batch and update Q-values
        batch = list(self.experience_buffer)[-32:]  # Simple batch sampling
        
        total_reward = 0.0
        for experience in batch:
            state_key = json.dumps([round(x, 3) for x in experience['state']])
            action = experience['action']
            reward = experience['reward']
            
            if state_key not in self.q_network:
                self.q_network[state_key] = defaultdict(float)
            
            # Simple Q-learning update
            current_q = self.q_network[state_key][action]
            new_q = current_q + self.learning_rate * (reward - current_q)
            self.q_network[state_key][action] = new_q
            
            total_reward += reward
        
        self.training_episodes += 1
        self.total_rewards += total_reward
        
        return {
            'status': 'success',
            'avg_reward': total_reward / len(batch),
            'episodes': self.training_episodes
        }
    
    def get_action(self, state: Tuple[float, ...]) -> int:
        """Get action using epsilon-greedy policy"""
        if np.random.random() < 0.1:  # 10% exploration
            return np.random.randint(0, 21)
        
        state_key = json.dumps([round(x, 3) for x in state])
        actions = self.q_network.get(state_key, {})
        
        if actions:
            return max(actions.keys(), key=lambda k: actions[k])
        else:
            return 10  # Default action
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'training_episodes': self.training_episodes,
            'total_rewards': self.total_rewards,
            'buffer_size': len(self.experience_buffer),
            'q_network_size': len(self.q_network)
        }

class WalkForwardOptimizer:
    """Walk-forward optimization with rolling windows"""
    
    def __init__(self, window_size_days: int = 30, step_size_days: int = 7):
        self.window_size_days = window_size_days
        self.step_size_days = step_size_days
        self.model_versions = {}
        self.performance_history = deque(maxlen=1000)
        
        logger.info("WalkForwardOptimizer initialized")
    
    async def optimize_window(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize model for a data window"""
        if len(data) < 50:
            return {'status': 'insufficient_data'}
        
        try:
            # Split data
            split_idx = int(len(data) * 0.8)
            training_data = data[:split_idx]
            validation_data = data[split_idx:]
            
            # Train model
            model, training_metrics = await self._train_model(training_data)
            
            # Validate model
            validation_metrics = await self._validate_model(model, validation_data)
            
            # Store model version
            version = f"v{len(self.model_versions) + 1}"
            self.model_versions[version] = {
                'model': model,
                'performance': validation_metrics,
                'timestamp': datetime.now()
            }
            
            return {
                'status': 'success',
                'version': version,
                'training_metrics': training_metrics,
                'validation_metrics': validation_metrics
            }
            
        except Exception as e:
            logger.error(f"Walk-forward optimization error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _train_model(self, training_data: List[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
        """Train model on training data"""
        if not ML_AVAILABLE:
            return self._create_simple_model(), {'accuracy': 0.5}
        
        try:
            # Extract features and labels
            features = []
            labels = []
            
            for data_point in training_data:
                feature_vector = [
                    data_point.get('confidence', 0.5),
                    data_point.get('threshold', 0.6),
                    data_point.get('risk_score', 0.5),
                    data_point.get('volatility', 0.5)
                ]
                label = 1 if data_point.get('success', False) else 0
                
                features.append(feature_vector)
                labels.append(label)
            
            if len(features) < 10:
                return self._create_simple_model(), {'accuracy': 0.5}
            
            # Train Random Forest
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(features, labels)
            
            # Calculate metrics
            predictions = model.predict(features)
            metrics = {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'f1_score': f1_score(labels, predictions, zero_division=0)
            }
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return self._create_simple_model(), {'accuracy': 0.5}
    
    async def _validate_model(self, model: Any, validation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate model on validation data"""
        try:
            features = []
            labels = []
            
            for data_point in validation_data:
                feature_vector = [
                    data_point.get('confidence', 0.5),
                    data_point.get('threshold', 0.6),
                    data_point.get('risk_score', 0.5),
                    data_point.get('volatility', 0.5)
                ]
                label = 1 if data_point.get('success', False) else 0
                
                features.append(feature_vector)
                labels.append(label)
            
            if not features:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            
            predictions = model.predict(features)
            
            return {
                'accuracy': accuracy_score(labels, predictions),
                'precision': precision_score(labels, predictions, zero_division=0),
                'recall': recall_score(labels, predictions, zero_division=0),
                'f1_score': f1_score(labels, predictions, zero_division=0)
            }
            
        except Exception as e:
            logger.error(f"Model validation error: {e}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def _create_simple_model(self) -> Any:
        """Create simple fallback model"""
        class SimpleModel:
            def predict(self, features):
                return [1 if f[0] > 0.6 else 0 for f in features]
        return SimpleModel()
    
    def get_best_model(self) -> Optional[Any]:
        """Get best performing model"""
        if not self.model_versions:
            return None
        
        best_version = max(
            self.model_versions.keys(),
            key=lambda v: self.model_versions[v]['performance']['f1_score']
        )
        
        return self.model_versions[best_version]['model']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.model_versions:
            return {'status': 'no_models'}
        
        performances = [v['performance']['f1_score'] for v in self.model_versions.values()]
        
        return {
            'model_versions': len(self.model_versions),
            'avg_performance': np.mean(performances),
            'best_performance': max(performances),
            'worst_performance': min(performances)
        }

class MonteCarloSimulator:
    """Monte Carlo simulation for model testing"""
    
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
        self.simulation_results = []
        
        logger.info(f"Monte Carlo Simulator initialized ({num_simulations} simulations)")
    
    async def run_simulation(self, model: Any, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        if not historical_data:
            return {'status': 'no_data'}
        
        try:
            simulation_results = []
            
            for i in range(self.num_simulations):
                # Generate random scenario
                scenario_data = self._generate_scenario(historical_data)
                
                # Run simulation
                result = await self._simulate_scenario(model, scenario_data)
                simulation_results.append(result)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Monte Carlo progress: {i + 1}/{self.num_simulations}")
            
            # Analyze results
            analysis = self._analyze_results(simulation_results)
            
            self.simulation_results = simulation_results
            
            return {
                'status': 'success',
                'num_simulations': self.num_simulations,
                'analysis': analysis
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_scenario(self, historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate random scenario from historical data"""
        scenario_length = min(30, len(historical_data))
        scenario_indices = np.random.choice(len(historical_data), scenario_length, replace=True)
        
        scenario_data = []
        for idx in scenario_indices:
            data_point = historical_data[idx].copy()
            
            # Add random noise
            data_point['confidence'] += np.random.normal(0, 0.05)
            data_point['confidence'] = max(0.0, min(1.0, data_point['confidence']))
            
            scenario_data.append(data_point)
        
        return scenario_data
    
    async def _simulate_scenario(self, model: Any, scenario_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate single scenario"""
        try:
            total_pnl = 0.0
            successful_trades = 0
            total_trades = 0
            
            for data_point in scenario_data:
                # Make prediction
                features = [
                    data_point.get('confidence', 0.5),
                    data_point.get('threshold', 0.6),
                    data_point.get('risk_score', 0.5),
                    data_point.get('volatility', 0.5)
                ]
                prediction = model.predict([features])[0]
                
                # Simulate trade outcome
                if prediction == 1:
                    total_trades += 1
                    success_prob = data_point.get('success_probability', 0.6)
                    success = np.random.random() < success_prob
                    
                    if success:
                        successful_trades += 1
                        pnl = 0.02
                    else:
                        pnl = -0.01
                    
                    total_pnl += pnl
            
            return {
                'total_pnl': total_pnl,
                'successful_trades': successful_trades,
                'total_trades': total_trades,
                'success_rate': successful_trades / max(1, total_trades)
            }
            
        except Exception as e:
            logger.error(f"Scenario simulation error: {e}")
            return {
                'total_pnl': 0.0,
                'successful_trades': 0,
                'total_trades': 0,
                'success_rate': 0.0
            }
    
    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze simulation results"""
        try:
            pnls = [r['total_pnl'] for r in results]
            success_rates = [r['success_rate'] for r in results]
            
            analysis = {
                'pnl': {
                    'mean': np.mean(pnls),
                    'std': np.std(pnls),
                    'min': np.min(pnls),
                    'max': np.max(pnls),
                    'var_95': np.percentile(pnls, 5)
                },
                'success_rate': {
                    'mean': np.mean(success_rates),
                    'std': np.std(success_rates)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Simulation analysis error: {e}")
            return {}

class FeedbackLoop:
    """Main feedback loop system"""
    
    def __init__(self, 
                enable_rl_training: bool = True,
                enable_walk_forward: bool = True,
                enable_monte_carlo: bool = True,
                retraining_interval: int = 86400):
        
        # Initialize components
        self.rl_trainer = ReinforcementLearningTrainer() if enable_rl_training else None
        self.walk_forward_optimizer = WalkForwardOptimizer() if enable_walk_forward else None
        self.monte_carlo_simulator = MonteCarloSimulator() if enable_monte_carlo else None
        
        # Configuration
        self.retraining_interval = retraining_interval
        
        # State tracking
        self.current_model_state = ModelState.ACTIVE
        self.last_retraining = datetime.now()
        self.retraining_history = deque(maxlen=100)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.current_performance = 0.0
        
        # Background tasks
        self.retraining_task = None
        self.running = False
        
        logger.info("Feedback Loop System initialized")
    
    async def start(self):
        """Start the feedback loop system"""
        if self.running:
            return
        
        self.running = True
        self.retraining_task = asyncio.create_task(self._retraining_loop())
        logger.info("Feedback Loop System started")
    
    async def stop(self):
        """Stop the feedback loop system"""
        if not self.running:
            return
        
        self.running = False
        
        if self.retraining_task:
            self.retraining_task.cancel()
            try:
                await self.retraining_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Feedback Loop System stopped")
    
    async def add_performance_data(self, performance_data: Dict[str, Any]):
        """Add performance data for feedback loop"""
        try:
            # Calculate overall performance
            accuracy = performance_data.get('accuracy', 0.0)
            precision = performance_data.get('precision', 0.0)
            recall = performance_data.get('recall', 0.0)
            f1_score = performance_data.get('f1_score', 0.0)
            
            overall_performance = (accuracy + precision + recall + f1_score) / 4
            self.current_performance = overall_performance
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'performance': overall_performance,
                'metrics': performance_data
            })
            
            # Add to RL trainer
            if self.rl_trainer:
                state = (overall_performance, performance_data.get('threshold', 0.6))
                reward = overall_performance - 0.5
                self.rl_trainer.add_experience(state, 0, reward, state)
            
        except Exception as e:
            logger.error(f"Performance data addition error: {e}")
    
    async def trigger_retraining(self, trigger: RetrainingTrigger = RetrainingTrigger.SCHEDULED) -> bool:
        """Trigger model retraining"""
        try:
            if self.current_model_state == ModelState.TRAINING:
                return False
            
            self.current_model_state = ModelState.TRAINING
            
            event = RetrainingEvent(
                id=hashlib.md5(f"{trigger}{time.time()}".encode()).hexdigest()[:8],
                trigger=trigger,
                start_time=datetime.now()
            )
            
            logger.info(f"Starting retraining (trigger: {trigger.value})")
            
            # Perform retraining
            success = await self._perform_retraining(event)
            
            event.end_time = datetime.now()
            event.success = success
            
            self.retraining_history.append(event)
            self.last_retraining = datetime.now()
            
            self.current_model_state = ModelState.ACTIVE if success else ModelState.FAILED
            
            return success
            
        except Exception as e:
            logger.error(f"Retraining trigger error: {e}")
            self.current_model_state = ModelState.FAILED
            return False
    
    async def _perform_retraining(self, event: RetrainingEvent) -> bool:
        """Perform the actual retraining process"""
        try:
            # Train RL agent
            if self.rl_trainer:
                for _ in range(10):
                    await self.rl_trainer.train_step()
            
            # Walk-forward optimization
            if self.walk_forward_optimizer and self.performance_history:
                historical_data = []
                for perf in self.performance_history:
                    historical_data.append({
                        'timestamp': perf['timestamp'],
                        'performance': perf['performance'],
                        'success': perf['performance'] > 0.5,
                        **perf['metrics']
                    })
                
                if len(historical_data) >= 50:
                    await self.walk_forward_optimizer.optimize_window(historical_data)
            
            # Monte Carlo simulation
            if self.monte_carlo_simulator and self.performance_history:
                best_model = self.walk_forward_optimizer.get_best_model() if self.walk_forward_optimizer else None
                
                if best_model:
                    historical_data = []
                    for perf in self.performance_history:
                        historical_data.append({
                            'timestamp': perf['timestamp'],
                            'performance': perf['performance'],
                            'success_probability': perf['performance'],
                            **perf['metrics']
                        })
                    
                    await self.monte_carlo_simulator.run_simulation(best_model, historical_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Retraining process error: {e}")
            return False
    
    async def _retraining_loop(self):
        """Periodic retraining loop"""
        while self.running:
            try:
                time_since_retraining = (datetime.now() - self.last_retraining).total_seconds()
                
                if time_since_retraining >= self.retraining_interval:
                    await self.trigger_retraining(RetrainingTrigger.SCHEDULED)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Retraining loop error: {e}")
                await asyncio.sleep(1800)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback loop statistics"""
        return {
            'current_model_state': self.current_model_state.value,
            'current_performance': self.current_performance,
            'last_retraining': self.last_retraining.isoformat(),
            'retraining_count': len(self.retraining_history),
            'performance_history_size': len(self.performance_history),
            'rl_trainer_stats': self.rl_trainer.get_stats() if self.rl_trainer else {},
            'walk_forward_stats': self.walk_forward_optimizer.get_stats() if self.walk_forward_optimizer else {},
            'monte_carlo_stats': {
                'num_simulations': self.monte_carlo_simulator.num_simulations if self.monte_carlo_simulator else 0,
                'simulation_results': len(self.monte_carlo_simulator.simulation_results) if self.monte_carlo_simulator else 0
            }
        }

# Global feedback loop instance
feedback_loop = FeedbackLoop(
    enable_rl_training=True,
    enable_walk_forward=True,
    enable_monte_carlo=True,
    retraining_interval=86400  # 24 hours
)
