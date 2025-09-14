"""
Strategy Optimizer for AlphaPulse
Provides parameter optimization and strategy tuning capabilities
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Optimization configuration"""
    strategy_name: str
    parameter_ranges: Dict[str, List[Any]]
    optimization_metric: str
    max_iterations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    metadata: Dict[str, Any]

@dataclass
class OptimizationResult:
    """Optimization result"""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    execution_time: float
    metadata: Dict[str, Any]

class StrategyOptimizer:
    """Strategy parameter optimizer using genetic algorithm"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Optimization configuration
        self.optimization_method = self.config.get('optimization_method', 'genetic_algorithm')
        self.max_workers = self.config.get('max_workers', 4)
        self.enable_parallel = self.config.get('enable_parallel', True)
        
        # Optimization state
        self.active_optimizations = {}  # optimization_id -> optimization_task
        self.optimization_results = {}  # optimization_id -> result
        
        # Performance tracking
        self.stats = {
            'total_optimizations': 0,
            'completed_optimizations': 0,
            'failed_optimizations': 0,
            'last_optimization': None,
            'execution_times': deque(maxlen=100)
        }
        
        # Callbacks
        self.optimization_callbacks = defaultdict(list)  # event_type -> [callback]
    
    async def optimize_strategy(self, config: OptimizationConfig, 
                              fitness_function: Callable) -> str:
        """Start strategy optimization"""
        try:
            optimization_id = f"optimization_{int(time.time())}_{len(self.active_optimizations)}"
            
            # Create optimization task
            optimization_task = asyncio.create_task(
                self._execute_optimization(optimization_id, config, fitness_function)
            )
            self.active_optimizations[optimization_id] = optimization_task
            
            # Update statistics
            self.stats['total_optimizations'] += 1
            self.stats['last_optimization'] = datetime.now()
            
            self.logger.info(f"Started optimization {optimization_id} for {config.strategy_name}")
            return optimization_id
            
        except Exception as e:
            self.logger.error(f"Failed to start optimization: {e}")
            raise
    
    async def _execute_optimization(self, optimization_id: str, config: OptimizationConfig, 
                                  fitness_function: Callable) -> OptimizationResult:
        """Execute strategy optimization"""
        try:
            start_time = time.time()
            
            # Initialize population
            population = self._initialize_population(config)
            best_score = float('-inf')
            best_parameters = None
            optimization_history = []
            
            # Run optimization iterations
            for generation in range(config.max_iterations):
                try:
                    # Evaluate fitness for current population
                    fitness_scores = []
                    for individual in population:
                        score = await fitness_function(individual)
                        fitness_scores.append(score)
                        
                        if score > best_score:
                            best_score = score
                            best_parameters = individual.copy()
                    
                    # Record generation results
                    generation_result = {
                        'generation': generation,
                        'best_score': best_score,
                        'avg_score': np.mean(fitness_scores),
                        'population_size': len(population)
                    }
                    optimization_history.append(generation_result)
                    
                    # Check for convergence
                    if len(optimization_history) > 10:
                        recent_scores = [r['best_score'] for r in optimization_history[-10:]]
                        if max(recent_scores) - min(recent_scores) < 0.001:
                            self.logger.info(f"Optimization {optimization_id} converged at generation {generation}")
                            break
                    
                    # Generate next generation
                    population = self._evolve_population(population, fitness_scores, config)
                    
                except Exception as e:
                    self.logger.warning(f"Error in generation {generation}: {e}")
                    continue
            
            # Create optimization result
            execution_time = time.time() - start_time
            result = OptimizationResult(
                best_parameters=best_parameters or {},
                best_score=best_score,
                optimization_history=optimization_history,
                execution_time=execution_time,
                metadata={
                    'optimization_id': optimization_id,
                    'strategy_name': config.strategy_name,
                    'generations_completed': len(optimization_history),
                    'final_population_size': len(population)
                }
            )
            
            # Store result
            self.optimization_results[optimization_id] = result
            
            # Update statistics
            self.stats['completed_optimizations'] += 1
            self.stats['execution_times'].append(execution_time)
            
            # Trigger callbacks
            await self._trigger_callbacks('optimization_completed', result)
            
            self.logger.info(f"Completed optimization {optimization_id} in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization {optimization_id} failed: {e}")
            self.stats['failed_optimizations'] += 1
            raise
        finally:
            # Clean up active optimization
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
    
    def _initialize_population(self, config: OptimizationConfig) -> List[Dict[str, Any]]:
        """Initialize random population"""
        try:
            population = []
            
            for _ in range(config.population_size):
                individual = {}
                
                for param_name, param_range in config.parameter_ranges.items():
                    if isinstance(param_range, list):
                        # Discrete parameter
                        individual[param_name] = np.random.choice(param_range)
                    elif isinstance(param_range, tuple) and len(param_range) == 2:
                        # Continuous parameter range
                        min_val, max_val = param_range
                        if isinstance(min_val, int):
                            individual[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            individual[param_name] = np.random.uniform(min_val, max_val)
                    else:
                        # Single value parameter
                        individual[param_name] = param_range
                
                population.append(individual)
            
            return population
            
        except Exception as e:
            self.logger.error(f"Failed to initialize population: {e}")
            return []
    
    def _evolve_population(self, population: List[Dict[str, Any]], 
                          fitness_scores: List[float], 
                          config: OptimizationConfig) -> List[Dict[str, Any]]:
        """Evolve population using genetic operators"""
        try:
            new_population = []
            
            # Elitism - keep best individual
            best_idx = np.argmax(fitness_scores)
            new_population.append(population[best_idx].copy())
            
            # Generate rest of population
            while len(new_population) < config.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.random() < config.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.random() < config.mutation_rate:
                    child = self._mutate(child, config.parameter_ranges)
                
                new_population.append(child)
            
            return new_population
            
        except Exception as e:
            self.logger.error(f"Failed to evolve population: {e}")
            return population
    
    def _tournament_selection(self, population: List[Dict[str, Any]], 
                             fitness_scores: List[float], 
                             tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection"""
        try:
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_scores = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_scores)]
            
            return population[winner_idx]
            
        except Exception as e:
            self.logger.error(f"Failed to perform tournament selection: {e}")
            return population[0] if population else {}
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Single-point crossover"""
        try:
            child = {}
            params = list(parent1.keys())
            crossover_point = np.random.randint(1, len(params))
            
            for i, param in enumerate(params):
                if i < crossover_point:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
            
            return child
            
        except Exception as e:
            self.logger.error(f"Failed to perform crossover: {e}")
            return parent1.copy()
    
    def _mutate(self, individual: Dict[str, Any], 
                parameter_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Random mutation"""
        try:
            mutated = individual.copy()
            
            for param_name, param_range in parameter_ranges.items():
                if np.random.random() < 0.1:  # 10% mutation chance
                    if isinstance(param_range, list):
                        # Discrete parameter
                        mutated[param_name] = np.random.choice(param_range)
                    elif isinstance(param_range, tuple) and len(param_range) == 2:
                        # Continuous parameter range
                        min_val, max_val = param_range
                        if isinstance(min_val, int):
                            mutated[param_name] = np.random.randint(min_val, max_val + 1)
                        else:
                            mutated[param_name] = np.random.uniform(min_val, max_val)
            
            return mutated
            
        except Exception as e:
            self.logger.error(f"Failed to perform mutation: {e}")
            return individual
    
    # Public methods
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for optimization events"""
        self.optimization_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for optimization events"""
        callbacks = self.optimization_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization framework statistics"""
        return {
            'stats': self.stats,
            'active_optimizations': len(self.active_optimizations),
            'completed_optimizations': len(self.optimization_results),
            'last_optimization_time': self.stats['last_optimization'].isoformat() if self.stats['last_optimization'] else None
        }
    
    async def close(self):
        """Close the strategy optimizer"""
        try:
            # Cancel active optimizations
            for optimization_id, task in self.active_optimizations.items():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("Strategy Optimizer closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close strategy optimizer: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
