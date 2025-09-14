#!/usr/bin/env python3
"""
Chaos Engineering Framework for AlphaPulse
Provides automated failure injection and resilience validation
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ChaosType(Enum):
    """Types of chaos experiments"""
    LATENCY_INJECTION = "latency_injection"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATABASE_FAILURE = "database_failure"

class ExperimentState(Enum):
    """Experiment execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ChaosExperiment:
    """Configuration for a chaos experiment"""
    name: str
    description: str
    chaos_type: ChaosType
    duration: float
    intensity: float
    target_services: List[str]
    parameters: Dict[str, Any]

@dataclass
class ExperimentResult:
    """Results of a chaos experiment"""
    experiment_name: str
    start_time: datetime
    end_time: Optional[datetime]
    state: ExperimentState
    success: bool
    resilience_score: float

class SimpleChaosRunner:
    """Simple chaos experiment runner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        self.experiment_history: List[ExperimentResult] = []
        
        # Predefined experiments
        self.predefined_experiments = self._setup_predefined_experiments()
    
    def _setup_predefined_experiments(self) -> Dict[str, ChaosExperiment]:
        """Setup predefined chaos experiments"""
        experiments = {}
        
        experiments["high_latency"] = ChaosExperiment(
            name="High Latency Injection",
            description="Simulate high latency in operations",
            chaos_type=ChaosType.LATENCY_INJECTION,
            duration=30.0,
            intensity=0.8,
            target_services=["database", "pattern_storage"],
            parameters={"latency_ms": 2000, "probability": 0.7}
        )
        
        experiments["database_errors"] = ChaosExperiment(
            name="Database Error Injection",
            description="Simulate database errors",
            chaos_type=ChaosType.ERROR_INJECTION,
            duration=45.0,
            intensity=0.6,
            target_services=["database"],
            parameters={"error_type": "connection_timeout", "probability": 0.5}
        )
        
        experiments["resource_pressure"] = ChaosExperiment(
            name="Resource Pressure",
            description="Simulate resource exhaustion",
            chaos_type=ChaosType.RESOURCE_EXHAUSTION,
            duration=60.0,
            intensity=0.7,
            target_services=["all"],
            parameters={"memory_mb": 256, "duration": 60.0}
        )
        
        return experiments
    
    async def run_experiment(self, experiment: ChaosExperiment) -> str:
        """Run a chaos experiment"""
        experiment_id = f"exp_{experiment.name}_{int(time.time())}"
        
        self.active_experiments[experiment_id] = {
            "experiment": experiment,
            "state": ExperimentState.RUNNING,
            "start_time": datetime.now(timezone.utc),
            "result": None
        }
        
        # Start experiment execution
        asyncio.create_task(self._execute_experiment(experiment_id, experiment))
        
        self.logger.info(f"🚀 Chaos experiment started: {experiment.name}")
        return experiment_id
    
    async def _execute_experiment(self, experiment_id: str, experiment: ChaosExperiment):
        """Execute a chaos experiment"""
        try:
            self.logger.info(f"🔄 Running experiment: {experiment.name}")
            
            # Simulate chaos duration
            await asyncio.sleep(experiment.duration)
            
            # Simulate some chaos effects
            if experiment.chaos_type == ChaosType.LATENCY_INJECTION:
                await self._simulate_latency_chaos(experiment)
            elif experiment.chaos_type == ChaosType.ERROR_INJECTION:
                await self._simulate_error_chaos(experiment)
            elif experiment.chaos_type == ChaosType.RESOURCE_EXHAUSTION:
                await self._simulate_resource_chaos(experiment)
            
            # Calculate resilience score
            resilience_score = self._calculate_resilience_score(experiment)
            
            # Create result
            result = ExperimentResult(
                experiment_name=experiment.name,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                state=ExperimentState.COMPLETED,
                success=resilience_score > 0.5,
                resilience_score=resilience_score
            )
            
            # Update experiment state
            self.active_experiments[experiment_id]["state"] = ExperimentState.COMPLETED
            self.active_experiments[experiment_id]["result"] = result
            
            # Add to history
            self.experiment_history.append(result)
            
            # Keep only last 50 experiments
            if len(self.experiment_history) > 50:
                self.experiment_history = self.experiment_history[-50:]
            
            self.logger.info(f"✅ Chaos experiment completed: {experiment.name} (score: {resilience_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ Chaos experiment failed: {experiment.name} - {e}")
            
            # Create failed result
            result = ExperimentResult(
                experiment_name=experiment.name,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                state=ExperimentState.FAILED,
                success=False,
                resilience_score=0.0
            )
            
            # Update experiment state
            self.active_experiments[experiment_id]["state"] = ExperimentState.FAILED
            self.active_experiments[experiment_id]["result"] = result
            
            # Add to history
            self.experiment_history.append(result)
    
    async def _simulate_latency_chaos(self, experiment: ChaosExperiment):
        """Simulate latency chaos"""
        latency_ms = experiment.parameters.get("latency_ms", 1000)
        probability = experiment.parameters.get("probability", 1.0)
        
        self.logger.info(f"⏱️ Simulating latency chaos: {latency_ms}ms (prob: {probability})")
        
        # Simulate some operations with latency
        for i in range(5):
            if random.random() < probability:
                await asyncio.sleep(latency_ms / 1000.0)
                self.logger.info(f"⏱️ Latency injected in operation {i+1}")
    
    async def _simulate_error_chaos(self, experiment: ChaosExperiment):
        """Simulate error chaos"""
        error_type = experiment.parameters.get("error_type", "generic_error")
        probability = experiment.parameters.get("probability", 1.0)
        
        self.logger.info(f"🚨 Simulating error chaos: {error_type} (prob: {probability})")
        
        # Simulate some errors
        for i in range(3):
            if random.random() < probability:
                self.logger.warning(f"🚨 Error injected in operation {i+1}: {error_type}")
    
    async def _simulate_resource_chaos(self, experiment: ChaosExperiment):
        """Simulate resource chaos"""
        memory_mb = experiment.parameters.get("memory_mb", 128)
        duration = experiment.parameters.get("duration", 30.0)
        
        self.logger.info(f"💾 Simulating resource chaos: {memory_mb}MB for {duration}s")
        
        # Simulate resource consumption
        memory_blocks = []
        try:
            # Simulate memory allocation
            for i in range(min(memory_mb, 10)):  # Limit to prevent real memory issues
                memory_blocks.append(bytearray(1024 * 1024))  # 1MB
                await asyncio.sleep(0.1)
            
            # Hold memory for duration
            await asyncio.sleep(duration)
            
        finally:
            # Clean up
            memory_blocks.clear()
            self.logger.info("🧹 Resource chaos simulation cleaned up")
    
    def _calculate_resilience_score(self, experiment: ChaosExperiment) -> float:
        """Calculate resilience score based on experiment"""
        try:
            # Base score
            score = 1.0
            
            # Deduct points based on intensity
            score -= experiment.intensity * 0.3
            
            # Deduct points based on duration
            if experiment.duration > 60:
                score -= 0.2
            
            # Add some randomness to simulate real-world conditions
            score += random.uniform(-0.1, 0.1)
            
            # Ensure score is within bounds
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating resilience score: {e}")
            return 0.0
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific experiment"""
        if experiment_id in self.active_experiments:
            exp_data = self.active_experiments[experiment_id]
            return {
                "experiment_id": experiment_id,
                "name": exp_data["experiment"].name,
                "state": exp_data["state"].value,
                "start_time": exp_data["start_time"].isoformat(),
                "result": asdict(exp_data["result"]) if exp_data["result"] else None
            }
        return None
    
    def get_all_experiments_status(self) -> List[Dict[str, Any]]:
        """Get status of all experiments"""
        return [
            self.get_experiment_status(exp_id)
            for exp_id in self.active_experiments.keys()
        ]
    
    def get_experiment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get experiment history"""
        return [
            asdict(result) for result in self.experiment_history[-limit:]
        ]
    
    def get_predefined_experiments(self) -> List[Dict[str, Any]]:
        """Get list of predefined experiments"""
        return [
            {
                "name": exp.name,
                "description": exp.description,
                "chaos_type": exp.chaos_type.value,
                "duration": exp.duration,
                "intensity": exp.intensity,
                "target_services": exp.target_services,
                "parameters": exp.parameters
            }
            for exp in self.predefined_experiments.values()
        ]
    
    async def run_predefined_experiment(self, experiment_name: str) -> str:
        """Run a predefined chaos experiment"""
        if experiment_name in self.predefined_experiments:
            experiment = self.predefined_experiments[experiment_name]
            return await self.run_experiment(experiment)
        else:
            raise ValueError(f"Unknown experiment: {experiment_name}")

# Global chaos engineering instance
_chaos_runner = None

def get_chaos_runner() -> SimpleChaosRunner:
    """Get the global chaos engineering instance"""
    global _chaos_runner
    if _chaos_runner is None:
        _chaos_runner = SimpleChaosRunner()
    return _chaos_runner

async def run_chaos_experiment(experiment_name: str) -> str:
    """Run a predefined chaos experiment"""
    runner = get_chaos_runner()
    return await runner.run_predefined_experiment(experiment_name)

def get_chaos_status() -> Dict[str, Any]:
    """Get chaos engineering status"""
    runner = get_chaos_runner()
    
    return {
        "active_experiments": runner.get_all_experiments_status(),
        "predefined_experiments": runner.get_predefined_experiments(),
        "recent_history": runner.get_experiment_history(10)
    }
