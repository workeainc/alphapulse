"""
SDE Advanced Calibration System
Implements isotonic, Platt, and temperature scaling for probability calibration
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import asyncpg
import json

logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    method: str
    calibrated_probability: float
    reliability_score: float
    calibration_error: float
    confidence_interval: Tuple[float, float]

@dataclass
class CalibrationMetrics:
    brier_score: float
    reliability_score: float
    resolution_score: float
    uncertainty_score: float
    calibration_error: float

class SDECalibrationSystem:
    """Advanced calibration system for SDE framework"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.calibrators = {}
        self.calibration_history = {}
        
    async def calibrate_probability(self, 
                                  raw_probability: float, 
                                  method: str = 'isotonic',
                                  model_name: str = 'head_a',
                                  symbol: str = 'BTCUSDT',
                                  timeframe: str = '15m') -> CalibrationResult:
        """Calibrate raw probability using specified method"""
        try:
            if method == 'isotonic':
                return await self._isotonic_calibration(raw_probability, model_name, symbol, timeframe)
            elif method == 'platt':
                return await self._platt_calibration(raw_probability, model_name, symbol, timeframe)
            elif method == 'temperature':
                return await self._temperature_calibration(raw_probability, model_name, symbol, timeframe)
            else:
                raise ValueError(f"Unknown calibration method: {method}")
                
        except Exception as e:
            logger.error(f"❌ Calibration failed: {e}")
            return CalibrationResult(
                method=method,
                calibrated_probability=raw_probability,
                reliability_score=0.5,
                calibration_error=1.0,
                confidence_interval=(0.0, 1.0)
            )
    
    async def _isotonic_calibration(self, 
                                   raw_probability: float, 
                                   model_name: str,
                                   symbol: str,
                                   timeframe: str) -> CalibrationResult:
        """Isotonic regression calibration"""
        try:
            # Get historical data for calibration
            historical_data = await self._get_calibration_data(model_name, symbol, timeframe)
            
            if len(historical_data) < 100:
                logger.warning(f"Insufficient data for isotonic calibration: {len(historical_data)} samples")
                return self._fallback_calibration(raw_probability, 'isotonic')
            
            # Prepare data
            raw_probs = np.array([d['raw_probability'] for d in historical_data])
            actual_outcomes = np.array([d['actual_outcome'] for d in historical_data])
            
            # Fit isotonic regression
            isotonic = IsotonicRegression(out_of_bounds='clip')
            isotonic.fit(raw_probs, actual_outcomes)
            
            # Calibrate probability
            calibrated_prob = isotonic.predict([raw_probability])[0]
            
            # Calculate reliability metrics
            reliability_score = self._calculate_reliability_score(raw_probs, actual_outcomes)
            calibration_error = self._calculate_calibration_error(raw_probs, actual_outcomes)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(calibrated_prob, len(historical_data))
            
            return CalibrationResult(
                method='isotonic',
                calibrated_probability=calibrated_prob,
                reliability_score=reliability_score,
                calibration_error=calibration_error,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"❌ Isotonic calibration failed: {e}")
            return self._fallback_calibration(raw_probability, 'isotonic')
    
    async def _platt_calibration(self, 
                                raw_probability: float, 
                                model_name: str,
                                symbol: str,
                                timeframe: str) -> CalibrationResult:
        """Platt scaling calibration"""
        try:
            # Get historical data for calibration
            historical_data = await self._get_calibration_data(model_name, symbol, timeframe)
            
            if len(historical_data) < 50:
                logger.warning(f"Insufficient data for Platt calibration: {len(historical_data)} samples")
                return self._fallback_calibration(raw_probability, 'platt')
            
            # Prepare data
            raw_probs = np.array([d['raw_probability'] for d in historical_data]).reshape(-1, 1)
            actual_outcomes = np.array([d['actual_outcome'] for d in historical_data])
            
            # Fit Platt scaling
            platt = LogisticRegression()
            platt.fit(raw_probs, actual_outcomes)
            
            # Calibrate probability
            calibrated_prob = platt.predict_proba([[raw_probability]])[0][1]
            
            # Calculate reliability metrics
            reliability_score = self._calculate_reliability_score(raw_probs.flatten(), actual_outcomes)
            calibration_error = self._calculate_calibration_error(raw_probs.flatten(), actual_outcomes)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(calibrated_prob, len(historical_data))
            
            return CalibrationResult(
                method='platt',
                calibrated_probability=calibrated_prob,
                reliability_score=reliability_score,
                calibration_error=calibration_error,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"❌ Platt calibration failed: {e}")
            return self._fallback_calibration(raw_probability, 'platt')
    
    async def _temperature_calibration(self, 
                                     raw_probability: float, 
                                     model_name: str,
                                     symbol: str,
                                     timeframe: str) -> CalibrationResult:
        """Temperature scaling calibration"""
        try:
            # Get historical data for calibration
            historical_data = await self._get_calibration_data(model_name, symbol, timeframe)
            
            if len(historical_data) < 30:
                logger.warning(f"Insufficient data for temperature calibration: {len(historical_data)} samples")
                return self._fallback_calibration(raw_probability, 'temperature')
            
            # Prepare data
            raw_probs = np.array([d['raw_probability'] for d in historical_data])
            actual_outcomes = np.array([d['actual_outcome'] for d in historical_data])
            
            # Find optimal temperature parameter
            temperature = self._optimize_temperature(raw_probs, actual_outcomes)
            
            # Apply temperature scaling
            logits = np.log(raw_probability / (1 - raw_probability))
            calibrated_logits = logits / temperature
            calibrated_prob = 1 / (1 + np.exp(-calibrated_logits))
            
            # Calculate reliability metrics
            reliability_score = self._calculate_reliability_score(raw_probs, actual_outcomes)
            calibration_error = self._calculate_calibration_error(raw_probs, actual_outcomes)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(calibrated_prob, len(historical_data))
            
            return CalibrationResult(
                method='temperature',
                calibrated_probability=calibrated_prob,
                reliability_score=reliability_score,
                calibration_error=calibration_error,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"❌ Temperature calibration failed: {e}")
            return self._fallback_calibration(raw_probability, 'temperature')
    
    async def _get_calibration_data(self, model_name: str, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Get historical calibration data from database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent signal history with outcomes
                data = await conn.fetch("""
                    SELECT 
                        raw_probability,
                        confidence_score as calibrated_probability,
                        sde_final_decision as actual_outcome,
                        timestamp as signal_timestamp
                    FROM sde_signal_history
                    WHERE symbol = $1 
                    AND timeframe = $2
                    AND raw_probability IS NOT NULL
                    AND timestamp >= NOW() - INTERVAL '30 days'
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """, symbol, timeframe)
                
                return [
                    {
                        'raw_probability': row['raw_probability'],
                        'calibrated_probability': row['calibrated_probability'],
                        'actual_outcome': row['actual_outcome'],
                        'timestamp': row['signal_timestamp']
                    }
                    for row in data
                ]
                
        except Exception as e:
            logger.error(f"❌ Failed to get calibration data: {e}")
            return []
    
    def _optimize_temperature(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Optimize temperature parameter for temperature scaling"""
        try:
            # Use cross-validation to find optimal temperature
            temperatures = np.logspace(-2, 2, 50)
            best_score = -np.inf
            best_temp = 1.0
            
            for temp in temperatures:
                # Apply temperature scaling
                logits = np.log(raw_probs / (1 - raw_probs))
                calibrated_logits = logits / temp
                calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
                
                # Calculate reliability score
                score = self._calculate_reliability_score(calibrated_probs, actual_outcomes)
                
                if score > best_score:
                    best_score = score
                    best_temp = temp
            
            return best_temp
            
        except Exception as e:
            logger.error(f"❌ Temperature optimization failed: {e}")
            return 1.0
    
    def _calculate_reliability_score(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate reliability score (how well calibrated the probabilities are)"""
        try:
            # Bin the predictions
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(predicted_probs, bins) - 1
            
            reliability_scores = []
            for i in range(len(bins) - 1):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    avg_pred = np.mean(predicted_probs[mask])
                    avg_actual = np.mean(actual_outcomes[mask])
                    reliability_scores.append(abs(avg_pred - avg_actual))
            
            return 1.0 - np.mean(reliability_scores) if reliability_scores else 0.5
            
        except Exception as e:
            logger.error(f"❌ Reliability score calculation failed: {e}")
            return 0.5
    
    def _calculate_calibration_error(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """Calculate calibration error"""
        try:
            return np.mean((predicted_probs - actual_outcomes) ** 2)
        except Exception as e:
            logger.error(f"❌ Calibration error calculation failed: {e}")
            return 1.0
    
    def _calculate_confidence_interval(self, probability: float, sample_size: int) -> Tuple[float, float]:
        """Calculate confidence interval for calibrated probability"""
        try:
            # Wilson score interval
            z = 1.96  # 95% confidence
            denominator = 1 + z**2 / sample_size
            centre_adjusted_probability = (probability + z * z / (2 * sample_size)) / denominator
            adjusted_standard_error = z * np.sqrt((probability * (1 - probability) + z * z / (4 * sample_size)) / sample_size) / denominator
            
            lower_bound = max(0, centre_adjusted_probability - adjusted_standard_error)
            upper_bound = min(1, centre_adjusted_probability + adjusted_standard_error)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            logger.error(f"❌ Confidence interval calculation failed: {e}")
            return (0.0, 1.0)
    
    def _fallback_calibration(self, raw_probability: float, method: str) -> CalibrationResult:
        """Fallback calibration when insufficient data"""
        return CalibrationResult(
            method=method,
            calibrated_probability=raw_probability,
            reliability_score=0.5,
            calibration_error=1.0,
            confidence_interval=(0.0, 1.0)
        )
    
    async def calculate_calibration_metrics(self, 
                                          model_name: str, 
                                          symbol: str, 
                                          timeframe: str) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics"""
        try:
            historical_data = await self._get_calibration_data(model_name, symbol, timeframe)
            
            if len(historical_data) < 10:
                return CalibrationMetrics(
                    brier_score=1.0,
                    reliability_score=0.5,
                    resolution_score=0.0,
                    uncertainty_score=0.25,
                    calibration_error=1.0
                )
            
            raw_probs = np.array([d['raw_probability'] for d in historical_data])
            actual_outcomes = np.array([d['actual_outcome'] for d in historical_data])
            
            # Brier score
            brier_score = np.mean((raw_probs - actual_outcomes) ** 2)
            
            # Reliability score
            reliability_score = self._calculate_reliability_score(raw_probs, actual_outcomes)
            
            # Resolution score
            resolution_score = np.var(raw_probs)
            
            # Uncertainty score
            uncertainty_score = np.mean(actual_outcomes) * (1 - np.mean(actual_outcomes))
            
            # Calibration error
            calibration_error = self._calculate_calibration_error(raw_probs, actual_outcomes)
            
            return CalibrationMetrics(
                brier_score=brier_score,
                reliability_score=reliability_score,
                resolution_score=resolution_score,
                uncertainty_score=uncertainty_score,
                calibration_error=calibration_error
            )
            
        except Exception as e:
            logger.error(f"❌ Calibration metrics calculation failed: {e}")
            return CalibrationMetrics(
                brier_score=1.0,
                reliability_score=0.5,
                resolution_score=0.0,
                uncertainty_score=0.25,
                calibration_error=1.0
            )
    
    async def store_calibration_result(self, 
                                     model_name: str,
                                     symbol: str,
                                     timeframe: str,
                                     raw_probability: float,
                                     calibrated_probability: float,
                                     method: str,
                                     reliability_score: float) -> None:
        """Store calibration result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sde_calibration_history (
                        model_name, symbol, timeframe, raw_probability, 
                        calibrated_probability, calibration_method, reliability_score,
                        calibration_timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                """, model_name, symbol, timeframe, raw_probability, 
                     calibrated_probability, method, reliability_score)
                
        except Exception as e:
            logger.error(f"❌ Failed to store calibration result: {e}")
