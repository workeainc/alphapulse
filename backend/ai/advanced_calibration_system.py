"""
Advanced Calibration System for Signal Accuracy Improvement
Implements multiple calibration methods to achieve 90%+ signal accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
import asyncpg
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import json

logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Result from calibration process"""
    calibrated_probability: float
    calibration_method: str
    confidence_interval: Tuple[float, float]
    reliability_score: float
    method_performance: Dict[str, float]

class AdvancedCalibrationSystem:
    """Advanced calibration system for improving signal accuracy"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.calibration_models = {}
        self.performance_history = {}
        self.method_weights = {
            'isotonic': 0.4,
            'platt': 0.3,
            'temperature': 0.2,
            'ensemble': 0.1
        }
        
    async def initialize(self):
        """Initialize calibration system"""
        # Initialize with default settings
        logger.info("✅ Advanced Calibration System initialized")
    
    async def calibrate_probability(self, 
                                  raw_probability: float,
                                  calibration_method: str = 'ensemble',
                                  model_name: str = None,
                                  symbol: str = None,
                                  timeframe: str = None,
                                  features: Dict[str, float] = None,
                                  market_regime: str = 'unknown') -> CalibrationResult:
        """Calibrate probability using multiple methods"""
        try:
            # Get historical data for calibration
            historical_data = await self._get_calibration_data(symbol, timeframe, market_regime)
            
            if len(historical_data) < 100:
                logger.warning(f"Insufficient calibration data for {symbol}: {len(historical_data)} samples")
                return self._fallback_calibration(raw_probability)
            
            # Apply multiple calibration methods
            calibration_results = {}
            
            # 1. Isotonic Regression Calibration
            iso_result = await self._isotonic_calibration(raw_probability, historical_data)
            calibration_results['isotonic'] = iso_result
            
            # 2. Platt Scaling Calibration
            platt_result = await self._platt_calibration(raw_probability, historical_data)
            calibration_results['platt'] = platt_result
            
            # 3. Temperature Scaling Calibration
            temp_result = await self._temperature_calibration(raw_probability, historical_data)
            calibration_results['temperature'] = temp_result
            
            # 4. Ensemble Calibration
            ensemble_result = await self._ensemble_calibration(calibration_results, features, market_regime)
            calibration_results['ensemble'] = ensemble_result
            
            # Combine results using weighted average
            final_probability = self._combine_calibrations(calibration_results)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(calibration_results)
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score(calibration_results, historical_data)
            
            # Store calibration result
            await self._store_calibration_result(
                symbol, timeframe, raw_probability, final_probability, 
                calibration_results, reliability_score
            )
            
            return CalibrationResult(
                calibrated_probability=final_probability,
                calibration_method='ensemble',
                confidence_interval=confidence_interval,
                reliability_score=reliability_score,
                method_performance={k: v.get('performance', 0.0) for k, v in calibration_results.items()}
            )
            
        except Exception as e:
            logger.error(f"❌ Calibration failed: {e}")
            return self._fallback_calibration(raw_probability)
    
    async def _isotonic_calibration(self, probability: float, historical_data: List[Dict]) -> Dict[str, Any]:
        """Isotonic regression calibration"""
        try:
            # Prepare data
            predictions = [d['prediction'] for d in historical_data]
            actuals = [d['actual'] for d in historical_data]
            
            # Fit isotonic regression
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(predictions, actuals)
            
            # Calibrate probability
            calibrated_prob = iso_reg.predict([probability])[0]
            
            # Calculate performance
            calibrated_predictions = iso_reg.predict(predictions)
            performance = 1 - brier_score_loss(actuals, calibrated_predictions)
            
            return {
                'calibrated_probability': calibrated_prob,
                'performance': performance,
                'method': 'isotonic'
            }
            
        except Exception as e:
            logger.error(f"Isotonic calibration failed: {e}")
            return {
                'calibrated_probability': probability,
                'performance': 0.5,
                'method': 'isotonic'
            }
    
    async def _platt_calibration(self, probability: float, historical_data: List[Dict]) -> Dict[str, Any]:
        """Platt scaling calibration"""
        try:
            # Prepare data
            predictions = np.array([d['prediction'] for d in historical_data]).reshape(-1, 1)
            actuals = np.array([d['actual'] for d in historical_data])
            
            # Fit Platt scaling
            platt_reg = LogisticRegression()
            platt_reg.fit(predictions, actuals)
            
            # Calibrate probability
            calibrated_prob = platt_reg.predict_proba([[probability]])[0][1]
            
            # Calculate performance
            calibrated_predictions = platt_reg.predict_proba(predictions)[:, 1]
            performance = 1 - brier_score_loss(actuals, calibrated_predictions)
            
            return {
                'calibrated_probability': calibrated_prob,
                'performance': performance,
                'method': 'platt'
            }
            
        except Exception as e:
            logger.error(f"Platt calibration failed: {e}")
            return {
                'calibrated_probability': probability,
                'performance': 0.5,
                'method': 'platt'
            }
    
    async def _temperature_calibration(self, probability: float, historical_data: List[Dict]) -> Dict[str, Any]:
        """Advanced temperature scaling calibration"""
        try:
            # Find optimal temperature using validation
            temperatures = np.linspace(0.1, 5.0, 50)
            best_temp = 1.0
            best_score = 0.0
            
            predictions = [d['prediction'] for d in historical_data]
            actuals = [d['actual'] for d in historical_data]
            
            for temp in temperatures:
                # Apply temperature scaling
                scaled_probs = []
                for pred in predictions:
                    logit = np.log(pred / (1 - pred))
                    scaled_logit = logit / temp
                    scaled_prob = 1 / (1 + np.exp(-scaled_logit))
                    scaled_probs.append(scaled_prob)
                
                # Calculate performance
                score = 1 - brier_score_loss(actuals, scaled_probs)
                if score > best_score:
                    best_score = score
                    best_temp = temp
            
            # Apply best temperature
            logit = np.log(probability / (1 - probability))
            scaled_logit = logit / best_temp
            calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
            
            return {
                'calibrated_probability': calibrated_prob,
                'performance': best_score,
                'method': 'temperature',
                'optimal_temperature': best_temp
            }
            
        except Exception as e:
            logger.error(f"Temperature calibration failed: {e}")
            return {
                'calibrated_probability': probability,
                'performance': 0.5,
                'method': 'temperature'
            }
    
    async def _ensemble_calibration(self, 
                                  individual_results: Dict[str, Dict],
                                  features: Dict[str, float],
                                  market_regime: str) -> Dict[str, Any]:
        """Ensemble calibration combining multiple methods"""
        try:
            # Dynamic weighting based on market regime and features
            regime_weights = {
                'bullish': {'isotonic': 0.5, 'platt': 0.3, 'temperature': 0.2},
                'bearish': {'isotonic': 0.3, 'platt': 0.5, 'temperature': 0.2},
                'sideways': {'isotonic': 0.4, 'platt': 0.3, 'temperature': 0.3},
                'volatile': {'isotonic': 0.2, 'platt': 0.2, 'temperature': 0.6}
            }
            
            weights = regime_weights.get(market_regime, self.method_weights)
            
            # Weighted combination
            total_weight = 0.0
            weighted_prob = 0.0
            
            for method, result in individual_results.items():
                if method in weights and 'calibrated_probability' in result:
                    weight = weights[method] * result.get('performance', 0.5)
                    weighted_prob += result['calibrated_probability'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_prob = weighted_prob / total_weight
            else:
                ensemble_prob = sum(r.get('calibrated_probability', 0.5) for r in individual_results.values()) / len(individual_results)
            
            # Calculate ensemble performance
            ensemble_performance = np.mean([r.get('performance', 0.5) for r in individual_results.values()])
            
            return {
                'calibrated_probability': ensemble_prob,
                'performance': ensemble_performance,
                'method': 'ensemble',
                'weights_used': weights
            }
            
        except Exception as e:
            logger.error(f"Ensemble calibration failed: {e}")
            return {
                'calibrated_probability': 0.5,
                'performance': 0.5,
                'method': 'ensemble'
            }
    
    def _combine_calibrations(self, calibration_results: Dict[str, Dict]) -> float:
        """Combine multiple calibration results"""
        try:
            total_weight = 0.0
            weighted_prob = 0.0
            
            for method, result in calibration_results.items():
                weight = self.method_weights.get(method, 0.1)
                performance = result.get('performance', 0.5)
                adjusted_weight = weight * performance
                
                weighted_prob += result.get('calibrated_probability', 0.5) * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                return weighted_prob / total_weight
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Combination failed: {e}")
            return 0.5
    
    def _calculate_confidence_interval(self, calibration_results: Dict[str, Dict]) -> Tuple[float, float]:
        """Calculate confidence interval for calibrated probability"""
        try:
            probabilities = [r.get('calibrated_probability', 0.5) for r in calibration_results.values()]
            mean_prob = np.mean(probabilities)
            std_prob = np.std(probabilities)
            
            # 95% confidence interval
            margin = 1.96 * std_prob
            lower = max(0.0, mean_prob - margin)
            upper = min(1.0, mean_prob + margin)
            
            return (lower, upper)
            
        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return (0.0, 1.0)
    
    def _calculate_reliability_score(self, 
                                   calibration_results: Dict[str, Dict],
                                   historical_data: List[Dict]) -> float:
        """Calculate reliability score based on calibration consistency"""
        try:
            # Method agreement score
            probabilities = [r.get('calibrated_probability', 0.5) for r in calibration_results.values()]
            agreement_score = 1.0 - np.std(probabilities)
            
            # Historical performance score
            if historical_data:
                recent_performance = np.mean([d.get('accuracy', 0.5) for d in historical_data[-50:]])
            else:
                recent_performance = 0.5
            
            # Data quality score
            data_quality = min(1.0, len(historical_data) / 1000.0)
            
            # Combined reliability score
            reliability = (agreement_score * 0.4 + recent_performance * 0.4 + data_quality * 0.2)
            
            return min(1.0, max(0.0, reliability))
            
        except Exception as e:
            logger.error(f"Reliability score calculation failed: {e}")
            return 0.5
    
    async def _get_calibration_data(self, symbol: str, timeframe: str, market_regime: str) -> List[Dict]:
        """Get historical data for calibration"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent signal predictions and outcomes
                query = """
                    SELECT ensemble_prediction as prediction, confidence_score, signal_direction, fusion_method
                    FROM sde_ensemble_predictions ep
                    WHERE ep.symbol = $1 AND ep.timeframe = $2
                        AND ep.timestamp > $3
                    ORDER BY ep.timestamp DESC
                    LIMIT 1000
                """
                
                cutoff_time = datetime.now() - timedelta(days=30)
                rows = await conn.fetch(query, symbol, timeframe, cutoff_time)
                
                calibration_data = []
                for row in rows:
                    # Convert signal direction to numeric for calibration
                    direction_value = 1.0 if row['signal_direction'] == 'LONG' else 0.0
                    calibration_data.append({
                        'prediction': float(row['prediction']),
                        'actual': direction_value,
                        'confidence': float(row['confidence_score']),
                        'regime': row['fusion_method'] or 'unknown'
                    })
                
                return calibration_data
                
        except Exception as e:
            logger.error(f"Failed to get calibration data: {e}")
            return []
    
    async def _store_calibration_result(self, 
                                      symbol: str, 
                                      timeframe: str,
                                      raw_probability: float,
                                      calibrated_probability: float,
                                      calibration_results: Dict[str, Dict],
                                      reliability_score: float):
        """Store calibration result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sde_calibration_history 
                    (symbol, timeframe, raw_probability, calibrated_probability, 
                     calibration_methods, reliability_score, calibration_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, symbol, timeframe, raw_probability, calibrated_probability,
                     json.dumps(calibration_results), reliability_score, datetime.now())
                
        except Exception as e:
            logger.error(f"Failed to store calibration result: {e}")
    
    def _fallback_calibration(self, probability: float) -> CalibrationResult:
        """Fallback calibration when advanced methods fail"""
        return CalibrationResult(
            calibrated_probability=probability,
            calibration_method='fallback',
            confidence_interval=(probability - 0.1, probability + 0.1),
            reliability_score=0.5,
            method_performance={'fallback': 0.5}
        )
    
    async def get_calibration_performance(self, symbol: str = None, timeframe: str = None) -> Dict[str, Any]:
        """Get calibration performance statistics"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT AVG(reliability_score) as avg_reliability,
                           COUNT(*) as total_calibrations,
                           AVG(ABS(raw_probability - calibrated_probability)) as avg_adjustment
                    FROM sde_calibration_history
                    WHERE 1=1
                """
                params = []
                
                if symbol:
                    query += " AND symbol = $1"
                    params.append(symbol)
                
                if timeframe:
                    query += f" AND timeframe = ${len(params) + 1}"
                    params.append(timeframe)
                
                row = await conn.fetchrow(query, *params)
                
                return {
                    'avg_reliability': float(row['avg_reliability']) if row['avg_reliability'] else 0.0,
                    'total_calibrations': row['total_calibrations'],
                    'avg_adjustment': float(row['avg_adjustment']) if row['avg_adjustment'] else 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get calibration performance: {e}")
            return {'avg_reliability': 0.0, 'total_calibrations': 0, 'avg_adjustment': 0.0}
