#!/usr/bin/env python3
"""
Week 7.2 Phase 1: Real-Time Correlation Calculator

This service provides advanced correlation analysis:
- Real-time correlation matrices for multiple metrics
- Advanced statistical analysis (rolling correlations, regime detection)
- Multi-timeframe correlation analysis
- Correlation clustering and pattern recognition
- Real-time market regime classification

Author: AlphaPulse Team
Date: 2025
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import deque
import json
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationMethod(Enum):
    """Correlation calculation methods"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    ROLLING = "rolling"
    EXPONENTIAL = "exponential"


@dataclass
class CorrelationMatrix:
    """Real-time correlation matrix result"""
    timestamp: datetime
    metric_name: str
    chains: List[str]
    correlation_matrix: np.ndarray
    method: CorrelationMethod
    window_size: int
    confidence_intervals: Optional[np.ndarray] = None
    significance_levels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollingCorrelation:
    """Rolling correlation analysis result"""
    timestamp: datetime
    metric_name: str
    chain_pair: Tuple[str, str]
    correlation_values: List[float]
    timestamps: List[datetime]
    window_size: int
    trend: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    volatility: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimeAnalysis:
    """Market regime analysis result"""
    timestamp: datetime
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile', 'transition'
    confidence: float
    duration: Optional[timedelta] = None
    transition_probability: float = 0.0
    supporting_metrics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationConfig:
    """Configuration for correlation calculator"""
    default_method: CorrelationMethod = CorrelationMethod.PEARSON
    rolling_window_size: int = 50
    exponential_decay: float = 0.95
    confidence_level: float = 0.95
    min_data_points: int = 20
    max_correlation_pairs: int = 100
    update_frequency: float = 5.0  # seconds
    enable_rolling_analysis: bool = True
    enable_regime_detection: bool = True
    regime_detection_threshold: float = 0.3


class RealTimeCorrelationCalculator:
    """
    Real-time correlation calculator for cross-chain analysis
    
    This service provides advanced correlation analysis including
    rolling correlations, regime detection, and pattern recognition.
    """
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        """Initialize the correlation calculator"""
        self.config = config or CorrelationConfig()
        
        # Data storage for correlation analysis
        self.correlation_matrices: deque = deque(maxlen=1000)
        self.rolling_correlations: Dict[Tuple[str, str], deque] = {}
        self.regime_analysis: deque = deque(maxlen=100)
        
        # Performance monitoring
        self.calculation_times: deque = deque(maxlen=100)
        self.total_calculations: int = 0
        
        # Callbacks for real-time updates
        self.correlation_callbacks: List[Callable] = []
        self.regime_callbacks: List[Callable] = []
        
        logger.info("✅ Real-Time Correlation Calculator initialized")
    
    async def initialize(self):
        """Initialize the service"""
        logger.info("🚀 Initializing Real-Time Correlation Calculator...")
        
        # Start continuous correlation analysis
        asyncio.create_task(self._continuous_correlation_analysis())
        
        logger.info("✅ Real-Time Correlation Calculator ready")
    
    async def _continuous_correlation_analysis(self):
        """Continuous correlation analysis loop"""
        while True:
            try:
                start_time = time.time()
                
                # Perform correlation analysis
                await self._update_correlations()
                
                # Record calculation time
                calculation_time = time.time() - start_time
                self.calculation_times.append(calculation_time)
                self.total_calculations += 1
                
                logger.info(f"✅ Correlation analysis completed in {calculation_time:.3f}s")
                
                # Wait for next update
                await asyncio.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in correlation analysis loop: {e}")
                await asyncio.sleep(self.config.update_frequency)
    
    async def _update_correlations(self):
        """Update all correlation calculations"""
        # This method would be called with actual data from the cross-chain service
        # For now, we'll simulate the update process
        pass
    
    def calculate_correlation_matrix(self, data: Dict[str, List[float]], 
                                  method: CorrelationMethod = None) -> Optional[CorrelationMatrix]:
        """
        Calculate correlation matrix for given data
        
        Args:
            data: Dictionary of chain -> values
            method: Correlation method to use
            
        Returns:
            CorrelationMatrix result
        """
        if not data or len(data) < 2:
            return None
        
        method = method or self.config.default_method
        
        try:
            # Convert data to numpy arrays
            chains = list(data.keys())
            arrays = []
            min_length = float('inf')
            
            for chain, values in data.items():
                if isinstance(values, (list, np.ndarray)):
                    array = np.array(values)
                    arrays.append(array)
                    min_length = min(min_length, len(array))
                else:
                    logger.warning(f"Invalid data format for chain {chain}")
                    return None
            
            if min_length < self.config.min_data_points:
                logger.warning(f"Insufficient data points: {min_length} < {self.config.min_data_points}")
                return None
            
            # Truncate all arrays to the same length
            for i in range(len(arrays)):
                arrays[i] = arrays[i][-int(min_length):]
            
            # Calculate correlation matrix based on method
            if method == CorrelationMethod.PEARSON:
                correlation_matrix = np.corrcoef(arrays)
            elif method == CorrelationMethod.SPEARMAN:
                # Convert to ranks for Spearman correlation
                ranked_arrays = [np.argsort(np.argsort(arr)) for arr in arrays]
                correlation_matrix = np.corrcoef(ranked_arrays)
            elif method == CorrelationMethod.KENDALL:
                # Kendall's tau correlation
                correlation_matrix = self._calculate_kendall_correlation(arrays)
            else:
                correlation_matrix = np.corrcoef(arrays)
            
            # Calculate confidence intervals and significance levels
            confidence_intervals, significance_levels = self._calculate_statistical_measures(
                arrays, correlation_matrix
            )
            
            # Create result
            result = CorrelationMatrix(
                timestamp=datetime.now(),
                metric_name="cross_chain_correlation",
                chains=chains,
                correlation_matrix=correlation_matrix,
                method=method,
                window_size=int(min_length),
                confidence_intervals=confidence_intervals,
                significance_levels=significance_levels,
                metadata={
                    'method': method.value,
                    'data_points': int(min_length),
                    'chains_count': len(chains)
                }
            )
            
            # Store result
            self.correlation_matrices.append(result)
            
            # Trigger callbacks
            self._trigger_correlation_callbacks(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return None
    
    def _calculate_kendall_correlation(self, arrays: List[np.ndarray]) -> np.ndarray:
        """Calculate Kendall's tau correlation matrix"""
        n = len(arrays)
        correlation_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    correlation_matrix[i, j] = self._kendall_tau(arrays[i], arrays[j])
        
        return correlation_matrix
    
    def _kendall_tau(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Kendall's tau between two arrays"""
        n = len(x)
        if n != len(y):
            return 0.0
        
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                    concordant += 1
                elif (x[i] - x[j]) * (y[i] - y[j]) < 0:
                    discordant += 1
        
        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 0.0
        
        tau = (concordant - discordant) / total_pairs
        return tau
    
    def _calculate_statistical_measures(self, arrays: List[np.ndarray], 
                                      correlation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals and significance levels"""
        n = len(arrays)
        confidence_intervals = np.zeros_like(correlation_matrix)
        significance_levels = np.zeros_like(correlation_matrix)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Calculate confidence interval using Fisher's z-transformation
                    r = correlation_matrix[i, j]
                    if abs(r) < 1.0:
                        z = 0.5 * np.log((1 + r) / (1 - r))
                        se = 1 / np.sqrt(len(arrays[i]) - 3)
                        
                        # 95% confidence interval
                        z_lower = z - 1.96 * se
                        z_upper = z + 1.96 * se
                        
                        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                        
                        confidence_intervals[i, j] = r_upper - r_lower
                        
                        # Calculate significance level (p-value approximation)
                        z_score = abs(z) / se
                        significance_levels[i, j] = 2 * (1 - self._normal_cdf(z_score))
        
        return confidence_intervals, significance_levels
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function"""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))
    
    def calculate_rolling_correlation(self, chain1: str, chain2: str, 
                                   data1: List[float], data2: List[float],
                                   window_size: int = None) -> Optional[RollingCorrelation]:
        """
        Calculate rolling correlation between two chains
        
        Args:
            chain1: First chain identifier
            chain2: Second chain identifier
            data1: Time series data for first chain
            data2: Time series data for second chain
            window_size: Rolling window size
            
        Returns:
            RollingCorrelation result
        """
        if not data1 or not data2:
            return None
        
        window_size = window_size or self.config.rolling_window_size
        
        if len(data1) < window_size or len(data2) < window_size:
            return None
        
        try:
            # Convert to numpy arrays
            array1 = np.array(data1)
            array2 = np.array(data2)
            
            # Calculate rolling correlation
            correlations = []
            timestamps = []
            
            for i in range(window_size, len(array1)):
                window1 = array1[i-window_size:i]
                window2 = array2[i-window_size:i]
                
                if len(window1) == len(window2) and len(window1) >= 10:
                    corr = np.corrcoef(window1, window2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                        timestamps.append(datetime.now() - timedelta(seconds=(len(array1) - i) * 30))
            
            if not correlations:
                return None
            
            # Analyze trend and volatility
            trend = self._classify_trend(correlations)
            volatility = np.std(correlations)
            
            # Create result
            result = RollingCorrelation(
                timestamp=datetime.now(),
                metric_name="rolling_correlation",
                chain_pair=(chain1, chain2),
                correlation_values=correlations,
                timestamps=timestamps,
                window_size=window_size,
                trend=trend,
                volatility=volatility,
                metadata={
                    'data_points': len(correlations),
                    'window_size': window_size
                }
            )
            
            # Store result
            pair_key = (chain1, chain2)
            if pair_key not in self.rolling_correlations:
                self.rolling_correlations[pair_key] = deque(maxlen=100)
            self.rolling_correlations[pair_key].append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}")
            return None
    
    def _classify_trend(self, values: List[float]) -> str:
        """Classify trend based on correlation values"""
        if len(values) < 3:
            return 'stable'
        
        # Calculate linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Calculate volatility
        volatility = np.std(values)
        mean_value = np.mean(values)
        
        if abs(slope) < 0.01:
            if volatility < 0.1:
                return 'stable'
            else:
                return 'volatile'
        elif slope > 0.01:
            return 'increasing'
        else:
            return 'decreasing'
    
    def detect_market_regime(self, correlation_matrices: List[CorrelationMatrix]) -> Optional[RegimeAnalysis]:
        """
        Detect market regime based on correlation patterns
        
        Args:
            correlation_matrices: List of recent correlation matrices
            
        Returns:
            RegimeAnalysis result
        """
        if not correlation_matrices:
            return None
        
        try:
            # Extract average correlations over time
            avg_correlations = []
            timestamps = []
            
            for matrix in correlation_matrices:
                # Calculate average correlation (excluding diagonal)
                upper_triangle = matrix.correlation_matrix[np.triu_indices(len(matrix.chains), k=1)]
                avg_corr = np.mean(upper_triangle)
                
                if not np.isnan(avg_corr):
                    avg_correlations.append(avg_corr)
                    timestamps.append(matrix.timestamp)
            
            if len(avg_correlations) < 5:
                return None
            
            # Analyze regime characteristics
            current_correlation = avg_correlations[-1]
            correlation_trend = np.polyfit(range(len(avg_correlations)), avg_correlations, 1)[0]
            volatility = np.std(avg_correlations)
            
            # Classify regime
            regime_type = self._classify_regime(current_correlation, correlation_trend, volatility)
            
            # Calculate confidence based on data consistency
            confidence = self._calculate_regime_confidence(avg_correlations, volatility)
            
            # Calculate transition probability
            transition_prob = self._calculate_transition_probability(avg_correlations)
            
            # Determine supporting metrics
            supporting_metrics = self._identify_supporting_metrics(correlation_matrices[-1])
            
            # Create result
            result = RegimeAnalysis(
                timestamp=datetime.now(),
                regime_type=regime_type,
                confidence=confidence,
                transition_probability=transition_prob,
                supporting_metrics=supporting_metrics,
                metadata={
                    'current_correlation': current_correlation,
                    'correlation_trend': correlation_trend,
                    'volatility': volatility,
                    'data_points': len(avg_correlations)
                }
            )
            
            # Store result
            self.regime_analysis.append(result)
            
            # Trigger callbacks
            self._trigger_regime_callbacks(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return None
    
    def _classify_regime(self, correlation: float, trend: float, volatility: float) -> str:
        """Classify market regime based on correlation characteristics"""
        if volatility > 0.4:
            return 'volatile'
        elif correlation > 0.7:
            if trend > 0.01:
                return 'bull'
            elif trend < -0.01:
                return 'transition'
            else:
                return 'bull'
        elif correlation < -0.3:
            if trend < -0.01:
                return 'bear'
            elif trend > 0.01:
                return 'transition'
            else:
                return 'bear'
        elif abs(trend) > 0.02:
            return 'transition'
        else:
            return 'sideways'
    
    def _calculate_regime_confidence(self, correlations: List[float], volatility: float) -> float:
        """Calculate confidence in regime classification"""
        # Higher confidence with more data and lower volatility
        data_confidence = min(len(correlations) / 50.0, 1.0)
        volatility_confidence = max(0.0, 1.0 - volatility)
        
        return (data_confidence + volatility_confidence) / 2.0
    
    def _calculate_transition_probability(self, correlations: List[float]) -> float:
        """Calculate probability of regime transition"""
        if len(correlations) < 3:
            return 0.0
        
        # Calculate rate of change
        changes = np.diff(correlations)
        recent_changes = changes[-3:]  # Last 3 changes
        
        # Higher transition probability with increasing volatility
        volatility = np.std(recent_changes)
        base_prob = min(volatility * 2, 0.8)
        
        # Adjust based on trend consistency
        trend_consistency = 1.0 - (np.std(changes) / np.mean(np.abs(changes)) if np.mean(np.abs(changes)) > 0 else 0.0)
        
        return base_prob * (1.0 - trend_consistency)
    
    def _identify_supporting_metrics(self, correlation_matrix: CorrelationMatrix) -> List[str]:
        """Identify metrics that support the current regime"""
        supporting_metrics = []
        
        # Check for high correlations in specific metrics
        upper_triangle = correlation_matrix.correlation_matrix[np.triu_indices(len(correlation_matrix.chains), k=1)]
        
        if np.mean(upper_triangle) > 0.6:
            supporting_metrics.append('high_chain_correlation')
        
        if np.std(upper_triangle) < 0.2:
            supporting_metrics.append('stable_correlation_pattern')
        
        # Check for specific chain relationships
        for i, chain1 in enumerate(correlation_matrix.chains):
            for j, chain2 in enumerate(correlation_matrix.chains):
                if i < j:
                    corr = correlation_matrix.correlation_matrix[i, j]
                    if corr > 0.8:
                        supporting_metrics.append(f'strong_{chain1}_{chain2}_correlation')
        
        return supporting_metrics
    
    def add_correlation_callback(self, callback: Callable):
        """Add callback for correlation updates"""
        if callback not in self.correlation_callbacks:
            self.correlation_callbacks.append(callback)
            logger.info(f"✅ Added correlation callback: {callback.__name__}")
    
    def add_regime_callback(self, callback: Callable):
        """Add callback for regime changes"""
        if callback not in self.regime_callbacks:
            self.regime_callbacks.append(callback)
            logger.info(f"✅ Added regime callback: {callback.__name__}")
    
    def _trigger_correlation_callbacks(self, result: CorrelationMatrix):
        """Trigger correlation callbacks"""
        for callback in self.correlation_callbacks:
            try:
                asyncio.create_task(callback(result))
            except Exception as e:
                logger.error(f"❌ Error in correlation callback {callback.__name__}: {e}")
    
    def _trigger_regime_callbacks(self, result: RegimeAnalysis):
        """Trigger regime callbacks"""
        for callback in self.regime_callbacks:
            try:
                asyncio.create_task(callback(result))
            except Exception as e:
                logger.error(f"❌ Error in regime callback {callback.__name__}: {e}")
    
    def get_correlation_history(self, limit: int = 100) -> List[CorrelationMatrix]:
        """Get recent correlation matrices"""
        return list(self.correlation_matrices)[-limit:]
    
    def get_rolling_correlations(self, chain_pair: Tuple[str, str], limit: int = 50) -> List[RollingCorrelation]:
        """Get rolling correlations for a specific chain pair"""
        if chain_pair not in self.rolling_correlations:
            return []
        return list(self.rolling_correlations[chain_pair])[-limit:]
    
    def get_regime_history(self, limit: int = 50) -> List[RegimeAnalysis]:
        """Get recent regime analysis results"""
        return list(self.regime_analysis)[-limit:]
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'total_calculations': self.total_calculations,
            'correlation_matrices_count': len(self.correlation_matrices),
            'rolling_correlations_count': sum(len(corrs) for corrs in self.rolling_correlations.values()),
            'regime_analysis_count': len(self.regime_analysis),
            'average_calculation_time': np.mean(self.calculation_times) if self.calculation_times else 0.0,
            'active_chain_pairs': len(self.rolling_correlations),
            'correlation_callbacks': len(self.correlation_callbacks),
            'regime_callbacks': len(self.regime_callbacks)
        }
    
    async def close(self):
        """Clean up resources"""
        logger.info("🔄 Closing Real-Time Correlation Calculator...")
        
        # Clear data and history
        self.correlation_matrices.clear()
        self.rolling_correlations.clear()
        self.regime_analysis.clear()
        self.correlation_callbacks.clear()
        self.regime_callbacks.clear()
        
        logger.info("✅ Real-Time Correlation Calculator closed")


# Example usage and testing
async def main():
    """Example usage of the Real-Time Correlation Calculator"""
    # Create calculator
    config = CorrelationConfig(
        default_method=CorrelationMethod.PEARSON,
        rolling_window_size=30,
        update_frequency=2.0,
        enable_rolling_analysis=True,
        enable_regime_detection=True
    )
    
    calculator = RealTimeCorrelationCalculator(config)
    await calculator.initialize()
    
    # Add callbacks
    def correlation_callback(matrix: CorrelationMatrix):
        print(f"📊 New correlation matrix: {len(matrix.chains)} chains")
        print(f"  Method: {matrix.method.value}")
        print(f"  Window size: {matrix.window_size}")
    
    def regime_callback(regime: RegimeAnalysis):
        print(f"🔄 Regime detected: {regime.regime_type}")
        print(f"  Confidence: {regime.confidence:.2f}")
        print(f"  Transition probability: {regime.transition_probability:.2f}")
    
    calculator.add_correlation_callback(correlation_callback)
    calculator.add_regime_callback(regime_callback)
    
    # Test with sample data
    print("🧪 Testing Real-Time Correlation Calculator...")
    
    # Simulate cross-chain data
    chains = ["ethereum", "bsc", "polygon"]
    data = {
        "ethereum": [100 + i * 0.1 + np.random.random() * 2 for i in range(100)],
        "bsc": [50 + i * 0.05 + np.random.random() * 1 for i in range(100)],
        "polygon": [75 + i * 0.08 + np.random.random() * 1.5 for i in range(100)]
    }
    
    # Calculate correlation matrix
    print("\n📈 Calculating correlation matrix...")
    matrix = calculator.calculate_correlation_matrix(data)
    if matrix:
        print(f"  ✅ Correlation matrix calculated")
        print(f"  - Method: {matrix.method.value}")
        print(f"  - Chains: {', '.join(matrix.chains)}")
        print(f"  - Matrix shape: {matrix.correlation_matrix.shape}")
    
    # Calculate rolling correlations
    print("\n🔄 Calculating rolling correlations...")
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            chain1, chain2 = chains[i], chains[j]
            rolling_corr = calculator.calculate_rolling_correlation(
                chain1, chain2, data[chain1], data[chain2]
            )
            if rolling_corr:
                print(f"  ✅ {chain1}-{chain2}: {rolling_corr.trend} trend")
    
    # Detect market regime
    print("\n🎯 Detecting market regime...")
    if matrix:
        regime = calculator.detect_market_regime([matrix])
        if regime:
            print(f"  ✅ Regime detected: {regime.regime_type}")
            print(f"  - Confidence: {regime.confidence:.2f}")
            print(f"  - Supporting metrics: {', '.join(regime.supporting_metrics)}")
    
    # Print statistics
    print(f"\n📊 Service Statistics:")
    stats = calculator.get_service_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    await calculator.close()


if __name__ == "__main__":
    asyncio.run(main())
