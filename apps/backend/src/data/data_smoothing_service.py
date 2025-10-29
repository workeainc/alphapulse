#!/usr/bin/env python3
"""
Data Smoothing Service for AlphaPulse
Phase 3: Advanced Smoothing
Handles indicator-based smoothing and adaptive smoothing based on market volatility
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import talib

logger = logging.getLogger(__name__)

@dataclass
class SmoothingConfig:
    """Configuration for data smoothing"""
    method: str  # 'rolling_median', 'gaussian', 'savitzky_golay', 'exponential', 'adaptive'
    window_size: int = 20
    std_dev: float = 2.0
    polynomial_order: int = 3
    alpha: float = 0.1
    volatility_threshold: float = 0.02
    min_window: int = 5
    max_window: int = 100

@dataclass
class SmoothingResult:
    """Result of data smoothing process"""
    smoothed_data: pd.DataFrame
    smoothing_report: Dict[str, Any]
    quality_score: float
    volatility_profile: Dict[str, float]
    warnings: List[str]
    errors: List[str]

class DataSmoothingService:
    """
    Advanced data smoothing service for AlphaPulse
    Handles indicator-based smoothing and adaptive smoothing based on market volatility
    """
    
    def __init__(self):
        # Default smoothing configurations
        self.default_configs = {
            'price': SmoothingConfig(
                method='adaptive',
                window_size=20,
                std_dev=2.0,
                volatility_threshold=0.02
            ),
            'volume': SmoothingConfig(
                method='rolling_median',
                window_size=15,
                std_dev=1.5
            ),
            'indicators': SmoothingConfig(
                method='gaussian',
                window_size=10,
                std_dev=1.0
            )
        }
        
        # Performance tracking
        self.stats = {
            'total_smoothing_operations': 0,
            'successful_smoothing': 0,
            'failed_smoothing': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0,
            'methods_used': {}
        }
        
        logger.info("🚀 Data Smoothing Service initialized")
    
    def smooth_candlestick_data(
        self, 
        data: pd.DataFrame, 
        config: Optional[SmoothingConfig] = None,
        smooth_indicators: bool = True,
        preserve_original: bool = True
    ) -> SmoothingResult:
        """
        Smooth candlestick data using advanced smoothing techniques
        
        Args:
            data: DataFrame with OHLCV data
            config: Smoothing configuration (None for default)
            smooth_indicators: Whether to smooth technical indicators
            preserve_original: Whether to preserve original columns
            
        Returns:
            SmoothingResult with smoothed data and report
        """
        start_time = datetime.now()
        
        try:
            # Use default config if none provided
            if config is None:
                config = self.default_configs['price']
            
            # Create copy for smoothing
            smoothed_df = data.copy()
            
            # Initialize report
            smoothing_report = {
                'original_shape': data.shape,
                'smoothed_shape': None,
                'smoothing_method': config.method,
                'window_size': config.window_size,
                'columns_smoothed': [],
                'indicators_smoothed': [],
                'volatility_reduction': {},
                'timestamp': datetime.now().isoformat()
            }
            
            warnings = []
            errors = []
            
            # 1. Calculate market volatility profile
            volatility_profile = self._calculate_volatility_profile(smoothed_df)
            
            # 2. Smooth price data
            smoothed_df, price_report = self._smooth_price_data(
                smoothed_df, config, volatility_profile
            )
            smoothing_report['columns_smoothed'].extend(price_report['columns_smoothed'])
            smoothing_report['volatility_reduction'].update(price_report['volatility_reduction'])
            
            # 3. Smooth volume data
            if 'volume' in smoothed_df.columns:
                smoothed_df, volume_report = self._smooth_volume_data(
                    smoothed_df, self.default_configs['volume']
                )
                smoothing_report['columns_smoothed'].extend(volume_report['columns_smoothed'])
                smoothing_report['volatility_reduction'].update(volume_report['volatility_reduction'])
            
            # 4. Smooth technical indicators if requested
            if smooth_indicators:
                smoothed_df, indicator_report = self._smooth_technical_indicators(
                    smoothed_df, self.default_configs['indicators']
                )
                smoothing_report['indicators_smoothed'] = indicator_report['indicators_smoothed']
                smoothing_report['volatility_reduction'].update(indicator_report['volatility_reduction'])
            
            # 5. Calculate quality score
            quality_score = self._calculate_smoothing_quality(
                data, smoothed_df, volatility_profile
            )
            
            # 6. Update final shape
            smoothing_report['smoothed_shape'] = smoothed_df.shape
            
            # 7. Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(True, processing_time, config.method)
            
            logger.info(f"✅ Data smoothing completed using {config.method} in {processing_time:.2f}ms")
            
            return SmoothingResult(
                smoothed_data=smoothed_df,
                smoothing_report=smoothing_report,
                quality_score=quality_score,
                volatility_profile=volatility_profile,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(False, processing_time)
            error_msg = f"Data smoothing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return SmoothingResult(
                smoothed_data=data,  # Return original data on failure
                smoothing_report={'error': error_msg},
                quality_score=0.0,
                volatility_profile={},
                warnings=[],
                errors=[error_msg]
            )
    
    def _calculate_volatility_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market volatility profile"""
        volatility_profile = {}
        
        if 'close' in df.columns and len(df) > 20:
            # Price volatility
            returns = df['close'].pct_change(fill_method=None).dropna()
            volatility_profile['price_volatility'] = returns.std()
            volatility_profile['price_volatility_annualized'] = returns.std() * np.sqrt(252)
            
            # Rolling volatility
            rolling_vol = returns.rolling(20).std()
            volatility_profile['rolling_volatility_mean'] = rolling_vol.mean()
            volatility_profile['rolling_volatility_std'] = rolling_vol.std()
            
            # Volatility regime classification
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0
            if current_vol > volatility_profile['price_volatility'] * 1.5:
                volatility_profile['regime'] = 'high_volatility'
            elif current_vol < volatility_profile['price_volatility'] * 0.5:
                volatility_profile['regime'] = 'low_volatility'
            else:
                volatility_profile['regime'] = 'normal_volatility'
        
        if 'volume' in df.columns and len(df) > 20:
            # Volume volatility
            volume_returns = df['volume'].pct_change(fill_method=None).dropna()
            volatility_profile['volume_volatility'] = volume_returns.std()
            
            # Volume spike detection
            volume_ma = df['volume'].rolling(20).mean()
            volume_spikes = (df['volume'] > volume_ma * 2).sum()
            volatility_profile['volume_spike_frequency'] = volume_spikes / len(df)
        
        return volatility_profile
    
    def _smooth_price_data(
        self, 
        df: pd.DataFrame, 
        config: SmoothingConfig, 
        volatility_profile: Dict[str, float]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Smooth price data using specified method"""
        report = {
            'columns_smoothed': [],
            'volatility_reduction': {},
            'method_used': config.method
        }
        
        price_columns = ['open', 'high', 'low', 'close']
        
        # Adjust window size based on volatility if using adaptive method
        if config.method == 'adaptive':
            config = self._adjust_config_for_volatility(config, volatility_profile)
        
        for col in price_columns:
            if col in df.columns:
                original_volatility = df[col].pct_change(fill_method=None).std()
                
                # Apply smoothing based on method
                if config.method == 'rolling_median':
                    df[f'{col}_smoothed'] = df[col].rolling(
                        window=config.window_size, center=True, min_periods=1
                    ).median()
                elif config.method == 'gaussian':
                    df[f'{col}_smoothed'] = gaussian_filter1d(
                        df[col].ffill().values, sigma=config.std_dev
                    )
                elif config.method == 'savitzky_golay':
                    df[f'{col}_smoothed'] = signal.savgol_filter(
                        df[col].values, 
                        window_length=config.window_size, 
                        polyorder=config.polynomial_order
                    )
                elif config.method == 'exponential':
                    df[f'{col}_smoothed'] = df[col].ewm(
                        span=config.window_size, adjust=False
                    ).mean()
                elif config.method == 'adaptive':
                    df[f'{col}_smoothed'] = self._adaptive_smoothing(
                        df[col], config, volatility_profile
                    )
                else:
                    # Default to rolling median
                    df[f'{col}_smoothed'] = df[col].rolling(
                        window=config.window_size, center=True, min_periods=1
                    ).median()
                
                # Calculate volatility reduction
                smoothed_volatility = df[f'{col}_smoothed'].pct_change(fill_method=None).std()
                volatility_reduction = (original_volatility - smoothed_volatility) / original_volatility
                
                report['columns_smoothed'].append(col)
                report['volatility_reduction'][col] = max(0, volatility_reduction)
        
        return df, report
    
    def _smooth_volume_data(
        self, 
        df: pd.DataFrame, 
        config: SmoothingConfig
    ) -> Tuple[pd.DataFrame, Dict]:
        """Smooth volume data"""
        report = {
            'columns_smoothed': [],
            'volatility_reduction': {},
            'method_used': config.method
        }
        
        if 'volume' in df.columns:
            original_volatility = df['volume'].pct_change(fill_method=None).std()
            
            # Apply smoothing
            if config.method == 'rolling_median':
                df['volume_smoothed'] = df['volume'].rolling(
                    window=config.window_size, center=True, min_periods=1
                ).median()
            elif config.method == 'gaussian':
                df['volume_smoothed'] = gaussian_filter1d(
                    df['volume'].values, sigma=config.std_dev
                )
            else:
                # Default to rolling median
                df['volume_smoothed'] = df['volume'].rolling(
                    window=config.window_size, center=True, min_periods=1
                ).median()
            
            # Calculate volatility reduction
            smoothed_volatility = df['volume_smoothed'].pct_change(fill_method=None).std()
            volatility_reduction = (original_volatility - smoothed_volatility) / original_volatility
            
            report['columns_smoothed'].append('volume')
            report['volatility_reduction']['volume'] = max(0, volatility_reduction)
        
        return df, report
    
    def _smooth_technical_indicators(
        self, 
        df: pd.DataFrame, 
        config: SmoothingConfig
    ) -> Tuple[pd.DataFrame, Dict]:
        """Smooth technical indicators"""
        report = {
            'indicators_smoothed': [],
            'volatility_reduction': {},
            'method_used': config.method
        }
        
        # Calculate basic indicators if not present
        if 'close' in df.columns:
            # RSI
            if 'rsi' not in df.columns:
                try:
                    df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
                except:
                    pass
            
            # MACD
            if 'macd' not in df.columns:
                try:
                    macd, macd_signal, macd_hist = talib.MACD(
                        df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    df['macd'] = macd
                    df['macd_signal'] = macd_signal
                    df['macd_histogram'] = macd_hist
                except:
                    pass
            
            # Bollinger Bands
            if 'bb_upper' not in df.columns:
                try:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(
                        df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2
                    )
                    df['bb_upper'] = bb_upper
                    df['bb_middle'] = bb_middle
                    df['bb_lower'] = bb_lower
                except:
                    pass
        
        # Smooth indicators
        indicator_columns = ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_middle', 'bb_lower']
        
        for col in indicator_columns:
            if col in df.columns and not df[col].isna().all():
                original_volatility = df[col].std()
                
                # Apply smoothing
                if config.method == 'gaussian':
                    df[f'{col}_smoothed'] = gaussian_filter1d(
                        df[col].ffill().values, sigma=config.std_dev
                    )
                elif config.method == 'rolling_median':
                    df[f'{col}_smoothed'] = df[col].rolling(
                        window=config.window_size, center=True, min_periods=1
                    ).median()
                else:
                    # Default to rolling median
                    df[f'{col}_smoothed'] = df[col].rolling(
                        window=config.window_size, center=True, min_periods=1
                    ).median()
                
                # Calculate volatility reduction
                smoothed_volatility = df[f'{col}_smoothed'].std()
                if original_volatility > 0:
                    volatility_reduction = (original_volatility - smoothed_volatility) / original_volatility
                    report['volatility_reduction'][col] = max(0, volatility_reduction)
                
                report['indicators_smoothed'].append(col)
        
        return df, report
    
    def _adjust_config_for_volatility(
        self, 
        config: SmoothingConfig, 
        volatility_profile: Dict[str, float]
    ) -> SmoothingConfig:
        """Adjust smoothing configuration based on volatility"""
        adjusted_config = SmoothingConfig(
            method=config.method,
            window_size=config.window_size,
            std_dev=config.std_dev,
            polynomial_order=config.polynomial_order,
            alpha=config.alpha,
            volatility_threshold=config.volatility_threshold,
            min_window=config.min_window,
            max_window=config.max_window
        )
        
        # Adjust window size based on volatility regime
        if 'regime' in volatility_profile:
            if volatility_profile['regime'] == 'high_volatility':
                # Use larger window for high volatility
                adjusted_config.window_size = min(
                    config.max_window, 
                    int(config.window_size * 1.5)
                )
            elif volatility_profile['regime'] == 'low_volatility':
                # Use smaller window for low volatility
                adjusted_config.window_size = max(
                    config.min_window, 
                    int(config.window_size * 0.7)
                )
        
        # Adjust smoothing parameters based on current volatility
        if 'price_volatility' in volatility_profile:
            current_vol = volatility_profile['price_volatility']
            if current_vol > config.volatility_threshold:
                # High volatility: use more aggressive smoothing
                adjusted_config.alpha = min(0.3, config.alpha * 1.5)
                adjusted_config.std_dev = min(3.0, config.std_dev * 1.2)
            else:
                # Low volatility: use gentler smoothing
                adjusted_config.alpha = max(0.05, config.alpha * 0.8)
                adjusted_config.std_dev = max(1.0, config.std_dev * 0.8)
        
        return adjusted_config
    
    def _adaptive_smoothing(
        self, 
        series: pd.Series, 
        config: SmoothingConfig, 
        volatility_profile: Dict[str, float]
    ) -> pd.Series:
        """Apply adaptive smoothing based on local volatility"""
        if len(series) < config.window_size:
            return series
        
        # Calculate local volatility
        local_vol = series.rolling(config.window_size).std()
        
        # Create adaptive weights
        weights = np.ones(len(series))
        
        # Adjust weights based on local volatility
        if 'price_volatility' in volatility_profile:
            baseline_vol = volatility_profile['price_volatility']
            for i in range(len(series)):
                if i >= config.window_size and not pd.isna(local_vol.iloc[i]):
                    local_vol_ratio = local_vol.iloc[i] / baseline_vol
                    if local_vol_ratio > 1.5:  # High local volatility
                        weights[i] = 0.3  # More smoothing
                    elif local_vol_ratio < 0.5:  # Low local volatility
                        weights[i] = 0.8  # Less smoothing
                    else:
                        weights[i] = 0.5  # Normal smoothing
        
        # Apply weighted smoothing
        smoothed = series.copy()
        for i in range(config.window_size, len(series)):
            if not pd.isna(weights[i]):
                window_data = series.iloc[i-config.window_size:i]
                smoothed.iloc[i] = (
                    weights[i] * series.iloc[i] + 
                    (1 - weights[i]) * window_data.mean()
                )
        
        return smoothed
    
    def _calculate_smoothing_quality(
        self, 
        original_df: pd.DataFrame, 
        smoothed_df: pd.DataFrame, 
        volatility_profile: Dict[str, float]
    ) -> float:
        """Calculate quality score for smoothing operation"""
        if original_df.empty or smoothed_df.empty:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        # 1. Volatility reduction score (40 points)
        if 'close' in original_df.columns and 'close_smoothed' in smoothed_df.columns:
            original_vol = original_df['close'].pct_change(fill_method=None).std()
            smoothed_vol = smoothed_df['close_smoothed'].pct_change(fill_method=None).std()
            
            if original_vol > 0:
                vol_reduction = (original_vol - smoothed_vol) / original_vol
                vol_score = min(40, max(0, vol_reduction * 40))
                total_score += vol_score
            max_score += 40
        
        # 2. Data preservation score (30 points)
        if 'close' in original_df.columns and 'close_smoothed' in smoothed_df.columns:
            # Check correlation between original and smoothed
            correlation = original_df['close'].corr(smoothed_df['close_smoothed'])
            corr_score = max(0, correlation * 30)
            total_score += corr_score
            max_score += 30
        
        # 3. Smoothness score (30 points)
        if 'close_smoothed' in smoothed_df.columns:
            # Calculate smoothness (lower second derivative = smoother)
            second_deriv = smoothed_df['close_smoothed'].diff().diff().abs().mean()
            if second_deriv > 0:
                smoothness_score = max(0, 30 - (second_deriv * 100))  # Scale factor
                total_score += smoothness_score
            max_score += 30
        
        # Calculate overall quality score
        quality_score = total_score / max_score if max_score > 0 else 0.0
        
        return max(0.0, min(1.0, quality_score))
    
    def _update_stats(self, success: bool, processing_time: float, method: str = "unknown"):
        """Update performance statistics"""
        self.stats['total_smoothing_operations'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        
        if success:
            self.stats['successful_smoothing'] += 1
        else:
            self.stats['failed_smoothing'] += 1
        
        # Track method usage
        if method not in self.stats['methods_used']:
            self.stats['methods_used'][method] = 0
        self.stats['methods_used'][method] += 1
        
        # Update average processing time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_smoothing_operations']
        )
    
    def get_smoothing_stats(self) -> Dict:
        """Get smoothing service statistics"""
        return self.stats.copy()

# Example usage and testing
def test_smoothing_service():
    """Test the data smoothing service"""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Add some noise to simulate real market data
    data['close'] = data['close'] + np.random.normal(0, 2, 100)
    data['volume'] = data['volume'] + np.random.normal(0, 500, 100)
    
    # Initialize service
    service = DataSmoothingService()
    
    # Test different smoothing methods
    methods = ['rolling_median', 'gaussian', 'savitzky_golay', 'exponential', 'adaptive']
    
    for method in methods:
        print(f"\n=== Testing {method.upper()} Smoothing ===")
        
        config = SmoothingConfig(method=method, window_size=20)
        
        result = service.smooth_candlestick_data(
            data=data,
            config=config,
            smooth_indicators=True,
            preserve_original=True
        )
        
        print(f"Quality Score: {result.quality_score:.2%}")
        print(f"Original Shape: {data.shape}")
        print(f"Smoothed Shape: {result.smoothed_data.shape}")
        print(f"Columns Smoothed: {len(result.smoothing_report['columns_smoothed'])}")
        print(f"Indicators Smoothed: {len(result.smoothing_report['indicators_smoothed'])}")
        
        if result.errors:
            print(f"Errors: {result.errors}")
    
    # Print statistics
    stats = service.get_smoothing_stats()
    print(f"\nService Statistics:")
    print(f"Total Operations: {stats['total_smoothing_operations']}")
    print(f"Success Rate: {stats['successful_smoothing']/stats['total_smoothing_operations']:.1%}")
    print(f"Methods Used: {stats['methods_used']}")
    print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.2f}ms")
    
    return True

if __name__ == "__main__":
    # Run test if script is executed directly
    test_smoothing_service()
