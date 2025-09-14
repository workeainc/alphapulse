#!/usr/bin/env python3
"""
Data Normalization Service for AlphaPulse
Phase 1: Complete Data Normalization
Handles decimal precision, cross-pair normalization, and scale consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from decimal import Decimal, ROUND_HALF_UP
import json

logger = logging.getLogger(__name__)

@dataclass
class TokenConfig:
    """Configuration for token-specific normalization"""
    symbol: str
    base_currency: str
    quote_currency: str
    price_precision: int
    volume_precision: int
    min_price: float
    max_price: float
    min_volume: float
    max_volume: float
    price_scale_factor: float = 1.0
    volume_scale_factor: float = 1.0
    exchange: str = "unknown"

@dataclass
class NormalizationResult:
    """Result of data normalization process"""
    normalized_data: pd.DataFrame
    normalization_report: Dict[str, Any]
    quality_score: float
    warnings: List[str]
    errors: List[str]

class DataNormalizationService:
    """
    Comprehensive data normalization service for AlphaPulse
    Handles decimal precision, cross-pair normalization, and scale consistency
    """
    
    def __init__(self):
        # Default token configurations for major pairs
        self.default_token_configs = {
            "BTCUSDT": TokenConfig(
                symbol="BTCUSDT",
                base_currency="BTC",
                quote_currency="USDT",
                price_precision=2,
                volume_precision=6,
                min_price=0.01,
                max_price=1000000.0,
                min_volume=0.000001,
                max_volume=1000000.0,
                price_scale_factor=1.0,
                volume_scale_factor=1.0
            ),
            "ETHUSDT": TokenConfig(
                symbol="ETHUSDT",
                base_currency="ETH",
                quote_currency="USDT",
                price_precision=2,
                volume_precision=5,
                min_price=0.01,
                max_price=100000.0,
                min_volume=0.00001,
                max_volume=1000000.0,
                price_scale_factor=1.0,
                volume_scale_factor=1.0
            ),
            "ADAUSDT": TokenConfig(
                symbol="ADAUSDT",
                base_currency="ADA",
                quote_currency="USDT",
                price_precision=4,
                volume_precision=0,
                min_price=0.0001,
                max_price=100.0,
                min_volume=1.0,
                max_volume=1000000000.0,
                price_scale_factor=1.0,
                volume_scale_factor=1.0
            ),
            "DOGEUSDT": TokenConfig(
                symbol="DOGEUSDT",
                base_currency="DOGE",
                quote_currency="USDT",
                price_precision=6,
                volume_precision=0,
                min_price=0.000001,
                max_price=10.0,
                min_volume=1.0,
                max_volume=10000000000.0,
                price_scale_factor=1.0,
                volume_scale_factor=1.0
            )
        }
        
        # Exchange-specific configurations
        self.exchange_configs = {
            "binance": {
                "price_precision_override": {},
                "volume_precision_override": {},
                "scale_factors": {}
            },
            "coinbase": {
                "price_precision_override": {},
                "volume_precision_override": {},
                "scale_factors": {}
            },
            "kraken": {
                "price_precision_override": {},
                "volume_precision_override": {},
                "scale_factors": {}
            }
        }
        
        # Performance tracking
        self.stats = {
            'total_normalizations': 0,
            'successful_normalizations': 0,
            'failed_normalizations': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0
        }
        
        logger.info("🚀 Data Normalization Service initialized")
    
    def normalize_candlestick_data(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        exchange: str = "unknown",
        target_precision: Optional[int] = None,
        normalize_volume: bool = True,
        validate_scale: bool = True
    ) -> NormalizationResult:
        """
        Normalize candlestick data with comprehensive precision and scale handling
        
        Args:
            data: Raw OHLCV DataFrame
            symbol: Trading pair symbol
            exchange: Exchange name
            target_precision: Target decimal precision (None for auto)
            normalize_volume: Whether to normalize volume
            validate_scale: Whether to validate price/volume scales
            
        Returns:
            NormalizationResult with normalized data and report
        """
        start_time = datetime.now()
        
        try:
            # Get token configuration
            token_config = self._get_token_config(symbol, exchange)
            
            # Create copy for normalization
            normalized_df = data.copy()
            
            # Initialize report
            normalization_report = {
                'symbol': symbol,
                'exchange': exchange,
                'original_shape': data.shape,
                'normalized_shape': None,
                'precision_changes': {},
                'scale_adjustments': {},
                'volume_normalizations': {},
                'quality_checks': {},
                'timestamp': datetime.now().isoformat()
            }
            
            warnings = []
            errors = []
            
            # 1. Price Normalization
            normalized_df, price_report = self._normalize_prices(
                normalized_df, token_config, target_precision
            )
            normalization_report['precision_changes'].update(price_report)
            
            # 2. Volume Normalization
            if normalize_volume:
                normalized_df, volume_report = self._normalize_volumes(
                    normalized_df, token_config
                )
                normalization_report['volume_normalizations'].update(volume_report)
            
            # 3. Scale Validation and Adjustment
            if validate_scale:
                normalized_df, scale_report = self._validate_and_adjust_scales(
                    normalized_df, token_config
                )
                normalization_report['scale_adjustments'].update(scale_report)
            
            # 4. Quality Checks
            quality_score, quality_report = self._perform_quality_checks(
                normalized_df, token_config
            )
            normalization_report['quality_checks'] = quality_report
            
            # Update final shape
            normalization_report['normalized_shape'] = normalized_df.shape
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(True, processing_time)
            
            logger.info(f"✅ Data normalization completed for {symbol} in {processing_time:.2f}ms")
            
            return NormalizationResult(
                normalized_data=normalized_df,
                normalization_report=normalization_report,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(False, processing_time)
            error_msg = f"Data normalization failed for {symbol}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return NormalizationResult(
                normalized_data=data,  # Return original data on failure
                normalization_report={'error': error_msg},
                quality_score=0.0,
                warnings=[],
                errors=[error_msg]
            )
    
    def _get_token_config(self, symbol: str, exchange: str) -> TokenConfig:
        """Get token configuration, creating default if not exists"""
        if symbol in self.default_token_configs:
            config = self.default_token_configs[symbol]
            config.exchange = exchange
            return config
        
        # Create default config for unknown symbols
        logger.warning(f"⚠️ No configuration found for {symbol}, using defaults")
        return TokenConfig(
            symbol=symbol,
            base_currency=symbol.split('USDT')[0] if 'USDT' in symbol else 'UNKNOWN',
            quote_currency=symbol.split('USDT')[1] if 'USDT' in symbol else 'USDT',
            price_precision=4,
            volume_precision=2,
            min_price=0.0001,
            max_price=1000.0,
            min_volume=0.01,
            max_volume=1000000.0,
            price_scale_factor=1.0,
            volume_scale_factor=1.0,
            exchange=exchange
        )
    
    def _normalize_prices(
        self, 
        df: pd.DataFrame, 
        token_config: TokenConfig, 
        target_precision: Optional[int]
    ) -> Tuple[pd.DataFrame, Dict]:
        """Normalize price columns with proper decimal precision"""
        report = {
            'price_precision_applied': token_config.price_precision,
            'price_columns_processed': [],
            'precision_rounding_applied': False
        }
        
        price_columns = ['open', 'high', 'low', 'close']
        
        # Remove rows with NaN values in price columns
        original_count = len(df)
        df = df.dropna(subset=price_columns)
        cleaned_count = len(df)
        
        if original_count != cleaned_count:
            report['rows_cleaned'] = original_count - cleaned_count
        
        for col in price_columns:
            if col in df.columns:
                # Apply precision rounding
                if target_precision is not None:
                    precision = target_precision
                else:
                    precision = token_config.price_precision
                
                # Round to specified precision
                df[col] = df[col].round(precision)
                
                # Ensure values are within bounds
                df[col] = df[col].clip(
                    lower=token_config.min_price,
                    upper=token_config.max_price
                )
                
                report['price_columns_processed'].append(col)
                report['precision_rounding_applied'] = True
        
        return df, report
    
    def _normalize_volumes(
        self, 
        df: pd.DataFrame, 
        token_config: TokenConfig
    ) -> Tuple[pd.DataFrame, Dict]:
        """Normalize volume with proper precision and scale"""
        report = {
            'volume_precision_applied': token_config.volume_precision,
            'volume_scale_factor': token_config.volume_scale_factor,
            'volume_normalization_applied': False
        }
        
        if 'volume' in df.columns:
            # Apply volume precision
            df['volume'] = df['volume'].round(token_config.volume_precision)
            
            # Ensure volume is positive
            df['volume'] = df['volume'].clip(lower=token_config.min_volume)
            
            # Apply scale factor if needed
            if token_config.volume_scale_factor != 1.0:
                df['volume'] = df['volume'] * token_config.volume_scale_factor
                report['volume_normalization_applied'] = True
            
            # Add normalized volume features
            df['volume_normalized'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20, min_periods=1).mean()) / df['volume'].rolling(20, min_periods=1).std()
            
            report['volume_normalization_applied'] = True
        
        return df, report
    
    def _validate_and_adjust_scales(
        self, 
        df: pd.DataFrame, 
        token_config: TokenConfig
    ) -> Tuple[pd.DataFrame, Dict]:
        """Validate and adjust price/volume scales if needed"""
        report = {
            'scale_validation_passed': True,
            'scale_adjustments_made': False,
            'price_scale_issues': [],
            'volume_scale_issues': []
        }
        
        # Check price scale consistency
        if 'close' in df.columns:
            price_range = df['close'].max() - df['close'].min()
            expected_range = token_config.max_price - token_config.min_price
            
            # More sensitive scale detection - detect if price range is significantly different from expected
            if price_range > expected_range * 5 or price_range < expected_range * 0.01:
                scale_issue_msg = f"Price range {price_range:.8f} significantly different from expected {expected_range:.2f}"
                report['price_scale_issues'].append(scale_issue_msg)
                report['scale_validation_passed'] = False
                
                # Apply scale adjustment if scale factor is available
                if token_config.price_scale_factor != 1.0:
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = df[col] * token_config.price_scale_factor
                    report['scale_adjustments_made'] = True
        
        # Check volume scale consistency
        if 'volume' in df.columns:
            volume_range = df['volume'].max() - df['volume'].min()
            expected_volume_range = token_config.max_volume - token_config.min_volume
            
            # More sensitive volume scale detection
            if volume_range > expected_volume_range * 50 or volume_range < expected_volume_range * 0.01:
                scale_issue_msg = f"Volume range {volume_range:.2f} significantly different from expected {expected_volume_range:.2f}"
                report['volume_scale_issues'].append(scale_issue_msg)
                report['scale_validation_passed'] = False
                
                # Apply volume scale adjustment if scale factor is available
                if token_config.volume_scale_factor != 1.0:
                    df['volume'] = df['volume'] * token_config.volume_scale_factor
                    report['scale_adjustments_made'] = True
        
        return df, report
    
    def _perform_quality_checks(
        self, 
        df: pd.DataFrame, 
        token_config: TokenConfig
    ) -> Tuple[float, Dict]:
        """Perform comprehensive quality checks on normalized data"""
        quality_report = {
            'data_integrity': {},
            'precision_accuracy': {},
            'scale_consistency': {},
            'overall_score': 0.0
        }
        
        total_score = 0.0
        max_score = 0.0
        
        # 1. Data Integrity Check (30 points)
        integrity_score = 0.0
        if not df.empty:
            integrity_score += 10  # Data exists
            if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                integrity_score += 10  # Required columns exist
            if df.isnull().sum().sum() == 0:
                integrity_score += 10  # No NaN values
        
        quality_report['data_integrity'] = {
            'score': integrity_score,
            'max_score': 30,
            'details': f"Data integrity check: {integrity_score}/30"
        }
        total_score += integrity_score
        max_score += 30
        
        # 2. Precision Accuracy Check (40 points)
        precision_score = 0.0
        if 'close' in df.columns:
            # Check if prices are within expected precision
            price_precision_check = df['close'].apply(
                lambda x: len(str(x).split('.')[-1]) <= token_config.price_precision
            ).mean() * 20
            
            # Check if prices are within bounds
            price_bounds_check = df['close'].between(
                token_config.min_price, 
                token_config.max_price
            ).mean() * 20
            
            precision_score = price_precision_check + price_bounds_check
        
        quality_report['precision_accuracy'] = {
            'score': precision_score,
            'max_score': 40,
            'details': f"Precision accuracy check: {precision_score:.1f}/40"
        }
        total_score += precision_score
        max_score += 40
        
        # 3. Scale Consistency Check (30 points)
        scale_score = 0.0
        if 'close' in df.columns and 'volume' in df.columns:
            # Check price scale consistency
            price_volatility = df['close'].pct_change().std()
            if price_volatility < 0.5:  # Reasonable volatility
                scale_score += 15
            
            # Check volume scale consistency
            volume_volatility = df['volume'].pct_change().std()
            if volume_volatility < 2.0:  # Reasonable volume volatility
                scale_score += 15
        
        quality_report['scale_consistency'] = {
            'score': scale_score,
            'max_score': 30,
            'details': f"Scale consistency check: {scale_score:.1f}/30"
        }
        total_score += scale_score
        max_score += 30
        
        # Calculate overall quality score
        overall_score = total_score / max_score if max_score > 0 else 0.0
        quality_report['overall_score'] = overall_score
        
        return overall_score, quality_report
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update performance statistics"""
        self.stats['total_normalizations'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        
        if success:
            self.stats['successful_normalizations'] += 1
        else:
            self.stats['failed_normalizations'] += 1
        
        # Update average processing time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_normalizations']
        )
    
    def get_normalization_stats(self) -> Dict:
        """Get normalization service statistics"""
        return self.stats.copy()
    
    def add_token_config(self, config: TokenConfig):
        """Add or update token configuration"""
        self.default_token_configs[config.symbol] = config
        logger.info(f"✅ Added token configuration for {config.symbol}")
    
    def update_exchange_config(self, exchange: str, config: Dict):
        """Update exchange-specific configuration"""
        if exchange in self.exchange_configs:
            self.exchange_configs[exchange].update(config)
            logger.info(f"✅ Updated exchange configuration for {exchange}")
        else:
            self.exchange_configs[exchange] = config
            logger.info(f"✅ Added exchange configuration for {exchange}")

# Example usage and testing
def test_normalization_service():
    """Test the data normalization service"""
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Initialize service
    service = DataNormalizationService()
    
    # Test normalization
    result = service.normalize_candlestick_data(
        data=data,
        symbol="BTCUSDT",
        exchange="binance",
        normalize_volume=True,
        validate_scale=True
    )
    
    # Print results
    print("=== Data Normalization Test Results ===")
    print(f"Quality Score: {result.quality_score:.2%}")
    print(f"Original Shape: {data.shape}")
    print(f"Normalized Shape: {result.normalized_data.shape}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.errors:
        print("Errors:", result.errors)
    
    # Print statistics
    stats = service.get_normalization_stats()
    print(f"\nService Statistics:")
    print(f"Total Normalizations: {stats['total_normalizations']}")
    print(f"Success Rate: {stats['successful_normalizations']/stats['total_normalizations']:.1%}")
    print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.2f}ms")
    
    return result

if __name__ == "__main__":
    # Run test if script is executed directly
    result = test_normalization_service()
