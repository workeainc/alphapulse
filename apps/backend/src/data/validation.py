#!/usr/bin/env python3
"""
Data Validation Module for AlphaPulse
Ensures candlestick data quality and consistency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CandlestickValidator:
    """Validates candlestick data quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'price_consistency': self._validate_price_consistency,
            'volume_consistency': self._validate_volume_consistency,
            'time_sequence': self._validate_time_sequence,
            'outlier_detection': self._detect_outliers,
            'data_completeness': self._validate_data_completeness
        }
        
        # Validation thresholds
        self.thresholds = {
            'max_price_change_pct': 50.0,  # Maximum 50% price change in one candle
            'min_volume': 0.0,  # Minimum volume threshold
            'max_time_gap_hours': 24,  # Maximum gap between candles
            'outlier_iqr_multiplier': 3.0,  # IQR multiplier for outlier detection
            'min_candles_for_analysis': 20  # Minimum candles needed for analysis
        }
    
    def validate_candlestick_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate candlestick data quality
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (cleaned_dataframe, validation_report)
        """
        original_count = len(df)
        validation_report = {
            'original_count': original_count,
            'validation_errors': [],
            'cleaned_count': 0,
            'quality_score': 0.0,
            'validation_timestamp': datetime.now(),
            'data_range': {
                'start': None,
                'end': None
            }
        }
        
        if df.empty:
            validation_report['validation_errors'].append({
                'rule': 'data_completeness',
                'issue': 'empty_dataframe',
                'count': 0
            })
            return df, validation_report
        
        # Set data range
        if not df.empty:
            validation_report['data_range']['start'] = df.index.min()
            validation_report['data_range']['end'] = df.index.max()
        
        # Apply validation rules
        cleaned_df = df.copy()
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                cleaned_df, errors = rule_func(cleaned_df)
                validation_report['validation_errors'].extend(errors)
            except Exception as e:
                logger.error(f"Error applying validation rule {rule_name}: {e}")
                validation_report['validation_errors'].append({
                    'rule': rule_name,
                    'error': str(e)
                })
        
        # Calculate quality score
        validation_report['cleaned_count'] = len(cleaned_df)
        validation_report['quality_score'] = len(cleaned_df) / original_count if original_count > 0 else 0
        
        # Add summary statistics
        if not cleaned_df.empty:
            validation_report['summary_stats'] = self._calculate_summary_stats(cleaned_df)
        
        return cleaned_df, validation_report
    
    def _validate_price_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Validate price consistency (high >= low, etc.)"""
        errors = []
        original_count = len(df)
        
        # Check high >= low
        invalid_high_low = df[df['high'] < df['low']]
        if not invalid_high_low.empty:
            errors.append({
                'rule': 'price_consistency',
                'issue': 'high_price_below_low',
                'count': len(invalid_high_low),
                'details': f"Found {len(invalid_high_low)} candles where high < low"
            })
            # Fix by swapping high and low
            df.loc[invalid_high_low.index, ['high', 'low']] = \
                df.loc[invalid_high_low.index, ['low', 'high']].values
        
        # Check open and close within high-low range
        invalid_open = df[
            (df['open'] > df['high']) | (df['open'] < df['low'])
        ]
        if not invalid_open.empty:
            errors.append({
                'rule': 'price_consistency',
                'issue': 'open_price_outside_range',
                'count': len(invalid_open),
                'details': f"Found {len(invalid_open)} candles where open price outside high-low range"
            })
            # Fix by clamping open price
            df.loc[invalid_open.index, 'open'] = df.loc[invalid_open.index, 'open'].clip(
                lower=df.loc[invalid_open.index, 'low'],
                upper=df.loc[invalid_open.index, 'high']
            )
        
        invalid_close = df[
            (df['close'] > df['high']) | (df['close'] < df['low'])
        ]
        if not invalid_close.empty:
            errors.append({
                'rule': 'price_consistency',
                'issue': 'close_price_outside_range',
                'count': len(invalid_close),
                'details': f"Found {len(invalid_close)} candles where close price outside high-low range"
            })
            # Fix by clamping close price
            df.loc[invalid_close.index, 'close'] = df.loc[invalid_close.index, 'close'].clip(
                lower=df.loc[invalid_close.index, 'low'],
                upper=df.loc[invalid_close.index, 'high']
            )
        
        # Check for extreme price changes
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs() * 100
            extreme_changes = price_changes[price_changes > self.thresholds['max_price_change_pct']]
            
            if not extreme_changes.empty:
                errors.append({
                    'rule': 'price_consistency',
                    'issue': 'extreme_price_changes',
                    'count': len(extreme_changes),
                    'details': f"Found {len(extreme_changes)} candles with >{self.thresholds['max_price_change_pct']}% price change"
                })
                # Mark extreme changes for review
                df.loc[extreme_changes.index, 'is_extreme_change'] = True
        
        logger.info(f"Price consistency validation: {len(errors)} issues found, {len(df)} candles remaining")
        return df, errors
    
    def _validate_volume_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Validate volume consistency"""
        errors = []
        
        # Check for negative volumes
        negative_volume = df[df['volume'] < 0]
        if not negative_volume.empty:
            errors.append({
                'rule': 'volume_consistency',
                'issue': 'negative_volume',
                'count': len(negative_volume),
                'details': f"Found {len(negative_volume)} candles with negative volume"
            })
            # Fix by taking absolute value
            df.loc[negative_volume.index, 'volume'] = \
                df.loc[negative_volume.index, 'volume'].abs()
        
        # Check for zero volumes (might indicate data issues)
        zero_volume = df[df['volume'] == 0]
        if not zero_volume.empty:
            errors.append({
                'rule': 'volume_consistency',
                'issue': 'zero_volume',
                'count': len(zero_volume),
                'details': f"Found {len(zero_volume)} candles with zero volume"
            })
            # Mark for review
            df.loc[zero_volume.index, 'is_zero_volume'] = True
        
        # Check for extremely high volumes (outliers)
        if len(df) > 1:
            volume_ratio = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
            extreme_volume = volume_ratio[volume_ratio > 10]  # 10x average volume
            
            if not extreme_volume.empty:
                errors.append({
                    'rule': 'volume_consistency',
                    'issue': 'extreme_volume',
                    'count': len(extreme_volume),
                    'details': f"Found {len(extreme_volume)} candles with >10x average volume"
                })
                # Mark for review
                df.loc[extreme_volume.index, 'is_extreme_volume'] = True
        
        logger.info(f"Volume consistency validation: {len(errors)} issues found")
        return df, errors
    
    def _validate_time_sequence(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Validate time sequence consistency"""
        errors = []
        
        # Sort by time
        df_sorted = df.sort_index()
        
        # Check for duplicate timestamps
        duplicate_times = df_sorted.index.duplicated()
        if duplicate_times.any():
            errors.append({
                'rule': 'time_sequence',
                'issue': 'duplicate_timestamps',
                'count': duplicate_times.sum(),
                'details': f"Found {duplicate_times.sum()} duplicate timestamps"
            })
            # Remove duplicates, keeping last occurrence
            df_sorted = df_sorted[~duplicate_times]
        
        # Check for gaps in time sequence
        if len(df_sorted) > 1:
            time_diffs = df_sorted.index.to_series().diff()
            expected_interval = self._get_expected_interval(df_sorted)
            
            if expected_interval:
                large_gaps = time_diffs[time_diffs > expected_interval * 2]
                if not large_gaps.empty:
                    errors.append({
                        'rule': 'time_sequence',
                        'issue': 'large_time_gaps',
                        'count': len(large_gaps),
                        'details': f"Found {len(large_gaps)} large time gaps >{expected_interval * 2}"
                    })
                    
                    # Mark gaps for review
                    gap_indices = large_gaps.index
                    df_sorted.loc[gap_indices, 'has_large_gap'] = True
        
        # Check for future timestamps
        future_timestamps = df_sorted[df_sorted.index > datetime.now()]
        if not future_timestamps.empty:
            errors.append({
                'rule': 'time_sequence',
                'issue': 'future_timestamps',
                'count': len(future_timestamps),
                'details': f"Found {len(future_timestamps)} timestamps in the future"
            })
            # Remove future timestamps
            df_sorted = df_sorted[df_sorted.index <= datetime.now()]
        
        logger.info(f"Time sequence validation: {len(errors)} issues found")
        return df_sorted, errors
    
    def _detect_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Detect and handle price outliers"""
        errors = []
        
        if len(df) < 10:  # Need enough data for outlier detection
            return df, errors
        
        # Calculate price changes
        price_changes = df['close'].pct_change().abs()
        
        # Detect outliers using IQR method
        Q1 = price_changes.quantile(0.25)
        Q3 = price_changes.quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold = Q3 + self.thresholds['outlier_iqr_multiplier'] * IQR
        
        outliers = price_changes[price_changes > outlier_threshold]
        
        if not outliers.empty:
            errors.append({
                'rule': 'outlier_detection',
                'issue': 'price_outliers',
                'count': len(outliers),
                'details': f"Found {len(outliers)} price outliers using IQR method"
            })
            
            # Mark outliers for review (don't remove automatically)
            df.loc[outliers.index, 'is_outlier'] = True
        
        # Detect volume outliers
        if 'volume' in df.columns and len(df) > 10:
            volume_log = np.log1p(df['volume'])  # Log transform for better distribution
            Q1_vol = volume_log.quantile(0.25)
            Q3_vol = volume_log.quantile(0.75)
            IQR_vol = Q3_vol - Q1_vol
            outlier_threshold_vol = Q3_vol + self.thresholds['outlier_iqr_multiplier'] * IQR_vol
            
            volume_outliers = volume_log[volume_log > outlier_threshold_vol]
            
            if not volume_outliers.empty:
                errors.append({
                    'rule': 'outlier_detection',
                    'issue': 'volume_outliers',
                    'count': len(volume_outliers),
                    'details': f"Found {len(volume_outliers)} volume outliers using IQR method"
                })
                
                # Mark volume outliers for review
                df.loc[volume_outliers.index, 'is_volume_outlier'] = True
        
        logger.info(f"Outlier detection: {len(errors)} issues found")
        return df, errors
    
    def _validate_data_completeness(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
        """Validate data completeness and required columns"""
        errors = []
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append({
                'rule': 'data_completeness',
                'issue': 'missing_required_columns',
                'count': len(missing_columns),
                'details': f"Missing required columns: {missing_columns}"
            })
            return df, errors
        
        # Check for NaN values in required columns
        for col in required_columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                errors.append({
                    'rule': 'data_completeness',
                    'issue': f'nan_values_in_{col}',
                    'count': nan_count,
                    'details': f"Found {nan_count} NaN values in {col} column"
                })
                
                # Remove rows with NaN values in required columns
                df = df.dropna(subset=[col])
        
        # Check minimum data requirements
        if len(df) < self.thresholds['min_candles_for_analysis']:
            errors.append({
                'rule': 'data_completeness',
                'issue': 'insufficient_data',
                'count': len(df),
                'details': f"Only {len(df)} candles available, need at least {self.thresholds['min_candles_for_analysis']}"
            })
        
        logger.info(f"Data completeness validation: {len(errors)} issues found")
        return df, errors
    
    def _get_expected_interval(self, df: pd.DataFrame) -> Optional[timedelta]:
        """Get expected time interval from data"""
        if len(df) < 2:
            return None
        
        time_diffs = df.index.to_series().diff().dropna()
        if time_diffs.empty:
            return None
        
        # Get most common interval
        most_common_interval = time_diffs.mode().iloc[0]
        return most_common_interval
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for validated data"""
        stats = {
            'total_candles': len(df),
            'date_range': {
                'start': df.index.min().isoformat() if not df.empty else None,
                'end': df.index.max().isoformat() if not df.empty else None
            },
            'price_stats': {},
            'volume_stats': {}
        }
        
        if not df.empty:
            # Price statistics
            stats['price_stats'] = {
                'min_price': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max_price': float(df[['open', 'high', 'low', 'close']].max().max()),
                'avg_close': float(df['close'].mean()),
                'price_volatility': float(df['close'].pct_change().std())
            }
            
            # Volume statistics
            if 'volume' in df.columns:
                stats['volume_stats'] = {
                    'min_volume': float(df['volume'].min()),
                    'max_volume': float(df['volume'].max()),
                    'avg_volume': float(df['volume'].mean()),
                    'total_volume': float(df['volume'].sum())
                }
        
        return stats
    
    def get_validation_summary(self, validation_report: Dict) -> str:
        """Get human-readable validation summary"""
        summary = f"Data Validation Summary\n"
        summary += f"{'='*50}\n"
        summary += f"Original count: {validation_report['original_count']}\n"
        summary += f"Cleaned count: {validation_report['cleaned_count']}\n"
        summary += f"Quality score: {validation_report['quality_score']:.2%}\n"
        summary += f"Validation timestamp: {validation_report['validation_timestamp']}\n"
        
        if 'data_range' in validation_report and validation_report['data_range']['start']:
            summary += f"Data range: {validation_report['data_range']['start']} to {validation_report['data_range']['end']}\n"
        
        if validation_report['validation_errors']:
            summary += f"\nValidation Issues Found:\n"
            for error in validation_report['validation_errors']:
                summary += f"- {error['rule']}: {error['issue']} ({error['count']} instances)\n"
                if 'details' in error:
                    summary += f"  Details: {error['details']}\n"
        
        if 'summary_stats' in validation_report:
            stats = validation_report['summary_stats']
            summary += f"\nData Statistics:\n"
            summary += f"- Total candles: {stats['total_candles']}\n"
            if stats['price_stats']:
                summary += f"- Price range: ${stats['price_stats']['min_price']:.2f} - ${stats['price_stats']['max_price']:.2f}\n"
                summary += f"- Average close: ${stats['price_stats']['avg_close']:.2f}\n"
                summary += f"- Price volatility: {stats['price_stats']['price_volatility']:.2%}\n"
        
        return summary

# Example usage and testing
def test_validation():
    """Test the validation functionality"""
    # Create sample data with some issues
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    data = {
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Introduce some data quality issues
    df.loc[df.index[10], 'high'] = 50  # High < Low
    df.loc[df.index[20], 'volume'] = -1000  # Negative volume
    df.loc[df.index[30], 'close'] = 300  # Close > High
    
    # Validate the data
    validator = CandlestickValidator()
    cleaned_df, report = validator.validate_candlestick_data(df)
    
    # Print results
    print(validator.get_validation_summary(report))
    
    return cleaned_df, report

if __name__ == "__main__":
    # Run test if script is executed directly
    cleaned_df, report = test_validation()
