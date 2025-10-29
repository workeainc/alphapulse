#!/usr/bin/env python3
"""
Timeframe Alignment Service for AlphaPulse
Phase 4: Timeframe Alignment
Handles multi-timeframe synchronization, incomplete candle handling, and timeframe validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class TimeframeUnit(Enum):
    """Timeframe units"""
    MINUTE = "m"
    HOUR = "h"
    DAY = "d"
    WEEK = "w"
    MONTH = "M"

@dataclass
class TimeframeConfig:
    """Configuration for timeframe alignment"""
    primary_timeframe: str = "1h"
    secondary_timeframes: List[str] = None
    sync_tolerance_minutes: int = 5
    handle_incomplete_candles: bool = True
    validate_data_integrity: bool = True
    min_data_points: int = 20
    max_gap_minutes: int = 60

@dataclass
class TimeframeInfo:
    """Information about a specific timeframe"""
    timeframe: str
    unit: TimeframeUnit
    value: int
    minutes: int
    is_higher: bool
    data_range: Tuple[datetime, datetime]
    data_points: int
    completeness: float

@dataclass
class AlignmentResult:
    """Result of timeframe alignment process"""
    aligned_data: Dict[str, pd.DataFrame]
    alignment_report: Dict[str, Any]
    timeframe_info: List[TimeframeInfo]
    quality_score: float
    warnings: List[str]
    errors: List[str]
    
    @property
    def success(self) -> bool:
        """Check if alignment was successful"""
        return len(self.errors) == 0

class TimeframeAlignmentService:
    """
    Comprehensive timeframe alignment service for AlphaPulse
    Handles multi-timeframe synchronization, incomplete candle handling, and timeframe validation
    """
    
    def __init__(self):
        # Supported timeframes and their minute equivalents
        self.timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '6h': 360, '12h': 720,
            '1d': 1440, '1w': 10080, '1M': 43200
        }
        
        # Performance tracking
        self.stats = {
            'total_alignments': 0,
            'successful_alignments': 0,
            'failed_alignments': 0,
            'timeframes_processed': 0,
            'avg_processing_time_ms': 0.0,
            'total_processing_time_ms': 0.0
        }
        
        logger.info("🚀 Timeframe Alignment Service initialized")
    
    def align_timeframes(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        config: Optional[TimeframeConfig] = None
    ) -> AlignmentResult:
        """
        Align multiple timeframes for synchronized analysis
        
        Args:
            data_dict: Dictionary of {timeframe: DataFrame} pairs
            config: Timeframe alignment configuration
            
        Returns:
            AlignmentResult with aligned data and report
        """
        start_time = datetime.now()
        
        try:
            # Use default config if none provided
            if config is None:
                config = TimeframeConfig()
            
            # Initialize report
            alignment_report = {
                'timeframes_processed': len(data_dict),
                'primary_timeframe': config.primary_timeframe,
                'secondary_timeframes': config.secondary_timeframes or [],
                'sync_tolerance_minutes': config.sync_tolerance_minutes,
                'alignment_methods': [],
                'data_quality_checks': {},
                'timestamp': datetime.now().isoformat()
            }
            
            warnings = []
            errors = []
            
            # 1. Validate and parse timeframes
            timeframe_info_list = self._validate_timeframes(data_dict, config)
            if not timeframe_info_list:
                raise ValueError("No valid timeframes found")
            
            # 2. Sort timeframes by resolution (highest to lowest)
            timeframe_info_list.sort(key=lambda x: x.minutes, reverse=True)
            
            # 3. Align timeframes
            aligned_data = self._align_timeframes_internal(
                data_dict, timeframe_info_list, config
            )
            
            # 4. Handle incomplete candles if requested
            if config.handle_incomplete_candles:
                aligned_data = self._handle_incomplete_candles(
                    aligned_data, timeframe_info_list, config
                )
            
            # 5. Validate data integrity if requested
            if config.validate_data_integrity:
                data_quality = self._validate_data_integrity(
                    aligned_data, timeframe_info_list, config
                )
                alignment_report['data_quality_checks'] = data_quality
            
            # 6. Calculate quality score
            quality_score = self._calculate_alignment_quality(
                aligned_data, timeframe_info_list, config
            )
            
            # 7. Update report
            alignment_report['timeframes_processed'] = len(timeframe_info_list)
            alignment_report['aligned_shape'] = {tf: df.shape for tf, df in aligned_data.items()}
            
            # 8. Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(True, processing_time, len(timeframe_info_list))
            
            logger.info(f"✅ Timeframe alignment completed: {len(timeframe_info_list)} timeframes in {processing_time:.2f}ms")
            
            return AlignmentResult(
                aligned_data=aligned_data,
                alignment_report=alignment_report,
                timeframe_info=timeframe_info_list,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(False, processing_time)
            error_msg = f"Timeframe alignment failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return AlignmentResult(
                aligned_data=data_dict,  # Return original data on failure
                alignment_report={'error': error_msg},
                timeframe_info=[],
                quality_score=0.0,
                warnings=[],
                errors=[error_msg]
            )
    
    def _validate_timeframes(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        config: TimeframeConfig
    ) -> List[TimeframeInfo]:
        """Validate and parse timeframes"""
        timeframe_info_list = []
        
        for timeframe, data in data_dict.items():
            try:
                # Parse timeframe
                unit, value = self._parse_timeframe(timeframe)
                if unit is None or value is None:
                    logger.warning(f"⚠️ Invalid timeframe format: {timeframe}")
                    continue
                
                # Calculate minutes
                minutes = self._timeframe_to_minutes(timeframe)
                if minutes is None:
                    logger.warning(f"⚠️ Unsupported timeframe: {timeframe}")
                    continue
                
                # Check if this is the primary timeframe
                is_higher = minutes <= self._timeframe_to_minutes(config.primary_timeframe)
                
                # Calculate data range and completeness
                if not data.empty:
                    data_range = (data.index.min(), data.index.max())
                    data_points = len(data)
                    
                    # Calculate completeness (percentage of expected candles)
                    expected_candles = self._calculate_expected_candles(
                        data_range[0], data_range[1], minutes
                    )
                    completeness = data_points / expected_candles if expected_candles > 0 else 0.0
                else:
                    data_range = (datetime.now(), datetime.now())
                    data_points = 0
                    completeness = 0.0
                
                timeframe_info = TimeframeInfo(
                    timeframe=timeframe,
                    unit=unit,
                    value=value,
                    minutes=minutes,
                    is_higher=is_higher,
                    data_range=data_range,
                    data_points=data_points,
                    completeness=completeness
                )
                
                timeframe_info_list.append(timeframe_info)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing timeframe {timeframe}: {e}")
                continue
        
        return timeframe_info_list
    
    def _parse_timeframe(self, timeframe: str) -> Tuple[Optional[TimeframeUnit], Optional[int]]:
        """Parse timeframe string to unit and value"""
        try:
            if timeframe.endswith('m'):
                return TimeframeUnit.MINUTE, int(timeframe[:-1])
            elif timeframe.endswith('h'):
                return TimeframeUnit.HOUR, int(timeframe[:-1])
            elif timeframe.endswith('d'):
                return TimeframeUnit.DAY, int(timeframe[:-1])
            elif timeframe.endswith('w'):
                return TimeframeUnit.WEEK, int(timeframe[:-1])
            elif timeframe.endswith('M'):
                return TimeframeUnit.MONTH, int(timeframe[:-1])
            else:
                return None, None
        except ValueError:
            return None, None
    
    def _timeframe_to_minutes(self, timeframe: str) -> Optional[int]:
        """Convert timeframe to minutes"""
        return self.timeframe_minutes.get(timeframe)
    
    def _calculate_expected_candles(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        timeframe_minutes: int
    ) -> int:
        """Calculate expected number of candles for a time period"""
        if start_time >= end_time:
            return 0
        
        time_diff = end_time - start_time
        total_minutes = time_diff.total_seconds() / 60
        expected_candles = int(total_minutes / timeframe_minutes) + 1
        
        return max(1, expected_candles)
    
    def _align_timeframes_internal(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        timeframe_info_list: List[TimeframeInfo], 
        config: TimeframeConfig
    ) -> Dict[str, pd.DataFrame]:
        """Internal timeframe alignment logic"""
        aligned_data = {}
        
        # Use the configured primary timeframe as the target
        primary_timeframe = config.primary_timeframe
        
        # We don't need primary_data for alignment - we're resampling TO the primary timeframe
        
        # Align all timeframes to the primary timeframe
        for timeframe_info in timeframe_info_list:
            timeframe = timeframe_info.timeframe
            data = data_dict[timeframe]
            
            if data.empty:
                continue
            
            # Resample data to align with primary timeframe
            aligned_df = self._resample_to_timeframe(
                data, timeframe_info, primary_timeframe, config
            )
            
            if aligned_df is not None and not aligned_df.empty:
                aligned_data[timeframe] = aligned_df
            else:
                # Fallback to original data if resampling fails
                aligned_data[timeframe] = data
        
        return aligned_data
    
    def _resample_to_timeframe(
        self, 
        data: pd.DataFrame, 
        timeframe_info: TimeframeInfo, 
        target_timeframe: str, 
        config: TimeframeConfig
    ) -> Optional[pd.DataFrame]:
        """Resample data to target timeframe"""
        try:
            target_minutes = self._timeframe_to_minutes(target_timeframe)
            if target_minutes is None:
                return None
            
            logger.info(f"🔄 Resampling {timeframe_info.timeframe} ({timeframe_info.minutes}min) to {target_timeframe} ({target_minutes}min)")
            
            # If source timeframe is higher resolution than target, aggregate
            if timeframe_info.minutes < target_minutes:
                logger.info(f"📊 Aggregating {timeframe_info.timeframe} to {target_timeframe}")
                resampled = self._aggregate_to_higher_timeframe(
                    data, timeframe_info.minutes, target_minutes
                )
                logger.info(f"✅ Aggregated from {len(data)} to {len(resampled)} rows")
            # If source timeframe is lower resolution than target, interpolate
            elif timeframe_info.minutes > target_minutes:
                logger.info(f"📈 Interpolating {timeframe_info.timeframe} to {target_timeframe}")
                resampled = self._interpolate_to_lower_timeframe(
                    data, timeframe_info.minutes, target_minutes
                )
                logger.info(f"✅ Interpolated from {len(data)} to {len(resampled)} rows")
            else:
                # Same timeframe, just ensure proper alignment
                logger.info(f"🔧 Aligning {timeframe_info.timeframe} to boundaries")
                resampled = self._align_to_timeframe_boundaries(data, target_minutes)
                logger.info(f"✅ Aligned from {len(data)} to {len(resampled)} rows")
            
            return resampled
            
        except Exception as e:
            logger.warning(f"⚠️ Error resampling {timeframe_info.timeframe}: {e}")
            return None
    
    def _aggregate_to_higher_timeframe(
        self, 
        data: pd.DataFrame, 
        source_minutes: int, 
        target_minutes: int
    ) -> pd.DataFrame:
        """Aggregate data to higher timeframe"""
        # Create time-based grouping
        data_copy = data.copy()
        data_copy.index = pd.to_datetime(data_copy.index)
        
        # Resample to target timeframe
        resampled = data_copy.resample(f'{target_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Remove rows with all NaN values
        resampled = resampled.dropna(how='all')
        
        return resampled
    
    def _interpolate_to_lower_timeframe(
        self, 
        data: pd.DataFrame, 
        source_minutes: int, 
        target_minutes: int
    ) -> pd.DataFrame:
        """Interpolate data to lower timeframe"""
        # This is a simplified approach - in practice, you might want more sophisticated interpolation
        data_copy = data.copy()
        data_copy.index = pd.to_datetime(data_copy.index)
        
        # Create target timeframe index
        start_time = data_copy.index.min()
        end_time = data_copy.index.max()
        target_index = pd.date_range(
            start=start_time, 
            end=end_time, 
            freq=f'{target_minutes}min'
        )
        
        # Reindex and forward fill
        resampled = data_copy.reindex(target_index, method='ffill')
        
        return resampled
    
    def _align_to_timeframe_boundaries(
        self, 
        data: pd.DataFrame, 
        timeframe_minutes: int
    ) -> pd.DataFrame:
        """Align data to exact timeframe boundaries"""
        data_copy = data.copy()
        data_copy.index = pd.to_datetime(data_copy.index)
        
        # Round timestamps to timeframe boundaries
        aligned_index = data_copy.index.round(f'{timeframe_minutes}min')
        data_copy.index = aligned_index
        
        # Remove duplicates and sort
        data_copy = data_copy[~data_copy.index.duplicated(keep='first')]
        data_copy = data_copy.sort_index()
        
        return data_copy
    
    def _handle_incomplete_candles(
        self, 
        aligned_data: Dict[str, pd.DataFrame], 
        timeframe_info_list: List[TimeframeInfo], 
        config: TimeframeConfig
    ) -> Dict[str, pd.DataFrame]:
        """Handle incomplete candles in aligned data"""
        processed_data = {}
        
        for timeframe, data in aligned_data.items():
            if data.empty:
                continue
            
            # Find timeframe info
            timeframe_info = next(
                (tf for tf in timeframe_info_list if tf.timeframe == timeframe), 
                None
            )
            
            if timeframe_info is None:
                processed_data[timeframe] = data
                continue
            
            # Check for incomplete candles
            current_time = datetime.now()
            last_candle_time = data.index.max()
            
            # Calculate expected next candle time
            expected_next = last_candle_time + timedelta(minutes=timeframe_info.minutes)
            
            # If current time is close to expected next candle, the last candle might be incomplete
            if (current_time - expected_next).total_seconds() < timeframe_info.minutes * 60:
                # Mark last candle as potentially incomplete
                data.loc[data.index[-1], 'is_incomplete'] = True
                
                # Optionally, you could remove the incomplete candle or mark it differently
                # For now, we'll keep it but mark it
            
            processed_data[timeframe] = data
        
        return processed_data
    
    def _validate_data_integrity(
        self, 
        aligned_data: Dict[str, pd.DataFrame], 
        timeframe_info_list: List[TimeframeInfo], 
        config: TimeframeConfig
    ) -> Dict[str, Any]:
        """Validate data integrity across timeframes"""
        integrity_report = {}
        
        for timeframe, data in aligned_data.items():
            if data.empty:
                continue
            
            timeframe_info = next(
                (tf for tf in timeframe_info_list if tf.timeframe == timeframe), 
                None
            )
            
            if timeframe_info is None:
                continue
            
            # Basic integrity checks
            integrity_checks = {
                'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close']),
                'no_duplicate_timestamps': not data.index.duplicated().any(),
                'timestamps_sorted': data.index.is_monotonic_increasing,
                'no_nan_prices': not data[['open', 'high', 'low', 'close']].isna().any().any(),
                'price_logic_valid': all(
                    (data['high'] >= data['low']) & 
                    (data['high'] >= data['open']) & 
                    (data['high'] >= data['close']) &
                    (data['low'] <= data['open']) & 
                    (data['low'] <= data['close'])
                ),
                'data_completeness': timeframe_info.completeness,
                'min_data_points_met': len(data) >= config.min_data_points
            }
            
            integrity_report[timeframe] = integrity_checks
        
        return integrity_report
    
    def _calculate_alignment_quality(
        self, 
        aligned_data: Dict[str, pd.DataFrame], 
        timeframe_info_list: List[TimeframeInfo], 
        config: TimeframeConfig
    ) -> float:
        """Calculate quality score for timeframe alignment"""
        if not aligned_data or not timeframe_info_list:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        # 1. Data completeness score (40 points)
        completeness_scores = [tf.completeness for tf in timeframe_info_list]
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0.0
        completeness_score = avg_completeness * 40
        total_score += completeness_score
        max_score += 40
        
        # 2. Timeframe coverage score (30 points)
        coverage_score = min(30, len(timeframe_info_list) * 5)  # 5 points per timeframe
        total_score += coverage_score
        max_score += 30
        
        # 3. Data integrity score (30 points)
        if aligned_data:
            integrity_scores = []
            for timeframe, data in aligned_data.items():
                if not data.empty:
                    # Basic integrity check
                    has_required_cols = all(col in data.columns for col in ['open', 'high', 'low', 'close'])
                    no_duplicates = not data.index.duplicated().any()
                    is_sorted = data.index.is_monotonic_increasing
                    
                    integrity_score = sum([has_required_cols, no_duplicates, is_sorted]) / 3
                    integrity_scores.append(integrity_score)
            
            if integrity_scores:
                avg_integrity = np.mean(integrity_scores)
                integrity_score = avg_integrity * 30
                total_score += integrity_score
            max_score += 30
        
        # Calculate overall quality score
        quality_score = total_score / max_score if max_score > 0 else 0.0
        
        return max(0.0, min(1.0, quality_score))
    
    def _update_stats(self, success: bool, processing_time: float, timeframes_processed: int = 0):
        """Update performance statistics"""
        self.stats['total_alignments'] += 1
        self.stats['total_processing_time_ms'] += processing_time
        self.stats['timeframes_processed'] += timeframes_processed
        
        if success:
            self.stats['successful_alignments'] += 1
        else:
            self.stats['failed_alignments'] += 1
        
        # Update average processing time
        self.stats['avg_processing_time_ms'] = (
            self.stats['total_processing_time_ms'] / self.stats['total_alignments']
        )
    
    def get_alignment_stats(self) -> Dict:
        """Get timeframe alignment service statistics"""
        return self.stats.copy()

# Example usage and testing
def test_timeframe_alignment_service():
    """Test the timeframe alignment service"""
    # Create sample data for different timeframes
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # 1-minute data
    dates_1m = pd.date_range(base_time, periods=120, freq='1min')
    data_1m = pd.DataFrame({
        'open': np.random.uniform(100, 200, 120),
        'high': np.random.uniform(100, 200, 120),
        'low': np.random.uniform(100, 200, 120),
        'close': np.random.uniform(100, 200, 120),
        'volume': np.random.uniform(1000, 10000, 120)
    }, index=dates_1m)
    
    # 5-minute data
    dates_5m = pd.date_range(base_time, periods=24, freq='5min')
    data_5m = pd.DataFrame({
        'open': np.random.uniform(100, 200, 24),
        'high': np.random.uniform(100, 200, 24),
        'low': np.random.uniform(100, 200, 24),
        'close': np.random.uniform(100, 200, 24),
        'volume': np.random.uniform(1000, 10000, 24)
    }, index=dates_5m)
    
    # 1-hour data
    dates_1h = pd.date_range(base_time, periods=2, freq='1H')
    data_1h = pd.DataFrame({
        'open': np.random.uniform(100, 200, 2),
        'high': np.random.uniform(100, 200, 2),
        'low': np.random.uniform(100, 200, 2),
        'close': np.random.uniform(100, 200, 2),
        'volume': np.random.uniform(1000, 10000, 2)
    }, index=dates_1h)
    
    # Create data dictionary
    data_dict = {
        '1m': data_1m,
        '5m': data_5m,
        '1h': data_1h
    }
    
    # Initialize service
    service = TimeframeAlignmentService()
    
    # Test timeframe alignment
    config = TimeframeConfig(
        primary_timeframe='5m',
        secondary_timeframes=['1m', '1h'],
        sync_tolerance_minutes=5,
        handle_incomplete_candles=True,
        validate_data_integrity=True
    )
    
    result = service.align_timeframes(data_dict, config)
    
    # Print results
    print("=== Timeframe Alignment Test Results ===")
    print(f"Quality Score: {result.quality_score:.2%}")
    print(f"Timeframes Processed: {len(result.timeframe_info)}")
    print(f"Aligned Data Keys: {list(result.aligned_data.keys())}")
    
    for tf_info in result.timeframe_info:
        print(f"\nTimeframe: {tf_info.timeframe}")
        print(f"  Unit: {tf_info.unit.value}")
        print(f"  Value: {tf_info.value}")
        print(f"  Minutes: {tf_info.minutes}")
        print(f"  Data Points: {tf_info.data_points}")
        print(f"  Completeness: {tf_info.completeness:.2%}")
    
    # Print statistics
    stats = service.get_alignment_stats()
    print(f"\nService Statistics:")
    print(f"Total Alignments: {stats['total_alignments']}")
    print(f"Success Rate: {stats['successful_alignments']/stats['total_alignments']:.1%}")
    print(f"Timeframes Processed: {stats['timeframes_processed']}")
    print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.2f}ms")
    
    return result

if __name__ == "__main__":
    # Run test if script is executed directly
    result = test_timeframe_alignment_service()
