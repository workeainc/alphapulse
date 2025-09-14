#!/usr/bin/env python3
"""
Gap Filling Service for AlphaPulse
Phase 2: Enhanced Gap Filling
Handles interpolated price filling, zero-volume flat candles, and news correlation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.interpolate import interp1d
import requests
import json
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class GapInfo:
    """Information about a detected gap"""
    start_time: datetime
    end_time: datetime
    gap_duration: timedelta
    gap_type: str  # 'price', 'volume', 'time', 'mixed'
    severity: str  # 'low', 'medium', 'high'
    affected_columns: List[str]
    estimated_rows_missing: int

@dataclass
class GapFillingResult:
    """Result of gap filling process"""
    filled_data: pd.DataFrame
    gaps_filled: List[GapInfo]
    filling_report: Dict[str, Any]
    quality_score: float
    warnings: List[str]
    errors: List[str]

class GapFillingService:
    """
    Comprehensive gap filling service for AlphaPulse
    Handles interpolated price filling, zero-volume flat candles, and news correlation
    """
    
    def __init__(self, news_api_key: Optional[str] = None):
        self.news_api_key = news_api_key
        
        # Gap detection thresholds
        self.thresholds = {
            'min_gap_minutes': 5,  # Minimum gap size to consider
            'max_gap_hours': 24,    # Maximum gap size to fill
            'price_change_threshold': 0.1,  # 10% price change threshold
            'volume_spike_threshold': 5.0,  # 5x volume spike threshold
            'interpolation_method': 'linear'  # 'linear', 'cubic', 'nearest'
        }
        
        # News correlation settings
        self.news_correlation = {
            'enabled': news_api_key is not None,
            'search_radius_hours': 2,  # Search for news within 2 hours of gap
            'keywords': ['crypto', 'bitcoin', 'ethereum', 'trading', 'market'],
            'correlation_threshold': 0.6
        }
        
        # Performance tracking
        self.stats = {
            'total_gaps_detected': 0,
            'gaps_filled': 0,
            'interpolation_used': 0,
            'zero_volume_candles_created': 0,
            'news_correlations_found': 0,
            'avg_filling_time_ms': 0.0,
            'total_filling_time_ms': 0.0
        }
        
        logger.info("🚀 Gap Filling Service initialized")
    
    async def fill_gaps_in_data(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        timeframe: str = "1h",
        fill_method: str = "auto",
        enable_news_correlation: bool = True
    ) -> GapFillingResult:
        """
        Fill gaps in candlestick data using various methods
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading pair symbol
            timeframe: Data timeframe (e.g., "1h", "15m")
            fill_method: Filling method ('auto', 'interpolate', 'zero_volume', 'news_correlated')
            enable_news_correlation: Whether to use news correlation for volume spikes
            
        Returns:
            GapFillingResult with filled data and report
        """
        start_time = datetime.now()
        
        try:
            # Create copy for gap filling
            filled_df = data.copy()
            
            # Initialize report
            filling_report = {
                'symbol': symbol,
                'timeframe': timeframe,
                'original_shape': data.shape,
                'filled_shape': None,
                'gaps_detected': 0,
                'gaps_filled': 0,
                'filling_methods_used': [],
                'news_correlations': [],
                'timestamp': datetime.now().isoformat()
            }
            
            warnings = []
            errors = []
            
            # 1. Detect gaps
            gaps = self._detect_gaps(filled_df, timeframe)
            filling_report['gaps_detected'] = len(gaps)
            
            if not gaps:
                logger.info(f"✅ No gaps detected in {symbol} data")
                return GapFillingResult(
                    filled_data=filled_df,
                    gaps_filled=[],
                    filling_report=filling_report,
                    quality_score=1.0,
                    warnings=warnings,
                    errors=errors
                )
            
            # 2. Fill gaps based on method
            if fill_method == "auto":
                filled_df, gaps_filled = await self._fill_gaps_auto(
                    filled_df, gaps, symbol, enable_news_correlation
                )
            elif fill_method == "interpolate":
                filled_df, gaps_filled = self._fill_gaps_interpolate(filled_df, gaps)
            elif fill_method == "zero_volume":
                filled_df, gaps_filled = self._fill_gaps_zero_volume(filled_df, gaps, timeframe)
            elif fill_method == "news_correlated":
                filled_df, gaps_filled = await self._fill_gaps_news_correlated(filled_df, gaps, symbol)
            else:
                raise ValueError(f"Unknown fill method: {fill_method}")
            
            # 3. Update report
            filling_report['gaps_filled'] = len(gaps_filled)
            filling_report['filled_shape'] = filled_df.shape
            
            # 4. Calculate quality score
            quality_score = self._calculate_filling_quality(filled_df, gaps, gaps_filled)
            
            # 5. Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(len(gaps_filled), processing_time)
            
            logger.info(f"✅ Gap filling completed for {symbol}: {len(gaps_filled)}/{len(gaps)} gaps filled in {processing_time:.2f}ms")
            
            return GapFillingResult(
                filled_data=filled_df,
                gaps_filled=gaps_filled,
                filling_report=filling_report,
                quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_stats(0, processing_time)
            error_msg = f"Gap filling failed for {symbol}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return GapFillingResult(
                filled_data=data,  # Return original data on failure
                gaps_filled=[],
                filling_report={'error': error_msg},
                quality_score=0.0,
                warnings=[],
                errors=[error_msg]
            )
    
    def _detect_gaps(self, df: pd.DataFrame, timeframe: str) -> List[GapInfo]:
        """Detect gaps in the data"""
        gaps = []
        
        if len(df) < 2:
            return gaps
        
        # Parse timeframe to get expected interval
        expected_interval = self._parse_timeframe(timeframe)
        if not expected_interval:
            return gaps
        
        # Sort by time index
        df_sorted = df.sort_index()
        
        # Find gaps by checking time differences between consecutive rows
        for i in range(len(df_sorted) - 1):
            current_time = df_sorted.index[i]
            next_time = df_sorted.index[i + 1]
            
            # Calculate actual time difference
            time_diff = next_time - current_time
            
            # Check if this is a gap (more than 2x expected interval)
            if time_diff > expected_interval * 2:
                # Calculate gap duration
                gap_duration = time_diff - expected_interval
                
                # Skip if gap is too small or too large
                if gap_duration < timedelta(minutes=self.thresholds['min_gap_minutes']):
                    continue
                if gap_duration > timedelta(hours=self.thresholds['max_gap_hours']):
                    continue
                
                # Determine gap type and severity
                gap_type, severity = self._classify_gap(df_sorted, current_time, next_time, gap_duration)
                
                # Estimate missing rows
                estimated_rows = int(gap_duration / expected_interval)
                
                gap_info = GapInfo(
                    start_time=current_time,
                    end_time=next_time,
                    gap_duration=gap_duration,
                    gap_type=gap_type,
                    severity=severity,
                    affected_columns=['open', 'high', 'low', 'close', 'volume'],
                    estimated_rows_missing=estimated_rows
                )
                
                gaps.append(gap_info)
        
        return gaps
    
    def _parse_timeframe(self, timeframe: str) -> Optional[timedelta]:
        """Parse timeframe string to timedelta"""
        timeframe_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        return timeframe_map.get(timeframe)
    
    def _classify_gap(self, df: pd.DataFrame, gap_start: datetime, gap_end: datetime, gap_duration: timedelta) -> Tuple[str, str]:
        """Classify gap type and severity"""
        # Get data before and after gap
        before_gap = df[df.index < gap_start].tail(5)
        after_gap = df[df.index > gap_end].head(5)
        
        if before_gap.empty or after_gap.empty:
            return 'time', 'medium'
        
        # Check for price gaps
        last_close = before_gap['close'].iloc[-1]
        first_open = after_gap['open'].iloc[0]
        price_change = abs(first_open - last_close) / last_close
        
        # Check for volume gaps
        avg_volume_before = before_gap['volume'].mean()
        avg_volume_after = after_gap['volume'].mean()
        volume_change = abs(avg_volume_after - avg_volume_before) / avg_volume_before if avg_volume_before > 0 else 0
        
        # Classify gap type
        if price_change > self.thresholds['price_change_threshold']:
            gap_type = 'price'
        elif volume_change > self.thresholds['volume_spike_threshold']:
            gap_type = 'volume'
        elif price_change > 0.05 or volume_change > 2.0:
            gap_type = 'mixed'
        else:
            gap_type = 'time'
        
        # Classify severity
        if gap_duration > timedelta(hours=12):
            severity = 'high'
        elif gap_duration > timedelta(hours=2):
            severity = 'medium'
        else:
            severity = 'low'
        
        return gap_type, severity
    
    async def _fill_gaps_auto(
        self, 
        df: pd.DataFrame, 
        gaps: List[GapInfo], 
        symbol: str,
        enable_news_correlation: bool
    ) -> Tuple[pd.DataFrame, List[GapInfo]]:
        """Automatically choose best filling method for each gap"""
        filled_df = df.copy()
        gaps_filled = []
        
        for gap in gaps:
            try:
                if gap.gap_type == 'price' and gap.severity == 'high':
                    # High severity price gaps: use news correlation if available
                    if enable_news_correlation and self.news_correlation['enabled']:
                        filled_df, success = await self._fill_gap_news_correlated(filled_df, gap, symbol)
                    else:
                        filled_df, success = self._fill_gap_interpolate(filled_df, gap)
                elif gap.gap_type == 'volume':
                    # Volume gaps: use zero-volume candles
                    filled_df, success = self._fill_gap_zero_volume(filled_df, gap, '1h')
                else:
                    # Time gaps: use interpolation
                    filled_df, success = self._fill_gap_interpolate(filled_df, gap)
                
                if success:
                    gaps_filled.append(gap)
                    
            except Exception as e:
                logger.warning(f"Failed to fill gap {gap.start_time}: {e}")
        
        return filled_df, gaps_filled
    
    def _fill_gaps_interpolate(self, df: pd.DataFrame, gaps: List[GapInfo]) -> Tuple[pd.DataFrame, List[GapInfo]]:
        """Fill gaps using interpolation"""
        filled_df = df.copy()
        gaps_filled = []
        
        for gap in gaps:
            try:
                filled_df, success = self._fill_gap_interpolate(filled_df, gap)
                if success:
                    gaps_filled.append(gap)
                    self.stats['interpolation_used'] += 1
            except Exception as e:
                logger.warning(f"Failed to interpolate gap {gap.start_time}: {e}")
        
        return filled_df, gaps_filled
    
    def _fill_gap_interpolate(self, df: pd.DataFrame, gap: GapInfo) -> Tuple[pd.DataFrame, bool]:
        """Fill a single gap using interpolation"""
        try:
            # Get data before and after gap
            before_gap = df[df.index < gap.start_time].tail(10)
            after_gap = df[df.index > gap.end_time].head(10)
            
            if before_gap.empty or after_gap.empty:
                return df, False
            
            # Create interpolation function for each column
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    # Combine before and after data for interpolation
                    x_before = [i for i in range(len(before_gap))]
                    x_after = [len(before_gap) + gap.estimated_rows_missing + i for i in range(len(after_gap))]
                    y_before = before_gap[col].values
                    y_after = after_gap[col].values
                    
                    # Create interpolation function
                    x_combined = x_before + x_after
                    y_combined = np.concatenate([y_before, y_after])
                    
                    if len(x_combined) >= 2:
                        interp_func = interp1d(x_combined, y_combined, kind=self.thresholds['interpolation_method'])
                        
                        # Generate interpolated values
                        x_interp = range(len(before_gap), len(before_gap) + gap.estimated_rows_missing)
                        y_interp = interp_func(x_interp)
                        
                        # Insert interpolated values
                        for i, (x_val, y_val) in enumerate(zip(x_interp, y_interp)):
                            insert_time = gap.start_time + timedelta(minutes=i * 60)  # Assuming hourly data
                            if insert_time not in df.index:
                                new_row = pd.Series({
                                    'open': y_val,
                                    'high': y_val,
                                    'low': y_val,
                                    'close': y_val,
                                    'volume': 0  # Zero volume for interpolated candles
                                }, name=insert_time)
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=False)
            
            return df, True
            
        except Exception as e:
            logger.error(f"Error interpolating gap: {e}")
            return df, False
    
    def _fill_gaps_zero_volume(self, df: pd.DataFrame, gaps: List[GapInfo], timeframe: str) -> Tuple[pd.DataFrame, List[GapInfo]]:
        """Fill gaps using zero-volume flat candles"""
        filled_df = df.copy()
        gaps_filled = []
        
        for gap in gaps:
            try:
                filled_df, success = self._fill_gap_zero_volume(filled_df, gap, timeframe)
                if success:
                    gaps_filled.append(gap)
                    self.stats['zero_volume_candles_created'] += 1
            except Exception as e:
                logger.warning(f"Failed to create zero-volume candles for gap {gap.start_time}: {e}")
        
        return filled_df, gaps_filled
    
    def _fill_gap_zero_volume(self, df: pd.DataFrame, gap: GapInfo, timeframe: str) -> Tuple[pd.DataFrame, bool]:
        """Fill a single gap using zero-volume flat candles"""
        try:
            # Get the last price before the gap
            before_gap = df[df.index < gap.start_time].tail(1)
            if before_gap.empty:
                return df, False
            
            last_close = before_gap['close'].iloc[-1]
            expected_interval = self._parse_timeframe(timeframe)
            
            # Create zero-volume flat candles
            for i in range(gap.estimated_rows_missing):
                insert_time = gap.start_time + timedelta(minutes=i * expected_interval.total_seconds() / 60)
                if insert_time not in df.index:
                    new_row = pd.Series({
                        'open': last_close,
                        'high': last_close,
                        'low': last_close,
                        'close': last_close,
                        'volume': 0
                    }, name=insert_time)
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=False)
            
            return df, True
            
        except Exception as e:
            logger.error(f"Error creating zero-volume candles: {e}")
            return df, False
    
    async def _fill_gaps_news_correlated(self, df: pd.DataFrame, gaps: List[GapInfo], symbol: str) -> Tuple[pd.DataFrame, List[GapInfo]]:
        """Fill gaps using news correlation"""
        filled_df = df.copy()
        gaps_filled = []
        
        for gap in gaps:
            try:
                filled_df, success = await self._fill_gap_news_correlated(filled_df, gap, symbol)
                if success:
                    gaps_filled.append(gap)
                    self.stats['news_correlations_found'] += 1
            except Exception as e:
                logger.warning(f"Failed to fill gap with news correlation {gap.start_time}: {e}")
        
        return filled_df, gaps_filled
    
    async def _fill_gap_news_correlated(self, df: pd.DataFrame, gap: GapInfo, symbol: str) -> Tuple[pd.DataFrame, bool]:
        """Fill a single gap using news correlation"""
        try:
            if not self.news_correlation['enabled']:
                return df, False
            
            # Search for news around the gap time
            news_data = await self._search_news_around_gap(symbol, gap.start_time)
            
            if news_data and len(news_data) > 0:
                # Use news sentiment to adjust filling strategy
                sentiment_score = self._calculate_news_sentiment(news_data)
                
                # Adjust interpolation based on sentiment
                if sentiment_score > 0.6:  # Positive news
                    # Use upward interpolation
                    return self._fill_gap_interpolate_with_sentiment(df, gap, sentiment_score)
                elif sentiment_score < -0.6:  # Negative news
                    # Use downward interpolation
                    return self._fill_gap_interpolate_with_sentiment(df, gap, sentiment_score)
                else:
                    # Neutral news, use standard interpolation
                    return self._fill_gap_interpolate(df, gap)
            else:
                # No news found, fall back to standard interpolation
                return self._fill_gap_interpolate(df, gap)
                
        except Exception as e:
            logger.error(f"Error in news-correlated gap filling: {e}")
            return df, False
    
    async def _search_news_around_gap(self, symbol: str, gap_time: datetime) -> List[Dict]:
        """Search for news around the gap time"""
        try:
            if not self.news_api_key:
                return []
            
            # Search radius
            search_start = gap_time - timedelta(hours=self.news_correlation['search_radius_hours'])
            search_end = gap_time + timedelta(hours=self.news_correlation['search_radius_hours'])
            
            # Search for crypto-related news
            search_query = f"crypto {symbol.split('USDT')[0]}"
            
            # This is a placeholder - you would integrate with a real news API
            # For now, return mock data
            mock_news = [
                {
                    'title': f'Crypto market update for {symbol}',
                    'description': f'Market analysis for {symbol} during the gap period',
                    'published_at': gap_time.isoformat(),
                    'sentiment': 'neutral'
                }
            ]
            
            return mock_news
            
        except Exception as e:
            logger.warning(f"News search failed: {e}")
            return []
    
    def _calculate_news_sentiment(self, news_data: List[Dict]) -> float:
        """Calculate overall sentiment score from news data"""
        if not news_data:
            return 0.0
        
        sentiment_scores = []
        for news in news_data:
            sentiment = news.get('sentiment', 'neutral')
            if sentiment == 'positive':
                sentiment_scores.append(1.0)
            elif sentiment == 'negative':
                sentiment_scores.append(-1.0)
            else:
                sentiment_scores.append(0.0)
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.0
    
    def _fill_gap_interpolate_with_sentiment(self, df: pd.DataFrame, gap: GapInfo, sentiment_score: float) -> Tuple[pd.DataFrame, bool]:
        """Fill gap with sentiment-adjusted interpolation"""
        # This is a simplified version - you could implement more sophisticated sentiment-based filling
        return self._fill_gap_interpolate(df, gap)
    
    def _calculate_filling_quality(self, df: pd.DataFrame, original_gaps: List[GapInfo], filled_gaps: List[GapInfo]) -> float:
        """Calculate quality score for gap filling"""
        if not original_gaps:
            return 1.0
        
        # Base score from filling success rate
        filling_success_rate = len(filled_gaps) / len(original_gaps)
        
        # Quality score based on data continuity
        continuity_score = 0.0
        if len(df) > 1:
            # Check for price continuity
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.5]  # 50% changes
            continuity_score = 1.0 - (len(extreme_changes) / len(df))
        
        # Overall quality score
        quality_score = (filling_success_rate * 0.7) + (continuity_score * 0.3)
        
        return max(0.0, min(1.0, quality_score))
    
    def _update_stats(self, gaps_filled: int, processing_time: float):
        """Update performance statistics"""
        self.stats['gaps_filled'] += gaps_filled
        self.stats['total_filling_time_ms'] += processing_time
        
        # Update average processing time
        if self.stats['gaps_filled'] > 0:
            self.stats['avg_filling_time_ms'] = (
                self.stats['total_filling_time_ms'] / self.stats['gaps_filled']
            )
    
    def get_gap_filling_stats(self) -> Dict:
        """Get gap filling service statistics"""
        return self.stats.copy()

# Example usage and testing
async def test_gap_filling_service():
    """Test the gap filling service"""
    # Create sample data with gaps
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Introduce some gaps
    data = data.drop(data.index[20:25])  # 5-hour gap
    data = data.drop(data.index[50:55])  # Another 5-hour gap
    
    # Initialize service
    service = GapFillingService()
    
    # Test gap filling
    result = await service.fill_gaps_in_data(
        data=data,
        symbol="BTCUSDT",
        timeframe="1h",
        fill_method="auto",
        enable_news_correlation=False
    )
    
    # Print results
    print("=== Gap Filling Test Results ===")
    print(f"Quality Score: {result.quality_score:.2%}")
    print(f"Original Shape: {data.shape}")
    print(f"Filled Shape: {result.filled_data.shape}")
    print(f"Gaps Detected: {len(result.filling_report.get('gaps_detected', 0))}")
    print(f"Gaps Filled: {len(result.gaps_filled)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.errors:
        print("Errors:", result.errors)
    
    # Print statistics
    stats = service.get_gap_filling_stats()
    print(f"\nService Statistics:")
    print(f"Total Gaps Detected: {stats['total_gaps_detected']}")
    print(f"Gaps Filled: {stats['gaps_filled']}")
    print(f"Interpolation Used: {stats['interpolation_used']}")
    print(f"Zero Volume Candles: {stats['zero_volume_candles_created']}")
    print(f"Avg Filling Time: {stats['avg_filling_time_ms']:.2f}ms")
    
    return result

if __name__ == "__main__":
    # Run test if script is executed directly
    result = asyncio.run(test_gap_filling_service())
