"""
ETL Processor for AlphaPulse
Provides data transformation, cleaning, and preparation for ML processing
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ETLConfig:
    """ETL processing configuration"""
    enable_data_cleaning: bool = True
    enable_feature_engineering: bool = True
    enable_aggregation: bool = True
    enable_validation: bool = True
    batch_size: int = 1000
    processing_timeout: float = 30.0

@dataclass
class ETLResult:
    """ETL processing result"""
    input_count: int
    output_count: int
    processing_time: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

@dataclass
class DataQualityMetrics:
    """Data quality metrics"""
    completeness: float  # Percentage of non-null values
    accuracy: float      # Data accuracy score
    consistency: float   # Data consistency score
    timeliness: float    # Data freshness score
    validity: float      # Data validation score

class ETLProcessor:
    """ETL processor for data transformation and cleaning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # ETL configuration
        self.enable_data_cleaning = self.config.get('enable_data_cleaning', True)
        self.enable_feature_engineering = self.config.get('enable_feature_engineering', True)
        self.enable_aggregation = self.config.get('enable_aggregation', True)
        self.enable_validation = self.config.get('enable_validation', True)
        self.batch_size = self.config.get('batch_size', 1000)
        self.processing_timeout = self.config.get('processing_timeout', 30.0)
        
        # Processing state
        self.is_processing = False
        self.current_batch = None
        
        # Performance tracking
        self.stats = {
            'total_batches_processed': 0,
            'total_records_processed': 0,
            'total_processing_time': 0.0,
            'successful_batches': 0,
            'failed_batches': 0,
            'last_processing_time': None,
            'processing_times': deque(maxlen=100)
        }
        
        # Callbacks
        self.etl_callbacks = defaultdict(list)  # event_type -> [callback]
    
    async def process_market_data(self, raw_data: List[Dict[str, Any]]) -> ETLResult:
        """Process market data through ETL pipeline"""
        try:
            start_time = time.time()
            self.is_processing = True
            
            self.logger.info(f"Starting ETL processing for {len(raw_data)} records")
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(raw_data)
            input_count = len(df)
            
            # Data cleaning
            if self.enable_data_cleaning:
                df = await self._clean_market_data(df)
            
            # Feature engineering
            if self.enable_feature_engineering:
                df = await self._engineer_features(df)
            
            # Data aggregation
            if self.enable_aggregation:
                df = await self._aggregate_data(df)
            
            # Data validation
            if self.enable_validation:
                validation_result = await self._validate_data(df)
                if not validation_result['is_valid']:
                    self.logger.warning(f"Data validation failed: {validation_result['errors']}")
            
            # Convert back to list of dictionaries
            processed_data = df.to_dict('records')
            output_count = len(processed_data)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create ETL result
            result = ETLResult(
                input_count=input_count,
                output_count=output_count,
                processing_time=processing_time,
                errors=[],
                warnings=[],
                metadata={
                    'data_type': 'market_data',
                    'processing_steps': ['cleaning', 'feature_engineering', 'aggregation', 'validation'],
                    'quality_metrics': await self._calculate_quality_metrics(df)
                }
            )
            
            # Update statistics
            self._update_statistics(processing_time, True)
            
            self.logger.info(f"ETL processing completed: {input_count} -> {output_count} records in {processing_time:.2f}s")
            
            # Trigger callbacks
            await self._trigger_callbacks('etl_completed', result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"ETL processing failed: {e}")
            processing_time = time.time() - start_time
            
            # Create error result
            result = ETLResult(
                input_count=len(raw_data) if raw_data else 0,
                output_count=0,
                processing_time=processing_time,
                errors=[str(e)],
                warnings=[],
                metadata={'data_type': 'market_data', 'error': str(e)}
            )
            
            # Update statistics
            self._update_statistics(processing_time, False)
            
            # Trigger callbacks
            await self._trigger_callbacks('etl_failed', result)
            
            return result
            
        finally:
            self.is_processing = False
    
    async def _clean_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean market data"""
        try:
            original_count = len(df)
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().sum() > 0:
                    # Fill with forward fill, then backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Remove rows with critical missing values
            critical_columns = ['symbol', 'timestamp', 'price']
            df = df.dropna(subset=critical_columns)
            
            # Remove outliers using IQR method for numeric columns
            for col in numeric_columns:
                if col in ['price', 'volume', 'bid', 'ask']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            cleaned_count = len(df)
            self.logger.info(f"Data cleaning: {original_count} -> {cleaned_count} records")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {e}")
            return df
    
    async def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from market data"""
        try:
            if 'price' not in df.columns:
                return df
            
            # Price-based features
            df['price_change'] = df['price'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            df['log_return'] = np.log(df['price'] / df['price'].shift(1))
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['price'].ewm(span=window).mean()
            
            # Volatility features
            df['volatility_5'] = df['price_change'].rolling(5).std()
            df['volatility_20'] = df['price_change'].rolling(20).std()
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_ma_5'] = df['volume'].rolling(5).mean()
                df['volume_ma_20'] = df['volume'].rolling(20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            
            # Bid-ask spread features
            if 'bid' in df.columns and 'ask' in df.columns:
                df['spread'] = df['ask'] - df['bid']
                df['spread_pct'] = df['spread'] / df['bid']
                df['mid_price'] = (df['bid'] + df['ask']) / 2
            
            # Time-based features
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Technical indicators
            df = self._add_technical_indicators(df)
            
            self.logger.info(f"Feature engineering completed: {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            if 'price' not in df.columns:
                return df
            
            # RSI
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['price'].ewm(span=12).mean()
            exp2 = df['price'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['price'].rolling(window=20).mean()
            bb_std = df['price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Stochastic Oscillator
            if 'high' in df.columns and 'low' in df.columns:
                lowest_low = df['low'].rolling(window=14).min()
                highest_high = df['high'].rolling(window=14).max()
                df['stoch_k'] = 100 * ((df['price'] - lowest_low) / (highest_high - lowest_low))
                df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    async def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data for different timeframes"""
        try:
            if 'timestamp' not in df.columns:
                return df
            
            # Set timestamp as index for resampling
            df = df.set_index('timestamp')
            
            # Resample to different timeframes
            timeframes = {
                '1min': '1T',
                '5min': '5T',
                '15min': '15T',
                '1hour': '1H',
                '4hour': '4H',
                '1day': '1D'
            }
            
            aggregated_data = {}
            
            for timeframe_name, timeframe_code in timeframes.items():
                try:
                    # OHLCV aggregation
                    ohlcv = df.resample(timeframe_code).agg({
                        'price': 'ohlc',
                        'volume': 'sum' if 'volume' in df.columns else 'count'
                    })
                    
                    # Flatten column names
                    ohlcv.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in ohlcv.columns]
                    
                    # Add other aggregated features
                    if 'price_change' in df.columns:
                        ohlcv['price_change_mean'] = df['price_change'].resample(timeframe_code).mean()
                        ohlcv['price_change_std'] = df['price_change'].resample(timeframe_code).std()
                    
                    if 'rsi' in df.columns:
                        ohlcv['rsi_mean'] = df['rsi'].resample(timeframe_code).mean()
                    
                    aggregated_data[timeframe_name] = ohlcv
                    
                except Exception as e:
                    self.logger.warning(f"Failed to aggregate {timeframe_name}: {e}")
                    continue
            
            # Store aggregated data in metadata
            self.current_batch = {
                'raw_data': df,
                'aggregated_data': aggregated_data
            }
            
            self.logger.info(f"Data aggregation completed: {len(aggregated_data)} timeframes")
            
            # Return the original data (aggregated data is stored separately)
            return df.reset_index()
            
        except Exception as e:
            self.logger.error(f"Error in data aggregation: {e}")
            return df
    
    async def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate processed data"""
        try:
            validation_result = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'metrics': {}
            }
            
            # Check for required columns
            required_columns = ['symbol', 'timestamp', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                validation_result['is_valid'] = False
            
            # Check data types
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                validation_result['warnings'].append("Timestamp column is not datetime type")
            
            if 'price' in df.columns and not pd.api.types.is_numeric_dtype(df['price']):
                validation_result['errors'].append("Price column is not numeric")
                validation_result['is_valid'] = False
            
            # Check for infinite values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if np.isinf(df[col]).any():
                    validation_result['warnings'].append(f"Column {col} contains infinite values")
            
            # Check for extreme outliers
            for col in numeric_columns:
                if col in ['price', 'volume']:
                    Q1 = df[col].quantile(0.01)
                    Q99 = df[col].quantile(0.99)
                    extreme_outliers = df[(df[col] < Q1) | (df[col] > Q99)]
                    if len(extreme_outliers) > 0:
                        validation_result['warnings'].append(f"Column {col} has {len(extreme_outliers)} extreme outliers")
            
            # Calculate quality metrics
            validation_result['metrics'] = await self._calculate_quality_metrics(df)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': [],
                'metrics': {}
            }
    
    async def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate data quality metrics"""
        try:
            metrics = {}
            
            # Completeness
            total_cells = df.size
            non_null_cells = df.count().sum()
            metrics['completeness'] = non_null_cells / total_cells if total_cells > 0 else 0.0
            
            # Timeliness (if timestamp column exists)
            if 'timestamp' in df.columns:
                now = pd.Timestamp.now()
                time_diffs = (now - df['timestamp']).dt.total_seconds()
                avg_age_hours = time_diffs.mean() / 3600
                metrics['timeliness'] = max(0, 1 - (avg_age_hours / 24))  # 1 = fresh, 0 = old
            
            # Consistency (check for duplicate timestamps per symbol)
            if 'timestamp' in df.columns and 'symbol' in df.columns:
                duplicates = df.groupby(['symbol', 'timestamp']).size()
                duplicate_count = (duplicates > 1).sum()
                total_unique = len(df.groupby(['symbol', 'timestamp']))
                metrics['consistency'] = 1 - (duplicate_count / total_unique) if total_unique > 0 else 1.0
            
            # Validity (check for reasonable price ranges)
            if 'price' in df.columns:
                valid_prices = df[(df['price'] > 0) & (df['price'] < 1e10)]
                metrics['validity'] = len(valid_prices) / len(df) if len(df) > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            return {
                'completeness': 0.0,
                'timeliness': 0.0,
                'consistency': 0.0,
                'validity': 0.0
            }
    
    def _update_statistics(self, processing_time: float, success: bool):
        """Update processing statistics"""
        try:
            self.stats['total_batches_processed'] += 1
            self.stats['total_processing_time'] += processing_time
            self.stats['last_processing_time'] = datetime.now()
            self.stats['processing_times'].append(processing_time)
            
            if success:
                self.stats['successful_batches'] += 1
            else:
                self.stats['failed_batches'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")
    
    # Public methods
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for ETL events"""
        self.etl_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for ETL events"""
        callbacks = self.etl_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_processor_statistics(self) -> Dict[str, Any]:
        """Get ETL processor statistics"""
        return {
            'stats': self.stats,
            'config': {
                'enable_data_cleaning': self.enable_data_cleaning,
                'enable_feature_engineering': self.enable_feature_engineering,
                'enable_aggregation': self.enable_aggregation,
                'enable_validation': self.enable_validation,
                'batch_size': self.batch_size
            },
            'current_status': {
                'is_processing': self.is_processing,
                'current_batch_size': len(self.current_batch['raw_data']) if self.current_batch else 0
            }
        }
    
    async def close(self):
        """Close the ETL processor"""
        try:
            self.logger.info("ETL Processor closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close ETL processor: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
