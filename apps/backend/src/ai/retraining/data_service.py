"""
Retraining Data Service for AlphaPulse
Phase 5: Consolidated Retraining System

Provides data preparation for different retraining cadences:
1. Weekly quick retrain (8-12 weeks data)
2. Monthly full retrain (12-24 months data)
3. Nightly incremental updates (daily data)

Now integrates with real data sources via RealDataIntegrationService
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
import time

# Local imports
from ...core.prefect_config import prefect_settings
from ..advanced_logging_system import redis_logger, EventType, LogLevel
from ..feature_engineering import FeatureExtractor
from ..real_data_integration_service import real_data_integration_service

logger = logging.getLogger(__name__)

class RetrainingDataService:
    """
    Service for preparing training data for different retraining cadences
    Optimizes data selection and preprocessing for efficiency
    Now integrates with real data sources
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        
        # Data cache for efficiency
        self.data_cache = {}
        self.cache_ttl = prefect_settings.DATA_CACHE_TTL
        
        # Data quality thresholds
        self.quality_thresholds = {
            'min_data_points': {
                'weekly_quick': 1000,
                'monthly_full': 5000,
                'nightly_incremental': 100
            },
            'min_feature_completeness': 0.95,  # 95% feature completeness
            'max_missing_values': 0.05,  # 5% max missing values
            'min_data_quality_score': 0.8  # 80% quality score
        }
        
        # Feature selection for different cadences
        self.feature_configs = {
            'weekly_quick': {
                'use_technical_indicators': True,
                'use_sentiment': True,
                'use_market_regime': True,
                'use_volume_analysis': True,
                'feature_count': 50  # Lightweight feature set
            },
            'monthly_full': {
                'use_technical_indicators': True,
                'use_sentiment': True,
                'use_market_regime': True,
                'use_volume_analysis': True,
                'use_advanced_features': True,
                'feature_count': 100  # Comprehensive feature set
            },
            'nightly_incremental': {
                'use_technical_indicators': True,
                'use_sentiment': False,  # Skip heavy sentiment for speed
                'use_market_regime': True,
                'use_volume_analysis': True,
                'use_advanced_features': False,
                'feature_count': 25  # Minimal feature set
            }
        }
        
        # Default symbols for trading
        self.default_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        
        logger.info("🚀 Retraining Data Service initialized with real data integration")
    
    async def prepare_weekly_training_data(self, 
                                         symbols: List[str] = None,
                                         weeks: int = None) -> Optional[pd.DataFrame]:
        """
        Prepare training data for weekly quick retrain (8-12 weeks)
        
        Args:
            symbols: List of trading symbols
            weeks: Number of weeks (default: 8-12 weeks)
            
        Returns:
            DataFrame with training data or None if insufficient data
        """
        try:
            symbols = symbols or self.default_symbols
            weeks = weeks or 10  # Default to 10 weeks
            
            logger.info(f"🔄 Preparing weekly training data for {len(symbols)} symbols over {weeks} weeks")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)
            
            # Get data from real data integration service
            training_data = await real_data_integration_service.get_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1h"  # Hourly data for weekly retrain
            )
            
            if training_data is None or training_data.empty:
                logger.warning("⚠️ No training data available for weekly retrain")
                return None
            
            # Apply feature engineering
            training_data = await self._apply_feature_engineering(
                training_data, 
                feature_config=self.feature_configs['weekly_quick']
            )
            
            # Validate data quality
            if not await self._validate_data_quality(training_data, 'weekly_quick'):
                logger.warning("⚠️ Data quality insufficient for weekly retrain")
                return None
            
            # Cache the data
            cache_key = f"weekly_quick_{hashlib.md5(str(symbols).encode()).hexdigest()[:8]}"
            self.data_cache[cache_key] = {
                'data': training_data,
                'timestamp': time.time(),
                'ttl': self.cache_ttl
            }
            
            logger.info(f"✅ Weekly training data prepared: {len(training_data)} rows, {training_data.shape[1]} features")
            return training_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing weekly training data: {e}")
            return None
    
    async def prepare_monthly_training_data(self, 
                                          symbols: List[str] = None,
                                          months: int = None) -> Optional[pd.DataFrame]:
        """
        Prepare training data for monthly full retrain (12-24 months)
        
        Args:
            symbols: List of trading symbols
            months: Number of months (default: 12-24 months)
            
        Returns:
            DataFrame with training data or None if insufficient data
        """
        try:
            symbols = symbols or self.default_symbols
            months = months or 18  # Default to 18 months
            
            logger.info(f"🔄 Preparing monthly training data for {len(symbols)} symbols over {months} months")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=months*30)
            
            # Get data from real data integration service
            training_data = await real_data_integration_service.get_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="4h"  # 4-hour data for monthly retrain
            )
            
            if training_data is None or training_data.empty:
                logger.warning("⚠️ No training data available for monthly retrain")
                return None
            
            # Apply feature engineering
            training_data = await self._apply_feature_engineering(
                training_data, 
                feature_config=self.feature_configs['monthly_full']
            )
            
            # Validate data quality
            if not await self._validate_data_quality(training_data, 'monthly_full'):
                logger.warning("⚠️ Data quality insufficient for monthly retrain")
                return None
            
            # Cache the data
            cache_key = f"monthly_full_{hashlib.md5(str(symbols).encode()).hexdigest()[:8]}"
            self.data_cache[cache_key] = {
                'data': training_data,
                'timestamp': time.time(),
                'ttl': self.cache_ttl
            }
            
            logger.info(f"✅ Monthly training data prepared: {len(training_data)} rows, {training_data.shape[1]} features")
            return training_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing monthly training data: {e}")
            return None
    
    async def prepare_nightly_training_data(self, 
                                          symbols: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Prepare training data for nightly incremental update (daily data)
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            DataFrame with training data or None if insufficient data
        """
        try:
            symbols = symbols or self.default_symbols
            
            logger.info(f"🔄 Preparing nightly training data for {len(symbols)} symbols")
            
            # Calculate date range (last 24 hours)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            # Get data from real data integration service
            training_data = await real_data_integration_service.get_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="15m"  # 15-minute data for nightly update
            )
            
            if training_data is None or training_data.empty:
                logger.warning("⚠️ No training data available for nightly update")
                return None
            
            # Apply feature engineering
            training_data = await self._apply_feature_engineering(
                training_data, 
                feature_config=self.feature_configs['nightly_incremental']
            )
            
            # Validate data quality
            if not await self._validate_data_quality(training_data, 'nightly_incremental'):
                logger.warning("⚠️ Data quality insufficient for nightly update")
                return None
            
            # Cache the data
            cache_key = f"nightly_incremental_{hashlib.md5(str(symbols).encode()).hexdigest()[:8]}"
            self.data_cache[cache_key] = {
                'data': training_data,
                'timestamp': time.time(),
                'ttl': self.cache_ttl
            }
            
            logger.info(f"✅ Nightly training data prepared: {len(training_data)} rows, {training_data.shape[1]} features")
            return training_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing nightly training data: {e}")
            return None
    
    async def _apply_feature_engineering(self, 
                                       data: pd.DataFrame, 
                                       feature_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply feature engineering based on configuration
        
        Args:
            data: Raw training data
            feature_config: Feature configuration dictionary
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info(f"🔧 Applying feature engineering with config: {feature_config['feature_count']} features")
            
            # Extract features based on configuration
            if feature_config.get('use_technical_indicators', False):
                data = await self.feature_extractor.add_technical_indicators(data)
            
            if feature_config.get('use_sentiment', False):
                data = await self.feature_extractor.add_sentiment_features(data)
            
            if feature_config.get('use_market_regime', False):
                data = await self.feature_extractor.add_market_regime_features(data)
            
            if feature_config.get('use_volume_analysis', False):
                data = await self.feature_extractor.add_volume_features(data)
            
            if feature_config.get('use_advanced_features', False):
                data = await self.feature_extractor.add_advanced_features(data)
            
            # Ensure we have the target number of features
            if len(data.columns) > feature_config['feature_count']:
                # Select most important features
                data = await self._select_important_features(data, feature_config['feature_count'])
            
            logger.info(f"✅ Feature engineering completed: {data.shape[1]} features")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error in feature engineering: {e}")
            return data
    
    async def _select_important_features(self, 
                                       data: pd.DataFrame, 
                                       target_count: int) -> pd.DataFrame:
        """
        Select most important features to meet target count
        
        Args:
            data: DataFrame with all features
            target_count: Target number of features
            
        Returns:
            DataFrame with selected features
        """
        try:
            # Simple feature selection based on variance and correlation
            # In production, use more sophisticated methods like mutual information, SHAP values, etc.
            
            # Remove constant features
            constant_features = data.columns[data.nunique() == 1]
            if len(constant_features) > 0:
                data = data.drop(columns=constant_features)
                logger.info(f"🧹 Removed {len(constant_features)} constant features")
            
            # Remove highly correlated features
            if len(data.columns) > target_count:
                correlation_matrix = data.corr().abs()
                upper_triangle = correlation_matrix.where(
                    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
                )
                
                # Find highly correlated features (correlation > 0.95)
                high_corr_features = [column for column in upper_triangle.columns 
                                    if any(upper_triangle[column] > 0.95)]
                
                if len(high_corr_features) > 0:
                    data = data.drop(columns=high_corr_features)
                    logger.info(f"🧹 Removed {len(high_corr_features)} highly correlated features")
            
            # If still too many features, select by variance
            if len(data.columns) > target_count:
                feature_variance = data.var().sort_values(ascending=False)
                selected_features = feature_variance.head(target_count).index.tolist()
                data = data[selected_features]
                logger.info(f"📊 Selected {len(selected_features)} features by variance")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error in feature selection: {e}")
            return data
    
    async def _validate_data_quality(self, 
                                   data: pd.DataFrame, 
                                   cadence_type: str) -> bool:
        """
        Validate data quality for retraining
        
        Args:
            data: Training data to validate
            cadence_type: Type of retraining cadence
            
        Returns:
            True if data quality is sufficient, False otherwise
        """
        try:
            thresholds = self.quality_thresholds
            
            # Check minimum data points
            min_points = thresholds['min_data_points'].get(cadence_type, 100)
            if len(data) < min_points:
                logger.warning(f"⚠️ Insufficient data points: {len(data)} < {min_points}")
                return False
            
            # Check feature completeness
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio > thresholds['max_missing_values']:
                logger.warning(f"⚠️ Too many missing values: {missing_ratio:.2%} > {thresholds['max_missing_values']:.2%}")
                return False
            
            # Check data quality score
            quality_score = self._calculate_data_quality_score(data)
            if quality_score < thresholds['min_data_quality_score']:
                logger.warning(f"⚠️ Data quality score too low: {quality_score:.2f} < {thresholds['min_data_quality_score']}")
                return False
            
            logger.info(f"✅ Data quality validation passed: {len(data)} rows, quality score: {quality_score:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in data quality validation: {e}")
            return False
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate overall data quality score
        
        Args:
            data: Training data
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Calculate various quality metrics
            completeness_score = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            # Check for infinite values
            infinite_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
            infinite_score = 1 - (infinite_count / (len(data) * len(data.columns)))
            
            # Check for duplicate rows
            duplicate_ratio = data.duplicated().sum() / len(data)
            duplicate_score = 1 - duplicate_ratio
            
            # Check for outliers (simple approach using IQR)
            outlier_scores = []
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_ratio = outlier_count / len(data)
                outlier_scores.append(1 - outlier_ratio)
            
            outlier_score = np.mean(outlier_scores) if outlier_scores else 1.0
            
            # Combine scores with weights
            quality_score = (
                0.4 * completeness_score +
                0.2 * infinite_score +
                0.2 * duplicate_score +
                0.2 * outlier_score
            )
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating data quality score: {e}")
            return 0.0
    
    async def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Get cached training data
        
        Args:
            cache_key: Cache key for the data
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        try:
            if cache_key in self.data_cache:
                cache_entry = self.data_cache[cache_key]
                
                # Check if cache is still valid
                if time.time() - cache_entry['timestamp'] < cache_entry['ttl']:
                    logger.info(f"✅ Retrieved cached data for key: {cache_key}")
                    return cache_entry['data']
                else:
                    # Remove expired cache entry
                    del self.data_cache[cache_key]
                    logger.info(f"🧹 Removed expired cache entry: {cache_key}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error retrieving cached data: {e}")
            return None
    
    async def clear_cache(self):
        """Clear all cached data"""
        try:
            cache_size = len(self.data_cache)
            self.data_cache.clear()
            logger.info(f"🧹 Cache cleared: {cache_size} entries removed")
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return {
                'cache_size': len(self.data_cache),
                'cache_keys': list(self.data_cache.keys()),
                'total_memory_mb': sum(
                    entry['data'].memory_usage(deep=True).sum() / 1024 / 1024
                    for entry in self.data_cache.values()
                ),
                'oldest_entry': min(
                    entry['timestamp'] for entry in self.data_cache.values()
                ) if self.data_cache else None,
                'newest_entry': max(
                    entry['timestamp'] for entry in self.data_cache.values()
                ) if self.data_cache else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            return {'error': str(e)}
    
    async def add_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced features for comprehensive retraining
        
        Args:
            data: Base training data
            
        Returns:
            DataFrame with advanced features added
        """
        try:
            logger.info("🔧 Adding advanced features for comprehensive retraining")
            
            # Add time-based features
            if 'timestamp' in data.columns:
                data = data.copy()
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                data['month'] = pd.to_datetime(data['timestamp']).dt.month
                data['quarter'] = pd.to_datetime(data['timestamp']).dt.quarter
                
                # Cyclical encoding for time features
                data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
                data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
                data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
                data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
                data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
                data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Add lag features for time series
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns[:10]:  # Limit to first 10 numeric columns
                if col not in ['hour', 'day_of_week', 'month', 'quarter']:
                    # Add lag 1 and lag 2
                    data[f'{col}_lag1'] = data[col].shift(1)
                    data[f'{col}_lag2'] = data[col].shift(2)
                    
                    # Add rolling statistics
                    data[f'{col}_rolling_mean_5'] = data[col].rolling(window=5).mean()
                    data[f'{col}_rolling_std_5'] = data[col].rolling(window=5).std()
                    data[f'{col}_rolling_min_5'] = data[col].rolling(window=5).min()
                    data[f'{col}_rolling_max_5'] = data[col].rolling(window=5).max()
            
            # Add interaction features
            if len(numeric_columns) >= 2:
                # Simple pairwise interactions for first few features
                for i, col1 in enumerate(numeric_columns[:3]):
                    for col2 in numeric_columns[i+1:4]:
                        if col1 != col2:
                            data[f'{col1}_{col2}_interaction'] = data[col1] * data[col2]
                            data[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-8)
            
            # Fill NaN values created by lag features
            data = data.fillna(method='bfill').fillna(method='ffill')
            
            logger.info(f"✅ Advanced features added: {data.shape[1]} total features")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error adding advanced features: {e}")
            return data

# Global instance
retraining_data_service = RetrainingDataService()
