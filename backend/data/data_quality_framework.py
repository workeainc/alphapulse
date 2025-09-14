"""
Data Quality Framework for AlphaPlus
Ensures data integrity, handles anomalies, and maintains data quality
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncpg
from dataclasses import dataclass
import json
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring"""
    completeness: float  # Percentage of non-null values
    accuracy: float      # Data accuracy score
    consistency: float   # Data consistency score
    timeliness: float    # Data freshness score
    validity: float      # Data validity score
    overall_score: float # Overall quality score

@dataclass
class DataAnomaly:
    """Data anomaly detection result"""
    timestamp: datetime
    symbol: str
    timeframe: str
    anomaly_type: str  # price_spike, volume_spike, missing_data, etc.
    severity: str      # low, medium, high, critical
    description: str
    suggested_action: str

class DataQualityFramework:
    """Comprehensive data quality management system"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Quality thresholds
        self.price_change_threshold = 0.15  # 15% price change threshold
        self.volume_spike_threshold = 5.0   # 5x volume increase threshold
        self.missing_data_threshold = 0.05  # 5% missing data threshold
        self.data_freshness_threshold = 300  # 5 minutes data freshness
        
    async def validate_data_integrity(self, data_points: List[Dict[str, Any]]) -> DataQualityMetrics:
        """Validate data integrity and calculate quality metrics"""
        try:
            if not data_points:
                return DataQualityMetrics(0, 0, 0, 0, 0, 0)
            
            df = pd.DataFrame(data_points)
            
            # Calculate completeness
            completeness = self._calculate_completeness(df)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(df)
            
            # Calculate consistency
            consistency = self._calculate_consistency(df)
            
            # Calculate timeliness
            timeliness = self._calculate_timeliness(df)
            
            # Calculate validity
            validity = self._calculate_validity(df)
            
            # Calculate overall score
            overall_score = (completeness + accuracy + consistency + timeliness + validity) / 5
            
            return DataQualityMetrics(
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                timeliness=timeliness,
                validity=validity,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"❌ Error validating data integrity: {e}")
            return DataQualityMetrics(0, 0, 0, 0, 0, 0)
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness score"""
        try:
            # Check for missing values in critical columns
            critical_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            missing_ratios = []
            
            for col in critical_columns:
                if col in df.columns:
                    missing_ratio = df[col].isnull().sum() / len(df)
                    missing_ratios.append(missing_ratio)
            
            if missing_ratios:
                avg_missing_ratio = np.mean(missing_ratios)
                completeness = 1.0 - avg_missing_ratio
                return max(0, min(1, completeness))
            
            return 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating completeness: {e}")
            return 0.0
    
    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """Calculate data accuracy score"""
        try:
            accuracy_scores = []
            
            # Check for logical price relationships
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= low
                high_low_valid = (df['high'] >= df['low']).sum() / len(df)
                accuracy_scores.append(high_low_valid)
                
                # High should be >= open and close
                high_valid = ((df['high'] >= df['open']) & (df['high'] >= df['close'])).sum() / len(df)
                accuracy_scores.append(high_valid)
                
                # Low should be <= open and close
                low_valid = ((df['low'] <= df['open']) & (df['low'] <= df['close'])).sum() / len(df)
                accuracy_scores.append(low_valid)
            
            # Check for positive volume
            if 'volume' in df.columns:
                volume_valid = (df['volume'] >= 0).sum() / len(df)
                accuracy_scores.append(volume_valid)
            
            if accuracy_scores:
                return np.mean(accuracy_scores)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating accuracy: {e}")
            return 0.0
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        try:
            consistency_scores = []
            
            # Check for consistent price movements
            if 'close' in df.columns and len(df) > 1:
                price_changes = df['close'].pct_change().abs()
                reasonable_changes = (price_changes <= self.price_change_threshold).sum() / len(price_changes)
                consistency_scores.append(reasonable_changes)
            
            # Check for consistent volume patterns
            if 'volume' in df.columns and len(df) > 1:
                volume_changes = df['volume'].pct_change().abs()
                reasonable_volume = (volume_changes <= self.volume_spike_threshold).sum() / len(volume_changes)
                consistency_scores.append(reasonable_volume)
            
            if consistency_scores:
                return np.mean(consistency_scores)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating consistency: {e}")
            return 0.0
    
    def _calculate_timeliness(self, df: pd.DataFrame) -> float:
        """Calculate data timeliness score"""
        try:
            if 'timestamp' not in df.columns:
                return 0.0
            
            # Check if timestamps are recent
            now = datetime.utcnow()
            time_diffs = []
            
            for timestamp in df['timestamp']:
                if pd.isna(timestamp):
                    time_diffs.append(self.data_freshness_threshold)
                else:
                    diff_seconds = (now - timestamp).total_seconds()
                    time_diffs.append(diff_seconds)
            
            if time_diffs:
                avg_time_diff = np.mean(time_diffs)
                timeliness = max(0, 1 - (avg_time_diff / self.data_freshness_threshold))
                return timeliness
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating timeliness: {e}")
            return 0.0
    
    def _calculate_validity(self, df: pd.DataFrame) -> float:
        """Calculate data validity score"""
        try:
            validity_scores = []
            
            # Check for valid price ranges
            if 'close' in df.columns:
                valid_prices = (df['close'] > 0).sum() / len(df)
                validity_scores.append(valid_prices)
            
            # Check for valid volume ranges
            if 'volume' in df.columns:
                valid_volumes = (df['volume'] >= 0).sum() / len(df)
                validity_scores.append(valid_volumes)
            
            # Check for valid timestamps
            if 'timestamp' in df.columns:
                valid_timestamps = df['timestamp'].notna().sum() / len(df)
                validity_scores.append(valid_timestamps)
            
            if validity_scores:
                return np.mean(validity_scores)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating validity: {e}")
            return 0.0
    
    async def detect_outliers(self, data_points: List[Dict[str, Any]]) -> List[DataAnomaly]:
        """Detect outliers and anomalies in market data"""
        try:
            if not data_points or len(data_points) < 10:
                return []
            
            df = pd.DataFrame(data_points)
            anomalies = []
            
            # Detect price outliers
            price_anomalies = self._detect_price_anomalies(df)
            anomalies.extend(price_anomalies)
            
            # Detect volume outliers
            volume_anomalies = self._detect_volume_anomalies(df)
            anomalies.extend(volume_anomalies)
            
            # Detect missing data patterns
            missing_anomalies = self._detect_missing_data_anomalies(df)
            anomalies.extend(missing_anomalies)
            
            # Detect temporal anomalies
            temporal_anomalies = self._detect_temporal_anomalies(df)
            anomalies.extend(temporal_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting outliers: {e}")
            return []
    
    def _detect_price_anomalies(self, df: pd.DataFrame) -> List[DataAnomaly]:
        """Detect price-related anomalies"""
        anomalies = []
        
        try:
            if 'close' not in df.columns or 'timestamp' not in df.columns:
                return anomalies
            
            # Calculate price changes
            price_changes = df['close'].pct_change().abs()
            
            # Use Z-score method for outlier detection
            z_scores = np.abs(stats.zscore(price_changes.dropna()))
            
            for i, (timestamp, change, z_score) in enumerate(zip(df['timestamp'], price_changes, z_scores)):
                if pd.isna(change) or pd.isna(z_score):
                    continue
                
                if z_score > 3.0:  # 3 standard deviations
                    severity = 'critical' if z_score > 5.0 else 'high' if z_score > 4.0 else 'medium'
                    
                    anomaly = DataAnomaly(
                        timestamp=timestamp,
                        symbol=df.get('symbol', ['Unknown'])[0] if 'symbol' in df.columns else 'Unknown',
                        timeframe=df.get('timeframe', ['Unknown'])[0] if 'timeframe' in df.columns else 'Unknown',
                        anomaly_type='price_spike',
                        severity=severity,
                        description=f"Price change of {change:.4f} ({change*100:.2f}%) detected (Z-score: {z_score:.2f})",
                        suggested_action="Verify data source and check for market events"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting price anomalies: {e}")
            return anomalies
    
    def _detect_volume_anomalies(self, df: pd.DataFrame) -> List[DataAnomaly]:
        """Detect volume-related anomalies"""
        anomalies = []
        
        try:
            if 'volume' not in df.columns or 'timestamp' not in df.columns:
                return anomalies
            
            # Calculate volume changes
            volume_changes = df['volume'].pct_change().abs()
            
            # Use Z-score method for outlier detection
            z_scores = np.abs(stats.zscore(volume_changes.dropna()))
            
            for i, (timestamp, change, z_score) in enumerate(zip(df['timestamp'], volume_changes, z_scores)):
                if pd.isna(change) or pd.isna(z_score):
                    continue
                
                if z_score > 3.0:  # 3 standard deviations
                    severity = 'critical' if z_score > 5.0 else 'high' if z_score > 4.0 else 'medium'
                    
                    anomaly = DataAnomaly(
                        timestamp=timestamp,
                        symbol=df.get('symbol', ['Unknown'])[0] if 'symbol' in df.columns else 'Unknown',
                        timeframe=df.get('timeframe', ['Unknown'])[0] if 'timeframe' in df.columns else 'Unknown',
                        anomaly_type='volume_spike',
                        severity=severity,
                        description=f"Volume change of {change:.4f} ({change*100:.2f}%) detected (Z-score: {z_score:.2f})",
                        suggested_action="Check for news events or market manipulation"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting volume anomalies: {e}")
            return anomalies
    
    def _detect_missing_data_anomalies(self, df: pd.DataFrame) -> List[DataAnomaly]:
        """Detect missing data patterns"""
        anomalies = []
        
        try:
            if 'timestamp' not in df.columns:
                return anomalies
            
            # Check for gaps in timestamps
            timestamps = pd.to_datetime(df['timestamp'])
            time_diffs = timestamps.diff()
            
            # Expected time differences based on timeframe (simplified)
            expected_diff = pd.Timedelta(minutes=1)  # Default to 1 minute
            
            for i, diff in enumerate(time_diffs):
                if pd.isna(diff):
                    continue
                
                if diff > expected_diff * 2:  # More than 2x expected gap
                    severity = 'high' if diff > expected_diff * 5 else 'medium'
                    
                    anomaly = DataAnomaly(
                        timestamp=timestamps.iloc[i],
                        symbol=df.get('symbol', ['Unknown'])[0] if 'symbol' in df.columns else 'Unknown',
                        timeframe=df.get('timeframe', ['Unknown'])[0] if 'timeframe' in df.columns else 'Unknown',
                        anomaly_type='missing_data',
                        severity=severity,
                        description=f"Data gap of {diff} detected",
                        suggested_action="Check data source connectivity and fill missing data"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting missing data anomalies: {e}")
            return anomalies
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame) -> List[DataAnomaly]:
        """Detect temporal anomalies (future timestamps, etc.)"""
        anomalies = []
        
        try:
            if 'timestamp' not in df.columns:
                return anomalies
            
            now = datetime.utcnow()
            
            for timestamp in df['timestamp']:
                if pd.isna(timestamp):
                    continue
                
                # Check for future timestamps
                if timestamp > now:
                    anomaly = DataAnomaly(
                        timestamp=timestamp,
                        symbol=df.get('symbol', ['Unknown'])[0] if 'symbol' in df.columns else 'Unknown',
                        timeframe=df.get('timeframe', ['Unknown'])[0] if 'timeframe' in df.columns else 'Unknown',
                        anomaly_type='future_timestamp',
                        severity='high',
                        description=f"Future timestamp detected: {timestamp}",
                        suggested_action="Check system clock and data source"
                    )
                    anomalies.append(anomaly)
                
                # Check for very old timestamps
                age = (now - timestamp).total_seconds()
                if age > 86400:  # More than 24 hours old
                    anomaly = DataAnomaly(
                        timestamp=timestamp,
                        symbol=df.get('symbol', ['Unknown'])[0] if 'symbol' in df.columns else 'Unknown',
                        timeframe=df.get('timeframe', ['Unknown'])[0] if 'timeframe' in df.columns else 'Unknown',
                        anomaly_type='stale_data',
                        severity='medium',
                        description=f"Stale data detected: {age/3600:.1f} hours old",
                        suggested_action="Check data source and update frequency"
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Error detecting temporal anomalies: {e}")
            return anomalies
    
    async def fill_missing_data(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fill missing data points intelligently"""
        try:
            if not data_points:
                return data_points
            
            df = pd.DataFrame(data_points)
            filled_df = df.copy()
            
            # Fill missing OHLCV data using forward fill and interpolation
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in ohlcv_columns:
                if col in filled_df.columns:
                    # Forward fill for short gaps
                    filled_df[col] = filled_df[col].fillna(method='ffill', limit=3)
                    
                    # Linear interpolation for longer gaps
                    filled_df[col] = filled_df[col].interpolate(method='linear', limit_direction='both')
            
            # Fill missing technical indicators
            tech_columns = ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'bollinger_middle', 'atr']
            
            for col in tech_columns:
                if col in filled_df.columns:
                    filled_df[col] = filled_df[col].interpolate(method='linear', limit_direction='both')
            
            # Convert back to list of dictionaries
            filled_data = filled_df.to_dict('records')
            
            logger.info(f"✅ Filled missing data: {len(filled_data)} points")
            return filled_data
            
        except Exception as e:
            logger.error(f"❌ Error filling missing data: {e}")
            return data_points
    
    async def store_data_quality_metrics(self, symbol: str, timeframe: str, metrics: DataQualityMetrics) -> bool:
        """Store data quality metrics in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO data_quality_metrics (
                        symbol, timeframe, completeness, accuracy, consistency, timeliness, validity, overall_score, timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                        completeness = EXCLUDED.completeness,
                        accuracy = EXCLUDED.accuracy,
                        consistency = EXCLUDED.consistency,
                        timeliness = EXCLUDED.timeliness,
                        validity = EXCLUDED.validity,
                        overall_score = EXCLUDED.overall_score
                """, symbol, timeframe, metrics.completeness, metrics.accuracy, 
                     metrics.consistency, metrics.timeliness, metrics.validity, 
                     metrics.overall_score, datetime.utcnow())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing data quality metrics: {e}")
            return False
    
    async def store_data_anomalies(self, anomalies: List[DataAnomaly]) -> bool:
        """Store detected anomalies in database"""
        try:
            if not anomalies:
                return True
            
            async with self.db_pool.acquire() as conn:
                for anomaly in anomalies:
                    await conn.execute("""
                        INSERT INTO data_anomalies (
                            symbol, timeframe, timestamp, anomaly_type, severity, description, suggested_action
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (symbol, timeframe, timestamp, anomaly_type) DO UPDATE SET
                            severity = EXCLUDED.severity,
                            description = EXCLUDED.description,
                            suggested_action = EXCLUDED.suggested_action
                    """, anomaly.symbol, anomaly.timeframe, anomaly.timestamp, 
                         anomaly.anomaly_type, anomaly.severity, anomaly.description, 
                         anomaly.suggested_action)
            
            logger.info(f"✅ Stored {len(anomalies)} data anomalies")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing data anomalies: {e}")
            return False
