import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncpg

# Try to import ML libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class AnomalyType(Enum):
    MANIPULATION = "manipulation"
    NEWS_EVENT = "news_event"
    TECHNICAL_ANOMALY = "technical_anomaly"

class DetectionMethod(Enum):
    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL = "statistical"
    AUTOENCODER = "autoencoder"

class SeverityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnomalyResult:
    anomaly_type: AnomalyType
    anomaly_score: float
    confidence_score: float
    detection_method: DetectionMethod
    severity_level: SeverityLevel
    anomaly_features: Dict
    metadata: Dict

class AnomalyDetectionService:
    """Service for detecting trading anomalies including manipulation, news events, and technical anomalies"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Anomaly detection parameters
        self.manipulation_thresholds = {
            'volume_spike_threshold': 5.0,  # 5x average volume
            'price_impact_threshold': 0.02,  # 2% price change
            'order_imbalance_threshold': 0.8,  # 80% imbalance
            'time_window_minutes': 5
        }
        
        self.news_event_thresholds = {
            'volatility_spike_threshold': 3.0,  # 3x average volatility
            'volume_surge_threshold': 3.0,  # 3x average volume
            'price_gap_threshold': 0.01,  # 1% price gap
            'recovery_time_minutes': 30
        }
        
        self.technical_anomaly_thresholds = {
            'z_score_threshold': 3.0,  # 3 standard deviations
            'percentile_threshold': 0.99,  # 99th percentile
            'rolling_window': 100
        }
        
        # Initialize ML models if available
        if ML_AVAILABLE:
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
            self.models_trained = False
        else:
            self.logger.warning("⚠️ ML libraries not available - using statistical methods only")
        
        self.logger.info("🔍 Anomaly Detection Service initialized")
    
    async def detect_anomalies(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]) -> List[AnomalyResult]:
        """Detect anomalies in the given OHLCV data"""
        try:
            if not ohlcv_data or len(ohlcv_data) < 10:
                return []
            
            df = pd.DataFrame(ohlcv_data)
            anomalies = []
            
            # Detect manipulation
            manipulation_anomalies = await self._detect_manipulation(symbol, timeframe, df)
            anomalies.extend(manipulation_anomalies)
            
            # Detect news events
            news_anomalies = await self._detect_news_events(symbol, timeframe, df)
            anomalies.extend(news_anomalies)
            
            # Detect technical anomalies
            technical_anomalies = await self._detect_technical_anomalies(symbol, timeframe, df)
            anomalies.extend(technical_anomalies)
            
            # Store anomalies in database
            if anomalies:
                await self._store_anomalies(symbol, timeframe, anomalies)
            
            self.logger.info(f"🔍 Detected {len(anomalies)} anomalies for {symbol}")
            return anomalies
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting anomalies: {e}")
            return []
    
    async def _detect_manipulation(self, symbol: str, timeframe: str, df: pd.DataFrame) -> List[AnomalyResult]:
        """Detect potential market manipulation"""
        anomalies = []
        
        try:
            # Calculate manipulation indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_change'] = (df['close'] - df['open']) / df['open']
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            
            # Detect volume spikes with minimal price impact (potential spoofing)
            volume_spikes = df[
                (df['volume_ratio'] > self.manipulation_thresholds['volume_spike_threshold']) &
                (abs(df['price_change']) < self.manipulation_thresholds['price_impact_threshold'])
            ]
            
            for idx, row in volume_spikes.iterrows():
                anomaly_score = min(row['volume_ratio'] / 10.0, 1.0)  # Normalize to 0-1
                confidence_score = 0.7 if row['volume_ratio'] > 10 else 0.5
                
                anomaly = AnomalyResult(
                    anomaly_type=AnomalyType.MANIPULATION,
                    anomaly_score=anomaly_score,
                    confidence_score=confidence_score,
                    detection_method=DetectionMethod.STATISTICAL,
                    severity_level=self._get_severity_level(anomaly_score),
                    anomaly_features={
                        'volume_ratio': float(row['volume_ratio']),
                        'price_change': float(row['price_change']),
                        'high_low_ratio': float(row['high_low_ratio']),
                        'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                    },
                    metadata={
                        'detection_reason': 'volume_spike_with_minimal_price_impact',
                        'threshold_exceeded': self.manipulation_thresholds['volume_spike_threshold']
                    }
                )
                anomalies.append(anomaly)
            
            # Detect price manipulation patterns
            price_manipulation = df[
                (abs(df['price_change']) > self.manipulation_thresholds['price_impact_threshold']) &
                (df['volume_ratio'] < 0.5)  # Low volume price moves
            ]
            
            for idx, row in price_manipulation.iterrows():
                anomaly_score = min(abs(row['price_change']) / 0.05, 1.0)  # Normalize to 0-1
                confidence_score = 0.6
                
                anomaly = AnomalyResult(
                    anomaly_type=AnomalyType.MANIPULATION,
                    anomaly_score=anomaly_score,
                    confidence_score=confidence_score,
                    detection_method=DetectionMethod.STATISTICAL,
                    severity_level=self._get_severity_level(anomaly_score),
                    anomaly_features={
                        'price_change': float(row['price_change']),
                        'volume_ratio': float(row['volume_ratio']),
                        'high_low_ratio': float(row['high_low_ratio']),
                        'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                    },
                    metadata={
                        'detection_reason': 'price_manipulation_with_low_volume',
                        'threshold_exceeded': self.manipulation_thresholds['price_impact_threshold']
                    }
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"❌ Error in manipulation detection: {e}")
        
        return anomalies
    
    async def _detect_news_events(self, symbol: str, timeframe: str, df: pd.DataFrame) -> List[AnomalyResult]:
        """Detect potential news-driven events"""
        anomalies = []
        
        try:
            # Calculate volatility and volume metrics
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['volatility_ma'] = df['volatility'].rolling(window=50).mean()
            df['volatility_ratio'] = df['volatility'] / df['volatility_ma']
            
            # Detect volatility spikes with volume surges
            news_events = df[
                (df['volatility_ratio'] > self.news_event_thresholds['volatility_spike_threshold']) &
                (df['volume_ratio'] > self.news_event_thresholds['volume_surge_threshold'])
            ]
            
            for idx, row in news_events.iterrows():
                anomaly_score = min(
                    (row['volatility_ratio'] + row['volume_ratio']) / 10.0, 
                    1.0
                )
                confidence_score = 0.8 if row['volatility_ratio'] > 5 else 0.6
                
                anomaly = AnomalyResult(
                    anomaly_type=AnomalyType.NEWS_EVENT,
                    anomaly_score=anomaly_score,
                    confidence_score=confidence_score,
                    detection_method=DetectionMethod.STATISTICAL,
                    severity_level=self._get_severity_level(anomaly_score),
                    anomaly_features={
                        'volatility_ratio': float(row['volatility_ratio']),
                        'volume_ratio': float(row['volume_ratio']),
                        'returns': float(row['returns']) if not pd.isna(row['returns']) else 0.0,
                        'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                    },
                    metadata={
                        'detection_reason': 'volatility_spike_with_volume_surge',
                        'threshold_exceeded': self.news_event_thresholds['volatility_spike_threshold']
                    }
                )
                anomalies.append(anomaly)
            
            # Detect price gaps (potential news gaps)
            df['price_gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            price_gaps = df[
                (df['price_gap'] > self.news_event_thresholds['price_gap_threshold']) &
                (df['volume_ratio'] > 2.0)  # Above average volume
            ]
            
            for idx, row in price_gaps.iterrows():
                anomaly_score = min(row['price_gap'] / 0.05, 1.0)
                confidence_score = 0.7
                
                anomaly = AnomalyResult(
                    anomaly_type=AnomalyType.NEWS_EVENT,
                    anomaly_score=anomaly_score,
                    confidence_score=confidence_score,
                    detection_method=DetectionMethod.STATISTICAL,
                    severity_level=self._get_severity_level(anomaly_score),
                    anomaly_features={
                        'price_gap': float(row['price_gap']),
                        'volume_ratio': float(row['volume_ratio']),
                        'returns': float(row['returns']) if not pd.isna(row['returns']) else 0.0,
                        'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                    },
                    metadata={
                        'detection_reason': 'price_gap_with_volume',
                        'threshold_exceeded': self.news_event_thresholds['price_gap_threshold']
                    }
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"❌ Error in news event detection: {e}")
        
        return anomalies
    
    async def _detect_technical_anomalies(self, symbol: str, timeframe: str, df: pd.DataFrame) -> List[AnomalyResult]:
        """Detect technical anomalies using statistical methods and ML"""
        anomalies = []
        
        try:
            # Statistical anomaly detection
            statistical_anomalies = await self._detect_statistical_anomalies(df)
            anomalies.extend(statistical_anomalies)
            
            # ML-based anomaly detection
            if ML_AVAILABLE and len(df) >= 50:
                ml_anomalies = await self._detect_ml_anomalies(df)
                anomalies.extend(ml_anomalies)
            
        except Exception as e:
            self.logger.error(f"❌ Error in technical anomaly detection: {e}")
        
        return anomalies
    
    async def _detect_statistical_anomalies(self, df: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        try:
            # Z-score based detection
            for column in ['volume', 'close', 'high_low_ratio']:
                if column in df.columns:
                    z_scores = np.abs((df[column] - df[column].rolling(window=self.technical_anomaly_thresholds['rolling_window']).mean()) / 
                                    df[column].rolling(window=self.technical_anomaly_thresholds['rolling_window']).std())
                    
                    anomaly_indices = z_scores > self.technical_anomaly_thresholds['z_score_threshold']
                    
                    for idx in df[anomaly_indices].index:
                        row = df.loc[idx]
                        anomaly_score = min(z_scores.loc[idx] / 5.0, 1.0)  # Normalize to 0-1
                        confidence_score = 0.6
                        
                        anomaly = AnomalyResult(
                            anomaly_type=AnomalyType.TECHNICAL_ANOMALY,
                            anomaly_score=anomaly_score,
                            confidence_score=confidence_score,
                            detection_method=DetectionMethod.STATISTICAL,
                            severity_level=self._get_severity_level(anomaly_score),
                            anomaly_features={
                                'metric': column,
                                'z_score': float(z_scores.loc[idx]),
                                'value': float(row[column]),
                                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                            },
                            metadata={
                                'detection_reason': f'z_score_anomaly_{column}',
                                'threshold_exceeded': self.technical_anomaly_thresholds['z_score_threshold']
                            }
                        )
                        anomalies.append(anomaly)
            
            # Percentile based detection
            for column in ['volume', 'close']:
                if column in df.columns:
                    rolling_percentile = df[column].rolling(window=self.technical_anomaly_thresholds['rolling_window']).quantile(
                        self.technical_anomaly_thresholds['percentile_threshold']
                    )
                    
                    anomaly_indices = df[column] > rolling_percentile
                    
                    for idx in df[anomaly_indices].index:
                        row = df.loc[idx]
                        percentile_rank = (df[column] <= row[column]).rolling(window=self.technical_anomaly_thresholds['rolling_window']).mean().loc[idx]
                        anomaly_score = percentile_rank
                        confidence_score = 0.5
                        
                        anomaly = AnomalyResult(
                            anomaly_type=AnomalyType.TECHNICAL_ANOMALY,
                            anomaly_score=anomaly_score,
                            confidence_score=confidence_score,
                            detection_method=DetectionMethod.STATISTICAL,
                            severity_level=self._get_severity_level(anomaly_score),
                            anomaly_features={
                                'metric': column,
                                'percentile_rank': float(percentile_rank),
                                'value': float(row[column]),
                                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                            },
                            metadata={
                                'detection_reason': f'percentile_anomaly_{column}',
                                'threshold_exceeded': self.technical_anomaly_thresholds['percentile_threshold']
                            }
                        )
                        anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"❌ Error in statistical anomaly detection: {e}")
        
        return anomalies
    
    async def _detect_ml_anomalies(self, df: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies using ML models"""
        anomalies = []
        
        try:
            # Prepare features for ML
            features = ['volume', 'close', 'high', 'low', 'open']
            available_features = [f for f in features if f in df.columns]
            
            if len(available_features) < 2:
                return anomalies
            
            # Create feature matrix
            X = df[available_features].ffill().fillna(0)
            
            # Train model if not trained
            if not self.models_trained:
                X_scaled = self.scaler.fit_transform(X)
                self.isolation_forest.fit(X_scaled)
                self.models_trained = True
            
            # Predict anomalies
            X_scaled = self.scaler.transform(X)
            predictions = self.isolation_forest.predict(X_scaled)
            scores = self.isolation_forest.decision_function(X_scaled)
            
            # Convert scores to anomaly scores (0-1 scale)
            anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            
            # Find anomalies (predictions == -1)
            anomaly_indices = predictions == -1
            
            for idx in df[anomaly_indices].index:
                row = df.loc[idx]
                anomaly_score = float(anomaly_scores[df.index.get_loc(idx)])
                confidence_score = 0.7
                
                anomaly = AnomalyResult(
                    anomaly_type=AnomalyType.TECHNICAL_ANOMALY,
                    anomaly_score=anomaly_score,
                    confidence_score=confidence_score,
                    detection_method=DetectionMethod.ISOLATION_FOREST,
                    severity_level=self._get_severity_level(anomaly_score),
                    anomaly_features={
                        'ml_score': float(scores[df.index.get_loc(idx)]),
                        'features_used': available_features,
                        'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
                    },
                    metadata={
                        'detection_reason': 'ml_isolation_forest',
                        'model_type': 'IsolationForest'
                    }
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"❌ Error in ML anomaly detection: {e}")
        
        return anomalies
    
    def _get_severity_level(self, anomaly_score: float) -> SeverityLevel:
        """Determine severity level based on anomaly score"""
        if anomaly_score >= 0.8:
            return SeverityLevel.CRITICAL
        elif anomaly_score >= 0.6:
            return SeverityLevel.HIGH
        elif anomaly_score >= 0.4:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    async def _store_anomalies(self, symbol: str, timeframe: str, anomalies: List[AnomalyResult]):
        """Store detected anomalies in the database"""
        try:
            async with self.db_pool.acquire() as conn:
                for anomaly in anomalies:
                    await conn.execute("""
                        INSERT INTO anomaly_detection (
                            symbol, timeframe, timestamp, anomaly_type, anomaly_score,
                            confidence_score, anomaly_features, detection_method,
                            severity_level, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """, 
                    symbol, timeframe, datetime.now(), anomaly.anomaly_type.value,
                    anomaly.anomaly_score, anomaly.confidence_score, anomaly.anomaly_features,
                    anomaly.detection_method.value, anomaly.severity_level.value, anomaly.metadata
                    )
            
            self.logger.info(f"💾 Stored {len(anomalies)} anomalies for {symbol}")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing anomalies: {e}")
    
    async def get_recent_anomalies(self, symbol: str, timeframe: str, hours: int = 24) -> List[Dict]:
        """Get recent anomalies for a symbol"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM anomaly_detection 
                    WHERE symbol = $1 AND timeframe = $2 
                    AND timestamp >= NOW() - INTERVAL '1 hour' * $3
                    ORDER BY timestamp DESC
                """, symbol, timeframe, hours)
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching recent anomalies: {e}")
            return []
