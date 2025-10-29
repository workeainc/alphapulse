"""
Enhanced Regime Detection with Advanced K-Means Clustering
Advanced market regime classification with periodic updates and stability monitoring
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import threading
import json

# Machine learning imports
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MarketFeatures:
    """Market features for regime classification"""
    volume: float
    volatility: float
    trend_strength: float
    price_momentum: float
    volume_trend: float
    price_range: float
    rsi: float
    macd: float
    bollinger_position: float
    atr: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RegimeClassification:
    """Market regime classification result"""
    regime: str  # trending, ranging, volatile, breakout, consolidation
    confidence: float
    cluster_id: int
    distance_to_center: float
    features: MarketFeatures
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RegimeThresholds:
    """Regime-specific thresholds"""
    volume_threshold: float
    trend_threshold: float
    confidence_threshold: float
    risk_multiplier: float
    position_size_multiplier: float

class EnhancedRegimeDetector:
    """
    Enhanced market regime detection using advanced K-Means clustering.
    Features periodic model updates, stability monitoring, and regime-specific thresholds.
    """
    
    def __init__(self, 
                 n_clusters: int = 5,
                 update_interval: int = 3600,  # 1 hour
                 min_samples_for_training: int = 1000,
                 feature_window: int = 100,
                 stability_threshold: float = 0.6):
        
        self.n_clusters = n_clusters
        self.update_interval = update_interval
        self.min_samples_for_training = min_samples_for_training
        self.feature_window = feature_window
        self.stability_threshold = stability_threshold
        
        # Model components
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        
        # Data buffers
        self.feature_buffer = deque(maxlen=feature_window)
        self.regime_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
        # Regime labels and thresholds
        self.regime_labels = [
            'low_volatility_ranging',
            'normal_trending', 
            'high_volatility_breakout',
            'consolidation',
            'extreme_volatility'
        ]
        
        self.regime_thresholds = {
            'low_volatility_ranging': RegimeThresholds(
                volume_threshold=300.0,
                trend_threshold=0.3,
                confidence_threshold=0.5,
                risk_multiplier=0.8,
                position_size_multiplier=1.2
            ),
            'normal_trending': RegimeThresholds(
                volume_threshold=500.0,
                trend_threshold=0.5,
                confidence_threshold=0.6,
                risk_multiplier=1.0,
                position_size_multiplier=1.0
            ),
            'high_volatility_breakout': RegimeThresholds(
                volume_threshold=800.0,
                trend_threshold=0.7,
                confidence_threshold=0.8,
                risk_multiplier=1.5,
                position_size_multiplier=0.7
            ),
            'consolidation': RegimeThresholds(
                volume_threshold=200.0,
                trend_threshold=0.2,
                confidence_threshold=0.4,
                risk_multiplier=0.6,
                position_size_multiplier=1.5
            ),
            'extreme_volatility': RegimeThresholds(
                volume_threshold=1000.0,
                trend_threshold=0.8,
                confidence_threshold=0.9,
                risk_multiplier=2.0,
                position_size_multiplier=0.5
            )
        }
        
        # Performance tracking
        self.last_update = datetime.now()
        self.classification_count = 0
        self.model_stability_score = 0.0
        self.regime_stability_score = 0.0
        
        # Threading for background updates
        self.update_lock = threading.Lock()
        self.background_task = None
        
        logger.info(f"Enhanced Regime Detector initialized with {n_clusters} clusters")
    
    async def start(self):
        """Start the regime detector with background updates"""
        if self.background_task is None:
            self.background_task = asyncio.create_task(self._background_update_loop())
            logger.info("Enhanced Regime Detector started")
    
    async def stop(self):
        """Stop the regime detector"""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Enhanced Regime Detector stopped")
    
    def add_market_data(self, 
                       prices: List[float], 
                       volumes: List[float], 
                       indicators: Optional[Dict[str, List[float]]] = None):
        """Add market data for feature extraction"""
        if len(prices) < 20 or len(volumes) < 20:
            return
        
        try:
            # Extract features
            features = self._extract_features(prices, volumes, indicators)
            
            # Add to buffer
            self.feature_buffer.append(features)
            
            # Trigger model update if needed
            if len(self.feature_buffer) >= self.min_samples_for_training:
                asyncio.create_task(self._check_and_update_model())
                
        except Exception as e:
            logger.error(f"Error adding market data: {e}")
    
    def classify_regime(self, 
                       prices: List[float], 
                       volumes: List[float], 
                       indicators: Optional[Dict[str, List[float]]] = None) -> RegimeClassification:
        """Classify current market regime"""
        try:
            # Extract features
            features = self._extract_features(prices, volumes, indicators)
            
            # Add to buffer
            self.feature_buffer.append(features)
            
            # Classify if model is available
            if self.kmeans is not None and len(self.feature_buffer) >= 10:
                return self._classify_with_model(features)
            else:
                # Fallback classification
                return self._fallback_classification(features)
                
        except Exception as e:
            logger.error(f"Regime classification error: {e}")
            return self._fallback_classification(None)
    
    def _extract_features(self, 
                         prices: List[float], 
                         volumes: List[float], 
                         indicators: Optional[Dict[str, List[float]]] = None) -> MarketFeatures:
        """Extract comprehensive market features"""
        if len(prices) < 20:
            # Return default features if insufficient data
            return MarketFeatures(
                volume=500.0,
                volatility=0.02,
                trend_strength=0.5,
                price_momentum=0.0,
                volume_trend=1.0,
                price_range=0.02,
                rsi=50.0,
                macd=0.0,
                bollinger_position=0.5,
                atr=0.02
            )
        
        # Calculate basic features
        recent_prices = prices[-20:]
        recent_volumes = volumes[-20:]
        
        # Volume features
        volume = np.mean(recent_volumes)
        volume_trend = np.mean(recent_volumes[-10:]) / np.mean(recent_volumes[-20:-10]) if len(recent_volumes) >= 20 else 1.0
        
        # Volatility features
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.02
        
        # Trend features
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0.0
        trend_strength = abs(price_momentum) / volatility if volatility > 0 else 0.5
        
        # Price range
        price_range = (max(recent_prices) - min(recent_prices)) / np.mean(recent_prices)
        
        # Technical indicators (if available)
        rsi = indicators.get('rsi', [50.0])[-1] if indicators and 'rsi' in indicators else 50.0
        macd = indicators.get('macd', [0.0])[-1] if indicators and 'macd' in indicators else 0.0
        bollinger_position = indicators.get('bb_position', [0.5])[-1] if indicators and 'bb_position' in indicators else 0.5
        atr = indicators.get('atr', [0.02])[-1] if indicators and 'atr' in indicators else 0.02
        
        return MarketFeatures(
            volume=volume,
            volatility=volatility,
            trend_strength=trend_strength,
            price_momentum=price_momentum,
            volume_trend=volume_trend,
            price_range=price_range,
            rsi=rsi,
            macd=macd,
            bollinger_position=bollinger_position,
            atr=atr
        )
    
    def _classify_with_model(self, features: MarketFeatures) -> RegimeClassification:
        """Classify regime using trained K-Means model"""
        try:
            # Prepare feature vector
            feature_vector = self._features_to_vector(features)
            
            # Scale features
            scaled_features = self.scaler.transform([feature_vector])
            
            # Apply PCA if available
            if self.pca is not None:
                scaled_features = self.pca.transform(scaled_features)
            
            # Predict cluster
            cluster_id = self.kmeans.predict(scaled_features)[0]
            
            # Calculate distance to cluster center
            distance_to_center = np.linalg.norm(
                scaled_features[0] - self.kmeans.cluster_centers_[cluster_id]
            )
            
            # Get regime label
            regime = self.regime_labels[cluster_id] if cluster_id < len(self.regime_labels) else 'normal_trending'
            
            # Calculate confidence based on distance
            confidence = max(0.1, 1.0 - distance_to_center / 10.0)
            
            # Update history
            self.regime_history.append({
                'regime': regime,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
            
            self.classification_count += 1
            
            return RegimeClassification(
                regime=regime,
                confidence=confidence,
                cluster_id=cluster_id,
                distance_to_center=distance_to_center,
                features=features,
                metadata={
                    'model_stability': self.model_stability_score,
                    'regime_stability': self.regime_stability_score
                }
            )
            
        except Exception as e:
            logger.error(f"Model classification error: {e}")
            return self._fallback_classification(features)
    
    def _fallback_classification(self, features: Optional[MarketFeatures]) -> RegimeClassification:
        """Fallback classification when model is not available"""
        if features is None:
            features = MarketFeatures(
                volume=500.0,
                volatility=0.02,
                trend_strength=0.5,
                price_momentum=0.0,
                volume_trend=1.0,
                price_range=0.02,
                rsi=50.0,
                macd=0.0,
                bollinger_position=0.5,
                atr=0.02
            )
        
        # Simple rule-based classification
        if features.volatility > 0.05:
            regime = 'extreme_volatility'
        elif features.volatility > 0.03:
            regime = 'high_volatility_breakout'
        elif features.trend_strength > 0.7:
            regime = 'normal_trending'
        elif features.volatility < 0.015:
            regime = 'low_volatility_ranging'
        else:
            regime = 'consolidation'
        
        return RegimeClassification(
            regime=regime,
            confidence=0.5,
            cluster_id=-1,
            distance_to_center=0.0,
            features=features,
            metadata={'method': 'fallback'}
        )
    
    def _features_to_vector(self, features: MarketFeatures) -> List[float]:
        """Convert features to vector for model input"""
        return [
            features.volume,
            features.volatility,
            features.trend_strength,
            features.price_momentum,
            features.volume_trend,
            features.price_range,
            features.rsi / 100.0,  # Normalize RSI to [0,1]
            features.macd,
            features.bollinger_position,
            features.atr
        ]
    
    async def _check_and_update_model(self):
        """Check if model update is needed and perform update"""
        if (datetime.now() - self.last_update).seconds < self.update_interval:
            return
        
        if len(self.feature_buffer) < self.min_samples_for_training:
            return
        
        await self._update_model()
    
    async def _update_model(self):
        """Update the K-Means model with new data"""
        with self.update_lock:
            try:
                logger.info("Updating regime detection model...")
                
                # Prepare training data
                feature_vectors = []
                for features in list(self.feature_buffer):
                    feature_vectors.append(self._features_to_vector(features))
                
                if len(feature_vectors) < self.min_samples_for_training:
                    logger.warning("Insufficient data for model update")
                    return
                
                # Convert to numpy array
                X = np.array(feature_vectors)
                
                # Scale features
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                
                # Apply PCA for dimensionality reduction
                if X_scaled.shape[1] > 5:
                    self.pca = PCA(n_components=5, random_state=42)
                    X_scaled = self.pca.fit_transform(X_scaled)
                
                # Train K-Means
                self.kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
                self.kmeans.fit(X_scaled)
                
                # Calculate model stability
                self.model_stability_score = self._calculate_model_stability(X_scaled)
                
                # Calculate regime stability
                self.regime_stability_score = self._calculate_regime_stability()
                
                self.last_update = datetime.now()
                
                logger.info(f"Model updated successfully. Stability: {self.model_stability_score:.3f}")
                
            except Exception as e:
                logger.error(f"Model update error: {e}")
    
    def _calculate_model_stability(self, X_scaled: np.ndarray) -> float:
        """Calculate model stability using silhouette score"""
        try:
            if self.kmeans is None:
                return 0.0
            
            labels = self.kmeans.labels_
            
            # Silhouette score (higher is better)
            silhouette = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0.0
            
            # Calinski-Harabasz score (higher is better)
            calinski = calinski_harabasz_score(X_scaled, labels) if len(np.unique(labels)) > 1 else 0.0
            
            # Normalize scores
            silhouette_norm = max(0.0, min(1.0, silhouette))
            calinski_norm = max(0.0, min(1.0, calinski / 1000.0))  # Normalize to [0,1]
            
            # Combined stability score
            stability = (silhouette_norm + calinski_norm) / 2.0
            
            return stability
            
        except Exception as e:
            logger.error(f"Stability calculation error: {e}")
            return 0.0
    
    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability based on recent classifications"""
        try:
            if len(self.regime_history) < 10:
                return 0.0
            
            # Get recent regimes
            recent_regimes = [entry['regime'] for entry in list(self.regime_history)[-20:]]
            
            # Calculate regime change frequency
            changes = sum(1 for i in range(1, len(recent_regimes)) 
                         if recent_regimes[i] != recent_regimes[i-1])
            
            # Stability is inverse of change frequency
            stability = 1.0 - (changes / max(1, len(recent_regimes) - 1))
            
            return stability
            
        except Exception as e:
            logger.error(f"Regime stability calculation error: {e}")
            return 0.0
    
    async def _background_update_loop(self):
        """Background task for periodic model updates"""
        while True:
            try:
                await asyncio.sleep(self.update_interval)
                await self._check_and_update_model()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background update error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_regime_thresholds(self, regime: str) -> RegimeThresholds:
        """Get regime-specific thresholds"""
        return self.regime_thresholds.get(regime, self.regime_thresholds['normal_trending'])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'classification_count': self.classification_count,
            'model_stability_score': self.model_stability_score,
            'regime_stability_score': self.regime_stability_score,
            'feature_buffer_size': len(self.feature_buffer),
            'regime_history_size': len(self.regime_history),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'model_available': self.kmeans is not None,
            'regime_distribution': self._get_regime_distribution()
        }
    
    def _get_regime_distribution(self) -> Dict[str, int]:
        """Get distribution of recent regime classifications"""
        if len(self.regime_history) == 0:
            return {}
        
        recent_regimes = [entry['regime'] for entry in list(self.regime_history)[-100:]]
        distribution = {}
        
        for regime in recent_regimes:
            distribution[regime] = distribution.get(regime, 0) + 1
        
        return distribution

# Global enhanced regime detector instance
enhanced_regime_detector = EnhancedRegimeDetector(
    n_clusters=5,
    update_interval=3600,
    min_samples_for_training=1000,
    feature_window=100,
    stability_threshold=0.6
)
