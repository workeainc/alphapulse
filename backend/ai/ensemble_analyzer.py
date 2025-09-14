"""
Ensemble Analyzer for AlphaPulse
Phase 4: Automated ensemble methods for log analysis and anomaly detection
"""

import asyncio
import logging
import time
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from datetime import datetime, timedelta

# Machine learning imports
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Scikit-learn not available - ML features disabled")

logger = logging.getLogger(__name__)

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    features: List[float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleAnalysis:
    """Ensemble analysis result"""
    prediction: str
    confidence: float
    individual_predictions: Dict[str, float]
    ensemble_method: str
    features_used: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnsembleAnalyzer:
    """Ensemble methods for log analysis and anomaly detection"""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 contamination: float = 0.1,
                 random_state: int = 42):
        
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        
        self.anomaly_detector = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.vectorizer = None
        
        self.training_data = []
        self.predictions_history = deque(maxlen=1000)
        
        if ML_AVAILABLE:
            self._initialize_models()
        
        logger.info("Ensemble Analyzer initialized")
    
    def _initialize_models(self):
        """Initialize ensemble models"""
        try:
            # Anomaly detection ensemble
            self.anomaly_detector = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state
            )
            
            # Classification ensemble
            rf1 = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            rf2 = RandomForestClassifier(n_estimators=100, random_state=self.random_state + 1)
            
            self.classifier = VotingClassifier(
                estimators=[('rf1', rf1), ('rf2', rf2)],
                voting='soft'
            )
            
            # Text vectorizer for log messages
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("✅ Ensemble models initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
    
    def extract_features(self, log_data: Dict[str, Any]) -> List[float]:
        """Extract features from log data"""
        features = []
        
        try:
            # Time-based features
            timestamp = log_data.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            features.extend([hour, day_of_week])
            
            # Event type encoding
            event_type = log_data.get('event_type', 'unknown')
            event_type_encoding = {
                'signal_generated': 1,
                'signal_validated': 2,
                'signal_rejected': 3,
                'trade_executed': 4,
                'trade_closed': 5,
                'system_error': 6,
                'performance_update': 7,
                'model_update': 8
            }
            features.append(event_type_encoding.get(event_type, 0))
            
            # Log level encoding
            log_level = log_data.get('log_level', 'info')
            log_level_encoding = {
                'debug': 1,
                'info': 2,
                'warning': 3,
                'error': 4,
                'critical': 5
            }
            features.append(log_level_encoding.get(log_level, 2))
            
            # Data-based features
            data = log_data.get('data', {})
            if data:
                # Extract numeric values
                numeric_values = []
                for value in data.values():
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                    elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                        numeric_values.append(float(value))
                
                if numeric_values:
                    features.extend([
                        np.mean(numeric_values),
                        np.std(numeric_values),
                        len(numeric_values)
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
                
                # Data size features
                features.append(len(str(data)))
            else:
                features.extend([0.0, 0.0, 0.0, 0])
            
            # Metadata features
            metadata = log_data.get('metadata', {})
            features.append(len(metadata))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return [0.0] * 10  # Default features
    
    async def detect_anomaly(self, log_data: Dict[str, Any]) -> AnomalyDetection:
        """Detect anomalies in log data"""
        if self.anomaly_detector is None:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.5,
                features=[],
                timestamp=datetime.now()
            )
        
        try:
            # Extract features
            features = self.extract_features(log_data)
            
            # Check if anomaly detector is fitted
            if not hasattr(self.anomaly_detector, 'estimators_'):
                # Return default result if not fitted
                return AnomalyDetection(
                    is_anomaly=False,
                    anomaly_score=0.0,
                    confidence=0.5,
                    features=features,
                    timestamp=datetime.now()
                )
            
            # Normalize features
            features_scaled = self.scaler.fit_transform([features])[0]
            
            # Detect anomaly
            anomaly_score = self.anomaly_detector.decision_function([features_scaled])[0]
            is_anomaly = self.anomaly_detector.predict([features_scaled])[0] == -1
            
            # Calculate confidence based on score
            confidence = 1.0 - abs(anomaly_score)
            
            return AnomalyDetection(
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                confidence=confidence,
                features=features,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.5,
                features=[],
                timestamp=datetime.now()
            )
    
    async def classify_event(self, log_data: Dict[str, Any]) -> EnsembleAnalysis:
        """Classify events using ensemble methods"""
        if self.classifier is None:
            return EnsembleAnalysis(
                prediction="unknown",
                confidence=0.5,
                individual_predictions={},
                ensemble_method="none",
                features_used=[],
                timestamp=datetime.now()
            )
        
        try:
            # Extract features
            features = self.extract_features(log_data)
            features_scaled = self.scaler.fit_transform([features])[0]
            
            # Check if classifier is fitted
            if not hasattr(self.classifier, 'estimators_'):
                return EnsembleAnalysis(
                    prediction="unknown",
                    confidence=0.5,
                    individual_predictions={},
                    ensemble_method="not_fitted",
                    features_used=[f"feature_{i}" for i in range(len(features))],
                    timestamp=datetime.now()
                )
            
            # Get predictions from all models
            individual_predictions = {}
            try:
                # Check if classifier has been fitted
                if hasattr(self.classifier, 'estimators_'):
                    for i, (name, model) in enumerate(self.classifier.estimators_):
                        try:
                            pred = model.predict([features_scaled])[0]
                            prob = model.predict_proba([features_scaled])[0].max()
                            individual_predictions[name] = {'prediction': pred, 'confidence': prob}
                        except Exception as e:
                            logger.warning(f"Model {name} prediction failed: {e}")
                else:
                    logger.warning("Classifier not fitted yet - skipping individual predictions")
            except Exception as e:
                logger.warning(f"Individual predictions failed: {e}")
            
            # Ensemble prediction
            ensemble_pred = self.classifier.predict([features_scaled])[0]
            ensemble_prob = self.classifier.predict_proba([features_scaled])[0].max()
            
            return EnsembleAnalysis(
                prediction=str(ensemble_pred),
                confidence=ensemble_prob,
                individual_predictions=individual_predictions,
                ensemble_method="voting",
                features_used=[f"feature_{i}" for i in range(len(features))],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error classifying event: {e}")
            return EnsembleAnalysis(
                prediction="error",
                confidence=0.0,
                individual_predictions={},
                ensemble_method="error",
                features_used=[],
                timestamp=datetime.now()
            )
    
    def add_training_data(self, log_data: Dict[str, Any], label: str):
        """Add training data for model updates"""
        if not ML_AVAILABLE:
            return
        
        try:
            features = self.extract_features(log_data)
            self.training_data.append({
                'features': features,
                'label': label,
                'timestamp': datetime.now()
            })
            
            # Keep only recent data
            if len(self.training_data) > 10000:
                self.training_data = self.training_data[-5000:]
                
        except Exception as e:
            logger.error(f"Error adding training data: {e}")
    
    async def retrain_models(self) -> Dict[str, Any]:
        """Retrain ensemble models"""
        if not ML_AVAILABLE or len(self.training_data) < 100:
            return {'status': 'insufficient_data'}
        
        try:
            # Prepare training data
            X = [item['features'] for item in self.training_data]
            y = [item['label'] for item in self.training_data]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            # Retrain models
            start_time = time.time()
            
            # Retrain anomaly detector
            self.anomaly_detector.fit(X_train)
            
            # Retrain classifier
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            training_time = time.time() - start_time
            
            logger.info(f"✅ Models retrained - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'samples_used': len(self.training_data)
            }
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            'training_data_size': len(self.training_data),
            'predictions_history_size': len(self.predictions_history),
            'anomaly_detector_available': self.anomaly_detector is not None,
            'classifier_available': self.classifier is not None,
            'ml_available': ML_AVAILABLE
        }

# Global instance
ensemble_analyzer = EnsembleAnalyzer(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)
