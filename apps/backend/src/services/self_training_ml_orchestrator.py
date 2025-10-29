#!/usr/bin/env python3
"""
Self-Training ML Orchestrator
Coordinates the entire self-training ML system for news impact prediction
"""

import asyncio
import logging
import asyncpg
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb

from .auto_labeling_service import AutoLabelingService, LabelingConfig
from .feature_engineering_service_simple import SimpleFeatureEngineeringService as FeatureEngineeringService, FeatureConfig

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    # Model types
    model_types: List[str] = None
    
    # Training parameters
    test_size: float = 0.2
    random_state: int = 42
    min_samples_for_training: int = 100
    
    # Model-specific parameters
    lgb_params: Dict[str, Any] = None
    xgb_params: Dict[str, Any] = None
    rf_params: Dict[str, Any] = None
    lr_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['lightgbm', 'xgboost', 'random_forest', 'logistic_regression']
        
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
        
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        
        if self.lr_params is None:
            self.lr_params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }

@dataclass
class TrainingResult:
    """Result of model training"""
    model_name: str
    target_variable: str
    model_type: str
    model_version: str
    
    # Training metrics
    training_samples: int
    validation_samples: int
    test_samples: int
    
    # Performance metrics
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    auc_score: float
    
    # Model details
    hyperparameters: Dict[str, Any]
    feature_importance: Dict[str, float]
    training_duration_seconds: float
    
    # Metadata
    training_metadata: Dict[str, Any]

class SelfTrainingMLOrchestrator:
    """Main orchestrator for the self-training ML system"""
    
    def __init__(self, db_pool: asyncpg.Pool, 
                 labeling_config: LabelingConfig = None,
                 feature_config: FeatureConfig = None,
                 model_config: ModelConfig = None):
        
        self.db_pool = db_pool
        self.labeling_config = labeling_config or LabelingConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.model_config = model_config or ModelConfig()
        
        # Initialize services
        self.auto_labeling_service = AutoLabelingService(db_pool, self.labeling_config)
        self.feature_engineering_service = FeatureEngineeringService(db_pool, self.feature_config)
        
        # Model storage
        self.model_dir = "models/self_training"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Target variables
        self.target_variables = ['y_30m', 'y_2h', 'y_24h']
        
        # Phase 5: Automated Retraining Integration
        self.retraining_enabled = True
        self.retraining_config = {
            'performance_threshold': 0.7,
            'drift_threshold': 0.1,
            'data_threshold': 1000,
            'max_model_versions': 3,
            'rollback_threshold': 0.05
        }
        self.performance_monitors = {}
        self.drift_detectors = {}
        self.scheduled_tasks = {}
        
        # Kubernetes integration
        self.k8s_enabled = False  # Will be set from config
        self.k8s_namespace = 'alphapulse'
        
    async def run_full_pipeline(self, 
                              start_time: datetime = None,
                              end_time: datetime = None,
                              symbols: List[str] = None) -> Dict[str, Any]:
        """
        Run the complete self-training ML pipeline
        
        Args:
            start_time: Start time for processing
            end_time: End time for processing
            symbols: List of symbols to process
            
        Returns:
            Pipeline results summary
        """
        
        logger.info("[PIPELINE] Starting self-training ML pipeline...")
        
        try:
            # Step 1: Auto-labeling
            logger.info("[STEP1] Auto-labeling news articles...")
            labeled_data = await self.auto_labeling_service.generate_labels_for_news_articles(
                start_time, end_time, symbols
            )
            
            if not labeled_data:
                logger.warning("[WARNING] No labeled data generated, skipping pipeline")
                return {'status': 'no_data', 'message': 'No labeled data available'}
            
            # Store labeled data
            stored_count = await self.auto_labeling_service.store_labeled_data(labeled_data)
            logger.info(f"[STORED] Stored {stored_count} labeled data points")
            
            # Step 2: Feature engineering
            logger.info("[STEP2] Feature engineering...")
            feature_sets = await self._extract_features_for_labeled_data(labeled_data)
            
            if not feature_sets:
                logger.warning("[WARNING] No features extracted, skipping pipeline")
                return {'status': 'no_features', 'message': 'No features extracted'}
            
            # Store feature sets
            stored_features = await self._store_feature_sets(feature_sets)
            logger.info(f"[STORED] Stored {stored_features} feature sets")
            
            # Step 3: Model training
            logger.info("[STEP3] Model training...")
            training_results = await self._train_models_for_all_targets()
            
            # Step 4: Store training results
            logger.info("[STEP4] Storing training results...")
            stored_results = await self._store_training_results(training_results)
            
            # Step 5: Online learning setup
            logger.info("[STEP5] Setting up online learning...")
            online_learning_setup = await self._setup_online_learning()
            
            # Generate summary
            summary = {
                'status': 'success',
                'labeled_data_count': len(labeled_data),
                'feature_sets_count': len(feature_sets),
                'training_results_count': len(training_results),
                'stored_results_count': stored_results,
                'online_learning_setup': online_learning_setup,
                'pipeline_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("[SUCCESS] Self-training ML pipeline completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"[ERROR] Error in self-training ML pipeline: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _extract_features_for_labeled_data(self, 
                                               labeled_data: List) -> List:
        """Extract features for all labeled data points"""
        
        feature_sets = []
        
        for data_point in labeled_data:
            try:
                # Get article data
                article_data = await self._get_article_data(data_point.news_id)
                
                if not article_data:
                    continue
                
                # Extract features
                feature_set = await self.feature_engineering_service.extract_features_for_news_article(
                    data_point.news_id,
                    data_point.symbol,
                    article_data
                )
                
                if feature_set:
                    feature_sets.append(feature_set)
                    
            except Exception as e:
                logger.error(f"❌ Error extracting features for news {data_point.news_id}: {e}")
                continue
        
        return feature_sets
    
    async def _get_article_data(self, news_id: int) -> Optional[Dict[str, Any]]:
        """Get article data from database"""
        
        query = """
            SELECT id, title, description, content, source, published_at,
                   sentiment_score, breaking_news, verified_source,
                   entities, metadata, keywords
            FROM raw_news_content
            WHERE id = $1
        """
        
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, news_id)
            return dict(row) if row else None
    
    async def _store_feature_sets(self, feature_sets: List) -> int:
        """Store feature sets in database"""
        
        stored_count = 0
        for feature_set in feature_sets:
            try:
                success = await self.feature_engineering_service.store_feature_set(feature_set)
                if success:
                    stored_count += 1
            except Exception as e:
                logger.error(f"❌ Error storing feature set: {e}")
                continue
        
        return stored_count
    
    async def _train_models_for_all_targets(self) -> List[TrainingResult]:
        """Train models for all target variables"""
        
        training_results = []
        
        for target_var in self.target_variables:
            try:
                logger.info(f"🎯 Training models for target: {target_var}")
                
                # Get training data
                training_data = await self._get_training_data(target_var)
                
                if len(training_data) < self.model_config.min_samples_for_training:
                    logger.warning(f"⚠️ Insufficient training data for {target_var}: {len(training_data)} samples")
                    continue
                
                # Train models
                for model_type in self.model_config.model_types:
                    try:
                        result = await self._train_single_model(training_data, target_var, model_type)
                        if result:
                            training_results.append(result)
                    except Exception as e:
                        logger.error(f"❌ Error training {model_type} for {target_var}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"❌ Error training models for {target_var}: {e}")
                continue
        
        return training_results
    
    async def _get_training_data(self, target_variable: str) -> List[Dict[str, Any]]:
        """Get training data for a specific target variable"""
        
        query = """
            SELECT 
                l.news_id, l.symbol, l.publish_time,
                l.y_30m, l.y_2h, l.y_24h,
                f.title_tfidf_ngrams, f.embedding_384d, f.entities, f.event_tags,
                f.source_trust, f.is_breaking, f.is_important, f.is_hot,
                f.publish_hour, f.day_of_week, f.dedup_cluster_size,
                f.social_volume_zscore_30m, f.social_volume_zscore_neg30m,
                f.dev_activity_7d_change, f.whale_tx_usd_1m_plus_24h_change,
                f.btc_dominance, f.total_mc_zscore, f.asset_vol_10d, f.atr_14, f.funding_rate
            FROM labels_news_market l
            JOIN feature_engineering_pipeline f ON l.news_id = f.news_id AND l.symbol = f.symbol
            WHERE l.confidence_score >= 0.5
            ORDER BY l.publish_time DESC
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query)
            
            training_data = []
            for row in rows:
                data_point = dict(row)
                
                # Parse JSON fields
                data_point['title_tfidf_ngrams'] = json.loads(data_point['title_tfidf_ngrams'])
                data_point['entities'] = json.loads(data_point['entities'])
                
                training_data.append(data_point)
            
            return training_data
    
    async def _train_single_model(self, 
                                training_data: List[Dict[str, Any]],
                                target_variable: str,
                                model_type: str) -> Optional[TrainingResult]:
        """Train a single model"""
        
        try:
            start_time = datetime.utcnow()
            
            # Prepare features and labels
            X, y = await self._prepare_features_and_labels(training_data, target_variable)
            
            if len(X) < self.model_config.min_samples_for_training:
                logger.warning(f"⚠️ Insufficient samples for {model_type} on {target_variable}")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.model_config.test_size, 
                random_state=self.model_config.random_state, stratify=y
            )
            
            # Initialize model
            model = await self._initialize_model(model_type, target_variable)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = await self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Get feature importance
            feature_importance = await self._get_feature_importance(model, model_type)
            
            # Save model
            model_version = f"{model_type}_{target_variable}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            await self._save_model(model, model_type, target_variable, model_version)
            
            # Calculate training duration
            training_duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Create training result
            result = TrainingResult(
                model_name=f"{model_type}_{target_variable}",
                target_variable=target_variable,
                model_type=model_type,
                model_version=model_version,
                training_samples=len(X_train),
                validation_samples=len(X_test),
                test_samples=len(X_test),
                accuracy_score=metrics['accuracy'],
                precision_score=metrics['precision'],
                recall_score=metrics['recall'],
                f1_score=metrics['f1'],
                auc_score=metrics['auc'],
                hyperparameters=await self._get_hyperparameters(model_type),
                feature_importance=feature_importance,
                training_duration_seconds=training_duration,
                training_metadata={
                    'training_timestamp': datetime.utcnow().isoformat(),
                    'data_points_used': len(training_data),
                    'positive_class_ratio': np.mean(y)
                }
            )
            
            logger.info(f"✅ Trained {model_type} for {target_variable}: F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error training {model_type} for {target_variable}: {e}")
            return None
    
    async def _prepare_features_and_labels(self, 
                                         training_data: List[Dict[str, Any]],
                                         target_variable: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        
        features = []
        labels = []
        
        for data_point in training_data:
            try:
                # Extract numeric features
                feature_vector = [
                    data_point['source_trust'],
                    float(data_point['is_breaking']),
                    float(data_point['is_important']),
                    float(data_point['is_hot']),
                    data_point['publish_hour'] / 24.0,  # Normalize hour
                    data_point['day_of_week'] / 7.0,    # Normalize day
                    data_point['dedup_cluster_size'],
                    data_point['social_volume_zscore_30m'],
                    data_point['social_volume_zscore_neg30m'],
                    data_point['dev_activity_7d_change'],
                    data_point['whale_tx_usd_1m_plus_24h_change'],
                    data_point['btc_dominance'],
                    data_point['total_mc_zscore'],
                    data_point['asset_vol_10d'],
                    data_point['atr_14'],
                    data_point['funding_rate']
                ]
                
                # Add embedding features (first 50 dimensions to keep it manageable)
                embedding = data_point['embedding_384d'][:50]
                feature_vector.extend(embedding)
                
                # Add TF-IDF features (top 100 features)
                tfidf_features = list(data_point['title_tfidf_ngrams'].values())[:100]
                if len(tfidf_features) < 100:
                    tfidf_features.extend([0.0] * (100 - len(tfidf_features)))
                feature_vector.extend(tfidf_features)
                
                features.append(feature_vector)
                labels.append(int(data_point[target_variable]))
                
            except Exception as e:
                logger.error(f"❌ Error preparing features: {e}")
                continue
        
        return np.array(features), np.array(labels)
    
    async def _initialize_model(self, model_type: str, target_variable: str):
        """Initialize a model based on type"""
        
        if model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.model_config.lgb_params)
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(**self.model_config.xgb_params)
        elif model_type == 'random_forest':
            return RandomForestClassifier(**self.model_config.rf_params)
        elif model_type == 'logistic_regression':
            return LogisticRegression(**self.model_config.lr_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    async def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate performance metrics"""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else 0.0
        }
        
        return metrics
    
    async def _get_feature_importance(self, model, model_type: str) -> Dict[str, float]:
        """Get feature importance from model"""
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return {}
            
            # Create feature names
            feature_names = []
            feature_names.extend([
                'source_trust', 'is_breaking', 'is_important', 'is_hot',
                'publish_hour', 'day_of_week', 'dedup_cluster_size',
                'social_volume_zscore_30m', 'social_volume_zscore_neg30m',
                'dev_activity_7d_change', 'whale_tx_usd_1m_plus_24h_change',
                'btc_dominance', 'total_mc_zscore', 'asset_vol_10d', 'atr_14', 'funding_rate'
            ])
            
            # Add embedding feature names
            for i in range(50):
                feature_names.append(f'embedding_{i}')
            
            # Add TF-IDF feature names
            for i in range(100):
                feature_names.append(f'tfidf_{i}')
            
            # Create importance dictionary
            importance_dict = {}
            for i, importance in enumerate(importances):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"❌ Error getting feature importance: {e}")
            return {}
    
    async def _get_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters for model type"""
        
        if model_type == 'lightgbm':
            return self.model_config.lgb_params
        elif model_type == 'xgboost':
            return self.model_config.xgb_params
        elif model_type == 'random_forest':
            return self.model_config.rf_params
        elif model_type == 'logistic_regression':
            return self.model_config.lr_params
        else:
            return {}
    
    async def _save_model(self, model, model_type: str, target_variable: str, model_version: str):
        """Save trained model to disk"""
        
        try:
            model_path = os.path.join(self.model_dir, f"{model_version}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"💾 Saved model: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
    
    async def _store_training_results(self, training_results: List[TrainingResult]) -> int:
        """Store training results in database"""
        
        stored_count = 0
        
        for result in training_results:
            try:
                query = """
                    INSERT INTO model_training_history (
                        timestamp, model_name, model_version, target_variable,
                        training_samples, validation_samples, test_samples,
                        accuracy_score, precision_score, recall_score, f1_score, auc_score,
                        model_type, hyperparameters, feature_importance, training_duration_seconds,
                        training_metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        query,
                        datetime.utcnow(),
                        result.model_name,
                        result.model_version,
                        result.target_variable,
                        result.training_samples,
                        result.validation_samples,
                        result.test_samples,
                        result.accuracy_score,
                        result.precision_score,
                        result.recall_score,
                        result.f1_score,
                        result.auc_score,
                        result.model_type,
                        json.dumps(result.hyperparameters),
                        json.dumps(result.feature_importance),
                        result.training_duration_seconds,
                        json.dumps(result.training_metadata)
                    )
                
                stored_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error storing training result: {e}")
                continue
        
        return stored_count
    
    async def _setup_online_learning(self) -> Dict[str, Any]:
        """Set up online learning infrastructure"""
        
        try:
            # Create online learning buffer entries for recent data
            query = """
                INSERT INTO online_learning_buffer (
                    timestamp, news_id, symbol, features, label_30m, label_2h, label_24h,
                    prediction_30m, prediction_2h, prediction_24h, buffer_status, learning_priority
                )
                SELECT 
                    NOW(), l.news_id, l.symbol, 
                    jsonb_build_object(
                        'source_trust', f.source_trust,
                        'is_breaking', f.is_breaking,
                        'is_important', f.is_important,
                        'is_hot', f.is_hot,
                        'publish_hour', f.publish_hour,
                        'day_of_week', f.day_of_week,
                        'dedup_cluster_size', f.dedup_cluster_size,
                        'social_volume_zscore_30m', f.social_volume_zscore_30m,
                        'social_volume_zscore_neg30m', f.social_volume_zscore_neg30m,
                        'dev_activity_7d_change', f.dev_activity_7d_change,
                        'whale_tx_usd_1m_plus_24h_change', f.whale_tx_usd_1m_plus_24h_change,
                        'btc_dominance', f.btc_dominance,
                        'total_mc_zscore', f.total_mc_zscore,
                        'asset_vol_10d', f.asset_vol_10d,
                        'atr_14', f.atr_14,
                        'funding_rate', f.funding_rate
                    ) as features,
                    l.y_30m, l.y_2h, l.y_24h,
                    0.5, 0.5, 0.5,  -- Default predictions
                    'pending', 0.5   -- Default status and priority
                FROM labels_news_market l
                JOIN feature_engineering_pipeline f ON l.news_id = f.news_id AND l.symbol = f.symbol
                WHERE l.publish_time >= NOW() - INTERVAL '24 hours'
                AND l.confidence_score >= 0.7
                ON CONFLICT (news_id, symbol) DO NOTHING
            """
            
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(query)
            
            return {
                'status': 'setup_complete',
                'buffer_entries_created': 'recent_high_confidence_data',
                'online_learning_ready': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error setting up online learning: {e}")
            return {
                'status': 'setup_failed',
                'error': str(e),
                'online_learning_ready': False
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        try:
            # Get labeling statistics
            labeling_stats = await self.auto_labeling_service.get_labeling_statistics()
            
            # Get model performance
            query = """
                SELECT 
                    target_variable,
                    model_type,
                    AVG(f1_score) as avg_f1,
                    AVG(auc_score) as avg_auc,
                    COUNT(*) as model_count,
                    MAX(timestamp) as last_training
                FROM model_training_history
                WHERE timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY target_variable, model_type
                ORDER BY target_variable, avg_f1 DESC
            """
            
            async with self.db_pool.acquire() as conn:
                model_performance = await conn.fetch(query)
            
            # Get online learning buffer status
            buffer_query = """
                SELECT 
                    buffer_status,
                    COUNT(*) as count,
                    AVG(learning_priority) as avg_priority
                FROM online_learning_buffer
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY buffer_status
            """
            
            async with self.db_pool.acquire() as conn:
                buffer_status = await conn.fetch(buffer_query)
            
            return {
                'system_status': 'operational',
                'labeling_statistics': labeling_stats,
                'model_performance': [dict(row) for row in model_performance],
                'online_learning_buffer': [dict(row) for row in buffer_status],
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {
                'system_status': 'error',
                'error': str(e),
                'last_updated': datetime.utcnow().isoformat()
            }
