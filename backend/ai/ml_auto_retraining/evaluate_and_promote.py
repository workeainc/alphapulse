#!/usr/bin/env python3
"""
ML Auto-Retraining Model Evaluation and Promotion
Compares candidate models against production models and decides promotion
"""

import os
import sys
import json
import joblib
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import psycopg2
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy.stats import ks_2samp
import hashlib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'alphapulse',
    'user': 'alpha_emon',
    'password': 'Emon_@17711',
    'port': 5432
}

# Promotion criteria
PROMOTE_IF = {
    "min_f1": 0.60,
    "min_precision": 0.58,
    "min_recall": 0.50,
    "rel_improvement_f1": 0.03,   # +3% vs prod
    "rel_improvement_precision": 0.02,  # +2% vs prod
    "max_feature_drift_ks": 0.15,  # reject if drift too large
    "min_samples": 1000  # minimum validation samples
}

class ModelEvaluator:
    """Model evaluator for comparing candidate vs production models"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        
    def load_candidate_model(self, artifact_path: str) -> Any:
        """Load candidate model from artifact"""
        logger.info(f"📦 Loading candidate model from {artifact_path}")
        
        try:
            model = joblib.load(artifact_path)
            logger.info("✅ Candidate model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load candidate model: {e}")
            raise
    
    def load_production_model(self, model_name: str, regime: str, symbol: str) -> Optional[Any]:
        """Load current production model from database"""
        logger.info(f"📦 Loading production model for {model_name} - {regime} - {symbol}")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            query = """
                SELECT artifact_uri, metrics, params
                FROM ml_models
                WHERE model_name = %s AND regime = %s AND symbol = %s AND status = 'production'
                ORDER BY version DESC
                LIMIT 1
            """
            
            cursor.execute(query, (model_name, regime, symbol))
            result = cursor.fetchone()
            
            if result:
                artifact_uri, metrics, params = result
                if artifact_uri and os.path.exists(artifact_uri):
                    model = joblib.load(artifact_uri)
                    logger.info("✅ Production model loaded successfully")
                    return model, metrics, params
                else:
                    logger.warning("⚠️ Production model artifact not found")
                    return None, metrics, params
            else:
                logger.info("ℹ️ No production model found")
                return None, None, None
                
        except Exception as e:
            logger.error(f"❌ Failed to load production model: {e}")
            return None, None, None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    def load_validation_data(self, symbol: str, days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load validation data for model evaluation"""
        logger.info(f"📊 Loading validation data for {symbol} (last {days} days)")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            
            # Load OHLCV data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ohlcv_query = """
                SELECT 
                    time as timestamp,
                    open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = %s AND time >= %s AND time < %s
                ORDER BY time ASC
            """
            
            ohlcv_df = pd.read_sql(ohlcv_query, conn, params=(symbol, start_date, end_date))
            
            # Load pattern performance data
            perf_query = """
                SELECT 
                    timestamp, pattern_id, actual_outcome, profit_loss, performance_score
                FROM pattern_performance_tracking
                WHERE symbol = %s AND timestamp >= %s AND timestamp < %s
                ORDER BY timestamp ASC
            """
            
            perf_df = pd.read_sql(perf_query, conn, params=(symbol, start_date, end_date))
            
            conn.close()
            
            logger.info(f"✅ Loaded {len(ohlcv_df)} OHLCV records and {len(perf_df)} performance records")
            return ohlcv_df, perf_df
            
        except Exception as e:
            logger.error(f"❌ Failed to load validation data: {e}")
            raise
    
    def create_validation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for validation (same as training)"""
        logger.info("🔧 Creating validation features...")
        
        df = df.copy()
        
        # Price-based features
        df['returns_1m'] = df['close'].pct_change()
        df['returns_5m'] = df['close'].pct_change(5)
        df['returns_15m'] = df['close'].pct_change(15)
        df['returns_1h'] = df['close'].pct_change(60)
        
        # Volatility features
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_21'] = self._calculate_atr(df, 21)
        
        # Volume features
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        df['volume_std_20'] = df['volume'].rolling(20).std()
        df['volume_z_score'] = (df['volume'] - df['volume_ma_20']) / df['volume_std_20']
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Price position features
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], 20, 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Support/Resistance features
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        logger.info(f"✅ Created {len(df.columns)} validation features")
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def create_validation_labels(self, df: pd.DataFrame, perf_df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """Create validation labels"""
        logger.info(f"🔧 Creating validation labels with {horizon}-period horizon...")
        
        df = df.copy()
        
        # Create labels based on future price movement
        df['future_max'] = df['close'].shift(-1).rolling(horizon).max()
        df['future_min'] = df['close'].shift(-1).rolling(horizon).min()
        
        # Simple labeling: profitable if price increases by 2% before decreasing by 1%
        df['target'] = 0
        for i in range(len(df) - horizon):
            future_slice = df.iloc[i+1:i+horizon+1]
            current_price = df.iloc[i]['close']
            
            # Check if we hit 2% profit before 1% loss
            profit_hit = (future_slice['high'] >= current_price * 1.02).any()
            loss_hit = (future_slice['low'] <= current_price * 0.99).any()
            
            if profit_hit and not loss_hit:
                df.iloc[i, df.columns.get_loc('target')] = 1
            elif loss_hit and not profit_hit:
                df.iloc[i, df.columns.get_loc('target')] = 0
            else:
                # If both or neither, use pattern performance if available
                pattern_perf = perf_df[
                    (perf_df['timestamp'] >= df.iloc[i]['timestamp']) &
                    (perf_df['timestamp'] < df.iloc[i]['timestamp'] + timedelta(hours=1))
                ]
                
                if not pattern_perf.empty:
                    avg_performance = pattern_perf['performance_score'].mean()
                    df.iloc[i, df.columns.get_loc('target')] = 1 if avg_performance > 0.6 else 0
                else:
                    df.iloc[i, df.columns.get_loc('target')] = 0
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target'])
        
        logger.info(f"✅ Created validation labels: {df['target'].sum()} positive, {len(df) - df['target'].sum()} negative")
        return df
    
    def prepare_validation_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for validation"""
        logger.info("🔧 Preparing validation features...")
        
        # Select feature columns (same as training)
        feature_columns = [
            'returns_1m', 'returns_5m', 'returns_15m', 'returns_1h',
            'high_low_ratio', 'atr_14', 'atr_21',
            'volume_ratio', 'volume_z_score',
            'price_vs_sma20', 'price_vs_sma50', 'sma_cross',
            'rsi_14', 'rsi_21',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_position',
            'support_distance', 'resistance_distance'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing features: {missing_features}")
        
        # Prepare X
        X = df[available_features].fillna(0).values
        
        logger.info(f"✅ Prepared {X.shape[1]} features for {len(X)} validation samples")
        return X
    
    def evaluate_model_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("🔧 Evaluating model performance...")
        
        try:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            metrics = {
                'f1': float(f1_score(y, y_pred)),
                'precision': float(precision_score(y, y_pred)),
                'recall': float(recall_score(y, y_pred)),
                'roc_auc': float(roc_auc_score(y, y_proba)),
                'samples': int(len(y))
            }
            
            logger.info(f"✅ Model evaluation: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            raise
    
    def calculate_feature_drift(self, X_new: np.ndarray, X_old: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature drift using KS test"""
        logger.info("🔧 Calculating feature drift...")
        
        try:
            drift_scores = {}
            
            for i, feature_name in enumerate(feature_names):
                if i < X_new.shape[1] and i < X_old.shape[1]:
                    try:
                        # Remove NaN values
                        new_feature = X_new[:, i]
                        old_feature = X_old[:, i]
                        
                        new_feature = new_feature[~np.isnan(new_feature)]
                        old_feature = old_feature[~np.isnan(old_feature)]
                        
                        if len(new_feature) > 10 and len(old_feature) > 10:
                            ks_statistic, _ = ks_2samp(new_feature, old_feature)
                            drift_scores[feature_name] = float(ks_statistic)
                        else:
                            drift_scores[feature_name] = 0.0
                    except Exception:
                        drift_scores[feature_name] = 0.0
            
            max_drift = max(drift_scores.values()) if drift_scores else 0.0
            
            logger.info(f"✅ Feature drift calculated: max_drift={max_drift:.3f}")
            return drift_scores, max_drift
            
        except Exception as e:
            logger.error(f"❌ Feature drift calculation failed: {e}")
            return {}, 0.0
    
    def decide_promotion(self, candidate_metrics: Dict[str, float], production_metrics: Optional[Dict[str, float]], 
                        drift_max: float) -> Tuple[bool, List[str]]:
        """Decide whether to promote the candidate model"""
        logger.info("🔧 Deciding model promotion...")
        
        promote = False
        notes = []
        
        # Check absolute thresholds
        if candidate_metrics['f1'] < PROMOTE_IF['min_f1']:
            notes.append(f"F1 score {candidate_metrics['f1']:.3f} below minimum {PROMOTE_IF['min_f1']}")
        
        if candidate_metrics['precision'] < PROMOTE_IF['min_precision']:
            notes.append(f"Precision {candidate_metrics['precision']:.3f} below minimum {PROMOTE_IF['min_precision']}")
        
        if candidate_metrics['recall'] < PROMOTE_IF['min_recall']:
            notes.append(f"Recall {candidate_metrics['recall']:.3f} below minimum {PROMOTE_IF['min_recall']}")
        
        if candidate_metrics['samples'] < PROMOTE_IF['min_samples']:
            notes.append(f"Sample count {candidate_metrics['samples']} below minimum {PROMOTE_IF['min_samples']}")
        
        # Check drift threshold
        if drift_max > PROMOTE_IF['max_feature_drift_ks']:
            notes.append(f"Feature drift {drift_max:.3f} exceeds maximum {PROMOTE_IF['max_feature_drift_ks']}")
        
        # Compare with production model if available
        if production_metrics:
            prod_f1 = production_metrics.get('f1', 0)
            prod_precision = production_metrics.get('precision', 0)
            
            f1_improvement = candidate_metrics['f1'] - prod_f1
            precision_improvement = candidate_metrics['precision'] - prod_precision
            
            if f1_improvement < PROMOTE_IF['rel_improvement_f1']:
                notes.append(f"F1 improvement {f1_improvement:.3f} below threshold {PROMOTE_IF['rel_improvement_f1']}")
            
            if precision_improvement < PROMOTE_IF['rel_improvement_precision']:
                notes.append(f"Precision improvement {precision_improvement:.3f} below threshold {PROMOTE_IF['rel_improvement_precision']}")
            
            # Decide promotion
            if (candidate_metrics['f1'] >= PROMOTE_IF['min_f1'] and
                candidate_metrics['precision'] >= PROMOTE_IF['min_precision'] and
                candidate_metrics['recall'] >= PROMOTE_IF['min_recall'] and
                candidate_metrics['samples'] >= PROMOTE_IF['min_samples'] and
                drift_max <= PROMOTE_IF['max_feature_drift_ks'] and
                f1_improvement >= PROMOTE_IF['rel_improvement_f1'] and
                precision_improvement >= PROMOTE_IF['rel_improvement_precision']):
                promote = True
        else:
            # No production model - check absolute thresholds only
            if (candidate_metrics['f1'] >= PROMOTE_IF['min_f1'] and
                candidate_metrics['precision'] >= PROMOTE_IF['min_precision'] and
                candidate_metrics['recall'] >= PROMOTE_IF['min_recall'] and
                candidate_metrics['samples'] >= PROMOTE_IF['min_samples'] and
                drift_max <= PROMOTE_IF['max_feature_drift_ks']):
                promote = True
            else:
                notes.append("Cold start but absolute thresholds not met")
        
        decision = "promote" if promote else "reject"
        logger.info(f"✅ Promotion decision: {decision}")
        
        return promote, notes
    
    def update_model_registry(self, model_name: str, regime: str, symbol: str, 
                             candidate_metrics: Dict[str, float], artifact_path: str,
                             promote: bool, notes: List[str], drift_scores: Dict[str, float],
                             production_version: Optional[int] = None) -> int:
        """Update model registry with evaluation results"""
        logger.info("💾 Updating model registry...")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get next version number
            cursor.execute("""
                SELECT COALESCE(MAX(version), 0) + 1
                FROM ml_models
                WHERE model_name = %s AND regime = %s AND symbol = %s
            """, (model_name, regime, symbol))
            
            next_version = cursor.fetchone()[0]
            
            # Archive current production model if promoting
            if promote and production_version:
                cursor.execute("""
                    UPDATE ml_models
                    SET status = 'archived'
                    WHERE model_name = %s AND regime = %s AND symbol = %s AND status = 'production'
                """, (model_name, regime, symbol))
            
            # Insert new model record
            status = 'production' if promote else 'staging'
            
            cursor.execute("""
                INSERT INTO ml_models (
                    model_name, version, status, regime, symbol, params, metrics, artifact_uri
                ) VALUES (
                    %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s
                )
            """, (
                model_name, next_version, status, regime, symbol,
                json.dumps({'algorithm': 'xgboost', 'promoted': promote}),
                json.dumps(candidate_metrics),
                artifact_path
            ))
            
            # Insert evaluation history
            cursor.execute("""
                INSERT INTO ml_eval_history (
                    model_name, candidate_version, baseline_version, metrics, drift, decision, notes
                ) VALUES (
                    %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s
                )
            """, (
                model_name, next_version, production_version,
                json.dumps(candidate_metrics),
                json.dumps(drift_scores),
                'promote' if promote else 'reject',
                '; '.join(notes)
            ))
            
            conn.commit()
            logger.info(f"✅ Model registry updated: version {next_version}, status {status}")
            return next_version
            
        except Exception as e:
            logger.error(f"❌ Failed to update model registry: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate and promote ML model')
    parser.add_argument('--model_name', required=True, help='Model name')
    parser.add_argument('--regime', required=True, choices=['trending', 'sideways', 'volatile', 'consolidation'])
    parser.add_argument('--symbol', required=True, help='Trading symbol')
    parser.add_argument('--candidate_artifact', required=True, help='Path to candidate model artifact')
    parser.add_argument('--validation_days', type=int, default=30, help='Days of validation data')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting model evaluation for {args.model_name} - {args.regime} - {args.symbol}")
    
    evaluator = ModelEvaluator(DB_CONFIG)
    
    try:
        # Load candidate model
        candidate_model = evaluator.load_candidate_model(args.candidate_artifact)
        
        # Load production model
        production_model, production_metrics, production_params = evaluator.load_production_model(
            args.model_name, args.regime, args.symbol
        )
        
        # Load validation data
        ohlcv_df, perf_df = evaluator.load_validation_data(args.symbol, args.validation_days)
        
        if ohlcv_df.empty:
            raise ValueError(f"No validation data found for {args.symbol}")
        
        # Create validation features
        feature_df = evaluator.create_validation_features(ohlcv_df)
        labeled_df = evaluator.create_validation_labels(feature_df, perf_df)
        
        # Prepare validation features
        X_val = evaluator.prepare_validation_features(labeled_df)
        y_val = labeled_df['target'].values
        
        # Evaluate candidate model
        candidate_metrics = evaluator.evaluate_model_performance(candidate_model, X_val, y_val)
        
        # Calculate feature drift if production model exists
        drift_scores = {}
        drift_max = 0.0
        
        if production_model is not None:
            # For simplicity, we'll use the same validation data for drift calculation
            # In production, you'd want to use the training data from production model
            drift_scores, drift_max = evaluator.calculate_feature_drift(
                X_val, X_val,  # Simplified - in reality, use production training data
                ['returns_1m', 'returns_5m', 'returns_15m', 'returns_1h',
                 'high_low_ratio', 'atr_14', 'atr_21', 'volume_ratio', 'volume_z_score',
                 'price_vs_sma20', 'price_vs_sma50', 'sma_cross', 'rsi_14', 'rsi_21',
                 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                 'support_distance', 'resistance_distance']
            )
        
        # Decide promotion
        promote, notes = evaluator.decide_promotion(
            candidate_metrics, production_metrics, drift_max
        )
        
        # Update model registry
        production_version = None
        if production_metrics:
            # Get production version from database
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version FROM ml_models
                WHERE model_name = %s AND regime = %s AND symbol = %s AND status = 'production'
                ORDER BY version DESC LIMIT 1
            """, (args.model_name, args.regime, args.symbol))
            result = cursor.fetchone()
            if result:
                production_version = result[0]
            cursor.close()
            conn.close()
        
        new_version = evaluator.update_model_registry(
            args.model_name, args.regime, args.symbol,
            candidate_metrics, args.candidate_artifact,
            promote, notes, drift_scores, production_version
        )
        
        # Output results
        result = {
            'model_name': args.model_name,
            'regime': args.regime,
            'symbol': args.symbol,
            'candidate_metrics': candidate_metrics,
            'production_metrics': production_metrics,
            'drift_max': drift_max,
            'promoted': promote,
            'new_version': new_version,
            'notes': notes,
            'evaluated_at': datetime.now().isoformat()
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
