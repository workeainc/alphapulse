#!/usr/bin/env python3
"""
ML Auto-Retraining Training CLI
Integrates with existing noise filtering and market regime classification
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
import hashlib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.noise_filter_engine import NoiseFilterEngine
from ai.market_regime_classifier import MarketRegimeClassifier
from ai.adaptive_learning_engine import AdaptiveLearningEngine
from ai.ml_auto_retraining.model_versioning_manager import ModelVersioningManager
from ai.ml_auto_retraining.rollback_manager import RollbackManager

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

class MLModelTrainer:
    """ML Model Trainer that integrates with existing AlphaPlus infrastructure"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.noise_filter = None
        self.market_regime_classifier = None
        self.adaptive_learning = None
        self.versioning_manager = ModelVersioningManager(db_config)
        self.rollback_manager = RollbackManager(db_config)
        
    async def initialize_components(self):
        """Initialize noise filtering and market regime components"""
        logger.info("🔧 Initializing ML training components...")
        
        try:
            self.noise_filter = NoiseFilterEngine(self.db_config)
            await self.noise_filter.initialize()
            
            self.market_regime_classifier = MarketRegimeClassifier(self.db_config)
            await self.market_regime_classifier.initialize()
            
            self.adaptive_learning = AdaptiveLearningEngine(self.db_config)
            await self.adaptive_learning.initialize()
            
            logger.info("✅ ML training components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def load_ohlcv_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load OHLCV data from TimescaleDB"""
        logger.info(f"📊 Loading OHLCV data for {symbol} from {start_date} to {end_date}")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            
            query = """
                SELECT 
                    time as timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM ohlcv
                WHERE symbol = %s AND time >= %s AND time < %s
                ORDER BY time ASC
            """
            
            df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            logger.info(f"✅ Loaded {len(df)} OHLCV records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load OHLCV data: {e}")
            raise
    
    def load_pattern_performance_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load pattern performance data for labeling"""
        logger.info(f"📊 Loading pattern performance data for {symbol}")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            
            query = """
                SELECT 
                    timestamp,
                    tracking_id,
                    pattern_id,
                    symbol,
                    pattern_name,
                    timeframe,
                    pattern_confidence,
                    predicted_outcome,
                    actual_outcome,
                    market_regime,
                    volume_ratio,
                    volatility_level,
                    spread_impact,
                    noise_filter_score,
                    performance_score,
                    outcome_timestamp,
                    outcome_price,
                    profit_loss,
                    market_conditions
                FROM pattern_performance_tracking
                WHERE symbol = %s AND timestamp >= %s AND timestamp < %s
                ORDER BY timestamp ASC
            """
            
            df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
            conn.close()
            
            logger.info(f"✅ Loaded {len(df)} pattern performance records")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load pattern performance data: {e}")
            raise
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical features from OHLCV data"""
        logger.info("🔧 Creating technical features...")
        
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
        
        logger.info(f"✅ Created {len(df.columns)} technical features")
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
    
    async def apply_noise_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply noise filtering to the dataset"""
        logger.info("🔧 Applying noise filtering...")
        
        if self.noise_filter is None:
            logger.warning("⚠️ Noise filter not initialized, skipping filtering")
            return df
        
        try:
            # Create a sample pattern data for filtering
            sample_pattern = {
                'symbol': df['symbol'].iloc[0] if 'symbol' in df.columns else 'BTCUSDT',
                'pattern_name': 'sample_pattern',
                'timeframe': '1m',
                'confidence': 0.8,
                'timestamp': df.index[0] if hasattr(df.index[0], 'timestamp') else datetime.now()
            }
            
            # Apply filtering to each row (simplified approach)
            filtered_rows = []
            for idx, row in df.iterrows():
                # Create market data slice for this row
                market_data = df.loc[:idx].tail(50)  # Last 50 rows for context
                
                # Apply noise filter
                passed_filter, _ = await self.noise_filter.filter_pattern(sample_pattern, market_data)
                
                if passed_filter:
                    filtered_rows.append(row)
            
            filtered_df = pd.DataFrame(filtered_rows)
            logger.info(f"✅ Noise filtering: {len(df)} -> {len(filtered_df)} records")
            return filtered_df
            
        except Exception as e:
            logger.error(f"❌ Noise filtering failed: {e}")
            return df
    
    async def classify_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market regime for each data point"""
        logger.info("🔧 Classifying market regime...")
        
        if self.market_regime_classifier is None:
            logger.warning("⚠️ Market regime classifier not initialized, skipping classification")
            df['market_regime'] = 'unknown'
            return df
        
        try:
            regimes = []
            for idx, row in df.iterrows():
                # Create market data slice for this row
                market_data = df.loc[:idx].tail(50)  # Last 50 rows for context
                
                if len(market_data) >= 10:
                    regime_result = await self.market_regime_classifier.classify_market_regime(
                        market_data, 'BTCUSDT', '1m'
                    )
                    regimes.append(regime_result['regime'])
                else:
                    regimes.append('unknown')
            
            df['market_regime'] = regimes
            logger.info(f"✅ Market regime classification completed")
            return df
            
        except Exception as e:
            logger.error(f"❌ Market regime classification failed: {e}")
            df['market_regime'] = 'unknown'
            return df
    
    def create_labels(self, df: pd.DataFrame, performance_df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """Create labels for ML training"""
        logger.info(f"🔧 Creating labels with {horizon}-period horizon...")
        
        df = df.copy()
        
        # Merge with performance data
        if not performance_df.empty:
            # Create labels based on pattern performance
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
                    pattern_perf = performance_df[
                        (performance_df['timestamp'] >= df.iloc[i]['timestamp']) &
                        (performance_df['timestamp'] < df.iloc[i]['timestamp'] + timedelta(hours=1))
                    ]
                    
                    if not pattern_perf.empty:
                        avg_performance = pattern_perf['performance_score'].mean()
                        df.iloc[i, df.columns.get_loc('target')] = 1 if avg_performance > 0.6 else 0
                    else:
                        df.iloc[i, df.columns.get_loc('target')] = 0
        else:
            # Fallback labeling based on price movement
            df['future_returns'] = df['close'].shift(-horizon) / df['close'] - 1
            df['target'] = (df['future_returns'] > 0.01).astype(int)  # 1% threshold
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target'])
        
        logger.info(f"✅ Created labels: {df['target'].sum()} positive, {len(df) - df['target'].sum()} negative")
        return df
    
    def prepare_features(self, df: pd.DataFrame, regime: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for ML training"""
        logger.info(f"🔧 Preparing features for {regime} regime...")
        
        # Filter by regime
        if regime != 'all':
            df = df[df['market_regime'] == regime].copy()
        
        # Select feature columns
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
        
        # Prepare X and y
        X = df[available_features].fillna(0).values
        y = df['target'].values
        
        logger.info(f"✅ Prepared {X.shape[1]} features for {len(X)} samples")
        return X, y
    
    def _train_xgboost_model(self, X_train: np.ndarray, y_train: np.ndarray, params: Dict[str, Any]) -> XGBClassifier:
        """Train XGBoost model"""
        logger.info("🔧 Training XGBoost model...")
        
        model = XGBClassifier(
            n_estimators=params.get('n_estimators', 400),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.9),
            colsample_bytree=params.get('colsample_bytree', 0.9),
            reg_lambda=params.get('reg_lambda', 1.0),
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        logger.info("✅ Model training completed")
        return model
    
    def evaluate_model(self, model: XGBClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("🔧 Evaluating model performance...")
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'f1': float(f1_score(y_val, y_pred)),
            'precision': float(precision_score(y_val, y_pred)),
            'recall': float(recall_score(y_val, y_pred)),
            'roc_auc': float(roc_auc_score(y_val, y_proba)),
            'samples': int(len(y_val))
        }
        
        logger.info(f"✅ Model evaluation: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}")
        return metrics
    
    def save_model(self, model: XGBClassifier, model_name: str, regime: str, symbol: str, 
                   metrics: Dict[str, float], params: Dict[str, Any]) -> str:
        """Save model and metadata"""
        logger.info("💾 Saving model and metadata...")
        
        # Create artifacts directory
        artifacts_dir = f"artifacts/{model_name}"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save model
        model_path = f"{artifacts_dir}/{regime}_{symbol}.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'regime': regime,
            'symbol': symbol,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'params': params,
            'model_path': model_path
        }
        
        metadata_path = f"{artifacts_dir}/{regime}_{symbol}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model saved to {model_path}")
        return model_path
    
    async def train_model(self, symbol: str, regime: str, start_date: datetime, end_date: datetime, 
                         model_name: str = 'alphaplus_pattern_classifier', horizon: int = 10) -> Optional[str]:
        """Train ML model with the given parameters and versioning integration"""
        logger.info(f"🚀 Starting ML training for {symbol} - {regime} regime")
        
        training_start_time = datetime.now()
        
        try:
            # Load data
            ohlcv_df = self.load_ohlcv_data(symbol, start_date, end_date)
            performance_df = self.load_pattern_performance_data(symbol, start_date, end_date)
            
            if ohlcv_df.empty:
                logger.warning(f"No OHLCV data found for {symbol}")
                return None
            
            # Add symbol column
            ohlcv_df['symbol'] = symbol
            
            # Create technical features
            feature_df = self.create_technical_features(ohlcv_df)
            
            # Apply noise filtering
            filtered_df = await self.apply_noise_filtering(feature_df)
            
            # Classify market regime
            regime_df = await self.classify_market_regime(filtered_df)
            
            # Create labels
            labeled_df = self.create_labels(regime_df, performance_df, horizon)
            
            if len(labeled_df) < 1000:
                logger.warning(f"Insufficient data after filtering: {len(labeled_df)} samples")
                return None
            
            # Prepare features
            X, y = self.prepare_features(labeled_df, regime)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            # Train model
            params = {
                'n_estimators': 400,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_lambda': 1.0
            }
            
            model = self._train_xgboost_model(X_train, y_train, params)
            
            # Evaluate model
            metrics = self.evaluate_model(model, X_val, y_val)
            
            # Save model
            model_path = self.save_model(model, model_name, regime, symbol, metrics, params)
            
            # Calculate training duration
            training_duration = (datetime.now() - training_start_time).total_seconds()
            
            # Get current production model for versioning
            current_production = await self.versioning_manager.get_production_model(model_name, regime, symbol)
            new_version = (current_production.version + 1) if current_production else 1
            
            # Create model lineage
            lineage_id = await self.versioning_manager.create_model_lineage(
                model_name=model_name,
                model_version=new_version,
                parent_model_name=current_production.model_name if current_production else None,
                parent_model_version=current_production.version if current_production else None,
                training_data=labeled_df,
                feature_set={'features': list(X.columns) if hasattr(X, 'columns') else []},
                hyperparameters=params,
                training_environment="production_v1",
                training_duration_seconds=int(training_duration),
                training_samples=len(X_train),
                validation_samples=len(X_val),
                lineage_metadata={
                    'symbol': symbol,
                    'regime': regime,
                    'horizon': horizon,
                    'phase': 'phase1_enhancement'
                }
            )
            
            # Calculate model artifact size
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0
            
            # Generate model hash
            model_hash = hashlib.sha256(open(model_path, 'rb').read()).hexdigest() if os.path.exists(model_path) else "unknown"
            
            # Register new model version
            model_version = await self.versioning_manager.register_model_version(
                model_name=model_name,
                version=new_version,
                status='staging',  # Start as staging, will be promoted later
                regime=regime,
                symbol=symbol,
                model_artifact_path=model_path,
                model_artifact_size_mb=model_size_mb,
                model_artifact_hash=model_hash,
                training_metrics=metrics,
                validation_metrics=metrics,  # Using same metrics for now
                performance_metrics=None,  # Will be updated during production
                version_metadata={
                    'lineage_id': lineage_id,
                    'symbol': symbol,
                    'regime': regime,
                    'horizon': horizon,
                    'training_duration_seconds': training_duration,
                    'phase': 'phase1_enhancement'
                }
            )
            
            logger.info(f"✅ Training completed with versioning: {model_path} (v{new_version})")
            logger.info(f"✅ Model lineage created: {lineage_id}")
            logger.info(f"✅ Model version registered: {model_name} v{new_version}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up ML training components...")
        
        if self.noise_filter:
            await self.noise_filter.cleanup()
        
        if self.market_regime_classifier:
            await self.market_regime_classifier.cleanup()
        
        if self.adaptive_learning:
            await self.adaptive_learning.cleanup()
        
        logger.info("✅ Cleanup completed")

async def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ML model for pattern classification')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--regime', required=True, choices=['trending', 'sideways', 'volatile', 'consolidation', 'all'])
    parser.add_argument('--days', type=int, default=120, help='Number of days of training data')
    parser.add_argument('--model_name', default='alphaplus_pattern_classifier', help='Model name')
    parser.add_argument('--horizon', type=int, default=10, help='Prediction horizon in periods')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting ML training for {args.symbol} - {args.regime} regime")
    
    trainer = MLModelTrainer(DB_CONFIG)
    
    try:
        # Initialize components
        await trainer.initialize_components()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        # Load data
        ohlcv_df = trainer.load_ohlcv_data(args.symbol, start_date, end_date)
        performance_df = trainer.load_pattern_performance_data(args.symbol, start_date, end_date)
        
        if ohlcv_df.empty:
            raise ValueError(f"No OHLCV data found for {args.symbol}")
        
        # Add symbol column
        ohlcv_df['symbol'] = args.symbol
        
        # Create technical features
        feature_df = trainer.create_technical_features(ohlcv_df)
        
        # Apply noise filtering
        filtered_df = await trainer.apply_noise_filtering(feature_df)
        
        # Classify market regime
        regime_df = await trainer.classify_market_regime(filtered_df)
        
        # Create labels
        labeled_df = trainer.create_labels(regime_df, performance_df, args.horizon)
        
        if len(labeled_df) < 1000:
            raise ValueError(f"Insufficient data after filtering: {len(labeled_df)} samples")
        
        # Prepare features
        X, y = trainer.prepare_features(labeled_df, args.regime)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        # Train model
        params = {
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_lambda': 1.0
        }
        
        model = trainer.train_model(X_train, y_train, params)
        
        # Evaluate model
        metrics = trainer.evaluate_model(model, X_val, y_val)
        
        # Save model
        model_path = trainer.save_model(model, args.model_name, args.regime, args.symbol, metrics, params)
        
        # Output results
        result = {
            'model_name': args.model_name,
            'regime': args.regime,
            'symbol': args.symbol,
            'artifact_path': model_path,
            'metrics': metrics,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'created_at': datetime.now().isoformat()
        }
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    finally:
        await trainer.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
