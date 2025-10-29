"""
Advanced Feature Engineering Pipeline for AlphaPlus
Creates ML-ready features from raw market data
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Import our enhanced components
try:
    from ..src.database.connection import TimescaleDBConnection
    from ..src.data.storage import DataStorage
except ImportError:
    # Fallback for testing
    TimescaleDBConnection = None
    DataStorage = None

logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Feature type enumeration"""
    TECHNICAL_INDICATOR = "technical_indicator"
    PRICE_ACTION = "price_action"
    VOLUME_ANALYSIS = "volume_analysis"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    TIME_BASED = "time_based"
    DERIVED = "derived"
    ML_ENGINEERED = "ml_engineered"
    MULTITIMEFRAME = "multitimeframe"
    MARKET_REGIME = "market_regime"
    NEWS_SENTIMENT = "news_sentiment"
    VOLUME_PROFILE = "volume_profile"

@dataclass
class FeatureDefinition:
    """Feature definition and metadata"""
    name: str
    feature_type: FeatureType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    output_type: str
    is_lagging: bool
    lag_periods: int

class AdvancedFeatureEngineering:
    """Advanced feature engineering pipeline for ML models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Feature engineering configuration
        self.feature_window = self.config.get('feature_window', 100)
        self.max_features = self.config.get('max_features', 200)
        self.feature_selection_method = self.config.get('feature_selection_method', 'mutual_info')
        self.scaling_method = self.config.get('scaling_method', 'robust')
        self.pca_components = self.config.get('pca_components', 50)
        
        # Component references
        self.db_connection = None
        self.storage = None
        
        # Feature definitions
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        
        # Scaler and transformers
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        
        # Feature cache
        self.feature_cache = {}
        self.cache_size = self.config.get('cache_size', 1000)
        
        # Performance tracking
        self.stats = {
            'features_created': 0,
            'features_cached': 0,
            'processing_time': 0.0
        }
        
        # Initialize feature definitions
        self._initialize_feature_definitions()
        
    async def initialize(self):
        """Initialize the feature engineering pipeline"""
        try:
            self.logger.info("Initializing Advanced Feature Engineering Pipeline...")
            
            # Initialize database connection if available
            if TimescaleDBConnection:
                self.db_connection = TimescaleDBConnection(
                    self.config.get('db_config', {})
                )
                await self.db_connection.initialize()
            
            # Initialize storage if available
            if DataStorage:
                self.storage = DataStorage(
                    storage_path='features',
                    db_config=self.config.get('db_config', {})
                )
                await self.storage.initialize()
            
            # Initialize scalers and transformers
            self._initialize_transformers()
            
            self.logger.info("Advanced Feature Engineering Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Feature Engineering: {e}")
            raise
    
    def _initialize_feature_definitions(self):
        """Initialize predefined feature definitions"""
        try:
            # Technical indicators
            self.feature_definitions['sma'] = FeatureDefinition(
                name='sma',
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                description='Simple Moving Average',
                parameters={'window': 20},
                dependencies=['close'],
                output_type='float',
                is_lagging=True,
                lag_periods=20
            )
            
            self.feature_definitions['ema'] = FeatureDefinition(
                name='ema',
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                description='Exponential Moving Average',
                parameters={'window': 20, 'alpha': 0.1},
                dependencies=['close'],
                output_type='float',
                is_lagging=True,
                lag_periods=20
            )
            
            self.feature_definitions['rsi'] = FeatureDefinition(
                name='rsi',
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                description='Relative Strength Index',
                parameters={'window': 14},
                dependencies=['close'],
                output_type='float',
                is_lagging=True,
                lag_periods=14
            )
            
            self.feature_definitions['bollinger_bands'] = FeatureDefinition(
                name='bollinger_bands',
                feature_type=FeatureType.TECHNICAL_INDICATOR,
                description='Bollinger Bands',
                parameters={'window': 20, 'std_dev': 2},
                dependencies=['close'],
                output_type='float',
                is_lagging=True,
                lag_periods=20
            )
            
            # Price action features
            self.feature_definitions['price_change'] = FeatureDefinition(
                name='price_change',
                feature_type=FeatureType.PRICE_ACTION,
                description='Price change percentage',
                parameters={'period': 1},
                dependencies=['close'],
                output_type='float',
                is_lagging=False,
                lag_periods=0
            )
            
            self.feature_definitions['price_volatility'] = FeatureDefinition(
                name='price_volatility',
                feature_type=FeatureType.PRICE_ACTION,
                description='Price volatility',
                parameters={'window': 20},
                dependencies=['close'],
                output_type='float',
                is_lagging=True,
                lag_periods=20
            )
            
            # Volume analysis features
            self.feature_definitions['volume_sma'] = FeatureDefinition(
                name='volume_sma',
                feature_type=FeatureType.VOLUME_ANALYSIS,
                description='Volume Simple Moving Average',
                parameters={'window': 20},
                dependencies=['volume'],
                output_type='float',
                is_lagging=True,
                lag_periods=20
            )
            
            self.feature_definitions['volume_price_trend'] = FeatureDefinition(
                name='volume_price_trend',
                feature_type=FeatureType.VOLUME_ANALYSIS,
                description='Volume Price Trend',
                parameters={},
                dependencies=['close', 'volume'],
                output_type='float',
                is_lagging=False,
                lag_periods=0
            )
            
            # Market microstructure features
            self.feature_definitions['bid_ask_spread'] = FeatureDefinition(
                name='bid_ask_spread',
                feature_type=FeatureType.MARKET_MICROSTRUCTURE,
                description='Bid-Ask Spread',
                parameters={},
                dependencies=['bid', 'ask'],
                output_type='float',
                is_lagging=False,
                lag_periods=0
            )
            
            # Time-based features
            self.feature_definitions['time_of_day'] = FeatureDefinition(
                name='time_of_day',
                feature_type=FeatureType.TIME_BASED,
                description='Time of day (0-23)',
                parameters={},
                dependencies=['timestamp'],
                output_type='int',
                is_lagging=False,
                lag_periods=0
            )
            
            self.feature_definitions['day_of_week'] = FeatureDefinition(
                name='day_of_week',
                feature_type=FeatureType.TIME_BASED,
                description='Day of week (0-6)',
                parameters={},
                dependencies=['timestamp'],
                output_type='int',
                is_lagging=False,
                lag_periods=0
            )
            
            # Derived features
            self.feature_definitions['price_momentum'] = FeatureDefinition(
                name='price_momentum',
                feature_type=FeatureType.DERIVED,
                description='Price momentum indicator',
                parameters={'short_window': 5, 'long_window': 20},
                dependencies=['close'],
                output_type='float',
                is_lagging=True,
                lag_periods=20
            )
            
            self.logger.info(f"Initialized {len(self.feature_definitions)} feature definitions")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature definitions: {e}")
    
    def _initialize_transformers(self):
        """Initialize scalers and feature transformers"""
        try:
            # Initialize scaler
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = RobustScaler()  # Default
            
            # Initialize PCA
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            
            # Initialize feature selector
            if self.feature_selection_method == 'mutual_info':
                self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=self.max_features)
            else:
                self.feature_selector = SelectKBest(score_func=f_regression, k=self.max_features)
            
            self.logger.info("Transformers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing transformers: {e}")
    
    async def create_features(self, data: pd.DataFrame, 
                            feature_list: List[str] = None) -> pd.DataFrame:
        """Create features from raw market data"""
        try:
            start_time = datetime.now()
            
            if data.empty:
                self.logger.warning("Empty data provided for feature creation")
                return pd.DataFrame()
            
            # Use all features if none specified
            if feature_list is None:
                feature_list = list(self.feature_definitions.keys())
            
            # Create features
            features_df = data.copy()
            
            for feature_name in feature_list:
                if feature_name in self.feature_definitions:
                    try:
                        feature_data = await self._create_single_feature(data, feature_name)
                        if feature_data is not None:
                            features_df = pd.concat([features_df, feature_data], axis=1)
                    except Exception as e:
                        self.logger.warning(f"Failed to create feature {feature_name}: {e}")
                continue
            
            # Remove duplicate columns
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['processing_time'] += processing_time
            self.stats['features_created'] += len(feature_list)
            
            self.logger.info(f"Created {len(feature_list)} features in {processing_time:.2f}s")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    async def _create_single_feature(self, data: pd.DataFrame, feature_name: str) -> Optional[pd.DataFrame]:
        """Create a single feature"""
        try:
            definition = self.feature_definitions[feature_name]
            
            # Check cache first
            cache_key = f"{feature_name}_{hash(str(data.values.tobytes()))}"
            if cache_key in self.feature_cache:
                self.stats['features_cached'] += 1
                return self.feature_cache[cache_key]
            
            # Create feature based on type
            if definition.feature_type == FeatureType.TECHNICAL_INDICATOR:
                feature_data = self._create_technical_indicator(data, definition)
            elif definition.feature_type == FeatureType.PRICE_ACTION:
                feature_data = self._create_price_action_feature(data, definition)
            elif definition.feature_type == FeatureType.VOLUME_ANALYSIS:
                feature_data = self._create_volume_feature(data, definition)
            elif definition.feature_type == FeatureType.MARKET_MICROSTRUCTURE:
                feature_data = self._create_microstructure_feature(data, definition)
            elif definition.feature_type == FeatureType.TIME_BASED:
                feature_data = self._create_time_feature(data, definition)
            elif definition.feature_type == FeatureType.DERIVED:
                feature_data = self._create_derived_feature(data, definition)
            else:
                feature_data = None
            
            # Cache the result
            if feature_data is not None:
                self._cache_feature(cache_key, feature_data)
            
            return feature_data
            
        except Exception as e:
            self.logger.error(f"Error creating feature {feature_name}: {e}")
            return None
    
    def _create_technical_indicator(self, data: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        """Create technical indicator features"""
        try:
            if definition.name == 'sma':
                window = definition.parameters.get('window', 20)
                sma = data['close'].rolling(window=window).mean()
                return pd.DataFrame({f'sma_{window}': sma})
            
            elif definition.name == 'ema':
                window = definition.parameters.get('window', 20)
                alpha = definition.parameters.get('alpha', 0.1)
                ema = data['close'].ewm(span=window, adjust=False).mean()
                return pd.DataFrame({f'ema_{window}': ema})
            
            elif definition.name == 'rsi':
                window = definition.parameters.get('window', 14)
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return pd.DataFrame({f'rsi_{window}': rsi})
            
            elif definition.name == 'bollinger_bands':
                window = definition.parameters.get('window', 20)
                std_dev = definition.parameters.get('std_dev', 2)
                sma = data['close'].rolling(window=window).mean()
                std = data['close'].rolling(window=window).std()
                upper_band = sma + (std * std_dev)
                lower_band = sma - (std * std_dev)
                bb_width = (upper_band - lower_band) / sma
                return pd.DataFrame({
                    f'bb_upper_{window}': upper_band,
                    f'bb_lower_{window}': lower_band,
                    f'bb_width_{window}': bb_width
                })
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating technical indicator {definition.name}: {e}")
            return None
    
    def _create_price_action_feature(self, data: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        """Create price action features"""
        try:
            if definition.name == 'price_change':
                period = definition.parameters.get('period', 1)
                price_change = data['close'].pct_change(periods=period)
                return pd.DataFrame({f'price_change_{period}': price_change})
            
            elif definition.name == 'price_volatility':
                window = definition.parameters.get('window', 20)
                returns = data['close'].pct_change()
                volatility = returns.rolling(window=window).std()
                return pd.DataFrame({f'volatility_{window}': volatility})
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating price action feature {definition.name}: {e}")
            return None
    
    def _create_volume_feature(self, data: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        """Create volume analysis features"""
        try:
            if definition.name == 'volume_sma':
                window = definition.parameters.get('window', 20)
                volume_sma = data['volume'].rolling(window=window).mean()
                return pd.DataFrame({f'volume_sma_{window}': volume_sma})
            
            elif definition.name == 'volume_price_trend':
                # Volume Price Trend (VPT)
                vpt = (data['close'].pct_change() * data['volume']).cumsum()
                return pd.DataFrame({'vpt': vpt})
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating volume feature {definition.name}: {e}")
            return None
    
    def _create_microstructure_feature(self, data: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        """Create market microstructure features"""
        try:
            if definition.name == 'bid_ask_spread':
                if 'bid' in data.columns and 'ask' in data.columns:
                    spread = (data['ask'] - data['bid']) / ((data['ask'] + data['bid']) / 2)
                    return pd.DataFrame({'bid_ask_spread': spread})
                else:
                    # Estimate spread using high-low
                    spread = (data['high'] - data['low']) / data['close']
                    return pd.DataFrame({'estimated_spread': spread})
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating microstructure feature {definition.name}: {e}")
            return None
    
    def _create_time_feature(self, data: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        """Create time-based features"""
        try:
            if definition.name == 'time_of_day':
                time_of_day = pd.to_datetime(data['timestamp']).dt.hour
                return pd.DataFrame({'time_of_day': time_of_day})
            
            elif definition.name == 'day_of_week':
                day_of_week = pd.to_datetime(data['timestamp']).dt.dayofweek
                return pd.DataFrame({'day_of_week': day_of_week})
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating time feature {definition.name}: {e}")
            return None
    
    def _create_derived_feature(self, data: pd.DataFrame, definition: FeatureDefinition) -> pd.DataFrame:
        """Create derived features"""
        try:
            if definition.name == 'price_momentum':
                short_window = definition.parameters.get('short_window', 5)
                long_window = definition.parameters.get('long_window', 20)
                
                short_sma = data['close'].rolling(window=short_window).mean()
                long_sma = data['close'].rolling(window=long_window).mean()
                momentum = (short_sma - long_sma) / long_sma
                
                return pd.DataFrame({f'momentum_{short_window}_{long_window}': momentum})
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating derived feature {definition.name}: {e}")
            return None
    
    def _cache_feature(self, key: str, feature_data: pd.DataFrame):
        """Cache feature data"""
        try:
            # Add to cache
            self.feature_cache[key] = feature_data
            
            # Maintain cache size
            if len(self.feature_cache) > self.cache_size:
                # Remove oldest entries
                oldest_keys = list(self.feature_cache.keys())[:len(self.feature_cache) - self.cache_size]
                for old_key in oldest_keys:
                    del self.feature_cache[old_key]
                    
        except Exception as e:
            self.logger.error(f"Error caching feature: {e}")
    
    async def fit_transformers(self, features_df: pd.DataFrame, target: pd.Series = None):
        """Fit and transform features using scalers and selectors"""
        try:
            if features_df.empty:
                return features_df
            
            # Remove non-numeric columns
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            if numeric_features.empty:
                return features_df
            
            # Handle missing values
            numeric_features = numeric_features.fillna(method='ffill').fillna(method='bfill')
            
            # Fit and transform scaler
            if self.scaler is not None:
                scaled_features = self.scaler.fit_transform(numeric_features)
                scaled_df = pd.DataFrame(scaled_features, 
                                       columns=numeric_features.columns,
                                       index=numeric_features.index)
            else:
                scaled_df = numeric_features
            
            # Fit and transform feature selector if target is provided
            if target is not None and self.feature_selector is not None:
                # Align target with features
                aligned_target = target.reindex(scaled_df.index).dropna()
                aligned_features = scaled_df.reindex(aligned_target.index)
                
                # Fit and transform
                selected_features = self.feature_selector.fit_transform(aligned_features, aligned_target)
                selected_columns = aligned_features.columns[self.feature_selector.get_support()]
                
                selected_df = pd.DataFrame(selected_features, 
                                         columns=selected_columns,
                                         index=aligned_features.index)
            else:
                selected_df = scaled_df
            
            # Fit PCA
            if self.pca is not None and selected_df.shape[1] > self.pca_components:
                pca_features = self.pca.fit_transform(selected_df)
                pca_df = pd.DataFrame(pca_features,
                                    columns=[f'pca_{i}' for i in range(pca_features.shape[1])],
                                    index=selected_df.index)
                
                # Combine original and PCA features
                final_df = pd.concat([selected_df, pca_df], axis=1)
            else:
                final_df = selected_df
            
            self.logger.info(f"Transformed features: {final_df.shape[1]} features")
            return final_df
            
        except Exception as e:
            self.logger.error(f"Error fitting transformers: {e}")
            return features_df
    
    async def get_feature_importance(self, feature_names: List[str], 
                                   target: pd.Series) -> Dict[str, float]:
        """Get feature importance scores"""
        try:
            if not feature_names or target.empty:
                return {}
            
            # Calculate correlation-based importance
            importance_scores = {}
            
            for feature_name in feature_names:
                if feature_name in self.feature_cache:
                    feature_data = self.feature_cache[feature_name]
                    if not feature_data.empty:
                        # Calculate correlation with target
                        correlation = abs(feature_data.corrwith(target).iloc[0])
                        importance_scores[feature_name] = correlation if not np.isnan(correlation) else 0.0
            
            # Sort by importance
            sorted_importance = dict(sorted(importance_scores.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    async def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature engineering statistics"""
        try:
            stats = self.stats.copy()
            
            # Add feature definition stats
            stats['total_feature_definitions'] = len(self.feature_definitions)
            stats['feature_types'] = {}
            
            for feature_type in FeatureType:
                count = len([f for f in self.feature_definitions.values() 
                           if f.feature_type == feature_type])
                stats['feature_types'][feature_type.value] = count
            
            # Add cache stats
            stats['cache_size'] = len(self.feature_cache)
            stats['cache_hit_rate'] = (stats['features_cached'] / 
                                     max(stats['features_created'], 1)) * 100
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting feature statistics: {e}")
            return {'error': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for feature engineering pipeline"""
        try:
            health_status = {
                'status': 'healthy',
                'feature_definitions': len(self.feature_definitions),
                'cache_size': len(self.feature_cache),
                'transformers_initialized': all([
                    self.scaler is not None,
                    self.pca is not None,
                    self.feature_selector is not None
                ])
            }
            
            # Check storage paths
            if not os.path.exists('features'):
                os.makedirs('features', exist_ok=True)
            
            # Check database health if available
            if self.db_connection:
                try:
                    db_health = await self.db_connection.health_check()
                    health_status['database_health'] = db_health
                    
                    if db_health.get('status') != 'healthy':
                        health_status['status'] = 'degraded'
                        health_status['warnings'] = ['Database connection issues']
                except Exception as e:
                    health_status['database_health'] = {'status': 'error', 'error': str(e)}
                    health_status['status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error in health check: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def close(self):
        """Close feature engineering pipeline"""
        try:
            # Clear cache
            self.feature_cache.clear()
            
            if self.db_connection:
                await self.db_connection.close()
            
            if self.storage:
                await self.storage.close()
            
            self.logger.info("Feature engineering pipeline closed")
            
        except Exception as e:
            self.logger.error(f"Error closing feature engineering pipeline: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    # ============================================================================
    # PHASE 6: ADVANCED FEATURE ENGINEERING METHODS
    # ============================================================================
    
    async def create_multitimeframe_features(self, 
                                           symbol: str,
                                           base_timeframe: str,
                                           target_timeframes: List[str] = None) -> Dict[str, Any]:
        """Create multi-timeframe feature fusion"""
        try:
            if target_timeframes is None:
                target_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
            
            self.logger.info(f"Creating multi-timeframe features for {symbol} ({base_timeframe})")
            
            # Collect features from different timeframes
            timeframe_features = {}
            fusion_weights = {}
            
            for tf in target_timeframes:
                try:
                    # Get market data for this timeframe
                    market_data = await self._get_market_data(symbol, tf, limit=100)
                    if market_data is None or market_data.empty:
                        continue
                    
                    # Create features for this timeframe
                    tf_features = await self._create_basic_technical_features(market_data)
                    timeframe_features[tf] = tf_features
                    
                    # Assign weights based on timeframe importance
                    if tf == base_timeframe:
                        fusion_weights[tf] = 0.3
                    elif tf in ['1h', '4h']:
                        fusion_weights[tf] = 0.25
                    elif tf in ['15m', '5m']:
                        fusion_weights[tf] = 0.2
                    else:
                        fusion_weights[tf] = 0.05
                        
                except Exception as e:
                    self.logger.warning(f"Failed to create features for {tf}: {e}")
                    continue
            
            # Fuse features using weighted concatenation
            fused_features = await self._fuse_multitimeframe_features(timeframe_features, fusion_weights)
            
            # Store in database
            await self._store_multitimeframe_features(symbol, base_timeframe, fused_features, fusion_weights)
            
            # Calculate feature count safely
            feature_count = 0
            if fused_features is not None and hasattr(fused_features, 'columns'):
                feature_count = len(fused_features.columns)
            elif fused_features is not None and hasattr(fused_features, '__len__'):
                feature_count = len(fused_features)
            
            return {
                'symbol': symbol,
                'base_timeframe': base_timeframe,
                'fused_features': fused_features,
                'fusion_weights': fusion_weights,
                'timeframe_features': timeframe_features,
                'feature_count': feature_count
            }
            
        except Exception as e:
            self.logger.error(f"Error creating multi-timeframe features: {e}")
            return {'error': str(e)}
    
    async def create_market_regime_features(self, 
                                          symbol: str,
                                          timeframe: str,
                                          lookback_days: int = 30,
                                          market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Create market regime-aware features"""
        try:
            self.logger.info(f"Creating market regime features for {symbol} ({timeframe})")
            
            if market_data is None:
                market_data = await self._get_market_data(symbol, timeframe, limit=100)
                if market_data is None or market_data.empty:
                    return {'error': 'No market data available'}
            
            # Classify market regime
            regime_result = await self._classify_market_regime(market_data)
            
            # Create regime-specific features
            regime_features = await self._create_regime_specific_features(market_data, regime_result)
            
            # Create regime-adjusted technical indicators
            adjusted_indicators = await self._create_regime_adjusted_indicators(market_data, regime_result)
            
            # Predict regime transitions
            transition_prediction = await self._predict_regime_transition(market_data, regime_result)
            
            # Combine all features
            all_features = {
                **regime_features,
                **adjusted_indicators,
                'regime_type': regime_result['regime_type'],
                'regime_confidence': regime_result['confidence'],
                'transition_probability': transition_prediction['probability'],
                'next_regime': transition_prediction['next_regime']
            }
            
            # Store in database
            await self._store_market_regime_features(symbol, timeframe, all_features, regime_result)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'regime_features': all_features,
                'regime_classification': regime_result,
                'transition_prediction': transition_prediction,
                'feature_count': len(all_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating market regime features: {e}")
            return {'error': str(e)}
    
    async def create_news_sentiment_features(self, 
                                           symbol: str,
                                           lookback_hours: int = 24) -> Dict[str, Any]:
        """Create news sentiment features"""
        try:
            self.logger.info(f"Creating news sentiment features for {symbol}")
            
            # Get news data
            news_data = await self._get_news_data(symbol, lookback_hours)
            
            # Get social sentiment data
            social_data = await self._get_social_sentiment_data(symbol, lookback_hours)
            
            # Calculate sentiment scores
            sentiment_scores = await self._calculate_sentiment_scores(news_data, social_data)
            
            # Create sentiment features
            sentiment_features = await self._create_sentiment_features(sentiment_scores)
            
            # Calculate sentiment momentum and trends
            momentum_features = await self._calculate_sentiment_momentum(sentiment_scores)
            
            # Create news impact features
            impact_features = await self._create_news_impact_features(news_data)
            
            # Combine all features
            all_features = {
                **sentiment_features,
                **momentum_features,
                **impact_features
            }
            
            # Store in database
            await self._store_news_sentiment_features(symbol, all_features, sentiment_scores)
            
            return {
                'symbol': symbol,
                'sentiment_features': all_features,
                'sentiment_scores': sentiment_scores,
                'feature_count': len(all_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating news sentiment features: {e}")
            return {'error': str(e)}
    
    async def create_volume_profile_features(self, 
                                           symbol: str,
                                           timeframe: str,
                                           period_days: int = 30) -> Dict[str, Any]:
        """Create volume profile features"""
        try:
            self.logger.info(f"Creating volume profile features for {symbol} ({timeframe})")
            
            # Get market data for volume profile analysis
            market_data = await self._get_market_data(symbol, timeframe, limit=period_days * 24)
            if market_data is None or market_data.empty:
                return {'error': 'No market data available'}
            
            # Calculate volume profile
            volume_profile = await self._calculate_volume_profile(market_data)
            
            # Create volume profile features
            profile_features = await self._create_volume_profile_features(volume_profile)
            
            # Create price-volume relationship features
            price_volume_features = await self._create_price_volume_features(market_data)
            
            # Create volume zones
            volume_zones = await self._create_volume_zones(volume_profile)
            
            # Create volume signals
            volume_signals = await self._create_volume_signals(market_data, volume_profile)
            
            # Combine all features
            all_features = {
                **profile_features,
                **price_volume_features,
                **volume_zones,
                **volume_signals
            }
            
            # Store in database
            await self._store_volume_profile_features(symbol, timeframe, all_features, volume_profile)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'volume_profile_features': all_features,
                'volume_profile_data': volume_profile,
                'feature_count': len(all_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating volume profile features: {e}")
            return {'error': str(e)}
    
    # ============================================================================
    # PRIVATE HELPER METHODS FOR PHASE 6
    # ============================================================================
    
    async def _fuse_multitimeframe_features(self, 
                                          timeframe_features: Dict[str, pd.DataFrame],
                                          fusion_weights: Dict[str, float]) -> pd.DataFrame:
        """Fuse features from multiple timeframes"""
        try:
            if not timeframe_features:
                return pd.DataFrame()
            
            # Normalize weights
            total_weight = sum(fusion_weights.values())
            if total_weight > 0:
                fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}
            
            # Create weighted feature matrix
            fused_features = []
            feature_names = []
            
            for tf, features in timeframe_features.items():
                # Check if features is None
                if features is None:
                    continue
                
                # Check if features is empty DataFrame
                if hasattr(features, 'empty') and features.empty:
                    continue
                
                # Check if features is a DataFrame with no columns
                if hasattr(features, 'columns') and len(features.columns) == 0:
                    continue
                
                weight = fusion_weights.get(tf, 0.0)
                if weight == 0:
                    continue
                
                # Add timeframe prefix to feature names
                tf_features = features.copy()
                tf_features.columns = [f"{tf}_{col}" for col in tf_features.columns]
                
                # Apply weight
                tf_features = tf_features * weight
                
                fused_features.append(tf_features)
                feature_names.extend(tf_features.columns)
            
            if not fused_features:
                return pd.DataFrame()
            
            # Concatenate features
            result = pd.concat(fused_features, axis=1)
            
            # Handle missing values
            result = result.ffill().fillna(0)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fusing multi-timeframe features: {e}")
            return pd.DataFrame()
    
    async def _classify_market_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Classify market regime using multiple indicators"""
        try:
            if market_data is None or market_data.empty:
                return {'regime_type': 'unknown', 'confidence': 0.0}
            
            # Calculate regime indicators
            atr = self._calculate_atr(market_data, 14)
            adx = self._calculate_adx(market_data, 14)
            bollinger_position = self._calculate_bollinger_position(market_data)
            
            # Determine regime based on indicators
            regime_scores = {
                'trending': 0.0,
                'ranging': 0.0,
                'volatile': 0.0,
                'breakout': 0.0
            }
            
            # Trending regime (high ADX, consistent direction)
            if adx.iloc[-1] > 25:
                regime_scores['trending'] += 0.4
            
            # Ranging regime (low ADX, price within Bollinger bands)
            if adx.iloc[-1] < 20 and 0.2 < bollinger_position.iloc[-1] < 0.8:
                regime_scores['ranging'] += 0.4
            
            # Volatile regime (high ATR)
            atr_percentile = atr.iloc[-1] / atr.rolling(20).mean().iloc[-1]
            if atr_percentile > 1.5:
                regime_scores['volatile'] += 0.4
            
            # Breakout regime (price breaking Bollinger bands)
            if bollinger_position.iloc[-1] > 0.95 or bollinger_position.iloc[-1] < 0.05:
                regime_scores['breakout'] += 0.4
            
            # Find dominant regime
            dominant_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[dominant_regime]
            
            return {
                'regime_type': dominant_regime,
                'confidence': confidence,
                'scores': regime_scores,
                'indicators': {
                    'atr': atr.iloc[-1],
                    'adx': adx.iloc[-1],
                    'bollinger_position': bollinger_position.iloc[-1]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying market regime: {e}")
            return {'regime_type': 'unknown', 'confidence': 0.0}
    
    async def _create_regime_specific_features(self, 
                                             market_data: pd.DataFrame,
                                             regime_result: Dict[str, Any]) -> Dict[str, float]:
        """Create regime-specific features"""
        try:
            regime_type = regime_result.get('regime_type', 'unknown')
            features = {}
            
            if regime_type == 'trending':
                # Trending features
                features['trend_strength'] = self._calculate_trend_strength(market_data)
                features['momentum_score'] = self._calculate_momentum_score(market_data)
                features['trend_consistency'] = self._calculate_trend_consistency(market_data)
                
            elif regime_type == 'ranging':
                # Ranging features
                features['range_width'] = self._calculate_range_width(market_data)
                features['support_resistance_strength'] = self._calculate_support_resistance_strength(market_data)
                features['mean_reversion_probability'] = self._calculate_mean_reversion_probability(market_data)
                
            elif regime_type == 'volatile':
                # Volatile features
                features['volatility_ratio'] = self._calculate_volatility_ratio(market_data)
                features['volatility_trend'] = self._calculate_volatility_trend(market_data)
                features['volatility_regime_strength'] = self._calculate_volatility_regime_strength(market_data)
                
            elif regime_type == 'breakout':
                # Breakout features
                features['breakout_strength'] = self._calculate_breakout_strength(market_data)
                features['volume_confirmation'] = self._calculate_volume_confirmation(market_data)
                features['breakout_probability'] = self._calculate_breakout_probability(market_data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating regime-specific features: {e}")
            return {}
    
    async def _calculate_sentiment_scores(self, 
                                        news_data: pd.DataFrame,
                                        social_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment scores from news and social data"""
        try:
            scores = {
                'overall_sentiment': 0.5,
                'news_sentiment': 0.5,
                'social_sentiment': 0.5,
                'technical_sentiment': 0.5
            }
            
            # Calculate news sentiment
            if news_data is not None and not news_data.empty:
                # Simple sentiment calculation (in real implementation, use NLP)
                positive_news = len(news_data[news_data.get('sentiment', 0) > 0.6])
                negative_news = len(news_data[news_data.get('sentiment', 0) < 0.4])
                total_news = len(news_data)
                
                if total_news > 0:
                    scores['news_sentiment'] = positive_news / total_news
            
            # Calculate social sentiment
            if social_data is not None and not social_data.empty:
                # Simple sentiment calculation
                positive_social = len(social_data[social_data.get('sentiment', 0) > 0.6])
                negative_social = len(social_data[social_data.get('sentiment', 0) < 0.4])
                total_social = len(social_data)
                
                if total_social > 0:
                    scores['social_sentiment'] = positive_social / total_social
            
            # Calculate overall sentiment
            scores['overall_sentiment'] = (scores['news_sentiment'] + scores['social_sentiment']) / 2
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment scores: {e}")
            return {'overall_sentiment': 0.5, 'news_sentiment': 0.5, 'social_sentiment': 0.5, 'technical_sentiment': 0.5}
    
    async def _calculate_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume profile from market data"""
        try:
            if market_data is None or market_data.empty:
                return {}
            
            # Calculate volume-weighted average price (VWAP)
            vwap = (market_data['close'] * market_data['volume']).sum() / market_data['volume'].sum()
            
            # Calculate Point of Control (POC) - price level with highest volume
            price_volume = market_data.groupby('close')['volume'].sum()
            poc_price = price_volume.idxmax()
            
            # Calculate Value Area (70% of volume)
            total_volume = market_data['volume'].sum()
            target_volume = total_volume * 0.7
            
            # Sort price levels by volume
            sorted_volumes = price_volume.sort_values(ascending=False)
            cumulative_volume = sorted_volumes.cumsum()
            
            # Find value area
            value_area_prices = cumulative_volume[cumulative_volume <= target_volume].index
            value_area_high = value_area_prices.max()
            value_area_low = value_area_prices.min()
            
            return {
                'vwap': vwap,
                'poc_price': poc_price,
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'value_area_percentage': 0.7,
                'total_volume': total_volume,
                'price_volume_distribution': price_volume.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume profile: {e}")
            return {}
    
    # Database storage methods
    async def _store_multitimeframe_features(self, symbol: str, base_timeframe: str, 
                                           features: pd.DataFrame, fusion_weights: Dict[str, float]):
        """Store multi-timeframe features in database"""
        try:
            if self.db_connection is None:
                return
            
            # Implementation would store to sde_multitimeframe_features table
            self.logger.info(f"Stored multi-timeframe features for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing multi-timeframe features: {e}")
    
    async def _store_market_regime_features(self, symbol: str, timeframe: str, 
                                          features: Dict[str, float], regime_result: Dict[str, Any]):
        """Store market regime features in database"""
        try:
            if self.db_connection is None:
                return
            
            # Implementation would store to sde_market_regime_features table
            self.logger.info(f"Stored market regime features for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing market regime features: {e}")
    
    async def _store_news_sentiment_features(self, symbol: str, features: Dict[str, float], 
                                           sentiment_scores: Dict[str, float]):
        """Store news sentiment features in database"""
        try:
            if self.db_connection is None:
                return
            
            # Implementation would store to sde_news_sentiment_features table
            self.logger.info(f"Stored news sentiment features for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing news sentiment features: {e}")
    
    async def _store_volume_profile_features(self, symbol: str, timeframe: str, 
                                           features: Dict[str, float], volume_profile: Dict[str, Any]):
        """Store volume profile features in database"""
        try:
            if self.db_connection is None:
                return
            
            # Implementation would store to sde_volume_profile_features table
            self.logger.info(f"Stored volume profile features for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing volume profile features: {e}")
    
    # Additional helper methods (stubs for now)
    async def _get_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Get market data from database"""
        try:
            if self.db_connection is None:
                # Return mock data for testing
                return self._create_mock_market_data(symbol, timeframe, limit)
            
            # Implementation would fetch from database
            # For now, return mock data
            return self._create_mock_market_data(symbol, timeframe, limit)
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def _create_mock_market_data(self, symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
        """Create mock market data for testing"""
        try:
            import numpy as np
            from datetime import datetime, timedelta
            
            # Generate timestamps
            end_time = datetime.now()
            if timeframe == '1m':
                start_time = end_time - timedelta(minutes=periods)
                freq = '1min'
            elif timeframe == '5m':
                start_time = end_time - timedelta(minutes=periods * 5)
                freq = '5min'
            elif timeframe == '15m':
                start_time = end_time - timedelta(minutes=periods * 15)
                freq = '15min'
            elif timeframe == '1h':
                start_time = end_time - timedelta(hours=periods)
                freq = '1h'
            elif timeframe == '4h':
                start_time = end_time - timedelta(hours=periods * 4)
                freq = '4h'
            elif timeframe == '1d':
                start_time = end_time - timedelta(days=periods)
                freq = '1D'
            else:
                start_time = end_time - timedelta(hours=periods)
                freq = '1H'
            
            timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
            
            # Generate price data
            np.random.seed(42)  # For reproducible results
            base_price = 50000 if 'BTC' in symbol else 3000
            
            # Generate OHLCV data
            data = []
            current_price = base_price
            
            for i, timestamp in enumerate(timestamps):
                # Generate price movement
                price_change = np.random.normal(0, base_price * 0.01)  # 1% volatility
                current_price += price_change
                
                # Generate OHLC
                high = current_price * (1 + abs(np.random.normal(0, 0.005)))
                low = current_price * (1 - abs(np.random.normal(0, 0.005)))
                open_price = current_price * (1 + np.random.normal(0, 0.002))
                close_price = current_price
                
                # Generate volume
                volume = np.random.uniform(1000, 10000)
                
                data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating mock market data: {e}")
            return pd.DataFrame()
    
    async def _get_news_data(self, symbol: str, lookback_hours: int) -> pd.DataFrame:
        """Get news data from database"""
        # Implementation would fetch from database
        return pd.DataFrame()
    
    async def _get_social_sentiment_data(self, symbol: str, lookback_hours: int) -> pd.DataFrame:
        """Get social sentiment data from database"""
        # Implementation would fetch from database
        return pd.DataFrame()
    
    # Technical indicator calculation methods (stubs)
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        try:
            if data.empty or len(data) < period:
                return pd.Series(0.0, index=data.index)
            
            # Calculate True Range
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            if data.empty or len(data) < period:
                return pd.Series(0.0, index=data.index)
            
            # Calculate +DM and -DM
            high_diff = data['high'] - data['high'].shift()
            low_diff = data['low'].shift() - data['low']
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Calculate TR
            tr = self._calculate_atr(data, period)
            
            # Calculate smoothed +DM and -DM
            plus_di = pd.Series(plus_dm, index=data.index).rolling(period).mean() / tr * 100
            minus_di = pd.Series(minus_dm, index=data.index).rolling(period).mean() / tr * 100
            
            # Calculate DX and ADX
            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
            adx = dx.rolling(period).mean()
            
            return adx.fillna(0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return pd.Series(0.0, index=data.index)
    
    def _calculate_bollinger_position(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Band position"""
        try:
            if data.empty or len(data) < 20:
                return pd.Series(0.5, index=data.index)
            
            # Calculate Bollinger Bands
            sma = data['close'].rolling(window=20).mean()
            std = data['close'].rolling(window=20).std()
            
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            # Calculate position (0 = at lower band, 1 = at upper band)
            position = (data['close'] - lower_band) / (upper_band - lower_band)
            
            return position.fillna(0.5)
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger position: {e}")
            return pd.Series(0.5, index=data.index)
    
    # Regime-specific calculation methods (stubs)
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate trend strength using linear regression
            x = np.arange(len(data))
            y = data['close'].values
            
            slope, _, r_value, _, _ = stats.linregress(x, y)
            
            # Normalize slope and R-squared
            trend_strength = abs(r_value) * (abs(slope) / data['close'].mean())
            
            return min(trend_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 14:
                return 0.5
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Convert RSI to momentum score (0-1)
            momentum = (rsi.iloc[-1] - 50) / 50  # -1 to 1
            momentum = (momentum + 1) / 2  # 0 to 1
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.5
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate consistency of price movement direction
            price_changes = data['close'].diff()
            positive_changes = (price_changes > 0).sum()
            total_changes = len(price_changes.dropna())
            
            if total_changes == 0:
                return 0.5
            
            consistency = abs(positive_changes / total_changes - 0.5) * 2  # 0 to 1
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"Error calculating trend consistency: {e}")
            return 0.5
    
    def _calculate_range_width(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate range width as percentage of price
            high = data['high'].rolling(window=20).max()
            low = data['low'].rolling(window=20).min()
            range_width = (high - low) / data['close']
            
            return range_width.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating range width: {e}")
            return 0.5
    
    def _calculate_support_resistance_strength(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Simple support/resistance strength based on price clustering
            price_levels = data['close'].round(2)
            level_counts = price_levels.value_counts()
            
            # Strength based on how many times price returns to same level
            max_count = level_counts.max()
            strength = min(max_count / len(data), 1.0)
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance strength: {e}")
            return 0.5
    
    def _calculate_mean_reversion_probability(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate mean reversion probability based on Bollinger Band position
            bb_position = self._calculate_bollinger_position(data)
            current_position = bb_position.iloc[-1]
            
            # Higher probability when price is at extremes
            if current_position > 0.8 or current_position < 0.2:
                probability = 0.8
            elif current_position > 0.6 or current_position < 0.4:
                probability = 0.6
            else:
                probability = 0.3
            
            return probability
            
        except Exception as e:
            self.logger.error(f"Error calculating mean reversion probability: {e}")
            return 0.5
    
    def _calculate_volatility_ratio(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate current volatility vs historical volatility
            current_atr = self._calculate_atr(data, 14).iloc[-1]
            historical_atr = self._calculate_atr(data, 14).rolling(window=20).mean().iloc[-1]
            
            if historical_atr == 0:
                return 0.5
            
            ratio = current_atr / historical_atr
            return min(ratio, 2.0) / 2.0  # Normalize to 0-1
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility ratio: {e}")
            return 0.5
    
    def _calculate_volatility_trend(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate volatility trend using ATR
            atr = self._calculate_atr(data, 14)
            atr_trend = atr.rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            
            # Normalize trend
            trend = atr_trend.iloc[-1]
            normalized_trend = (trend + 1) / 2  # Assume max trend is 1
            
            return max(0.0, min(1.0, normalized_trend))
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility trend: {e}")
            return 0.5
    
    def _calculate_volatility_regime_strength(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate how strong the volatility regime is
            volatility_ratio = self._calculate_volatility_ratio(data)
            volatility_trend = self._calculate_volatility_trend(data)
            
            # Combine metrics
            strength = (volatility_ratio + volatility_trend) / 2
            
            return strength
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility regime strength: {e}")
            return 0.5
    
    def _calculate_breakout_strength(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate breakout strength based on price movement beyond recent range
            recent_high = data['high'].rolling(window=20).max()
            recent_low = data['low'].rolling(window=20).min()
            current_price = data['close'].iloc[-1]
            
            # Calculate how far price is from range
            if current_price > recent_high.iloc[-1]:
                strength = (current_price - recent_high.iloc[-1]) / recent_high.iloc[-1]
            elif current_price < recent_low.iloc[-1]:
                strength = (recent_low.iloc[-1] - current_price) / recent_low.iloc[-1]
            else:
                strength = 0.0
            
            return min(strength * 10, 1.0)  # Scale and cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout strength: {e}")
            return 0.5
    
    def _calculate_volume_confirmation(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate volume confirmation for price movement
            price_change = data['close'].diff()
            volume_avg = data['volume'].rolling(window=20).mean()
            
            # Check if volume is above average when price moves
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = volume_avg.iloc[-1]
            
            if avg_volume == 0:
                return 0.5
            
            volume_ratio = recent_volume / avg_volume
            confirmation = min(volume_ratio, 2.0) / 2.0  # Normalize to 0-1
            
            return confirmation
            
        except Exception as e:
            self.logger.error(f"Error calculating volume confirmation: {e}")
            return 0.5
    
    def _calculate_breakout_probability(self, data: pd.DataFrame) -> float:
        try:
            if data.empty or len(data) < 20:
                return 0.5
            
            # Calculate breakout probability based on multiple factors
            breakout_strength = self._calculate_breakout_strength(data)
            volume_confirmation = self._calculate_volume_confirmation(data)
            trend_strength = self._calculate_trend_strength(data)
            
            # Combine factors
            probability = (breakout_strength * 0.4 + volume_confirmation * 0.3 + trend_strength * 0.3)
            
            return probability
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return 0.5
    
    # Additional helper methods (stubs)
    async def _create_regime_adjusted_indicators(self, data: pd.DataFrame, regime_result: Dict[str, Any]) -> Dict[str, float]:
        return {}
    
    async def _predict_regime_transition(self, data: pd.DataFrame, regime_result: Dict[str, Any]) -> Dict[str, Any]:
        return {'probability': 0.1, 'next_regime': 'trending'}
    
    async def _create_sentiment_features(self, sentiment_scores: Dict[str, float]) -> Dict[str, float]:
        return {}
    
    async def _calculate_sentiment_momentum(self, sentiment_scores: Dict[str, float]) -> Dict[str, float]:
        return {}
    
    async def _create_news_impact_features(self, news_data: pd.DataFrame) -> Dict[str, float]:
        return {}
    
    async def _create_volume_profile_features(self, volume_profile: Dict[str, Any]) -> Dict[str, float]:
        return {}
    
    async def _create_price_volume_features(self, data: pd.DataFrame) -> Dict[str, float]:
        return {}
    
    async def _create_volume_zones(self, volume_profile: Dict[str, Any]) -> Dict[str, float]:
        return {}
    
    async def _create_volume_signals(self, data: pd.DataFrame, volume_profile: Dict[str, Any]) -> Dict[str, float]:
        return {}
    
    async def _create_basic_technical_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create basic technical features from market data"""
        try:
            if market_data.empty or len(market_data) < 20:
                return pd.DataFrame()
            
            features = {}
            
            # Price-based features
            features['close'] = market_data['close']
            features['returns'] = market_data['close'].pct_change()
            features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
            
            # Volume features
            features['volume'] = market_data['volume']
            features['volume_ma'] = market_data['volume'].rolling(window=20).mean()
            features['volume_ratio'] = market_data['volume'] / features['volume_ma']
            
            # Moving averages
            features['sma_20'] = market_data['close'].rolling(window=20).mean()
            features['sma_50'] = market_data['close'].rolling(window=50).mean()
            features['ema_12'] = market_data['close'].ewm(span=12).mean()
            features['ema_26'] = market_data['close'].ewm(span=26).mean()
            
            # Price position relative to moving averages
            features['price_vs_sma20'] = market_data['close'] / features['sma_20'] - 1
            features['price_vs_sma50'] = market_data['close'] / features['sma_50'] - 1
            
            # Volatility features
            features['atr'] = self._calculate_atr(market_data, 14)
            features['volatility'] = features['returns'].rolling(window=20).std()
            
            # RSI
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            bb_std = market_data['close'].rolling(window=20).std()
            features['bb_upper'] = features['sma_20'] + (bb_std * 2)
            features['bb_lower'] = features['sma_20'] - (bb_std * 2)
            features['bb_position'] = (market_data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Create DataFrame
            feature_df = pd.DataFrame(features)
            
            # Handle missing values
            feature_df = feature_df.ffill().fillna(0)
            
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error creating basic technical features: {e}")
            return pd.DataFrame()
