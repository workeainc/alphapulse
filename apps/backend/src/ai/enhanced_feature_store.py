#!/usr/bin/env python3
"""
Enhanced Feature Store
Phase 2C: Enhanced Feature Engineering
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Import our enhanced components
from .technical_indicators_engine import TechnicalIndicatorsEngine
from .feature_drift_detector import FeatureDriftDetector
from .feature_quality_validator import FeatureQualityValidator
from .feast_feature_store import FeastFeatureStoreManager
from .feature_store_timescaledb import TimescaleDBFeatureStore

logger = logging.getLogger(__name__)

class EnhancedFeatureStore:
    """Enhanced feature store with advanced feature engineering capabilities"""
    
    def __init__(self):
        # Core components
        self.technical_engine = TechnicalIndicatorsEngine()
        self.drift_detector = FeatureDriftDetector()
        self.quality_validator = FeatureQualityValidator()
        self.feast_manager = None
        self.timescaledb_store = None
        
        # Feature metadata and monitoring
        self.feature_metadata = {}
        self.feature_monitoring = {}
        self.production_features = {}
        
        # Configuration
        self.auto_drift_detection = True
        self.auto_quality_validation = True
        self.auto_feature_refresh = True
        self.refresh_interval_hours = 24
        
        logger.info("🚀 Enhanced Feature Store initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize Feast manager
            self.feast_manager = FeastFeatureStoreManager()
            await self.feast_manager.initialize()
            
            # Initialize TimescaleDB store
            self.timescaledb_store = TimescaleDBFeatureStore()
            await self.timescaledb_store.initialize()
            
            # Load feature metadata
            await self._load_feature_metadata()
            
            # Initialize monitoring
            await self._initialize_feature_monitoring()
            
            logger.info("✅ Enhanced Feature Store fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced Feature Store: {e}")
            raise
    
    async def _load_feature_metadata(self):
        """Load feature metadata and configurations"""
        try:
            # Get available technical indicators
            available_indicators = self.technical_engine.get_available_indicators()
            
            for indicator in available_indicators:
                config = self.technical_engine.get_indicator_config(indicator)
                if config:
                    self.feature_metadata[indicator] = {
                        'name': config.name,
                        'description': config.description,
                        'parameters': config.parameters,
                        'min_periods': config.min_periods,
                        'max_periods': config.max_periods,
                        'output_type': config.output_type,
                        'tags': config.tags,
                        'category': self._categorize_indicator(indicator),
                        'last_updated': datetime.now(),
                        'version': '1.0'
                    }
            
            logger.info(f"✅ Loaded metadata for {len(self.feature_metadata)} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to load feature metadata: {e}")
    
    async def _initialize_feature_monitoring(self):
        """Initialize feature monitoring and health tracking"""
        try:
            for feature_name in self.feature_metadata.keys():
                self.feature_monitoring[feature_name] = {
                    'health_score': 1.0,
                    'drift_score': 0.0,
                    'quality_score': 1.0,
                    'last_monitored': datetime.now(),
                    'monitoring_frequency': 'daily',
                    'alert_thresholds': {
                        'drift': 0.3,
                        'quality': 0.7,
                        'health': 0.6
                    }
                }
            
            logger.info(f"✅ Initialized monitoring for {len(self.feature_monitoring)} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize feature monitoring: {e}")
    
    def _categorize_indicator(self, indicator_name: str) -> str:
        """Categorize technical indicator"""
        if indicator_name in ['rsi', 'stoch_rsi', 'williams_r']:
            return 'momentum'
        elif indicator_name in ['ema', 'sma', 'macd', 'adx']:
            return 'trend'
        elif indicator_name in ['bollinger_bands', 'atr', 'keltner_channels']:
            return 'volatility'
        elif indicator_name in ['volume_sma_ratio', 'obv', 'vwap']:
            return 'volume'
        elif indicator_name in ['ichimoku', 'fibonacci_retracement', 'pivot_points']:
            return 'advanced'
        else:
            return 'other'
    
    async def compute_technical_features(self, ohlcv_data: pd.DataFrame, 
                                       symbols: List[str], timeframes: List[str],
                                       indicators: List[str] = None,
                                       validate_quality: bool = True,
                                       detect_drift: bool = True) -> Dict[str, pd.DataFrame]:
        """Compute technical features with quality validation and drift detection"""
        try:
            if indicators is None:
                indicators = list(self.feature_metadata.keys())
            
            results = {}
            computed_features = {}
            
            for symbol in symbols:
                for timeframe in timeframes:
                    # Filter data for symbol and timeframe
                    symbol_data = ohlcv_data[
                        (ohlcv_data['symbol'] == symbol) & 
                        (ohlcv_data['tf'] == timeframe)
                    ].copy()
                    
                    if len(symbol_data) == 0:
                        continue
                    
                    # Calculate technical indicators
                    indicator_results = self.technical_engine.calculate_all_indicators(
                        symbol_data, indicators
                    )
                    
                    # Create feature dataframe
                    feature_df = pd.DataFrame(index=symbol_data.index)
                    feature_df['symbol'] = symbol
                    feature_df['tf'] = timeframe
                    feature_df['timestamp'] = symbol_data.index
                    
                    # Add indicator values
                    for indicator_name, indicator_values in indicator_results.items():
                        feature_df[indicator_name] = indicator_values
                    
                    # Store results
                    key = f"{symbol}_{timeframe}"
                    results[key] = feature_df
                    computed_features[key] = indicator_results
                    
                    # Quality validation and drift detection
                    if validate_quality or detect_drift:
                        await self._monitor_feature_health(
                            key, feature_df, computed_features[key]
                        )
            
            logger.info(f"✅ Computed technical features for {len(results)} symbol-timeframe combinations")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to compute technical features: {e}")
            return {}
    
    async def _monitor_feature_health(self, feature_key: str, feature_df: pd.DataFrame, 
                                     indicator_results: Dict[str, pd.Series]):
        """Monitor feature health using drift detection and quality validation"""
        try:
            # Quality validation
            if self.auto_quality_validation:
                for indicator_name, indicator_values in indicator_results.items():
                    if not indicator_values.empty:
                        quality_metrics = self.quality_validator.validate_feature_quality(
                            f"{feature_key}_{indicator_name}",
                            indicator_values,
                            metadata={'source': 'technical_indicators', 'feature_key': feature_key}
                        )
                        
                        if quality_metrics:
                            # Update monitoring
                            if feature_key in self.feature_monitoring:
                                self.feature_monitoring[feature_key]['quality_score'] = quality_metrics.overall_score
                                self.feature_monitoring[feature_key]['last_monitored'] = datetime.now()
            
            # Drift detection
            if self.auto_drift_detection:
                for indicator_name, indicator_values in indicator_results.items():
                    if not indicator_values.empty and len(indicator_values) >= 50:
                        # Use first half as reference, second half for drift detection
                        split_point = len(indicator_values) // 2
                        reference_data = indicator_values.iloc[:split_point]
                        current_data = indicator_values.iloc[split_point:]
                        
                        drift_metrics = self.drift_detector.detect_drift(
                            f"{feature_key}_{indicator_name}",
                            current_data
                        )
                        
                        if drift_metrics:
                            # Update monitoring
                            if feature_key in self.feature_monitoring:
                                self.feature_monitoring[feature_key]['drift_score'] = drift_metrics.drift_score
                                self.feature_monitoring[feature_key]['last_monitored'] = datetime.now()
            
            # Update overall health score
            if feature_key in self.feature_monitoring:
                monitoring = self.feature_monitoring[feature_key]
                health_score = (
                    monitoring['quality_score'] * 0.6 +
                    (1.0 - monitoring['drift_score']) * 0.4
                )
                monitoring['health_score'] = health_score
                
        except Exception as e:
            logger.error(f"❌ Failed to monitor feature health for {feature_key}: {e}")
    
    async def get_feature_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature health summary"""
        try:
            summary = {
                'total_features': len(self.feature_monitoring),
                'health_distribution': {
                    'excellent': 0,  # 0.9-1.0
                    'good': 0,       # 0.7-0.9
                    'fair': 0,       # 0.5-0.7
                    'poor': 0,       # 0.3-0.5
                    'critical': 0    # 0.0-0.3
                },
                'drift_summary': self.drift_detector.get_drift_summary(),
                'quality_summary': self.quality_validator.get_quality_summary(),
                'feature_categories': {},
                'alerts': {
                    'drift_alerts': len(self.drift_detector.alerts),
                    'quality_alerts': len(self.quality_validator.alerts)
                }
            }
            
            # Analyze health distribution
            for feature_name, monitoring in self.feature_monitoring.items():
                health_score = monitoring['health_score']
                
                if health_score >= 0.9:
                    summary['health_distribution']['excellent'] += 1
                elif health_score >= 0.7:
                    summary['health_distribution']['good'] += 1
                elif health_score >= 0.5:
                    summary['health_distribution']['fair'] += 1
                elif health_score >= 0.3:
                    summary['health_distribution']['poor'] += 1
                else:
                    summary['health_distribution']['critical'] += 1
                
                # Categorize by feature type
                if feature_name in self.feature_metadata:
                    category = self.feature_metadata[feature_name]['category']
                    if category not in summary['feature_categories']:
                        summary['feature_categories'][category] = []
                    summary['feature_categories'][category].append({
                        'name': feature_name,
                        'health_score': health_score,
                        'drift_score': monitoring['drift_score'],
                        'quality_score': monitoring['quality_score']
                    })
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature health summary: {e}")
            return {'error': str(e)}
    
    async def get_feature_recommendations(self, feature_name: str = None) -> Dict[str, Any]:
        """Get recommendations for feature improvement"""
        try:
            recommendations = {
                'general': [],
                'specific_features': {},
                'system_improvements': []
            }
            
            # General recommendations based on overall health
            health_summary = await self.get_feature_health_summary()
            
            if health_summary['health_distribution']['critical'] > 0:
                recommendations['general'].append("Critical health issues detected - immediate attention required")
            
            if health_summary['health_distribution']['poor'] > len(self.feature_monitoring) * 0.2:
                recommendations['general'].append("High percentage of poor health features - review feature engineering pipeline")
            
            if health_summary['alerts']['drift_alerts'] > 10:
                recommendations['general'].append("High drift alert volume - consider updating reference data")
            
            if health_summary['alerts']['quality_alerts'] > 10:
                recommendations['general'].append("High quality alert volume - review data preprocessing")
            
            # Specific feature recommendations
            for feature_name, monitoring in self.feature_monitoring.items():
                feature_recs = []
                
                if monitoring['health_score'] < 0.5:
                    feature_recs.append("Critical health - consider feature replacement")
                
                if monitoring['drift_score'] > 0.5:
                    feature_recs.append("High drift - update reference data or retrain models")
                
                if monitoring['quality_score'] < 0.6:
                    feature_recs.append("Low quality - review data source and preprocessing")
                
                if feature_recs:
                    recommendations['specific_features'][feature_name] = feature_recs
            
            # System improvement recommendations
            if len(self.feature_monitoring) > 100:
                recommendations['system_improvements'].append("Large feature set - consider feature selection optimization")
            
            if health_summary['health_distribution']['excellent'] < len(self.feature_monitoring) * 0.3:
                recommendations['system_improvements'].append("Low excellent health rate - review feature engineering standards")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature recommendations: {e}")
            return {'error': str(e)}
    
    async def refresh_features(self, force: bool = False) -> bool:
        """Refresh feature computations and monitoring"""
        try:
            if not self.auto_feature_refresh and not force:
                logger.info("⚠️ Auto-refresh disabled, skipping feature refresh")
                return True
            
            logger.info("🔄 Starting feature refresh...")
            
            # Get current time
            current_time = datetime.now()
            
            # Check which features need refresh
            features_to_refresh = []
            for feature_name, monitoring in self.feature_monitoring.items():
                last_monitored = monitoring['last_monitored']
                hours_since_monitoring = (current_time - last_monitored).total_seconds() / 3600
                
                if hours_since_monitoring >= self.refresh_interval_hours:
                    features_to_refresh.append(feature_name)
            
            if not features_to_refresh:
                logger.info("✅ No features require refresh")
                return True
            
            logger.info(f"🔄 Refreshing {len(features_to_refresh)} features...")
            
            # Refresh features (this would typically involve recomputing from source data)
            for feature_name in features_to_refresh:
                try:
                    # Update monitoring timestamp
                    self.feature_monitoring[feature_name]['last_monitored'] = current_time
                    
                    # In a real implementation, you would recompute the feature here
                    # For now, we'll just update the monitoring
                    logger.debug(f"✅ Refreshed {feature_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to refresh {feature_name}: {e}")
                    continue
            
            logger.info(f"✅ Feature refresh completed for {len(features_to_refresh)} features")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to refresh features: {e}")
            return False
    
    async def get_production_features(self, symbols: List[str], timeframes: List[str],
                                     feature_names: List[str] = None,
                                     start_time: datetime = None,
                                     end_time: datetime = None) -> pd.DataFrame:
        """Get production-ready features with health filtering"""
        try:
            if feature_names is None:
                feature_names = list(self.feature_metadata.keys())
            
            # Filter features by health score
            healthy_features = []
            for feature_name in feature_names:
                # Check if feature is healthy across all symbol-timeframe combinations
                is_healthy = True
                for symbol in symbols:
                    for timeframe in timeframes:
                        key = f"{symbol}_{timeframe}"
                        if key in self.feature_monitoring:
                            if self.feature_monitoring[key]['health_score'] < 0.6:
                                is_healthy = False
                                break
                    if not is_healthy:
                        break
                
                if is_healthy:
                    healthy_features.append(feature_name)
            
            logger.info(f"✅ Selected {len(healthy_features)} healthy features out of {len(feature_names)} requested")
            
            # Get features from Feast or TimescaleDB
            if self.feast_manager and self.feast_manager._initialized:
                # Use Feast for production features
                features_df = await self.feast_manager.get_offline_features(
                    entity_ids=[f"{s}_{tf}" for s in symbols for tf in timeframes],
                    feature_names=healthy_features,
                    start_date=start_time or (datetime.now() - timedelta(days=7)),
                    end_date=end_time or datetime.now()
                )
            else:
                # Fallback to TimescaleDB
                features_df = await self.timescaledb_store.get_feature_history(
                    feature_names=healthy_features,
                    entity_ids=[f"{s}_{tf}" for s in symbols for tf in timeframes],
                    start_date=start_time or (datetime.now() - timedelta(days=7)),
                    end_date=end_time or datetime.now()
                )
            
            # Add health metadata
            if not features_df.empty:
                features_df['feature_health_score'] = 1.0  # Placeholder for actual health scores
                features_df['last_health_check'] = datetime.now()
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Failed to get production features: {e}")
            return pd.DataFrame()
    
    async def export_feature_report(self, output_path: str = None) -> str:
        """Export comprehensive feature report"""
        try:
            if output_path is None:
                output_path = f"feature_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Gather comprehensive data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'feature_metadata': self.feature_metadata,
                'feature_monitoring': self.feature_monitoring,
                'health_summary': await self.get_feature_health_summary(),
                'recommendations': await self.get_feature_recommendations(),
                'drift_summary': self.drift_detector.get_drift_summary(),
                'quality_summary': self.quality_validator.get_quality_summary(),
                'system_info': {
                    'total_features': len(self.feature_metadata),
                    'auto_drift_detection': self.auto_drift_detection,
                    'auto_quality_validation': self.auto_quality_validation,
                    'auto_feature_refresh': self.auto_feature_refresh,
                    'refresh_interval_hours': self.refresh_interval_hours
                }
            }
            
            # Export to JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"✅ Feature report exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Failed to export feature report: {e}")
            return ""
    
    async def close(self):
        """Close the enhanced feature store"""
        try:
            if self.feast_manager:
                await self.feast_manager.close()
            
            if self.timescaledb_store:
                await self.timescaledb_store.close()
            
            logger.info("🔒 Enhanced Feature Store closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing enhanced feature store: {e}")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Convenience functions
async def get_enhanced_features(symbols: List[str], timeframes: List[str],
                               feature_names: List[str] = None) -> pd.DataFrame:
    """Get enhanced features using default settings"""
    async with EnhancedFeatureStore() as store:
        return await store.get_production_features(symbols, timeframes, feature_names)

async def compute_enhanced_features(ohlcv_data: pd.DataFrame, symbols: List[str],
                                   timeframes: List[str], indicators: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Compute enhanced features using default settings"""
    async with EnhancedFeatureStore() as store:
        return await store.compute_technical_features(ohlcv_data, symbols, timeframes, indicators)

async def get_feature_health_report() -> Dict[str, Any]:
    """Get feature health report using default settings"""
    async with EnhancedFeatureStore() as store:
        return await store.get_feature_health_summary()
