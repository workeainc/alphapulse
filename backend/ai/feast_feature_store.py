#!/usr/bin/env python3
"""
Feast Feature Store Manager
Phase 2B: Feast Framework Integration
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Feast imports
try:
    from feast import FeatureStore, FeatureService
    from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgresSource
    from feast.types import Float32, Float64, Int64, String, Bool
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logging.warning("⚠️ Feast not available - using fallback implementation")

# Import our existing components
from .feature_store_timescaledb import TimescaleDBFeatureStore
from .feast_config import get_feast_paths, FEAST_CONFIG

logger = logging.getLogger(__name__)

class FeastFeatureStoreManager:
    """Feast feature store manager for online/offline feature serving"""
    
    def __init__(self, feature_store_path: str = None):
        self.feature_store_path = feature_store_path or get_feast_paths()["feature_store_yaml"]
        self.feature_store = None
        self.timescaledb_store = None
        self._initialized = False
        
        logger.info("🚀 Feast Feature Store Manager initialized")
    
    async def initialize(self):
        """Initialize the Feast feature store"""
        try:
            if not FEAST_AVAILABLE:
                logger.warning("⚠️ Feast not available - using TimescaleDB fallback")
                await self._initialize_fallback()
                return
            
            # Initialize Feast feature store
            if Path(self.feature_store_path).exists():
                self.feature_store = FeatureStore(repo_path=str(Path(self.feature_store_path).parent))
                logger.info("✅ Feast feature store initialized")
            else:
                logger.warning("⚠️ Feast config not found - using TimescaleDB fallback")
                await self._initialize_fallback()
                return
            
            # Initialize TimescaleDB store as backup
            self.timescaledb_store = TimescaleDBFeatureStore()
            await self.timescaledb_store.initialize()
            
            self._initialized = True
            logger.info("✅ Feast Feature Store Manager fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Feast feature store: {e}")
            logger.info("🔄 Falling back to TimescaleDB implementation")
            await self._initialize_fallback()
    
    async def _initialize_fallback(self):
        """Initialize fallback TimescaleDB implementation"""
        try:
            self.timescaledb_store = TimescaleDBFeatureStore()
            await self.timescaledb_store.initialize()
            self._initialized = True
            logger.info("✅ TimescaleDB fallback initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB fallback: {e}")
            raise
    
    async def get_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        feature_service_name: str = None
    ) -> pd.DataFrame:
        """Get online features for real-time serving"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if self.feature_store and FEAST_AVAILABLE:
                # Use Feast online serving
                return await self._get_feast_online_features(
                    entity_ids, feature_names, feature_service_name
                )
            else:
                # Use TimescaleDB fallback
                return await self._get_timescaledb_online_features(
                    entity_ids, feature_names
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get online features: {e}")
            return pd.DataFrame()
    
    async def get_offline_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime,
        feature_service_name: str = None
    ) -> pd.DataFrame:
        """Get offline features for training and batch processing"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if self.feature_store and FEAST_AVAILABLE:
                # Use Feast offline serving
                return await self._get_feast_offline_features(
                    entity_ids, feature_names, start_date, end_date, feature_service_name
                )
            else:
                # Use TimescaleDB fallback
                return await self._get_timescaledb_offline_features(
                    entity_ids, feature_names, start_date, end_date
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to get offline features: {e}")
            return pd.DataFrame()
    
    async def _get_feast_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        feature_service_name: str = None
    ) -> pd.DataFrame:
        """Get online features using Feast"""
        try:
            # Create entity dataframe
            entity_df = pd.DataFrame({
                "symbol": [entity_id.split("_")[0] for entity_id in entity_ids],
                "tf": [entity_id.split("_")[1] if "_" in entity_id else "1h" for entity_id in entity_ids]
            })
            
            # Get features from Feast
            if feature_service_name:
                features = self.feature_store.get_online_features(
                    features=feature_service_name,
                    entity_rows=[entity_df.to_dict("records")]
                )
            else:
                features = self.feature_store.get_online_features(
                    features=feature_names,
                    entity_rows=[entity_df.to_dict("records")]
                )
            
            # Convert to DataFrame
            result_df = features.to_df()
            logger.info(f"✅ Retrieved {len(result_df)} online features from Feast")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Feast online features failed: {e}")
            raise
    
    async def _get_feast_offline_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime,
        feature_service_name: str = None
    ) -> pd.DataFrame:
        """Get offline features using Feast"""
        try:
            # Create entity dataframe
            entity_df = pd.DataFrame({
                "symbol": [entity_id.split("_")[0] for entity_id in entity_ids],
                "tf": [entity_id.split("_")[1] if "_" in entity_id else "1h" for entity_id in entity_ids]
            })
            
            # Get features from Feast
            if feature_service_name:
                features = self.feature_store.get_historical_features(
                    features=feature_service_name,
                    entity_df=entity_df,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                features = self.feature_store.get_historical_features(
                    features=feature_names,
                    entity_df=entity_df,
                    start_date=start_date,
                    end_date=end_date
                )
            
            # Convert to DataFrame
            result_df = features.to_df()
            logger.info(f"✅ Retrieved {len(result_df)} offline features from Feast")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Feast offline features failed: {e}")
            raise
    
    async def _get_timescaledb_online_features(
        self,
        entity_ids: List[str],
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Get online features using TimescaleDB fallback"""
        try:
            results = []
            timestamp = datetime.now()
            
            for entity_id in entity_ids:
                entity_features = await self.timescaledb_store.get_latest_features(
                    entity_id, feature_names
                )
                
                if entity_features:
                    row = {"entity_id": entity_id, "timestamp": timestamp}
                    for feature_name, feature_data in entity_features.items():
                        row[feature_name] = feature_data["value"]
                    results.append(row)
            
            result_df = pd.DataFrame(results)
            logger.info(f"✅ Retrieved {len(result_df)} online features from TimescaleDB")
            return result_df
            
        except Exception as e:
            logger.error(f"❌ TimescaleDB online features failed: {e}")
            raise
    
    async def _get_timescaledb_offline_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get offline features using TimescaleDB fallback"""
        try:
            results = []
            
            for entity_id in entity_ids:
                for feature_name in feature_names:
                    feature_history = await self.timescaledb_store.get_feature_history(
                        feature_name, entity_id, start_date, end_date
                    )
                    
                    if not feature_history.empty:
                        feature_history["entity_id"] = entity_id
                        feature_history["feature_name"] = feature_name
                        results.append(feature_history)
            
            if results:
                result_df = pd.concat(results, ignore_index=True)
                logger.info(f"✅ Retrieved {len(result_df)} offline features from TimescaleDB")
                return result_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ TimescaleDB offline features failed: {e}")
            raise
    
    async def compute_and_store_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        timestamp: datetime = None
    ) -> bool:
        """Compute and store features for given entities"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if not self.timescaledb_store:
                logger.error("❌ TimescaleDB store not available")
                return False
            
            timestamp = timestamp or datetime.now()
            success_count = 0
            
            for entity_id in entity_ids:
                for feature_name in feature_names:
                    try:
                        value = await self.timescaledb_store.compute_feature(
                            feature_name, entity_id, timestamp
                        )
                        if value is not None:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"❌ Failed to compute {feature_name} for {entity_id}: {e}")
            
            logger.info(f"✅ Successfully computed {success_count}/{len(entity_ids) * len(feature_names)} features")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to compute and store features: {e}")
            return False
    
    async def get_feature_service_info(self, service_name: str = None) -> Dict[str, Any]:
        """Get information about available feature services"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if self.feature_store and FEAST_AVAILABLE:
                # Get Feast service info
                if service_name:
                    service = self.feature_store.get_feature_service(service_name)
                    return {
                        "name": service.name,
                        "description": service.description,
                        "features": [f.name for f in service.features],
                        "tags": service.tags
                    }
                else:
                    # List all services
                    services = self.feature_store.list_feature_services()
                    return {
                        "available_services": [s.name for s in services],
                        "total_services": len(services)
                    }
            else:
                # Return TimescaleDB service info
                return {
                    "store_type": "timescaledb",
                    "available_features": [
                        "rsi_14", "macd", "ema_20", "bollinger_bands_position",
                        "atr", "volume_sma_ratio"
                    ],
                    "message": "Using TimescaleDB fallback - Feast not available"
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get feature service info: {e}")
            return {"error": str(e)}
    
    async def get_feature_statistics(
        self,
        feature_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get feature statistics for a given period"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if self.timescaledb_store:
                return await self.timescaledb_store.get_feature_statistics(
                    feature_name, start_date, end_date
                )
            else:
                return {"error": "No feature store available"}
                
        except Exception as e:
            logger.error(f"❌ Failed to get feature statistics: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the feature store manager"""
        try:
            if self.timescaledb_store:
                await self.timescaledb_store.close()
            
            if self.feature_store:
                # Feast doesn't have a close method, but we can clean up
                pass
            
            logger.info("🔒 Feast Feature Store Manager closed")
            
        except Exception as e:
            logger.error(f"❌ Error closing feature store manager: {e}")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Convenience functions for easy usage
async def get_online_features(
    entity_ids: List[str],
    feature_names: List[str],
    feature_service_name: str = None
) -> pd.DataFrame:
    """Get online features for real-time serving"""
    async with FeastFeatureStoreManager() as manager:
        return await manager.get_online_features(entity_ids, feature_names, feature_service_name)

async def get_offline_features(
    entity_ids: List[str],
    feature_names: List[str],
    start_date: datetime,
    end_date: datetime,
    feature_service_name: str = None
) -> pd.DataFrame:
    """Get offline features for training and batch processing"""
    async with FeastFeatureStoreManager() as manager:
        return await manager.get_offline_features(
            entity_ids, feature_names, start_date, end_date, feature_service_name
        )

async def compute_features(
    entity_ids: List[str],
    feature_names: List[str],
    timestamp: datetime = None
) -> bool:
    """Compute and store features for given entities"""
    async with FeastFeatureStoreManager() as manager:
        return await manager.compute_and_store_features(entity_ids, feature_names, timestamp)
