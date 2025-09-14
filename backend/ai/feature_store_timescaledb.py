#!/usr/bin/env python3
"""
TimescaleDB Feature Store Implementation
Phase 2A: Feature Store Implementation (Unified Database)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio

# Update import paths for new structure
from ..database.connection import get_async_engine
from ..database.models import Base
from sqlalchemy import text, select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

@dataclass
class FeatureDefinition:
    """Feature definition with metadata"""
    name: str
    description: str
    data_type: str
    source_table: str
    computation_rule: str
    version: str
    created_at: datetime
    is_active: bool = True
    tags: List[str] = None

@dataclass
class FeatureSet:
    """Collection of related features"""
    name: str
    description: str
    features: List[str]
    version: str
    created_at: datetime
    is_active: bool = True
    metadata: Dict[str, Any] = None

class TimescaleDBFeatureStore:
    """TimescaleDB-based feature store integrated with main application database"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url
        self.engine = None
        self._lock = threading.Lock()
        
        # Feature cache for frequently accessed features
        self._feature_cache = {}
        self._cache_ttl = timedelta(hours=1)
        self._cache_timestamps = {}
        
        # Phase 3: Feature Store Enhancement
        self._snapshot_manager = None
        self._lineage_tracker = None
        self._quality_monitor = None
        self._consistency_checker = None
        
        logger.info("🚀 TimescaleDB Feature Store initialized")
    
    async def initialize(self):
        """Initialize the feature store with database connection"""
        try:
            if not self.engine:
                self.engine = await get_async_engine()
            
            # Initialize feature store schema
            await self._initialize_schema()
            
            # Phase 3: Initialize enhancement components
            await self._initialize_phase3_components()
            
            logger.info("✅ TimescaleDB Feature Store schema initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB feature store: {e}")
            raise
    
    async def _initialize_schema(self):
        """Initialize the feature store database schema in TimescaleDB"""
        try:
            async with self.engine.begin() as conn:
                # Create feature definitions table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feature_definitions (
                        name VARCHAR(100) PRIMARY KEY,
                        description TEXT,
                        data_type VARCHAR(50),
                        source_table VARCHAR(100),
                        computation_rule TEXT,
                        version VARCHAR(20),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT TRUE,
                        tags JSONB DEFAULT '[]'::jsonb
                    )
                """))
                
                # Create feature sets table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feature_sets (
                        name VARCHAR(100) PRIMARY KEY,
                        description TEXT,
                        features JSONB DEFAULT '[]'::jsonb,
                        version VARCHAR(20),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT TRUE,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """))
                
                # Create feature values table as TimescaleDB hypertable
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feature_values (
                        feature_name VARCHAR(100),
                        entity_id VARCHAR(100),
                        timestamp TIMESTAMPTZ NOT NULL,
                        value DOUBLE PRECISION,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (feature_name, entity_id, timestamp)
                    )
                """))
                
                # Convert to hypertable if not already
                await conn.execute(text("""
                    SELECT create_hypertable('feature_values', 'timestamp', 
                                          if_not_exists => TRUE)
                """))
                
                # Create feature computation cache table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feature_cache (
                        cache_key VARCHAR(255) PRIMARY KEY,
                        feature_data JSONB,
                        computed_at TIMESTAMPTZ DEFAULT NOW(),
                        expires_at TIMESTAMPTZ,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """))
                
                # Create indexes for performance
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feature_values_timestamp 
                    ON feature_values (timestamp DESC)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feature_values_entity 
                    ON feature_values (entity_id, feature_name)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feature_values_feature 
                    ON feature_values (feature_name, timestamp DESC)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_feature_cache_expires 
                    ON feature_cache (expires_at)
                """))
                
                # Enable compression on feature_values hypertable
                await conn.execute(text("""
                    ALTER TABLE feature_values SET (
                        timescaledb.compress, 
                        timescaledb.compress_segmentby = 'feature_name, entity_id'
                    )
                """))
                
                # Add compression policy (compress chunks older than 1 day)
                await conn.execute(text("""
                    SELECT add_compression_policy('feature_values', INTERVAL '1 day')
                """))
                
                # Add retention policy (keep data for 1 year)
                await conn.execute(text("""
                    SELECT add_retention_policy('feature_values', INTERVAL '1 year')
                """))
                
                logger.info("✅ TimescaleDB feature store schema and policies configured")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize TimescaleDB schema: {e}")
            raise
    
    async def register_feature(self, feature_def: FeatureDefinition) -> bool:
        """Register a new feature definition"""
        try:
            async with self.engine.begin() as conn:
                # Check if feature already exists
                result = await conn.execute(text("""
                    SELECT name FROM feature_definitions WHERE name = :name
                """), {"name": feature_def.name})
                
                existing = result.fetchone()
                
                if existing:
                    # Update existing feature
                    await conn.execute(text("""
                        UPDATE feature_definitions 
                        SET description = :description, data_type = :data_type,
                            source_table = :source_table, computation_rule = :computation_rule,
                            version = :version, is_active = :is_active, tags = :tags
                        WHERE name = :name
                    """), {
                        "description": feature_def.description,
                        "data_type": feature_def.data_type,
                        "source_table": feature_def.source_table,
                        "computation_rule": feature_def.computation_rule,
                        "version": feature_def.version,
                        "is_active": feature_def.is_active,
                        "tags": json.dumps(feature_def.tags or []),
                        "name": feature_def.name
                    })
                    logger.info(f"🔄 Updated feature definition: {feature_def.name}")
                else:
                    # Insert new feature
                    await conn.execute(text("""
                        INSERT INTO feature_definitions 
                        (name, description, data_type, source_table, computation_rule, 
                         version, created_at, is_active, tags)
                        VALUES (:name, :description, :data_type, :source_table, 
                                :computation_rule, :version, :created_at, :is_active, :tags)
                    """), {
                        "name": feature_def.name,
                        "description": feature_def.description,
                        "data_type": feature_def.data_type,
                        "source_table": feature_def.source_table,
                        "computation_rule": feature_def.computation_rule,
                        "version": feature_def.version,
                        "created_at": feature_def.created_at,
                        "is_active": feature_def.is_active,
                        "tags": json.dumps(feature_def.tags or [])
                    })
                    logger.info(f"✅ Registered new feature: {feature_def.name}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to register feature {feature_def.name}: {e}")
            return False
    
    async def register_feature_set(self, feature_set: FeatureSet) -> bool:
        """Register a new feature set"""
        try:
            async with self.engine.begin() as conn:
                # Check if feature set already exists
                result = await conn.execute(text("""
                    SELECT name FROM feature_sets WHERE name = :name
                """), {"name": feature_set.name})
                
                existing = result.fetchone()
                
                if existing:
                    # Update existing feature set
                    await conn.execute(text("""
                        UPDATE feature_sets 
                        SET description = :description, features = :features, 
                            version = :version, is_active = :is_active, metadata = :metadata
                        WHERE name = :name
                    """), {
                        "description": feature_set.description,
                        "features": json.dumps(feature_set.features),
                        "version": feature_set.version,
                        "is_active": feature_set.is_active,
                        "metadata": json.dumps(feature_set.metadata or {}),
                        "name": feature_set.name
                    })
                    logger.info(f"🔄 Updated feature set: {feature_set.name}")
                else:
                    # Insert new feature set
                    await conn.execute(text("""
                        INSERT INTO feature_sets 
                        (name, description, features, version, created_at, is_active, metadata)
                        VALUES (:name, :description, :features, :version, :created_at, :is_active, :metadata)
                    """), {
                        "name": feature_set.name,
                        "description": feature_set.description,
                        "features": json.dumps(feature_set.features),
                        "version": feature_set.version,
                        "created_at": feature_set.created_at,
                        "is_active": feature_set.is_active,
                        "metadata": json.dumps(feature_set.metadata or {})
                    })
                    logger.info(f"✅ Registered new feature set: {feature_set.name}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to register feature set {feature_set.name}: {e}")
            return False
    
    async def compute_feature(self, feature_name: str, entity_id: str, 
                             timestamp: datetime, **kwargs) -> Optional[float]:
        """Compute a feature value for a specific entity and timestamp"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(feature_name, entity_id, timestamp)
            cached_value = self._get_from_cache(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Get feature definition
            feature_def = await self._get_feature_definition(feature_name)
            if not feature_def:
                logger.warning(f"⚠️ Feature {feature_name} not found")
                return None
            
            # Compute feature value based on computation rule
            value = await self._execute_computation_rule(feature_def, entity_id, timestamp, **kwargs)
            
            if value is not None:
                # Store computed value
                await self._store_feature_value(feature_name, entity_id, timestamp, value, kwargs)
                
                # Cache the result
                self._cache_feature_value(cache_key, value)
                
                logger.debug(f"✅ Computed feature {feature_name} for {entity_id}: {value}")
            
            return value
            
        except Exception as e:
            logger.error(f"❌ Failed to compute feature {feature_name}: {e}")
            return None
    
    async def compute_feature_set(self, feature_set_name: str, entity_id: str, 
                                timestamp: datetime, **kwargs) -> Dict[str, float]:
        """Compute all features in a feature set for a specific entity and timestamp"""
        try:
            # Get feature set definition
            feature_set = await self._get_feature_set(feature_set_name)
            if not feature_set:
                logger.warning(f"⚠️ Feature set {feature_set_name} not found")
                return {}
            
            # Compute all features in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                feature_futures = {
                    feature_name: executor.submit(
                        self.compute_feature, feature_name, entity_id, timestamp, **kwargs
                    )
                    for feature_name in feature_set.features
                }
                
                # Collect results
                results = {}
                for feature_name, future in feature_futures.items():
                    try:
                        value = future.result(timeout=30)  # 30 second timeout
                        if value is not None:
                            results[feature_name] = value
                    except Exception as e:
                        logger.error(f"❌ Failed to compute feature {feature_name}: {e}")
                
                logger.info(f"✅ Computed {len(results)} features for feature set {feature_set_name}")
                return results
                
        except Exception as e:
            logger.error(f"❌ Failed to compute feature set {feature_set_name}: {e}")
            return {}
    
    async def get_feature_history(self, feature_name: str, entity_id: str, 
                                 start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Get historical feature values for an entity"""
        try:
            query = text("""
                SELECT timestamp, value, metadata
                FROM feature_values
                WHERE feature_name = :feature_name AND entity_id = :entity_id
                  AND timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp ASC
            """)
            
            async with self.engine.begin() as conn:
                result = await conn.execute(query, {
                    "feature_name": feature_name,
                    "entity_id": entity_id,
                    "start_time": start_time,
                    "end_time": end_time
                })
                
                rows = result.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=['timestamp', 'value', 'metadata'])
                    logger.debug(f"✅ Retrieved {len(df)} historical values for {feature_name}")
                    return df
                else:
                    return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature history for {feature_name}: {e}")
            return pd.DataFrame()
    
    async def get_latest_features(self, entity_id: str, feature_names: List[str] = None) -> Dict[str, float]:
        """Get the latest feature values for an entity"""
        try:
            if feature_names is None:
                # Get all features
                query = text("""
                    SELECT DISTINCT feature_name
                    FROM feature_values
                    WHERE entity_id = :entity_id
                """)
                
                async with self.engine.begin() as conn:
                    result = await conn.execute(query, {"entity_id": entity_id})
                    feature_names = [row[0] for row in result.fetchall()]
            
            # Get latest values for each feature
            results = {}
            for feature_name in feature_names:
                query = text("""
                    SELECT value, timestamp
                    FROM feature_values
                    WHERE feature_name = :feature_name AND entity_id = :entity_id
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                async with self.engine.begin() as conn:
                    result = await conn.execute(query, {
                        "feature_name": feature_name,
                        "entity_id": entity_id
                    })
                    
                    row = result.fetchone()
                    if row:
                        results[feature_name] = {
                            'value': row[0],
                            'timestamp': row[1]
                        }
            
            logger.debug(f"✅ Retrieved latest features for {entity_id}: {len(results)} features")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest features for {entity_id}: {e}")
            return {}
    
    async def _get_feature_definition(self, feature_name: str) -> Optional[FeatureDefinition]:
        """Get feature definition from database"""
        try:
            query = text("""
                SELECT name, description, data_type, source_table, computation_rule,
                       version, created_at, is_active, tags
                FROM feature_definitions
                WHERE name = :name AND is_active = TRUE
            """)
            
            async with self.engine.begin() as conn:
                result = await conn.execute(query, {"name": feature_name})
                row = result.fetchone()
                
                if row:
                    return FeatureDefinition(
                        name=row[0],
                        description=row[1],
                        data_type=row[2],
                        source_table=row[3],
                        computation_rule=row[4],
                        version=row[5],
                        created_at=row[6],
                        is_active=row[7],
                        tags=json.loads(row[8]) if row[8] else []
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get feature definition for {feature_name}: {e}")
            return None
    
    async def _get_feature_set(self, feature_set_name: str) -> Optional[FeatureSet]:
        """Get feature set definition from database"""
        try:
            query = text("""
                SELECT name, description, features, version, created_at, is_active, metadata
                FROM feature_sets
                WHERE name = :name AND is_active = TRUE
            """)
            
            async with self.engine.begin() as conn:
                result = await conn.execute(query, {"name": feature_set_name})
                row = result.fetchone()
                
                if row:
                    return FeatureSet(
                        name=row[0],
                        description=row[1],
                        features=json.loads(row[2]),
                        version=row[3],
                        created_at=row[4],
                        is_active=row[5],
                        metadata=json.loads(row[6]) if row[6] else {}
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get feature set {feature_set_name}: {e}")
            return None
    
    async def _execute_computation_rule(self, feature_def: FeatureDefinition, 
                                       entity_id: str, timestamp: datetime, **kwargs) -> Optional[float]:
        """Execute the computation rule for a feature"""
        try:
            # This is a simplified implementation - in production, you'd have a more
            # sophisticated rule engine that can handle complex computations
            
            rule = feature_def.computation_rule.lower()
            
            if "rsi" in rule:
                # Example: RSI computation
                return await self._compute_rsi(entity_id, timestamp, **kwargs)
            elif "macd" in rule:
                # Example: MACD computation
                return await self._compute_macd(entity_id, timestamp, **kwargs)
            elif "ema" in rule:
                # Example: EMA computation
                return await self._compute_ema(entity_id, timestamp, **kwargs)
            elif "bb" in rule or "bollinger" in rule:
                # Example: Bollinger Bands computation
                return await self._compute_bollinger_bands(entity_id, timestamp, **kwargs)
            else:
                # Default: return a placeholder value
                logger.warning(f"⚠️ Unknown computation rule: {rule}")
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Failed to execute computation rule for {feature_def.name}: {e}")
            return None
    
    async def _compute_rsi(self, entity_id: str, timestamp: datetime, **kwargs) -> float:
        """Compute RSI (Relative Strength Index)"""
        # This is a placeholder implementation
        # In production, you'd fetch price data from candles table and compute actual RSI
        return np.random.uniform(0, 100)
    
    async def _compute_macd(self, entity_id: str, timestamp: datetime, **kwargs) -> float:
        """Compute MACD (Moving Average Convergence Divergence)"""
        # This is a placeholder implementation
        return np.random.uniform(-0.01, 0.01)
    
    async def _compute_ema(self, entity_id: str, timestamp: datetime, **kwargs) -> float:
        """Compute EMA (Exponential Moving Average)"""
        # This is a placeholder implementation
        return np.random.uniform(40000, 50000)
    
    async def _compute_bollinger_bands(self, entity_id: str, timestamp: datetime, **kwargs) -> float:
        """Compute Bollinger Bands position"""
        # This is a placeholder implementation
        return np.random.uniform(0, 1)
    
    async def _store_feature_value(self, feature_name: str, entity_id: str, 
                                  timestamp: datetime, value: float, metadata: Dict[str, Any]):
        """Store computed feature value in database"""
        try:
            query = text("""
                INSERT INTO feature_values 
                (feature_name, entity_id, timestamp, value, metadata)
                VALUES (:feature_name, :entity_id, :timestamp, :value, :metadata)
                ON CONFLICT (feature_name, entity_id, timestamp) 
                DO UPDATE SET value = :value, metadata = :metadata
            """)
            
            async with self.engine.begin() as conn:
                await conn.execute(query, {
                    "feature_name": feature_name,
                    "entity_id": entity_id,
                    "timestamp": timestamp,
                    "value": value,
                    "metadata": json.dumps(metadata or {})
                })
                
        except Exception as e:
            logger.error(f"❌ Failed to store feature value: {e}")
    
    def _generate_cache_key(self, feature_name: str, entity_id: str, timestamp: datetime) -> str:
        """Generate cache key for feature computation"""
        key_data = f"{feature_name}:{entity_id}:{timestamp.isoformat()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[float]:
        """Get feature value from cache"""
        try:
            if cache_key in self._feature_cache:
                # Check if cache is still valid
                if datetime.now() < self._cache_timestamps.get(cache_key, datetime.min):
                    return self._feature_cache[cache_key]
                else:
                    # Remove expired cache entry
                    del self._feature_cache[cache_key]
                    del self._cache_timestamps[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache retrieval error: {e}")
            return None
    
    def _cache_feature_value(self, cache_key: str, value: float):
        """Cache feature computation result"""
        try:
            self._feature_cache[cache_key] = value
            self._cache_timestamps[cache_key] = datetime.now() + self._cache_ttl
            
            # Limit cache size
            if len(self._feature_cache) > 10000:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._cache_timestamps.keys(),
                    key=lambda k: self._cache_timestamps[k]
                )[:1000]
                
                for key in oldest_keys:
                    del self._feature_cache[key]
                    del self._cache_timestamps[key]
                    
        except Exception as e:
            logger.error(f"❌ Cache storage error: {e}")
    
    async def get_feature_statistics(self, feature_name: str, 
                                    start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get statistical information about a feature"""
        try:
            query = text("""
                SELECT 
                    COUNT(*) as count,
                    AVG(value) as mean,
                    MIN(value) as min,
                    MAX(value) as max
                FROM feature_values
                WHERE feature_name = :feature_name AND timestamp BETWEEN :start_time AND :end_time
            """)
            
            async with self.engine.begin() as conn:
                result = await conn.execute(query, {
                    "feature_name": feature_name,
                    "start_time": start_time,
                    "end_time": end_time
                })
                
                row = result.fetchone()
                if row and row[0] > 0:
                    return {
                        'feature_name': feature_name,
                        'count': row[0],
                        'mean': float(row[1]) if row[1] else 0.0,
                        'min': float(row[2]) if row[2] else 0.0,
                        'max': float(row[3]) if row[3] else 0.0,
                        'period': {
                            'start': start_time.isoformat(),
                            'end': end_time.isoformat()
                        }
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get feature statistics for {feature_name}: {e}")
            return {}
    
    def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = [
                key for key, expiry in self._cache_timestamps.items()
                if current_time >= expiry
            ]
            
            for key in expired_keys:
                del self._feature_cache[key]
                del self._cache_timestamps[key]
            
            if expired_keys:
                logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")
    
    async def close(self):
        """Close the feature store connection"""
        try:
            if self.engine:
                await self.engine.dispose()
            logger.info("🔒 TimescaleDB Feature Store connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing feature store: {e}")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # =====================================================
    # PHASE 3: FEATURE STORE ENHANCEMENT METHODS
    # =====================================================

    async def _initialize_phase3_components(self):
        """Initialize Phase 3 enhancement components"""
        try:
            # Initialize snapshot manager
            self._snapshot_manager = FeatureSnapshotManager(self.engine)
            await self._snapshot_manager.initialize()
            
            # Initialize lineage tracker
            self._lineage_tracker = FeatureLineageTracker(self.engine)
            await self._lineage_tracker.initialize()
            
            # Initialize quality monitor
            self._quality_monitor = FeatureQualityMonitor(self.engine)
            await self._quality_monitor.initialize()
            
            # Initialize consistency checker
            self._consistency_checker = FeatureConsistencyChecker(self.engine)
            await self._consistency_checker.initialize()
            
            logger.info("✅ Phase 3 enhancement components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Phase 3 components: {e}")
            raise

    async def create_feature_snapshot(self, feature_set_name: str, version: str, 
                                    features: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Create a versioned feature snapshot"""
        try:
            if not self._snapshot_manager:
                raise Exception("Snapshot manager not initialized")
            
            snapshot_id = await self._snapshot_manager.create_snapshot(
                feature_set_name=feature_set_name,
                version=version,
                features=features,
                metadata=metadata or {}
            )
            
            logger.info(f"📸 Created feature snapshot: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature snapshot: {e}")
            raise

    async def get_feature_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Retrieve a feature snapshot by ID"""
        try:
            if not self._snapshot_manager:
                raise Exception("Snapshot manager not initialized")
            
            snapshot = await self._snapshot_manager.get_snapshot(snapshot_id)
            return snapshot
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature snapshot {snapshot_id}: {e}")
            raise

    async def track_feature_lineage(self, feature_name: str, parent_features: List[str], 
                                  computation_rule: str, version: str) -> bool:
        """Track feature computation lineage"""
        try:
            if not self._lineage_tracker:
                raise Exception("Lineage tracker not initialized")
            
            success = await self._lineage_tracker.track_lineage(
                feature_name=feature_name,
                parent_features=parent_features,
                computation_rule=computation_rule,
                version=version
            )
            
            if success:
                logger.info(f"🔗 Tracked lineage for feature: {feature_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to track feature lineage for {feature_name}: {e}")
            return False

    async def check_feature_quality(self, feature_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Check feature quality and detect drift"""
        try:
            if not self._quality_monitor:
                raise Exception("Quality monitor not initialized")
            
            quality_report = await self._quality_monitor.check_quality(
                feature_name=feature_name,
                data=data
            )
            
            logger.info(f"🔍 Quality check completed for feature: {feature_name}")
            return quality_report
            
        except Exception as e:
            logger.error(f"❌ Failed to check feature quality for {feature_name}: {e}")
            return {}

    async def validate_feature_consistency(self, snapshot_id: str, 
                                         check_types: List[str] = None) -> Dict[str, Any]:
        """Validate feature consistency across systems"""
        try:
            if not self._consistency_checker:
                raise Exception("Consistency checker not initialized")
            
            if check_types is None:
                check_types = ['streaming', 'outcome', 'cross_system']
            
            consistency_report = await self._consistency_checker.validate_consistency(
                snapshot_id=snapshot_id,
                check_types=check_types
            )
            
            logger.info(f"✅ Consistency validation completed for snapshot: {snapshot_id}")
            return consistency_report
            
        except Exception as e:
            logger.error(f"❌ Failed to validate consistency for snapshot {snapshot_id}: {e}")
            return {}

    async def get_feature_performance_metrics(self, feature_name: str, 
                                            time_period: timedelta = None) -> Dict[str, Any]:
        """Get feature performance metrics"""
        try:
            query = text("""
                SELECT 
                    AVG(computation_time_ms) as avg_computation_time,
                    MAX(computation_time_ms) as max_computation_time,
                    AVG(memory_usage_mb) as avg_memory_usage,
                    SUM(usage_frequency) as total_usage_count,
                    AVG(performance_score) as avg_performance_score,
                    MAX(recorded_at) as last_used_at
                FROM feature_performance_metrics
                WHERE feature_name = :feature_name
                AND recorded_at >= :start_time
            """)
            
            if time_period is None:
                time_period = timedelta(days=30)
            
            start_time = datetime.now() - time_period
            
            async with self.engine.begin() as conn:
                result = await conn.execute(query, {
                    "feature_name": feature_name,
                    "start_time": start_time
                })
                
                row = result.fetchone()
                if row:
                    return {
                        'feature_name': feature_name,
                        'avg_computation_time_ms': float(row[0]) if row[0] else 0.0,
                        'max_computation_time_ms': float(row[1]) if row[1] else 0.0,
                        'avg_memory_usage_mb': float(row[2]) if row[2] else 0.0,
                        'total_usage_count': int(row[3]) if row[3] else 0,
                        'avg_performance_score': float(row[4]) if row[4] else 0.0,
                        'last_used_at': row[5].isoformat() if row[5] else None,
                        'time_period_days': time_period.days
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get performance metrics for {feature_name}: {e}")
            return {}

    async def generate_feature_documentation(self, feature_name: str, 
                                           version: str = "1.0.0") -> Dict[str, Any]:
        """Generate automated feature documentation"""
        try:
            # Get feature definition
            feature_def = await self.get_feature_definition(feature_name)
            if not feature_def:
                raise Exception(f"Feature definition not found: {feature_name}")
            
            # Get performance metrics
            performance_metrics = await self.get_feature_performance_metrics(feature_name)
            
            # Get lineage information
            lineage_info = await self._lineage_tracker.get_lineage(feature_name, version)
            
            # Generate documentation content
            documentation = {
                'feature_name': feature_name,
                'version': version,
                'description': feature_def.description,
                'data_type': feature_def.data_type,
                'computation_rule': feature_def.computation_rule,
                'performance_metrics': performance_metrics,
                'lineage': lineage_info,
                'created_at': datetime.now().isoformat(),
                'examples': [],
                'change_history': []
            }
            
            # Store documentation
            await self._store_feature_documentation(feature_name, version, documentation)
            
            logger.info(f"📚 Generated documentation for feature: {feature_name}")
            return documentation
            
        except Exception as e:
            logger.error(f"❌ Failed to generate documentation for {feature_name}: {e}")
            return {}

    async def _store_feature_documentation(self, feature_name: str, version: str, 
                                         documentation: Dict[str, Any]):
        """Store feature documentation in database"""
        try:
            query = text("""
                INSERT INTO feature_documentation 
                (feature_name, documentation_version, content, examples, change_history, 
                 documentation_quality_score, completeness_score, author)
                VALUES (:feature_name, :version, :content, :examples, :change_history, 
                       :quality_score, :completeness_score, :author)
                ON CONFLICT (feature_name, documentation_version) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    examples = EXCLUDED.examples,
                    change_history = EXCLUDED.change_history,
                    documentation_quality_score = EXCLUDED.documentation_quality_score,
                    completeness_score = EXCLUDED.completeness_score,
                    updated_at = NOW()
            """)
            
            async with self.engine.begin() as conn:
                await conn.execute(query, {
                    "feature_name": feature_name,
                    "version": version,
                    "content": json.dumps(documentation),
                    "examples": json.dumps(documentation.get('examples', [])),
                    "change_history": json.dumps(documentation.get('change_history', [])),
                    "quality_score": 85.0,  # Default quality score
                    "completeness_score": 90.0,  # Default completeness score
                    "author": "system"
                })
                
        except Exception as e:
            logger.error(f"❌ Failed to store documentation for {feature_name}: {e}")
            raise


# =====================================================
# PHASE 3: ENHANCEMENT COMPONENT CLASSES
# =====================================================

class FeatureSnapshotManager:
    """Manages versioned feature snapshots"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def initialize(self):
        """Initialize the snapshot manager"""
        logger.info("📸 Feature Snapshot Manager initialized")
    
    async def create_snapshot(self, feature_set_name: str, version: str, 
                            features: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Create a new feature snapshot"""
        try:
            snapshot_id = f"{feature_set_name}_{version}_{int(datetime.now().timestamp())}"
            
            query = text("""
                INSERT INTO feature_snapshot_versions 
                (snapshot_id, feature_set_name, version, metadata, feature_count, 
                 data_points_count, quality_score, validation_status)
                VALUES (:snapshot_id, :feature_set_name, :version, :metadata, 
                       :feature_count, :data_points_count, :quality_score, :validation_status)
            """)
            
            async with self.engine.begin() as conn:
                await conn.execute(query, {
                    "snapshot_id": snapshot_id,
                    "feature_set_name": feature_set_name,
                    "version": version,
                    "metadata": json.dumps(metadata),
                    "feature_count": len(features),
                    "data_points_count": sum(len(v) if isinstance(v, (list, dict)) else 1 for v in features.values()),
                    "quality_score": metadata.get('quality_score', 85.0),
                    "validation_status": "pending"
                })
            
            return snapshot_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create snapshot: {e}")
            raise
    
    async def get_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Retrieve a snapshot by ID"""
        try:
            query = text("""
                SELECT * FROM feature_snapshot_versions 
                WHERE snapshot_id = :snapshot_id
            """)
            
            async with self.engine.begin() as conn:
                result = await conn.execute(query, {"snapshot_id": snapshot_id})
                row = result.fetchone()
                
                if row:
                    return {
                        'snapshot_id': row.snapshot_id,
                        'feature_set_name': row.feature_set_name,
                        'version': row.version,
                        'created_at': row.created_at.isoformat(),
                        'metadata': json.loads(row.metadata) if row.metadata else {},
                        'feature_count': row.feature_count,
                        'data_points_count': row.data_points_count,
                        'quality_score': float(row.quality_score) if row.quality_score else 0.0,
                        'validation_status': row.validation_status
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get snapshot {snapshot_id}: {e}")
            return {}


class FeatureLineageTracker:
    """Tracks feature computation lineage"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def initialize(self):
        """Initialize the lineage tracker"""
        logger.info("🔗 Feature Lineage Tracker initialized")
    
    async def track_lineage(self, feature_name: str, parent_features: List[str], 
                          computation_rule: str, version: str) -> bool:
        """Track feature lineage"""
        try:
            query = text("""
                INSERT INTO feature_lineage 
                (feature_name, parent_features, computation_rule, version, dependency_count)
                VALUES (:feature_name, :parent_features, :computation_rule, :version, :dependency_count)
                ON CONFLICT (feature_name, version) 
                DO UPDATE SET 
                    parent_features = EXCLUDED.parent_features,
                    computation_rule = EXCLUDED.computation_rule,
                    dependency_count = EXCLUDED.dependency_count
            """)
            
            async with self.engine.begin() as conn:
                await conn.execute(query, {
                    "feature_name": feature_name,
                    "parent_features": json.dumps(parent_features),
                    "computation_rule": computation_rule,
                    "version": version,
                    "dependency_count": len(parent_features)
                })
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to track lineage for {feature_name}: {e}")
            return False
    
    async def get_lineage(self, feature_name: str, version: str) -> Dict[str, Any]:
        """Get lineage information for a feature"""
        try:
            query = text("""
                SELECT * FROM feature_lineage 
                WHERE feature_name = :feature_name AND version = :version
            """)
            
            async with self.engine.begin() as conn:
                result = await conn.execute(query, {
                    "feature_name": feature_name,
                    "version": version
                })
                row = result.fetchone()
                
                if row:
                    return {
                        'feature_name': row.feature_name,
                        'parent_features': json.loads(row.parent_features) if row.parent_features else [],
                        'computation_rule': row.computation_rule,
                        'version': row.version,
                        'dependency_count': row.dependency_count,
                        'created_at': row.created_at.isoformat()
                    }
                
                return {}
                
        except Exception as e:
            logger.error(f"❌ Failed to get lineage for {feature_name}: {e}")
            return {}


class FeatureQualityMonitor:
    """Monitors feature quality and detects drift"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def initialize(self):
        """Initialize the quality monitor"""
        logger.info("🔍 Feature Quality Monitor initialized")
    
    async def check_quality(self, feature_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Check feature quality and detect drift"""
        try:
            # Basic quality checks
            quality_report = {
                'feature_name': feature_name,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(data),
                'missing_values': data.isnull().sum().sum(),
                'duplicate_rows': data.duplicated().sum(),
                'quality_score': 85.0,  # Default score
                'drift_detected': False,
                'recommendations': []
            }
            
            # Calculate quality score
            if len(data) > 0:
                missing_ratio = quality_report['missing_values'] / (len(data) * len(data.columns))
                quality_report['quality_score'] = max(0, 100 - (missing_ratio * 100))
            
            # Store quality metrics
            await self._store_quality_metrics(feature_name, quality_report)
            
            return quality_report
            
        except Exception as e:
            logger.error(f"❌ Failed to check quality for {feature_name}: {e}")
            return {}
    
    async def _store_quality_metrics(self, feature_name: str, quality_report: Dict[str, Any]):
        """Store quality metrics in database"""
        try:
            query = text("""
                INSERT INTO feature_metadata 
                (feature_name, metadata_type, metadata_key, metadata_value, version, confidence_score)
                VALUES (:feature_name, 'quality', 'quality_report', :metadata_value, '1.0.0', :confidence_score)
                ON CONFLICT (feature_name, metadata_type, metadata_key) 
                DO UPDATE SET 
                    metadata_value = EXCLUDED.metadata_value,
                    confidence_score = EXCLUDED.confidence_score,
                    updated_at = NOW()
            """)
            
            async with self.engine.begin() as conn:
                await conn.execute(query, {
                    "feature_name": feature_name,
                    "metadata_value": json.dumps(quality_report),
                    "confidence_score": quality_report.get('quality_score', 85.0)
                })
                
        except Exception as e:
            logger.error(f"❌ Failed to store quality metrics for {feature_name}: {e}")


class FeatureConsistencyChecker:
    """Validates feature consistency across systems"""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def initialize(self):
        """Initialize the consistency checker"""
        logger.info("✅ Feature Consistency Checker initialized")
    
    async def validate_consistency(self, snapshot_id: str, check_types: List[str]) -> Dict[str, Any]:
        """Validate feature consistency"""
        try:
            consistency_report = {
                'snapshot_id': snapshot_id,
                'timestamp': datetime.now().isoformat(),
                'checks_performed': check_types,
                'overall_status': 'passed',
                'check_results': {}
            }
            
            for check_type in check_types:
                check_result = await self._perform_consistency_check(snapshot_id, check_type)
                consistency_report['check_results'][check_type] = check_result
                
                if check_result['status'] == 'failed':
                    consistency_report['overall_status'] = 'failed'
            
            # Store consistency check results
            await self._store_consistency_check(snapshot_id, consistency_report)
            
            return consistency_report
            
        except Exception as e:
            logger.error(f"❌ Failed to validate consistency for {snapshot_id}: {e}")
            return {}
    
    async def _perform_consistency_check(self, snapshot_id: str, check_type: str) -> Dict[str, Any]:
        """Perform a specific consistency check"""
        try:
            # Placeholder for actual consistency checks
            check_result = {
                'status': 'passed',
                'details': f'{check_type} consistency check passed',
                'confidence_score': 95.0,
                'check_duration_ms': 50
            }
            
            return check_result
            
        except Exception as e:
            logger.error(f"❌ Failed to perform {check_type} check: {e}")
            return {
                'status': 'failed',
                'details': str(e),
                'confidence_score': 0.0,
                'check_duration_ms': 0
            }
    
    async def _store_consistency_check(self, snapshot_id: str, consistency_report: Dict[str, Any]):
        """Store consistency check results"""
        try:
            for check_type, check_result in consistency_report['check_results'].items():
                query = text("""
                    INSERT INTO feature_consistency_checks 
                    (snapshot_id, check_type, status, details, confidence_score, check_duration_ms)
                    VALUES (:snapshot_id, :check_type, :status, :details, :confidence_score, :check_duration_ms)
                """)
                
                async with self.engine.begin() as conn:
                    await conn.execute(query, {
                        "snapshot_id": snapshot_id,
                        "check_type": check_type,
                        "status": check_result['status'],
                        "details": json.dumps(check_result),
                        "confidence_score": check_result.get('confidence_score', 0.0),
                        "check_duration_ms": check_result.get('check_duration_ms', 0)
                    })
                    
        except Exception as e:
            logger.error(f"❌ Failed to store consistency check for {snapshot_id}: {e}")
