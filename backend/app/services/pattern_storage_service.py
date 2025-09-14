#!/usr/bin/env python3
"""
Pattern Storage Service for AlphaPulse Trading Bot
Provides optimized storage and retrieval of candlestick patterns using TimescaleDB
Enhanced with resilience features: retry logic, circuit breaker, and dead letter queue
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import uuid

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import uuid
import concurrent.futures
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil

from sqlalchemy import text, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from ..database.connection import TimescaleDBConnection
from app.core.config import settings
from app.core.resilience import (
    execute_with_retry, execute_with_circuit_breaker, 
    execute_with_timeout, execute_with_resilience,
    RetryConfig, get_resilience_manager
)
from app.core.query_performance import log_slow_queries
from app.core.latency_tracker import track_latency
from app.core.performance_alerting import performance_alerting

logger = logging.getLogger(__name__)

@dataclass
class PatternData:
    """Data structure for candlestick patterns"""
    pattern_id: Optional[int] = None
    symbol: str = ""
    timeframe: str = ""
    pattern_name: str = ""
    timestamp: datetime = None
    confidence: float = 0.0
    strength: str = ""
    price_level: float = 0.0
    volume_confirmation: bool = False
    volume_confidence: float = 0.0  # Volume confirmation confidence score (0.0-1.0)
    volume_pattern_type: str = ""    # Type of volume pattern (e.g., "spike", "divergence", "climax", "dry-up")
    volume_strength: str = ""        # Qualitative volume strength ("weak", "moderate", "strong")
    volume_context: Dict = None      # Multi-timeframe volume context storage
    trend_alignment: str = ""
    metadata: Dict = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.volume_context is None:
            self.volume_context = {}
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)

class PatternStorageService:
    """Service for storing and retrieving candlestick patterns with TimescaleDB optimization and resilience"""
    
    def __init__(self):
        self.db_connection = TimescaleDBConnection()
        self._initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Resilience configuration
        self.resilience_manager = get_resilience_manager()
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # Performance monitoring
        self.performance_stats = {
            'total_patterns_stored': 0,
            'total_batches_processed': 0,
            'copy_operations': 0,
            'regular_batches': 0,
            'total_insertion_time': 0.0,
            'fastest_batch_time': float('inf'),
            'slowest_batch_time': 0.0,
            'last_batch_performance': {
                'method': None,
                'batch_size': 0,
                'insertion_time': 0.0,
                'patterns_per_second': 0.0
            },
            'resilience_stats': {
                'retry_attempts': 0,
                'circuit_breaker_trips': 0,
                'dead_letter_queue_entries': 0,
                'timeout_occurrences': 0
            }
        }
        
    async def initialize(self):
        """Initialize the pattern storage service"""
        if self._initialized:
            return
            
        try:
            # Use resilience wrapper for initialization
            await execute_with_resilience(
                self._initialize_internal,
                "pattern_storage_initialization",
                retry_config=self.retry_config,
                timeout=60.0
            )
            
            self._initialized = True
            self.logger.info("✅ Pattern Storage Service initialized successfully with resilience")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Pattern Storage Service: {e}")
            raise
    
    async def _initialize_internal(self):
        """Internal initialization logic"""
        self.db_connection.initialize()
        await self._ensure_pattern_table_exists()
        await self._create_indexes()
        await self._setup_compression_policies()
    
    async def _ensure_pattern_table_exists(self):
        """Ensure the candlestick_patterns table exists with proper structure"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Check if table exists
                table_exists = await session.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'candlestick_patterns'
                    );
                """))
                
                if not table_exists.scalar():
                    # Create the table with TimescaleDB optimization
                    await session.execute(text("""
                        CREATE TABLE candlestick_patterns (
                            pattern_id SERIAL PRIMARY KEY,
                            symbol VARCHAR(20) NOT NULL,
                            timeframe VARCHAR(10) NOT NULL,
                            pattern_name VARCHAR(100) NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                            strength VARCHAR(20) NOT NULL,
                            price_level DECIMAL(18,8) NOT NULL,
                            volume_confirmation BOOLEAN NOT NULL,
                            volume_confidence DECIMAL(3,2) NOT NULL DEFAULT 0.0 CHECK (volume_confidence >= 0.0 AND volume_confidence <= 1.0),
                            volume_pattern_type VARCHAR(50),
                            volume_strength VARCHAR(20),
                            volume_context JSONB,
                            trend_alignment VARCHAR(20) NOT NULL,
                            metadata JSONB,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            updated_at TIMESTAMPTZ DEFAULT NOW()
                        );
                    """))
                    
                    # Convert to TimescaleDB hypertable
                    await session.execute(text("""
                        SELECT create_hypertable(
                            'candlestick_patterns', 
                            'timestamp',
                            chunk_time_interval => INTERVAL '1 day',
                            if_not_exists => TRUE
                        );
                    """))
                    
                    await session.commit()
                    self.logger.info("✅ Created candlestick_patterns hypertable")
                else:
                    # Check if new volume fields exist, add them if not
                    await self._migrate_table_schema(session)
                    self.logger.info("ℹ️ candlestick_patterns table already exists")
                    
        except Exception as e:
            self.logger.error(f"❌ Error ensuring pattern table exists: {e}")
            raise
    
    async def _migrate_table_schema(self, session: AsyncSession):
        """Migrate existing table to include new volume fields"""
        try:
            # Check if volume_confirmation column exists
            column_exists = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'candlestick_patterns' 
                    AND column_name = 'volume_confirmation'
                );
            """))
            
            if not column_exists.scalar():
                # Add volume_confirmation column
                await session.execute(text("""
                    ALTER TABLE candlestick_patterns 
                    ADD COLUMN volume_confirmation BOOLEAN DEFAULT FALSE;
                """))
                self.logger.info("✅ Added volume_confirmation column")
            
            # Check if volume_confidence column exists
            column_exists = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'candlestick_patterns' 
                    AND column_name = 'volume_confidence'
                );
            """))
            
            if not column_exists.scalar():
                # Add volume_confidence column
                await session.execute(text("""
                    ALTER TABLE candlestick_patterns 
                    ADD COLUMN volume_confidence FLOAT DEFAULT 0.0;
                """))
                self.logger.info("✅ Added volume_confidence column")
            
            # Check if volume_pattern_type column exists
            column_exists = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'candlestick_patterns' 
                    AND column_name = 'volume_pattern_type'
                );
            """))
            
            if not column_exists.scalar():
                # Add volume_pattern_type column
                await session.execute(text("""
                    ALTER TABLE candlestick_patterns 
                    ADD COLUMN volume_pattern_type VARCHAR(50);
                """))
                self.logger.info("✅ Added volume_pattern_type column")
            
            # Check if volume_strength column exists
            column_exists = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'candlestick_patterns' 
                    AND column_name = 'volume_strength'
                );
            """))
            
            if not column_exists.scalar():
                # Add volume_strength column
                await session.execute(text("""
                    ALTER TABLE candlestick_patterns 
                    ADD COLUMN volume_strength VARCHAR(20);
                """))
                self.logger.info("✅ Added volume_strength column")
            
            # Check if volume_context column exists
            column_exists = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'candlestick_patterns' 
                    AND column_name = 'volume_context'
                );
            """))
            
            if not column_exists.scalar():
                # Add volume_context column
                await session.execute(text("""
                    ALTER TABLE candlestick_patterns 
                    ADD COLUMN volume_context JSONB;
                """))
                self.logger.info("✅ Added volume_context column")
            
            await session.commit()
            
        except Exception as e:
            self.logger.error(f"❌ Error migrating table schema: {e}")
            await session.rollback()
            raise
    
    async def _create_indexes(self):
        """Create optimized indexes for pattern queries"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Create indexes for common query patterns
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_patterns_symbol_timeframe_timestamp ON candlestick_patterns (symbol, timeframe, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_pattern_name_timestamp ON candlestick_patterns (pattern_name, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_confidence_timestamp ON candlestick_patterns (confidence, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_trend_alignment_timestamp ON candlestick_patterns (trend_alignment, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_volume_confirmation_timestamp ON candlestick_patterns (volume_confirmation, timestamp DESC);",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_metadata_gin ON candlestick_patterns USING GIN (metadata);",
                    "CREATE INDEX IF NOT EXISTS idx_patterns_volume_context_gin ON candlestick_patterns USING GIN (volume_context);"
                ]
                
                for index_sql in indexes:
                    await session.execute(text(index_sql))
                
                await session.commit()
                self.logger.info("✅ Created optimized indexes for pattern queries")
                
        except Exception as e:
            self.logger.error(f"❌ Error creating indexes: {e}")
            raise
    
    async def _setup_compression_policies(self):
        """Setup TimescaleDB compression policies"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Enable compression
                await session.execute(text("""
                    ALTER TABLE candlestick_patterns SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'symbol,timeframe,pattern_name',
                        timescaledb.compress_orderby = 'timestamp DESC'
                    );
                """))
                
                # Add compression policy (compress chunks older than 7 days)
                await session.execute(text("""
                    SELECT add_compression_policy('candlestick_patterns', INTERVAL '7 days');
                """))
                
                await session.commit()
                self.logger.info("✅ Setup TimescaleDB compression policies")
                
        except Exception as e:
            self.logger.error(f"❌ Error setting up compression policies: {e}")
            # Don't fail initialization for compression setup
            pass
    
    async def store_pattern(self, pattern: PatternData) -> bool:
        """Store a single pattern with resilience"""
        try:
            # Use resilience wrapper for single pattern storage
            return await execute_with_resilience(
                self._store_pattern_internal,
                "store_single_pattern",
                pattern,
                retry_config=self.retry_config,
                timeout=30.0
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store pattern after resilience attempts: {e}")
            return False
    
    async def _store_pattern_internal(self, pattern: PatternData) -> bool:
        """Internal single pattern storage logic"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Validate pattern data
            if not self._validate_pattern(pattern):
                return False
            
            async with self.db_connection.get_async_session() as session:
                try:
                    # Prepare pattern data
                    pattern.updated_at = datetime.now(timezone.utc)
                    pattern_dict = asdict(pattern)
                    
                    # Convert metadata and volume_context to JSON strings
                    pattern_dict['metadata'] = json.dumps(pattern_dict.get('metadata', {}))
                    pattern_dict['volume_context'] = json.dumps(pattern_dict.get('volume_context', {}))
                    
                    # Execute insert with explicit transaction
                    query = text("""
                        INSERT INTO candlestick_patterns (
                            symbol, timeframe, pattern_name, timestamp, confidence,
                            strength, price_level, volume_confirmation, volume_confidence,
                            volume_pattern_type, volume_strength, volume_context, trend_alignment,
                            metadata, created_at, updated_at
                        ) VALUES (
                            :symbol, :timeframe, :pattern_name, :timestamp, :confidence,
                            :strength, :price_level, :volume_confirmation, :volume_confidence,
                            :volume_pattern_type, :volume_strength, :volume_context, :trend_alignment,
                            :metadata, :created_at, :updated_at
                        ) RETURNING pattern_id;
                    """)
                    
                    result = await session.execute(query, pattern_dict)
                    await session.commit()
                    
                    # Get the inserted pattern ID
                    pattern_id = result.scalar()
                    if pattern_id:
                        pattern.pattern_id = pattern_id
                        self.logger.info(f"✅ Stored pattern {pattern_id} for {pattern.symbol}")
                        
                        # Update performance stats
                        self.performance_stats['total_patterns_stored'] += 1
                        
                        return True
                    else:
                        self.logger.error("❌ Failed to get pattern ID after insertion")
                        await session.rollback()
                        return False
                        
                except Exception as e:
                    self.logger.error(f"❌ Error in transaction: {e}")
                    await session.rollback()
                    raise
                    
        except Exception as e:
            self.logger.error(f"❌ Error storing pattern: {e}")
            return False
    
    def _validate_pattern(self, pattern: PatternData) -> bool:
        """Validate pattern data before storage"""
        try:
            # Check required fields
            if not pattern.symbol or not pattern.timeframe or not pattern.pattern_name:
                self.logger.warning(f"⚠️ Pattern validation failed: missing required fields")
                return False
            
            # Check confidence range
            if pattern.confidence < 0.0 or pattern.confidence > 1.0:
                self.logger.warning(f"⚠️ Pattern validation failed: confidence {pattern.confidence} out of range [0.0, 1.0]")
                return False
            
            # Check volume confidence range
            if pattern.volume_confidence < 0.0 or pattern.volume_confidence > 1.0:
                self.logger.warning(f"⚠️ Pattern validation failed: volume_confidence {pattern.volume_confidence} out of range [0.0, 1.0]")
                return False
            
            # Validate metadata JSON
            if pattern.metadata:
                try:
                    json.dumps(pattern.metadata)
                except (TypeError, ValueError) as e:
                    self.logger.warning(f"⚠️ Pattern validation failed: invalid metadata format: {e}")
                    return False
            
            # Validate volume context JSON
            if pattern.volume_context:
                try:
                    json.dumps(pattern.volume_context)
                except (TypeError, ValueError) as e:
                    self.logger.warning(f"⚠️ Pattern validation failed: invalid volume_context format: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating pattern: {e}")
            return False
    
    @log_slow_queries(threshold_ms=500)  # Higher threshold for batch operations
    @track_latency("batch_insert")
    async def store_patterns_batch(self, patterns: List[PatternData]) -> Tuple[int, int]:
        """Store multiple patterns in a batch with resilience and transaction safety"""
        try:
            # Use resilience wrapper for batch storage
            return await execute_with_resilience(
                self._store_patterns_batch_internal,
                "store_patterns_batch",
                patterns,
                retry_config=self.retry_config,
                timeout=120.0  # Longer timeout for batch operations
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to store patterns batch after resilience attempts: {e}")
            return 0, len(patterns)
    
    async def _store_patterns_batch_internal(self, patterns: List[PatternData]) -> Tuple[int, int]:
        """Internal batch storage logic with transaction safety"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if not patterns:
                return 0, 0
            
            # Validate all patterns first
            valid_patterns = []
            invalid_count = 0
            
            for pattern in patterns:
                if self._validate_pattern(pattern):
                    valid_patterns.append(pattern)
                else:
                    invalid_count += 1
            
            if not valid_patterns:
                self.logger.warning("⚠️ No valid patterns to insert")
                return 0, len(patterns)
            
            # Choose optimal insertion method based on batch size
            if len(valid_patterns) >= 1000:
                # Use COPY for large batches
                result = await self._execute_copy_batch(valid_patterns)
            else:
                # Use regular batch insert for smaller batches
                result = await self._execute_regular_batch(valid_patterns)
            
            # Update performance stats
            end_time = asyncio.get_event_loop().time()
            insertion_time = end_time - start_time
            self._update_performance_stats('BATCH', len(valid_patterns), insertion_time)
            
            # Update resilience stats
            self.performance_stats['resilience_stats']['retry_attempts'] += 1
            
            return result[0], invalid_count + result[1]
            
        except Exception as e:
            self.logger.error(f"❌ Error in batch insert: {e}")
            return 0, len(patterns)
    
    async def _execute_regular_batch(self, patterns: List[PatternData]) -> Tuple[int, int]:
        """Execute regular batch insert with transaction safety"""
        try:
            async with self.db_connection.get_async_session() as session:
                try:
                    # Prepare batch data
                    values = []
                    for pattern in patterns:
                        pattern.updated_at = datetime.now(timezone.utc)
                        pattern_dict = asdict(pattern)
                        
                        # Convert metadata and volume_context to JSON strings
                        pattern_dict['metadata'] = json.dumps(pattern_dict.get('metadata', {}))
                        pattern_dict['volume_context'] = json.dumps(pattern_dict.get('volume_context', {}))
                        
                        values.append(pattern_dict)
                    
                    # Execute batch insert with explicit transaction
                    query = text("""
                        INSERT INTO candlestick_patterns (
                            symbol, timeframe, pattern_name, timestamp, confidence,
                            strength, price_level, volume_confirmation, volume_confidence,
                            volume_pattern_type, volume_strength, volume_context, trend_alignment,
                            metadata, created_at, updated_at
                        ) VALUES (
                            :symbol, :timeframe, :pattern_name, :timestamp, :confidence,
                            :strength, :price_level, :volume_confirmation, :volume_confidence,
                            :volume_pattern_type, :volume_strength, :volume_context, :trend_alignment,
                            :metadata, :created_at, :updated_at
                        );
                    """)
                    
                    # Execute all inserts in transaction
                    for value in values:
                        await session.execute(query, value)
                    
                    # Commit transaction
                    await session.commit()
                    
                    # Update pattern IDs by fetching the inserted patterns
                    await self._update_pattern_ids(patterns)
                    
                    self.logger.info(f"✅ Regular batch insert completed: {len(values)} patterns")
                    return len(values), 0
                    
                except Exception as e:
                    self.logger.error(f"❌ Error in batch transaction: {e}")
                    await session.rollback()
                    raise
                    
        except Exception as e:
            self.logger.error(f"❌ Error in regular batch insert: {e}")
            return 0, len(patterns)
    
    async def _execute_copy_batch(self, patterns: List[PatternData]) -> Tuple[int, int]:
        """Execute COPY-based batch insert with transaction safety"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Convert patterns to dictionary format
            values = []
            for pattern in patterns:
                pattern_dict = asdict(pattern)
                
                # Convert metadata and volume_context to JSON strings
                pattern_dict['metadata'] = json.dumps(pattern_dict.get('metadata', {}))
                pattern_dict['volume_context'] = json.dumps(pattern_dict.get('volume_context', {}))
                
                values.append(pattern_dict)
            
            # Try TimescaleDB parallel copy first
            if len(patterns) >= 5000:  # Use parallel copy for very large batches
                try:
                    return await self._execute_timescaledb_parallel_copy(patterns, values)
                except Exception as e:
                    self.logger.warning(f"⚠️ TimescaleDB parallel copy failed, falling back to regular COPY: {e}")
            
            # Execute regular COPY operation with transaction safety
            raw_conn = self.db_connection.get_raw_connection()
            
            try:
                with raw_conn.cursor() as cursor:
                    copy_sql = """
                        COPY candlestick_patterns (
                            symbol, timeframe, pattern_name, timestamp, confidence,
                            strength, price_level, volume_confirmation, volume_confidence,
                            volume_pattern_type, volume_strength, volume_context, trend_alignment,
                            metadata, created_at, updated_at
                        ) FROM STDIN
                    """
                    
                    cursor.copy_expert(copy_sql, self._prepare_copy_data(values))
                    
                    # Commit the transaction
                    raw_conn.commit()
                
                # Update pattern IDs (COPY doesn't return IDs, so we need to fetch them)
                await self._update_pattern_ids(patterns)
                
                # Track performance
                end_time = asyncio.get_event_loop().time()
                insertion_time = end_time - start_time
                self._update_performance_stats('COPY', len(patterns), insertion_time)
                
                self.logger.info(f"✅ COPY batch completed: {len(patterns)} patterns")
                return len(patterns), 0
                
            finally:
                raw_conn.close()
                
        except Exception as e:
            self.logger.error(f"❌ COPY batch failed: {e}")
            # Fall back to regular batch insert
            return await self._execute_regular_batch(patterns)

    def _prepare_copy_data(self, values: List[Dict]) -> str:
        """Prepare data in COPY format for PostgreSQL"""
        import io
        
        output = io.StringIO()
        
        for value in values:
            # Format each row according to COPY format
            row = [
                str(value.get('symbol', '')),
                str(value.get('timeframe', '')),
                str(value.get('pattern_name', '')),
                value.get('timestamp', '').isoformat() if value.get('timestamp') else '',
                str(value.get('confidence', 0.0)),
                str(value.get('strength', '')),
                str(value.get('price_level', 0.0)),
                't' if value.get('volume_confirmation') else 'f',
                str(value.get('volume_confidence', 0.0)),
                str(value.get('volume_pattern_type', '')),
                str(value.get('volume_strength', '')),
                value.get('volume_context', '{}'),
                str(value.get('trend_alignment', '')),
                value.get('metadata', '{}'),
                value.get('created_at', '').isoformat() if value.get('created_at') else '',
                value.get('updated_at', '').isoformat() if value.get('updated_at') else ''
            ]
            
            # Join with tab delimiter and add newline
            output.write('\t'.join(row) + '\n')
        
        output.seek(0)
        return output

    async def store_patterns_batch_optimized(self, patterns: List[PatternData], 
                                           batch_size: Optional[int] = None,
                                           use_parallel: bool = False,
                                           max_workers: Optional[int] = None) -> Tuple[int, int]:
        """Advanced optimized batch storage with parallel processing and intelligent batching"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            if not patterns:
                return 0, 0
            
            # Determine optimal batch size
            if batch_size is None:
                batch_size = self._calculate_optimal_batch_size(len(patterns))
            
            # Split patterns into optimal batches
            batches = [patterns[i:i + batch_size] for i in range(0, len(patterns), batch_size)]
            
            total_inserted = 0
            total_skipped = 0
            
            if use_parallel and len(batches) > 1:
                # Use parallel processing for multiple batches
                inserted, skipped = await self._execute_parallel_batches(batches, max_workers)
                total_inserted = inserted
                total_skipped = skipped
            else:
                # Sequential processing
                for batch in batches:
                    try:
                        # Validate batch
                        valid_patterns = self._validate_pattern_batch(batch)
                        
                        if not valid_patterns:
                            total_skipped += len(batch)
                            continue
                        
                        # Choose insert method based on batch size
                        if len(valid_patterns) >= 1000:  # Use COPY for large batches
                            inserted, skipped = await self._execute_copy_batch(valid_patterns)
                        else:  # Use regular batch insert for smaller batches
                            inserted, skipped = await self._execute_regular_batch(valid_patterns)
                        
                        total_inserted += inserted
                        total_skipped += skipped
                        
                    except Exception as e:
                        self.logger.error(f"❌ Batch processing failed: {e}")
                        total_skipped += len(batch)
            
            # Track performance
            end_time = asyncio.get_event_loop().time()
            insertion_time = end_time - start_time
            method = 'PARALLEL_OPTIMIZED' if use_parallel else 'OPTIMIZED'
            self._update_performance_stats(method, total_inserted, insertion_time)
            
            self.logger.info(f"✅ {method} batch storage completed: {total_inserted} inserted, {total_skipped} skipped")
            return total_inserted, total_skipped
            
        except Exception as e:
            self.logger.error(f"❌ Advanced batch storage failed: {e}")
            return 0, len(patterns)

    async def _execute_parallel_batches(self, batches: List[List[PatternData]], 
                                      max_workers: Optional[int] = None) -> Tuple[int, int]:
        """Execute multiple batches in parallel for maximum performance"""
        if not max_workers:
            max_workers = self._calculate_optimal_worker_count()
        
        self.logger.info(f"🚀 Starting parallel processing with {max_workers} workers for {len(batches)} batches")
        
        # Create tasks for parallel execution
        tasks = []
        for batch in batches:
            task = self._process_batch_async(batch)
            tasks.append(task)
        
        # Execute all batches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_inserted = 0
        total_skipped = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"❌ Batch {i} failed: {result}")
                total_skipped += len(batches[i])
            else:
                inserted, skipped = result
                total_inserted += inserted
                total_skipped += skipped
        
        return total_inserted, total_skipped

    async def _process_batch_async(self, batch: List[PatternData]) -> Tuple[int, int]:
        """Process a single batch asynchronously"""
        try:
            # Validate batch
            valid_patterns = self._validate_pattern_batch(batch)
            
            if not valid_patterns:
                return 0, len(batch)
            
            # Choose insert method based on batch size
            if len(valid_patterns) >= 1000:  # Use COPY for large batches
                return await self._execute_copy_batch(valid_patterns)
            else:  # Use regular batch insert for smaller batches
                return await self._execute_regular_batch(valid_patterns)
                
        except Exception as e:
            self.logger.error(f"❌ Async batch processing failed: {e}")
            return 0, len(batch)

    def _calculate_optimal_worker_count(self) -> int:
        """Calculate optimal number of parallel workers based on system resources"""
        try:
            # Get CPU count
            cpu_count = multiprocessing.cpu_count()
            
            # Get available memory
            available_memory = psutil.virtual_memory().available
            
            # Estimate memory per worker (conservative estimate: 100MB per worker)
            memory_per_worker = 100 * 1024 * 1024  # 100MB
            
            # Calculate memory-based worker count
            memory_based_workers = max(1, available_memory // memory_per_worker)
            
            # Use the smaller of CPU count and memory-based workers
            optimal_workers = min(cpu_count, memory_based_workers)
            
            # Cap at reasonable maximum
            optimal_workers = min(optimal_workers, 8)
            
            self.logger.info(f"📊 Calculated optimal worker count: {optimal_workers} (CPU: {cpu_count}, Memory-based: {memory_based_workers})")
            return optimal_workers
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not calculate optimal worker count: {e}")
            return 2  # Fallback to 2 workers

    def _calculate_optimal_batch_size(self, total_patterns: int) -> int:
        """Calculate optimal batch size based on data volume and available memory"""
        try:
            import psutil
            
            # Get available memory
            available_memory = psutil.virtual_memory().available
            
            # Estimate memory per pattern (rough estimate: 2KB per pattern)
            memory_per_pattern = 2 * 1024  # 2KB
            
            # Calculate batch size based on available memory (use 20% of available memory)
            memory_based_batch = int((available_memory * 0.2) / memory_per_pattern)
            
            # Set reasonable limits
            min_batch = 100
            max_batch = 10000
            
            # For very large datasets, use larger batches
            if total_patterns > 100000:
                optimal_batch = min(max_batch, max(min_batch, memory_based_batch))
            elif total_patterns > 10000:
                optimal_batch = min(5000, max(min_batch, memory_based_batch))
            else:
                optimal_batch = min(1000, max(min_batch, memory_based_batch))
            
            self.logger.info(f"📊 Calculated optimal batch size: {optimal_batch} (total: {total_patterns})")
            return optimal_batch
            
        except ImportError:
            # Fallback to conservative batch size
            return min(1000, max(100, total_patterns // 10))

    def _validate_pattern_batch(self, patterns: List[PatternData]) -> List[PatternData]:
        """Validate a batch of patterns and return valid ones"""
        valid_patterns = []
        
        for pattern in patterns:
            # Basic validation
            if not pattern.symbol or not pattern.timeframe or not pattern.pattern_name:
                self.logger.warning(f"⚠️ Skipping invalid pattern: missing required fields")
                continue
            
            if pattern.confidence < 0.0 or pattern.confidence > 1.0:
                self.logger.warning(f"⚠️ Skipping invalid pattern: confidence {pattern.confidence} out of range")
                continue
            
            # Update timestamp
            pattern.updated_at = datetime.now(timezone.utc)
            valid_patterns.append(pattern)
        
        return valid_patterns

    async def _execute_copy_batch(self, patterns: List[PatternData]) -> Tuple[int, int]:
        """Execute a batch using COPY operations with TimescaleDB optimization"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Convert patterns to dictionary format
            values = []
            for pattern in patterns:
                pattern_dict = asdict(pattern)
                
                # Convert metadata and volume_context to JSON strings
                if pattern_dict.get('metadata'):
                    try:
                        pattern_dict['metadata'] = json.dumps(pattern_dict['metadata'])
                    except (TypeError, ValueError):
                        pattern_dict['metadata'] = json.dumps({})
                else:
                    pattern_dict['metadata'] = json.dumps({})
                
                if pattern_dict.get('volume_context'):
                    try:
                        pattern_dict['volume_context'] = json.dumps(pattern_dict['volume_context'])
                    except (TypeError, ValueError):
                        pattern_dict['volume_context'] = json.dumps({})
                else:
                    pattern_dict['volume_context'] = json.dumps({})
                
                values.append(pattern_dict)
            
            # Try TimescaleDB parallel copy first
            if len(patterns) >= 5000:  # Use parallel copy for very large batches
                try:
                    return await self._execute_timescaledb_parallel_copy(patterns, values)
                except Exception as e:
                    self.logger.warning(f"⚠️ TimescaleDB parallel copy failed, falling back to regular COPY: {e}")
            
            # Execute regular COPY operation
            raw_conn = self.db_connection.get_raw_connection()
            
            with raw_conn.cursor() as cursor:
                copy_sql = """
                    COPY candlestick_patterns (
                        symbol, timeframe, pattern_name, timestamp, confidence,
                        strength, price_level, volume_confirmation, volume_confidence,
                        volume_pattern_type, volume_strength, volume_context, trend_alignment,
                        metadata, created_at, updated_at
                    ) FROM STDIN
                """
                
                cursor.copy_expert(copy_sql, self._prepare_copy_data(values))
                raw_conn.commit()
            
            raw_conn.close()
            
            # Update pattern IDs (COPY doesn't return IDs, so we need to fetch them)
            await self._update_pattern_ids(patterns)
            
            # Track performance
            end_time = asyncio.get_event_loop().time()
            insertion_time = end_time - start_time
            self._update_performance_stats('COPY', len(patterns), insertion_time)
            
            self.logger.info(f"✅ COPY batch completed: {len(patterns)} patterns")
            return len(patterns), 0
            
        except Exception as e:
            self.logger.error(f"❌ COPY batch failed: {e}")
            # Fall back to regular batch insert
            return await self._execute_regular_batch(patterns)

    async def _execute_timescaledb_parallel_copy(self, patterns: List[PatternData], 
                                                values: List[Dict]) -> Tuple[int, int]:
        """Execute TimescaleDB parallel copy for maximum performance on large datasets"""
        try:
            # Check if TimescaleDB parallel copy is available
            if not self._check_timescaledb_parallel_copy_available():
                raise Exception("TimescaleDB parallel copy not available")
            
            # Create temporary file for parallel copy
            temp_file_path = await self._create_temp_copy_file(values)
            
            try:
                # Execute parallel copy using psql command
                copy_result = await self._run_timescaledb_parallel_copy(temp_file_path)
                
                if copy_result:
                    # Update pattern IDs
                    await self._update_pattern_ids(patterns)
                    
                    # Track performance
                    self._update_performance_stats('TIMESCALEDB_PARALLEL_COPY', len(patterns), 0.0)
                    
                    self.logger.info(f"✅ TimescaleDB parallel copy completed: {len(patterns)} patterns")
                    return len(patterns), 0
                else:
                    raise Exception("Parallel copy command failed")
                    
            finally:
                # Clean up temporary file
                await self._cleanup_temp_copy_file(temp_file_path)
                
        except Exception as e:
            self.logger.error(f"❌ TimescaleDB parallel copy failed: {e}")
            raise

    def _check_timescaledb_parallel_copy_available(self) -> bool:
        """Check if TimescaleDB parallel copy tools are available"""
        try:
            import subprocess
            result = subprocess.run(['timescaledb-parallel-copy', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    async def _create_temp_copy_file(self, values: List[Dict]) -> str:
        """Create a temporary file for parallel copy operations"""
        import tempfile
        import os
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        
        try:
            # Write data in COPY format
            for value in values:
                row = [
                    str(value.get('symbol', '')),
                    str(value.get('timeframe', '')),
                    str(value.get('pattern_name', '')),
                    value.get('timestamp', '').isoformat() if value.get('timestamp') else '',
                    str(value.get('confidence', 0.0)),
                    str(value.get('strength', '')),
                    str(value.get('price_level', 0.0)),
                    't' if value.get('volume_confirmation') else 'f',
                    str(value.get('volume_confidence', 0.0)),
                    str(value.get('volume_pattern_type', '')),
                    str(value.get('volume_strength', '')),
                    value.get('volume_context', '{}'),
                    str(value.get('trend_alignment', '')),
                    value.get('metadata', '{}'),
                    value.get('created_at', '').isoformat() if value.get('created_at') else '',
                    value.get('updated_at', '').isoformat() if value.get('updated_at') else ''
                ]
                
                # Join with tab delimiter and add newline
                temp_file.write('\t'.join(row) + '\n')
            
            temp_file.close()
            return temp_file.name
            
        except Exception as e:
            temp_file.close()
            os.unlink(temp_file.name)
            raise e

    async def _run_timescaledb_parallel_copy(self, temp_file_path: str) -> bool:
        """Run TimescaleDB parallel copy command"""
        try:
            import subprocess
            import asyncio
            
            # Get database connection details
            db_url = self.db_connection.engine.url
            host = db_url.host or 'localhost'
            port = db_url.port or 5432
            database = db_url.database
            username = db_url.username
            password = db_url.password
            
            # Build parallel copy command
            cmd = [
                'timescaledb-parallel-copy',
                '--db-name', database,
                '--table', 'candlestick_patterns',
                '--file', temp_file_path,
                '--workers', str(self._calculate_optimal_worker_count()),
                '--copy-options', 'FORMAT csv, DELIMITER E\'\\t\'',
                '--columns', 'symbol,timeframe,pattern_name,timestamp,confidence,strength,price_level,volume_confirmation,volume_confidence,volume_pattern_type,volume_strength,volume_context,trend_alignment,metadata,created_at,updated_at'
            ]
            
            # Set environment variables for authentication
            env = os.environ.copy()
            env['PGHOST'] = host
            env['PGPORT'] = str(port)
            env['PGDATABASE'] = database
            env['PGUSER'] = username
            if password:
                env['PGPASSWORD'] = password
            
            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"✅ TimescaleDB parallel copy successful: {stdout.decode()}")
                return True
            else:
                self.logger.error(f"❌ TimescaleDB parallel copy failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error running TimescaleDB parallel copy: {e}")
            return False

    async def _cleanup_temp_copy_file(self, temp_file_path: str):
        """Clean up temporary copy file"""
        try:
            import os
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not cleanup temp file {temp_file_path}: {e}")

    async def _execute_regular_batch(self, patterns: List[PatternData]) -> Tuple[int, int]:
        """Execute a batch using regular batch insert"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Convert patterns to dictionary format
            values = []
            for pattern in patterns:
                pattern_dict = asdict(pattern)
                
                # Convert metadata and volume_context to JSON strings
                if pattern_dict.get('metadata'):
                    try:
                        pattern_dict['metadata'] = json.dumps(pattern_dict['metadata'])
                    except (TypeError, ValueError):
                        pattern_dict['metadata'] = json.dumps({})
                else:
                    pattern_dict['metadata'] = json.dumps({})
                
                if pattern_dict.get('volume_context'):
                    try:
                        pattern_dict['volume_context'] = json.dumps(pattern_dict['volume_context'])
                    except (TypeError, ValueError):
                        pattern_dict['volume_context'] = json.dumps({})
                else:
                    pattern_dict['volume_context'] = json.dumps({})
                
                values.append(pattern_dict)
            
            # Execute regular batch insert
            async with self.db_connection.get_async_session() as session:
                query = text("""
                    INSERT INTO candlestick_patterns (
                        symbol, timeframe, pattern_name, timestamp, confidence,
                        strength, price_level, volume_confirmation, volume_confidence,
                        volume_pattern_type, volume_strength, volume_context, trend_alignment,
                        metadata, created_at, updated_at
                    ) VALUES (
                        :symbol, :timeframe, :pattern_name, :timestamp, :confidence,
                        :strength, :price_level, :volume_confirmation, :volume_confidence,
                        :volume_pattern_type, :volume_strength, :volume_context, :trend_alignment,
                        :metadata, :created_at, :updated_at
                    );
                """)
                
                for value in values:
                    await session.execute(query, value)
                
                await session.commit()
            
            # Update pattern IDs
            await self._update_pattern_ids(patterns)
            
            # Track performance
            end_time = asyncio.get_event_loop().time()
            insertion_time = end_time - start_time
            self._update_performance_stats('REGULAR', len(patterns), insertion_time)
            
            self.logger.info(f"✅ Regular batch completed: {len(patterns)} patterns")
            return len(patterns), 0
            
        except Exception as e:
            self.logger.error(f"❌ Regular batch failed: {e}")
            return 0, len(patterns)

    async def _update_pattern_ids(self, patterns: List[PatternData]):
        """Update pattern IDs after batch insertion"""
        try:
            async with self.db_connection.get_async_session() as session:
                for pattern in patterns:
                    # Find the pattern in the database by its unique combination
                    fetch_query = text("""
                        SELECT pattern_id FROM candlestick_patterns 
                        WHERE symbol = :symbol 
                        AND timeframe = :timeframe 
                        AND pattern_name = :pattern_name 
                        AND timestamp = :timestamp
                        ORDER BY pattern_id DESC 
                        LIMIT 1
                    """)
                    
                    result = await session.execute(fetch_query, {
                        'symbol': pattern.symbol,
                        'timeframe': pattern.timeframe,
                        'pattern_name': pattern.pattern_name,
                        'timestamp': pattern.timestamp
                    })
                    
                    pattern_id = result.scalar()
                    if pattern_id:
                        pattern.pattern_id = pattern_id
                        
        except Exception as e:
            self.logger.warning(f"⚠️ Could not update pattern IDs: {e}")

    def _update_performance_stats(self, method: str, batch_size: int, insertion_time: float):
        """Update performance statistics"""
        patterns_per_second = batch_size / insertion_time if insertion_time > 0 else 0
        
        # Update overall stats
        self.performance_stats['total_patterns_stored'] += batch_size
        self.performance_stats['total_batches_processed'] += 1
        self.performance_stats['total_insertion_time'] += insertion_time
        
        if method == 'COPY':
            self.performance_stats['copy_operations'] += 1
        else:
            self.performance_stats['regular_batches'] += 1
        
        # Update timing stats
        if insertion_time < self.performance_stats['fastest_batch_time']:
            self.performance_stats['fastest_batch_time'] = insertion_time
        
        if insertion_time > self.performance_stats['slowest_batch_time']:
            self.performance_stats['slowest_batch_time'] = insertion_time
        
        # Update last batch performance
        self.performance_stats['last_batch_performance'] = {
            'method': method,
            'batch_size': batch_size,
            'insertion_time': insertion_time,
            'patterns_per_second': patterns_per_second
        }
        
        self.logger.info(f"📊 Performance: {method} batch of {batch_size} patterns in {insertion_time:.3f}s ({patterns_per_second:.1f} patterns/s)")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if self.performance_stats['total_batches_processed'] == 0:
            return self.performance_stats
        
        # Calculate averages
        avg_batch_time = self.performance_stats['total_insertion_time'] / self.performance_stats['total_batches_processed']
        avg_patterns_per_second = self.performance_stats['total_patterns_stored'] / self.performance_stats['total_insertion_time'] if self.performance_stats['total_insertion_time'] > 0 else 0
        
        return {
            **self.performance_stats,
            'avg_batch_time': avg_batch_time,
            'avg_patterns_per_second': avg_patterns_per_second,
            'copy_efficiency': self.performance_stats['copy_operations'] / self.performance_stats['total_batches_processed'] if self.performance_stats['total_batches_processed'] > 0 else 0
        }

    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_patterns_stored': 0,
            'total_batches_processed': 0,
            'copy_operations': 0,
            'regular_batches': 0,
            'total_insertion_time': 0.0,
            'fastest_batch_time': float('inf'),
            'slowest_batch_time': 0.0,
            'last_batch_performance': {
                'method': None,
                'batch_size': 0,
                'insertion_time': 0.0,
                'patterns_per_second': 0.0
            },
            'resilience_stats': {
                'retry_attempts': 0,
                'circuit_breaker_trips': 0,
                'dead_letter_queue_entries': 0,
                'timeout_occurrences': 0
            }
        }
        self.logger.info("🔄 Performance statistics reset")

    async def benchmark_insertion_methods(self, test_patterns: List[PatternData], 
                                        iterations: int = 3,
                                        include_parallel: bool = True,
                                        include_storage_analysis: bool = True) -> Dict[str, Any]:
        """Advanced benchmark with parallel processing and storage format analysis"""
        if not test_patterns:
            return {"error": "No test patterns provided"}
        
        results = {
            'test_config': {
                'total_patterns': len(test_patterns),
                'iterations': iterations,
                'include_parallel': include_parallel,
                'include_storage_analysis': include_storage_analysis
            },
            'regular_batch': {'times': [], 'avg_time': 0.0, 'avg_patterns_per_second': 0.0},
            'copy_batch': {'times': [], 'avg_time': 0.0, 'avg_patterns_per_second': 0.0},
            'optimized_batch': {'times': [], 'avg_time': 0.0, 'avg_patterns_per_second': 0.0},
            'parallel_optimized': {'times': [], 'avg_time': 0.0, 'avg_patterns_per_second': 0.0},
            'timescaledb_parallel_copy': {'times': [], 'avg_time': 0.0, 'avg_patterns_per_second': 0.0}
        }
        
        self.logger.info(f"🚀 Starting advanced benchmark with {len(test_patterns)} patterns, {iterations} iterations")
        
        # Test regular batch insert
        for i in range(iterations):
            try:
                start_time = asyncio.get_event_loop().time()
                inserted, skipped = await self._execute_regular_batch(test_patterns.copy())
                end_time = asyncio.get_event_loop().time()
                
                insertion_time = end_time - start_time
                results['regular_batch']['times'].append(insertion_time)
                
                self.logger.info(f"📊 Regular batch iteration {i+1}: {insertion_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Regular batch benchmark failed: {e}")
        
        # Test COPY batch insert
        for i in range(iterations):
            try:
                start_time = asyncio.get_event_loop().time()
                inserted, skipped = await self._execute_copy_batch(test_patterns.copy())
                end_time = asyncio.get_event_loop().time()
                
                insertion_time = end_time - start_time
                results['copy_batch']['times'].append(insertion_time)
                
                self.logger.info(f"📊 COPY batch iteration {i+1}: {insertion_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"❌ COPY batch benchmark failed: {e}")
        
        # Test optimized batch insert
        for i in range(iterations):
            try:
                start_time = asyncio.get_event_loop().time()
                inserted, skipped = await self.store_patterns_batch_optimized(test_patterns.copy())
                end_time = asyncio.get_event_loop().time()
                
                insertion_time = end_time - start_time
                results['optimized_batch']['times'].append(insertion_time)
                
                self.logger.info(f"📊 Optimized batch iteration {i+1}: {insertion_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Optimized batch benchmark failed: {e}")
        
        # Test parallel optimized batch insert
        if include_parallel and len(test_patterns) > 1000:
            for i in range(iterations):
                try:
                    start_time = asyncio.get_event_loop().time()
                    inserted, skipped = await self.store_patterns_batch_optimized(
                        test_patterns.copy(), use_parallel=True
                    )
                    end_time = asyncio.get_event_loop().time()
                    
                    insertion_time = end_time - start_time
                    results['parallel_optimized']['times'].append(insertion_time)
                    
                    self.logger.info(f"📊 Parallel optimized batch iteration {i+1}: {insertion_time:.3f}s")
                    
                except Exception as e:
                    self.logger.error(f"❌ Parallel optimized batch benchmark failed: {e}")
        
        # Test TimescaleDB parallel copy (if available)
        if include_parallel and len(test_patterns) >= 5000:
            for i in range(iterations):
                try:
                    start_time = asyncio.get_event_loop().time()
                    inserted, skipped = await self._execute_timescaledb_parallel_copy(
                        test_patterns.copy(), 
                        [asdict(p) for p in test_patterns]
                    )
                    end_time = asyncio.get_event_loop().time()
                    
                    insertion_time = end_time - start_time
                    results['timescaledb_parallel_copy']['times'].append(insertion_time)
                    
                    self.logger.info(f"📊 TimescaleDB parallel copy iteration {i+1}: {insertion_time:.3f}s")
                    
                except Exception as e:
                    self.logger.error(f"❌ TimescaleDB parallel copy benchmark failed: {e}")
        
        # Calculate averages for all methods
        for method in results.keys():
            if method != 'test_config' and results[method]['times']:
                avg_time = sum(results[method]['times']) / len(results[method]['times'])
                results[method]['avg_time'] = avg_time
                results[method]['avg_patterns_per_second'] = len(test_patterns) / avg_time if avg_time > 0 else 0
        
        # Find the fastest method
        valid_methods = [k for k, v in results.items() 
                        if k != 'test_config' and isinstance(v, dict) and 'avg_time' in v and v['avg_time'] > 0]
        
        if valid_methods:
            fastest_method = min(valid_methods, key=lambda k: results[k]['avg_time'])
            results['fastest_method'] = fastest_method
            results['performance_improvement'] = {}
            
            for method in valid_methods:
                if method != fastest_method:
                    improvement = ((results[method]['avg_time'] - results[fastest_method]['avg_time']) / results[method]['avg_time']) * 100
                    results['performance_improvement'][method] = f"{improvement:.1f}% faster"
        
        # Add storage format analysis if requested
        if include_storage_analysis:
            results['storage_analysis'] = await self._analyze_storage_format_performance()
        
        self.logger.info(f"🏆 Advanced benchmark completed. Fastest method: {results.get('fastest_method', 'N/A')}")
        return results

    async def _analyze_storage_format_performance(self) -> Dict[str, Any]:
        """Analyze storage format performance (JSONB vs flattened columns)"""
        try:
            self.logger.info("🔍 Analyzing storage format performance...")
            
            analysis = {
                'jsonb_performance': {},
                'flattened_performance': {},
                'recommendations': []
            }
            
            # Analyze JSONB column performance
            jsonb_stats = await self._get_jsonb_column_stats()
            analysis['jsonb_performance'] = jsonb_stats
            
            # Analyze potential flattened column performance
            flattened_analysis = await self._analyze_flattened_columns()
            analysis['flattened_performance'] = flattened_analysis
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_storage_recommendations(
                jsonb_stats, flattened_analysis
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Storage format analysis failed: {e}")
            return {"error": str(e)}

    async def _get_jsonb_column_stats(self) -> Dict[str, Any]:
        """Get performance statistics for JSONB columns"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Get table size and JSONB column sizes
                size_query = text("""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('candlestick_patterns')) as total_size,
                        pg_size_pretty(pg_column_size(metadata)) as metadata_size,
                        pg_size_pretty(pg_column_size(volume_context)) as volume_context_size,
                        COUNT(*) as total_rows,
                        AVG(pg_column_size(metadata)) as avg_metadata_size,
                        AVG(pg_column_size(volume_context)) as avg_volume_context_size
                    FROM candlestick_patterns
                    LIMIT 1000;
                """)
                
                result = await session.execute(size_query)
                stats = result.fetchone()
                
                if stats:
                    return {
                        'total_table_size': stats.total_size,
                        'metadata_column_size': stats.metadata_size,
                        'volume_context_column_size': stats.volume_context_size,
                        'total_rows_analyzed': stats.total_rows,
                        'avg_metadata_size_bytes': float(stats.avg_metadata_size) if stats.avg_metadata_size else 0,
                        'avg_volume_context_size_bytes': float(stats.avg_volume_context_size) if stats.avg_volume_context_size else 0
                    }
                
                return {}
                
        except Exception as e:
            self.logger.error(f"❌ Error getting JSONB stats: {e}")
            return {}

    async def _analyze_flattened_columns(self) -> Dict[str, Any]:
        """Analyze potential performance of flattened columns"""
        try:
            async with self.db_connection.get_async_session() as session:
                # Analyze metadata structure to suggest flattened columns
                metadata_query = text("""
                    SELECT 
                        metadata->>'rsi' as rsi,
                        metadata->>'macd' as macd,
                        metadata->>'bollinger_position' as bollinger_position,
                        volume_context->>'volume_trend' as volume_trend,
                        volume_context->>'volume_ratio' as volume_ratio
                    FROM candlestick_patterns
                    WHERE metadata IS NOT NULL OR volume_context IS NOT NULL
                    LIMIT 1000;
                """)
                
                result = await session.execute(metadata_query)
                rows = result.fetchall()
                
                if not rows:
                    return {}
                
                # Analyze data distribution
                analysis = {
                    'suggested_flattened_columns': [],
                    'data_distribution': {},
                    'estimated_storage_savings': 0
                }
                
                # Check which fields are commonly used
                field_usage = {
                    'rsi': 0, 'macd': 0, 'bollinger_position': 0,
                    'volume_trend': 0, 'volume_ratio': 0
                }
                
                for row in rows:
                    for field in field_usage.keys():
                        if getattr(row, field) is not None:
                            field_usage[field] += 1
                
                # Suggest columns for frequently used fields
                total_rows = len(rows)
                for field, usage_count in field_usage.items():
                    usage_percentage = (usage_count / total_rows) * 100
                    if usage_percentage > 50:  # If field is used in more than 50% of rows
                        analysis['suggested_flattened_columns'].append({
                            'field': field,
                            'usage_percentage': usage_percentage,
                            'estimated_storage': '4-8 bytes' if field in ['rsi', 'macd', 'bollinger_position'] else '20-50 bytes'
                        })
                
                # Estimate storage savings
                current_jsonb_size = 100  # Estimated average JSONB size per row
                estimated_flattened_size = 50  # Estimated flattened column size
                analysis['estimated_storage_savings'] = ((current_jsonb_size - estimated_flattened_size) / current_jsonb_size) * 100
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"❌ Error analyzing flattened columns: {e}")
            return {}

    def _generate_storage_recommendations(self, jsonb_stats: Dict, flattened_analysis: Dict) -> List[str]:
        """Generate storage optimization recommendations"""
        recommendations = []
        
        try:
            # Analyze JSONB performance
            if jsonb_stats.get('avg_metadata_size_bytes', 0) > 200:
                recommendations.append("Consider flattening frequently accessed metadata fields (metadata > 200 bytes average)")
            
            if jsonb_stats.get('avg_volume_context_size_bytes', 0) > 150:
                recommendations.append("Consider flattening volume context fields for better query performance")
            
            # Analyze flattened column suggestions
            suggested_columns = flattened_analysis.get('suggested_flattened_columns', [])
            if suggested_columns:
                recommendations.append(f"High-usage fields detected: {', '.join([col['field'] for col in suggested_columns])}")
                recommendations.append("Consider adding dedicated columns for these frequently queried fields")
            
            # Storage savings recommendation
            savings = flattened_analysis.get('estimated_storage_savings', 0)
            if savings > 20:
                recommendations.append(f"Potential storage savings: {savings:.1f}% by flattening common fields")
            
            # Performance recommendations
            if not recommendations:
                recommendations.append("Current JSONB structure appears optimal for your usage patterns")
            else:
                recommendations.append("Use GIN indexes on JSONB columns for complex queries")
                recommendations.append("Consider partial indexes on JSONB fields for specific query patterns")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Error generating recommendations: {e}")
            return ["Error generating storage recommendations"]

    @log_slow_queries(threshold_ms=300)  # Threshold for retrieval operations
    @track_latency("retrieval")
    async def get_patterns(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        pattern_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        trend_alignment: Optional[str] = None,
        strength: Optional[str] = None,
        volume_confirmation: Optional[bool] = None,
        volume_pattern_type: Optional[str] = None,
        volume_strength: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "timestamp DESC"
    ) -> List[PatternData]:
        """Retrieve patterns with flexible filtering and optimization"""
        try:
            if not self._initialized:
                await self.initialize()
            
            async with self.db_connection.get_async_session() as session:
                # Build dynamic query with conditions
                conditions = []
                params = {}
                
                if symbol:
                    conditions.append("symbol = :symbol")
                    params['symbol'] = symbol
                
                if timeframe:
                    conditions.append("timeframe = :timeframe")
                    params['timeframe'] = timeframe
                
                if pattern_name:
                    conditions.append("pattern_name = :pattern_name")
                    params['pattern_name'] = pattern_name
                
                if start_time:
                    conditions.append("timestamp >= :start_time")
                    params['start_time'] = start_time
                
                if end_time:
                    conditions.append("timestamp <= :end_time")
                    params['end_time'] = end_time
                
                if min_confidence is not None:
                    conditions.append("confidence >= :min_confidence")
                    params['min_confidence'] = min_confidence
                
                if max_confidence is not None:
                    conditions.append("confidence <= :max_confidence")
                    params['max_confidence'] = max_confidence
                
                if trend_alignment:
                    conditions.append("trend_alignment = :trend_alignment")
                    params['trend_alignment'] = trend_alignment
                
                if strength:
                    conditions.append("strength = :strength")
                    params['strength'] = strength
                
                if volume_confirmation is not None:
                    conditions.append("volume_confirmation = :volume_confirmation")
                    params['volume_confirmation'] = volume_confirmation
                
                if volume_pattern_type:
                    conditions.append("volume_pattern_type = :volume_pattern_type")
                    params['volume_pattern_type'] = volume_pattern_type
                
                if volume_strength:
                    conditions.append("volume_strength = :volume_strength")
                    params['volume_strength'] = volume_strength
                
                # Build WHERE clause
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # Build ORDER BY clause
                order_clause = self._build_order_clause(order_by)
                
                # Execute query
                query = text(f"""
                    SELECT 
                        pattern_id, symbol, timeframe, pattern_name, timestamp, confidence,
                        strength, price_level, volume_confirmation, volume_confidence,
                        volume_pattern_type, volume_strength, volume_context, trend_alignment,
                        metadata, created_at, updated_at
                    FROM candlestick_patterns
                    WHERE {where_clause}
                    ORDER BY {order_clause}
                    LIMIT :limit OFFSET :offset;
                """)
                
                # Add limit and offset to params
                params['limit'] = limit
                params['offset'] = offset
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                # Convert rows to PatternData objects
                patterns = []
                for row in rows:
                    # Parse JSON fields
                    metadata = {}
                    volume_context = {}
                    
                    if row.metadata:
                        try:
                            metadata = json.loads(row.metadata) if isinstance(row.metadata, str) else row.metadata
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    
                    if row.volume_context:
                        try:
                            volume_context = json.loads(row.volume_context) if isinstance(row.volume_context, str) else row.volume_context
                        except (json.JSONDecodeError, TypeError):
                            volume_context = {}
                    
                    pattern = PatternData(
                        pattern_id=row.pattern_id,
                        symbol=row.symbol,
                        timeframe=row.timeframe,
                        pattern_name=row.pattern_name,
                        timestamp=row.timestamp,
                        confidence=float(row.confidence),
                        strength=row.strength,
                        price_level=float(row.price_level),
                        volume_confirmation=row.volume_confirmation,
                        volume_confidence=float(row.volume_confidence),
                        volume_pattern_type=row.volume_pattern_type,
                        volume_strength=row.volume_strength,
                        volume_context=volume_context,
                        trend_alignment=row.trend_alignment,
                        metadata=metadata,
                        created_at=row.created_at,
                        updated_at=row.updated_at
                    )
                    patterns.append(pattern)
                
                self.logger.info(f"✅ Retrieved {len(patterns)} patterns")
                return patterns
                
        except Exception as e:
            self.logger.error(f"❌ Error retrieving patterns: {e}")
            return []
    
    def _build_order_clause(self, order_by: str) -> str:
        """Build safe ORDER BY clause"""
        allowed_columns = {
            'timestamp', 'confidence', 'price_level', 'created_at', 'updated_at'
        }
        allowed_directions = {'ASC', 'DESC'}
        
        parts = order_by.strip().split()
        if len(parts) != 2:
            return "timestamp DESC"
        
        column, direction = parts[0].lower(), parts[1].upper()
        
        if column not in allowed_columns or direction not in allowed_directions:
            return "timestamp DESC"
        
        return f"{column} {direction}"
    
    async def get_pattern_statistics(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get statistical information about stored patterns"""
        try:
            if not self._initialized:
                await self.initialize()
            
            async with self.db_connection.get_async_session() as session:
                # Build conditions
                conditions = []
                params = {}
                
                if symbol:
                    conditions.append("symbol = :symbol")
                    params['symbol'] = symbol
                
                if timeframe:
                    conditions.append("timeframe = :timeframe")
                    params['timeframe'] = timeframe
                
                if start_time:
                    conditions.append("timestamp >= :start_time")
                    params['start_time'] = start_time
                
                if end_time:
                    conditions.append("timestamp <= :end_time")
                    params['end_time'] = end_time
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                # Get various statistics
                stats_query = text(f"""
                    SELECT 
                        COUNT(*) as total_patterns,
                        COUNT(DISTINCT symbol) as unique_symbols,
                        COUNT(DISTINCT timeframe) as unique_timeframes,
                        COUNT(DISTINCT pattern_name) as unique_patterns,
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence,
                        COUNT(CASE WHEN volume_confirmation = true THEN 1 END) as volume_confirmed_patterns,
                        COUNT(CASE WHEN trend_alignment = 'bullish' THEN 1 END) as bullish_patterns,
                        COUNT(CASE WHEN trend_alignment = 'bearish' THEN 1 END) as bearish_patterns,
                        COUNT(CASE WHEN trend_alignment = 'neutral' THEN 1 END) as neutral_patterns
                    FROM candlestick_patterns
                    WHERE {where_clause};
                """)
                
                result = await session.execute(stats_query, params)
                stats = result.fetchone()
                
                # Get pattern distribution
                pattern_dist_query = text(f"""
                    SELECT 
                        pattern_name,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM candlestick_patterns
                    WHERE {where_clause}
                    GROUP BY pattern_name
                    ORDER BY count DESC
                    LIMIT 10;
                """)
                
                pattern_result = await session.execute(pattern_dist_query, params)
                pattern_distribution = [
                    {"pattern_name": row.pattern_name, "count": row.count, "avg_confidence": float(row.avg_confidence)}
                    for row in pattern_result.fetchall()
                ]
                
                # Get time-based statistics
                time_stats_query = text(f"""
                    SELECT 
                        DATE_TRUNC('hour', timestamp) as hour_bucket,
                        COUNT(*) as pattern_count
                    FROM candlestick_patterns
                    WHERE {where_clause}
                    GROUP BY hour_bucket
                    ORDER BY hour_bucket DESC
                    LIMIT 24;
                """)
                
                time_result = await session.execute(time_stats_query, params)
                time_distribution = [
                    {"hour": row.hour_bucket.isoformat(), "count": row.count}
                    for row in time_result.fetchall()
                ]
                
                return {
                    "total_patterns": stats.total_patterns,
                    "unique_symbols": stats.unique_symbols,
                    "unique_timeframes": stats.unique_timeframes,
                    "unique_patterns": stats.unique_patterns,
                    "confidence_stats": {
                        "average": float(stats.avg_confidence) if stats.avg_confidence else 0.0,
                        "minimum": float(stats.min_confidence) if stats.min_confidence else 0.0,
                        "maximum": float(stats.max_confidence) if stats.max_confidence else 0.0
                    },
                    "volume_confirmation_rate": stats.volume_confirmed_patterns / stats.total_patterns if stats.total_patterns > 0 else 0.0,
                    "trend_distribution": {
                        "bullish": stats.bullish_patterns,
                        "bearish": stats.bearish_patterns,
                        "neutral": stats.neutral_patterns
                    },
                    "pattern_distribution": pattern_distribution,
                    "time_distribution": time_distribution
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting pattern statistics: {e}")
            return {}
    
    async def delete_patterns(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        pattern_name: Optional[str] = None
    ) -> int:
        """Delete patterns based on criteria (returns count of deleted patterns)"""
        try:
            if not self._initialized:
                await self.initialize()
            
            async with self.db_connection.get_async_session() as session:
                # Build conditions
                conditions = []
                params = {}
                
                if symbol:
                    conditions.append("symbol = :symbol")
                    params['symbol'] = symbol
                
                if timeframe:
                    conditions.append("timeframe = :timeframe")
                    params['timeframe'] = timeframe
                
                if start_time:
                    conditions.append("timestamp >= :start_time")
                    params['start_time'] = start_time
                
                if end_time:
                    conditions.append("timestamp <= :end_time")
                    params['end_time'] = end_time
                
                if pattern_name:
                    conditions.append("pattern_name = :pattern_name")
                    params['pattern_name'] = pattern_name
                
                if not conditions:
                    # Prevent accidental deletion of all patterns
                    self.logger.warning("⚠️ No deletion criteria specified - operation cancelled")
                    return 0
                
                where_clause = " AND ".join(conditions)
                
                # Get count before deletion
                count_query = text(f"SELECT COUNT(*) FROM candlestick_patterns WHERE {where_clause}")
                count_result = await session.execute(count_query, params)
                count_before = count_result.scalar()
                
                # Delete patterns
                delete_query = text(f"DELETE FROM candlestick_patterns WHERE {where_clause}")
                await session.execute(delete_query, params)
                await session.commit()
                
                self.logger.info(f"✅ Deleted {count_before} patterns")
                return count_before
                
        except Exception as e:
            self.logger.error(f"❌ Error deleting patterns: {e}")
            return 0
    
    async def cleanup_old_patterns(self, days_to_keep: int = 30) -> int:
        """Clean up patterns older than specified days"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            return await self.delete_patterns(start_time=None, end_time=cutoff_date)
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up old patterns: {e}")
            return 0
    
    async def get_storage_health(self) -> Dict[str, Any]:
        """Get storage system health information"""
        try:
            if not self._initialized:
                await self.initialize()
            
            async with self.db_connection.get_async_session() as session:
                # Check table size and chunk information
                size_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE tablename = 'candlestick_patterns';
                """)
                
                size_result = await session.execute(size_query)
                size_info = size_result.fetchone()
                
                # Check chunk information
                chunk_query = text("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(CASE WHEN is_compressed THEN 1 END) as compressed_chunks,
                        COUNT(CASE WHEN NOT is_compressed THEN 1 END) as uncompressed_chunks
                    FROM timescaledb_information.chunks 
                    WHERE hypertable_name = 'candlestick_patterns';
                """)
                
                chunk_result = await session.execute(chunk_query)
                chunk_info = chunk_result.fetchone()
                
                # Check index usage
                index_query = text("""
                    SELECT 
                        indexrelname,
                        idx_scan as scans,
                        idx_tup_read as tuples_read,
                        idx_tup_fetch as tuples_fetched
                    FROM pg_stat_user_indexes 
                    WHERE relname = 'candlestick_patterns'
                    ORDER BY idx_scan DESC;
                """)
                
                index_result = await session.execute(index_query)
                index_stats = [
                    {
                        "index_name": row.indexrelname,
                        "scans": row.scans,
                        "tuples_read": row.tuples_read,
                        "tuples_fetched": row.tuples_fetched
                    }
                    for row in index_result.fetchall()
                ]
                
                return {
                    "table_size": size_info.size if size_info else "Unknown",
                    "size_bytes": size_info.size_bytes if size_info else 0,
                    "chunks": {
                        "total": chunk_info.total_chunks if chunk_info else 0,
                        "compressed": chunk_info.compressed_chunks if chunk_info else 0,
                        "uncompressed": chunk_info.uncompressed_chunks if chunk_info else 0
                    },
                    "index_usage": index_stats,
                    "initialized": self._initialized,
                    "connection_healthy": await self.db_connection.health_check()
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error getting storage health: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close the pattern storage service"""
        try:
            if self.db_connection:
                await self.db_connection.close()
            self._initialized = False
            self.logger.info("🔌 Pattern Storage Service closed")
        except Exception as e:
            self.logger.error(f"❌ Error closing Pattern Storage Service: {e}")

# Global instance for easy access
pattern_storage_service = PatternStorageService()
