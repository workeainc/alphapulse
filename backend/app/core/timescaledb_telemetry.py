"""
TimescaleDB Telemetry Framework
Monitors chunk usage, compression savings, index usage, and database health
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

@dataclass
class ChunkStats:
    """Statistics for a TimescaleDB chunk"""
    hypertable_name: str
    chunk_name: str
    table_bytes: int
    index_bytes: int
    total_bytes: int
    compression_status: str
    chunk_time_range: str

@dataclass
class CompressionStats:
    """Compression statistics for a hypertable"""
    hypertable_name: str
    total_bytes: int
    compressed_total_bytes: int
    compression_ratio: float
    compression_savings_bytes: int
    compression_savings_percent: float

@dataclass
class IndexStats:
    """Index usage statistics"""
    table_name: str
    index_name: str
    index_scans: int
    index_tuples_read: int
    index_tuples_fetched: int
    index_size_bytes: int
    last_used: Optional[datetime] = None

@dataclass
class TableStats:
    """Table statistics"""
    table_name: str
    total_rows: int
    table_size_bytes: int
    index_size_bytes: int
    total_size_bytes: int
    seq_scans: int
    seq_tuples_read: int
    idx_scans: int
    idx_tuples_fetched: int
    last_vacuum: Optional[datetime] = None
    last_analyze: Optional[datetime] = None

@dataclass
class TimescaleDBHealth:
    """Overall TimescaleDB health metrics"""
    total_hypertables: int
    total_chunks: int
    compressed_chunks: int
    compression_ratio: float
    total_size_bytes: int
    compressed_size_bytes: int
    space_savings_bytes: int
    space_savings_percent: float
    active_compression_jobs: int
    last_compression_run: Optional[datetime] = None

class TimescaleDBTelemetry:
    """Monitor TimescaleDB performance and health metrics"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.logger = logging.getLogger(__name__)
    
    async def get_chunk_stats(self, hypertable_name: Optional[str] = None) -> List[ChunkStats]:
        """Get chunk statistics for hypertables"""
        try:
            if hypertable_name:
                query = text("""
                    SELECT 
                        h.hypertable_name,
                        c.chunk_name,
                        pg_total_relation_size(c.chunk_name::regclass) as total_bytes,
                        pg_relation_size(c.chunk_name::regclass) as table_bytes,
                        pg_total_relation_size(c.chunk_name::regclass) - pg_relation_size(c.chunk_name::regclass) as index_bytes,
                        CASE 
                            WHEN c.is_compressed THEN 'compressed'
                            ELSE 'uncompressed'
                        END as compression_status,
                        format('[%s, %s]', c.range_start, c.range_end) as chunk_time_range
                    FROM timescaledb_information.chunks c
                    JOIN timescaledb_information.hypertables h ON c.hypertable_name = h.hypertable_name
                    WHERE h.hypertable_name = :hypertable_name
                    ORDER BY c.range_start DESC
                """)
                result = await self.session.execute(query, {"hypertable_name": hypertable_name})
            else:
                query = text("""
                    SELECT 
                        h.hypertable_name,
                        c.chunk_name,
                        pg_total_relation_size(c.chunk_name::regclass) as total_bytes,
                        pg_relation_size(c.chunk_name::regclass) as table_bytes,
                        pg_total_relation_size(c.chunk_name::regclass) - pg_relation_size(c.chunk_name::regclass) as index_bytes,
                        CASE 
                            WHEN c.is_compressed THEN 'compressed'
                            ELSE 'uncompressed'
                        END as compression_status,
                        format('[%s, %s]', c.range_start, c.range_end) as chunk_time_range
                    FROM timescaledb_information.chunks c
                    JOIN timescaledb_information.hypertables h ON c.hypertable_name = h.hypertable_name
                    ORDER BY h.hypertable_name, c.range_start DESC
                """)
                result = await self.session.execute(query)
            
            chunks = []
            for row in result:
                chunks.append(ChunkStats(
                    hypertable_name=row.hypertable_name,
                    chunk_name=row.chunk_name,
                    table_bytes=row.table_bytes or 0,
                    index_bytes=row.index_bytes or 0,
                    total_bytes=row.total_bytes or 0,
                    compression_status=row.compression_status,
                    chunk_time_range=row.chunk_time_range
                ))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error getting chunk stats: {e}")
            return []
    
    async def get_compression_stats(self, hypertable_name: Optional[str] = None) -> List[CompressionStats]:
        """Get compression statistics for hypertables"""
        try:
            if hypertable_name:
                query = text("""
                    SELECT 
                        h.hypertable_name,
                        COALESCE(SUM(pg_total_relation_size(c.chunk_name::regclass)), 0) as total_bytes,
                        COALESCE(SUM(CASE WHEN c.is_compressed THEN pg_total_relation_size(c.chunk_name::regclass) ELSE 0 END), 0) as compressed_total_bytes
                    FROM timescaledb_information.chunks c
                    JOIN timescaledb_information.hypertables h ON c.hypertable_name = h.hypertable_name
                    WHERE h.hypertable_name = :hypertable_name
                    GROUP BY h.hypertable_name
                """)
                result = await self.session.execute(query, {"hypertable_name": hypertable_name})
            else:
                query = text("""
                    SELECT 
                        h.hypertable_name,
                        COALESCE(SUM(pg_total_relation_size(c.chunk_name::regclass)), 0) as total_bytes,
                        COALESCE(SUM(CASE WHEN c.is_compressed THEN pg_total_relation_size(c.chunk_name::regclass) ELSE 0 END), 0) as compressed_total_bytes
                    FROM timescaledb_information.chunks c
                    JOIN timescaledb_information.hypertables h ON c.hypertable_name = h.hypertable_name
                    GROUP BY h.hypertable_name
                """)
                result = await self.session.execute(query)
            
            compression_stats = []
            for row in result:
                total_bytes = row.total_bytes or 0
                compressed_bytes = row.compressed_total_bytes or 0
                savings_bytes = total_bytes - compressed_bytes
                savings_percent = (savings_bytes / total_bytes * 100) if total_bytes > 0 else 0
                compression_ratio = (compressed_bytes / total_bytes) if total_bytes > 0 else 1.0
                
                compression_stats.append(CompressionStats(
                    hypertable_name=row.hypertable_name,
                    total_bytes=total_bytes,
                    compressed_total_bytes=compressed_bytes,
                    compression_ratio=compression_ratio,
                    compression_savings_bytes=savings_bytes,
                    compression_savings_percent=savings_percent
                ))
            
            return compression_stats
            
        except Exception as e:
            self.logger.error(f"Error getting compression stats: {e}")
            return []
    
    async def get_index_stats(self, table_name: Optional[str] = None) -> List[IndexStats]:
        """Get index usage statistics"""
        try:
            if table_name:
                query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch,
                        pg_relation_size(indexrelid) as index_size_bytes
                    FROM pg_stat_user_indexes
                    WHERE tablename = :table_name
                    ORDER BY idx_scan DESC
                """)
                result = await self.session.execute(query, {"table_name": table_name})
            else:
                query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch,
                        pg_relation_size(indexrelid) as index_size_bytes
                    FROM pg_stat_user_indexes
                    ORDER BY idx_scan DESC
                """)
                result = await self.session.execute(query)
            
            index_stats = []
            for row in result:
                index_stats.append(IndexStats(
                    table_name=f"{row.schemaname}.{row.tablename}",
                    index_name=row.indexname,
                    index_scans=row.idx_scan or 0,
                    index_tuples_read=row.idx_tup_read or 0,
                    index_tuples_fetched=row.idx_tup_fetch or 0,
                    index_size_bytes=row.index_size_bytes or 0
                ))
            
            return index_stats
            
        except Exception as e:
            self.logger.error(f"Error getting index stats: {e}")
            return []
    
    async def get_table_stats(self, table_name: Optional[str] = None) -> List[TableStats]:
        """Get table statistics"""
        try:
            if table_name:
                query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins + n_tup_upd + n_tup_del as total_rows,
                        pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes,
                        pg_relation_size(schemaname||'.'||tablename) as table_size_bytes,
                        pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename) as index_size_bytes,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        last_vacuum,
                        last_analyze
                    FROM pg_stat_user_tables
                    WHERE tablename = :table_name
                """)
                result = await self.session.execute(query, {"table_name": table_name})
            else:
                query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins + n_tup_upd + n_tup_del as total_rows,
                        pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes,
                        pg_relation_size(schemaname||'.'||tablename) as table_size_bytes,
                        pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename) as index_size_bytes,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        last_vacuum,
                        last_analyze
                    FROM pg_stat_user_tables
                    ORDER BY total_size_bytes DESC
                """)
                result = await self.session.execute(query)
            
            table_stats = []
            for row in result:
                table_stats.append(TableStats(
                    table_name=f"{row.schemaname}.{row.tablename}",
                    total_rows=row.total_rows or 0,
                    table_size_bytes=row.table_size_bytes or 0,
                    index_size_bytes=row.index_size_bytes or 0,
                    total_size_bytes=row.total_size_bytes or 0,
                    seq_scans=row.seq_scan or 0,
                    seq_tuples_read=row.seq_tup_read or 0,
                    idx_scans=row.idx_scan or 0,
                    idx_tuples_fetched=row.idx_tup_fetch or 0,
                    last_vacuum=row.last_vacuum,
                    last_analyze=row.last_analyze
                ))
            
            return table_stats
            
        except Exception as e:
            self.logger.error(f"Error getting table stats: {e}")
            return []
    
    async def get_timescaledb_health(self) -> TimescaleDBHealth:
        """Get overall TimescaleDB health metrics"""
        try:
            # Get basic hypertable info
            query = text("""
                SELECT 
                    COUNT(DISTINCT h.hypertable_name) as total_hypertables,
                    COUNT(c.chunk_name) as total_chunks,
                    COUNT(CASE WHEN c.is_compressed THEN 1 END) as compressed_chunks,
                    COALESCE(SUM(pg_total_relation_size(c.chunk_name::regclass)), 0) as total_size_bytes,
                    COALESCE(SUM(CASE WHEN c.is_compressed THEN pg_total_relation_size(c.chunk_name::regclass) ELSE 0 END), 0) as compressed_size_bytes
                FROM timescaledb_information.hypertables h
                LEFT JOIN timescaledb_information.chunks c ON h.hypertable_name = c.hypertable_name
            """)
            result = await self.session.execute(query)
            row = result.fetchone()
            
            if not row:
                return TimescaleDBHealth(
                    total_hypertables=0,
                    total_chunks=0,
                    compressed_chunks=0,
                    compression_ratio=0.0,
                    total_size_bytes=0,
                    compressed_size_bytes=0,
                    space_savings_bytes=0,
                    space_savings_percent=0.0,
                    active_compression_jobs=0
                )
            
            total_size = row.total_size_bytes or 0
            compressed_size = row.compressed_size_bytes or 0
            space_savings = total_size - compressed_size
            space_savings_percent = (space_savings / total_size * 100) if total_size > 0 else 0
            compression_ratio = (compressed_size / total_size) if total_size > 0 else 1.0
            
            # Get compression job info
            compression_query = text("""
                SELECT COUNT(*) as active_jobs
                FROM timescaledb_information.jobs
                WHERE proc_name = 'policy_compression'
                AND scheduled = true
            """)
            compression_result = await self.session.execute(compression_query)
            compression_row = compression_result.fetchone()
            active_compression_jobs = compression_row.active_jobs if compression_row else 0
            
            return TimescaleDBHealth(
                total_hypertables=row.total_hypertables or 0,
                total_chunks=row.total_chunks or 0,
                compressed_chunks=row.compressed_chunks or 0,
                compression_ratio=compression_ratio,
                total_size_bytes=total_size,
                compressed_size_bytes=compressed_size,
                space_savings_bytes=space_savings,
                space_savings_percent=space_savings_percent,
                active_compression_jobs=active_compression_jobs
            )
            
        except Exception as e:
            self.logger.error(f"Error getting TimescaleDB health: {e}")
            return TimescaleDBHealth(
                total_hypertables=0,
                total_chunks=0,
                compressed_chunks=0,
                compression_ratio=0.0,
                total_size_bytes=0,
                compressed_size_bytes=0,
                space_savings_bytes=0,
                space_savings_percent=0.0,
                active_compression_jobs=0
            )
    
    async def get_unused_indexes(self, min_scans: int = 0) -> List[IndexStats]:
        """Get indexes that are rarely or never used"""
        try:
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch,
                    pg_relation_size(indexrelid) as index_size_bytes
                FROM pg_stat_user_indexes
                WHERE idx_scan <= :min_scans
                ORDER BY pg_relation_size(indexrelid) DESC
            """)
            result = await self.session.execute(query, {"min_scans": min_scans})
            
            unused_indexes = []
            for row in result:
                unused_indexes.append(IndexStats(
                    table_name=f"{row.schemaname}.{row.tablename}",
                    index_name=row.indexname,
                    index_scans=row.idx_scan or 0,
                    index_tuples_read=row.idx_tup_read or 0,
                    index_tuples_fetched=row.idx_tup_fetch or 0,
                    index_size_bytes=row.index_size_bytes or 0
                ))
            
            return unused_indexes
            
        except Exception as e:
            self.logger.error(f"Error getting unused indexes: {e}")
            return []
    
    async def get_large_tables(self, min_size_mb: int = 100) -> List[TableStats]:
        """Get tables larger than specified size"""
        try:
            query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins + n_tup_upd + n_tup_del as total_rows,
                    pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes,
                    pg_relation_size(schemaname||'.'||tablename) as table_size_bytes,
                    pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename) as index_size_bytes,
                    seq_scan,
                    seq_tup_read,
                    idx_scan,
                    idx_tup_fetch,
                    last_vacuum,
                    last_analyze
                FROM pg_stat_user_tables
                WHERE pg_total_relation_size(schemaname||'.'||tablename) > :min_size_bytes
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            result = await self.session.execute(query, {"min_size_bytes": min_size_mb * 1024 * 1024})
            
            large_tables = []
            for row in result:
                large_tables.append(TableStats(
                    table_name=f"{row.schemaname}.{row.tablename}",
                    total_rows=row.total_rows or 0,
                    table_size_bytes=row.table_size_bytes or 0,
                    index_size_bytes=row.index_size_bytes or 0,
                    total_size_bytes=row.total_size_bytes or 0,
                    seq_scans=row.seq_scan or 0,
                    seq_tuples_read=row.seq_tup_read or 0,
                    idx_scans=row.idx_scan or 0,
                    idx_tuples_fetched=row.idx_tup_fetch or 0,
                    last_vacuum=row.last_vacuum,
                    last_analyze=row.last_analyze
                ))
            
            return large_tables
            
        except Exception as e:
            self.logger.error(f"Error getting large tables: {e}")
            return []
