"""
Data Lifecycle Management for AlphaPlus
Handles retention policies, compression, archiving, and cleanup operations
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

class DataLifecycleManager:
    """Comprehensive data lifecycle management for TimescaleDB"""
    
    def __init__(self, async_engine: AsyncEngine):
        self.async_engine = async_engine
        self.logger = logger
        
        # Configuration
        self.archive_base_path = Path("archives")
        self.archive_base_path.mkdir(exist_ok=True)
        
        # Default policies
        self.default_retention_policies = {
            'stream_data': 30,  # 30 days
            'signals': 365,     # 1 year
            'signal_outcomes': 365,  # 1 year
            'feature_snapshot_versions': 180,  # 6 months
            'lifecycle_executions': 90,  # 3 months
            'compression_metrics': 90,  # 3 months
            'cleanup_operations': 90,  # 3 months
        }
        
        self.default_compression_policies = {
            'stream_data': 7,   # 7 days
            'signals': 30,      # 30 days
            'signal_outcomes': 30,  # 30 days
            'feature_snapshot_versions': 14,  # 14 days
            'lifecycle_executions': 7,  # 7 days
            'compression_metrics': 7,  # 7 days
            'cleanup_operations': 7,  # 7 days
        }
    
    async def initialize(self):
        """Initialize the lifecycle manager"""
        try:
            # Create default policies for existing tables
            await self.create_default_policies()
            
            self.logger.info("✅ Data lifecycle manager initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize lifecycle manager: {e}")
            return False
    
    async def create_default_policies(self):
        """Create default lifecycle policies for all tables"""
        try:
            async with self.async_engine.begin() as conn:
                # Create retention policies
                for table_name, retention_days in self.default_retention_policies.items():
                    await self._create_retention_policy_internal(conn, table_name, retention_days)
                
                # Create compression policies
                for table_name, compress_after_days in self.default_compression_policies.items():
                    await self._create_compression_policy_internal(conn, table_name, compress_after_days)
                
            self.logger.info("✅ Default lifecycle policies created")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create default policies: {e}")
    
    async def create_retention_policy(self, table_name: str, retention_days: int, policy_name: str = None) -> bool:
        """Create a retention policy for a table"""
        try:
            async with self.async_engine.begin() as conn:
                return await self._create_retention_policy_internal(conn, table_name, retention_days, policy_name)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create retention policy: {e}")
            return False
    
    async def _create_retention_policy_internal(self, conn, table_name: str, retention_days: int, policy_name: str = None) -> bool:
        """Internal method to create retention policy"""
        try:
            if policy_name is None:
                policy_name = f"{table_name}_retention_{retention_days}d"
            
            # Check if table exists
            table_exists_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """
            result = await conn.execute(text(table_exists_query), {"table_name": table_name})
            table_exists = result.scalar()
            
            if not table_exists:
                self.logger.warning(f"Table {table_name} does not exist, skipping retention policy")
                return False
            
            # Create policy configuration
            policy_config = {
                "retention_days": retention_days,
                "policy_type": "retention",
                "created_at": datetime.now().isoformat()
            }
            
            # Insert or update policy
            upsert_query = """
                INSERT INTO lifecycle_policies (table_name, policy_name, policy_type, policy_config)
                VALUES (:table_name, :policy_name, :policy_type, :policy_config)
                ON CONFLICT (table_name, policy_name) 
                DO UPDATE SET 
                    policy_config = :policy_config,
                    updated_at = NOW()
            """
            
            await conn.execute(text(upsert_query), {
                "table_name": table_name,
                "policy_name": policy_name,
                "policy_type": "retention",
                "policy_config": json.dumps(policy_config)
            })
            
            # Add TimescaleDB retention policy
            retention_query = f"""
                SELECT add_retention_policy('{table_name}', INTERVAL '{retention_days} days')
            """
            await conn.execute(text(retention_query))
            
            self.logger.info(f"✅ Retention policy created: {policy_name} for table {table_name} with {retention_days} days retention")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create retention policy for {table_name}: {e}")
            return False
    
    async def create_compression_policy(self, table_name: str, compress_after_days: int = 7, policy_name: str = None) -> bool:
        """Create a compression policy for a table"""
        try:
            async with self.async_engine.begin() as conn:
                return await self._create_compression_policy_internal(conn, table_name, compress_after_days, policy_name)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create compression policy: {e}")
            return False
    
    async def _create_compression_policy_internal(self, conn, table_name: str, compress_after_days: int, policy_name: str = None) -> bool:
        """Internal method to create compression policy"""
        try:
            if policy_name is None:
                policy_name = f"{table_name}_compression_{compress_after_days}d"
            
            # Check if table exists and is a hypertable
            hypertable_query = """
                SELECT EXISTS (
                    SELECT FROM timescaledb_information.hypertables 
                    WHERE hypertable_name = :table_name
                )
            """
            result = await conn.execute(text(hypertable_query), {"table_name": table_name})
            is_hypertable = result.scalar()
            
            if not is_hypertable:
                self.logger.warning(f"Table {table_name} is not a hypertable, skipping compression policy")
                return False
            
            # Create policy configuration
            policy_config = {
                "compress_after_days": compress_after_days,
                "policy_type": "compression",
                "created_at": datetime.now().isoformat()
            }
            
            # Insert or update policy
            upsert_query = """
                INSERT INTO lifecycle_policies (table_name, policy_name, policy_type, policy_config)
                VALUES (:table_name, :policy_name, :policy_type, :policy_config)
                ON CONFLICT (table_name, policy_name) 
                DO UPDATE SET 
                    policy_config = :policy_config,
                    updated_at = NOW()
            """
            
            await conn.execute(text(upsert_query), {
                "table_name": table_name,
                "policy_name": policy_name,
                "policy_type": "compression",
                "policy_config": json.dumps(policy_config)
            })
            
            # Add TimescaleDB compression policy
            compression_query = f"""
                SELECT add_compression_policy('{table_name}', INTERVAL '{compress_after_days} days')
            """
            await conn.execute(text(compression_query))
            
            self.logger.info(f"✅ Compression policy created: {policy_name} for table {table_name} with {compress_after_days} days delay")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create compression policy for {table_name}: {e}")
            return False
    
    async def execute_cleanup_operation(self, table_name: str, cleanup_type: str, criteria: dict = None) -> int:
        """Execute a cleanup operation on a table"""
        try:
            async with self.async_engine.begin() as conn:
                return await self._execute_cleanup_operation_internal(conn, table_name, cleanup_type, criteria)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to execute cleanup operation: {e}")
            return 0
    
    async def _execute_cleanup_operation_internal(self, conn, table_name: str, cleanup_type: str, criteria: dict = None) -> int:
        """Internal method to execute cleanup operation"""
        try:
            if criteria is None:
                criteria = {}
            
            # Insert cleanup operation record
            cleanup_query = """
                INSERT INTO cleanup_operations (table_name, cleanup_type, cleanup_criteria, execution_status)
                VALUES (:table_name, :cleanup_type, :cleanup_criteria, 'running')
                RETURNING id
            """
            
            result = await conn.execute(text(cleanup_query), {
                "table_name": table_name,
                "cleanup_type": cleanup_type,
                "cleanup_criteria": json.dumps(criteria)
            })
            cleanup_id = result.scalar()
            
            records_removed = 0
            
            # Execute cleanup based on type
            if cleanup_type == "orphaned":
                records_removed = await self._cleanup_orphaned_records(conn, table_name, criteria)
            elif cleanup_type == "duplicate":
                records_removed = await self._cleanup_duplicate_records(conn, table_name, criteria)
            elif cleanup_type == "corrupted":
                records_removed = await self._cleanup_corrupted_records(conn, table_name, criteria)
            elif cleanup_type == "expired":
                records_removed = await self._cleanup_expired_records(conn, table_name, criteria)
            else:
                raise ValueError(f"Unknown cleanup type: {cleanup_type}")
            
            # Update cleanup operation record
            update_query = """
                UPDATE cleanup_operations 
                SET 
                    records_removed = :records_removed,
                    execution_status = 'success',
                    completed_at = NOW()
                WHERE id = :cleanup_id
            """
            
            await conn.execute(text(update_query), {
                "records_removed": records_removed,
                "cleanup_id": cleanup_id
            })
            
            self.logger.info(f"✅ Cleanup operation completed: {cleanup_type} on {table_name}, removed {records_removed} records")
            return records_removed
            
        except Exception as e:
            # Update cleanup operation record with error
            try:
                error_update_query = """
                    UPDATE cleanup_operations 
                    SET 
                        execution_status = 'failed',
                        error_message = :error_message,
                        completed_at = NOW()
                    WHERE id = :cleanup_id
                """
                await conn.execute(text(error_update_query), {
                    "error_message": str(e),
                    "cleanup_id": cleanup_id
                })
            except:
                pass
            
            self.logger.error(f"❌ Cleanup operation failed: {e}")
            return 0
    
    async def _cleanup_orphaned_records(self, conn, table_name: str, criteria: dict) -> int:
        """Clean up orphaned records"""
        try:
            min_age_days = criteria.get('min_age_days', 30)
            
            if table_name == 'signals':
                # Remove signals without outcomes older than min_age_days
                cleanup_query = """
                    DELETE FROM signals 
                    WHERE id NOT IN (SELECT DISTINCT signal_id FROM signal_outcomes WHERE signal_id IS NOT NULL)
                    AND created_at < NOW() - INTERVAL :min_age_days || ' days'
                """
            else:
                # Generic orphaned cleanup - remove records older than min_age_days
                cleanup_query = f"""
                    DELETE FROM {table_name} 
                    WHERE created_at < NOW() - INTERVAL :min_age_days || ' days'
                """
            
            result = await conn.execute(text(cleanup_query), {"min_age_days": min_age_days})
            return result.rowcount
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup orphaned records: {e}")
            return 0
    
    async def _cleanup_duplicate_records(self, conn, table_name: str, criteria: dict) -> int:
        """Clean up duplicate records"""
        try:
            # This is a simplified duplicate cleanup
            # In a real implementation, you'd define specific duplicate criteria per table
            
            if table_name == 'signals':
                # Remove duplicate signals based on symbol, timestamp, and signal_type
                cleanup_query = """
                    DELETE FROM signals 
                    WHERE id NOT IN (
                        SELECT MIN(id) 
                        FROM signals 
                        GROUP BY symbol, DATE_TRUNC('minute', created_at), signal_type
                    )
                """
            else:
                # Generic duplicate cleanup - keep only the latest record per day
                cleanup_query = f"""
                    DELETE FROM {table_name} 
                    WHERE id NOT IN (
                        SELECT MAX(id) 
                        FROM {table_name} 
                        GROUP BY DATE_TRUNC('day', created_at)
                    )
                """
            
            result = await conn.execute(text(cleanup_query))
            return result.rowcount
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup duplicate records: {e}")
            return 0
    
    async def _cleanup_corrupted_records(self, conn, table_name: str, criteria: dict) -> int:
        """Clean up corrupted records"""
        try:
            # Remove records with NULL critical fields
            if table_name == 'signals':
                cleanup_query = """
                    DELETE FROM signals 
                    WHERE symbol IS NULL OR signal_type IS NULL OR created_at IS NULL
                """
            else:
                cleanup_query = f"""
                    DELETE FROM {table_name} 
                    WHERE created_at IS NULL
                """
            
            result = await conn.execute(text(cleanup_query))
            return result.rowcount
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup corrupted records: {e}")
            return 0
    
    async def _cleanup_expired_records(self, conn, table_name: str, criteria: dict) -> int:
        """Clean up expired records"""
        try:
            max_age_days = criteria.get('max_age_days', 365)
            
            cleanup_query = f"""
                DELETE FROM {table_name} 
                WHERE created_at < NOW() - INTERVAL :max_age_days || ' days'
            """
            
            result = await conn.execute(text(cleanup_query), {"max_age_days": max_age_days})
            return result.rowcount
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup expired records: {e}")
            return 0
    
    async def get_statistics(self, table_name: str = None, days_back: int = 30) -> dict:
        """Get lifecycle management statistics"""
        try:
            async with self.async_engine.begin() as conn:
                # Get lifecycle execution statistics
                stats_query = """
                    SELECT 
                        table_name,
                        execution_type as policy_type,
                        COUNT(*) as executions_count,
                        COUNT(*) FILTER (WHERE execution_status = 'success') as success_count,
                        COUNT(*) FILTER (WHERE execution_status = 'failed') as failed_count,
                        COALESCE(SUM(records_affected), 0) as total_records_affected,
                        COALESCE(AVG(execution_duration_ms), 0)::INTEGER as avg_execution_time_ms,
                        MAX(started_at) as last_execution
                    FROM lifecycle_executions
                    WHERE started_at >= NOW() - INTERVAL :days_back || ' days'
                    AND (:table_name IS NULL OR table_name = :table_name)
                    GROUP BY table_name, execution_type
                    ORDER BY table_name, execution_type
                """
                
                result = await conn.execute(text(stats_query), {
                    "days_back": days_back,
                    "table_name": table_name
                })
                
                lifecycle_stats = [dict(row) for row in result.fetchall()]
                
                # Get compression statistics
                compression_query = """
                    SELECT 
                        table_name,
                        COUNT(*) as chunks_compressed,
                        AVG(compression_ratio) as avg_compression_ratio,
                        SUM(original_size_bytes) as total_original_size,
                        SUM(compressed_size_bytes) as total_compressed_size,
                        (SUM(original_size_bytes) - SUM(compressed_size_bytes)) / SUM(original_size_bytes) * 100 as total_space_saved_percent,
                        MAX(compressed_at) as last_compression
                    FROM compression_metrics
                    WHERE compressed_at >= NOW() - INTERVAL :days_back || ' days'
                    AND (:table_name IS NULL OR table_name = :table_name)
                    GROUP BY table_name
                """
                
                result = await conn.execute(text(compression_query), {
                    "days_back": days_back,
                    "table_name": table_name
                })
                
                compression_stats = [dict(row) for row in result.fetchall()]
                
                # Get cleanup statistics
                cleanup_query = """
                    SELECT 
                        table_name,
                        cleanup_type,
                        COUNT(*) as cleanup_operations,
                        SUM(records_removed) as total_records_removed,
                        AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds,
                        MAX(completed_at) as last_cleanup
                    FROM cleanup_operations
                    WHERE execution_status = 'success'
                    AND started_at >= NOW() - INTERVAL :days_back || ' days'
                    AND (:table_name IS NULL OR table_name = :table_name)
                    GROUP BY table_name, cleanup_type
                """
                
                result = await conn.execute(text(cleanup_query), {
                    "days_back": days_back,
                    "table_name": table_name
                })
                
                cleanup_stats = [dict(row) for row in result.fetchall()]
                
                return {
                    'lifecycle_executions': lifecycle_stats,
                    'compression_metrics': compression_stats,
                    'cleanup_operations': cleanup_stats,
                    'summary': {
                        'total_executions': sum(stat['executions_count'] for stat in lifecycle_stats),
                        'successful_executions': sum(stat['success_count'] for stat in lifecycle_stats),
                        'failed_executions': sum(stat['failed_count'] for stat in lifecycle_stats),
                        'total_records_affected': sum(stat['total_records_affected'] for stat in lifecycle_stats),
                        'total_space_saved_mb': sum(stat.get('total_space_saved_percent', 0) for stat in compression_stats),
                        'total_records_removed': sum(stat['total_records_removed'] for stat in cleanup_stats)
                    }
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get lifecycle statistics: {e}")
            return {}
    
    async def archive_table_data(self, table_name: str, date_range: tuple, archive_format: str = 'parquet') -> str:
        """Archive table data to external storage"""
        try:
            start_date, end_date = date_range
            
            # Create archive directory
            archive_dir = self.archive_base_path / table_name / start_date.strftime('%Y-%m')
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate archive filename
            archive_name = f"{table_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.{archive_format}"
            archive_path = archive_dir / archive_name
            
            # Export data to file
            if archive_format == 'parquet':
                await self._export_to_parquet(table_name, date_range, archive_path)
            elif archive_format == 'csv':
                await self._export_to_csv(table_name, date_range, archive_path)
            else:
                raise ValueError(f"Unsupported archive format: {archive_format}")
            
            # Record archive metadata
            await self._record_archive_metadata(table_name, archive_name, str(archive_path), date_range, archive_format)
            
            self.logger.info(f"✅ Data archived: {table_name} to {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to archive table data: {e}")
            return None
    
    async def _export_to_parquet(self, table_name: str, date_range: tuple, archive_path: Path):
        """Export table data to Parquet format"""
        try:
            start_date, end_date = date_range
            
            # Query data
            query = f"""
                SELECT * FROM {table_name}
                WHERE created_at >= :start_date AND created_at < :end_date
                ORDER BY created_at
            """
            
            # Use pandas to export to parquet
            engine = create_engine(str(self.async_engine.url))
            df = pd.read_sql_query(
                query, 
                engine, 
                params={"start_date": start_date, "end_date": end_date}
            )
            
            df.to_parquet(archive_path, index=False)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export to parquet: {e}")
            raise
    
    async def _export_to_csv(self, table_name: str, date_range: tuple, archive_path: Path):
        """Export table data to CSV format"""
        try:
            start_date, end_date = date_range
            
            # Query data
            query = f"""
                SELECT * FROM {table_name}
                WHERE created_at >= :start_date AND created_at < :end_date
                ORDER BY created_at
            """
            
            # Use pandas to export to CSV
            engine = create_engine(str(self.async_engine.url))
            df = pd.read_sql_query(
                query, 
                engine, 
                params={"start_date": start_date, "end_date": end_date}
            )
            
            df.to_csv(archive_path, index=False)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export to CSV: {e}")
            raise
    
    async def _record_archive_metadata(self, table_name: str, archive_name: str, archive_path: str, date_range: tuple, archive_format: str):
        """Record archive metadata in database"""
        try:
            start_date, end_date = date_range
            
            # Get file size
            file_size = os.path.getsize(archive_path)
            
            # Count records
            query = f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE created_at >= :start_date AND created_at < :end_date
            """
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(query), {
                    "start_date": start_date,
                    "end_date": end_date
                })
                records_count = result.scalar()
                
                # Insert archive metadata
                insert_query = """
                    INSERT INTO archive_metadata (
                        table_name, archive_name, archive_path, archive_size_bytes,
                        records_count, date_range_start, date_range_end,
                        archive_format, created_at
                    ) VALUES (
                        :table_name, :archive_name, :archive_path, :archive_size_bytes,
                        :records_count, :date_range_start, :date_range_end,
                        :archive_format, NOW()
                    )
                """
                
                await conn.execute(text(insert_query), {
                    "table_name": table_name,
                    "archive_name": archive_name,
                    "archive_path": archive_path,
                    "archive_size_bytes": file_size,
                    "records_count": records_count,
                    "date_range_start": start_date,
                    "date_range_end": end_date,
                    "archive_format": archive_format
                })
                
        except Exception as e:
            self.logger.error(f"❌ Failed to record archive metadata: {e}")
    
    async def restore_archived_data(self, archive_name: str, target_table: str = None) -> bool:
        """Restore archived data to a table"""
        try:
            # Get archive metadata
            async with self.async_engine.begin() as conn:
                metadata_query = """
                    SELECT * FROM archive_metadata 
                    WHERE archive_name = :archive_name AND is_active = true
                """
                
                result = await conn.execute(text(metadata_query), {"archive_name": archive_name})
                metadata = result.fetchone()
                
                if not metadata:
                    raise ValueError(f"Archive {archive_name} not found or not active")
                
                archive_path = Path(metadata.archive_path)
                if not archive_path.exists():
                    raise FileNotFoundError(f"Archive file not found: {archive_path}")
                
                # Use original table name if target not specified
                if target_table is None:
                    target_table = metadata.table_name
                
                # Restore data based on format
                if metadata.archive_format == 'parquet':
                    await self._restore_from_parquet(archive_path, target_table)
                elif metadata.archive_format == 'csv':
                    await self._restore_from_csv(archive_path, target_table)
                else:
                    raise ValueError(f"Unsupported archive format: {metadata.archive_format}")
                
                # Update archive metadata
                update_query = """
                    UPDATE archive_metadata 
                    SET restored_at = NOW() 
                    WHERE archive_name = :archive_name
                """
                await conn.execute(text(update_query), {"archive_name": archive_name})
                
            self.logger.info(f"✅ Data restored: {archive_name} to {target_table}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore archived data: {e}")
            return False
    
    async def _restore_from_parquet(self, archive_path: Path, target_table: str):
        """Restore data from Parquet file"""
        try:
            # Read parquet file
            df = pd.read_parquet(archive_path)
            
            # Insert data into table
            engine = create_engine(str(self.async_engine.url))
            df.to_sql(target_table, engine, if_exists='append', index=False)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore from parquet: {e}")
            raise
    
    async def _restore_from_csv(self, archive_path: Path, target_table: str):
        """Restore data from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(archive_path)
            
            # Insert data into table
            engine = create_engine(str(self.async_engine.url))
            df.to_sql(target_table, engine, if_exists='append', index=False)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to restore from CSV: {e}")
            raise
