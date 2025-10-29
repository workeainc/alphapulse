"""
Database Connection Module for AlphaPlus
Provides database connection utilities
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from src.app.core.database_manager import db_manager

logger = logging.getLogger(__name__)

async def get_db():
    """Get database connection"""
    try:
        return db_manager
    except Exception as e:
        logger.error(f"❌ Failed to get database connection: {e}")
        return None

async def get_connection():
    """Get database connection (alias for get_db)"""
    return await get_db()

async def execute_query(query: str, params: Optional[tuple] = None):
    """Execute a database query"""
    try:
        if params:
            return await db_manager.execute(query, *params)
        else:
            return await db_manager.execute(query)
    except Exception as e:
        logger.error(f"❌ Failed to execute query: {e}")
        return None

async def fetch_query(query: str, params: Optional[tuple] = None):
    """Fetch results from a database query"""
    try:
        if params:
            return await db_manager.fetch(query, *params)
        else:
            return await db_manager.fetch(query)
    except Exception as e:
        logger.error(f"❌ Failed to fetch query: {e}")
        return None

async def fetch_one(query: str, params: Optional[tuple] = None):
    """Fetch a single result from a database query"""
    try:
        if params:
            return await db_manager.fetchrow(query, *params)
        else:
            return await db_manager.fetchrow(query)
    except Exception as e:
        logger.error(f"❌ Failed to fetch one: {e}")
        return None
