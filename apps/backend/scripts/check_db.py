#!/usr/bin/env python3
import asyncio
from ..src.database.connection import TimescaleDBConnection
from sqlalchemy import text

async def check_db():
    db = TimescaleDBConnection()
    db.initialize()
    
    async with db.get_async_session() as session:
        # Check all tables
        result = await session.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [r[0] for r in result.fetchall()]
        print("All tables:", tables)
        
        # Check for accuracy table specifically
        result = await session.execute(text("SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%accuracy%'"))
        accuracy_tables = [r[0] for r in result.fetchall()]
        print("Accuracy tables:", accuracy_tables)

asyncio.run(check_db())
