"""
Dynamic Symbol List Manager for AlphaPulse
Automatically fetches and maintains top 100 Binance symbols (50 futures + 50 spot)
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
import aiohttp
import asyncpg
from dataclasses import dataclass
import yaml
import os

logger = logging.getLogger(__name__)

@dataclass
class SymbolInfo:
    """Information about a tracked symbol"""
    symbol: str
    market_type: str  # 'futures' or 'spot'
    base_asset: str
    quote_asset: str
    volume_24h: float
    volume_rank: int
    price_change_24h: float
    last_price: float
    is_active: bool
    last_updated: datetime

class DynamicSymbolManager:
    """
    Manages dynamic symbol list with automatic daily updates
    Fetches top 100 Binance symbols: 50 futures + 50 spot
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None, config_path: str = "config/symbol_config.yaml"):
        self.db_pool = db_pool
        self.config = self._load_config(config_path)
        self.logger = logger
        
        # Binance API endpoints
        self.binance_futures_api = "https://fapi.binance.com"
        self.binance_spot_api = "https://api.binance.com"
        
        # Cache
        self._symbol_cache: Dict[str, List[SymbolInfo]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 3600  # 1 hour cache
        
        # State
        self.is_initialized = False
        self.last_update_time: Optional[datetime] = None
        
        # Stats
        self.stats = {
            'total_updates': 0,
            'last_update_duration_seconds': 0.0,
            'symbols_added': 0,
            'symbols_removed': 0,
            'update_errors': 0
        }
        
        logger.info("✅ Dynamic Symbol Manager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            full_path = os.path.join(os.path.dirname(__file__), '..', '..', config_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"✅ Loaded config from {config_path}")
                    return config
        except Exception as e:
            logger.warning(f"⚠️ Could not load config file: {e}, using defaults")
        
        # Default configuration
        return {
            'symbol_management': {
                'total_symbols': 100,
                'futures_count': 50,
                'spot_count': 50,
                'update_interval_hours': 24,
                'min_volume_24h': 1000000
            }
        }
    
    async def initialize(self, db_pool: asyncpg.Pool):
        """Initialize with database connection"""
        self.db_pool = db_pool
        self.is_initialized = True
        
        # Load or create initial symbol list
        existing_symbols = await self._load_symbols_from_database()
        
        if not existing_symbols:
            logger.info("📋 No symbols in database, performing initial fetch...")
            await self.update_symbol_list()
        else:
            logger.info(f"✅ Loaded {len(existing_symbols)} symbols from database")
            self._symbol_cache['all'] = existing_symbols
            self._cache_timestamp = datetime.now(timezone.utc)
    
    async def fetch_top_futures(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch top USDT perpetual futures by 24h volume"""
        try:
            url = f"{self.binance_futures_api}/fapi/v1/ticker/24hr"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter for USDT perpetuals and sort by volume
                        usdt_futures = [
                            item for item in data 
                            if item['symbol'].endswith('USDT') and 
                            float(item['quoteVolume']) > self.config['symbol_management']['min_volume_24h']
                        ]
                        
                        # Sort by quote volume (USD volume)
                        usdt_futures.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                        
                        # Take top N
                        top_futures = usdt_futures[:limit]
                        
                        logger.info(f"✅ Fetched top {len(top_futures)} USDT perpetual futures")
                        return top_futures
                    else:
                        logger.error(f"❌ Binance futures API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Error fetching futures symbols: {e}")
            self.stats['update_errors'] += 1
            return []
    
    async def fetch_top_spot(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch top USDT spot pairs by 24h volume"""
        try:
            url = f"{self.binance_spot_api}/api/v3/ticker/24hr"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Filter for USDT spot pairs and sort by volume
                        usdt_spot = [
                            item for item in data 
                            if item['symbol'].endswith('USDT') and 
                            float(item['quoteVolume']) > self.config['symbol_management']['min_volume_24h']
                        ]
                        
                        # Sort by quote volume
                        usdt_spot.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
                        
                        # Take top N
                        top_spot = usdt_spot[:limit]
                        
                        logger.info(f"✅ Fetched top {len(top_spot)} USDT spot pairs")
                        return top_spot
                    else:
                        logger.error(f"❌ Binance spot API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Error fetching spot symbols: {e}")
            self.stats['update_errors'] += 1
            return []
    
    async def update_symbol_list(self) -> bool:
        """Update the symbol list from Binance API and store in database"""
        update_start = datetime.now(timezone.utc)
        
        try:
            logger.info("🔄 Updating symbol list from Binance...")
            
            # Fetch from Binance APIs in parallel
            futures_data, spot_data = await asyncio.gather(
                self.fetch_top_futures(self.config['symbol_management']['futures_count']),
                self.fetch_top_spot(self.config['symbol_management']['spot_count'])
            )
            
            if not futures_data and not spot_data:
                logger.error("❌ Failed to fetch any symbols from Binance")
                return False
            
            # Process and store symbols
            symbols_to_store = []
            rank = 1
            
            # Process futures
            for item in futures_data:
                symbol_info = self._parse_futures_ticker(item, rank)
                symbols_to_store.append(symbol_info)
                rank += 1
            
            # Process spot
            for item in spot_data:
                symbol_info = self._parse_spot_ticker(item, rank)
                symbols_to_store.append(symbol_info)
                rank += 1
            
            # Store in database
            success = await self._store_symbols_in_database(symbols_to_store)
            
            if success:
                # Archive snapshot to history
                await self._archive_snapshot()
                
                # Update cache
                self._symbol_cache['all'] = symbols_to_store
                self._symbol_cache['futures'] = [s for s in symbols_to_store if s.market_type == 'futures']
                self._symbol_cache['spot'] = [s for s in symbols_to_store if s.market_type == 'spot']
                self._cache_timestamp = datetime.now(timezone.utc)
                
                # Update stats
                self.last_update_time = update_start
                update_duration = (datetime.now(timezone.utc) - update_start).total_seconds()
                self.stats['total_updates'] += 1
                self.stats['last_update_duration_seconds'] = update_duration
                
                logger.info(f"✅ Updated {len(symbols_to_store)} symbols in {update_duration:.2f}s")
                return True
            else:
                logger.error("❌ Failed to store symbols in database")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating symbol list: {e}")
            self.stats['update_errors'] += 1
            return False
    
    def _parse_futures_ticker(self, item: Dict, rank: int) -> SymbolInfo:
        """Parse Binance futures ticker data"""
        symbol = item['symbol']
        base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol
        
        return SymbolInfo(
            symbol=symbol,
            market_type='futures',
            base_asset=base_asset,
            quote_asset='USDT',
            volume_24h=float(item['quoteVolume']),
            volume_rank=rank,
            price_change_24h=float(item.get('priceChangePercent', 0)),
            last_price=float(item['lastPrice']),
            is_active=True,
            last_updated=datetime.now(timezone.utc)
        )
    
    def _parse_spot_ticker(self, item: Dict, rank: int) -> SymbolInfo:
        """Parse Binance spot ticker data"""
        symbol = item['symbol']
        base_asset = symbol[:-4] if symbol.endswith('USDT') else symbol
        
        return SymbolInfo(
            symbol=symbol,
            market_type='spot',
            base_asset=base_asset,
            quote_asset='USDT',
            volume_24h=float(item['quoteVolume']),
            volume_rank=rank,
            price_change_24h=float(item.get('priceChangePercent', 0)),
            last_price=float(item['lastPrice']),
            is_active=True,
            last_updated=datetime.now(timezone.utc)
        )
    
    async def _store_symbols_in_database(self, symbols: List[SymbolInfo]) -> bool:
        """Store symbol list in database"""
        try:
            if not self.db_pool:
                logger.error("❌ Database pool not initialized")
                return False
            
            async with self.db_pool.acquire() as conn:
                # Start transaction
                async with conn.transaction():
                    # Deactivate all existing symbols first
                    await conn.execute("UPDATE tracked_symbols SET is_active = false")
                    
                    # Insert or update new symbols
                    for symbol_info in symbols:
                        await conn.execute("""
                            INSERT INTO tracked_symbols (
                                symbol, market_type, base_asset, quote_asset,
                                volume_24h, volume_rank, price_change_24h, last_price,
                                is_active, last_updated
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT (symbol) DO UPDATE SET
                                market_type = EXCLUDED.market_type,
                                base_asset = EXCLUDED.base_asset,
                                quote_asset = EXCLUDED.quote_asset,
                                volume_24h = EXCLUDED.volume_24h,
                                volume_rank = EXCLUDED.volume_rank,
                                price_change_24h = EXCLUDED.price_change_24h,
                                last_price = EXCLUDED.last_price,
                                is_active = EXCLUDED.is_active,
                                last_updated = EXCLUDED.last_updated
                        """, symbol_info.symbol, symbol_info.market_type, symbol_info.base_asset,
                             symbol_info.quote_asset, symbol_info.volume_24h, symbol_info.volume_rank,
                             symbol_info.price_change_24h, symbol_info.last_price, 
                             symbol_info.is_active, symbol_info.last_updated)
                
                logger.info(f"✅ Stored {len(symbols)} symbols in database")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error storing symbols in database: {e}")
            return False
    
    async def _load_symbols_from_database(self) -> List[SymbolInfo]:
        """Load active symbols from database"""
        try:
            if not self.db_pool:
                return []
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT symbol, market_type, base_asset, quote_asset,
                           volume_24h, volume_rank, price_change_24h, last_price,
                           is_active, last_updated
                    FROM tracked_symbols
                    WHERE is_active = true
                    ORDER BY volume_rank
                """)
                
                symbols = []
                for row in rows:
                    symbols.append(SymbolInfo(
                        symbol=row['symbol'],
                        market_type=row['market_type'],
                        base_asset=row['base_asset'],
                        quote_asset=row['quote_asset'],
                        volume_24h=float(row['volume_24h']),
                        volume_rank=row['volume_rank'],
                        price_change_24h=float(row['price_change_24h']),
                        last_price=float(row['last_price']),
                        is_active=row['is_active'],
                        last_updated=row['last_updated']
                    ))
                
                return symbols
                
        except Exception as e:
            logger.error(f"❌ Error loading symbols from database: {e}")
            return []
    
    async def _archive_snapshot(self):
        """Archive current symbol list to history table"""
        try:
            if not self.db_pool:
                return
            
            async with self.db_pool.acquire() as conn:
                await conn.execute("SELECT archive_symbol_snapshot()")
                logger.debug("📊 Archived symbol snapshot to history")
                
        except Exception as e:
            logger.error(f"❌ Error archiving snapshot: {e}")
    
    async def get_active_symbols(self, market_type: Optional[str] = None) -> List[str]:
        """Get list of active symbols, optionally filtered by market type"""
        try:
            # Check cache first
            cache_key = market_type or 'all'
            if self._is_cache_valid() and cache_key in self._symbol_cache:
                symbols = self._symbol_cache[cache_key]
                return [s.symbol for s in symbols]
            
            # Load from database
            symbols = await self._load_symbols_from_database()
            
            # Filter by market type if specified
            if market_type:
                symbols = [s for s in symbols if s.market_type == market_type]
            
            # Update cache
            self._symbol_cache[cache_key] = symbols
            self._cache_timestamp = datetime.now(timezone.utc)
            
            return [s.symbol for s in symbols]
            
        except Exception as e:
            logger.error(f"❌ Error getting active symbols: {e}")
            return []
    
    async def get_futures_symbols(self) -> List[str]:
        """Get active futures symbols"""
        return await self.get_active_symbols(market_type='futures')
    
    async def get_spot_symbols(self) -> List[str]:
        """Get active spot symbols"""
        return await self.get_active_symbols(market_type='spot')
    
    async def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get detailed information for a specific symbol"""
        try:
            if not self.db_pool:
                return None
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT symbol, market_type, base_asset, quote_asset,
                           volume_24h, volume_rank, price_change_24h, last_price,
                           is_active, last_updated
                    FROM tracked_symbols
                    WHERE symbol = $1
                """, symbol)
                
                if row:
                    return SymbolInfo(
                        symbol=row['symbol'],
                        market_type=row['market_type'],
                        base_asset=row['base_asset'],
                        quote_asset=row['quote_asset'],
                        volume_24h=float(row['volume_24h']),
                        volume_rank=row['volume_rank'],
                        price_change_24h=float(row['price_change_24h']),
                        last_price=float(row['last_price']),
                        is_active=row['is_active'],
                        last_updated=row['last_updated']
                    )
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting symbol info for {symbol}: {e}")
            return None
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._cache_timestamp:
            return False
        
        age = (datetime.now(timezone.utc) - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl_seconds
    
    async def should_update(self) -> bool:
        """Check if symbol list should be updated"""
        if not self.last_update_time:
            return True
        
        update_interval_hours = self.config['symbol_management']['update_interval_hours']
        age = datetime.now(timezone.utc) - self.last_update_time
        return age > timedelta(hours=update_interval_hours)
    
    async def auto_update_loop(self):
        """Background task to automatically update symbol list"""
        logger.info("🔄 Starting auto-update loop...")
        
        while True:
            try:
                if await self.should_update():
                    logger.info("⏰ Time to update symbol list")
                    await self.update_symbol_list()
                
                # Check every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"❌ Error in auto-update loop: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    def get_stats(self) -> Dict[str, Any]:
        """Get symbol manager statistics"""
        return {
            'stats': self.stats,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'cache_valid': self._is_cache_valid(),
            'cached_symbols': {
                'all': len(self._symbol_cache.get('all', [])),
                'futures': len(self._symbol_cache.get('futures', [])),
                'spot': len(self._symbol_cache.get('spot', []))
            }
        }

# Global instance
symbol_manager = DynamicSymbolManager()

