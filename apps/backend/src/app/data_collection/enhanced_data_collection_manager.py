"""
Enhanced Data Collection Manager for AlphaPulse
Unified manager for collecting and storing all market intelligence data
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncpg
import ccxt

from .market_intelligence_collector import MarketIntelligenceCollector, MarketIntelligenceData
from .volume_positioning_analyzer import VolumePositioningAnalyzer, VolumeAnalysis

logger = logging.getLogger(__name__)

class EnhancedDataCollectionManager:
    """
    Unified data collection manager
    Coordinates collection of all market intelligence data
    """
    
    def __init__(self, db_pool: asyncpg.Pool, exchange: ccxt.Exchange):
        self.db_pool = db_pool
        self.exchange = exchange
        
        # Initialize data collectors
        self.market_intelligence_collector = MarketIntelligenceCollector(db_pool)
        self.volume_analyzer = VolumePositioningAnalyzer(db_pool, exchange)
        
        # Collection intervals
        self.market_intelligence_interval = 300  # 5 minutes
        self.volume_analysis_interval = 60  # 1 minute
        
        # Symbols to analyze
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        
        # Collection tasks
        self.collection_tasks = []
        self.is_running = False
        
        logger.info("Enhanced Data Collection Manager initialized")
    
    async def start_collection(self):
        """Start all data collection processes"""
        try:
            logger.info("🚀 Starting enhanced data collection...")
            self.is_running = True
            
            # Start market intelligence collection
            market_intelligence_task = asyncio.create_task(
                self._run_market_intelligence_collection()
            )
            self.collection_tasks.append(market_intelligence_task)
            
            # Start volume analysis collection
            volume_analysis_task = asyncio.create_task(
                self._run_volume_analysis_collection()
            )
            self.collection_tasks.append(volume_analysis_task)
            
            logger.info("✅ Enhanced data collection started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting data collection: {e}")
            self.is_running = False
    
    async def stop_collection(self):
        """Stop all data collection processes"""
        try:
            logger.info("🛑 Stopping enhanced data collection...")
            self.is_running = False
            
            # Cancel all collection tasks
            for task in self.collection_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.collection_tasks, return_exceptions=True)
            self.collection_tasks.clear()
            
            logger.info("✅ Enhanced data collection stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping data collection: {e}")
    
    async def get_collection_status(self) -> Dict[str, Any]:
        """Get data collection status"""
        try:
            return {
                "is_running": self.is_running,
                "active_tasks": len(self.collection_tasks),
                "symbols": self.symbols,
                "market_intelligence_interval": self.market_intelligence_interval,
                "volume_analysis_interval": self.volume_analysis_interval,
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting collection status: {e}")
            return {
                "is_running": False,
                "active_tasks": 0,
                "symbols": [],
                "market_intelligence_interval": 0,
                "volume_analysis_interval": 0,
                "last_update": None
            }
    
    async def _run_market_intelligence_collection(self):
        """Run market intelligence collection loop"""
        while self.is_running:
            try:
                logger.info("🔄 Running market intelligence collection cycle...")
                
                async with self.market_intelligence_collector:
                    # Collect market intelligence
                    intelligence_data = await self.market_intelligence_collector.collect_market_intelligence()
                    
                    # Store in database
                    success = await self.market_intelligence_collector.store_market_intelligence(intelligence_data)
                    
                    if success:
                        logger.info("✅ Market intelligence collection completed")
                    else:
                        logger.error("❌ Market intelligence collection failed")
                
                # Wait for next collection cycle
                await asyncio.sleep(self.market_intelligence_interval)
                
            except asyncio.CancelledError:
                logger.info("Market intelligence collection cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in market intelligence collection: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _run_volume_analysis_collection(self):
        """Run volume analysis collection loop"""
        while self.is_running:
            try:
                logger.info("🔄 Running volume analysis collection cycle...")
                
                # Analyze volume positioning for all symbols
                for symbol in self.symbols:
                    try:
                        # Perform volume analysis
                        volume_analysis = await self.volume_analyzer.analyze_volume_positioning(symbol, '1h')
                        
                        # Store in database
                        success = await self.volume_analyzer.store_volume_analysis(volume_analysis, '1h')
                        
                        if success:
                            logger.info(f"✅ Volume analysis completed for {symbol}")
                        else:
                            logger.error(f"❌ Volume analysis failed for {symbol}")
                        
                        # Small delay between symbols to avoid rate limiting
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"❌ Error analyzing volume for {symbol}: {e}")
                        continue
                
                # Wait for next collection cycle
                await asyncio.sleep(self.volume_analysis_interval)
                
            except asyncio.CancelledError:
                logger.info("Volume analysis collection cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in volume analysis collection: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def get_latest_market_data(self) -> Dict[str, Any]:
        """Get latest market intelligence and volume data"""
        try:
            # Get latest market intelligence
            market_intelligence = await self.market_intelligence_collector.get_latest_market_intelligence()
            
            # Get latest volume analysis for each symbol
            volume_data = {}
            for symbol in self.symbols:
                try:
                    # Get latest volume analysis from database
                    async with self.db_pool.acquire() as conn:
                        row = await conn.fetchrow("""
                            SELECT * FROM volume_analysis 
                            WHERE symbol = $1 
                            ORDER BY timestamp DESC 
                            LIMIT 1
                        """, symbol)
                        
                        if row:
                            volume_data[symbol] = {
                                'volume_ratio': float(row['volume_ratio']),
                                'volume_trend': row['volume_trend'],
                                'order_book_imbalance': float(row['order_book_imbalance']),
                                'volume_positioning_score': float(row['volume_positioning_score']),
                                'volume_analysis': row['volume_analysis'],
                                'timestamp': row['timestamp']
                            }
                except Exception as e:
                    logger.error(f"Error getting volume data for {symbol}: {e}")
                    continue
            
            return {
                'market_intelligence': market_intelligence,
                'volume_data': volume_data,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting latest market data: {e}")
            return {
                'market_intelligence': None,
                'volume_data': {},
                'timestamp': datetime.utcnow()
            }
    
    async def run_single_collection_cycle(self):
        """Run a single collection cycle for testing"""
        try:
            logger.info("🔄 Running single collection cycle...")
            
            # Collect market intelligence
            async with self.market_intelligence_collector:
                intelligence_data = await self.market_intelligence_collector.collect_market_intelligence()
                await self.market_intelligence_collector.store_market_intelligence(intelligence_data)
            
            # Collect volume analysis for one symbol
            symbol = 'BTC/USDT'
            volume_analysis = await self.volume_analyzer.analyze_volume_positioning(symbol, '1h')
            await self.volume_analyzer.store_volume_analysis(volume_analysis, '1h')
            
            logger.info("✅ Single collection cycle completed")
            
            return {
                'market_intelligence': intelligence_data,
                'volume_analysis': volume_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Error in single collection cycle: {e}")
            return None
    
    async def get_collection_status(self) -> Dict[str, Any]:
        """Get status of data collection processes"""
        try:
            # Check if processes are running
            status = {
                'is_running': self.is_running,
                'active_tasks': len(self.collection_tasks),
                'symbols': self.symbols,
                'intervals': {
                    'market_intelligence': self.market_intelligence_interval,
                    'volume_analysis': self.volume_analysis_interval
                },
                'last_update': datetime.utcnow()
            }
            
            # Get latest data timestamps
            async with self.db_pool.acquire() as conn:
                # Latest market intelligence
                mi_row = await conn.fetchrow("""
                    SELECT timestamp FROM market_intelligence 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                if mi_row:
                    status['last_market_intelligence'] = mi_row['timestamp']
                
                # Latest volume analysis
                va_row = await conn.fetchrow("""
                    SELECT symbol, timestamp FROM volume_analysis 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                if va_row:
                    status['last_volume_analysis'] = {
                        'symbol': va_row['symbol'],
                        'timestamp': va_row['timestamp']
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting collection status: {e}")
            return {
                'is_running': self.is_running,
                'error': str(e),
                'last_update': datetime.utcnow()
            }

# Example usage
async def main():
    """Example usage of Enhanced Data Collection Manager"""
    # Initialize database pool
    db_pool = await asyncpg.create_pool(
        host='postgres',
        port=5432,
        database='alphapulse',
        user='alpha_emon',
        password='Emon_@17711',
        min_size=5,
        max_size=20
    )
    
    # Initialize exchange
    exchange = ccxt.binance({
        'sandbox': False,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    })
    
    # Create manager
    manager = EnhancedDataCollectionManager(db_pool, exchange)
    
    try:
        # Run a single collection cycle
        result = await manager.run_single_collection_cycle()
        
        if result:
            print("Collection Results:")
            print(f"BTC Dominance: {result['market_intelligence'].btc_dominance}%")
            print(f"Market Regime: {result['market_intelligence'].market_regime}")
            print(f"Volume Analysis: {result['volume_analysis'].volume_analysis}")
        
        # Get collection status
        status = await manager.get_collection_status()
        print(f"Collection Status: {status}")
        
    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())
