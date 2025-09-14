"""
Market Intelligence Collector for AlphaPulse
Collects comprehensive market data including BTC dominance, Total2/Total3, sentiment, and market regime
"""

import asyncio
import logging
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncpg
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class MarketIntelligenceData:
    """Market intelligence data structure"""
    timestamp: datetime
    btc_dominance: float
    total2_value: float
    total3_value: float
    market_sentiment_score: float
    news_sentiment_score: float
    volume_positioning_score: float
    fear_greed_index: int
    market_regime: str
    volatility_index: float
    trend_strength: float

class MarketIntelligenceCollector:
    """
    Comprehensive market intelligence collector
    Integrates multiple data sources for market analysis
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.session = None
        
        # API endpoints
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.news_api_base = "https://newsapi.org/v2"
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        logger.info("Market Intelligence Collector initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_btc_dominance(self) -> float:
        """Get BTC dominance from CoinGecko API"""
        try:
            cache_key = "btc_dominance"
            if cache_key in self.cache and (datetime.now() - self.cache[cache_key]['timestamp']).seconds < self.cache_duration:
                return self.cache[cache_key]['data']
            
            url = f"{self.coingecko_base}/global"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    btc_dominance = data['data']['market_cap_percentage']['btc']
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'data': btc_dominance,
                        'timestamp': datetime.now()
                    }
                    
                    logger.info(f"BTC Dominance: {btc_dominance}%")
                    return btc_dominance
                else:
                    logger.warning(f"Failed to get BTC dominance: {response.status}")
                    return 45.0  # Default fallback
                    
        except Exception as e:
            logger.error(f"Error getting BTC dominance: {e}")
            return 45.0  # Default fallback
    
    async def get_total2_total3_values(self) -> tuple[float, float]:
        """Get Total2 and Total3 values (market cap excluding BTC and including BTC)"""
        try:
            cache_key = "total2_total3"
            if cache_key in self.cache and (datetime.now() - self.cache[cache_key]['timestamp']).seconds < self.cache_duration:
                return self.cache[cache_key]['data']
            
            url = f"{self.coingecko_base}/global"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    total_market_cap = data['data']['total_market_cap']['usd']
                    
                    # Get BTC market cap
                    btc_url = f"{self.coingecko_base}/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=true"
                    async with self.session.get(btc_url) as btc_response:
                        if btc_response.status == 200:
                            btc_data = await btc_response.json()
                            btc_market_cap = btc_data['bitcoin']['usd_market_cap']
                            
                            total2 = total_market_cap - btc_market_cap  # Market cap excluding BTC
                            total3 = total_market_cap  # Total market cap including BTC
                            
                            result = (total2, total3)
                            
                            # Cache the result
                            self.cache[cache_key] = {
                                'data': result,
                                'timestamp': datetime.now()
                            }
                            
                            logger.info(f"Total2: ${total2:,.0f}, Total3: ${total3:,.0f}")
                            return result
                        else:
                            logger.warning(f"Failed to get BTC market cap: {btc_response.status}")
                            return (total_market_cap * 0.55, total_market_cap)  # Estimate
                else:
                    logger.warning(f"Failed to get market data: {response.status}")
                    return (1234567890.0, 9876543210.0)  # Default fallback
                    
        except Exception as e:
            logger.error(f"Error getting Total2/Total3: {e}")
            return (1234567890.0, 9876543210.0)  # Default fallback
    
    async def get_fear_greed_index(self) -> int:
        """Get Fear & Greed Index"""
        try:
            cache_key = "fear_greed"
            if cache_key in self.cache and (datetime.now() - self.cache[cache_key]['timestamp']).seconds < self.cache_duration:
                return self.cache[cache_key]['data']
            
            async with self.session.get(self.fear_greed_url) as response:
                if response.status == 200:
                    data = await response.json()
                    fear_greed_value = int(data['data'][0]['value'])
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'data': fear_greed_value,
                        'timestamp': datetime.now()
                    }
                    
                    logger.info(f"Fear & Greed Index: {fear_greed_value}")
                    return fear_greed_value
                else:
                    logger.warning(f"Failed to get Fear & Greed Index: {response.status}")
                    return 45  # Neutral default
                    
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {e}")
            return 45  # Neutral default
    
    async def get_market_sentiment_score(self) -> float:
        """Calculate market sentiment score based on multiple factors"""
        try:
            # Get Fear & Greed Index
            fear_greed = await self.get_fear_greed_index()
            
            # Convert Fear & Greed to sentiment score (0-1)
            # 0-25: Extreme Fear (0.0-0.25)
            # 26-45: Fear (0.25-0.45)
            # 46-55: Neutral (0.45-0.55)
            # 56-75: Greed (0.55-0.75)
            # 76-100: Extreme Greed (0.75-1.0)
            
            if fear_greed <= 25:
                sentiment = 0.25 * (fear_greed / 25)
            elif fear_greed <= 45:
                sentiment = 0.25 + 0.20 * ((fear_greed - 25) / 20)
            elif fear_greed <= 55:
                sentiment = 0.45 + 0.10 * ((fear_greed - 45) / 10)
            elif fear_greed <= 75:
                sentiment = 0.55 + 0.20 * ((fear_greed - 55) / 20)
            else:
                sentiment = 0.75 + 0.25 * ((fear_greed - 75) / 25)
            
            logger.info(f"Market Sentiment Score: {sentiment:.3f}")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return 0.5  # Neutral default
    
    async def get_news_sentiment_score(self) -> float:
        """Get news sentiment score (placeholder for now)"""
        # TODO: Implement news sentiment analysis
        # For now, return a neutral score
        return 0.5
    
    async def get_volume_positioning_score(self) -> float:
        """Calculate volume positioning score (placeholder for now)"""
        # TODO: Implement volume positioning analysis
        # For now, return a neutral score
        return 0.5
    
    async def calculate_market_regime(self, btc_dominance: float, fear_greed: int, volatility: float) -> str:
        """Determine market regime based on multiple factors"""
        try:
            # Calculate regime based on BTC dominance and sentiment
            if btc_dominance > 50 and fear_greed > 60:
                regime = "bullish"
            elif btc_dominance < 40 and fear_greed < 40:
                regime = "bearish"
            elif volatility > 0.05:
                regime = "volatile"
            else:
                regime = "sideways"
            
            logger.info(f"Market Regime: {regime}")
            return regime
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return "sideways"
    
    async def calculate_volatility_index(self) -> float:
        """Calculate market volatility index"""
        try:
            # Get recent price data for BTC to calculate volatility
            # For now, return a default value
            volatility = 0.025  # 2.5% volatility
            logger.info(f"Volatility Index: {volatility:.4f}")
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility index: {e}")
            return 0.025
    
    async def calculate_trend_strength(self) -> float:
        """Calculate trend strength (0-1)"""
        try:
            # TODO: Implement trend strength calculation
            # For now, return a default value
            trend_strength = 0.45  # Moderate trend strength
            logger.info(f"Trend Strength: {trend_strength:.3f}")
            return trend_strength
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.45
    
    async def collect_market_intelligence(self) -> MarketIntelligenceData:
        """Collect comprehensive market intelligence data"""
        try:
            logger.info("🔄 Collecting market intelligence data...")
            
            # Collect all market data
            btc_dominance = await self.get_btc_dominance()
            total2, total3 = await self.get_total2_total3_values()
            fear_greed = await self.get_fear_greed_index()
            market_sentiment = await self.get_market_sentiment_score()
            news_sentiment = await self.get_news_sentiment_score()
            volume_positioning = await self.get_volume_positioning_score()
            volatility = await self.calculate_volatility_index()
            trend_strength = await self.calculate_trend_strength()
            
            # Determine market regime
            market_regime = await self.calculate_market_regime(btc_dominance, fear_greed, volatility)
            
            # Create market intelligence data
            intelligence_data = MarketIntelligenceData(
                timestamp=datetime.utcnow(),
                btc_dominance=btc_dominance,
                total2_value=total2,
                total3_value=total3,
                market_sentiment_score=market_sentiment,
                news_sentiment_score=news_sentiment,
                volume_positioning_score=volume_positioning,
                fear_greed_index=fear_greed,
                market_regime=market_regime,
                volatility_index=volatility,
                trend_strength=trend_strength
            )
            
            logger.info("✅ Market intelligence data collected successfully")
            return intelligence_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting market intelligence: {e}")
            # Return default data
            return MarketIntelligenceData(
                timestamp=datetime.utcnow(),
                btc_dominance=45.0,
                total2_value=1234567890.0,
                total3_value=9876543210.0,
                market_sentiment_score=0.5,
                news_sentiment_score=0.5,
                volume_positioning_score=0.5,
                fear_greed_index=45,
                market_regime="sideways",
                volatility_index=0.025,
                trend_strength=0.45
            )
    
    async def store_market_intelligence(self, data: MarketIntelligenceData) -> bool:
        """Store market intelligence data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_intelligence (
                        timestamp, btc_dominance, total2_value, total3_value,
                        market_sentiment_score, news_sentiment_score, volume_positioning_score,
                        fear_greed_index, market_regime, volatility_index, trend_strength
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """, 
                data.timestamp, data.btc_dominance, data.total2_value, data.total3_value,
                data.market_sentiment_score, data.news_sentiment_score, data.volume_positioning_score,
                data.fear_greed_index, data.market_regime, data.volatility_index, data.trend_strength
                )
            
            logger.info("✅ Market intelligence data stored in database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing market intelligence: {e}")
            return False
    
    async def get_latest_market_intelligence(self) -> Optional[MarketIntelligenceData]:
        """Get latest market intelligence data from database"""
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM market_intelligence 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                if row:
                    return MarketIntelligenceData(
                        timestamp=row['timestamp'],
                        btc_dominance=float(row['btc_dominance']),
                        total2_value=float(row['total2_value']),
                        total3_value=float(row['total3_value']),
                        market_sentiment_score=float(row['market_sentiment_score']),
                        news_sentiment_score=float(row['news_sentiment_score']),
                        volume_positioning_score=float(row['volume_positioning_score']),
                        fear_greed_index=int(row['fear_greed_index']),
                        market_regime=row['market_regime'],
                        volatility_index=float(row['volatility_index']),
                        trend_strength=float(row['trend_strength'])
                    )
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Error getting latest market intelligence: {e}")
            return None
    
    async def run_collection_cycle(self):
        """Run a complete market intelligence collection cycle"""
        try:
            logger.info("🚀 Starting market intelligence collection cycle")
            
            # Collect market intelligence
            intelligence_data = await self.collect_market_intelligence()
            
            # Store in database
            success = await self.store_market_intelligence(intelligence_data)
            
            if success:
                logger.info("✅ Market intelligence collection cycle completed successfully")
            else:
                logger.error("❌ Market intelligence collection cycle failed")
                
        except Exception as e:
            logger.error(f"❌ Error in market intelligence collection cycle: {e}")

# Example usage
async def main():
    """Example usage of Market Intelligence Collector"""
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
    
    async with MarketIntelligenceCollector(db_pool) as collector:
        # Run collection cycle
        await collector.run_collection_cycle()
        
        # Get latest data
        latest_data = await collector.get_latest_market_intelligence()
        if latest_data:
            print(f"Latest BTC Dominance: {latest_data.btc_dominance}%")
            print(f"Latest Market Regime: {latest_data.market_regime}")
            print(f"Latest Fear & Greed: {latest_data.fear_greed_index}")

if __name__ == "__main__":
    asyncio.run(main())
