"""
Market Intelligence Collector for AlphaPulse
Collects comprehensive market data including BTC dominance, Total2/Total3, sentiment, and market regime
"""

import asyncio
import logging
import aiohttp
import asyncpg
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

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
    social_sentiment_score: float
    volume_positioning_score: float
    fear_greed_index: int
    market_regime: str
    volatility_index: float
    trend_strength: float

class MarketIntelligenceCollector:
    """
    Collects market intelligence data from various sources
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.session = None
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
    
    async def collect_market_intelligence(self) -> MarketIntelligenceData:
        """Collect comprehensive market intelligence data"""
        try:
            logger.info("🔄 Collecting market intelligence data...")
            
            # Initialize session if not exists
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Collect BTC dominance and Total2/Total3
            btc_data = await self._get_btc_dominance_and_total()
            
            # Collect Fear & Greed Index
            fear_greed = await self._get_fear_greed_index()
            
            # Collect News Sentiment
            news_sentiment = await self._get_news_sentiment()
            
            # Collect Social Sentiment
            social_sentiment = await self._get_social_sentiment()
            
            # Calculate market sentiment
            market_sentiment = await self._calculate_market_sentiment(btc_data, fear_greed)
            
            # Determine market regime
            market_regime = await self._determine_market_regime(btc_data, fear_greed)
            
            # Calculate volatility index
            volatility_index = await self._calculate_volatility_index(btc_data)
            
            # Calculate trend strength
            trend_strength = await self._calculate_trend_strength(btc_data)
            
            # Create market intelligence data
            intelligence_data = MarketIntelligenceData(
                timestamp=datetime.utcnow(),
                btc_dominance=btc_data.get('btc_dominance', 0.0),
                total2_value=btc_data.get('total2', 0.0),
                total3_value=btc_data.get('total3', 0.0),
                market_sentiment_score=market_sentiment,
                news_sentiment_score=news_sentiment,
                social_sentiment_score=social_sentiment,
                volume_positioning_score=0.5,  # Placeholder for volume positioning
                fear_greed_index=fear_greed.get('value', 50),
                market_regime=market_regime,
                volatility_index=volatility_index,
                trend_strength=trend_strength
            )
            
            logger.info("✅ Market intelligence data collected successfully")
            return intelligence_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting market intelligence: {e}")
            # Return default data on error
            return MarketIntelligenceData(
                timestamp=datetime.utcnow(),
                btc_dominance=45.0,
                total2_value=1000000000.0,
                total3_value=8000000000.0,
                market_sentiment_score=0.5,
                news_sentiment_score=0.5,
                volume_positioning_score=0.5,
                fear_greed_index=50,
                market_regime='sideways',
                volatility_index=0.025,
                trend_strength=0.5
            )
    
    async def _get_btc_dominance_and_total(self) -> Dict[str, float]:
        """Get BTC dominance and Total2/Total3 values from CoinGecko"""
        try:
            cache_key = 'btc_data'
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_duration:
                    return cached_data['data']
            
            # Use CoinGecko API to get market data
            url = "https://api.coingecko.com/api/v3/global"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data.get('data', {})
                    
                    # Get BTC dominance with proper validation
                    btc_dominance_raw = market_data.get('market_cap_percentage', {}).get('btc')
                    btc_dominance = float(btc_dominance_raw) if btc_dominance_raw is not None else 45.0
                    
                    # Get total market cap with proper validation
                    total_market_cap_raw = market_data.get('total_market_cap', {}).get('usd')
                    total_market_cap = float(total_market_cap_raw) if total_market_cap_raw is not None else 1000000000000
                    
                    # Calculate Total2 (total market cap excluding BTC)
                    total2 = total_market_cap * (1 - btc_dominance / 100)
                    
                    # Calculate Total3 (total market cap excluding BTC and ETH)
                    eth_dominance = market_data.get('market_cap_percentage', {}).get('eth', 15.0)
                    total3 = total_market_cap * (1 - btc_dominance / 100 - eth_dominance / 100)
                    
                    result = {
                        'btc_dominance': btc_dominance,
                        'total2': total2,
                        'total3': total3
                    }
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': result
                    }
                    
                    return result
                else:
                    logger.warning(f"Failed to get BTC data: {response.status}")
                    return {'btc_dominance': 45.0, 'total2': 1000000000000, 'total3': 8000000000000}
                    
        except Exception as e:
            logger.error(f"Error getting BTC dominance and total: {e}")
            return {'btc_dominance': 45.0, 'total2': 1000000000000, 'total3': 8000000000000}
    
    async def _get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index from alternative.me"""
        try:
            cache_key = 'fear_greed'
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_duration:
                    return cached_data['data']
            
            url = "https://api.alternative.me/fng/"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    fng_data = data.get('data', [{}])[0]
                    
                    # Get Fear & Greed value with proper validation
                    fng_value_raw = fng_data.get('value')
                    fng_value = int(fng_value_raw) if fng_value_raw is not None else 50
                    
                    result = {
                        'value': fng_value,
                        'classification': fng_data.get('value_classification', 'Neutral')
                    }
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': result
                    }
                    
                    return result
                else:
                    logger.warning(f"Failed to get Fear & Greed Index: {response.status}")
                    return {'value': 50, 'classification': 'Neutral'}
                    
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {e}")
            return {'value': 50, 'classification': 'Neutral'}
    
    async def _get_news_sentiment(self) -> float:
        """Get news sentiment score from crypto news sources"""
        try:
            cache_key = 'news_sentiment'
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_duration:
                    return cached_data['data']
            
            # Use CryptoCompare News API (free tier)
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    news_data = data.get('Data', [])
                    
                    if not news_data:
                        return 0.5
                    
                    # Analyze recent news sentiment (last 10 articles)
                    recent_news = news_data[:10]
                    sentiment_scores = []
                    
                    for article in recent_news:
                        # Simple keyword-based sentiment analysis
                        title = article.get('title', '').lower()
                        body = article.get('body', '').lower()
                        text = f"{title} {body}"
                        
                        # Positive keywords
                        positive_words = ['bullish', 'surge', 'rally', 'gain', 'up', 'positive', 'growth', 'adoption']
                        # Negative keywords  
                        negative_words = ['bearish', 'crash', 'drop', 'fall', 'down', 'negative', 'decline', 'sell']
                        
                        positive_count = sum(1 for word in positive_words if word in text)
                        negative_count = sum(1 for word in negative_words if word in text)
                        
                        if positive_count > negative_count:
                            sentiment_scores.append(0.7)  # Positive
                        elif negative_count > positive_count:
                            sentiment_scores.append(0.3)  # Negative
                        else:
                            sentiment_scores.append(0.5)  # Neutral
                    
                    # Calculate average sentiment
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
                    
                    # Cache the result
                    self.cache[cache_key] = {
                        'timestamp': datetime.now(),
                        'data': avg_sentiment
                    }
                    
                    return avg_sentiment
                else:
                    logger.warning(f"Failed to get news sentiment: {response.status}")
                    return 0.5
                    
        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return 0.5
    
    async def _get_social_sentiment(self) -> float:
        """Get social sentiment score from Reddit and Twitter"""
        try:
            cache_key = 'social_sentiment'
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if (datetime.now() - cached_data['timestamp']).seconds < self.cache_duration:
                    return cached_data['data']
            
            # Use Reddit API for crypto sentiment (r/cryptocurrency, r/bitcoin)
            reddit_sentiment = await self._get_reddit_sentiment()
            
            # Use Twitter-like sentiment (simulated for now)
            twitter_sentiment = await self._get_twitter_sentiment()
            
            # Combine social sentiment scores
            social_sentiment = (reddit_sentiment * 0.6) + (twitter_sentiment * 0.4)
            
            # Cache the result
            self.cache[cache_key] = {
                'timestamp': datetime.now(),
                'data': social_sentiment
            }
            
            return social_sentiment
            
        except Exception as e:
            logger.error(f"Error getting social sentiment: {e}")
            return 0.5
    
    async def _get_reddit_sentiment(self) -> float:
        """Get Reddit sentiment for crypto communities"""
        try:
            # Use Reddit JSON API (no authentication required for public data)
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum']
            sentiment_scores = []
            
            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=10"
                async with self.session.get(url, headers={'User-Agent': 'AlphaPulse/1.0'}) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])
                        
                        for post in posts:
                            post_data = post.get('data', {})
                            title = post_data.get('title', '').lower()
                            
                            # Simple sentiment analysis based on title keywords
                            positive_words = ['bullish', 'moon', 'pump', 'surge', 'rally', 'adoption', 'partnership']
                            negative_words = ['bearish', 'dump', 'crash', 'sell', 'fud', 'scam', 'regulation']
                            
                            positive_count = sum(1 for word in positive_words if word in title)
                            negative_count = sum(1 for word in negative_words if word in title)
                            
                            if positive_count > negative_count:
                                sentiment_scores.append(0.7)
                            elif negative_count > positive_count:
                                sentiment_scores.append(0.3)
                            else:
                                sentiment_scores.append(0.5)
            
            return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return 0.5
    
    async def _get_twitter_sentiment(self) -> float:
        """Get Twitter sentiment (simulated for now)"""
        try:
            # Simulate Twitter sentiment based on market conditions
            # In a real implementation, you would use Twitter API
            import random
            
            # Simulate sentiment based on current market conditions
            base_sentiment = 0.5
            
            # Add some randomness to simulate real sentiment
            variation = random.uniform(-0.2, 0.2)
            
            return max(0, min(1, base_sentiment + variation))
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return 0.5
    
    async def _calculate_market_sentiment(self, btc_data: Dict[str, float], fear_greed: Dict[str, Any]) -> float:
        """Calculate market sentiment score (0-1)"""
        try:
            # Normalize BTC dominance (40-60% range)
            btc_dominance = btc_data.get('btc_dominance', 45.0)
            btc_sentiment = max(0, min(1, (btc_dominance - 40) / 20))
            
            # Normalize Fear & Greed Index (0-100 range)
            fng_value = fear_greed.get('value', 50)
            fng_sentiment = fng_value / 100
            
            # Combine sentiment scores (weighted average)
            sentiment_score = (btc_sentiment * 0.6) + (fng_sentiment * 0.4)
            
            return max(0, min(1, sentiment_score))
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return 0.5
    
    async def _determine_market_regime(self, btc_data: Dict[str, float], fear_greed: Dict[str, Any]) -> str:
        """Determine current market regime"""
        try:
            btc_dominance = btc_data.get('btc_dominance', 45.0)
            fng_value = fear_greed.get('value', 50)
            
            # High BTC dominance + high fear = bearish
            if btc_dominance > 55 and fng_value < 30:
                return 'bearish'
            # Low BTC dominance + high greed = bullish
            elif btc_dominance < 45 and fng_value > 70:
                return 'bullish'
            # Otherwise sideways
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return 'sideways'
    
    async def _calculate_volatility_index(self, btc_data: Dict[str, float]) -> float:
        """Calculate volatility index based on market data"""
        try:
            # Simple volatility calculation based on BTC dominance changes
            # In a real implementation, this would use historical price data
            btc_dominance = btc_data.get('btc_dominance', 45.0)
            
            # Normalize to 0-1 range (assuming 40-60% is normal range)
            volatility = abs(btc_dominance - 50) / 10
            
            return max(0, min(1, volatility))
            
        except Exception as e:
            logger.error(f"Error calculating volatility index: {e}")
            return 0.025
    
    async def _calculate_trend_strength(self, btc_data: Dict[str, float]) -> float:
        """Calculate trend strength based on market data"""
        try:
            # Simple trend strength calculation
            # In a real implementation, this would use technical indicators
            btc_dominance = btc_data.get('btc_dominance', 45.0)
            
            # Normalize to 0-1 range
            trend_strength = abs(btc_dominance - 50) / 50
            
            return max(0, min(1, trend_strength))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
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
            
            logger.info(f"✅ Market intelligence data stored: {data.timestamp}")
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
                        btc_dominance=float(row['btc_dominance']) if row['btc_dominance'] is not None else 45.0,
                        total2_value=float(row['total2_value']) if row['total2_value'] is not None else 1000000000000,
                        total3_value=float(row['total3_value']) if row['total3_value'] is not None else 8000000000000,
                        market_sentiment_score=float(row['market_sentiment_score']) if row['market_sentiment_score'] is not None else 0.5,
                        news_sentiment_score=float(row['news_sentiment_score']) if row['news_sentiment_score'] is not None else 0.5,
                        volume_positioning_score=float(row['volume_positioning_score']) if row['volume_positioning_score'] is not None else 0.5,
                        fear_greed_index=int(row['fear_greed_index']) if row['fear_greed_index'] is not None else 50,
                        market_regime=row['market_regime'] if row['market_regime'] is not None else 'sideways',
                        volatility_index=float(row['volatility_index']) if row['volatility_index'] is not None else 0.025,
                        trend_strength=float(row['trend_strength']) if row['trend_strength'] is not None else 0.5
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting latest market intelligence: {e}")
            return None
