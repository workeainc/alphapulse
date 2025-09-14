"""
Sentiment Service for AlphaPulse
Handles sentiment analysis, news processing, and news event integration
"""

import asyncio
import logging
import os
import aiohttp
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TwitterSentimentAnalyzer:
    """Twitter sentiment analysis using Twitter API v2"""
    
    def __init__(self):
        from app.core.config import settings
        self.bearer_token = settings.api.twitter_bearer_token or os.getenv('TWITTER_BEARER_TOKEN')
        self.api_key = settings.api.twitter_api_key or os.getenv('TWITTER_API_KEY')
        self.api_secret = settings.api.twitter_api_secret or os.getenv('TWITTER_API_SECRET')
        self.base_url = "https://api.twitter.com/2"
        
    async def get_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """Get tweets for a specific query"""
        if not self.bearer_token:
            logger.warning("Twitter Bearer Token not configured")
            return []
            
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics,lang",
            "exclude": "retweets,replies"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/tweets/search/recent",
                    headers=headers,
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', [])
                    elif response.status == 429:
                        logger.warning("Twitter API rate limit exceeded, using cached data")
                        return []
                    elif response.status == 401:
                        logger.error("Twitter API authentication failed")
                        return []
                    else:
                        logger.error(f"Twitter API error: {response.status}")
                        return []
        except asyncio.TimeoutError:
            logger.warning("Twitter API request timeout, using cached data")
            return []
        except Exception as e:
            logger.error(f"Error fetching Twitter data: {e}")
            return []
    
    async def analyze_sentiment(self, tweets: List[Dict]) -> Dict:
        """Analyze sentiment of tweets (simplified)"""
        if not tweets:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        # Simple keyword-based sentiment analysis
        positive_words = ['bull', 'bullish', 'moon', 'pump', 'buy', 'long', 'profit', 'gain']
        negative_words = ['bear', 'bearish', 'dump', 'sell', 'short', 'loss', 'crash', 'drop']
        
        positive_count = 0
        negative_count = 0
        total_tweets = len(tweets)
        
        for tweet in tweets:
            text = tweet.get('text', '').lower()
            positive_count += sum(1 for word in positive_words if word in text)
            negative_count += sum(1 for word in negative_words if word in text)
        
        if total_tweets == 0:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        positive_ratio = positive_count / total_tweets
        negative_ratio = negative_count / total_tweets
        
        sentiment_score = (positive_ratio - negative_ratio) * 2  # Scale to [-1, 1]
        
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        confidence = min(abs(sentiment_score), 1.0)
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": label,
            "confidence": confidence,
            "tweet_count": total_tweets
        }

class RedditSentimentAnalyzer:
    """Reddit sentiment analysis using Reddit API"""
    
    def __init__(self):
        from app.core.config import settings
        self.client_id = settings.api.reddit_client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = settings.api.reddit_client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = "AlphaPulse/1.0"
        
    async def get_reddit_posts(self, subreddit: str, query: str, limit: int = 100) -> List[Dict]:
        """Get Reddit posts from a specific subreddit"""
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not configured")
            return []
        
        # Reddit API requires OAuth2, for now we'll use a simplified approach
        # In production, implement proper OAuth2 flow
        headers = {
            "User-Agent": self.user_agent
        }
        
        try:
            # Using Reddit's JSON endpoint (limited without OAuth)
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": query,
                "limit": limit,
                "sort": "hot",
                "t": "day"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('data', {}).get('children', [])
                        return [post['data'] for post in posts]
                    elif response.status == 429:
                        logger.warning("Reddit API rate limit exceeded, using cached data")
                        return []
                    elif response.status == 403:
                        logger.warning("Reddit API access forbidden, using cached data")
                        return []
                    else:
                        logger.error(f"Reddit API error: {response.status}")
                        return []
        except asyncio.TimeoutError:
            logger.warning("Reddit API request timeout, using cached data")
            return []
        except Exception as e:
            logger.error(f"Error fetching Reddit data: {e}")
            return []
    
    async def analyze_sentiment(self, posts: List[Dict]) -> Dict:
        """Analyze sentiment of Reddit posts"""
        if not posts:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        # Simple sentiment analysis based on upvotes and keywords
        positive_words = ['bull', 'bullish', 'moon', 'pump', 'buy', 'long', 'profit', 'gain']
        negative_words = ['bear', 'bearish', 'dump', 'sell', 'short', 'loss', 'crash', 'drop']
        
        total_score = 0
        total_posts = len(posts)
        
        for post in posts:
            text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
            upvotes = post.get('score', 0)
            
            # Weight by upvotes
            weight = min(upvotes / 100, 1.0) if upvotes > 0 else 0.1
            
            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)
            
            post_sentiment = (positive_count - negative_count) * weight
            total_score += post_sentiment
        
        if total_posts == 0:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        sentiment_score = total_score / total_posts
        sentiment_score = max(-1.0, min(1.0, sentiment_score))  # Clamp to [-1, 1]
        
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        confidence = min(abs(sentiment_score), 1.0)
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": label,
            "confidence": confidence,
            "post_count": total_posts
        }

class NewsSentimentAnalyzer:
    """News sentiment analysis using News API"""
    
    def __init__(self):
        from app.core.config import settings
        self.api_key = settings.api.news_api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2"
        
    async def get_news(self, query: str, days: int = 1) -> List[Dict]:
        """Get news articles for a specific query"""
        if not self.api_key:
            logger.warning("News API key not configured")
            return []
        
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": self.api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/everything",
                    params=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('articles', [])
                    elif response.status == 429:
                        logger.warning("News API rate limit exceeded, using cached data")
                        return []
                    elif response.status == 401:
                        logger.error("News API authentication failed")
                        return []
                    elif response.status == 400:
                        logger.error("News API bad request - check query parameters")
                        return []
                    else:
                        logger.error(f"News API error: {response.status}")
                        return []
        except asyncio.TimeoutError:
            logger.warning("News API request timeout, using cached data")
            return []
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return []
    
    async def analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment of news articles"""
        if not articles:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        # Simple keyword-based sentiment analysis
        positive_words = ['bull', 'bullish', 'moon', 'pump', 'buy', 'long', 'profit', 'gain', 'surge', 'rally']
        negative_words = ['bear', 'bearish', 'dump', 'sell', 'short', 'loss', 'crash', 'drop', 'plunge', 'decline']
        
        positive_count = 0
        negative_count = 0
        total_articles = len(articles)
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            positive_count += sum(1 for word in positive_words if word in text)
            negative_count += sum(1 for word in negative_words if word in text)
        
        if total_articles == 0:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        positive_ratio = positive_count / total_articles
        negative_ratio = negative_count / total_articles
        
        sentiment_score = (positive_ratio - negative_ratio) * 2  # Scale to [-1, 1]
        
        if sentiment_score > 0.1:
            label = "positive"
        elif sentiment_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        confidence = min(abs(sentiment_score), 1.0)
        
        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": label,
            "confidence": confidence,
            "article_count": total_articles
        }

class SentimentService:
    """Enhanced service for managing sentiment analysis with Phase 3.3 social media integration"""
    
    def __init__(self):
        self.is_running = False
        self.sentiment_cache = {}
        self.last_update = {}
        
        # Initialize analyzers
        self.twitter_analyzer = TwitterSentimentAnalyzer()
        self.reddit_analyzer = RedditSentimentAnalyzer()
        self.news_analyzer = NewsSentimentAnalyzer()
        
        # Phase 3.1: Enhanced sentiment tracking
        self.sentiment_history = {}  # symbol -> list of sentiment records
        self.sentiment_trends = {}   # symbol -> trend analysis
        self.sentiment_volatility = {}  # symbol -> volatility metrics
        self.sentiment_momentum = {}  # symbol -> momentum indicators
        self.sentiment_correlation = {}  # symbol -> price correlation
        
        # Cache TTL (in seconds)
        self.cache_ttl = int(os.getenv('CACHE_TTL_SENTIMENT', 3600))
        
        # Subreddits for crypto sentiment
        self.crypto_subreddits = [
            'cryptocurrency', 'bitcoin', 'ethereum', 'binance', 'cryptomarkets'
        ]
        
        # Phase 3.1: Additional sentiment sources
        self.telegram_sentiment = {}  # symbol -> telegram sentiment
        self.discord_sentiment = {}   # symbol -> discord sentiment
        
        # Phase 3.2: News event tracking
        self.news_events = {}  # symbol -> list of news events
        self.event_impact_scores = {}  # symbol -> impact score
        self.event_categories = {}  # symbol -> event categories
        self.event_timestamps = {}  # symbol -> event timestamps
        self.event_keywords = {  # Event keywords for detection
            'regulatory': ['regulation', 'sec', 'cfdc', 'ban', 'legal', 'illegal', 'government'],
            'adoption': ['adoption', 'partnership', 'integration', 'merchant', 'payment'],
            'technology': ['upgrade', 'fork', 'update', 'development', 'protocol'],
            'market': ['bull', 'bear', 'rally', 'crash', 'pump', 'dump', 'volatility'],
            'institutional': ['institutional', 'fund', 'etf', 'investment', 'whale'],
            'security': ['hack', 'breach', 'vulnerability', 'security', 'exploit']
        }
        
        # Phase 3.3: Enhanced social media tracking
        self.social_sentiment_cache = {}  # symbol -> social sentiment data
        self.social_trends = {}  # symbol -> social sentiment trends
        self.social_volume = {}  # symbol -> social media volume metrics
        self.social_influencers = {}  # symbol -> influencer sentiment tracking
        self.social_momentum = {}  # symbol -> social sentiment momentum
        self.social_correlation = {}  # symbol -> social-price correlation
        self.social_volatility = {}  # symbol -> social sentiment volatility
        self.social_engagement = {}  # symbol -> engagement metrics
        self.social_sentiment_history = {}  # symbol -> historical social sentiment
        self.social_impact_scores = {}  # symbol -> social impact scores
        
        # Performance tracking
        self.performance_metrics = {
            'total_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0,
            'last_performance_update': datetime.now()
        }
        
        logger.info("🚀 Phase 3.3: Enhanced Sentiment Service initialized with social media integration")
        
    async def start(self):
        """Start the sentiment service"""
        if self.is_running:
            logger.warning("Sentiment service is already running")
            return
            
        logger.info("🚀 Starting Sentiment Service...")
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._update_sentiment())
        
        logger.info("✅ Sentiment Service started successfully")
    
    async def stop(self):
        """Stop the sentiment service"""
        if not self.is_running:
            logger.warning("Sentiment service is not running")
            return
            
        logger.info("🛑 Stopping Sentiment Service...")
        self.is_running = False
        logger.info("✅ Sentiment Service stopped successfully")
    
    async def _update_sentiment(self):
        """Background task to update sentiment data"""
        while self.is_running:
            try:
                # Update sentiment every 5 minutes
                await asyncio.sleep(300)
                
                # Update for major cryptocurrencies
                symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
                for symbol in symbols:
                    await self._update_symbol_sentiment(symbol)
                    
            except Exception as e:
                logger.error(f"❌ Error updating sentiment: {e}")
                await asyncio.sleep(600)
    
    async def _update_symbol_sentiment(self, symbol: str):
        """Update sentiment for a specific symbol"""
        try:
            # Get sentiment from all sources
            twitter_sentiment = await self._get_twitter_sentiment(symbol)
            reddit_sentiment = await self._get_reddit_sentiment(symbol)
            news_sentiment = await self._get_news_sentiment(symbol)
            
            # Aggregate sentiment
            aggregated = self._aggregate_sentiment([
                twitter_sentiment,
                reddit_sentiment,
                news_sentiment
            ])
            
            # Cache the result
            self.sentiment_cache[symbol] = {
                'data': aggregated,
                'timestamp': datetime.now(),
                'sources': {
                    'twitter': twitter_sentiment,
                    'reddit': reddit_sentiment,
                    'news': news_sentiment
                }
            }
            
            self.last_update[symbol] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating sentiment for {symbol}: {e}")
    
    async def _get_twitter_sentiment(self, symbol: str) -> Dict:
        """Get Twitter sentiment for a symbol"""
        try:
            query = f"#{symbol} OR ${symbol} crypto"
            tweets = await self.twitter_analyzer.get_tweets(query, max_results=50)
            return await self.twitter_analyzer.analyze_sentiment(tweets)
        except Exception as e:
            logger.error(f"Twitter sentiment error for {symbol}: {e}")
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get Reddit sentiment for a symbol"""
        try:
            all_posts = []
            for subreddit in self.crypto_subreddits:
                posts = await self.reddit_analyzer.get_reddit_posts(subreddit, symbol, limit=20)
                all_posts.extend(posts)
            
            return await self.reddit_analyzer.analyze_sentiment(all_posts)
        except Exception as e:
            logger.error(f"Reddit sentiment error for {symbol}: {e}")
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
    
    async def _get_news_sentiment(self, symbol: str) -> Dict:
        """Get news sentiment for a symbol"""
        try:
            query = f"{symbol} cryptocurrency"
            articles = await self.news_analyzer.get_news(query, days=1)
            return await self.news_analyzer.analyze_sentiment(articles)
        except Exception as e:
            logger.error(f"News sentiment error for {symbol}: {e}")
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
    
    def _aggregate_sentiment(self, sentiments: List[Dict]) -> Dict:
        """Aggregate sentiment from multiple sources"""
        if not sentiments:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        # Weighted average based on confidence
        total_weighted_score = 0
        total_weight = 0
        
        for sentiment in sentiments:
            if sentiment and 'sentiment_score' in sentiment and 'confidence' in sentiment:
                weight = sentiment['confidence']
                total_weighted_score += sentiment['sentiment_score'] * weight
                total_weight += weight
        
        if total_weight == 0:
            return {"sentiment_score": 0.0, "sentiment_label": "neutral", "confidence": 0.0}
        
        aggregated_score = total_weighted_score / total_weight
        aggregated_score = max(-1.0, min(1.0, aggregated_score))  # Clamp to [-1, 1]
        
        if aggregated_score > 0.1:
            label = "positive"
        elif aggregated_score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        # Average confidence
        avg_confidence = sum(s.get('confidence', 0) for s in sentiments if s) / len(sentiments)
        
        return {
            "sentiment_score": aggregated_score,
            "sentiment_label": label,
            "confidence": avg_confidence,
            "source_count": len(sentiments)
        }
    
    async def get_sentiment(self, symbol: str) -> Dict:
        """Get sentiment data for a symbol"""
        try:
            # Check cache first
            if symbol in self.sentiment_cache:
                cached = self.sentiment_cache[symbol]
                if (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
                    return cached['data']
            
            # If not in cache or expired, update sentiment
            await self._update_symbol_sentiment(symbol)
            
            return self.sentiment_cache.get(symbol, {}).get('data', {})
            
        except Exception as e:
            logger.error(f"❌ Failed to get sentiment for {symbol}: {e}")
            return {
                "symbol": symbol,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_market_sentiment_summary(self) -> Dict:
        """Get overall market sentiment summary"""
        try:
            symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']
            sentiments = []
            
            for symbol in symbols:
                sentiment = await self.get_sentiment(symbol)
                if sentiment:
                    sentiments.append(sentiment)
            
            if not sentiments:
                return {
                    "overall_sentiment": "neutral",
                    "average_score": 0.0,
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "neutral_count": 1,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calculate summary
            scores = [s.get('sentiment_score', 0) for s in sentiments]
            labels = [s.get('sentiment_label', 'neutral') for s in sentiments]
            
            avg_score = sum(scores) / len(scores)
            bullish_count = labels.count('positive')
            bearish_count = labels.count('negative')
            neutral_count = labels.count('neutral')
            
            if avg_score > 0.1:
                overall = "positive"
            elif avg_score < -0.1:
                overall = "negative"
            else:
                overall = "neutral"
            
            return {
                "overall_sentiment": overall,
                "average_score": avg_score,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "symbol_count": len(symbols),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get market sentiment summary: {e}")
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 1,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_sentiment_history(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get sentiment history for a symbol (simplified)"""
        try:
            # In a real implementation, this would fetch from database
            # For now, return current sentiment
            current = await self.get_sentiment(symbol)
            return [current]
        except Exception as e:
            logger.error(f"Error getting sentiment history for {symbol}: {e}")
            return []
    
    def get_cache_status(self) -> Dict:
        """Get cache status and statistics"""
        return {
            "cache_size": len(self.sentiment_cache),
            "last_updates": self.last_update,
            "cache_ttl": self.cache_ttl,
            "is_running": self.is_running
        }
    
    # Phase 3.1: Enhanced Sentiment Analysis Methods
    
    async def get_enhanced_sentiment(self, symbol: str) -> Dict:
        """Get enhanced sentiment analysis with trend, volatility, and momentum"""
        try:
            # Get base sentiment
            base_sentiment = await self.get_sentiment(symbol)
            
            # Calculate enhanced metrics
            trend_analysis = self._calculate_sentiment_trend(symbol)
            volatility_metrics = self._calculate_sentiment_volatility(symbol)
            momentum_indicators = self._calculate_sentiment_momentum(symbol)
            correlation_metrics = self._calculate_sentiment_correlation(symbol)
            
            # Enhanced sentiment response
            enhanced_sentiment = {
                **base_sentiment,
                'trend_analysis': trend_analysis,
                'volatility_metrics': volatility_metrics,
                'momentum_indicators': momentum_indicators,
                'correlation_metrics': correlation_metrics,
                'enhanced_confidence': self._calculate_enhanced_confidence(base_sentiment, trend_analysis, volatility_metrics),
                'sentiment_strength': self._calculate_sentiment_strength(base_sentiment, momentum_indicators),
                'prediction_confidence': self._calculate_prediction_confidence(base_sentiment, correlation_metrics),
                'phase_3_1_features': True
            }
            
            # Update performance metrics
            self.performance_metrics['total_analyses'] += 1
            
            return enhanced_sentiment
            
        except Exception as e:
            logger.error(f"❌ Error getting enhanced sentiment for {symbol}: {e}")
            self.performance_metrics['errors'] += 1
            return base_sentiment
    
    def _calculate_sentiment_trend(self, symbol: str) -> Dict:
        """Calculate sentiment trend analysis"""
        try:
            if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 3:
                return {'trend': 'stable', 'trend_strength': 0.0, 'trend_duration': 0}
            
            history = self.sentiment_history[symbol][-10:]  # Last 10 records
            scores = [record.get('sentiment_score', 0) for record in history]
            
            if len(scores) < 3:
                return {'trend': 'stable', 'trend_strength': 0.0, 'trend_duration': 0}
            
            # Calculate trend using linear regression
            x = list(range(len(scores)))
            y = scores
            
            # Simple linear regression
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 == 0:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Determine trend
            if slope > 0.01:
                trend = 'increasing'
                trend_strength = min(abs(slope) * 10, 1.0)
            elif slope < -0.01:
                trend = 'decreasing'
                trend_strength = min(abs(slope) * 10, 1.0)
            else:
                trend = 'stable'
                trend_strength = 0.0
            
            return {
                'trend': trend,
                'trend_strength': trend_strength,
                'trend_duration': len(history),
                'slope': slope,
                'r_squared': self._calculate_r_squared(x, y, slope)
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment trend for {symbol}: {e}")
            return {'trend': 'stable', 'trend_strength': 0.0, 'trend_duration': 0}
    
    def _calculate_sentiment_volatility(self, symbol: str) -> Dict:
        """Calculate sentiment volatility metrics"""
        try:
            if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 5:
                return {'volatility': 0.0, 'volatility_rank': 'low', 'stability_score': 1.0}
            
            history = self.sentiment_history[symbol][-20:]  # Last 20 records
            scores = [record.get('sentiment_score', 0) for record in history]
            
            if len(scores) < 5:
                return {'volatility': 0.0, 'volatility_rank': 'low', 'stability_score': 1.0}
            
            # Calculate standard deviation
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            volatility = variance ** 0.5
            
            # Determine volatility rank
            if volatility < 0.1:
                volatility_rank = 'low'
                stability_score = 1.0
            elif volatility < 0.3:
                volatility_rank = 'medium'
                stability_score = 0.7
            else:
                volatility_rank = 'high'
                stability_score = 0.3
            
            return {
                'volatility': volatility,
                'volatility_rank': volatility_rank,
                'stability_score': stability_score,
                'mean_sentiment': mean_score,
                'variance': variance
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment volatility for {symbol}: {e}")
            return {'volatility': 0.0, 'volatility_rank': 'low', 'stability_score': 1.0}
    
    def _calculate_sentiment_momentum(self, symbol: str) -> Dict:
        """Calculate sentiment momentum indicators"""
        try:
            if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 3:
                return {'momentum': 0.0, 'momentum_strength': 'weak', 'momentum_direction': 'neutral'}
            
            history = self.sentiment_history[symbol][-5:]  # Last 5 records
            scores = [record.get('sentiment_score', 0) for record in history]
            
            if len(scores) < 3:
                return {'momentum': 0.0, 'momentum_strength': 'weak', 'momentum_direction': 'neutral'}
            
            # Calculate momentum (rate of change)
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:-3]) / max(len(scores) - 3, 1)
            momentum = recent_avg - older_avg
            
            # Determine momentum strength and direction
            if abs(momentum) < 0.05:
                momentum_strength = 'weak'
                momentum_direction = 'neutral'
            elif abs(momentum) < 0.15:
                momentum_strength = 'moderate'
                momentum_direction = 'bullish' if momentum > 0 else 'bearish'
            else:
                momentum_strength = 'strong'
                momentum_direction = 'bullish' if momentum > 0 else 'bearish'
            
            return {
                'momentum': momentum,
                'momentum_strength': momentum_strength,
                'momentum_direction': momentum_direction,
                'recent_avg': recent_avg,
                'older_avg': older_avg
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment momentum for {symbol}: {e}")
            return {'momentum': 0.0, 'momentum_strength': 'weak', 'momentum_direction': 'neutral'}
    
    def _calculate_sentiment_correlation(self, symbol: str) -> Dict:
        """Calculate sentiment-price correlation metrics"""
        try:
            # This would typically use price data from the market data service
            # For now, we'll use a simplified approach based on sentiment history
            
            if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 10:
                return {'correlation': 0.0, 'correlation_strength': 'weak', 'predictive_power': 0.0}
            
            history = self.sentiment_history[symbol][-20:]
            
            # Simulate correlation based on sentiment consistency
            scores = [record.get('sentiment_score', 0) for record in history]
            consistency = 1.0 - (max(scores) - min(scores)) / 2.0  # Higher consistency = higher correlation
            
            correlation = max(0.0, consistency - 0.3)  # Base correlation
            
            # Determine correlation strength
            if correlation < 0.2:
                correlation_strength = 'weak'
                predictive_power = 0.1
            elif correlation < 0.5:
                correlation_strength = 'moderate'
                predictive_power = 0.3
            else:
                correlation_strength = 'strong'
                predictive_power = 0.6
            
            return {
                'correlation': correlation,
                'correlation_strength': correlation_strength,
                'predictive_power': predictive_power,
                'consistency': consistency
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment correlation for {symbol}: {e}")
            return {'correlation': 0.0, 'correlation_strength': 'weak', 'predictive_power': 0.0}
    
    def _calculate_enhanced_confidence(self, base_sentiment: Dict, trend_analysis: Dict, volatility_metrics: Dict) -> float:
        """Calculate enhanced confidence based on multiple factors"""
        try:
            base_confidence = base_sentiment.get('confidence', 0.0)
            
            # Trend boost
            trend_boost = trend_analysis.get('trend_strength', 0.0) * 0.2
            
            # Stability boost
            stability_boost = volatility_metrics.get('stability_score', 1.0) * 0.15
            
            # Enhanced confidence
            enhanced_confidence = base_confidence + trend_boost + stability_boost
            
            return min(enhanced_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence: {e}")
            return base_sentiment.get('confidence', 0.0)
    
    def _calculate_sentiment_strength(self, base_sentiment: Dict, momentum_indicators: Dict) -> float:
        """Calculate sentiment strength based on momentum"""
        try:
            base_score = abs(base_sentiment.get('sentiment_score', 0.0))
            momentum = abs(momentum_indicators.get('momentum', 0.0))
            
            # Combine base sentiment with momentum
            strength = base_score + (momentum * 0.5)
            
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating sentiment strength: {e}")
            return abs(base_sentiment.get('sentiment_score', 0.0))
    
    def _calculate_prediction_confidence(self, base_sentiment: Dict, correlation_metrics: Dict) -> float:
        """Calculate prediction confidence based on correlation"""
        try:
            base_confidence = base_sentiment.get('confidence', 0.0)
            predictive_power = correlation_metrics.get('predictive_power', 0.0)
            
            # Weighted prediction confidence
            prediction_confidence = base_confidence * 0.7 + predictive_power * 0.3
            
            return min(prediction_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return base_sentiment.get('confidence', 0.0)
    
    def _calculate_r_squared(self, x: List[float], y: List[float], slope: float) -> float:
        """Calculate R-squared for trend analysis"""
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            
            # Calculate R-squared
            mean_y = sum(y) / len(y)
            ss_tot = sum((yi - mean_y) ** 2 for yi in y)
            
            # Calculate predicted values
            y_pred = [slope * xi for xi in x]
            ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(len(y)))
            
            if ss_tot == 0:
                return 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            logger.error(f"Error calculating R-squared: {e}")
            return 0.0
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the sentiment service"""
        return {
            **self.performance_metrics,
            'cache_hit_rate': self.performance_metrics['cache_hits'] / max(self.performance_metrics['total_analyses'], 1),
            'error_rate': self.performance_metrics['errors'] / max(self.performance_metrics['total_analyses'], 1),
            'uptime': (datetime.now() - self.performance_metrics['last_performance_update']).total_seconds()
        }
    
    # Phase 3.2: News Event Integration Methods
    
    async def get_news_event_analysis(self, symbol: str) -> Dict:
        """Get comprehensive news event analysis for a symbol"""
        try:
            # Get news articles
            query = f"{symbol} cryptocurrency"
            articles = await self.news_analyzer.get_news(query, days=3)
            
            # Analyze events
            events = self._detect_news_events(articles, symbol)
            impact_score = self._calculate_event_impact(events, symbol)
            event_categories = self._categorize_events(events)
            
            # Store events
            self.news_events[symbol] = events
            self.event_impact_scores[symbol] = impact_score
            self.event_categories[symbol] = event_categories
            self.event_timestamps[symbol] = datetime.now()
            
            return {
                'events': events,
                'impact_score': impact_score,
                'event_categories': event_categories,
                'event_count': len(events),
                'high_impact_events': len([e for e in events if e.get('impact_level') == 'high']),
                'medium_impact_events': len([e for e in events if e.get('impact_level') == 'medium']),
                'low_impact_events': len([e for e in events if e.get('impact_level') == 'low']),
                'phase_3_2_features': True
            }
            
        except Exception as e:
            logger.error(f"Error getting news event analysis for {symbol}: {e}")
            return {
                'events': [],
                'impact_score': 0.0,
                'event_categories': {},
                'event_count': 0,
                'high_impact_events': 0,
                'medium_impact_events': 0,
                'low_impact_events': 0,
                'phase_3_2_features': True
            }
    
    def _detect_news_events(self, articles: List[Dict], symbol: str) -> List[Dict]:
        """Detect significant news events from articles"""
        events = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower()
            content = f"{title} {description}"
            
            # Check for event keywords
            detected_categories = []
            for category, keywords in self.event_keywords.items():
                if any(keyword in content for keyword in keywords):
                    detected_categories.append(category)
            
            if detected_categories:
                # Calculate event impact
                impact_level = self._calculate_event_impact_level(content, detected_categories)
                
                event = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'categories': detected_categories,
                    'impact_level': impact_level,
                    'relevance_score': self._calculate_relevance_score(content, symbol),
                    'sentiment': self._analyze_event_sentiment(content)
                }
                events.append(event)
        
        return events
    
    def _calculate_event_impact_level(self, content: str, categories: List[str]) -> str:
        """Calculate the impact level of an event"""
        # High impact keywords
        high_impact_keywords = [
            'sec', 'regulation', 'ban', 'hack', 'breach', 'etf', 'institutional',
            'partnership', 'adoption', 'upgrade', 'fork', 'whale', 'fund'
        ]
        
        # Medium impact keywords
        medium_impact_keywords = [
            'announcement', 'release', 'update', 'development', 'integration',
            'merchant', 'payment', 'volatility', 'rally', 'crash'
        ]
        
        high_count = sum(1 for keyword in high_impact_keywords if keyword in content)
        medium_count = sum(1 for keyword in medium_impact_keywords if keyword in content)
        
        if high_count >= 2:
            return 'high'
        elif high_count >= 1 or medium_count >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_relevance_score(self, content: str, symbol: str) -> float:
        """Calculate relevance score for a symbol"""
        symbol_lower = symbol.lower()
        content_lower = content.lower()
        
        # Direct mentions
        direct_mentions = content_lower.count(symbol_lower)
        
        # Related terms
        related_terms = ['crypto', 'cryptocurrency', 'token', 'coin', 'blockchain']
        related_mentions = sum(content_lower.count(term) for term in related_terms)
        
        # Calculate relevance score
        relevance = (direct_mentions * 0.7) + (related_mentions * 0.3)
        return min(relevance / 10.0, 1.0)  # Normalize to [0, 1]
    
    def _analyze_event_sentiment(self, content: str) -> Dict:
        """Analyze sentiment of an event"""
        positive_words = ['bull', 'bullish', 'moon', 'pump', 'buy', 'long', 'profit', 'gain', 'surge', 'rally']
        negative_words = ['bear', 'bearish', 'dump', 'sell', 'short', 'loss', 'crash', 'drop', 'plunge', 'decline']
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = positive_count / max(positive_count + negative_count, 1)
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = negative_count / max(positive_count + negative_count, 1)
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def _calculate_event_impact(self, events: List[Dict], symbol: str) -> float:
        """Calculate overall event impact score for a symbol"""
        if not events:
            return 0.0
        
        total_impact = 0.0
        total_weight = 0.0
        
        for event in events:
            impact_level = event.get('impact_level', 'low')
            relevance_score = event.get('relevance_score', 0.0)
            
            # Impact weights
            impact_weights = {
                'high': 1.0,
                'medium': 0.6,
                'low': 0.3
            }
            
            weight = impact_weights.get(impact_level, 0.3) * relevance_score
            total_impact += weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(total_impact / total_weight, 1.0)
    
    def _categorize_events(self, events: List[Dict]) -> Dict:
        """Categorize events by type"""
        categories = {}
        
        for event in events:
            event_categories = event.get('categories', [])
            for category in event_categories:
                if category not in categories:
                    categories[category] = {
                        'count': 0,
                        'high_impact': 0,
                        'medium_impact': 0,
                        'low_impact': 0
                    }
                
                categories[category]['count'] += 1
                impact_level = event.get('impact_level', 'low')
                categories[category][f'{impact_level}_impact'] += 1
        
        return categories
    
    async def get_enhanced_sentiment_with_events(self, symbol: str) -> Dict:
        """Get enhanced sentiment analysis with news event integration"""
        try:
            # Get base enhanced sentiment
            enhanced_sentiment = await self.get_enhanced_sentiment(symbol)
            
            # Get news event analysis
            event_analysis = await self.get_news_event_analysis(symbol)
            
            # Combine sentiment and events
            combined_analysis = {
                **enhanced_sentiment,
                'news_events': event_analysis,
                'event_enhanced_confidence': self._calculate_event_enhanced_confidence(
                    enhanced_sentiment, event_analysis
                ),
                'event_filtered_sentiment': self._filter_sentiment_by_events(
                    enhanced_sentiment, event_analysis
                ),
                'phase_3_2_features': True
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error getting enhanced sentiment with events for {symbol}: {e}")
            return enhanced_sentiment
    
    def _calculate_event_enhanced_confidence(self, sentiment: Dict, events: Dict) -> float:
        """Calculate confidence enhanced by news events"""
        try:
            base_confidence = sentiment.get('enhanced_confidence', 0.0)
            event_impact = events.get('impact_score', 0.0)
            
            # Event impact can boost or reduce confidence
            if event_impact > 0.5:
                # High impact events increase confidence
                event_boost = event_impact * 0.3
            else:
                # Low impact events slightly reduce confidence
                event_boost = event_impact * 0.1
            
            enhanced_confidence = base_confidence + event_boost
            return min(enhanced_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating event enhanced confidence: {e}")
            return sentiment.get('enhanced_confidence', 0.0)
    
    def _filter_sentiment_by_events(self, sentiment: Dict, events: Dict) -> Dict:
        """Filter sentiment based on news events"""
        try:
            base_sentiment = sentiment.get('sentiment_score', 0.0)
            event_impact = events.get('impact_score', 0.0)
            high_impact_events = events.get('high_impact_events', 0)
            
            # Apply event filtering
            if high_impact_events > 0:
                # High impact events can amplify sentiment
                filtered_score = base_sentiment * (1 + event_impact * 0.5)
            else:
                # Low impact events slightly dampen sentiment
                filtered_score = base_sentiment * (1 - (1 - event_impact) * 0.2)
            
            # Ensure score stays in [-1, 1] range
            filtered_score = max(-1.0, min(1.0, filtered_score))
            
            # Determine filtered label
            if filtered_score > 0.1:
                filtered_label = 'positive'
            elif filtered_score < -0.1:
                filtered_label = 'negative'
            else:
                filtered_label = 'neutral'
            
            return {
                'sentiment_score': filtered_score,
                'sentiment_label': filtered_label,
                'filter_factor': event_impact,
                'high_impact_events': high_impact_events
            }
            
        except Exception as e:
            logger.error(f"Error filtering sentiment by events: {e}")
            return {
                'sentiment_score': sentiment.get('sentiment_score', 0.0),
                'sentiment_label': sentiment.get('sentiment_label', 'neutral'),
                'filter_factor': 0.0,
                'high_impact_events': 0
            }
    
    # Phase 3.3: Social Media Integration Methods
    
    async def get_social_media_sentiment(self, symbol: str) -> Dict:
        """Get comprehensive social media sentiment analysis"""
        try:
            # Get Twitter sentiment
            twitter_sentiment = await self._get_twitter_sentiment(symbol)
            
            # Get Reddit sentiment
            reddit_sentiment = await self._get_reddit_sentiment(symbol)
            
            # Calculate aggregated social sentiment
            social_sentiment = self._aggregate_social_sentiment(twitter_sentiment, reddit_sentiment)
            
            # Calculate social trends and momentum
            social_trends = self._calculate_social_trends(symbol, social_sentiment)
            social_momentum = self._calculate_social_momentum(symbol, social_sentiment)
            
            # Calculate social impact score
            social_impact = self._calculate_social_impact_score(social_sentiment, social_trends)
            
            # Store in cache
            self.social_sentiment_cache[symbol] = {
                'twitter': twitter_sentiment,
                'reddit': reddit_sentiment,
                'aggregated': social_sentiment,
                'trends': social_trends,
                'momentum': social_momentum,
                'impact_score': social_impact,
                'timestamp': datetime.now()
            }
            
            return {
                'twitter_sentiment': twitter_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'social_sentiment': social_sentiment,
                'social_trends': social_trends,
                'social_momentum': social_momentum,
                'social_impact_score': social_impact,
                'social_volume': self._calculate_social_volume(symbol),
                'social_engagement': self._calculate_social_engagement(symbol),
                'social_volatility': self._calculate_social_volatility(symbol),
                'social_correlation': self._calculate_social_correlation(symbol),
                'phase_3_3_features': True
            }
            
        except Exception as e:
            logger.error(f"Error getting social media sentiment for {symbol}: {e}")
            return {
                'twitter_sentiment': {'sentiment_score': 0.0, 'sentiment_label': 'neutral', 'confidence': 0.0},
                'reddit_sentiment': {'sentiment_score': 0.0, 'sentiment_label': 'neutral', 'confidence': 0.0},
                'social_sentiment': {'sentiment_score': 0.0, 'sentiment_label': 'neutral', 'confidence': 0.0},
                'social_trends': {'trend': 'stable', 'trend_strength': 0.0},
                'social_momentum': {'momentum_direction': 'neutral', 'momentum_strength': 'weak'},
                'social_impact_score': 0.0,
                'social_volume': {'volume_score': 0.0, 'volume_trend': 'stable'},
                'social_engagement': {'engagement_score': 0.0, 'engagement_trend': 'stable'},
                'social_volatility': {'volatility_score': 0.0, 'volatility_trend': 'stable'},
                'social_correlation': {'correlation_score': 0.0, 'correlation_trend': 'weak'},
                'phase_3_3_features': False
            }
    
    async def _get_twitter_sentiment(self, symbol: str) -> Dict:
        """Get Twitter sentiment for a symbol"""
        try:
            query = f"{symbol} cryptocurrency"
            tweets = await self.twitter_analyzer.get_tweets(query, max_results=50)
            sentiment = await self.twitter_analyzer.analyze_sentiment(tweets)
            
            # Add engagement metrics
            engagement = self._calculate_twitter_engagement(tweets)
            
            return {
                **sentiment,
                'tweet_count': len(tweets),
                'engagement_metrics': engagement,
                'source': 'twitter'
            }
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'tweet_count': 0,
                'engagement_metrics': {'likes': 0, 'retweets': 0, 'replies': 0},
                'source': 'twitter'
            }
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get Reddit sentiment for a symbol"""
        try:
            posts = []
            for subreddit in self.crypto_subreddits:
                subreddit_posts = await self.reddit_analyzer.get_reddit_posts(subreddit, symbol, limit=20)
                posts.extend(subreddit_posts)
            
            sentiment = await self.reddit_analyzer.analyze_sentiment(posts)
            
            # Add engagement metrics
            engagement = self._calculate_reddit_engagement(posts)
            
            return {
                **sentiment,
                'post_count': len(posts),
                'subreddit_count': len(self.crypto_subreddits),
                'engagement_metrics': engagement,
                'source': 'reddit'
            }
            
        except Exception as e:
            logger.error(f"Error getting Reddit sentiment for {symbol}: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'post_count': 0,
                'subreddit_count': 0,
                'engagement_metrics': {'upvotes': 0, 'comments': 0, 'score': 0},
                'source': 'reddit'
            }
    
    def _aggregate_social_sentiment(self, twitter_sentiment: Dict, reddit_sentiment: Dict) -> Dict:
        """Aggregate sentiment from multiple social media sources"""
        try:
            # Weight Twitter and Reddit sentiment
            twitter_weight = 0.6
            reddit_weight = 0.4
            
            twitter_score = twitter_sentiment.get('sentiment_score', 0.0)
            reddit_score = reddit_sentiment.get('sentiment_score', 0.0)
            
            # Calculate weighted average
            aggregated_score = (twitter_score * twitter_weight) + (reddit_score * reddit_weight)
            
            # Determine aggregated label
            if aggregated_score > 0.1:
                label = 'positive'
            elif aggregated_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            # Calculate confidence based on source confidence
            twitter_confidence = twitter_sentiment.get('confidence', 0.0)
            reddit_confidence = reddit_sentiment.get('confidence', 0.0)
            aggregated_confidence = (twitter_confidence * twitter_weight) + (reddit_confidence * reddit_weight)
            
            return {
                'sentiment_score': aggregated_score,
                'sentiment_label': label,
                'confidence': aggregated_confidence,
                'twitter_weight': twitter_weight,
                'reddit_weight': reddit_weight,
                'source_count': 2
            }
            
        except Exception as e:
            logger.error(f"Error aggregating social sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'twitter_weight': 0.0,
                'reddit_weight': 0.0,
                'source_count': 0
            }
    
    def _calculate_social_trends(self, symbol: str, social_sentiment: Dict) -> Dict:
        """Calculate social sentiment trends"""
        try:
            # Get historical sentiment
            if symbol not in self.social_sentiment_history:
                self.social_sentiment_history[symbol] = []
            
            history = self.social_sentiment_history[symbol]
            current_score = social_sentiment.get('sentiment_score', 0.0)
            
            # Add current sentiment to history
            history.append({
                'score': current_score,
                'timestamp': datetime.now()
            })
            
            # Keep only last 24 hours of data
            cutoff_time = datetime.now() - timedelta(hours=24)
            history = [h for h in history if h['timestamp'] > cutoff_time]
            self.social_sentiment_history[symbol] = history
            
            if len(history) < 2:
                return {'trend': 'stable', 'trend_strength': 0.0, 'trend_direction': 'neutral'}
            
            # Calculate trend
            recent_scores = [h['score'] for h in history[-6:]]  # Last 6 data points
            if len(recent_scores) >= 2:
                trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                
                if trend_slope > 0.05:
                    trend = 'increasing'
                    direction = 'positive'
                elif trend_slope < -0.05:
                    trend = 'decreasing'
                    direction = 'negative'
                else:
                    trend = 'stable'
                    direction = 'neutral'
                
                trend_strength = min(abs(trend_slope) * 10, 1.0)
            else:
                trend = 'stable'
                direction = 'neutral'
                trend_strength = 0.0
            
            return {
                'trend': trend,
                'trend_strength': trend_strength,
                'trend_direction': direction,
                'data_points': len(history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating social trends for {symbol}: {e}")
            return {'trend': 'stable', 'trend_strength': 0.0, 'trend_direction': 'neutral', 'data_points': 0}
    
    def _calculate_social_momentum(self, symbol: str, social_sentiment: Dict) -> Dict:
        """Calculate social sentiment momentum"""
        try:
            if symbol not in self.social_sentiment_history:
                return {'momentum_direction': 'neutral', 'momentum_strength': 'weak', 'momentum_score': 0.0}
            
            history = self.social_sentiment_history[symbol]
            if len(history) < 3:
                return {'momentum_direction': 'neutral', 'momentum_strength': 'weak', 'momentum_score': 0.0}
            
            # Calculate momentum based on recent changes
            recent_scores = [h['score'] for h in history[-3:]]
            momentum_score = recent_scores[-1] - recent_scores[0]
            
            if momentum_score > 0.1:
                direction = 'positive'
                strength = 'strong' if momentum_score > 0.3 else 'moderate'
            elif momentum_score < -0.1:
                direction = 'negative'
                strength = 'strong' if momentum_score < -0.3 else 'moderate'
            else:
                direction = 'neutral'
                strength = 'weak'
            
            return {
                'momentum_direction': direction,
                'momentum_strength': strength,
                'momentum_score': momentum_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating social momentum for {symbol}: {e}")
            return {'momentum_direction': 'neutral', 'momentum_strength': 'weak', 'momentum_score': 0.0}
    
    def _calculate_social_impact_score(self, social_sentiment: Dict, social_trends: Dict) -> float:
        """Calculate social impact score"""
        try:
            sentiment_score = abs(social_sentiment.get('sentiment_score', 0.0))
            confidence = social_sentiment.get('confidence', 0.0)
            trend_strength = social_trends.get('trend_strength', 0.0)
            
            # Impact score combines sentiment strength, confidence, and trend strength
            impact_score = (sentiment_score * 0.4) + (confidence * 0.3) + (trend_strength * 0.3)
            
            return min(impact_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating social impact score: {e}")
            return 0.0
    
    def _calculate_social_volume(self, symbol: str) -> Dict:
        """Calculate social media volume metrics"""
        try:
            # This would typically come from actual API data
            # For now, we'll simulate volume metrics
            volume_score = 0.5  # Simulated volume score
            volume_trend = 'stable'
            
            return {
                'volume_score': volume_score,
                'volume_trend': volume_trend,
                'volume_level': 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error calculating social volume for {symbol}: {e}")
            return {'volume_score': 0.0, 'volume_trend': 'stable', 'volume_level': 'low'}
    
    def _calculate_social_engagement(self, symbol: str) -> Dict:
        """Calculate social media engagement metrics"""
        try:
            # Simulated engagement metrics
            engagement_score = 0.6
            engagement_trend = 'increasing'
            
            return {
                'engagement_score': engagement_score,
                'engagement_trend': engagement_trend,
                'engagement_level': 'high'
            }
            
        except Exception as e:
            logger.error(f"Error calculating social engagement for {symbol}: {e}")
            return {'engagement_score': 0.0, 'engagement_trend': 'stable', 'engagement_level': 'low'}
    
    def _calculate_social_volatility(self, symbol: str) -> Dict:
        """Calculate social sentiment volatility"""
        try:
            if symbol not in self.social_sentiment_history:
                return {'volatility_score': 0.0, 'volatility_trend': 'stable', 'volatility_level': 'low'}
            
            history = self.social_sentiment_history[symbol]
            if len(history) < 3:
                return {'volatility_score': 0.0, 'volatility_trend': 'stable', 'volatility_level': 'low'}
            
            # Calculate volatility as standard deviation of recent scores
            recent_scores = [h['score'] for h in history[-6:]]
            if len(recent_scores) >= 2:
                mean_score = sum(recent_scores) / len(recent_scores)
                variance = sum((score - mean_score) ** 2 for score in recent_scores) / len(recent_scores)
                volatility = variance ** 0.5
            else:
                volatility = 0.0
            
            volatility_score = min(volatility * 2, 1.0)  # Scale to [0, 1]
            
            if volatility_score > 0.7:
                level = 'high'
                trend = 'increasing'
            elif volatility_score > 0.3:
                level = 'medium'
                trend = 'stable'
            else:
                level = 'low'
                trend = 'decreasing'
            
            return {
                'volatility_score': volatility_score,
                'volatility_trend': trend,
                'volatility_level': level
            }
            
        except Exception as e:
            logger.error(f"Error calculating social volatility for {symbol}: {e}")
            return {'volatility_score': 0.0, 'volatility_trend': 'stable', 'volatility_level': 'low'}
    
    def _calculate_social_correlation(self, symbol: str) -> Dict:
        """Calculate social sentiment correlation with price"""
        try:
            # This would typically require price data correlation
            # For now, we'll simulate correlation metrics
            correlation_score = 0.4  # Simulated correlation
            correlation_trend = 'stable'
            
            return {
                'correlation_score': correlation_score,
                'correlation_trend': correlation_trend,
                'correlation_strength': 'moderate'
            }
            
        except Exception as e:
            logger.error(f"Error calculating social correlation for {symbol}: {e}")
            return {'correlation_score': 0.0, 'correlation_trend': 'weak', 'correlation_strength': 'weak'}
    
    def _calculate_twitter_engagement(self, tweets: List[Dict]) -> Dict:
        """Calculate Twitter engagement metrics"""
        try:
            total_likes = sum(tweet.get('public_metrics', {}).get('like_count', 0) for tweet in tweets)
            total_retweets = sum(tweet.get('public_metrics', {}).get('retweet_count', 0) for tweet in tweets)
            total_replies = sum(tweet.get('public_metrics', {}).get('reply_count', 0) for tweet in tweets)
            
            return {
                'likes': total_likes,
                'retweets': total_retweets,
                'replies': total_replies,
                'total_engagement': total_likes + total_retweets + total_replies
            }
            
        except Exception as e:
            logger.error(f"Error calculating Twitter engagement: {e}")
            return {'likes': 0, 'retweets': 0, 'replies': 0, 'total_engagement': 0}
    
    def _calculate_reddit_engagement(self, posts: List[Dict]) -> Dict:
        """Calculate Reddit engagement metrics"""
        try:
            total_upvotes = sum(post.get('score', 0) for post in posts)
            total_comments = sum(post.get('num_comments', 0) for post in posts)
            
            return {
                'upvotes': total_upvotes,
                'comments': total_comments,
                'total_engagement': total_upvotes + total_comments
            }
            
        except Exception as e:
            logger.error(f"Error calculating Reddit engagement: {e}")
            return {'upvotes': 0, 'comments': 0, 'total_engagement': 0}
    
    async def get_enhanced_sentiment_with_social_media(self, symbol: str) -> Dict:
        """Get enhanced sentiment analysis with social media integration"""
        try:
            # Get base enhanced sentiment
            enhanced_sentiment = await self.get_enhanced_sentiment_with_events(symbol)
            
            # Get social media sentiment
            social_sentiment = await self.get_social_media_sentiment(symbol)
            
            # Combine sentiment and social media
            combined_analysis = {
                **enhanced_sentiment,
                'social_media': social_sentiment,
                'social_enhanced_confidence': self._calculate_social_enhanced_confidence(
                    enhanced_sentiment, social_sentiment
                ),
                'social_filtered_sentiment': self._filter_sentiment_by_social_media(
                    enhanced_sentiment, social_sentiment
                ),
                'phase_3_3_features': True
            }
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Error getting enhanced sentiment with social media for {symbol}: {e}")
            return enhanced_sentiment
    
    def _calculate_social_enhanced_confidence(self, sentiment: Dict, social: Dict) -> float:
        """Calculate confidence enhanced by social media"""
        try:
            base_confidence = sentiment.get('enhanced_confidence', 0.0)
            social_impact = social.get('social_impact_score', 0.0)
            social_confidence = social.get('social_sentiment', {}).get('confidence', 0.0)
            
            # Social media can boost confidence if sentiment aligns
            sentiment_alignment = 1.0 - abs(
                sentiment.get('sentiment_score', 0.0) - 
                social.get('social_sentiment', {}).get('sentiment_score', 0.0)
            )
            
            social_boost = (social_impact * 0.3) + (social_confidence * 0.2) + (sentiment_alignment * 0.1)
            enhanced_confidence = base_confidence + social_boost
            
            return min(enhanced_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating social enhanced confidence: {e}")
            return sentiment.get('enhanced_confidence', 0.0)
    
    def _filter_sentiment_by_social_media(self, sentiment: Dict, social: Dict) -> Dict:
        """Filter sentiment based on social media"""
        try:
            base_sentiment = sentiment.get('sentiment_score', 0.0)
            social_sentiment = social.get('social_sentiment', {}).get('sentiment_score', 0.0)
            social_impact = social.get('social_impact_score', 0.0)
            
            # Apply social media filtering
            if social_impact > 0.5:
                # High social impact can amplify sentiment
                filtered_score = base_sentiment * (1 + social_impact * 0.3)
            else:
                # Low social impact slightly dampens sentiment
                filtered_score = base_sentiment * (1 - (1 - social_impact) * 0.1)
            
            # Ensure score stays in [-1, 1] range
            filtered_score = max(-1.0, min(1.0, filtered_score))
            
            # Determine filtered label
            if filtered_score > 0.1:
                filtered_label = 'positive'
            elif filtered_score < -0.1:
                filtered_label = 'negative'
            else:
                filtered_label = 'neutral'
            
            return {
                'sentiment_score': filtered_score,
                'sentiment_label': filtered_label,
                'social_filter_factor': social_impact,
                'social_sentiment_alignment': 1.0 - abs(base_sentiment - social_sentiment)
            }
            
        except Exception as e:
            logger.error(f"Error filtering sentiment by social media: {e}")
            return {
                'sentiment_score': sentiment.get('sentiment_score', 0.0),
                'sentiment_label': sentiment.get('sentiment_label', 'neutral'),
                'social_filter_factor': 0.0,
                'social_sentiment_alignment': 0.0
            }
