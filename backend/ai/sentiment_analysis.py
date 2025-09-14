import asyncio
import logging
from typing import Optional, Dict, List
import aiohttp
import json
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import praw
import tweepy
from sqlalchemy.orm import Session

from app.core.config import settings, NEWS_KEYWORDS
from app.models.database import SentimentData, get_db


class SentimentService:
    """
    Service for analyzing market sentiment from multiple sources.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize API clients
        self.twitter_api = None
        self.reddit_api = None
        self.news_api_key = settings.NEWS_API_KEY
        
        self._initialize_apis()
        
        # Cache for sentiment data
        self.sentiment_cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def _initialize_apis(self):
        """Initialize API connections."""
        try:
            # Initialize Twitter API
            if all([settings.TWITTER_API_KEY, settings.TWITTER_API_SECRET, 
                   settings.TWITTER_ACCESS_TOKEN, settings.TWITTER_ACCESS_TOKEN_SECRET]):
                auth = tweepy.OAuthHandler(settings.TWITTER_API_KEY, settings.TWITTER_API_SECRET)
                auth.set_access_token(settings.TWITTER_ACCESS_TOKEN, settings.TWITTER_ACCESS_TOKEN_SECRET)
                self.twitter_api = tweepy.API(auth, wait_on_rate_limit=True)
            
            # Initialize Reddit API
            if settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET:
                self.reddit_api = praw.Reddit(
                    client_id=settings.REDDIT_CLIENT_ID,
                    client_secret=settings.REDDIT_CLIENT_SECRET,
                    user_agent=settings.REDDIT_USER_AGENT
                )
            
        except Exception as e:
            self.logger.error(f"Error initializing APIs: {e}")
    
    async def get_sentiment(self, symbol: str) -> Optional[Dict]:
        """
        Get aggregated sentiment for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with sentiment data
        """
        try:
            # Check cache first
            cache_key = f"sentiment_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_timeout:
                    return cached_data
            
            # Get sentiment from multiple sources
            twitter_sentiment = await self._get_twitter_sentiment(symbol)
            reddit_sentiment = await self._get_reddit_sentiment(symbol)
            news_sentiment = await self._get_news_sentiment(symbol)
            
            # Aggregate sentiment
            aggregated_sentiment = self._aggregate_sentiment(
                twitter_sentiment, reddit_sentiment, news_sentiment
            )
            
            # Save to database
            await self._save_sentiment_data(symbol, aggregated_sentiment)
            
            # Update cache
            self.sentiment_cache[cache_key] = (aggregated_sentiment, datetime.now())
            
            return aggregated_sentiment
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol}: {e}")
            return None
    
    async def _get_twitter_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get sentiment from Twitter."""
        if not self.twitter_api:
            return None
        
        try:
            # Convert symbol to search terms
            search_terms = self._get_search_terms(symbol)
            
            all_tweets = []
            for term in search_terms:
                tweets = self.twitter_api.search_tweets(
                    q=term,
                    lang='en',
                    count=100,
                    tweet_mode='extended'
                )
                all_tweets.extend(tweets)
            
            if not all_tweets:
                return None
            
            # Analyze sentiment
            sentiments = []
            for tweet in all_tweets:
                text = tweet.full_text
                # Clean text
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'@\w+|#\w+', '', text)
                
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                sentiments.append({
                    'score': sentiment_score,
                    'text': text,
                    'created_at': tweet.created_at,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                })
            
            # Calculate weighted average
            total_weight = sum(tweet['retweet_count'] + tweet['favorite_count'] + 1 for tweet in sentiments)
            weighted_score = sum(
                tweet['score'] * (tweet['retweet_count'] + tweet['favorite_count'] + 1) 
                for tweet in sentiments
            ) / total_weight if total_weight > 0 else 0
            
            return {
                'score': weighted_score,
                'count': len(sentiments),
                'source': 'twitter',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Twitter sentiment: {e}")
            return None
    
    async def _get_reddit_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get sentiment from Reddit."""
        if not self.reddit_api:
            return None
        
        try:
            # Convert symbol to subreddit search terms
            search_terms = self._get_search_terms(symbol)
            
            all_posts = []
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'CryptoMarkets']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_api.subreddit(subreddit_name)
                    
                    for term in search_terms:
                        posts = subreddit.search(term, sort='hot', limit=50)
                        all_posts.extend(posts)
                        
                except Exception as e:
                    self.logger.warning(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
            
            if not all_posts:
                return None
            
            # Analyze sentiment
            sentiments = []
            for post in all_posts:
                text = f"{post.title} {post.selftext}"
                # Clean text
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                sentiments.append({
                    'score': sentiment_score,
                    'text': text[:200],  # Truncate for storage
                    'created_at': datetime.fromtimestamp(post.created_utc),
                    'score_reddit': post.score,
                    'num_comments': post.num_comments
                })
            
            # Calculate weighted average
            total_weight = sum(post['score_reddit'] + post['num_comments'] + 1 for post in sentiments)
            weighted_score = sum(
                post['score'] * (post['score_reddit'] + post['num_comments'] + 1) 
                for post in sentiments
            ) / total_weight if total_weight > 0 else 0
            
            return {
                'score': weighted_score,
                'count': len(sentiments),
                'source': 'reddit',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit sentiment: {e}")
            return None
    
    async def _get_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get sentiment from news articles."""
        if not self.news_api_key:
            return None
        
        try:
            # Convert symbol to search terms
            search_terms = self._get_search_terms(symbol)
            
            all_articles = []
            async with aiohttp.ClientSession() as session:
                for term in search_terms:
                    url = f"https://newsapi.org/v2/everything"
                    params = {
                        'q': term,
                        'apiKey': self.news_api_key,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 50
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get('articles', [])
                            all_articles.extend(articles)
            
            if not all_articles:
                return None
            
            # Analyze sentiment
            sentiments = []
            for article in all_articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                text = f"{title} {description} {content}"
                # Clean text
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                sentiments.append({
                    'score': sentiment_score,
                    'title': title,
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', '')
                })
            
            # Calculate average
            avg_score = sum(article['score'] for article in sentiments) / len(sentiments)
            
            return {
                'score': avg_score,
                'count': len(sentiments),
                'source': 'news',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting news sentiment: {e}")
            return None
    
    def _get_search_terms(self, symbol: str) -> List[str]:
        """Convert symbol to search terms."""
        # Extract base currency from symbol (e.g., 'BTC' from 'BTC/USDT')
        base_currency = symbol.split('/')[0]
        
        # Create search terms
        search_terms = [
            base_currency,
            f"#{base_currency}",
            f"${base_currency}",
            f"{base_currency} price",
            f"{base_currency} crypto"
        ]
        
        # Add common variations
        if base_currency == 'BTC':
            search_terms.extend(['bitcoin', 'Bitcoin', '#bitcoin'])
        elif base_currency == 'ETH':
            search_terms.extend(['ethereum', 'Ethereum', '#ethereum'])
        
        return search_terms
    
    def _aggregate_sentiment(self, twitter_sentiment: Optional[Dict], 
                           reddit_sentiment: Optional[Dict], 
                           news_sentiment: Optional[Dict]) -> Dict:
        """Aggregate sentiment from multiple sources."""
        sentiments = []
        weights = {
            'twitter': 0.4,
            'reddit': 0.3,
            'news': 0.3
        }
        
        if twitter_sentiment:
            sentiments.append((twitter_sentiment['score'], weights['twitter']))
        
        if reddit_sentiment:
            sentiments.append((reddit_sentiment['score'], weights['reddit']))
        
        if news_sentiment:
            sentiments.append((news_sentiment['score'], weights['news']))
        
        if not sentiments:
            return {
                'score': 0.0,
                'label': 'neutral',
                'confidence': 0.0,
                'sources': []
            }
        
        # Calculate weighted average
        total_weight = sum(weight for _, weight in sentiments)
        weighted_score = sum(score * weight for score, weight in sentiments) / total_weight
        
        # Determine label
        if weighted_score > 0.1:
            label = 'positive'
        elif weighted_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        # Calculate confidence based on number of sources
        confidence = min(len(sentiments) / 3, 1.0)
        
        return {
            'score': weighted_score,
            'label': label,
            'confidence': confidence,
            'sources': [
                {'source': 'twitter', 'data': twitter_sentiment},
                {'source': 'reddit', 'data': reddit_sentiment},
                {'source': 'news', 'data': news_sentiment}
            ],
            'timestamp': datetime.now()
        }
    
    async def _save_sentiment_data(self, symbol: str, sentiment_data: Dict):
        """Save sentiment data to database."""
        try:
            db = next(get_db())
            
            # Save aggregated sentiment
            sentiment_record = SentimentData(
                symbol=symbol,
                source='aggregated',
                timestamp=sentiment_data['timestamp'],
                sentiment_score=sentiment_data['score'],
                sentiment_label=sentiment_data['label'],
                confidence=sentiment_data['confidence']
            )
            
            db.add(sentiment_record)
            
            # Save individual source data
            for source_data in sentiment_data['sources']:
                if source_data['data']:
                    source_record = SentimentData(
                        symbol=symbol,
                        source=source_data['source'],
                        timestamp=source_data['data']['timestamp'],
                        sentiment_score=source_data['data']['score'],
                        sentiment_label=self._get_sentiment_label(source_data['data']['score']),
                        confidence=1.0
                    )
                    db.add(source_record)
            
            db.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    async def get_market_sentiment_summary(self) -> Optional[Dict]:
        """Get overall market sentiment summary."""
        try:
            # Get sentiment for major cryptocurrencies
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            
            sentiments = {}
            for symbol in symbols:
                sentiment = await self.get_sentiment(symbol)
                if sentiment:
                    sentiments[symbol] = sentiment
            
            if not sentiments:
                return None
            
            # Calculate market-wide sentiment
            avg_score = sum(sentiment['score'] for sentiment in sentiments.values()) / len(sentiments)
            
            return {
                'market_sentiment': avg_score,
                'market_label': self._get_sentiment_label(avg_score),
                'symbol_sentiments': sentiments,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market sentiment summary: {e}")
            return None
    
    async def get_sentiment_trend(self, symbol: str, hours: int = 24) -> Optional[Dict]:
        """Get sentiment trend over time."""
        try:
            db = next(get_db())
            
            # Get sentiment data from the last N hours
            since = datetime.now() - timedelta(hours=hours)
            
            sentiment_records = db.query(SentimentData).filter(
                SentimentData.symbol == symbol,
                SentimentData.source == 'aggregated',
                SentimentData.timestamp >= since
            ).order_by(SentimentData.timestamp.asc()).all()
            
            if not sentiment_records:
                return None
            
            # Calculate trend
            scores = [record.sentiment_score for record in sentiment_records]
            timestamps = [record.timestamp for record in sentiment_records]
            
            # Calculate trend direction
            if len(scores) >= 2:
                trend_direction = 'increasing' if scores[-1] > scores[0] else 'decreasing'
                trend_strength = abs(scores[-1] - scores[0])
            else:
                trend_direction = 'stable'
                trend_strength = 0
            
            return {
                'symbol': symbol,
                'current_score': scores[-1] if scores else 0,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'scores': scores,
                'timestamps': timestamps,
                'period_hours': hours
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment trend for {symbol}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the sentiment cache."""
        self.sentiment_cache.clear()
        self.logger.info("Sentiment cache cleared")
