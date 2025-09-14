"""
News Sentiment Integration Service
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class NewsSentimentService:
    """Service for integrating news sentiment analysis with real-time capabilities"""
    
    def __init__(self):
        self.news_api_key = "9d9a3e710a0a454f8bcee7e4f04e3c24"  # From config
        self.news_buffer = []
        
        # Real-time news tracking
        self.recent_news = {}
        self.sentiment_cache = {}
        self.breaking_news_alerts = []
        
        # News sources for real-time updates
        self.news_sources = [
            'https://newsapi.org/v2/everything',
            'https://api.coindesk.com/v1/news',
            'https://cryptopanic.com/api/v1/posts/'
        ]
        
    async def get_crypto_news_realtime(self, symbol: str = "BTC", limit: int = 20) -> List[Dict[str, Any]]:
        """Get cryptocurrency news with real-time updates"""
        try:
            # Get news from multiple sources
            all_news = []
            
            # NewsAPI
            newsapi_news = await self._get_newsapi_news(symbol, limit)
            all_news.extend(newsapi_news)
            
            # CryptoPanic (if available)
            try:
                cryptopanic_news = await self._get_cryptopanic_news(symbol, limit)
                all_news.extend(cryptopanic_news)
            except Exception as e:
                logger.warning(f"CryptoPanic API failed: {e}")
            
            # Sort by timestamp and remove duplicates
            all_news = self._deduplicate_news(all_news)
            all_news.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
            
            # Update cache
            self.recent_news[symbol] = all_news[:limit]
            
            return all_news[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching real-time news: {e}")
            return []
    
    async def _get_newsapi_news(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Get news from NewsAPI"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': f"{symbol} cryptocurrency",
                    'apiKey': self.news_api_key,
                    'sortBy': 'publishedAt',
                    'pageSize': limit,
                    'language': 'en',
                    'from': (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%S')
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        # Process articles
                        processed_articles = []
                        for article in articles:
                            processed_article = {
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'publishedAt': article.get('publishedAt', ''),
                                'source': article.get('source', {}).get('name', 'NewsAPI'),
                                'content': article.get('content', ''),
                                'sentiment': self._analyze_sentiment_simple(article.get('title', '') + ' ' + article.get('description', '')),
                                'symbol': symbol,
                                'is_breaking': self._is_breaking_news(article.get('title', ''), article.get('description', ''))
                            }
                            processed_articles.append(processed_article)
                            
                            # Check for breaking news
                            if processed_article['is_breaking']:
                                self._add_breaking_news_alert(processed_article)
                        
                        return processed_articles
                    else:
                        logger.error(f"NewsAPI error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI news: {e}")
            return []
    
    async def _get_cryptopanic_news(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Get news from CryptoPanic API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://cryptopanic.com/api/v1/posts/"
                params = {
                    'auth_token': '',  # Add API key if available
                    'currencies': symbol,
                    'kind': 'news',
                    'public': 'true',
                    'page': 1
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        posts = data.get('results', [])
                        
                        processed_posts = []
                        for post in posts[:limit]:
                            processed_post = {
                                'title': post.get('title', ''),
                                'description': post.get('domain', ''),
                                'url': post.get('url', ''),
                                'publishedAt': post.get('created_at', ''),
                                'source': 'CryptoPanic',
                                'content': post.get('title', ''),
                                'sentiment': self._analyze_sentiment_simple(post.get('title', '')),
                                'symbol': symbol,
                                'is_breaking': self._is_breaking_news(post.get('title', ''), ''),
                                'votes': post.get('votes', {}).get('positive', 0) - post.get('votes', {}).get('negative', 0)
                            }
                            processed_posts.append(processed_post)
                        
                        return processed_posts
                    else:
                        logger.warning(f"CryptoPanic API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching CryptoPanic news: {e}")
            return []
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """Remove duplicate news articles"""
        seen_titles = set()
        unique_news = []
        
        for article in news_list:
            title = article.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(article)
        
        return unique_news
    
    def _is_breaking_news(self, title: str, description: str) -> bool:
        """Detect if news is breaking/urgent"""
        breaking_keywords = [
            'breaking', 'urgent', 'alert', 'crash', 'surge', 'plunge',
            'hack', 'exploit', 'regulation', 'ban', 'approval', 'rejection',
            'partnership', 'acquisition', 'merger', 'launch', 'release'
        ]
        
        text = (title + ' ' + description).lower()
        return any(keyword in text for keyword in breaking_keywords)
    
    def _add_breaking_news_alert(self, article: Dict):
        """Add breaking news alert"""
        alert = {
            'title': article['title'],
            'symbol': article['symbol'],
            'timestamp': datetime.now().isoformat(),
            'sentiment': article['sentiment'],
            'url': article['url']
        }
        
        self.breaking_news_alerts.append(alert)
        
        # Keep only last 50 alerts
        if len(self.breaking_news_alerts) > 50:
            self.breaking_news_alerts = self.breaking_news_alerts[-50:]
    
    def get_breaking_news_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent breaking news alerts"""
        return self.breaking_news_alerts[-limit:] if self.breaking_news_alerts else []
    
    def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment summary for a symbol"""
        try:
            if symbol not in self.recent_news:
                return {'sentiment': 'neutral', 'confidence': 0.0, 'news_count': 0}
            
            news_articles = self.recent_news[symbol]
            if not news_articles:
                return {'sentiment': 'neutral', 'confidence': 0.0, 'news_count': 0}
            
            # Calculate sentiment scores
            sentiments = [article.get('sentiment', 0) for article in news_articles]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # Determine sentiment label
            if avg_sentiment > 0.1:
                sentiment_label = 'positive'
            elif avg_sentiment < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            # Calculate confidence based on sentiment consistency
            positive_count = sum(1 for s in sentiments if s > 0.1)
            negative_count = sum(1 for s in sentiments if s < -0.1)
            total_count = len(sentiments)
            
            confidence = max(positive_count, negative_count) / total_count if total_count > 0 else 0
            
            return {
                'sentiment': sentiment_label,
                'sentiment_score': avg_sentiment,
                'confidence': confidence,
                'news_count': total_count,
                'positive_news': positive_count,
                'negative_news': negative_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0, 'news_count': 0}
    
    def _analyze_sentiment_simple(self, text: str) -> float:
        """Simple sentiment analysis based on keywords"""
        try:
            # Simple sentiment analysis based on keywords
            positive_keywords = [
                'bullish', 'rise', 'up', 'gain', 'profit', 'success', 'positive',
                'growth', 'increase', 'surge', 'rally', 'breakthrough', 'breakout',
                'adoption', 'partnership', 'launch', 'approval', 'support'
            ]
            
            negative_keywords = [
                'bearish', 'fall', 'down', 'loss', 'fail', 'negative', 'decline',
                'decrease', 'crash', 'plunge', 'drop', 'rejection', 'ban',
                'regulation', 'hack', 'exploit', 'scam', 'fraud'
            ]
            
            text_lower = text.lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
            
            total_keywords = positive_count + negative_count
            
            if total_keywords == 0:
                return 0.0  # Neutral
            
            # Return sentiment score between -1 and 1
            sentiment_score = (positive_count - negative_count) / total_keywords
            return max(-1.0, min(1.0, sentiment_score))
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    async def get_crypto_news(self, symbol: str = "BTC") -> List[Dict[str, Any]]:
        """Get cryptocurrency news (legacy method for backward compatibility)"""
        return await self.get_crypto_news_realtime(symbol, 10)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using simple keyword analysis (legacy method)"""
        score = self._analyze_sentiment_simple(text)
        return {
            'sentiment_score': score,
            'sentiment_label': 'positive' if score > 0.1 else 'negative' if score < -0.1 else 'neutral'
        }
