"""
Natural Language Processing Engine for AlphaPulse
Comprehensive NLP implementation for sentiment analysis, news processing, and market commentary analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import warnings
import re
import json
warnings.filterwarnings('ignore')

# NLP imports
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available - using mock sentiment analysis")

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False
    logging.warning("Tweepy not available - Twitter analysis disabled")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logging.warning("PRAW not available - Reddit analysis disabled")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline as transformers_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - using TextBlob for sentiment analysis")

logger = logging.getLogger(__name__)

class NLPSource(Enum):
    """NLP data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    FORUMS = "forums"

class SentimentType(Enum):
    """Sentiment types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class NLPText:
    """NLP text representation"""
    text: str
    source: NLPSource
    timestamp: datetime
    author: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment_type: SentimentType
    sentiment_score: float
    confidence: float
    source: NLPSource
    timestamp: datetime
    keywords: List[str]
    entities: List[str]
    metadata: Dict[str, Any]

@dataclass
class NewsArticle:
    """News article representation"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    sentiment_score: float
    keywords: List[str]
    relevance_score: float

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis using multiple models"""
    
    def __init__(self):
        self.models = {}
        self.keywords = {
            'positive': [
                'bull', 'bullish', 'moon', 'pump', 'buy', 'long', 'profit', 'gain', 'surge', 'rally',
                'breakout', 'uptrend', 'strong', 'buying', 'accumulation', 'hodl', 'diamond hands',
                'to the moon', 'lambo', 'mooning', 'rocket', 'green', 'positive', 'optimistic'
            ],
            'negative': [
                'bear', 'bearish', 'dump', 'sell', 'short', 'loss', 'crash', 'drop', 'plunge', 'decline',
                'breakdown', 'downtrend', 'weak', 'selling', 'distribution', 'paper hands', 'rekt',
                'to zero', 'dead', 'dumpster', 'red', 'negative', 'pessimistic', 'fud'
            ],
            'volume': [
                'volume', 'liquidity', 'market cap', 'circulating supply', 'total supply',
                'buying pressure', 'selling pressure', 'whale', 'retail', 'institutional'
            ],
            'technical': [
                'support', 'resistance', 'breakout', 'breakdown', 'consolidation', 'accumulation',
                'distribution', 'trend', 'momentum', 'rsi', 'macd', 'moving average', 'fibonacci'
            ]
        }
        
        self._initialize_models()
        logger.info("Advanced Sentiment Analyzer initialized")
    
    def _initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Initialize transformer-based sentiment analysis
                self.models['transformer'] = transformers_pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    return_all_scores=True
                )
                logger.info("✅ Transformer sentiment model loaded")
            
            if TEXTBLOB_AVAILABLE:
                self.models['textblob'] = True
                logger.info("✅ TextBlob sentiment analysis available")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    def analyze_sentiment(self, text: str, source: NLPSource = NLPSource.NEWS) -> SentimentResult:
        """Analyze sentiment using multiple models"""
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Get sentiment scores from different models
            scores = {}
            
            # TextBlob analysis
            if 'textblob' in self.models:
                blob = TextBlob(cleaned_text)
                scores['textblob'] = blob.sentiment.polarity
            
            # Transformer analysis
            if 'transformer' in self.models:
                try:
                    result = self.models['transformer'](cleaned_text[:512])  # Limit length
                    if result and len(result) > 0:
                        # Extract positive and negative scores
                        positive_score = 0.0
                        negative_score = 0.0
                        
                        for item in result[0]:
                            if 'POSITIVE' in item['label'].upper():
                                positive_score = item['score']
                            elif 'NEGATIVE' in item['label'].upper():
                                negative_score = item['score']
                        
                        # Calculate sentiment score (-1 to 1)
                        transformer_score = positive_score - negative_score
                        scores['transformer'] = transformer_score
                except Exception as e:
                    logger.warning(f"Transformer analysis failed: {e}")
            
            # Keyword-based analysis
            keyword_score = self._analyze_keywords(cleaned_text)
            scores['keyword'] = keyword_score
            
            # Aggregate scores
            final_score = self._aggregate_scores(scores)
            
            # Determine sentiment type
            sentiment_type = self._determine_sentiment_type(final_score)
            
            # Extract keywords and entities
            keywords = self._extract_keywords(cleaned_text)
            entities = self._extract_entities(cleaned_text)
            
            # Calculate confidence
            confidence = self._calculate_confidence(scores, final_score)
            
            return SentimentResult(
                text=cleaned_text,
                sentiment_type=sentiment_type,
                sentiment_score=final_score,
                confidence=confidence,
                source=source,
                timestamp=datetime.now(),
                keywords=keywords,
                entities=entities,
                metadata={'model_scores': scores}
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return SentimentResult(
                text=text,
                sentiment_type=SentimentType.NEUTRAL,
                sentiment_score=0.0,
                confidence=0.0,
                source=source,
                timestamp=datetime.now(),
                keywords=[],
                entities=[],
                metadata={'error': str(e)}
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.,!?@#$%&*()]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def _analyze_keywords(self, text: str) -> float:
        """Analyze sentiment based on keywords"""
        positive_count = sum(1 for word in self.keywords['positive'] if word in text)
        negative_count = sum(1 for word in self.keywords['negative'] if word in text)
        
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0
        
        # Calculate score (-1 to 1)
        score = (positive_count - negative_count) / total_keywords
        return max(-1.0, min(1.0, score))
    
    def _aggregate_scores(self, scores: Dict[str, float]) -> float:
        """Aggregate scores from different models"""
        if not scores:
            return 0.0
        
        # Weight different models
        weights = {
            'transformer': 0.5,
            'textblob': 0.3,
            'keyword': 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model, score in scores.items():
            weight = weights.get(model, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _determine_sentiment_type(self, score: float) -> SentimentType:
        """Determine sentiment type based on score"""
        if score > 0.1:
            return SentimentType.POSITIVE
        elif score < -0.1:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        keywords = []
        
        # Extract keywords from all categories
        for category, words in self.keywords.items():
            for word in words:
                if word in text:
                    keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (simplified)"""
        entities = []
        
        # Extract cryptocurrency symbols
        crypto_pattern = r'\b[A-Z]{2,10}\b'
        crypto_matches = re.findall(crypto_pattern, text.upper())
        entities.extend(crypto_matches)
        
        # Extract numbers (prices, percentages)
        number_pattern = r'\$\d+\.?\d*|\d+%|\d+\.?\d*'
        number_matches = re.findall(number_pattern, text)
        entities.extend(number_matches)
        
        return list(set(entities))
    
    def _calculate_confidence(self, scores: Dict[str, float], final_score: float) -> float:
        """Calculate confidence in the sentiment analysis"""
        if not scores:
            return 0.0
        
        # Base confidence on agreement between models
        score_values = list(scores.values())
        if len(score_values) < 2:
            return 0.5
        
        # Calculate variance (lower variance = higher confidence)
        variance = np.var(score_values)
        confidence = max(0.0, 1.0 - variance)
        
        # Boost confidence for extreme scores
        if abs(final_score) > 0.7:
            confidence = min(1.0, confidence + 0.2)
        
        return confidence

class NewsProcessor:
    """Advanced news processing and analysis"""
    
    def __init__(self):
        self.news_api_key = None  # Will be set from environment
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.relevance_keywords = [
            'cryptocurrency', 'bitcoin', 'ethereum', 'crypto', 'blockchain',
            'trading', 'market', 'price', 'volume', 'bull', 'bear',
            'regulation', 'adoption', 'institutional', 'defi', 'nft'
        ]
        
        logger.info("News Processor initialized")
    
    async def fetch_news(self, query: str, days: int = 1) -> List[NewsArticle]:
        """Fetch news articles (mock implementation)"""
        try:
            # Mock news articles for demonstration
            mock_articles = [
                {
                    'title': f'{query} shows bullish momentum as institutional adoption increases',
                    'content': f'Recent developments in {query} indicate strong institutional interest...',
                    'url': f'https://example.com/news/{query.lower()}',
                    'source': 'CryptoNews',
                    'published_at': datetime.now() - timedelta(hours=2)
                },
                {
                    'title': f'{query} faces resistance at key levels amid market uncertainty',
                    'content': f'Technical analysis suggests {query} is struggling to break through...',
                    'url': f'https://example.com/analysis/{query.lower()}',
                    'source': 'TradingView',
                    'published_at': datetime.now() - timedelta(hours=4)
                }
            ]
            
            articles = []
            for article_data in mock_articles:
                # Analyze sentiment
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                    f"{article_data['title']} {article_data['content']}",
                    NLPSource.NEWS
                )
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance(
                    f"{article_data['title']} {article_data['content']}"
                )
                
                article = NewsArticle(
                    title=article_data['title'],
                    content=article_data['content'],
                    url=article_data['url'],
                    source=article_data['source'],
                    published_at=article_data['published_at'],
                    sentiment_score=sentiment_result.sentiment_score,
                    keywords=sentiment_result.keywords,
                    relevance_score=relevance_score
                )
                articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score for news article"""
        text_lower = text.lower()
        relevant_keywords = sum(1 for keyword in self.relevance_keywords if keyword in text_lower)
        return min(1.0, relevant_keywords / len(self.relevance_keywords))

class SocialMediaAnalyzer:
    """Social media sentiment analysis"""
    
    def __init__(self):
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.twitter_api = None
        self.reddit_api = None
        
        self._initialize_apis()
        logger.info("Social Media Analyzer initialized")
    
    def _initialize_apis(self):
        """Initialize social media APIs"""
        try:
            # Initialize Twitter API (if credentials available)
            if TWEEPY_AVAILABLE:
                # This would be configured with actual credentials
                pass
            
            # Initialize Reddit API (if credentials available)
            if PRAW_AVAILABLE:
                # This would be configured with actual credentials
                pass
                
        except Exception as e:
            logger.error(f"Error initializing social media APIs: {e}")
    
    async def analyze_twitter_sentiment(self, query: str, max_tweets: int = 100) -> List[SentimentResult]:
        """Analyze Twitter sentiment (mock implementation)"""
        try:
            # Mock Twitter data
            mock_tweets = [
                f"$BTC looking bullish! 🚀 #crypto #bitcoin",
                f"Market is bearish today, {query} dropping hard 📉",
                f"Great analysis on {query}, fundamentals are strong 💪",
                f"Not sure about {query}, market seems uncertain 🤔"
            ]
            
            results = []
            for tweet in mock_tweets:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                    tweet, NLPSource.TWITTER
                )
                results.append(sentiment_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Twitter sentiment analysis error: {e}")
            return []
    
    async def analyze_reddit_sentiment(self, subreddit: str, query: str, max_posts: int = 50) -> List[SentimentResult]:
        """Analyze Reddit sentiment (mock implementation)"""
        try:
            # Mock Reddit data
            mock_posts = [
                f"Bullish on {query}! The fundamentals are solid and adoption is growing.",
                f"Bearish sentiment on {query}. Technical indicators show weakness.",
                f"Neutral on {query}. Waiting for more confirmation signals.",
                f"Mixed feelings about {query}. Some good news, some concerns."
            ]
            
            results = []
            for post in mock_posts:
                sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                    post, NLPSource.REDDIT
                )
                results.append(sentiment_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis error: {e}")
            return []

class NaturalLanguageProcessingEngine:
    """Main NLP engine for AlphaPulse"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NLP engine"""
        self.config = config or {}
        self.is_running = False
        
        # NLP components
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.news_processor = NewsProcessor()
        self.social_analyzer = SocialMediaAnalyzer()
        
        # Data storage
        self.sentiment_cache = {}
        self.news_cache = {}
        self.social_cache = {}
        
        # Performance tracking
        self.analyses_performed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Configuration
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        self.update_frequency = self.config.get('update_frequency', 300)  # 5 minutes
        
        logger.info("🚀 Natural Language Processing Engine initialized")
    
    async def start(self):
        """Start the NLP engine"""
        if self.is_running:
            logger.warning("NLP engine already running")
            return
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._background_analysis())
        
        logger.info("✅ NLP engine started successfully")
    
    async def stop(self):
        """Stop the NLP engine"""
        self.is_running = False
        logger.info("🛑 NLP engine stopped")
    
    async def _background_analysis(self):
        """Background analysis loop"""
        while self.is_running:
            try:
                # Update sentiment for major cryptocurrencies
                symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
                for symbol in symbols:
                    await self._update_symbol_sentiment(symbol)
                
                # Sleep between updates
                await asyncio.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Background analysis error: {e}")
                await asyncio.sleep(60)
    
    async def _update_symbol_sentiment(self, symbol: str):
        """Update sentiment for a specific symbol"""
        try:
            # Analyze news sentiment
            news_articles = await self.news_processor.fetch_news(symbol, days=1)
            news_sentiment = self._aggregate_news_sentiment(news_articles)
            
            # Analyze social media sentiment
            twitter_sentiment = await self.social_analyzer.analyze_twitter_sentiment(symbol)
            reddit_sentiment = await self.social_analyzer.analyze_reddit_sentiment('cryptocurrency', symbol)
            
            # Aggregate all sentiment
            aggregated_sentiment = self._aggregate_all_sentiment(
                news_sentiment, twitter_sentiment, reddit_sentiment
            )
            
            # Cache results
            self.sentiment_cache[symbol] = {
                'sentiment': aggregated_sentiment,
                'news_articles': news_articles,
                'twitter_sentiment': twitter_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'timestamp': datetime.now()
            }
            
            self.analyses_performed += 1
            logger.debug(f"Updated sentiment for {symbol}: {aggregated_sentiment['overall_score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating sentiment for {symbol}: {e}")
    
    def _aggregate_news_sentiment(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Aggregate news sentiment"""
        if not articles:
            return {'overall_score': 0.0, 'confidence': 0.0, 'article_count': 0}
        
        scores = [article.sentiment_score for article in articles]
        confidences = [article.relevance_score for article in articles]
        
        # Weight by relevance
        weighted_score = sum(score * conf for score, conf in zip(scores, confidences))
        total_weight = sum(confidences)
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        confidence = np.mean(confidences)
        
        return {
            'overall_score': overall_score,
            'confidence': confidence,
            'article_count': len(articles),
            'articles': articles
        }
    
    def _aggregate_all_sentiment(self, news_sentiment: Dict, 
                                twitter_sentiment: List[SentimentResult],
                                reddit_sentiment: List[SentimentResult]) -> Dict[str, Any]:
        """Aggregate sentiment from all sources"""
        
        # Calculate Twitter sentiment
        twitter_scores = [result.sentiment_score for result in twitter_sentiment]
        twitter_confidence = np.mean([result.confidence for result in twitter_sentiment]) if twitter_sentiment else 0.0
        
        # Calculate Reddit sentiment
        reddit_scores = [result.sentiment_score for result in reddit_sentiment]
        reddit_confidence = np.mean([result.confidence for result in reddit_sentiment]) if reddit_sentiment else 0.0
        
        # Aggregate with weights
        weights = {
            'news': 0.4,
            'twitter': 0.35,
            'reddit': 0.25
        }
        
        news_score = news_sentiment.get('overall_score', 0.0)
        news_confidence = news_sentiment.get('confidence', 0.0)
        
        twitter_score = np.mean(twitter_scores) if twitter_scores else 0.0
        reddit_score = np.mean(reddit_scores) if reddit_scores else 0.0
        
        # Weighted average
        overall_score = (
            news_score * weights['news'] * news_confidence +
            twitter_score * weights['twitter'] * twitter_confidence +
            reddit_score * weights['reddit'] * reddit_confidence
        ) / (
            weights['news'] * news_confidence +
            weights['twitter'] * twitter_confidence +
            weights['reddit'] * reddit_confidence
        )
        
        overall_confidence = (
            news_confidence * weights['news'] +
            twitter_confidence * weights['twitter'] +
            reddit_confidence * weights['reddit']
        )
        
        return {
            'overall_score': overall_score,
            'overall_confidence': overall_confidence,
            'news_score': news_score,
            'news_confidence': news_confidence,
            'twitter_score': twitter_score,
            'twitter_confidence': twitter_confidence,
            'reddit_score': reddit_score,
            'reddit_confidence': reddit_confidence,
            'timestamp': datetime.now()
        }
    
    def get_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached sentiment for a symbol"""
        if symbol in self.sentiment_cache:
            cache_entry = self.sentiment_cache[symbol]
            
            # Check if cache is still valid
            if (datetime.now() - cache_entry['timestamp']).seconds < self.cache_ttl:
                self.cache_hits += 1
                return cache_entry['sentiment']
        
        self.cache_misses += 1
        return None
    
    def analyze_text_sentiment(self, text: str, source: NLPSource = NLPSource.NEWS) -> SentimentResult:
        """Analyze sentiment of arbitrary text"""
        return self.sentiment_analyzer.analyze_sentiment(text, source)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get NLP performance summary"""
        return {
            'is_running': self.is_running,
            'analyses_performed': self.analyses_performed,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            'cached_symbols': list(self.sentiment_cache.keys()),
            'models_available': {
                'textblob': TEXTBLOB_AVAILABLE,
                'transformers': TRANSFORMERS_AVAILABLE,
                'tweepy': TWEEPY_AVAILABLE,
                'praw': PRAW_AVAILABLE
            }
        }
