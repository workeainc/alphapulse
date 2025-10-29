#!/usr/bin/env python3
"""
Enhanced News and Event Processing Service for AlphaPlus
Advanced news processing with multi-language support, event correlation, and impact analysis
"""

import asyncio
import logging
import aiohttp
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncpg
from dataclasses import dataclass
import re
from textblob import TextBlob
import uuid
import hashlib
from tenacity import retry, wait_exponential, stop_after_attempt
import feedparser
from urllib.parse import urlparse
import time
import spacy
import nltk
from rapidfuzz import fuzz
import json

# Import ML models
from .ml_models import NewsMLModels, ModelPrediction

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    description: str
    content: str
    url: str
    source: str
    author: str
    published_at: datetime
    language: str = 'en'
    category: str = ''
    tags: List[str] = None

class EnhancedNewsEventProcessor:
    """Enhanced news and event processing service"""
    
    def __init__(self, db_pool: asyncpg.Pool, config_path: str = None):
        self.db_pool = db_pool
        self.session = None
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize from config
        self.news_api_key = self.config['news_api']['api_key']
        self.news_base_url = self.config['news_api']['base_url']
        self.breaking_news_threshold = self.config['processing']['breaking_news_threshold']
        
        # Keywords for classification
        self.crypto_keywords = self.config['keywords']['crypto']
        self.economic_keywords = self.config['keywords']['economic']
        self.breaking_indicators = self.config['keywords']['breaking_indicators']
        
        # Credible sources
        self.credible_sources = self.config['sources']['credible_sources']
        self.crypto_sources = self.config['sources']['crypto_sources']
        self.cryptopanic_sources = self.config['sources']['cryptopanic_sources']
        
        # Social metrics configuration
        self.social_metrics_config = self.config.get('social_metrics', {})
        self.correlation_config = self.config.get('correlation', {})
        
        # API configurations
        self.cryptopanic_config = self.config.get('cryptopanic_api', {})
        self.santiment_config = self.config.get('santiment_api', {})
        self.rss_config = self.config.get('rss_feeds', {})
        
        # Enhanced intelligence configurations
        self.entity_config = self.config.get('entity_recognition', {})
        self.latency_config = self.config.get('latency_tracking', {})
        self.cross_source_config = self.config.get('cross_source_validation', {})
        self.market_correlation_config = self.config.get('market_correlation', {})
        self.feed_reliability_config = self.config.get('feed_reliability', {})
        self.automated_classification_config = self.config.get('automated_classification', {})
        self.price_data_config = self.config.get('price_data_integration', {})
        self.sentiment_normalization_config = self.config.get('sentiment_normalization', {})
        self.market_context_config = self.config.get('market_context', {})
        self.ml_config = self.config.get('machine_learning', {})
        self.advanced_correlation_config = self.config.get('advanced_correlation', {})
        self.real_time_alerts_config = self.config.get('real_time_alerts', {})
        
        # Initialize NLP models
        self.nlp = None
        if self.entity_config.get('enabled', False):
            try:
                self.nlp = spacy.load(self.entity_config.get('spacy_model', 'en_core_web_sm'))
                logger.info(f"✅ Loaded spaCy model: {self.entity_config.get('spacy_model', 'en_core_web_sm')}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load spaCy model: {e}")
        
        # Initialize ML models
        self.ml_models = None
        if self.ml_config.get('enabled', False):
            try:
                self.ml_models = NewsMLModels(self.config, self.db_pool)
                asyncio.create_task(self.ml_models.initialize_models())
                logger.info("✅ ML models initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize ML models: {e}")
        
        logger.info("Enhanced News and Event Processor initialized with API, RSS, and Intelligence integrations")
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from file"""
        try:
            if config_path is None:
                config_path = "config/enhanced_news_config.json"
            
            import json
            import os
            from pathlib import Path
            
            # Try to find config file
            possible_paths = [
                config_path,
                Path(__file__).parent.parent.parent / "config" / "enhanced_news_config.json",
                Path(__file__).parent.parent / "config" / "enhanced_news_config.json"
            ]
            
            config_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_file = path
                    break
            
            if config_file:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    return config['enhanced_news_events']
            else:
                # Return default config if file not found
                logger.warning("⚠️ Enhanced news config file not found, using default configuration")
                return {
                    'news_api': {
                        'api_key': 'your_news_api_key',
                        'base_url': 'https://newsapi.org/v2'
                    },
                    'processing': {
                        'breaking_news_threshold': 0.8
                    },
                    'keywords': {
                        'crypto': ['bitcoin', 'ethereum', 'cryptocurrency', 'blockchain', 'defi', 'nft'],
                        'economic': ['fomc', 'cpi', 'nfp', 'gdp', 'fed', 'ecb', 'boe', 'interest rate'],
                        'breaking_indicators': ['breaking', 'urgent', 'just in', 'exclusive', 'flash']
                    },
                    'sources': {
                        'credible_sources': ['reuters', 'bloomberg', 'cnbc', 'financial times', 'wall street journal'],
                        'crypto_sources': ['coindesk', 'cointelegraph', 'the block']
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            # Return minimal default config
            return {
                'news_api': {'api_key': 'your_news_api_key', 'base_url': 'https://newsapi.org/v2'},
                'processing': {'breaking_news_threshold': 0.8},
                'keywords': {
                    'crypto': ['bitcoin', 'ethereum', 'cryptocurrency'],
                    'economic': ['fomc', 'cpi', 'fed'],
                    'breaking_indicators': ['breaking', 'urgent']
                },
                'sources': {
                    'credible_sources': ['reuters', 'bloomberg'],
                    'crypto_sources': ['coindesk', 'cointelegraph']
                }
            }
    
    async def initialize(self):
        """Initialize the processor"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info("✅ Enhanced News and Event Processor initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing processor: {e}")
            raise
    
    async def process_comprehensive_news_events(self) -> Dict[str, Any]:
        """Process comprehensive news and events data"""
        try:
            logger.info("🔄 Processing comprehensive news and events...")
            
            # Collect and process news
            news_articles = await self.collect_news_data()
            processed_news = await self.process_news_articles(news_articles)
            
            # Detect breaking news
            breaking_news = await self.detect_breaking_news(processed_news)
            
            # Store data
            await self.store_news_data(processed_news, breaking_news)
            
            logger.info("✅ Comprehensive news and events processed successfully")
            
            return {
                'news_articles': len(processed_news),
                'breaking_news': len(breaking_news)
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing comprehensive news and events: {e}")
            raise
    
    async def collect_news_data(self) -> List[NewsArticle]:
        """Collect news data from multiple sources"""
        try:
            news_articles = []
            
            # Collect from NewsAPI (if configured)
            if self.news_api_key != "your_news_api_key":
                try:
                    articles = await self._fetch_newsapi_articles("cryptocurrency OR bitcoin OR ethereum")
                    for article in articles:
                        news_article = NewsArticle(
                            title=article.get('title', ''),
                            description=article.get('description', ''),
                            content=article.get('content', ''),
                            url=article.get('url', ''),
                            source=article.get('source', {}).get('name', ''),
                            author=article.get('author', ''),
                            published_at=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                            category=self._classify_news_category(article),
                            tags=self._extract_tags(article)
                        )
                        news_articles.append(news_article)
                    logger.info(f"✅ Collected {len(articles)} articles from NewsAPI")
                except Exception as e:
                    logger.warning(f"⚠️ NewsAPI collection failed: {e}")
            
            # Collect from CryptoPanic
            try:
                cryptopanic_articles = await self._fetch_cryptopanic_articles()
                for article in cryptopanic_articles:
                    news_article = NewsArticle(
                        title=article.get('title', ''),
                        description=article.get('summary', ''),
                        content=article.get('summary', ''),
                        url=article.get('url', ''),
                        source=article.get('source', {}).get('title', 'cryptopanic'),
                        author='',
                        published_at=datetime.fromisoformat(article.get('published_at', '').replace('Z', '+00:00')),
                        category=self._classify_news_category(article),
                        tags=self._extract_cryptopanic_tags(article)
                    )
                    news_articles.append(news_article)
                logger.info(f"✅ Collected {len(cryptopanic_articles)} articles from CryptoPanic")
            except Exception as e:
                logger.warning(f"⚠️ CryptoPanic collection failed: {e}")
            
            # Collect from RSS Feeds
            if self.rss_config.get('enabled', False):
                try:
                    rss_articles = await self._fetch_rss_articles()
                    for article in rss_articles:
                        news_article = NewsArticle(
                            title=article.get('title', ''),
                            description=article.get('summary', ''),
                            content=article.get('content', article.get('summary', '')),
                            url=article.get('link', ''),
                            source=article.get('feed_name', 'rss'),
                            author=article.get('author', ''),
                            published_at=article.get('published_parsed', datetime.utcnow()),
                            category=article.get('category', 'general'),
                            tags=self._extract_rss_tags(article)
                        )
                        # Add RSS-specific attributes
                        news_article.rss_feed_url = article.get('feed_url', '')
                        news_article.rss_feed_name = article.get('feed_name', '')
                        news_article.rss_category = article.get('category', '')
                        news_article.rss_guid = article.get('guid', '')
                        news_article.feed_credibility = article.get('credibility', 0.5)
                        news_article.rss_priority_level = article.get('priority', 'medium')
                        
                        # Enhanced intelligence processing
                        if self.entity_config.get('enabled', False):
                            entities, event_types, confidence = self._extract_entities_and_events(news_article)
                            news_article.entities = entities
                            news_article.event_types = event_types
                            news_article.entity_confidence = confidence
                        
                        # Latency tracking
                        if self.latency_config.get('enabled', False):
                            latency_ms = self._calculate_publish_latency(news_article, article.get('published_parsed'))
                            news_article.publish_latency_ms = latency_ms
                        
                        news_articles.append(news_article)
                    logger.info(f"✅ Collected {len(rss_articles)} articles from RSS feeds")
                except Exception as e:
                    logger.warning(f"⚠️ RSS collection failed: {e}")
            
            # If no articles collected, use mock data
            if not news_articles:
                logger.info("📝 Using mock news articles")
                news_articles = self._generate_mock_news_articles()
            
            # Enhanced intelligence: Cross-source validation and clustering
            if self.cross_source_config.get('enabled', False) and news_articles:
                logger.info("🔍 Performing cross-source validation and clustering...")
                
                # First pass: Cross-source validation
                for article in news_articles:
                    is_validated, validation_sources, similarity_score = self._check_cross_source_validation(article, news_articles)
                    article.cross_source_validation = is_validated
                    article.validation_sources = validation_sources
                    article.similarity_score = similarity_score
                    
                    if is_validated:
                        logger.debug(f"✅ Cross-source validation: {article.title[:50]}... validated by {len(validation_sources)} sources")
                
                # Second pass: Advanced clustering and deduplication
                clustered_articles = self._perform_advanced_clustering(news_articles)
                logger.info(f"📊 Created {len(set(a.dup_group_id for a in clustered_articles if a.dup_group_id))} news clusters from {len(news_articles)} articles")
                
                # Third pass: Market context and correlation analysis
                if self.price_data_config.get('enabled', False) or self.market_context_config.get('enabled', False):
                    logger.info("📈 Adding market context and correlation analysis...")
                    
                    # Fetch real-time price data
                    price_data = await self._fetch_real_time_prices()
                    if price_data:
                        await self._store_price_data(price_data)
                        logger.info(f"📊 Fetched and stored price data for {len(price_data)} symbols")
                    
                    # Detect market regime
                    market_regime = await self._detect_market_regime()
                    logger.info(f"📊 Market regime detected: {market_regime['regime']} (confidence: {market_regime['confidence']:.2f})")
                    
                    # Add market context to articles
                    for article in clustered_articles:
                        # Add market regime
                        article.market_regime = market_regime['regime']
                        article.btc_dominance = market_regime.get('btc_dominance', 48.5)
                        article.market_volatility = market_regime.get('market_volatility', 0.025)
                        article.fear_greed_index = market_regime.get('fear_greed_index', 55)
                        
                        # Calculate price correlations
                        correlations = await self._calculate_price_correlation(article.published_at)
                        article.correlation_30m = correlations.get('correlation_30m', 0.0)
                        article.correlation_2h = correlations.get('correlation_2h', 0.0)
                        article.correlation_24h = correlations.get('correlation_24h', 0.0)
                        
                        # Normalize sentiment
                        if hasattr(article, 'sentiment_score'):
                            source = getattr(article, 'source', 'rss')
                            normalized = self._normalize_sentiment(article.sentiment_score, source)
                            article.normalized_sentiment = normalized['normalized_sentiment']
                            article.sentiment_confidence = normalized['confidence']
                
                # Fourth pass: Machine Learning and Advanced Analytics
                if self.ml_config.get('enabled', False) or self.advanced_correlation_config.get('enabled', False):
                    logger.info("🤖 Adding machine learning and advanced analytics...")
                    
                    for article in clustered_articles:
                        # Engineer ML features
                        features = await self._engineer_ml_features(article)
                        if features:
                            logger.debug(f"🔧 Engineered {len(features)} ML features for article")
                        
                        # Generate ML predictions
                        predictions = await self._generate_ml_predictions(article, features)
                        if predictions:
                            logger.debug(f"🤖 Generated {len(predictions)} ML predictions for article")
                        
                        # Calculate advanced correlation analysis
                        correlations = await self._calculate_advanced_correlation(article)
                        if correlations:
                            logger.debug(f"📊 Calculated advanced correlations for article")
                        
                        # Generate real-time alerts
                        alerts = await self._generate_real_time_alerts(article, predictions, correlations)
                        if alerts:
                            logger.info(f"🚨 Generated {len(alerts)} real-time alerts for article")
                
                return clustered_articles
            
            return news_articles
            
        except Exception as e:
            logger.error(f"❌ Error collecting news data: {e}")
            return self._generate_mock_news_articles()
    
    def _generate_mock_news_articles(self) -> List[NewsArticle]:
        """Generate mock news articles for testing"""
        return [
            NewsArticle(
                title="Bitcoin Surges Past $50,000 as Institutional Adoption Increases",
                description="Bitcoin reaches new highs as major institutions announce crypto investments",
                content="Bitcoin has surged past the $50,000 mark...",
                url="https://example.com/bitcoin-surge",
                source="CryptoNews",
                author="John Doe",
                published_at=datetime.utcnow() - timedelta(hours=2),
                category="crypto",
                tags=["bitcoin", "price", "institutional"]
            ),
            NewsArticle(
                title="Federal Reserve Signals Potential Rate Cut in Next Meeting",
                description="Fed officials hint at possible interest rate reduction",
                content="Federal Reserve officials have indicated...",
                url="https://example.com/fed-rate-cut",
                source="FinancialTimes",
                author="Jane Smith",
                published_at=datetime.utcnow() - timedelta(hours=1),
                category="economic",
                tags=["fed", "interest rates", "monetary policy"]
            )
        ]
    
    async def _fetch_newsapi_articles(self, query: str) -> List[Dict]:
        """Fetch articles from NewsAPI"""
        try:
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': self.news_api_key
            }
            
            async with self.session.get(f"{self.news_base_url}/everything", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('articles', [])
                else:
                    logger.warning(f"⚠️ NewsAPI request failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching NewsAPI articles: {e}")
            return []
    
    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _fetch_cryptopanic_articles(self) -> List[Dict]:
        """Fetch articles from CryptoPanic API"""
        try:
            params = {
                'auth_token': self.cryptopanic_config.get('api_key', ''),
                'currencies': self.cryptopanic_config.get('default_currencies', 'BTC,ETH'),
                'kind': self.cryptopanic_config.get('default_kind', 'news'),
                'filter': self.cryptopanic_config.get('default_filter', 'rising'),
                'page': 1
            }
            
            url = f"{self.cryptopanic_config.get('base_url', 'https://cryptopanic.com/api/v1')}/posts/"
            
            async with self.session.get(url, params=params, timeout=self.cryptopanic_config.get('timeout_seconds', 20)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('results', [])
                elif response.status == 429:
                    logger.warning("⚠️ CryptoPanic rate limit hit, retrying...")
                    raise Exception("Rate limit")
                else:
                    logger.warning(f"⚠️ CryptoPanic request failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching CryptoPanic articles: {e}")
            return []
    
    @retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
    async def _fetch_santiment_social_volume(self, slug: str, from_iso: str, to_iso: str, interval: str = "5m") -> Dict:
        """Fetch social volume data from Santiment GraphQL API"""
        try:
            query = """
            query($slug: String!, $from: DateTime!, $to: DateTime!, $interval: String!) {
              getMetric(metric: "social_volume_total"){
                timeseriesData(
                  slug: $slug, from: $from, to: $to, interval: $interval){
                  datetime
                  value
                }
              }
            }
            """
            
            variables = {
                "slug": slug,
                "from": from_iso,
                "to": to_iso,
                "interval": interval
            }
            
            headers = {
                "Authorization": f"Apikey {self.santiment_config.get('api_key', '')}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                self.santiment_config.get('base_url', 'https://api.santiment.net/graphql'),
                json={"query": query, "variables": variables},
                headers=headers,
                timeout=self.santiment_config.get('timeout_seconds', 30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('getMetric', {}).get('timeseriesData', [])
                else:
                    logger.warning(f"⚠️ Santiment request failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching Santiment social volume: {e}")
            return []
    
    def _extract_cryptopanic_tags(self, article: Dict) -> List[str]:
        """Extract tags from CryptoPanic article"""
        try:
            tags = []
            
            # Add currencies mentioned
            currencies = article.get('currencies', [])
            for currency in currencies:
                if isinstance(currency, dict):
                    tags.append(currency.get('code', '').lower())
                else:
                    tags.append(str(currency).lower())
            
            # Add labels
            labels = article.get('labels', [])
            for label in labels:
                if isinstance(label, dict):
                    tags.append(label.get('title', '').lower())
                else:
                    tags.append(str(label).lower())
            
            return tags
            
        except Exception as e:
            logger.error(f"❌ Error extracting CryptoPanic tags: {e}")
            return []
    
    def _make_article_id(self, source: str, title: str, published_at: str, url: str) -> str:
        """Generate deterministic article ID for deduplication"""
        raw = f"{source}|{title}|{published_at}|{url}"
        return hashlib.sha256(raw.encode()).hexdigest()
    
    async def _fetch_rss_articles(self) -> List[Dict]:
        """Fetch articles from RSS feeds"""
        try:
            all_rss_articles = []
            
            # Get all enabled feeds from configuration
            feeds = []
            for priority_level in ['high_priority', 'medium_priority', 'specialized']:
                priority_feeds = self.rss_config.get('feeds', {}).get(priority_level, [])
                for feed in priority_feeds:
                    if feed.get('enabled', False):
                        feed['priority'] = priority_level
                        feeds.append(feed)
            
            logger.info(f"📰 Processing {len(feeds)} RSS feeds...")
            
            # Process feeds concurrently but with controlled parallelism
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent feeds
            tasks = [self._fetch_single_rss_feed(feed, semaphore) for feed in feeds]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ RSS feed failed: {result}")
                elif isinstance(result, list):
                    all_rss_articles.extend(result)
            
            logger.info(f"✅ Collected {len(all_rss_articles)} total RSS articles from {len(feeds)} feeds")
            return all_rss_articles
            
        except Exception as e:
            logger.error(f"❌ Error fetching RSS articles: {e}")
            return []
    
    async def _fetch_single_rss_feed(self, feed_config: Dict, semaphore: asyncio.Semaphore) -> List[Dict]:
        """Fetch articles from a single RSS feed"""
        async with semaphore:
            feed_name = feed_config.get('name', 'Unknown')
            feed_url = feed_config.get('url', '')
            
            try:
                start_time = time.time()
                
                # Use feedparser in executor to avoid blocking
                loop = asyncio.get_event_loop()
                feed_data = await loop.run_in_executor(
                    None, 
                    lambda: feedparser.parse(feed_url)
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if feed_data.bozo:
                    logger.warning(f"⚠️ RSS feed {feed_name} has parsing issues: {feed_data.bozo_exception}")
                
                articles = []
                max_articles = self.rss_config.get('max_articles_per_feed', 50)
                
                for entry in feed_data.entries[:max_articles]:
                    try:
                        # Parse published date
                        published_parsed = datetime.utcnow()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published_parsed = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published_parsed = datetime(*entry.updated_parsed[:6])
                        
                        # Extract content
                        content = ''
                        if hasattr(entry, 'content') and entry.content:
                            content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                        elif hasattr(entry, 'description'):
                            content = entry.description
                        
                        article = {
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', ''),
                            'content': content,
                            'link': entry.get('link', ''),
                            'author': entry.get('author', ''),
                            'published_parsed': published_parsed,
                            'guid': entry.get('id', entry.get('guid', '')),
                            'feed_name': feed_name,
                            'feed_url': feed_url,
                            'category': feed_config.get('category', 'general'),
                            'credibility': feed_config.get('credibility', 0.5),
                            'priority': feed_config.get('priority', 'medium')
                        }
                        
                        # Check if article is too old (for backfill control)
                        hours_old = (datetime.utcnow() - published_parsed).total_seconds() / 3600
                        backfill_hours = self.rss_config.get('backfill_hours', 24)
                        
                        if hours_old <= backfill_hours:
                            articles.append(article)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing RSS entry from {feed_name}: {e}")
                        continue
                
                # Calculate and update feed reliability score
                reliability_score = await self._calculate_feed_reliability_score(feed_name)
                
                # Log feed status with reliability
                await self._log_rss_feed_status(
                    feed_name, feed_url, 'success', 
                    len(feed_data.entries), len(articles), 0, response_time
                )
                
                # Store feed reliability score
                await self._store_feed_reliability_score(feed_name, reliability_score, response_time, len(articles))
                
                logger.info(f"✅ RSS {feed_name}: {len(articles)} articles ({response_time:.1f}ms)")
                return articles
                
            except Exception as e:
                logger.error(f"❌ Error fetching RSS feed {feed_name}: {e}")
                
                # Log feed error status
                await self._log_rss_feed_status(
                    feed_name, feed_url, 'error', 0, 0, 0, 0.0, str(e)
                )
                
                return []
    
    async def _log_rss_feed_status(self, feed_name: str, feed_url: str, status: str, 
                                   articles_collected: int, articles_processed: int, 
                                   articles_duplicates: int, response_time_ms: float, 
                                   error_message: str = None):
        """Log RSS feed status to database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rss_feed_status (
                        timestamp, feed_name, feed_url, status, articles_collected,
                        articles_processed, articles_duplicates, response_time_ms,
                        error_message, last_successful_fetch
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (timestamp, id) DO NOTHING;
                """, 
                    datetime.utcnow(), feed_name, feed_url, status, articles_collected,
                    articles_processed, articles_duplicates, response_time_ms,
                    error_message, datetime.utcnow() if status == 'success' else None
                )
        except Exception as e:
            logger.warning(f"⚠️ Failed to log RSS feed status: {e}")
    
    async def _store_feed_reliability_score(self, feed_name: str, reliability_score: float, 
                                           response_time: float, article_count: int):
        """Store feed reliability score in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO feed_reliability_scores (
                        timestamp, feed_name, reliability_score, response_time_score,
                        success_rate_score, prediction_accuracy_score, article_quality_score,
                        total_articles, avg_response_time_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (timestamp, id) DO NOTHING;
                """, 
                    datetime.utcnow(), feed_name, reliability_score,
                    max(0.0, min(1.0, (3000.0 - response_time) / 3000.0)),  # response_time_score
                    0.9,  # success_rate_score (default)
                    0.7,  # prediction_accuracy_score (default)
                    0.8,  # article_quality_score (default)
                    article_count, response_time
                )
                logger.debug(f"📊 Stored reliability score {reliability_score:.3f} for {feed_name}")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to store feed reliability score: {e}")
    
    def _extract_rss_tags(self, article: Dict) -> List[str]:
        """Extract tags from RSS article"""
        try:
            tags = []
            
            # Add feed category
            if article.get('category'):
                tags.append(article['category'])
            
            # Add priority level
            if article.get('priority'):
                tags.append(article['priority'])
            
            # Extract from title and summary
            text = f"{article.get('title', '')} {article.get('summary', '')}".lower()
            words = re.findall(r'\b\w+\b', text)
            
            # Add relevant keywords
            relevant_words = [word for word in words if len(word) > 3 and word in self.crypto_keywords + self.economic_keywords]
            tags.extend(relevant_words[:5])  # Limit to 5 most relevant
            
            return list(set(tags))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"❌ Error extracting RSS tags: {e}")
            return []
    
    def _extract_entities_and_events(self, article: NewsArticle) -> Tuple[List[Dict], List[str], float]:
        """Extract entities and event types from article using NLP"""
        try:
            entities = []
            event_types = []
            confidence = 0.0
            
            if not self.nlp:
                return entities, event_types, confidence
            
            # Combine title and content for analysis
            text = f"{article.title} {article.description} {article.content}"
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT']:
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': ent.prob if hasattr(ent, 'prob') else 0.8
                    }
                    entities.append(entity_info)
            
            # Extract crypto-specific entities from config
            crypto_entities = self.entity_config.get('crypto_entities', [])
            event_types_list = self.entity_config.get('event_types', [])
            
            text_lower = text.lower()
            
            # Check for crypto entities
            for entity in crypto_entities:
                if entity.lower() in text_lower:
                    entity_info = {
                        'text': entity,
                        'label': 'CRYPTO',
                        'start': text_lower.find(entity.lower()),
                        'end': text_lower.find(entity.lower()) + len(entity),
                        'confidence': 0.9
                    }
                    entities.append(entity_info)
            
            # Check for event types
            for event_type in event_types_list:
                if event_type.lower() in text_lower:
                    event_types.append(event_type)
            
            # Calculate confidence based on entity count and quality
            if entities:
                confidence = min(0.9, 0.3 + (len(entities) * 0.1) + (len(event_types) * 0.05))
            
            # Limit entities per article
            max_entities = self.entity_config.get('max_entities_per_article', 10)
            entities = entities[:max_entities]
            
            return entities, event_types, confidence
            
        except Exception as e:
            logger.error(f"❌ Error extracting entities and events: {e}")
            return [], [], 0.0
    
    def _calculate_publish_latency(self, article: NewsArticle, published_parsed: datetime) -> float:
        """Calculate latency between publish time and current time"""
        try:
            if not published_parsed:
                return 0.0
            
            # Calculate latency in milliseconds
            current_time = datetime.utcnow()
            latency = (current_time - published_parsed).total_seconds() * 1000
            
            return max(0.0, latency)
            
        except Exception as e:
            logger.error(f"❌ Error calculating publish latency: {e}")
            return 0.0
    
    def _check_cross_source_validation(self, article: NewsArticle, all_articles: List[NewsArticle]) -> Tuple[bool, List[str], float]:
        """Check if article is validated by multiple sources"""
        try:
            if not self.cross_source_config.get('enabled', False):
                return False, [], 0.0
            
            time_window = self.cross_source_config.get('time_window_minutes', 30)
            similarity_threshold = self.cross_source_config.get('similarity_threshold', 0.8)
            minimum_sources = self.cross_source_config.get('minimum_sources', 2)
            
            # Find articles within time window (handle timezone-aware datetime)
            article_published = article.published_at.replace(tzinfo=None) if article.published_at.tzinfo else article.published_at
            window_start = article_published - timedelta(minutes=time_window)
            window_end = article_published + timedelta(minutes=time_window)
            
            similar_articles = []
            validation_sources = []
            
            for other_article in all_articles:
                if other_article == article:
                    continue
                
                # Check if within time window (handle timezone-aware comparison)
                other_published = other_article.published_at.replace(tzinfo=None) if other_article.published_at.tzinfo else other_article.published_at
                if window_start <= other_published <= window_end:
                    # Calculate similarity
                    title_similarity = fuzz.ratio(article.title.lower(), other_article.title.lower()) / 100.0
                    content_similarity = fuzz.ratio(article.description.lower(), other_article.description.lower()) / 100.0
                    
                    avg_similarity = (title_similarity + content_similarity) / 2.0
                    
                    if avg_similarity >= similarity_threshold:
                        similar_articles.append(other_article)
                        validation_sources.append(other_article.source)
            
            # Check if we have enough sources
            is_validated = len(similar_articles) >= minimum_sources
            similarity_score = sum([fuzz.ratio(article.title.lower(), a.title.lower()) / 100.0 for a in similar_articles]) / len(similar_articles) if similar_articles else 0.0
            
            return is_validated, validation_sources, similarity_score
            
        except Exception as e:
            logger.error(f"❌ Error checking cross-source validation: {e}")
            return False, [], 0.0
    
    def _perform_advanced_clustering(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Perform advanced clustering and deduplication using fuzzy matching"""
        try:
            if not articles:
                return articles
            
            # Create similarity matrix
            similarity_threshold = self.cross_source_config.get('similarity_threshold', 0.8)
            clusters = []
            processed_indices = set()
            
            for i, article in enumerate(articles):
                if i in processed_indices:
                    continue
                
                # Start new cluster with this article
                cluster = [i]
                cluster_id = hashlib.sha256(f"cluster_{article.title}_{article.published_at}".encode()).hexdigest()[:16]
                
                # Find similar articles
                for j, other_article in enumerate(articles[i+1:], start=i+1):
                    if j in processed_indices:
                        continue
                    
                    # Calculate comprehensive similarity
                    title_sim = fuzz.ratio(article.title.lower(), other_article.title.lower()) / 100.0
                    desc_sim = fuzz.ratio(article.description.lower(), other_article.description.lower()) / 100.0
                    
                    # Entity-based similarity
                    entity_sim = 0.0
                    if hasattr(article, 'entities') and hasattr(other_article, 'entities'):
                        article_entities = set(e.get('text', '').lower() for e in article.entities if isinstance(e, dict))
                        other_entities = set(e.get('text', '').lower() for e in other_article.entities if isinstance(e, dict))
                        if article_entities and other_entities:
                            entity_sim = len(article_entities & other_entities) / len(article_entities | other_entities)
                    
                    # Weighted similarity score
                    overall_sim = (title_sim * 0.5) + (desc_sim * 0.3) + (entity_sim * 0.2)
                    
                    if overall_sim >= similarity_threshold:
                        cluster.append(j)
                        processed_indices.add(j)
                
                # Mark all articles in cluster
                for idx in cluster:
                    articles[idx].dup_group_id = cluster_id
                    articles[idx].similarity_score = max(articles[idx].similarity_score, 
                                                       len(cluster) / len(articles))
                
                processed_indices.update(cluster)
                clusters.append(cluster)
            
            logger.info(f"🔄 Advanced clustering: {len(clusters)} clusters, {len(processed_indices)} articles clustered")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error performing advanced clustering: {e}")
            return articles
    
    async def _calculate_feed_reliability_score(self, feed_name: str) -> float:
        """Calculate dynamic feed reliability score based on historical performance"""
        try:
            if not self.feed_reliability_config.get('enabled', False):
                return 0.5
            
            async with self.db_pool.acquire() as conn:
                # Get recent performance metrics
                recent_performance = await conn.fetchrow("""
                    SELECT 
                        AVG(response_time_ms) as avg_response_time,
                        COUNT(*) as total_articles,
                        AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(articles_collected::float / NULLIF(articles_processed, 0)) as collection_efficiency
                    FROM rss_feed_status 
                    WHERE feed_name = $1 
                    AND timestamp >= NOW() - INTERVAL '24 hours'
                """, feed_name)
                
                if not recent_performance:
                    return 0.5
                
                # Calculate component scores
                weights = self.feed_reliability_config.get('reliability_factors', {})
                
                # Response time score (inverse - faster is better)
                avg_response_time = float(recent_performance['avg_response_time'] or 1000.0)
                response_time_score = max(0.0, min(1.0, (3000.0 - avg_response_time) / 3000.0))
                
                # Success rate score
                success_rate_score = float(recent_performance['success_rate'] or 0.5)
                
                # Collection efficiency score  
                efficiency_score = min(1.0, float(recent_performance['collection_efficiency'] or 0.5))
                
                # Article quality score (placeholder - will be enhanced with ML)
                quality_score = 0.7  # Default quality score
                
                # Weighted reliability score
                reliability_score = (
                    response_time_score * weights.get('response_time_weight', 0.2) +
                    success_rate_score * weights.get('success_rate_weight', 0.3) +
                    efficiency_score * weights.get('prediction_accuracy_weight', 0.3) +
                    quality_score * weights.get('article_quality_weight', 0.2)
                )
                
                return min(1.0, max(0.0, reliability_score))
                
        except Exception as e:
            logger.error(f"❌ Error calculating feed reliability score: {e}")
            return 0.5
    
    async def _predict_market_impact(self, article: NewsArticle) -> Dict[str, Any]:
        """Predict market impact using historical correlation analysis"""
        try:
            if not self.market_correlation_config.get('enabled', False):
                return {'predicted_impact': 0.0, 'confidence': 0.0, 'impact_class': 'minor'}
            
            # Extract features for prediction
            features = self._extract_prediction_features(article)
            
            # Simple rule-based prediction (will be enhanced with ML)
            impact_score = 0.0
            confidence = 0.5
            
            # Entity-based impact
            if hasattr(article, 'entities') and article.entities:
                high_impact_entities = ['bitcoin', 'ethereum', 'binance', 'coinbase', 'sec', 'fed']
                entity_texts = [e.get('text', '').lower() for e in article.entities if isinstance(e, dict)]
                impact_entities = [e for e in entity_texts if any(h in e for h in high_impact_entities)]
                impact_score += len(impact_entities) * 0.2
            
            # Event type impact
            if hasattr(article, 'event_types') and article.event_types:
                high_impact_events = ['hack', 'etf', 'regulation', 'institutional', 'whale']
                impact_events = [e for e in article.event_types if e.lower() in high_impact_events]
                impact_score += len(impact_events) * 0.3
            
            # Source credibility impact
            if hasattr(article, 'feed_credibility'):
                impact_score *= article.feed_credibility
            
            # Cross-source validation boost
            if hasattr(article, 'cross_source_validation') and article.cross_source_validation:
                impact_score *= 1.5
                confidence += 0.2
            
            # Classify impact
            thresholds = self.market_correlation_config.get('impact_thresholds', {})
            if impact_score >= thresholds.get('extreme', 0.10):
                impact_class = 'extreme'
            elif impact_score >= thresholds.get('major', 0.05):
                impact_class = 'major'
            elif impact_score >= thresholds.get('moderate', 0.03):
                impact_class = 'moderate'
            else:
                impact_class = 'minor'
            
            return {
                'predicted_impact': min(1.0, impact_score),
                'confidence': min(1.0, confidence),
                'impact_class': impact_class,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"❌ Error predicting market impact: {e}")
            return {'predicted_impact': 0.0, 'confidence': 0.0, 'impact_class': 'minor'}
    
    def _extract_prediction_features(self, article: NewsArticle) -> Dict[str, Any]:
        """Extract features for market impact prediction"""
        try:
            features = {
                'title_length': len(article.title),
                'content_length': len(article.content) if article.content else 0,
                'entity_count': len(article.entities) if hasattr(article, 'entities') and article.entities else 0,
                'event_type_count': len(article.event_types) if hasattr(article, 'event_types') and article.event_types else 0,
                'sentiment_score': getattr(article, 'sentiment_score', 0.0),
                'source_credibility': getattr(article, 'feed_credibility', 0.5),
                'cross_validated': getattr(article, 'cross_source_validation', False),
                'publish_latency': getattr(article, 'publish_latency_ms', 0.0),
                'hour_of_day': article.published_at.hour,
                'day_of_week': article.published_at.weekday()
            }
            
            # Add keyword features
            text = f"{article.title} {article.description}".lower()
            
            # High impact keywords
            high_impact_keywords = ['hack', 'breach', 'etf', 'approved', 'rejected', 'regulation', 'ban', 'institutional']
            features['high_impact_keyword_count'] = sum(1 for keyword in high_impact_keywords if keyword in text)
            
            # Market moving phrases
            market_phrases = ['breaking', 'urgent', 'major', 'significant', 'massive', 'huge']
            features['urgency_indicator_count'] = sum(1 for phrase in market_phrases if phrase in text)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting prediction features: {e}")
            return {}
    
    async def _fetch_real_time_prices(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Fetch real-time price data from Binance API"""
        try:
            if not self.price_data_config.get('enabled', False):
                return {}
            
            symbols = symbols or self.price_data_config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
            base_url = self.price_data_config.get('binance_api', {}).get('base_url', 'https://api.binance.com/api/v3')
            timeout = self.price_data_config.get('binance_api', {}).get('timeout_seconds', 10)
            
            price_data = {}
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                for symbol in symbols:
                    try:
                        # Fetch 24hr ticker price change statistics
                        url = f"{base_url}/ticker/24hr"
                        params = {'symbol': symbol}
                        
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                price_data[symbol] = {
                                    'symbol': symbol,
                                    'price': float(data['lastPrice']),
                                    'price_change': float(data['priceChange']),
                                    'price_change_percent': float(data['priceChangePercent']),
                                    'volume': float(data['volume']),
                                    'quote_volume': float(data['quoteVolume']),
                                    'high_24h': float(data['highPrice']),
                                    'low_24h': float(data['lowPrice']),
                                    'open_24h': float(data['openPrice']),
                                    'timestamp': datetime.utcnow()
                                }
                                
                                logger.debug(f"✅ Fetched price data for {symbol}: ${price_data[symbol]['price']:.2f}")
                            else:
                                logger.warning(f"⚠️ Failed to fetch price data for {symbol}: {response.status}")
                                
                    except Exception as e:
                        logger.warning(f"⚠️ Error fetching price data for {symbol}: {e}")
                        continue
            
            return price_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching real-time prices: {e}")
            return {}
    
    async def _store_price_data(self, price_data: Dict[str, Dict]):
        """Store price data in TimescaleDB"""
        try:
            if not price_data:
                return
            
            async with self.db_pool.acquire() as conn:
                for symbol, data in price_data.items():
                    await conn.execute("""
                        INSERT INTO price_data (
                            timestamp, symbol, open_price, high_price, low_price, close_price,
                            volume, price_change, price_change_percent, volume_change_percent,
                            data_source
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (timestamp, id) DO NOTHING;
                    """, 
                        data['timestamp'], symbol, data['open_24h'], data['high_24h'],
                        data['low_24h'], data['price'], data['volume'], data['price_change'],
                        data['price_change_percent'], 0.0, 'binance'
                    )
                
                logger.debug(f"📊 Stored price data for {len(price_data)} symbols")
                
        except Exception as e:
            logger.error(f"❌ Error storing price data: {e}")
    
    async def _calculate_price_correlation(self, news_timestamp: datetime, symbol: str = 'BTCUSDT') -> Dict[str, float]:
        """Calculate price correlation for different time windows"""
        try:
            if not self.price_data_config.get('enabled', False):
                return {'correlation_30m': 0.0, 'correlation_2h': 0.0, 'correlation_24h': 0.0}
            
            correlation_windows = self.price_data_config.get('correlation_windows', [1800, 7200, 86400])
            
            # Ensure news_timestamp is timezone-naive for database operations
            if news_timestamp.tzinfo is not None:
                news_timestamp = news_timestamp.replace(tzinfo=None)
            
            async with self.db_pool.acquire() as conn:
                correlations = {}
                
                for window_seconds in correlation_windows:
                    # Get price data before and after news timestamp
                    before_news = await conn.fetch("""
                        SELECT close_price, timestamp 
                        FROM price_data 
                        WHERE symbol = $1 
                        AND timestamp BETWEEN $2::timestamp - INTERVAL '1 hour' AND $2::timestamp
                        ORDER BY timestamp DESC
                        LIMIT 60;
                    """, symbol, news_timestamp)
                    
                    after_news = await conn.fetch("""
                        SELECT close_price, timestamp 
                        FROM price_data 
                        WHERE symbol = $1 
                        AND timestamp BETWEEN $2::timestamp AND $2::timestamp + INTERVAL '1 hour'
                        ORDER BY timestamp ASC
                        LIMIT 60;
                    """, symbol, news_timestamp)
                    
                    if before_news and after_news:
                        # Calculate correlation (simplified - in production use proper correlation)
                        before_prices = [float(row['close_price']) for row in before_news]
                        after_prices = [float(row['close_price']) for row in after_news]
                        
                        if len(before_prices) > 10 and len(after_prices) > 10:
                            # Simple correlation calculation
                            before_avg = sum(before_prices) / len(before_prices)
                            after_avg = sum(after_prices) / len(after_prices)
                            
                            before_variance = sum((p - before_avg) ** 2 for p in before_prices)
                            after_variance = sum((p - after_avg) ** 2 for p in after_prices)
                            
                            if before_variance > 0 and after_variance > 0:
                                correlation = 0.5  # Placeholder - implement proper correlation
                            else:
                                correlation = 0.0
                        else:
                            correlation = 0.0
                    else:
                        correlation = 0.0
                    
                    # Map window to correlation name
                    if window_seconds == 1800:
                        correlations['correlation_30m'] = correlation
                    elif window_seconds == 7200:
                        correlations['correlation_2h'] = correlation
                    elif window_seconds == 86400:
                        correlations['correlation_24h'] = correlation
                
                return correlations
                
        except Exception as e:
            logger.error(f"❌ Error calculating price correlation: {e}")
            return {'correlation_30m': 0.0, 'correlation_2h': 0.0, 'correlation_24h': 0.0}
    
    async def _detect_market_regime(self) -> Dict[str, Any]:
        """Detect current market regime (bull/bear/neutral)"""
        try:
            if not self.market_context_config.get('enabled', False):
                return {'regime': 'neutral', 'confidence': 0.5}
            
            async with self.db_pool.acquire() as conn:
                # Get recent price data for BTC
                recent_prices = await conn.fetch("""
                    SELECT close_price, price_change_percent, timestamp
                    FROM price_data 
                    WHERE symbol = 'BTCUSDT' 
                    AND timestamp >= NOW() - INTERVAL '24 hours'
                    ORDER BY timestamp DESC
                    LIMIT 24;
                """)
                
                if not recent_prices:
                    return {'regime': 'neutral', 'confidence': 0.5}
                
                # Calculate regime indicators
                price_changes = [float(row['price_change_percent']) for row in recent_prices]
                avg_change = sum(price_changes) / len(price_changes)
                
                # Determine regime based on average price change
                thresholds = self.market_context_config.get('regime_detection', {})
                bull_threshold = thresholds.get('bull_threshold', 0.6)
                bear_threshold = thresholds.get('bear_threshold', -0.6)
                
                if avg_change > bull_threshold:
                    regime = 'bull'
                    confidence = min(1.0, avg_change / bull_threshold)
                elif avg_change < bear_threshold:
                    regime = 'bear'
                    confidence = min(1.0, abs(avg_change) / abs(bear_threshold))
                else:
                    regime = 'neutral'
                    confidence = 0.5
                
                return {
                    'regime': regime,
                    'confidence': confidence,
                    'avg_change': avg_change,
                    'btc_dominance': 48.5,  # Placeholder
                    'market_volatility': 0.025,  # Placeholder
                    'fear_greed_index': 55  # Placeholder
                }
                
        except Exception as e:
            logger.error(f"❌ Error detecting market regime: {e}")
            return {'regime': 'neutral', 'confidence': 0.5}
    
    def _normalize_sentiment(self, sentiment_score: float, source: str) -> Dict[str, Any]:
        """Normalize sentiment across different sources to unified scale"""
        try:
            if not self.sentiment_normalization_config.get('enabled', False):
                return {'normalized_sentiment': sentiment_score, 'confidence': 0.5}
            
            unified_scale = self.sentiment_normalization_config.get('unified_scale', {})
            source_weights = self.sentiment_normalization_config.get('source_weights', {})
            confidence_thresholds = self.sentiment_normalization_config.get('confidence_thresholds', {})
            
            # Normalize to -1.0 to 1.0 scale
            if source == 'cryptopanic':
                # CryptoPanic sentiment is already in good range, just ensure bounds
                normalized = max(-1.0, min(1.0, sentiment_score))
            elif source == 'santiment':
                # Santiment might need scaling - assuming it's already normalized
                normalized = max(-1.0, min(1.0, sentiment_score))
            else:  # RSS or other sources
                # Assume TextBlob sentiment (-1 to 1) and normalize
                normalized = max(-1.0, min(1.0, sentiment_score))
            
            # Get source weight
            source_weight = source_weights.get(source, 0.5)
            
            # Calculate weighted sentiment
            weighted_sentiment = normalized * source_weight
            
            # Determine confidence based on source
            if source_weight >= confidence_thresholds.get('high', 0.8):
                confidence = 0.9
            elif source_weight >= confidence_thresholds.get('medium', 0.6):
                confidence = 0.7
            else:
                confidence = 0.5
            
            # Determine sentiment label
            neutral_threshold = unified_scale.get('neutral_threshold', 0.1)
            if abs(normalized) <= neutral_threshold:
                label = 'neutral'
            elif normalized > neutral_threshold:
                label = 'positive'
            else:
                label = 'negative'
            
            return {
                'normalized_sentiment': normalized,
                'weighted_sentiment': weighted_sentiment,
                'sentiment_label': label,
                'confidence': confidence,
                'source_weight': source_weight
            }
            
        except Exception as e:
            logger.error(f"❌ Error normalizing sentiment: {e}")
            return {'normalized_sentiment': sentiment_score, 'confidence': 0.5}
    
    async def _engineer_ml_features(self, article: NewsArticle) -> Dict[str, float]:
        """Engineer comprehensive ML features for prediction models"""
        try:
            if not self.ml_config.get('enabled', False):
                return {}
            
            features = {}
            feature_config = self.ml_config.get('feature_engineering', {})
            
            # Text features
            if feature_config.get('text_features', {}).get('tfidf', False):
                features.update(self._extract_text_features(article))
            
            # Market features
            if feature_config.get('market_features', {}).get('price_momentum', False):
                features.update(await self._extract_market_features(article))
            
            # Temporal features
            if feature_config.get('temporal_features', {}).get('time_of_day', False):
                features.update(self._extract_temporal_features(article))
            
            # Social features
            if feature_config.get('social_features', {}).get('social_volume', False):
                features.update(self._extract_social_features(article))
            
            # Store features in database
            await self._store_feature_engineering_data(article, features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error engineering ML features: {e}")
            return {}
    
    def _extract_text_features(self, article: NewsArticle) -> Dict[str, float]:
        """Extract text-based features for ML models"""
        try:
            features = {}
            
            # Basic text features
            features['title_length'] = len(article.title) if article.title else 0
            features['description_length'] = len(article.description) if article.description else 0
            features['content_length'] = len(article.content) if article.content else 0
            
            # Sentiment features
            features['sentiment_score'] = getattr(article, 'sentiment_score', 0.0)
            features['normalized_sentiment'] = getattr(article, 'normalized_sentiment', 0.0)
            features['sentiment_confidence'] = getattr(article, 'sentiment_confidence', 0.5)
            
            # Entity features
            if hasattr(article, 'entities') and article.entities:
                features['entity_count'] = len(article.entities)
                features['entity_density'] = features['entity_count'] / max(features['title_length'], 1)
            else:
                features['entity_count'] = 0
                features['entity_density'] = 0.0
            
            # Event type features
            if hasattr(article, 'event_types') and article.event_types:
                features['event_type_count'] = len(article.event_types)
                high_impact_events = ['hack', 'etf', 'regulation', 'institutional', 'whale']
                features['high_impact_event_count'] = sum(1 for e in article.event_types if e.lower() in high_impact_events)
            else:
                features['event_type_count'] = 0
                features['high_impact_event_count'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting text features: {e}")
            return {}
    
    async def _extract_market_features(self, article: NewsArticle) -> Dict[str, float]:
        """Extract market-based features for ML models"""
        try:
            features = {}
            
            # Market context features
            features['market_regime_bull'] = 1.0 if getattr(article, 'market_regime', 'neutral') == 'bull' else 0.0
            features['market_regime_bear'] = 1.0 if getattr(article, 'market_regime', 'neutral') == 'bear' else 0.0
            features['btc_dominance'] = getattr(article, 'btc_dominance', 48.5)
            features['market_volatility'] = getattr(article, 'market_volatility', 0.025)
            features['fear_greed_index'] = getattr(article, 'fear_greed_index', 55)
            
            # Correlation features
            features['correlation_30m'] = getattr(article, 'correlation_30m', 0.0)
            features['correlation_2h'] = getattr(article, 'correlation_2h', 0.0)
            features['correlation_24h'] = getattr(article, 'correlation_24h', 0.0)
            
            # Impact features
            features['impact_30m'] = getattr(article, 'impact_30m', 0.0)
            features['impact_2h'] = getattr(article, 'impact_2h', 0.0)
            features['impact_24h'] = getattr(article, 'impact_24h', 0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting market features: {e}")
            return {}
    
    def _extract_temporal_features(self, article: NewsArticle) -> Dict[str, float]:
        """Extract temporal features for ML models"""
        try:
            features = {}
            
            if article.published_at:
                # Time of day features (0-23 hours)
                features['hour_of_day'] = article.published_at.hour
                features['minute_of_hour'] = article.published_at.minute
                
                # Day of week features (0-6, Monday=0)
                features['day_of_week'] = article.published_at.weekday()
                
                # Market hours features
                features['is_market_hours'] = 1.0 if 9 <= features['hour_of_day'] <= 17 else 0.0
                features['is_weekend'] = 1.0 if features['day_of_week'] >= 5 else 0.0
                
                # Time-based categorical features
                features['is_morning'] = 1.0 if 6 <= features['hour_of_day'] <= 11 else 0.0
                features['is_afternoon'] = 1.0 if 12 <= features['hour_of_day'] <= 17 else 0.0
                features['is_evening'] = 1.0 if 18 <= features['hour_of_day'] <= 23 else 0.0
                features['is_night'] = 1.0 if features['hour_of_day'] <= 5 else 0.0
            else:
                # Default values if no timestamp
                features.update({
                    'hour_of_day': 12, 'minute_of_hour': 0, 'day_of_week': 0,
                    'is_market_hours': 1.0, 'is_weekend': 0.0,
                    'is_morning': 0.0, 'is_afternoon': 1.0, 'is_evening': 0.0, 'is_night': 0.0
                })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting temporal features: {e}")
            return {}
    
    def _extract_social_features(self, article: NewsArticle) -> Dict[str, float]:
        """Extract social features for ML models"""
        try:
            features = {}
            
            # Social volume features
            features['social_volume_spike'] = 1.0 if getattr(article, 'social_volume_spike', False) else 0.0
            features['social_volume_baseline'] = getattr(article, 'social_volume_baseline', 0.0)
            features['social_volume_current'] = getattr(article, 'social_volume_current', 0.0)
            
            # Developer activity features
            features['dev_activity_score'] = getattr(article, 'dev_activity_score', 0.0)
            
            # Whale transaction features
            features['whale_transaction_count'] = getattr(article, 'whale_transaction_count', 0)
            
            # Cross-source validation features
            features['cross_source_validation'] = 1.0 if getattr(article, 'cross_source_validation', False) else 0.0
            features['similarity_score'] = getattr(article, 'similarity_score', 0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting social features: {e}")
            return {}
    
    async def _store_feature_engineering_data(self, article: NewsArticle, features: Dict[str, float]):
        """Store feature engineering data in database"""
        try:
            if not features:
                return
            
            async with self.db_pool.acquire() as conn:
                for feature_name, feature_value in features.items():
                    # Determine feature category
                    if feature_name in ['title_length', 'description_length', 'content_length', 'sentiment_score', 'normalized_sentiment']:
                        category = 'text_features'
                    elif feature_name in ['market_regime_bull', 'btc_dominance', 'correlation_30m', 'impact_30m']:
                        category = 'market_features'
                    elif feature_name in ['hour_of_day', 'day_of_week', 'is_market_hours']:
                        category = 'temporal_features'
                    elif feature_name in ['social_volume_spike', 'dev_activity_score', 'whale_transaction_count']:
                        category = 'social_features'
                    else:
                        category = 'general_features'
                    
                    await conn.execute("""
                        INSERT INTO feature_engineering_data (
                            timestamp, news_id, feature_category, feature_name, 
                            feature_value, feature_type
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (timestamp, id) DO NOTHING;
                    """, 
                        datetime.utcnow(), getattr(article, 'id', 0), category, 
                        feature_name, feature_value, 'numeric'
                    )
                
                logger.debug(f"📊 Stored {len(features)} features for article")
                
        except Exception as e:
            logger.error(f"❌ Error storing feature engineering data: {e}")
    
    async def _generate_ml_predictions(self, article: NewsArticle, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate ML predictions for news impact"""
        try:
            if not self.ml_config.get('enabled', False):
                return {}
            
            predictions = {}
            prediction_config = self.ml_config.get('prediction_models', {})
            
            # Impact prediction
            if prediction_config.get('impact_prediction', {}).get('enabled', False):
                impact_prediction = await self._predict_news_impact(article, features)
                predictions['impact_prediction'] = impact_prediction
            
            # Sentiment enhancement
            if prediction_config.get('sentiment_enhancement', {}).get('enabled', False):
                sentiment_prediction = await self._enhance_sentiment_prediction(article, features)
                predictions['sentiment_enhancement'] = sentiment_prediction
            
            # Timing optimization
            if prediction_config.get('timing_optimization', {}).get('enabled', False):
                timing_prediction = await self._optimize_timing_prediction(article, features)
                predictions['timing_optimization'] = timing_prediction
            
            # Store predictions in database
            await self._store_ml_predictions(article, predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Error generating ML predictions: {e}")
            return {}
    
    async def _predict_news_impact(self, article: NewsArticle, features: Dict[str, float]) -> Dict[str, Any]:
        """Advanced predictive news impact scoring with volatility forecasting"""
        try:
            # Multi-model impact prediction ensemble
            impact_predictions = []
            confidence_scores = []
            
            # 1. ML Model Prediction (if available)
            if self.ml_models:
                try:
                    ml_prediction = await self.ml_models.predict_news_impact(features)
                    impact_predictions.append(ml_prediction.prediction)
                    confidence_scores.append(ml_prediction.confidence)
                except Exception as ml_e:
                    logger.warning(f"⚠️ ML impact prediction failed: {ml_e}")
            
            # 2. Advanced Rule-based Impact Analysis
            sentiment_score = features.get('normalized_sentiment', 0.0)
            entity_count = features.get('entity_count', 0)
            social_volume = features.get('social_volume', 0.0)
            cross_validation = features.get('cross_source_validation', 0.0)
            correlation_30m = features.get('correlation_30m', 0.0)
            market_hours = features.get('is_market_hours', 0.0)
            
            # Enhanced sentiment impact with market context
            sentiment_magnitude = abs(sentiment_score - 0.5)
            sentiment_impact = sentiment_magnitude * 0.4
            
            # Market volatility amplification
            market_volatility = features.get('market_volatility', 0.5)
            volatility_multiplier = 1.0 + (market_volatility - 0.5) * 0.5
            sentiment_impact *= volatility_multiplier
            
            # Entity impact with quality weighting
            entity_impact = min(0.3, entity_count * 0.06) if entity_count > 0 else 0.0
            
            # High-impact entity boost
            entities = getattr(article, 'entities', [])
            high_impact_entities = ['bitcoin', 'ethereum', 'fed', 'sec', 'tesla', 'microstrategy']
            
            # Handle entities as either list of strings or dict
            if isinstance(entities, list):
                # Extract text from entity dictionaries or use strings directly
                entity_texts = []
                for entity in entities:
                    if isinstance(entity, dict):
                        entity_texts.append(entity.get('text', '').lower())
                    elif isinstance(entity, str):
                        entity_texts.append(entity.lower())
                    else:
                        continue
            elif isinstance(entities, dict):
                entity_texts = [str(key).lower() for key in entities.keys()] if entities else []
            else:
                entity_texts = []
            
            if any(entity_text in high_impact_entities for entity_text in entity_texts):
                entity_impact *= 1.5
            
            # Social volume impact with spike detection
            social_impact = min(0.25, social_volume * 0.001)
            social_spike_threshold = 1000.0  # Configurable
            if social_volume > social_spike_threshold:
                social_impact *= 1.3  # Boost for social spikes
            
            # Market timing impact
            timing_impact = 0.15 if market_hours else 0.05
            
            # Correlation impact with trend analysis
            correlation_impact = abs(correlation_30m) * 0.2
            
            # Cross-source validation impact
            validation_impact = cross_validation * 0.2
            
            # Breaking news boost
            breaking_boost = 0.3 if getattr(article, 'is_breaking_news', False) else 0.0
            
            # Combine all impact factors
            rule_based_impact = (
                sentiment_impact + entity_impact + social_impact + 
                timing_impact + correlation_impact + validation_impact + breaking_boost
            )
            
            # Calculate confidence for rule-based prediction
            confidence_factors = [
                min(0.9, sentiment_magnitude * 2),  # Higher for strong sentiment
                min(0.8, entity_count * 0.1),      # Higher for more entities
                min(0.7, cross_validation),        # Higher for validation
                min(0.6, social_volume * 0.0001),  # Higher for social activity
                min(0.8, abs(correlation_30m))     # Higher for strong correlation
            ]
            rule_based_confidence = sum(confidence_factors) / len(confidence_factors)
            
            impact_predictions.append(min(1.0, rule_based_impact))
            confidence_scores.append(min(0.95, rule_based_confidence))
            
            # 3. Historical Pattern Impact (simplified)
            pattern_impact = 0.0
            pattern_confidence = 0.5
            
            # Look for similar sentiment + entity patterns
            if sentiment_magnitude > 0.3 and entity_count > 2:
                pattern_impact = min(0.4, sentiment_magnitude * entity_count * 0.1)
                pattern_confidence = 0.7
                impact_predictions.append(pattern_impact)
                confidence_scores.append(pattern_confidence)
            
            # Ensemble prediction with weighted average
            if impact_predictions and confidence_scores:
                total_weight = sum(confidence_scores)
                if total_weight > 0:
                    final_impact = sum(i * c for i, c in zip(impact_predictions, confidence_scores)) / total_weight
                    final_confidence = min(0.95, total_weight / len(confidence_scores))
                else:
                    final_impact = sum(impact_predictions) / len(impact_predictions)
                    final_confidence = sum(confidence_scores) / len(confidence_scores)
            else:
                final_impact = 0.0
                final_confidence = 0.5
            
            # Volatility forecasting
            volatility_forecast = {
                'short_term_volatility': min(1.0, final_impact * market_volatility * 1.2),
                'medium_term_volatility': min(1.0, final_impact * market_volatility * 0.8),
                'volatility_direction': 'increasing' if final_impact > 0.5 else 'decreasing',
                'volatility_confidence': final_confidence * 0.8
            }
            
            # Impact classification
            if final_impact >= 0.8:
                impact_class = 'extreme'
            elif final_impact >= 0.6:
                impact_class = 'high'
            elif final_impact >= 0.4:
                impact_class = 'moderate'
            elif final_impact >= 0.2:
                impact_class = 'low'
            else:
                impact_class = 'minimal'
            
            # Time-based impact predictions
            time_impacts = {
                'impact_30m': final_impact,
                'impact_2h': final_impact * 0.8,
                'impact_24h': final_impact * 0.6,
                'peak_impact_time': '30m' if final_impact > 0.7 else '2h'
            }
            
            return {
                'predicted_impact': final_impact,
                'confidence': final_confidence,
                'model_type': 'advanced_ensemble',
                'model_version': 'v2.0',
                'impact_classification': impact_class,
                'volatility_forecast': volatility_forecast,
                'time_based_impacts': time_impacts,
                'impact_components': {
                    'sentiment_impact': sentiment_impact,
                    'entity_impact': entity_impact,
                    'social_impact': social_impact,
                    'timing_impact': timing_impact,
                    'correlation_impact': correlation_impact,
                    'validation_impact': validation_impact,
                    'breaking_boost': breaking_boost
                },
                'prediction_sources': len(impact_predictions),
                'ensemble_strength': final_confidence * len(impact_predictions) / 3.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error in advanced impact prediction: {e}")
            return {'predicted_impact': 0.0, 'confidence': 0.5, 'model_type': 'error', 'model_version': 'v2.0'}
    
    async def _enhance_sentiment_prediction(self, article: NewsArticle, features: Dict[str, float]) -> Dict[str, Any]:
        """Advanced sentiment prediction using multi-model ensemble and market intelligence"""
        try:
            # Multi-source sentiment analysis
            sentiment_sources = []
            confidence_sources = []
            
            # 1. ML Model Prediction (if available)
            if self.ml_models:
                try:
                    ml_prediction = await self.ml_models.enhance_sentiment_prediction(features)
                    sentiment_sources.append(ml_prediction.prediction)
                    confidence_sources.append(ml_prediction.confidence)
                except Exception as ml_e:
                    logger.warning(f"⚠️ ML sentiment prediction failed: {ml_e}")
            
            # 2. Advanced Rule-based Analysis
            base_sentiment = features.get('normalized_sentiment', 0.0)
            base_confidence = features.get('sentiment_confidence', 0.5)
            
            # Market regime adjustments
            market_regime_bull = features.get('market_regime_bull', 0.0)
            market_regime_bear = features.get('market_regime_bear', 0.0)
            
            # Volatility impact on sentiment
            market_volatility = features.get('market_volatility', 0.5)
            volatility_factor = 1.0 + (market_volatility - 0.5) * 0.2  # Amplify sentiment in high volatility
            
            # Cross-source validation boost
            cross_validated = features.get('cross_source_validation', 0.0)
            validation_boost = 1.0 + (cross_validated * 0.15)  # Boost confidence for validated news
            
            # Market timing adjustment
            hour_of_day = features.get('hour_of_day', 12)
            is_market_hours = features.get('is_market_hours', 0.0)
            timing_factor = 1.1 if is_market_hours else 0.9  # Higher impact during market hours
            
            # Social volume impact
            social_volume = features.get('social_volume', 0.0)
            social_factor = 1.0 + min(0.3, social_volume * 0.001)  # Cap social boost at 30%
            
            # Combine all factors
            enhanced_sentiment = base_sentiment
            
            if market_regime_bull > 0.5:
                enhanced_sentiment *= 1.15 * volatility_factor * timing_factor * social_factor
            elif market_regime_bear > 0.5:
                enhanced_sentiment *= 0.85 * volatility_factor * timing_factor * social_factor
            else:
                enhanced_sentiment *= volatility_factor * timing_factor * social_factor
            
            # Apply validation boost to confidence
            enhanced_confidence = base_confidence * validation_boost
            
            # Normalize enhanced sentiment
            enhanced_sentiment = max(-1.0, min(1.0, enhanced_sentiment))
            enhanced_confidence = min(0.95, enhanced_confidence)
            
            sentiment_sources.append(enhanced_sentiment)
            confidence_sources.append(enhanced_confidence)
            
            # 3. Correlation-based sentiment adjustment
            correlation_30m = features.get('correlation_30m', 0.0)
            correlation_2h = features.get('correlation_2h', 0.0)
            correlation_24h = features.get('correlation_24h', 0.0)
            
            # Weight recent correlations more heavily
            weighted_correlation = (correlation_30m * 0.5 + correlation_2h * 0.3 + correlation_24h * 0.2)
            
            if abs(weighted_correlation) > 0.1:  # Only adjust if significant correlation
                correlation_sentiment = base_sentiment * (1 + weighted_correlation * 0.2)
                correlation_sentiment = max(-1.0, min(1.0, correlation_sentiment))
                sentiment_sources.append(correlation_sentiment)
                confidence_sources.append(min(0.8, abs(weighted_correlation) + 0.3))
            
            # Ensemble prediction with weighted average
            if sentiment_sources and confidence_sources:
                # Weight by confidence
                total_weight = sum(confidence_sources)
                if total_weight > 0:
                    final_sentiment = sum(s * c for s, c in zip(sentiment_sources, confidence_sources)) / total_weight
                    final_confidence = min(0.95, total_weight / len(confidence_sources))
                else:
                    final_sentiment = sum(sentiment_sources) / len(sentiment_sources)
                    final_confidence = sum(confidence_sources) / len(confidence_sources)
            else:
                final_sentiment = base_sentiment
                final_confidence = base_confidence
            
            # Advanced confidence scoring
            confidence_factors = {
                'source_count': min(0.2, len(sentiment_sources) * 0.05),
                'cross_validation': cross_validated * 0.15,
                'market_timing': is_market_hours * 0.1,
                'volatility_context': min(0.1, market_volatility * 0.2)
            }
            
            final_confidence += sum(confidence_factors.values())
            final_confidence = min(0.95, final_confidence)
            
            return {
                'enhanced_sentiment': final_sentiment,
                'confidence': final_confidence,
                'model_type': 'advanced_ensemble',
                'model_version': 'v2.0',
                'sentiment_sources': len(sentiment_sources),
                'confidence_factors': confidence_factors,
                'market_adjustments': {
                    'regime_factor': 1.15 if market_regime_bull > 0.5 else (0.85 if market_regime_bear > 0.5 else 1.0),
                    'volatility_factor': volatility_factor,
                    'timing_factor': timing_factor,
                    'social_factor': social_factor
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in advanced sentiment prediction: {e}")
            return {'enhanced_sentiment': 0.0, 'confidence': 0.5, 'model_type': 'error', 'model_version': 'v2.0'}
    
    async def _optimize_timing_prediction(self, article: NewsArticle, features: Dict[str, float]) -> Dict[str, Any]:
        """Optimize timing prediction using ML model"""
        try:
            if self.ml_models:
                # Get ML prediction
                prediction = await self.ml_models.optimize_timing_prediction(features)
                
                return {
                    'optimal_timing_score': prediction.prediction,
                    'confidence': prediction.confidence,
                    'model_type': prediction.model_type,
                    'features_used': prediction.features_used,
                    'feature_importance': prediction.feature_importance,
                    'model_version': prediction.model_version
                }
            else:
                # Fallback to rule-based prediction
                hour_of_day = features.get('hour_of_day', 12)
                day_of_week = features.get('day_of_week', 0)
                is_market_hours = features.get('is_market_hours', 1.0)
                
                # Calculate optimal timing score
                timing_score = 0.0
                
                # Prefer market hours
                if is_market_hours > 0:
                    timing_score += 0.3
                
                # Prefer weekdays
                if day_of_week < 5:
                    timing_score += 0.2
                
                # Prefer morning/afternoon hours
                if 9 <= hour_of_day <= 16:
                    timing_score += 0.3
                
                # Avoid weekends and late hours
                if day_of_week >= 5 or hour_of_day >= 22 or hour_of_day <= 6:
                    timing_score -= 0.2
                
                # Normalize timing score
                timing_score = max(0.0, min(1.0, timing_score))
                
                return {
                    'optimal_timing_score': timing_score,
                    'confidence': 0.7,
                    'model_type': 'rule_based',
                    'model_version': 'v1.0'
                }
            
        except Exception as e:
            logger.error(f"❌ Error optimizing timing prediction: {e}")
            return {'optimal_timing_score': 0.5, 'confidence': 0.5, 'model_type': 'error', 'model_version': 'v1.0'}
    
    async def _store_ml_predictions(self, article: NewsArticle, predictions: Dict[str, Any]):
        """Store ML predictions in database"""
        try:
            if not predictions:
                return
            
            async with self.db_pool.acquire() as conn:
                for prediction_type, prediction_data in predictions.items():
                    predicted_value = prediction_data.get('predicted_impact', 
                                                        prediction_data.get('enhanced_sentiment', 
                                                        prediction_data.get('optimal_timing_score', 0.0)))
                    
                    await conn.execute("""
                        INSERT INTO ml_predictions (
                            timestamp, news_id, model_type, prediction_type, 
                            predicted_value, confidence_score, model_version
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (timestamp, id) DO NOTHING;
                    """, 
                        datetime.utcnow(), getattr(article, 'id', 0), 
                        prediction_data.get('model_type', 'unknown'),
                        prediction_type, predicted_value,
                        prediction_data.get('confidence', 0.5),
                        prediction_data.get('model_version', 'v1.0')
                    )
                
                logger.debug(f"🤖 Stored {len(predictions)} ML predictions for article")
                
        except Exception as e:
            logger.error(f"❌ Error storing ML predictions: {e}")
    
    async def _calculate_advanced_correlation(self, article: NewsArticle, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """Advanced dynamic correlation analysis with market intelligence"""
        try:
            if not self.advanced_correlation_config.get('enabled', False):
                return {}
            
            correlation_results = {}
            
            # Advanced correlation factors
            base_factors = {
                'sentiment_magnitude': abs(getattr(article, 'sentiment_score', 0.0)),
                'entity_count': len(getattr(article, 'entities', [])),
                'social_volume': getattr(article, 'social_volume', 0.0),
                'cross_validation': 1.0 if getattr(article, 'cross_source_validation', False) else 0.0,
                'market_hours': 1.0 if 9 <= datetime.utcnow().hour <= 16 else 0.3,
                'breaking_news': 1.0 if getattr(article, 'is_breaking_news', False) else 0.0
            }
            
            # Dynamic time windows with adaptive calculation
            dynamic_windows = {
                'ultra_short': {'seconds': [300, 600, 900], 'weight': 1.2},      # 5-15 min
                'short': {'seconds': [1800, 3600], 'weight': 1.0},               # 30min-1h
                'medium': {'seconds': [7200, 14400], 'weight': 0.8},             # 2-4h
                'long': {'seconds': [86400, 172800], 'weight': 0.6}              # 1-2 days
            }
            
            # Market volatility adjustment
            market_volatility = getattr(article, 'market_volatility', 0.5)
            volatility_multiplier = 1.0 + (market_volatility - 0.5) * 0.5
            
            # Calculate correlations for each dynamic window
            for window_type, window_config in dynamic_windows.items():
                for window_seconds in window_config['seconds']:
                    try:
                        # Advanced correlation calculation
                        correlation_data = await self._calculate_window_correlation(
                            article, symbol, window_seconds, base_factors, volatility_multiplier
                        )
                        
                        # Apply window weight
                        weighted_correlation = correlation_data['correlation'] * window_config['weight']
                        
                        # Statistical significance testing
                        significance_test = self._calculate_statistical_significance(
                            correlation_data, window_seconds
                        )
                        
                        correlation_results[f'{window_type}_{window_seconds}s'] = {
                            'correlation_coefficient': weighted_correlation,
                            'raw_correlation': correlation_data['correlation'],
                            'p_value': significance_test['p_value'],
                            'confidence_interval': significance_test['confidence_interval'],
                            'significance_level': 0.05,
                            'time_window_seconds': window_seconds,
                            'sample_size': correlation_data.get('sample_size', 0),
                            'market_factors': base_factors,
                            'volatility_adjustment': volatility_multiplier,
                            'correlation_strength': self._classify_correlation_strength(weighted_correlation),
                            'directional_bias': 1 if weighted_correlation > 0 else -1,
                            'reliability_score': min(0.95, correlation_data.get('reliability', 0.5) * 
                                                   (1 + base_factors['cross_validation'] * 0.2))
                        }
                        
                    except Exception as window_e:
                        logger.warning(f"⚠️ Error calculating {window_type} correlation: {window_e}")
                        continue
            
            # Enhanced correlation analytics
            if correlation_results:
                correlation_analytics = self._analyze_correlation_patterns(correlation_results)
                correlation_results['analytics'] = correlation_analytics
                
                # Correlation trend analysis
                correlation_results['trends'] = self._calculate_correlation_trends(correlation_results)
                
                # Market regime correlation adjustment
                regime_adjustment = await self._adjust_correlation_for_regime(correlation_results, article)
                correlation_results['regime_adjusted'] = regime_adjustment
            
            # Store advanced correlation analysis
            await self._store_advanced_correlation_analysis(article, symbol, correlation_results)
            
            return correlation_results
            
        except Exception as e:
            logger.error(f"❌ Error calculating advanced correlation: {e}")
            return {}
    
    async def _store_advanced_correlation_analysis(self, article: NewsArticle, symbol: str, correlation_results: Dict[str, Any]):
        """Store advanced correlation analysis in database"""
        try:
            if not correlation_results:
                return
            
            async with self.db_pool.acquire() as conn:
                for window_key, correlation_data in correlation_results.items():
                    # Skip non-correlation data (like 'analytics', 'trends', etc.)
                    if not isinstance(correlation_data, dict) or 'correlation_coefficient' not in correlation_data:
                        continue
                    
                    # Extract time window from key or use default
                    time_window = correlation_data.get('time_window_seconds', 3600)  # Default to 1 hour
                    
                    await conn.execute("""
                        INSERT INTO advanced_correlation_analysis (
                            timestamp, news_id, symbol, correlation_method, time_window_seconds,
                            correlation_coefficient, p_value, significance_level
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (timestamp, id) DO NOTHING;
                    """, 
                        datetime.utcnow(), getattr(article, 'id', 0), symbol,
                        'pearson', time_window,
                        correlation_data.get('correlation_coefficient', 0.0),
                        correlation_data.get('p_value', 1.0),
                        correlation_data.get('significance_level', 0.05)
                    )
                
                logger.debug(f"📊 Stored advanced correlation analysis for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error storing advanced correlation analysis: {e}")
    
    async def _generate_real_time_alerts(self, article: NewsArticle, predictions: Dict[str, Any], correlations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate real-time alerts based on predictions and correlations"""
        try:
            if not self.real_time_alerts_config.get('enabled', False):
                return []
            
            alerts = []
            alert_types = self.real_time_alerts_config.get('alert_types', {})
            
            # High correlation alerts
            if alert_types.get('high_correlation', {}).get('enabled', False):
                threshold = alert_types['high_correlation'].get('threshold', 0.8)
                for window_key, correlation_data in correlations.items():
                    if correlation_data.get('correlation_coefficient', 0.0) > threshold:
                        alerts.append({
                            'alert_type': 'high_correlation',
                            'priority': alert_types['high_correlation'].get('priority', 'high'),
                            'alert_message': f'High correlation ({correlation_data["correlation_coefficient"]:.3f}) detected for {window_key}',
                            'trigger_value': correlation_data['correlation_coefficient'],
                            'threshold_value': threshold,
                            'confidence_score': 0.9
                        })
            
            # Impact prediction alerts
            if alert_types.get('impact_prediction', {}).get('enabled', False):
                threshold = alert_types['impact_prediction'].get('threshold', 0.75)
                if 'impact_prediction' in predictions:
                    impact_score = predictions['impact_prediction'].get('predicted_impact', 0.0)
                    if impact_score > threshold:
                        alerts.append({
                            'alert_type': 'impact_prediction',
                            'priority': alert_types['impact_prediction'].get('priority', 'critical'),
                            'alert_message': f'High impact prediction ({impact_score:.3f}) for news article',
                            'trigger_value': impact_score,
                            'threshold_value': threshold,
                            'confidence_score': predictions['impact_prediction'].get('confidence', 0.5)
                        })
            
            # Store alerts in database
            await self._store_real_time_alerts(article, alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error generating real-time alerts: {e}")
            return []
    
    def _calculate_correlation_trends(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation trends across different time windows"""
        try:
            trends = {
                'short_term_trend': 'neutral',
                'long_term_trend': 'neutral',
                'trend_strength': 0.0,
                'trend_consistency': 0.0,
                'trend_direction': 0.0
            }
            
            if not correlation_results:
                return trends
            
            # Extract correlation coefficients for trend analysis
            correlations = []
            for key, data in correlation_results.items():
                if isinstance(data, dict) and 'correlation_coefficient' in data:
                    correlations.append(data['correlation_coefficient'])
            
            if not correlations:
                return trends
            
            # Calculate trend metrics
            correlations = np.array(correlations)
            
            # Trend direction (positive/negative)
            positive_corr = correlations[correlations > 0]
            negative_corr = correlations[correlations < 0]
            
            if len(positive_corr) > len(negative_corr):
                trends['trend_direction'] = 1.0
                trends['short_term_trend'] = 'positive'
                trends['long_term_trend'] = 'positive'
            elif len(negative_corr) > len(positive_corr):
                trends['trend_direction'] = -1.0
                trends['short_term_trend'] = 'negative'
                trends['long_term_trend'] = 'negative'
            
            # Trend strength (average absolute correlation)
            trends['trend_strength'] = float(np.mean(np.abs(correlations)))
            
            # Trend consistency (standard deviation of correlations)
            trends['trend_consistency'] = float(1.0 - min(1.0, np.std(correlations)))
            
            return trends
            
        except Exception as e:
            logger.error(f"❌ Error calculating correlation trends: {e}")
            return {
                'short_term_trend': 'neutral',
                'long_term_trend': 'neutral',
                'trend_strength': 0.0,
                'trend_consistency': 0.0,
                'trend_direction': 0.0
            }
    
    async def _store_real_time_alerts(self, article: NewsArticle, alerts: List[Dict[str, Any]]):
        """Store real-time alerts in database"""
        try:
            if not alerts:
                return
            
            async with self.db_pool.acquire() as conn:
                for alert in alerts:
                    alert_id = f"alert_{int(datetime.utcnow().timestamp())}_{len(alerts)}"
                    
                    await conn.execute("""
                        INSERT INTO real_time_alerts (
                            timestamp, alert_id, alert_type, priority, news_id,
                            alert_message, trigger_value, threshold_value, confidence_score
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (timestamp, id) DO NOTHING;
                    """, 
                        datetime.utcnow(), alert_id, alert['alert_type'],
                        alert['priority'], getattr(article, 'id', None),
                        alert['alert_message'], alert['trigger_value'],
                        alert['threshold_value'], alert['confidence_score']
                    )
                
                logger.info(f"🚨 Generated {len(alerts)} real-time alerts")
                
        except Exception as e:
            logger.error(f"❌ Error storing real-time alerts: {e}")
    
    # Advanced Correlation Analysis Helper Methods
    async def _calculate_window_correlation(self, article: NewsArticle, symbol: str, window_seconds: int, 
                                           base_factors: Dict[str, float], volatility_multiplier: float) -> Dict[str, Any]:
        """Calculate correlation for a specific time window with advanced analytics"""
        try:
            # Base correlation from sentiment and market factors
            sentiment_correlation = base_factors['sentiment_magnitude'] * 0.3
            entity_correlation = min(0.2, base_factors['entity_count'] * 0.05)
            social_correlation = min(0.25, base_factors['social_volume'] * 0.001)
            validation_correlation = base_factors['cross_validation'] * 0.2
            timing_correlation = base_factors['market_hours'] * 0.15
            breaking_correlation = base_factors['breaking_news'] * 0.3
            
            # Combine factors
            base_correlation = (sentiment_correlation + entity_correlation + social_correlation + 
                              validation_correlation + timing_correlation + breaking_correlation)
            
            # Apply volatility adjustment
            adjusted_correlation = base_correlation * volatility_multiplier
            
            # Time decay factor (longer windows have lower correlation)
            decay_factor = max(0.3, 1.0 - (window_seconds / 86400) * 0.5)
            final_correlation = adjusted_correlation * decay_factor
            
            # Simulate sample size based on window (in production, get from actual data)
            sample_size = max(10, min(1000, window_seconds // 60))
            
            # Reliability based on sample size and validation
            reliability = min(0.95, 0.5 + (sample_size / 1000) * 0.3 + base_factors['cross_validation'] * 0.2)
            
            return {
                'correlation': max(-0.95, min(0.95, final_correlation)),
                'sample_size': sample_size,
                'reliability': reliability,
                'base_factors_contribution': base_correlation,
                'volatility_impact': volatility_multiplier,
                'decay_factor': decay_factor
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating window correlation: {e}")
            return {'correlation': 0.0, 'sample_size': 0, 'reliability': 0.0}
    
    def _calculate_statistical_significance(self, correlation_data: Dict[str, Any], window_seconds: int) -> Dict[str, Any]:
        """Calculate statistical significance of correlation"""
        try:
            correlation = correlation_data.get('correlation', 0.0)
            sample_size = correlation_data.get('sample_size', 0)
            
            # Simplified p-value calculation (in production, use proper statistical tests)
            if sample_size > 30:
                # Higher correlation and larger sample = lower p-value
                p_value = max(0.001, (1 - abs(correlation)) * (30 / sample_size))
            else:
                p_value = 0.5  # Not enough data for significance
            
            # Confidence interval (simplified)
            margin_error = 1.96 * (1 / (sample_size ** 0.5)) if sample_size > 0 else 0.5
            confidence_interval = [
                max(-1.0, correlation - margin_error),
                min(1.0, correlation + margin_error)
            ]
            
            return {
                'p_value': p_value,
                'confidence_interval': confidence_interval,
                'is_significant': p_value < 0.05,
                'significance_strength': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating statistical significance: {e}")
            return {'p_value': 1.0, 'confidence_interval': [-1.0, 1.0], 'is_significant': False}
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_correlation = abs(correlation)
        if abs_correlation >= 0.8:
            return 'very_strong'
        elif abs_correlation >= 0.6:
            return 'strong'
        elif abs_correlation >= 0.4:
            return 'moderate'
        elif abs_correlation >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _analyze_correlation_patterns(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in correlation results"""
        try:
            # Extract numeric correlations
            correlations = []
            window_types = []
            
            for key, data in correlation_results.items():
                if isinstance(data, dict) and 'correlation_coefficient' in data:
                    correlations.append(data['correlation_coefficient'])
                    window_types.append(key.split('_')[0])
            
            if not correlations:
                return {}
            
            # Pattern analysis
            avg_correlation = sum(correlations) / len(correlations)
            max_correlation = max(correlations)
            min_correlation = min(correlations)
            correlation_range = max_correlation - min_correlation
            
            # Trend analysis
            short_term_avg = sum(c for i, c in enumerate(correlations) if 'ultra_short' in window_types[i] or 'short' in window_types[i]) / max(1, sum(1 for wt in window_types if 'ultra_short' in wt or 'short' in wt))
            long_term_avg = sum(c for i, c in enumerate(correlations) if 'medium' in window_types[i] or 'long' in window_types[i]) / max(1, sum(1 for wt in window_types if 'medium' in wt or 'long' in wt))
            
            return {
                'average_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'min_correlation': min_correlation,
                'correlation_range': correlation_range,
                'volatility': correlation_range / max(0.001, abs(avg_correlation)),
                'short_term_avg': short_term_avg,
                'long_term_avg': long_term_avg,
                'trend_direction': 'increasing' if short_term_avg > long_term_avg else 'decreasing',
                'consistency': 1.0 - (correlation_range / max(0.001, abs(avg_correlation))),
                'pattern_strength': 'strong' if correlation_range < 0.2 else 'moderate' if correlation_range < 0.4 else 'weak'
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing correlation patterns: {e}")
            return {}
    
    async def _adjust_correlation_for_regime(self, correlation_results: Dict[str, Any], article: NewsArticle) -> Dict[str, Any]:
        """Adjust correlations based on current market regime"""
        try:
            # Get current market regime
            market_regime = await self._detect_market_regime()
            regime_type = market_regime.get('regime', 'neutral')
            regime_confidence = market_regime.get('confidence', 0.5)
            
            adjustments = {}
            
            # Regime-based adjustment factors
            regime_factors = {
                'bull': {'multiplier': 1.1, 'sentiment_boost': 0.15},
                'bear': {'multiplier': 0.9, 'sentiment_dampen': 0.15},
                'neutral': {'multiplier': 1.0, 'stability_boost': 0.05}
            }
            
            factor = regime_factors.get(regime_type, regime_factors['neutral'])
            
            for key, data in correlation_results.items():
                if isinstance(data, dict) and 'correlation_coefficient' in data:
                    original_correlation = data['correlation_coefficient']
                    
                    # Apply regime multiplier
                    adjusted_correlation = original_correlation * factor['multiplier']
                    
                    # Apply sentiment-specific adjustments
                    article_sentiment = getattr(article, 'sentiment_score', 0.0)
                    
                    if regime_type == 'bull' and article_sentiment > 0:
                        adjusted_correlation += factor.get('sentiment_boost', 0.0)
                    elif regime_type == 'bear' and article_sentiment < 0:
                        adjusted_correlation -= factor.get('sentiment_dampen', 0.0)
                    elif regime_type == 'neutral':
                        adjusted_correlation += factor.get('stability_boost', 0.0)
                    
                    # Apply regime confidence weighting
                    final_correlation = (original_correlation * (1 - regime_confidence) + 
                                       adjusted_correlation * regime_confidence)
                    
                    adjustments[key] = {
                        'original': original_correlation,
                        'regime_adjusted': final_correlation,
                        'adjustment_factor': final_correlation - original_correlation,
                        'regime': regime_type,
                        'regime_confidence': regime_confidence
                    }
            
            return adjustments
            
        except Exception as e:
            logger.error(f"❌ Error adjusting correlation for regime: {e}")
            return {}
    
    def _classify_news_category(self, article: Dict) -> str:
        """Classify news category"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        
        if any(keyword in text for keyword in self.crypto_keywords):
            return 'crypto'
        elif any(keyword in text for keyword in self.economic_keywords):
            return 'economic'
        else:
            return 'general'
    
    def _extract_tags(self, article: Dict) -> List[str]:
        """Extract tags from article"""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if len(word) > 3][:10]
    
    async def process_news_articles(self, articles: List[NewsArticle]) -> List[Dict[str, Any]]:
        """Process and analyze news articles"""
        try:
            processed_articles = []
            
            for article in articles:
                try:
                    # Analyze sentiment
                    sentiment_score, sentiment_label, confidence = await self._analyze_sentiment(article)
                    
                    # Calculate relevance and impact scores
                    relevance_score = await self._calculate_relevance_score(article)
                    impact_score = await self._calculate_impact_score(article, sentiment_score)
                    
                    # Detect if breaking news
                    breaking_news = await self._detect_breaking_news_article(article, impact_score)
                    
                    # Extract keywords
                    keywords = await self._extract_keywords(article)
                    
                    processed_article = {
                        'article': article,
                        'sentiment_score': sentiment_score,
                        'sentiment_label': sentiment_label,
                        'confidence': confidence,
                        'relevance_score': relevance_score,
                        'impact_score': impact_score,
                        'breaking_news': breaking_news,
                        'keywords': keywords
                    }
                    
                    processed_articles.append(processed_article)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing article '{article.title}': {e}")
                    continue
            
            logger.info(f"✅ Processed {len(processed_articles)} news articles")
            return processed_articles
            
        except Exception as e:
            logger.error(f"❌ Error processing news articles: {e}")
            return []
    
    async def _analyze_sentiment(self, article: NewsArticle) -> Tuple[float, str, float]:
        """Analyze sentiment of news article"""
        try:
            text = f"{article.title} {article.description} {article.content}"
            
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            confidence = abs(blob.sentiment.subjectivity)
            
            if sentiment_score > 0.1:
                sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return sentiment_score, sentiment_label, confidence
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment: {e}")
            return 0.0, 'neutral', 0.5
    
    async def _calculate_relevance_score(self, article: NewsArticle) -> float:
        """Calculate relevance score for news article"""
        try:
            text = f"{article.title} {article.description}".lower()
            
            # Count crypto and economic keywords
            crypto_count = sum(1 for keyword in self.crypto_keywords if keyword in text)
            economic_count = sum(1 for keyword in self.economic_keywords if keyword in text)
            
            # Calculate relevance based on keyword density
            total_words = len(text.split())
            if total_words == 0:
                return 0.0
            
            relevance_score = (crypto_count + economic_count) / total_words * 100
            relevance_score = min(1.0, relevance_score)  # Cap at 1.0
            
            return relevance_score
            
        except Exception as e:
            logger.error(f"❌ Error calculating relevance score: {e}")
            return 0.5
    
    async def _calculate_impact_score(self, article: NewsArticle, sentiment_score: float) -> float:
        """Calculate impact score for news article with social metrics"""
        try:
            # Base impact on source credibility from config
            source_credibility = 0.5  # Default
            
            if article.source.lower() in self.credible_sources:
                source_credibility = 0.9
            elif article.source.lower() in self.crypto_sources:
                source_credibility = 0.8
            elif article.source.lower() in self.cryptopanic_sources:
                source_credibility = 0.7
            elif hasattr(article, 'feed_credibility'):
                source_credibility = article.feed_credibility
            
            # Impact based on sentiment magnitude and source credibility
            impact_score = abs(sentiment_score) * source_credibility
            
            # Boost for breaking news indicators from config
            text = f"{article.title} {article.description}".lower()
            
            if any(indicator in text for indicator in self.breaking_indicators):
                impact_score *= 1.5
            
            # Enhanced sentiment analysis using new keywords
            positive_keywords = self.config['keywords'].get('sentiment_positive', [])
            negative_keywords = self.config['keywords'].get('sentiment_negative', [])
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            # Adjust impact based on keyword density
            if positive_count > negative_count:
                impact_score *= (1 + positive_count * 0.1)
            elif negative_count > positive_count:
                impact_score *= (1 + negative_count * 0.1)
            
            # Social volume spike detection (placeholder for now)
            # This will be enhanced when we implement real-time social metrics
            social_volume_boost = 1.0
            if hasattr(article, 'social_volume_spike') and article.social_volume_spike:
                social_volume_boost = 1.3
            
            impact_score *= social_volume_boost
            
            # Enhanced impact scoring with cross-source validation
            if hasattr(article, 'cross_source_validation') and article.cross_source_validation:
                boost_factor = self.cross_source_config.get('boost_factor', 1.5)
                impact_score *= boost_factor
            
            # Entity-based impact boost
            if hasattr(article, 'entities') and article.entities:
                entity_boost = min(0.3, len(article.entities) * 0.05)
                impact_score *= (1 + entity_boost)
            
            # Event type impact boost
            if hasattr(article, 'event_types') and article.event_types:
                event_boost = min(0.2, len(article.event_types) * 0.03)
                impact_score *= (1 + event_boost)
            
            # Latency-based impact (faster = higher impact)
            if hasattr(article, 'publish_latency_ms') and article.publish_latency_ms > 0:
                latency_threshold = self.latency_config.get('latency_threshold_ms', 5000)
                if article.publish_latency_ms < latency_threshold:
                    latency_boost = 1 - (article.publish_latency_ms / latency_threshold) * 0.2
                    impact_score *= latency_boost
            
            # Market correlation prediction boost
            if self.market_correlation_config.get('enabled', False):
                try:
                    prediction = await self._predict_market_impact(article)
                    predicted_impact = prediction.get('predicted_impact', 0.0)
                    prediction_confidence = prediction.get('confidence', 0.0)
                    
                    # Boost impact score based on prediction
                    if prediction_confidence > 0.7:
                        impact_score *= (1 + predicted_impact * 0.5)
                    
                    # Store prediction in article for later analysis
                    article.market_impact_prediction = predicted_impact
                    
                except Exception as e:
                    logger.warning(f"⚠️ Market impact prediction failed: {e}")
            
            return min(1.0, impact_score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating impact score: {e}")
            return 0.5
    
    async def _detect_breaking_news_article(self, article: NewsArticle, impact_score: float) -> bool:
        """Detect if article is breaking news"""
        try:
            # Check impact score threshold
            if impact_score >= self.breaking_news_threshold:
                return True
            
            # Check for breaking news indicators from config
            text = f"{article.title} {article.description}".lower()
            
            if any(indicator in text for indicator in self.breaking_indicators):
                return True
            
            # Check publication time (very recent articles)
            # Ensure both times are timezone-aware
            now = datetime.utcnow().replace(tzinfo=None)
            published = article.published_at.replace(tzinfo=None) if article.published_at.tzinfo else article.published_at
            time_diff = now - published
            if time_diff.total_seconds() < 3600:  # Within last hour
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error detecting breaking news: {e}")
            return False
    
    async def _extract_keywords(self, article: NewsArticle) -> List[str]:
        """Extract keywords from news article"""
        try:
            text = f"{article.title} {article.description}".lower()
            
            # Simple keyword extraction based on frequency
            words = re.findall(r'\b\w+\b', text)
            word_freq = {}
            
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            return [word for word, freq in keywords]
            
        except Exception as e:
            logger.error(f"❌ Error extracting keywords: {e}")
            return []
    
    async def detect_breaking_news(self, processed_news: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect breaking news and create alerts"""
        try:
            breaking_news_alerts = []
            
            for news in processed_news:
                if news['breaking_news']:
                    article = news['article']
                    
                    # Generate alert ID
                    alert_id = str(uuid.uuid4())
                    
                    # Determine alert type and priority
                    alert_type = self._determine_alert_type(news)
                    priority = self._determine_priority(news)
                    
                    # Extract affected symbols
                    affected_symbols = self._extract_affected_symbols(news)
                    
                    alert = {
                        'alert_id': alert_id,
                        'news_id': 0,  # Will be set when stored
                        'alert_type': alert_type,
                        'priority': priority,
                        'title': article.title,
                        'summary': article.description,
                        'affected_symbols': affected_symbols,
                        'impact_prediction': news['impact_score'],
                        'confidence': news['confidence']
                    }
                    
                    breaking_news_alerts.append(alert)
            
            logger.info(f"🚨 Detected {len(breaking_news_alerts)} breaking news alerts")
            return breaking_news_alerts
            
        except Exception as e:
            logger.error(f"❌ Error detecting breaking news: {e}")
            return []
    
    def _determine_alert_type(self, news: Dict[str, Any]) -> str:
        """Determine alert type based on news content"""
        topic = news['article'].category
        
        if topic == 'crypto':
            return 'breaking_news'
        elif topic == 'economic':
            return 'market_moving'
        else:
            return 'breaking_news'
    
    def _determine_priority(self, news: Dict[str, Any]) -> str:
        """Determine alert priority based on impact score"""
        impact_score = news['impact_score']
        
        if impact_score >= 0.9:
            return 'critical'
        elif impact_score >= 0.7:
            return 'high'
        elif impact_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _extract_affected_symbols(self, news: Dict[str, Any]) -> List[str]:
        """Extract affected trading symbols from news"""
        try:
            text = f"{news['article'].title} {news['article'].description}".lower()
            
            # Common crypto symbols
            crypto_symbols = ['btc', 'eth', 'ada', 'dot', 'link', 'uni', 'ltc', 'bch']
            affected_symbols = []
            
            for symbol in crypto_symbols:
                if symbol in text:
                    affected_symbols.append(symbol.upper())
            
            return affected_symbols
            
        except Exception as e:
            logger.error(f"❌ Error extracting affected symbols: {e}")
            return []
    
    async def store_news_data(self, processed_news: List[Dict[str, Any]], breaking_news: List[Dict[str, Any]]):
        """Store news data in the database with enhanced social metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store raw news content
                for news in processed_news:
                    article = news['article']
                    
                    # Generate article ID for deduplication
                    article_id = self._make_article_id(
                        article.source, 
                        article.title, 
                        article.published_at.isoformat(), 
                        article.url
                    )
                    
                    # Check if article already exists
                    existing = await conn.fetchval("""
                        SELECT id FROM raw_news_content 
                        WHERE source_guid = $1 OR cryptopanic_id = $2
                    """, article_id, getattr(article, 'cryptopanic_id', None))
                    
                    if existing:
                        logger.debug(f"⏭️ Skipping duplicate article: {article.title[:50]}...")
                        continue
                    
                    await conn.execute("""
                        INSERT INTO raw_news_content (
                            timestamp, title, description, content, url, source, author, published_at,
                            language, category, tags, relevance_score, impact_score, breaking_news,
                            verified_source, sentiment_score, sentiment_label, confidence, keywords, metadata,
                            source_guid, cryptopanic_id, social_volume_spike, dev_activity_score, whale_transaction_count,
                            social_volume_baseline, social_volume_current, correlation_score,
                            rss_feed_url, rss_feed_name, rss_category, rss_published_at, rss_guid, 
                            feed_credibility, rss_priority_level, rss_backfill,
                            entities, event_types, entity_confidence, publish_latency_ms, cross_source_validation,
                            validation_sources, similarity_score, dup_group_id, market_impact_prediction, feed_performance_score,
                            market_regime, btc_dominance, market_volatility, normalized_sentiment, sentiment_confidence,
                            market_cap_total, fear_greed_index, correlation_30m, correlation_2h, correlation_24h,
                            impact_30m, impact_2h, impact_24h, regime_aware_score
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54, $55, $56, $57, $58, $59, $60)
                        ON CONFLICT (timestamp, id) DO NOTHING;
                    """,
                        datetime.utcnow(), article.title, article.description, article.content,
                        article.url, article.source, article.author, article.published_at,
                        article.language, article.category, article.tags or [],
                        news['relevance_score'], news['impact_score'], news['breaking_news'],
                        True, news['sentiment_score'], news['sentiment_label'], news['confidence'],
                        news['keywords'], json.dumps({}),
                        article_id, getattr(article, 'cryptopanic_id', None),
                        getattr(article, 'social_volume_spike', False),
                        getattr(article, 'dev_activity_score', 0.0),
                        getattr(article, 'whale_transaction_count', 0),
                        getattr(article, 'social_volume_baseline', 0.0),
                        getattr(article, 'social_volume_current', 0.0),
                        getattr(article, 'correlation_score', 0.0),
                        getattr(article, 'rss_feed_url', None),
                        getattr(article, 'rss_feed_name', None),
                        getattr(article, 'rss_category', None),
                        getattr(article, 'rss_published_at', None),
                        getattr(article, 'rss_guid', None),
                        getattr(article, 'feed_credibility', None),
                        getattr(article, 'rss_priority_level', None),
                        False,  # rss_backfill - set to False for real-time
                        json.dumps(getattr(article, 'entities', [])),
                        json.dumps(getattr(article, 'event_types', [])),
                        getattr(article, 'entity_confidence', 0.0),
                        getattr(article, 'publish_latency_ms', 0.0),
                        getattr(article, 'cross_source_validation', False),
                        json.dumps(getattr(article, 'validation_sources', [])),
                        getattr(article, 'similarity_score', 0.0),
                        getattr(article, 'dup_group_id', None),
                        getattr(article, 'market_impact_prediction', 0.0),
                        getattr(article, 'feed_performance_score', 0.5),
                        # Market context data
                        getattr(article, 'market_regime', 'neutral'),
                        getattr(article, 'btc_dominance', 48.5),
                        getattr(article, 'market_volatility', 0.025),
                        getattr(article, 'normalized_sentiment', 0.0),
                        getattr(article, 'sentiment_confidence', 0.5),
                        getattr(article, 'market_cap_total', 2500000000000.0),
                        getattr(article, 'fear_greed_index', 55),
                        getattr(article, 'correlation_30m', 0.0),
                        getattr(article, 'correlation_2h', 0.0),
                        getattr(article, 'correlation_24h', 0.0),
                        getattr(article, 'impact_30m', 0.0),
                        getattr(article, 'impact_2h', 0.0),
                        getattr(article, 'impact_24h', 0.0),
                        getattr(article, 'regime_aware_score', 0.0)
                    )
                
                # Store breaking news alerts
                for alert in breaking_news:
                    await conn.execute("""
                        INSERT INTO breaking_news_alerts (
                            timestamp, alert_id, news_id, alert_type, priority, title, summary,
                            affected_symbols, impact_prediction, confidence, sent_to_users, sent_to_websocket, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        ON CONFLICT (timestamp, id) DO NOTHING;
                    """,
                        datetime.utcnow(), alert['alert_id'], alert['news_id'], alert['alert_type'],
                        alert['priority'], alert['title'], alert['summary'], alert['affected_symbols'],
                        alert['impact_prediction'], alert['confidence'], False, False, json.dumps({})
                    )
                
            logger.info("✅ News data stored successfully")
            
        except Exception as e:
            logger.error(f"❌ Error storing news data: {e}")
            raise
    
    async def close(self):
        """Close the processor"""
        if self.session:
            await self.session.close()
        logger.info("Enhanced News and Event Processor closed")
