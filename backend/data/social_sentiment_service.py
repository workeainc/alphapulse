"""
Social Sentiment Service for AlphaPulse
Provides social media sentiment analysis and clustering for market correlation
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import json
import numpy as np

# Import required libraries for sentiment analysis
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available - clustering disabled")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - using mock sentiment analysis")

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    """Social media post data"""
    post_id: str
    text: str
    timestamp: datetime
    source: str  # 'twitter', 'reddit', etc.
    author: Optional[str]
    engagement: Optional[Dict[str, int]]  # likes, retweets, etc.
    metadata: Dict[str, Any]

@dataclass
class SentimentAnalysis:
    """Sentiment analysis result"""
    post_id: str
    text: str
    timestamp: datetime
    sentiment_score: float
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class ClusteredSentiment:
    """Clustered sentiment data"""
    symbol: str
    timestamp: datetime
    cluster_label: int
    sentiment_score: float
    post_count: int
    cluster_characteristics: Dict[str, Any]
    metadata: Dict[str, Any]

class SocialSentimentService:
    """Social sentiment analysis and clustering service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Service configuration
        self.twitter_api_key = self.config.get('twitter_api_key', '')
        self.twitter_api_secret = self.config.get('twitter_api_secret', '')
        self.twitter_bearer_token = self.config.get('twitter_bearer_token', '')
        
        # Sentiment analysis configuration
        self.enable_sentiment_analysis = self.config.get('enable_sentiment_analysis', True)
        self.enable_clustering = self.config.get('enable_clustering', True)
        self.cluster_count = self.config.get('cluster_count', 3)
        
        # Data configuration
        self.symbols = self.config.get('symbols', ['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        self.search_keywords = self.config.get('search_keywords', ['crypto', 'bitcoin', 'ethereum'])
        self.update_frequency = self.config.get('update_frequency', 60.0)  # seconds
        
        # Sentiment analyzer
        self.sentiment_analyzer = None
        self.kmeans = None
        
        # Data buffers
        self.posts_buffer = defaultdict(list)  # symbol -> posts
        self.sentiment_buffer = defaultdict(list)  # symbol -> sentiment
        self.clusters_buffer = defaultdict(list)  # symbol -> clusters
        
        # Performance tracking
        self.stats = {
            'total_posts_processed': 0,
            'sentiment_analyses': 0,
            'clustering_operations': 0,
            'last_update': None,
            'processing_times': deque(maxlen=100)
        }
        
        # Callbacks
        self.sentiment_callbacks = defaultdict(list)  # event_type -> [callback]
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize sentiment analysis and clustering components"""
        try:
            # Initialize sentiment analyzer
            if TRANSFORMERS_AVAILABLE and self.enable_sentiment_analysis:
                try:
                    self.sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
                    self.logger.info("Sentiment analyzer initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize sentiment analyzer: {e}")
                    self.sentiment_analyzer = None
            
            # Initialize clustering
            if SKLEARN_AVAILABLE and self.enable_clustering:
                self.kmeans = KMeans(n_clusters=self.cluster_count, random_state=42)
                self.logger.info("K-means clustering initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
    
    async def initialize(self):
        """Initialize the social sentiment service"""
        try:
            self.logger.info("Initializing Social Sentiment Service...")
            
            # Test Twitter API connection if credentials are provided
            if self.twitter_bearer_token:
                await self._test_twitter_connection()
            
            self.logger.info("Social Sentiment Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize social sentiment service: {e}")
            raise
    
    async def _test_twitter_connection(self):
        """Test Twitter API connection"""
        try:
            # This is a placeholder for Twitter API testing
            # In production, you would make an actual API call
            self.logger.info("Twitter API credentials configured")
            
        except Exception as e:
            self.logger.warning(f"Twitter API connection test failed: {e}")
    
    async def fetch_social_posts(self, symbol: str, limit: int = 100) -> List[SocialPost]:
        """Fetch social media posts for a symbol"""
        try:
            # For now, generate mock data since we don't have actual API access
            # In production, this would integrate with Twitter/X API
            return self._generate_mock_social_posts(symbol, limit)
            
        except Exception as e:
            self.logger.error(f"Error fetching social posts for {symbol}: {e}")
            return []
    
    async def analyze_sentiment(self, posts: List[SocialPost]) -> List[SentimentAnalysis]:
        """Analyze sentiment for a list of posts"""
        try:
            if not self.sentiment_analyzer:
                # Use mock sentiment analysis
                return self._generate_mock_sentiment_analysis(posts)
            
            sentiment_results = []
            
            for post in posts:
                try:
                    # Analyze sentiment using transformers
                    result = self.sentiment_analyzer(post.text[:512])  # Limit text length
                    
                    sentiment_score = result[0]['score']
                    sentiment_label = result[0]['label'].lower()
                    
                    # Convert label to score (-1 to 1)
                    if sentiment_label == 'negative':
                        sentiment_score = -sentiment_score
                    elif sentiment_label == 'neutral':
                        sentiment_score = 0.0
                    
                    analysis = SentimentAnalysis(
                        post_id=post.post_id,
                        text=post.text,
                        timestamp=post.timestamp,
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label,
                        confidence=result[0]['score'],
                        metadata={'model': 'distilbert', 'text_length': len(post.text)}
                    )
                    
                    sentiment_results.append(analysis)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze sentiment for post {post.post_id}: {e}")
                    continue
            
            # Update statistics
            self.stats['sentiment_analyses'] += len(sentiment_results)
            
            return sentiment_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return []
    
    async def cluster_sentiment_data(self, symbol: str, sentiment_data: List[SentimentAnalysis]) -> List[ClusteredSentiment]:
        """Cluster sentiment data using K-means"""
        try:
            if not self.kmeans or not sentiment_data:
                return []
            
            # Prepare features for clustering
            features = []
            for analysis in sentiment_data:
                # Feature vector: [sentiment_score, timestamp_hour, text_length]
                hour = analysis.timestamp.hour
                text_length = len(analysis.text)
                features.append([analysis.sentiment_score, hour, text_length])
            
            if len(features) < self.cluster_count:
                self.logger.warning(f"Insufficient data for clustering {symbol}: {len(features)} < {self.cluster_count}")
                return []
            
            # Perform clustering
            features_array = np.array(features)
            cluster_labels = self.kmeans.fit_predict(features_array)
            
            # Group by clusters
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(sentiment_data[i])
            
            # Create cluster summaries
            cluster_results = []
            for label, cluster_posts in clusters.items():
                if not cluster_posts:
                    continue
                
                # Calculate cluster characteristics
                sentiment_scores = [post.sentiment_score for post in cluster_posts]
                avg_sentiment = np.mean(sentiment_scores)
                post_count = len(cluster_posts)
                
                cluster_characteristics = {
                    'avg_sentiment': avg_sentiment,
                    'sentiment_std': np.std(sentiment_scores),
                    'post_count': post_count,
                    'time_range': {
                        'start': min(post.timestamp for post in cluster_posts),
                        'end': max(post.timestamp for post in cluster_posts)
                    }
                }
                
                clustered_sentiment = ClusteredSentiment(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    cluster_label=label,
                    sentiment_score=avg_sentiment,
                    post_count=post_count,
                    cluster_characteristics=cluster_characteristics,
                    metadata={'cluster_size': post_count, 'features_used': ['sentiment', 'hour', 'length']}
                )
                
                cluster_results.append(clustered_sentiment)
            
            # Update statistics
            self.stats['clustering_operations'] += 1
            
            return cluster_results
            
        except Exception as e:
            self.logger.error(f"Error clustering sentiment data for {symbol}: {e}")
            return []
    
    async def process_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Process sentiment analysis for a symbol"""
        try:
            start_time = time.time()
            
            # Fetch social posts
            posts = await self.fetch_social_posts(symbol, 50)
            if not posts:
                return {}
            
            # Analyze sentiment
            sentiment_results = await self.analyze_sentiment(posts)
            if not sentiment_results:
                return {}
            
            # Cluster sentiment data
            clusters = await self.cluster_sentiment_data(symbol, sentiment_results)
            
            # Store in buffers
            self.posts_buffer[symbol].extend(posts)
            self.sentiment_buffer[symbol].extend(sentiment_results)
            self.clusters_buffer[symbol].extend(clusters)
            
            # Maintain buffer sizes
            self._maintain_buffer_sizes(symbol)
            
            # Calculate summary metrics
            summary = self._calculate_sentiment_summary(symbol, sentiment_results, clusters)
            
            # Update statistics
            self.stats['total_posts_processed'] += len(posts)
            self.stats['last_update'] = datetime.now()
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            # Trigger callbacks
            await self._trigger_callbacks('sentiment_updated', {
                'symbol': symbol,
                'summary': summary,
                'timestamp': datetime.now()
            })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error processing sentiment for {symbol}: {e}")
            return {}
    
    def _calculate_sentiment_summary(self, symbol: str, sentiment_results: List[SentimentAnalysis], 
                                   clusters: List[ClusteredSentiment]) -> Dict[str, Any]:
        """Calculate sentiment summary metrics"""
        try:
            if not sentiment_results:
                return {}
            
            sentiment_scores = [result.sentiment_score for result in sentiment_results]
            
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'total_posts': len(sentiment_results),
                'avg_sentiment': np.mean(sentiment_scores),
                'sentiment_std': np.std(sentiment_scores),
                'positive_posts': len([s for s in sentiment_scores if s > 0.1]),
                'negative_posts': len([s for s in sentiment_scores if s < -0.1]),
                'neutral_posts': len([s for s in sentiment_scores if -0.1 <= s <= 0.1]),
                'clusters': len(clusters),
                'cluster_summary': [
                    {
                        'label': cluster.cluster_label,
                        'avg_sentiment': cluster.sentiment_score,
                        'post_count': cluster.post_count
                    }
                    for cluster in clusters
                ]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment summary: {e}")
            return {}
    
    def _maintain_buffer_sizes(self, symbol: str):
        """Maintain buffer sizes for memory management"""
        try:
            max_posts = 1000
            max_sentiment = 1000
            max_clusters = 500
            
            # Trim posts buffer
            if len(self.posts_buffer[symbol]) > max_posts:
                excess = len(self.posts_buffer[symbol]) - max_posts
                for _ in range(excess):
                    self.posts_buffer[symbol].pop(0)
            
            # Trim sentiment buffer
            if len(self.sentiment_buffer[symbol]) > max_sentiment:
                excess = len(self.sentiment_buffer[symbol]) - max_sentiment
                for _ in range(excess):
                    self.sentiment_buffer[symbol].pop(0)
            
            # Trim clusters buffer
            if len(self.clusters_buffer[symbol]) > max_clusters:
                excess = len(self.clusters_buffer[symbol]) - max_clusters
                for _ in range(excess):
                    self.clusters_buffer[symbol].pop(0)
                    
        except Exception as e:
            self.logger.error(f"Error maintaining buffer sizes: {e}")
    
    # Mock data generation methods
    def _generate_mock_social_posts(self, symbol: str, limit: int) -> List[SocialPost]:
        """Generate mock social media posts for testing"""
        try:
            posts = []
            base_texts = [
                f"Just bought some {symbol.split('/')[0]}! 🚀",
                f"{symbol.split('/')[0]} looking bullish today! 📈",
                f"Market sentiment for {symbol.split('/')[0]} is mixed 🤔",
                f"Great time to accumulate {symbol.split('/')[0]} 💎",
                f"{symbol.split('/')[0]} price action is interesting 📊"
            ]
            
            for i in range(limit):
                post_id = f"post_{symbol.replace('/', '_')}_{int(time.time())}_{i}"
                text = np.random.choice(base_texts)
                timestamp = datetime.now() - timedelta(minutes=np.random.randint(1, 60))
                
                post = SocialPost(
                    post_id=post_id,
                    text=text,
                    timestamp=timestamp,
                    source='twitter',
                    author=f"user_{np.random.randint(1000, 9999)}",
                    engagement={
                        'likes': np.random.randint(0, 100),
                        'retweets': np.random.randint(0, 50),
                        'replies': np.random.randint(0, 20)
                    },
                    metadata={'mock': True, 'generated_at': datetime.now().isoformat()}
                )
                
                posts.append(post)
            
            return posts
            
        except Exception as e:
            self.logger.error(f"Error generating mock social posts: {e}")
            return []
    
    def _generate_mock_sentiment_analysis(self, posts: List[SocialPost]) -> List[SentimentAnalysis]:
        """Generate mock sentiment analysis for testing"""
        try:
            sentiment_results = []
            
            for post in posts:
                # Generate realistic sentiment scores
                sentiment_score = np.random.uniform(-1.0, 1.0)
                
                if sentiment_score > 0.3:
                    sentiment_label = 'positive'
                elif sentiment_score < -0.3:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                
                analysis = SentimentAnalysis(
                    post_id=post.post_id,
                    text=post.text,
                    timestamp=post.timestamp,
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label,
                    confidence=np.random.uniform(0.7, 0.95),
                    metadata={'mock': True, 'method': 'mock_analysis'}
                )
                
                sentiment_results.append(analysis)
            
            return sentiment_results
            
        except Exception as e:
            self.logger.error(f"Error generating mock sentiment analysis: {e}")
            return []
    
    # Public methods
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for sentiment events"""
        self.sentiment_callbacks[event_type].append(callback)
        self.logger.info(f"Added callback for {event_type} events")
    
    async def _trigger_callbacks(self, event_type: str, data: Any):
        """Trigger callbacks for sentiment events"""
        callbacks = self.sentiment_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"Error in {event_type} callback: {e}")
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'stats': self.stats,
            'symbols': self.symbols,
            'buffer_sizes': {
                'posts': sum(len(posts) for posts in self.posts_buffer.values()),
                'sentiment': sum(len(sentiment) for sentiment in self.sentiment_buffer.values()),
                'clusters': sum(len(clusters) for clusters in self.clusters_buffer.values())
            },
            'last_update': self.stats['last_update'].isoformat() if self.stats['last_update'] else None
        }
    
    async def close(self):
        """Close the social sentiment service"""
        try:
            self.logger.info("Social Sentiment Service closed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to close social sentiment service: {e}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
