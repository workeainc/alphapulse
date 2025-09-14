"""
Enhanced Sentiment Analysis Service for AlphaPlus
Advanced sentiment analysis with transformer models and real-time processing
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncpg
from dataclasses import dataclass
import json
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from transformers import pipeline
import torch
import redis
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Enhanced sentiment data structure"""
    symbol: str
    timestamp: datetime
    source: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    volume: int
    keywords: List[str]
    raw_text: str
    processed_text: str
    topic_classification: str
    sarcasm_detected: bool
    context_score: float

@dataclass
class SentimentAggregation:
    """Real-time sentiment aggregation structure"""
    symbol: str
    timestamp: datetime
    window_size: str
    overall_sentiment_score: float
    positive_sentiment_score: float
    negative_sentiment_score: float
    neutral_sentiment_score: float
    source_breakdown: Dict[str, float]
    volume_metrics: Dict[str, int]
    confidence_weighted_score: float
    sentiment_trend: str
    trend_strength: float

class EnhancedSentimentAnalyzer:
    """Advanced sentiment analyzer with transformer models"""
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize sentiment models
        self._initialize_models()
        
        # Processing queues
        self.sentiment_queue = queue.Queue()
        self.aggregation_queue = queue.Queue()
        
        # Start background processors
        self._start_background_processors()
        
    def _initialize_models(self):
        """Initialize advanced transformer-based sentiment models with ensemble capabilities"""
        try:
            logger.info("Initializing advanced transformer sentiment models...")
            
            # Primary sentiment pipeline (FinBERT for financial text)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            # Secondary pipeline for general sentiment (fallback)
            self.general_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Crypto-specific sentiment model (if available)
            try:
                self.crypto_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",  # Using FinBERT as crypto-specific model
                    return_all_scores=True
                )
                self.crypto_model_available = True
            except Exception:
                logger.warning("Crypto-specific model not available, using FinBERT")
                self.crypto_pipeline = self.sentiment_pipeline
                self.crypto_model_available = False
            
            # Initialize VADER for social media
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize TextBlob for general sentiment
            self.textblob_analyzer = TextBlob
            
            # Model ensemble weights
            self.model_weights = {
                'finbert': 0.4,
                'general': 0.2,
                'crypto': 0.2,
                'vader': 0.1,
                'textblob': 0.1
            }
            
            # Model performance tracking
            self.model_performance = {
                'finbert': {'accuracy': 0.85, 'confidence': 0.9},
                'general': {'accuracy': 0.80, 'confidence': 0.8},
                'crypto': {'accuracy': 0.82, 'confidence': 0.85},
                'vader': {'accuracy': 0.75, 'confidence': 0.7},
                'textblob': {'accuracy': 0.78, 'confidence': 0.75}
            }
            
            logger.info("✅ Advanced transformer models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}")
            # Fallback to basic models
            self.sentiment_pipeline = None
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.textblob_analyzer = TextBlob
    
    def _start_background_processors(self):
        """Start background processing threads"""
        self.sentiment_thread = threading.Thread(target=self._process_sentiment_queue, daemon=True)
        self.sentiment_thread.start()
        
        self.aggregation_thread = threading.Thread(target=self._process_aggregation_queue, daemon=True)
        self.aggregation_thread.start()
        
        logger.info("✅ Background processors started")
    
    def _process_sentiment_queue(self):
        """Background thread for processing sentiment data"""
        while True:
            try:
                sentiment_data = self.sentiment_queue.get(timeout=1)
                asyncio.run(self._store_sentiment_data(sentiment_data))
                self.sentiment_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing sentiment data: {e}")
    
    def _process_aggregation_queue(self):
        """Background thread for processing aggregation data"""
        while True:
            try:
                aggregation_data = self.aggregation_queue.get(timeout=1)
                asyncio.run(self._store_aggregation_data(aggregation_data))
                self.aggregation_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing aggregation data: {e}")
    
    async def analyze_text_sentiment(self, text: str, source: str = 'general') -> Dict[str, Any]:
        """Analyze sentiment of text using advanced ensemble models"""
        try:
            # Clean and preprocess text
            processed_text = self._preprocess_text(text)
            
            # Initialize results
            results = {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'sarcasm_detected': False,
                'topic_classification': 'noise',
                'context_score': 0.0,
                'model_breakdown': {},
                'ensemble_confidence': 0.0
            }
            
            # Get ensemble predictions from all models
            ensemble_results = await self._get_ensemble_predictions(processed_text, source)
            
            # Calculate weighted ensemble score
            weighted_score = 0.0
            total_weight = 0.0
            
            for model_name, prediction in ensemble_results.items():
                weight = self.model_weights.get(model_name, 0.1)
                weighted_score += prediction['score'] * weight
                total_weight += weight
                results['model_breakdown'][model_name] = prediction
            
            if total_weight > 0:
                results['sentiment_score'] = weighted_score / total_weight
                results['ensemble_confidence'] = self._calculate_ensemble_confidence(ensemble_results)
            
            # Determine sentiment label
            if results['sentiment_score'] > 0.1:
                results['sentiment_label'] = 'positive'
            elif results['sentiment_score'] < -0.1:
                results['sentiment_label'] = 'negative'
            else:
                results['sentiment_label'] = 'neutral'
            
            # Additional analysis
            results['sarcasm_detected'] = self._detect_sarcasm(processed_text)
            results['topic_classification'] = self._classify_topic(processed_text, source)
            results['context_score'] = self._calculate_context_score(processed_text, source)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'sarcasm_detected': False,
                'topic_classification': 'noise',
                'context_score': 0.0,
                'model_breakdown': {},
                'ensemble_confidence': 0.0
            }
    
    async def _get_ensemble_predictions(self, text: str, source: str) -> Dict[str, Dict]:
        """Get predictions from all ensemble models"""
        ensemble_results = {}
        
        try:
            # FinBERT prediction (primary model)
            if self.sentiment_pipeline:
                finbert_result = self.sentiment_pipeline(text)
                if finbert_result:
                    scores = finbert_result[0]
                    ensemble_results['finbert'] = {
                        'score': self._normalize_sentiment_score(scores),
                        'confidence': self.model_performance['finbert']['confidence'],
                        'raw_scores': scores
                    }
            
            # General model prediction
            if self.general_pipeline:
                general_result = self.general_pipeline(text)
                if general_result:
                    scores = general_result[0]
                    ensemble_results['general'] = {
                        'score': self._normalize_sentiment_score(scores),
                        'confidence': self.model_performance['general']['confidence'],
                        'raw_scores': scores
                    }
            
            # Crypto model prediction
            if self.crypto_pipeline and self.crypto_model_available:
                crypto_result = self.crypto_pipeline(text)
                if crypto_result:
                    scores = crypto_result[0]
                    ensemble_results['crypto'] = {
                        'score': self._normalize_sentiment_score(scores),
                        'confidence': self.model_performance['crypto']['confidence'],
                        'raw_scores': scores
                    }
            
            # VADER prediction
            vader_scores = self.vader_analyzer.polarity_scores(text)
            ensemble_results['vader'] = {
                'score': vader_scores['compound'],
                'confidence': self.model_performance['vader']['confidence'],
                'raw_scores': vader_scores
            }
            
            # TextBlob prediction
            blob = self.textblob_analyzer(text)
            ensemble_results['textblob'] = {
                'score': blob.sentiment.polarity,
                'confidence': self.model_performance['textblob']['confidence'],
                'raw_scores': {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble predictions: {e}")
        
        return ensemble_results
    
    def _normalize_sentiment_score(self, scores: List[Dict]) -> float:
        """Normalize sentiment scores to -1 to 1 range"""
        try:
            # Handle different score formats
            if isinstance(scores, list) and len(scores) > 0:
                # Find the highest scoring label
                max_score = max(scores, key=lambda x: x.get('score', 0))
                label = max_score.get('label', '').lower()
                score = max_score.get('score', 0)
                
                # Normalize based on label
                if 'positive' in label:
                    return score
                elif 'negative' in label:
                    return -score
                else:
                    return 0.0
            return 0.0
        except Exception as e:
            logger.error(f"Error normalizing sentiment score: {e}")
            return 0.0
    
    def _calculate_ensemble_confidence(self, ensemble_results: Dict) -> float:
        """Calculate confidence based on ensemble agreement"""
        if not ensemble_results:
            return 0.0
        
        # Calculate agreement between models
        scores = [result['score'] for result in ensemble_results.values()]
        if len(scores) < 2:
            return 0.5
        
        # Calculate standard deviation (lower = more agreement)
        std_dev = np.std(scores)
        agreement_score = max(0, 1 - std_dev)
        
        # Weight by individual model confidences
        weighted_confidence = sum(
            result['confidence'] * self.model_weights.get(model_name, 0.1)
            for model_name, result in ensemble_results.items()
        ) / sum(self.model_weights.values())
        
        return (agreement_score + weighted_confidence) / 2
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (keep hashtag text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _detect_sarcasm(self, text: str) -> bool:
        """Simple sarcasm detection heuristic"""
        sarcasm_indicators = [
            'lol', 'haha', 'yeah right', 'sure', 'obviously', 'clearly',
            'wow', 'amazing', 'brilliant', 'genius', 'sarcasm', 'irony'
        ]
        
        text_lower = text.lower()
        sarcasm_count = sum(1 for indicator in sarcasm_indicators if indicator in text_lower)
        
        return sarcasm_count >= 2
    
    def _classify_topic(self, text: str, source: str = 'general') -> str:
        """Classify text topic"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['price', 'market', 'trading', 'buy', 'sell']):
            return 'price_moving'
        elif any(word in text_lower for word in ['news', 'announcement', 'update']):
            return 'news'
        elif any(word in text_lower for word in ['think', 'believe', 'opinion']):
            return 'opinion'
        else:
            return 'noise'
    
    def _calculate_context_score(self, text: str, source: str) -> float:
        """Calculate context relevance score"""
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'trading', 'market', 'price', 'bull', 'bear', 'moon', 'dump',
            'pump', 'hodl', 'fomo', 'fud', 'altcoin', 'defi', 'nft'
        ]
        
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in crypto_keywords if keyword in text_lower)
        
        context_score = min(keyword_matches / len(crypto_keywords), 1.0)
        
        if source in ['news', 'financial']:
            context_score *= 1.2
        
        return min(context_score, 1.0)
    
    async def collect_twitter_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect sentiment from Twitter"""
        try:
            base_symbol = symbol.split('/')[0]
            
            # Simulate Twitter data collection
            mock_tweets = [
                f"$BTC looking bullish today! 🚀",
                f"Bitcoin price action is concerning...",
                f"Great analysis on {base_symbol} market structure"
            ]
            
            sentiment_data = []
            
            for tweet in mock_tweets:
                sentiment_result = await self.analyze_text_sentiment(tweet, 'twitter')
                
                sentiment_data.append(SentimentData(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    source='twitter',
                    sentiment_score=sentiment_result['sentiment_score'],
                    sentiment_label=sentiment_result['sentiment_label'],
                    confidence=sentiment_result['confidence'],
                    volume=1,
                    keywords=self._extract_keywords(tweet),
                    raw_text=tweet,
                    processed_text=self._preprocess_text(tweet),
                    topic_classification=sentiment_result['topic_classification'],
                    sarcasm_detected=sentiment_result['sarcasm_detected'],
                    context_score=sentiment_result['context_score']
                ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting Twitter sentiment: {e}")
            return []
    
    async def collect_reddit_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect sentiment from Reddit"""
        try:
            base_symbol = symbol.split('/')[0]
            
            # Simulate Reddit data collection
            mock_posts = [
                f"Thoughts on {base_symbol} current price action?",
                f"{base_symbol} is definitely going to the moon!",
                f"Market analysis: {base_symbol} showing bearish signals"
            ]
            
            sentiment_data = []
            
            for post in mock_posts:
                sentiment_result = await self.analyze_text_sentiment(post, 'reddit')
                
                sentiment_data.append(SentimentData(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    source='reddit',
                    sentiment_score=sentiment_result['sentiment_score'],
                    sentiment_label=sentiment_result['sentiment_label'],
                    confidence=sentiment_result['confidence'],
                    volume=1,
                    keywords=self._extract_keywords(post),
                    raw_text=post,
                    processed_text=self._preprocess_text(post),
                    topic_classification=sentiment_result['topic_classification'],
                    sarcasm_detected=sentiment_result['sarcasm_detected'],
                    context_score=sentiment_result['context_score']
                ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting Reddit sentiment: {e}")
            return []
    
    async def collect_news_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect sentiment from news sources"""
        try:
            base_symbol = symbol.split('/')[0]
            
            # Simulate news data collection
            mock_news = [
                f"Breaking: {base_symbol} reaches new all-time high",
                f"Market Update: {base_symbol} shows bearish momentum",
                f"Analysis: {base_symbol} fundamentals remain strong"
            ]
            
            sentiment_data = []
            
            for news in mock_news:
                sentiment_result = await self.analyze_text_sentiment(news, 'news')
                
                sentiment_data.append(SentimentData(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    source='news',
                    sentiment_score=sentiment_result['sentiment_score'],
                    sentiment_label=sentiment_result['sentiment_label'],
                    confidence=sentiment_result['confidence'],
                    volume=1,
                    keywords=self._extract_keywords(news),
                    raw_text=news,
                    processed_text=self._preprocess_text(news),
                    topic_classification=sentiment_result['topic_classification'],
                    sarcasm_detected=sentiment_result['sarcasm_detected'],
                    context_score=sentiment_result['context_score']
                ))
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error collecting news sentiment: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'trading', 'market', 'price', 'bull', 'bear', 'moon', 'dump',
            'pump', 'hodl', 'fomo', 'fud', 'altcoin', 'defi', 'nft'
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in crypto_keywords if keyword in text_lower]
        
        return found_keywords
    
    async def aggregate_sentiment(self, symbol: str, window_size: str = '5min') -> SentimentAggregation:
        """Aggregate sentiment data for a symbol and time window"""
        try:
            # Get sentiment data for the time window
            end_time = datetime.utcnow()
            if window_size == '1min':
                start_time = end_time - timedelta(minutes=1)
            elif window_size == '5min':
                start_time = end_time - timedelta(minutes=5)
            elif window_size == '15min':
                start_time = end_time - timedelta(minutes=15)
            elif window_size == '1hour':
                start_time = end_time - timedelta(hours=1)
            else:
                start_time = end_time - timedelta(minutes=5)
            
            # Query sentiment data from database
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT * FROM enhanced_sentiment_data 
                    WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                    ORDER BY timestamp DESC
                """
                rows = await conn.fetch(query, symbol, start_time, end_time)
            
            if not rows:
                return self._create_default_aggregation(symbol, window_size)
            
            # Process sentiment data
            sentiment_scores = []
            source_sentiments = {}
            source_volumes = {}
            
            for row in rows:
                sentiment_scores.append(row['sentiment_score'])
                
                source = row['source']
                if source not in source_sentiments:
                    source_sentiments[source] = []
                    source_volumes[source] = 0
                
                source_sentiments[source].append(row['sentiment_score'])
                source_volumes[source] += row['volume'] or 1
            
            # Calculate aggregated metrics
            overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            
            # Calculate breakdown
            positive_count = sum(1 for score in sentiment_scores if score > 0.1)
            negative_count = sum(1 for score in sentiment_scores if score < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            total_count = len(sentiment_scores)
            positive_sentiment = positive_count / total_count if total_count > 0 else 0.0
            negative_sentiment = negative_count / total_count if total_count > 0 else 0.0
            neutral_sentiment = neutral_count / total_count if total_count > 0 else 0.0
            
            # Calculate source breakdown
            source_breakdown = {}
            for source, scores in source_sentiments.items():
                source_breakdown[source] = np.mean(scores) if scores else 0.0
            
            # Calculate confidence weighted score
            confidence_scores = [row['confidence'] for row in rows]
            confidence_weighted_score = np.average(sentiment_scores, weights=confidence_scores) if confidence_scores else 0.0
            
            # Calculate trend
            sentiment_trend, trend_strength = self._calculate_trend(sentiment_scores)
            
            return SentimentAggregation(
                symbol=symbol,
                timestamp=end_time,
                window_size=window_size,
                overall_sentiment_score=overall_sentiment,
                positive_sentiment_score=positive_sentiment,
                negative_sentiment_score=negative_sentiment,
                neutral_sentiment_score=neutral_sentiment,
                source_breakdown=source_breakdown,
                volume_metrics=source_volumes,
                confidence_weighted_score=confidence_weighted_score,
                sentiment_trend=sentiment_trend,
                trend_strength=trend_strength
            )
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment: {e}")
            return self._create_default_aggregation(symbol, window_size)
    
    def _create_default_aggregation(self, symbol: str, window_size: str) -> SentimentAggregation:
        """Create default aggregation when no data is available"""
        return SentimentAggregation(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            window_size=window_size,
            overall_sentiment_score=0.0,
            positive_sentiment_score=0.0,
            negative_sentiment_score=0.0,
            neutral_sentiment_score=1.0,
            source_breakdown={},
            volume_metrics={},
            confidence_weighted_score=0.0,
            sentiment_trend='stable',
            trend_strength=0.0
        )
    
    def _calculate_trend(self, scores: List[float]) -> tuple[str, float]:
        """Calculate sentiment trend and strength"""
        if len(scores) < 2:
            return 'stable', 0.0
        
        # Calculate linear trend
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        
        # Determine trend direction
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate trend strength (absolute slope)
        trend_strength = abs(slope)
        
        return trend, trend_strength
    
    async def _store_sentiment_data(self, sentiment_data: SentimentData):
        """Store sentiment data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO enhanced_sentiment_data (
                        symbol, timestamp, source, sentiment_score, sentiment_label,
                        confidence, volume, keywords, raw_text, processed_text,
                        topic_classification, sarcasm_detected, context_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """
                
                await conn.execute(query,
                    sentiment_data.symbol,
                    sentiment_data.timestamp,
                    sentiment_data.source,
                    sentiment_data.sentiment_score,
                    sentiment_data.sentiment_label,
                    sentiment_data.confidence,
                    sentiment_data.volume,
                    sentiment_data.keywords,
                    sentiment_data.raw_text,
                    sentiment_data.processed_text,
                    sentiment_data.topic_classification,
                    sentiment_data.sarcasm_detected,
                    sentiment_data.context_score
                )
                
        except Exception as e:
            logger.error(f"Error storing sentiment data: {e}")
    
    async def _store_aggregation_data(self, aggregation_data: SentimentAggregation):
        """Store aggregation data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO real_time_sentiment_aggregation (
                        symbol, timestamp, window_size, overall_sentiment_score,
                        positive_sentiment_score, negative_sentiment_score, neutral_sentiment_score,
                        twitter_sentiment, reddit_sentiment, news_sentiment,
                        total_volume, twitter_volume, reddit_volume, news_volume,
                        confidence_weighted_score, sentiment_trend, trend_strength
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """
                
                await conn.execute(query,
                    aggregation_data.symbol,
                    aggregation_data.timestamp,
                    aggregation_data.window_size,
                    aggregation_data.overall_sentiment_score,
                    aggregation_data.positive_sentiment_score,
                    aggregation_data.negative_sentiment_score,
                    aggregation_data.neutral_sentiment_score,
                    aggregation_data.source_breakdown.get('twitter'),
                    aggregation_data.source_breakdown.get('reddit'),
                    aggregation_data.source_breakdown.get('news'),
                    sum(aggregation_data.volume_metrics.values()),
                    aggregation_data.volume_metrics.get('twitter', 0),
                    aggregation_data.volume_metrics.get('reddit', 0),
                    aggregation_data.volume_metrics.get('news', 0),
                    aggregation_data.confidence_weighted_score,
                    aggregation_data.sentiment_trend,
                    aggregation_data.trend_strength
                )
                
        except Exception as e:
            logger.error(f"Error storing aggregation data: {e}")
    
    async def collect_all_sentiment(self, symbol: str) -> List[SentimentData]:
        """Collect sentiment from all sources"""
        try:
            all_sentiment = []
            
            # Collect from different sources
            twitter_sentiment = await self.collect_twitter_sentiment(symbol)
            reddit_sentiment = await self.collect_reddit_sentiment(symbol)
            news_sentiment = await self.collect_news_sentiment(symbol)
            
            all_sentiment.extend(twitter_sentiment)
            all_sentiment.extend(reddit_sentiment)
            all_sentiment.extend(news_sentiment)
            
            # Queue for background processing
            for sentiment_data in all_sentiment:
                self.sentiment_queue.put(sentiment_data)
            
            return all_sentiment
            
        except Exception as e:
            logger.error(f"Error collecting all sentiment: {e}")
            return []
    
    async def get_latest_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get latest sentiment summary for a symbol"""
        try:
            # Get latest aggregation
            aggregation = await self.aggregate_sentiment(symbol, '5min')
            
            # Queue for background storage
            self.aggregation_queue.put(aggregation)
            
            return {
                'symbol': aggregation.symbol,
                'timestamp': aggregation.timestamp.isoformat(),
                'overall_sentiment_score': aggregation.overall_sentiment_score,
                'positive_sentiment_score': aggregation.positive_sentiment_score,
                'negative_sentiment_score': aggregation.negative_sentiment_score,
                'neutral_sentiment_score': aggregation.neutral_sentiment_score,
                'source_breakdown': aggregation.source_breakdown,
                'volume_metrics': aggregation.volume_metrics,
                'confidence_weighted_score': aggregation.confidence_weighted_score,
                'sentiment_trend': aggregation.sentiment_trend,
                'trend_strength': aggregation.trend_strength
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {e}")
            return {}
    
    async def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("Enhanced sentiment analyzer closed")

    # ===== PHASE 4A: PREDICTIVE ANALYTICS METHODS =====
    
    async def predict_price_movement(self, symbol: str, time_horizon: str = '4h') -> Dict[str, Any]:
        """Predict price movement based on sentiment analysis"""
        try:
            # Get recent sentiment data
            sentiment_summary = await self.get_latest_sentiment_summary(symbol)
            
            # Get technical indicators (placeholder for integration)
            technical_indicators = await self._get_technical_indicators(symbol)
            
            # Calculate prediction probability
            prediction = self._calculate_price_prediction(
                sentiment_summary, 
                technical_indicators, 
                time_horizon
            )
            
            # Store prediction
            await self._store_prediction(symbol, prediction, time_horizon)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return self._create_default_prediction(symbol, time_horizon)
    
    def _calculate_price_prediction(self, sentiment_data: Dict, technical_data: Dict, time_horizon: str) -> Dict[str, Any]:
        """Calculate price movement prediction probability"""
        try:
            # Base sentiment score
            sentiment_score = sentiment_data.get('overall_sentiment_score', 0.0)
            confidence = sentiment_data.get('confidence_weighted_score', 0.0)
            
            # Technical indicators (simplified)
            rsi = technical_data.get('rsi', 50)
            macd = technical_data.get('macd', 0.0)
            volume = technical_data.get('volume', 1.0)
            
            # Calculate prediction probability
            # Positive sentiment + bullish technical = higher probability of price increase
            base_probability = 0.5  # Neutral starting point
            
            # Sentiment contribution (40% weight)
            sentiment_contribution = sentiment_score * 0.4
            
            # Technical contribution (30% weight)
            technical_contribution = 0.0
            if rsi < 30:  # Oversold
                technical_contribution += 0.1
            elif rsi > 70:  # Overbought
                technical_contribution -= 0.1
            
            if macd > 0:  # Bullish MACD
                technical_contribution += 0.1
            elif macd < 0:  # Bearish MACD
                technical_contribution -= 0.1
            
            # Volume contribution (20% weight)
            volume_contribution = (volume - 1.0) * 0.2  # Normalize volume
            
            # Confidence contribution (10% weight)
            confidence_contribution = (confidence - 0.5) * 0.2
            
            # Calculate final probability
            final_probability = base_probability + sentiment_contribution + technical_contribution + volume_contribution + confidence_contribution
            
            # Clamp to 0-1 range
            final_probability = max(0.0, min(1.0, final_probability))
            
            # Determine direction
            if final_probability > 0.6:
                direction = 'bullish'
                strength = 'strong' if final_probability > 0.8 else 'moderate'
            elif final_probability < 0.4:
                direction = 'bearish'
                strength = 'strong' if final_probability < 0.2 else 'moderate'
            else:
                direction = 'neutral'
                strength = 'weak'
            
            return {
                'symbol': sentiment_data.get('symbol', ''),
                'timestamp': datetime.utcnow().isoformat(),
                'time_horizon': time_horizon,
                'prediction_probability': final_probability,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'sentiment_score': sentiment_score,
                'technical_indicators': technical_data,
                'factors': {
                    'sentiment_contribution': sentiment_contribution,
                    'technical_contribution': technical_contribution,
                    'volume_contribution': volume_contribution,
                    'confidence_contribution': confidence_contribution
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating price prediction: {e}")
            return self._create_default_prediction(sentiment_data.get('symbol', ''), time_horizon)
    
    async def _get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """Get technical indicators for prediction (placeholder for integration)"""
        # This would integrate with your existing technical analysis system
        # For now, return mock data
        return {
            'rsi': 55.0,  # Relative Strength Index
            'macd': 0.02,  # MACD line
            'volume': 1.2,  # Volume ratio
            'bollinger_position': 0.6,  # Position within Bollinger Bands
            'support_level': 45000.0,  # Nearest support
            'resistance_level': 48000.0  # Nearest resistance
        }
    
    def _create_default_prediction(self, symbol: str, time_horizon: str) -> Dict[str, Any]:
        """Create default prediction when calculation fails"""
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'time_horizon': time_horizon,
            'prediction_probability': 0.5,
            'direction': 'neutral',
            'strength': 'weak',
            'confidence': 0.0,
            'sentiment_score': 0.0,
            'technical_indicators': {},
            'factors': {
                'sentiment_contribution': 0.0,
                'technical_contribution': 0.0,
                'volume_contribution': 0.0,
                'confidence_contribution': 0.0
            }
        }
    
    async def _store_prediction(self, symbol: str, prediction: Dict, time_horizon: str):
        """Store prediction in database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO sentiment_predictions (
                        symbol, timestamp, time_horizon, prediction_probability,
                        direction, strength, confidence, sentiment_score,
                        technical_indicators, factors
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """
                
                await conn.execute(query,
                    symbol,
                    datetime.fromisoformat(prediction['timestamp']),
                    time_horizon,
                    prediction['prediction_probability'],
                    prediction['direction'],
                    prediction['strength'],
                    prediction['confidence'],
                    prediction['sentiment_score'],
                    json.dumps(prediction['technical_indicators']),
                    json.dumps(prediction['factors'])
                )
                
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    # ===== PHASE 4B: CROSS-ASSET CORRELATION METHODS =====
    
    async def analyze_cross_asset_sentiment(self, primary_symbol: str, secondary_symbols: List[str] = None) -> Dict[str, Any]:
        """Analyze sentiment correlation across multiple assets"""
        try:
            if secondary_symbols is None:
                secondary_symbols = ['ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
            
            # Get sentiment for all symbols
            all_sentiments = {}
            all_sentiments[primary_symbol] = await self.get_latest_sentiment_summary(primary_symbol)
            
            for symbol in secondary_symbols:
                all_sentiments[symbol] = await self.get_latest_sentiment_summary(symbol)
            
            # Calculate correlations
            correlations = self._calculate_sentiment_correlations(all_sentiments)
            
            # Calculate market-wide sentiment
            market_sentiment = self._calculate_market_sentiment(all_sentiments)
            
            # Store cross-asset data
            await self._store_cross_asset_data(primary_symbol, correlations, market_sentiment)
            
            return {
                'primary_symbol': primary_symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'individual_sentiments': all_sentiments,
                'correlations': correlations,
                'market_sentiment': market_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cross-asset sentiment: {e}")
            return {}
    
    def _calculate_sentiment_correlations(self, sentiments: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate sentiment correlations between assets"""
        correlations = {}
        symbols = list(sentiments.keys())
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                sentiment1 = sentiments[symbol1].get('overall_sentiment_score', 0.0)
                sentiment2 = sentiments[symbol2].get('overall_sentiment_score', 0.0)
                
                # Simple correlation calculation
                correlation = self._calculate_correlation([sentiment1], [sentiment2])
                pair_name = f"{symbol1}_vs_{symbol2}"
                correlations[pair_name] = correlation
        
        return correlations
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient between two lists"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            return np.corrcoef(x, y)[0, 1]
        except:
            return 0.0
    
    def _calculate_market_sentiment(self, sentiments: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall market sentiment"""
        try:
            sentiment_scores = [data.get('overall_sentiment_score', 0.0) for data in sentiments.values()]
            confidence_scores = [data.get('confidence_weighted_score', 0.0) for data in sentiments.values()]
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Determine market mood
            if avg_sentiment > 0.2:
                market_mood = 'bullish'
            elif avg_sentiment < -0.2:
                market_mood = 'bearish'
            else:
                market_mood = 'neutral'
            
            return {
                'average_sentiment': avg_sentiment,
                'average_confidence': avg_confidence,
                'market_mood': market_mood,
                'sentiment_volatility': np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0.0,
                'asset_count': len(sentiments)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return {
                'average_sentiment': 0.0,
                'average_confidence': 0.0,
                'market_mood': 'neutral',
                'sentiment_volatility': 0.0,
                'asset_count': 0
            }
    
    async def _store_cross_asset_data(self, primary_symbol: str, correlations: Dict, market_sentiment: Dict):
        """Store cross-asset sentiment data"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO cross_asset_sentiment (
                        primary_symbol, timestamp, correlations, market_sentiment
                    ) VALUES ($1, $2, $3, $4)
                """
                
                await conn.execute(query,
                    primary_symbol,
                    datetime.utcnow(),
                    json.dumps(correlations),
                    json.dumps(market_sentiment)
                )
                
        except Exception as e:
            logger.error(f"Error storing cross-asset data: {e}")
    
    # ===== PHASE 4C: CONTINUOUS LEARNING METHODS =====
    
    async def update_model_performance(self, actual_outcomes: List[Dict]):
        """Update model performance based on actual outcomes"""
        try:
            for outcome in actual_outcomes:
                symbol = outcome.get('symbol')
                predicted_probability = outcome.get('predicted_probability', 0.5)
                actual_direction = outcome.get('actual_direction', 'neutral')
                actual_price_change = outcome.get('actual_price_change', 0.0)
                
                # Calculate prediction accuracy
                predicted_direction = 'bullish' if predicted_probability > 0.6 else 'bearish' if predicted_probability < 0.4 else 'neutral'
                accuracy = 1.0 if predicted_direction == actual_direction else 0.0
                
                # Update model performance
                await self._update_model_metrics(symbol, accuracy, predicted_probability, actual_price_change)
                
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    async def _update_model_metrics(self, symbol: str, accuracy: float, predicted_prob: float, actual_change: float):
        """Update model performance metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    INSERT INTO model_performance_metrics (
                        symbol, timestamp, accuracy, predicted_probability,
                        actual_price_change, model_version
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """
                
                await conn.execute(query,
                    symbol,
                    datetime.utcnow(),
                    accuracy,
                    predicted_prob,
                    actual_change,
                    'v1.0'  # Current model version
                )
                
        except Exception as e:
            logger.error(f"Error updating model metrics: {e}")
    
    async def get_model_performance_summary(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get model performance summary"""
        try:
            async with self.db_pool.acquire() as conn:
                if symbol:
                    query = """
                        SELECT AVG(accuracy) as avg_accuracy, 
                               COUNT(*) as total_predictions,
                               AVG(predicted_probability) as avg_predicted_prob,
                               AVG(actual_price_change) as avg_actual_change
                        FROM model_performance_metrics 
                        WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '$2 days'
                    """
                    row = await conn.fetchrow(query, symbol, days)
                else:
                    query = """
                        SELECT AVG(accuracy) as avg_accuracy, 
                               COUNT(*) as total_predictions,
                               AVG(predicted_probability) as avg_predicted_prob,
                               AVG(actual_price_change) as avg_actual_change
                        FROM model_performance_metrics 
                        WHERE timestamp >= NOW() - INTERVAL '$1 days'
                    """
                    row = await conn.fetchrow(query, days)
                
                if row:
                    return {
                        'average_accuracy': float(row['avg_accuracy'] or 0.0),
                        'total_predictions': int(row['total_predictions'] or 0),
                        'average_predicted_probability': float(row['avg_predicted_prob'] or 0.0),
                        'average_actual_change': float(row['avg_actual_change'] or 0.0),
                        'performance_period_days': days
                    }
                else:
                    return {
                        'average_accuracy': 0.0,
                        'total_predictions': 0,
                        'average_predicted_probability': 0.0,
                        'average_actual_change': 0.0,
                        'performance_period_days': days
                    }
                    
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
