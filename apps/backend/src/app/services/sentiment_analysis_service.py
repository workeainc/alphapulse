#!/usr/bin/env python3
"""
Sentiment Analysis Service
Handles news and social media sentiment analysis for trading signals
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Sentiment analysis imports
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import requests
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    logging.warning("Sentiment analysis libraries not available. Install with: pip install textblob vaderSentiment")

logger = logging.getLogger(__name__)

class SentimentSource(Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    COMBINED = "combined"

class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    symbol: str
    source: SentimentSource
    sentiment_score: float
    sentiment_label: SentimentLabel
    confidence_score: float
    text_content: str
    source_url: Optional[str]
    author: Optional[str]
    engagement_metrics: Dict[str, Any]
    keywords: List[str]
    language: str
    timestamp: datetime

class SentimentAnalysisService:
    """Service for analyzing sentiment from news and social media"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentiment analyzers
        if SENTIMENT_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.crypto_keywords = [
                'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
                'blockchain', 'defi', 'nft', 'altcoin', 'moon', 'pump', 'dump',
                'hodl', 'fomo', 'fud', 'bull', 'bear', 'mooning', 'crashing'
            ]
        else:
            self.vader_analyzer = None
            self.crypto_keywords = []
        
        self.logger.info("📰 Sentiment Analysis Service initialized")
    
    async def analyze_sentiment(self, symbol: str, text_content: str, source: SentimentSource,
                              source_url: str = None, author: str = None) -> Optional[SentimentResult]:
        """Analyze sentiment of text content"""
        try:
            if not SENTIMENT_AVAILABLE:
                self.logger.warning("⚠️ Sentiment analysis libraries not available")
                return None
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text_content)
            
            # Analyze sentiment using multiple methods
            vader_scores = self.vader_analyzer.polarity_scores(cleaned_text)
            textblob_sentiment = TextBlob(cleaned_text).sentiment
            
            # Combine sentiment scores
            combined_score = self._combine_sentiment_scores(vader_scores, textblob_sentiment)
            
            # Determine sentiment label
            sentiment_label = self._determine_sentiment_label(combined_score)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(vader_scores, textblob_sentiment)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_text)
            
            # Calculate engagement metrics (simplified)
            engagement_metrics = self._calculate_engagement_metrics(cleaned_text)
            
            # Create sentiment result
            result = SentimentResult(
                symbol=symbol,
                source=source,
                sentiment_score=combined_score,
                sentiment_label=sentiment_label,
                confidence_score=confidence_score,
                text_content=cleaned_text,
                source_url=source_url,
                author=author,
                engagement_metrics=engagement_metrics,
                keywords=keywords,
                language='en',
                timestamp=datetime.now()
            )
            
            # Store sentiment result
            await self._store_sentiment_result(result)
            
            self.logger.info(f"📊 Sentiment analyzed for {symbol}: {sentiment_label.value} ({combined_score:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing sentiment: {e}")
            return None
    
    async def get_sentiment_summary(self, symbol: str, hours: int = 24) -> Dict:
        """Get sentiment summary for the last N hours"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT 
                    source,
                    sentiment_label,
                    COUNT(*) as count,
                    AVG(sentiment_score) as avg_score,
                    AVG(confidence_score) as avg_confidence
                FROM sentiment_analysis
                WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '$2 hours'
                GROUP BY source, sentiment_label
                ORDER BY source, sentiment_label
                """
                
                rows = await conn.fetch(query, symbol, hours)
                
                summary = {
                    'symbol': symbol,
                    'timeframe_hours': hours,
                    'total_mentions': 0,
                    'sources': {},
                    'overall_sentiment': {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    },
                    'sentiment_score': 0.0,
                    'confidence_score': 0.0
                }
                
                total_score = 0.0
                total_confidence = 0.0
                total_count = 0
                
                for row in rows:
                    source = row['source']
                    label = row['sentiment_label']
                    count = row['count']
                    avg_score = float(row['avg_score']) if row['avg_score'] else 0.0
                    avg_confidence = float(row['avg_confidence']) if row['avg_confidence'] else 0.0
                    
                    # Initialize source if not exists
                    if source not in summary['sources']:
                        summary['sources'][source] = {
                            'positive': 0,
                            'negative': 0,
                            'neutral': 0,
                            'total': 0,
                            'avg_score': 0.0
                        }
                    
                    # Update source statistics
                    summary['sources'][source][label] = count
                    summary['sources'][source]['total'] += count
                    summary['sources'][source]['avg_score'] = avg_score
                    
                    # Update overall statistics
                    summary['overall_sentiment'][label] += count
                    summary['total_mentions'] += count
                    
                    total_score += avg_score * count
                    total_confidence += avg_confidence * count
                    total_count += count
                
                # Calculate overall averages
                if total_count > 0:
                    summary['sentiment_score'] = total_score / total_count
                    summary['confidence_score'] = total_confidence / total_count
                
                return summary
                
        except Exception as e:
            self.logger.error(f"❌ Error getting sentiment summary: {e}")
            return {}
    
    async def get_sentiment_trends(self, symbol: str, hours: int = 24) -> Dict:
        """Get sentiment trends over time"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    AVG(sentiment_score) as avg_score,
                    COUNT(*) as mention_count,
                    AVG(confidence_score) as avg_confidence
                FROM sentiment_analysis
                WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '$2 hours'
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour
                """
                
                rows = await conn.fetch(query, symbol, hours)
                
                trends = {
                    'symbol': symbol,
                    'timeframe_hours': hours,
                    'hourly_data': []
                }
                
                for row in rows:
                    trends['hourly_data'].append({
                        'hour': row['hour'].isoformat(),
                        'avg_sentiment_score': float(row['avg_score']) if row['avg_score'] else 0.0,
                        'mention_count': row['mention_count'],
                        'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0.0
                    })
                
                return trends
                
        except Exception as e:
            self.logger.error(f"❌ Error getting sentiment trends: {e}")
            return {}
    
    async def get_sentiment_correlation(self, symbol: str, hours: int = 24) -> Dict:
        """Get correlation between sentiment and price movements"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get sentiment data
                sentiment_query = """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    AVG(sentiment_score) as avg_sentiment
                FROM sentiment_analysis
                WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '$2 hours'
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour
                """
                
                sentiment_rows = await conn.fetch(sentiment_query, symbol, hours)
                
                # Get price data (assuming you have a price table)
                price_query = """
                SELECT 
                    DATE_TRUNC('hour', timestamp) as hour,
                    AVG(close) as avg_price
                FROM market_data
                WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '$2 hours'
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour
                """
                
                price_rows = await conn.fetch(price_query, symbol, hours)
                
                # Calculate correlation
                sentiment_data = {row['hour']: float(row['avg_sentiment']) for row in sentiment_rows}
                price_data = {row['hour']: float(row['avg_price']) for row in price_rows}
                
                # Find common hours
                common_hours = set(sentiment_data.keys()) & set(price_data.keys())
                
                if len(common_hours) < 2:
                    return {'correlation': 0.0, 'data_points': 0}
                
                sentiment_values = [sentiment_data[hour] for hour in sorted(common_hours)]
                price_values = [price_data[hour] for hour in sorted(common_hours)]
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(sentiment_values, price_values)[0, 1]
                
                return {
                    'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                    'data_points': len(common_hours),
                    'sentiment_values': sentiment_values,
                    'price_values': price_values
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating sentiment correlation: {e}")
            return {'correlation': 0.0, 'data_points': 0}
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\.,!?@#$%&*()]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _combine_sentiment_scores(self, vader_scores: Dict, textblob_sentiment) -> float:
        """Combine sentiment scores from different analyzers"""
        # VADER compound score (-1 to 1)
        vader_compound = vader_scores['compound']
        
        # TextBlob polarity (-1 to 1)
        textblob_polarity = textblob_sentiment.polarity
        
        # Weighted combination (VADER is generally more accurate for social media)
        combined_score = (0.7 * vader_compound) + (0.3 * textblob_polarity)
        
        return combined_score
    
    def _determine_sentiment_label(self, sentiment_score: float) -> SentimentLabel:
        """Determine sentiment label based on score"""
        if sentiment_score > 0.1:
            return SentimentLabel.POSITIVE
        elif sentiment_score < -0.1:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def _calculate_confidence_score(self, vader_scores: Dict, textblob_sentiment) -> float:
        """Calculate confidence score for sentiment analysis"""
        # Use VADER's subjectivity and compound score
        vader_confidence = abs(vader_scores['compound'])
        textblob_subjectivity = textblob_sentiment.subjectivity
        
        # Higher confidence for more extreme scores and more subjective text
        confidence = (vader_confidence + textblob_subjectivity) / 2
        
        return min(confidence, 1.0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        words = text.split()
        keywords = []
        
        for word in words:
            if word in self.crypto_keywords:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_engagement_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate engagement metrics from text"""
        words = text.split()
        
        return {
            'word_count': len(words),
            'hashtag_count': text.count('#'),
            'mention_count': text.count('@'),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'emoji_count': len(re.findall(r'[^\w\s]', text))
        }
    
    async def _store_sentiment_result(self, result: SentimentResult):
        """Store sentiment result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                INSERT INTO sentiment_analysis 
                (timestamp, symbol, source, sentiment_score, sentiment_label, confidence_score,
                 text_content, source_url, author, engagement_metrics, keywords, language)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """
                
                await conn.execute(query,
                    result.timestamp,
                    result.symbol,
                    result.source.value,
                    result.sentiment_score,
                    result.sentiment_label.value,
                    result.confidence_score,
                    result.text_content,
                    result.source_url,
                    result.author,
                    json.dumps(result.engagement_metrics),
                    json.dumps(result.keywords),
                    result.language
                )
                
                self.logger.info(f"💾 Sentiment result stored for {result.symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error storing sentiment result: {e}")
    
    async def get_recent_sentiment(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent sentiment analysis results"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT timestamp, symbol, source, sentiment_score, sentiment_label, 
                       confidence_score, text_content, source_url, author, 
                       engagement_metrics, keywords, language
                FROM sentiment_analysis
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """
                
                rows = await conn.fetch(query, symbol, limit)
                
                results = []
                for row in rows:
                    results.append({
                        'timestamp': row['timestamp'].isoformat(),
                        'symbol': row['symbol'],
                        'source': row['source'],
                        'sentiment_score': float(row['sentiment_score']),
                        'sentiment_label': row['sentiment_label'],
                        'confidence_score': float(row['confidence_score']),
                        'text_content': row['text_content'],
                        'source_url': row['source_url'],
                        'author': row['author'],
                        'engagement_metrics': json.loads(row['engagement_metrics']) if row['engagement_metrics'] else {},
                        'keywords': json.loads(row['keywords']) if row['keywords'] else [],
                        'language': row['language']
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"❌ Error getting recent sentiment: {e}")
            return []
