"""
Real Market Data Integration Service for AlphaPlus
Integrates multiple data sources for comprehensive market analysis
"""

import asyncio
import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

class RealMarketDataService:
    """Real-time market data service integrating multiple sources"""
    
    def __init__(self):
        # API Keys (from environment variables)
        self.coinglass_api_key = os.getenv('COINGLASS_API_KEY', '')
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', '')
        self.coinmarketcap_api_key = os.getenv('COINMARKETCAP_API_KEY', '')
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY', '')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET', '')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY', '')
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Supported symbols
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
        
        # Initialize exchange (using Binance for demo)
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': True,  # Use sandbox for safety
            'enableRateLimit': True,
        })
        
        # Data buffers
        self.market_data_buffer = {}
        self.sentiment_buffer = {}
        self.news_buffer = []
        
    async def initialize(self):
        """Initialize the market data service"""
        try:
            await self.exchange.load_markets()
            logger.info("✅ Real Market Data Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize market data service: {e}")
            raise
    
    async def get_real_time_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> Dict[str, Any]:
        """Get real-time OHLCV data from exchange"""
        try:
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate additional metrics
            df['price_change'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['price_ma'] = df['close'].rolling(window=20).mean()
            
            # Get latest data point
            latest = df.iloc[-1]
            
            return {
                'symbol': symbol,
                'timestamp': latest['timestamp'],
                'open': float(latest['open']),
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': float(latest['close']),
                'volume': float(latest['volume']),
                'price_change': float(latest['price_change']),
                'volume_ma': float(latest['volume_ma']),
                'price_ma': float(latest['price_ma']),
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching OHLCV for {symbol}: {e}")
            return None
    
    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get market sentiment from multiple sources"""
        try:
            # Extract base symbol (e.g., 'BTC' from 'BTC/USDT')
            base_symbol = symbol.split('/')[0]
            
            # Get news sentiment
            news_sentiment = await self.get_news_sentiment(base_symbol)
            
            # Get social sentiment (simulated for now)
            social_sentiment = await self.get_social_sentiment(base_symbol)
            
            # Get fear & greed index (simulated)
            fear_greed = await self.get_fear_greed_index()
            
            # Calculate composite sentiment
            composite_sentiment = (
                news_sentiment.get('compound', 0) * 0.4 +
                social_sentiment.get('compound', 0) * 0.3 +
                fear_greed.get('value', 50) / 100 * 0.3
            )
            
            return {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'fear_greed_index': fear_greed,
                'composite_sentiment': composite_sentiment,
                'sentiment_score': self.classify_sentiment(composite_sentiment)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting sentiment for {symbol}: {e}")
            return None
    
    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment for a symbol"""
        try:
            # Simulate news API call (replace with actual API)
            keywords = f"{symbol} cryptocurrency"
            
            # Simulated news data
            news_data = [
                {"title": f"{symbol} shows bullish momentum", "content": f"{symbol} is gaining traction in the market"},
                {"title": f"{symbol} adoption increases", "content": f"More institutions are adopting {symbol}"},
                {"title": f"{symbol} technical analysis", "content": f"Technical indicators show positive signals for {symbol}"}
            ]
            
            # Analyze sentiment
            sentiments = []
            for news in news_data:
                text = f"{news['title']} {news['content']}"
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                sentiments.append(sentiment)
            
            # Calculate average sentiment
            avg_sentiment = {
                'neg': np.mean([s['neg'] for s in sentiments]),
                'neu': np.mean([s['neu'] for s in sentiments]),
                'pos': np.mean([s['pos'] for s in sentiments]),
                'compound': np.mean([s['compound'] for s in sentiments])
            }
            
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"❌ Error getting news sentiment: {e}")
            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
    
    async def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get social media sentiment (simulated)"""
        try:
            # Simulate social sentiment analysis
            # In production, this would use Twitter API, Reddit API, etc.
            
            # Simulated sentiment scores
            sentiment_score = np.random.normal(0.1, 0.3)  # Slightly positive bias
            sentiment_score = np.clip(sentiment_score, -1, 1)
            
            return {
                'neg': max(0, -sentiment_score),
                'neu': 0.5,
                'pos': max(0, sentiment_score),
                'compound': sentiment_score
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting social sentiment: {e}")
            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
    
    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get fear & greed index (simulated)"""
        try:
            # Simulate fear & greed index
            # In production, this would fetch from alternative.me API
            
            fear_greed_value = np.random.randint(20, 80)  # Random value between 20-80
            
            if fear_greed_value < 25:
                classification = "Extreme Fear"
            elif fear_greed_value < 45:
                classification = "Fear"
            elif fear_greed_value < 55:
                classification = "Neutral"
            elif fear_greed_value < 75:
                classification = "Greed"
            else:
                classification = "Extreme Greed"
            
            return {
                'value': fear_greed_value,
                'classification': classification,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting fear & greed index: {e}")
            return {'value': 50, 'classification': 'Neutral', 'timestamp': datetime.utcnow()}
    
    def classify_sentiment(self, sentiment_score: float) -> str:
        """Classify sentiment score into category"""
        if sentiment_score >= 0.05:
            return "bullish"
        elif sentiment_score <= -0.05:
            return "bearish"
        else:
            return "neutral"
    
    async def get_market_regime(self, symbol: str, data_points: List[Dict]) -> Dict[str, Any]:
        """Analyze market regime (trending, ranging, volatile)"""
        try:
            if len(data_points) < 20:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data_points)
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['volatility'] = df['close'].rolling(window=20).std()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Determine market regime
            price_trend = (latest['close'] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            volume_trend = (latest['volume'] - latest['volume_ma']) / latest['volume_ma']
            volatility_level = latest['volatility'] / latest['close']
            
            # Classify regime
            if abs(price_trend) > 0.05 and abs(volume_trend) > 0.2:
                regime_type = "trending"
                confidence = min(0.95, 0.7 + abs(price_trend) * 2)
            elif volatility_level > 0.03:
                regime_type = "volatile"
                confidence = min(0.95, 0.6 + volatility_level * 10)
            else:
                regime_type = "ranging"
                confidence = min(0.95, 0.8 - abs(price_trend) * 5)
            
            return {
                'symbol': symbol,
                'regime_type': regime_type,
                'confidence': confidence,
                'volatility_level': volatility_level,
                'trend_strength': abs(price_trend),
                'volume_trend': volume_trend,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market regime for {symbol}: {e}")
            return None
    
    async def collect_all_market_data(self) -> Dict[str, Any]:
        """Collect comprehensive market data for all symbols"""
        try:
            all_data = {}
            
            for symbol in self.symbols:
                # Get OHLCV data
                ohlcv_data = await self.get_real_time_ohlcv(symbol)
                if ohlcv_data:
                    all_data[symbol] = ohlcv_data
                    
                    # Store in buffer
                    if symbol not in self.market_data_buffer:
                        self.market_data_buffer[symbol] = []
                    self.market_data_buffer[symbol].append(ohlcv_data)
                    
                    # Keep buffer size manageable
                    if len(self.market_data_buffer[symbol]) > 200:
                        self.market_data_buffer[symbol] = self.market_data_buffer[symbol][-200:]
                
                # Get sentiment data
                sentiment_data = await self.get_market_sentiment(symbol)
                if sentiment_data:
                    if symbol not in self.sentiment_buffer:
                        self.sentiment_buffer[symbol] = []
                    self.sentiment_buffer[symbol].append(sentiment_data)
                    
                    # Keep sentiment buffer manageable
                    if len(self.sentiment_buffer[symbol]) > 50:
                        self.sentiment_buffer[symbol] = self.sentiment_buffer[symbol][-50:]
            
            logger.info(f"📊 Collected market data for {len(all_data)} symbols")
            return all_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting market data: {e}")
            return {}
    
    async def close(self):
        """Close the market data service"""
        try:
            await self.exchange.close()
            logger.info("✅ Market data service closed")
        except Exception as e:
            logger.error(f"❌ Error closing market data service: {e}")

# Global instance
market_data_service = RealMarketDataService()
