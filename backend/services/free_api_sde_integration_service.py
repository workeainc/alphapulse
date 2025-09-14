"""
Free API SDE Integration Service
Integrates free API data with SDE framework for enhanced signal generation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from services.free_api_database_service import FreeAPIDatabaseService
from services.free_api_manager import FreeAPIManager

logger = logging.getLogger(__name__)

@dataclass
class FreeAPISDEInput:
    """Input data for SDE framework from free APIs"""
    symbol: str
    timestamp: datetime
    market_data: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    news_data: Dict[str, Any]
    social_data: Dict[str, Any]
    liquidation_events: List[Dict]
    data_quality_score: float
    confidence_score: float

@dataclass
class FreeAPISDEResult:
    """SDE framework result with free API integration"""
    symbol: str
    timestamp: datetime
    sde_confidence: float
    market_regime: str
    sentiment_regime: str
    risk_level: str
    signal_strength: float
    confluence_score: float
    free_api_contributions: Dict[str, Any]
    final_recommendation: str
    risk_reward_ratio: float

class FreeAPISDEIntegrationService:
    """Service for integrating free API data with SDE framework"""
    
    def __init__(self, db_service: FreeAPIDatabaseService, free_api_manager: FreeAPIManager):
        self.db_service = db_service
        self.free_api_manager = free_api_manager
        self.logger = logging.getLogger(__name__)
        
        # SDE weights for different data sources
        self.sde_weights = {
            'market_data': 0.35,
            'sentiment_data': 0.25,
            'news_data': 0.20,
            'social_data': 0.15,
            'liquidation_events': 0.05
        }
        
        # Market regime thresholds
        self.market_regime_thresholds = {
            'bull': 0.3,
            'bear': -0.3,
            'sideways': 0.1
        }
        
        # Sentiment regime thresholds
        self.sentiment_regime_thresholds = {
            'bullish': 0.2,
            'bearish': -0.2,
            'neutral': 0.1
        }
    
    async def prepare_sde_input(self, symbol: str, hours: int = 24) -> Optional[FreeAPISDEInput]:
        """Prepare comprehensive input data for SDE framework"""
        try:
            # Get aggregated data from database
            market_data = await self.db_service.get_aggregated_market_data(symbol, hours)
            sentiment_data = await self.db_service.get_aggregated_sentiment(symbol, hours)
            liquidation_events = await self.db_service.get_recent_liquidation_events(symbol, hours)
            
            # Get recent news data
            news_data = await self._get_recent_news_data(symbol, hours)
            
            # Get recent social data
            social_data = await self._get_recent_social_data(symbol, hours)
            
            # Calculate data quality score
            data_quality_score = self._calculate_data_quality_score(
                market_data, sentiment_data, news_data, social_data, liquidation_events
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                market_data, sentiment_data, news_data, social_data, liquidation_events
            )
            
            return FreeAPISDEInput(
                symbol=symbol,
                timestamp=datetime.now(),
                market_data=market_data,
                sentiment_data=sentiment_data,
                news_data=news_data,
                social_data=social_data,
                liquidation_events=liquidation_events,
                data_quality_score=data_quality_score,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing SDE input for {symbol}: {e}")
            return None
    
    async def _get_recent_news_data(self, symbol: str, hours: int) -> Dict[str, Any]:
        """Get recent news data for SDE analysis"""
        try:
            async with self.db_service.db_pool.acquire() as conn:
                query = """
                    SELECT source, COUNT(*) as news_count, AVG(sentiment_score) as avg_sentiment,
                           AVG(relevance_score) as avg_relevance, MAX(timestamp) as last_updated
                    FROM free_api_news_data
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    GROUP BY source
                """ % hours
                
                rows = await conn.fetch(query, symbol)
                
                news_data = {
                    'symbol': symbol,
                    'timeframe_hours': hours,
                    'news_by_source': {},
                    'total_news_count': 0,
                    'avg_sentiment': 0.0,
                    'avg_relevance': 0.0,
                    'last_updated': None
                }
                
                total_sentiment = 0.0
                total_relevance = 0.0
                total_count = 0
                
                for row in rows:
                    source = row['source']
                    count = int(row['news_count'])
                    avg_sentiment = float(row['avg_sentiment']) if row['avg_sentiment'] else 0.0
                    avg_relevance = float(row['avg_relevance']) if row['avg_relevance'] else 0.0
                    
                    news_data['news_by_source'][source] = {
                        'count': count,
                        'avg_sentiment': avg_sentiment,
                        'avg_relevance': avg_relevance,
                        'last_updated': row['last_updated']
                    }
                    
                    total_sentiment += avg_sentiment * count
                    total_relevance += avg_relevance * count
                    total_count += count
                    
                    if not news_data['last_updated'] or row['last_updated'] > news_data['last_updated']:
                        news_data['last_updated'] = row['last_updated']
                
                if total_count > 0:
                    news_data['total_news_count'] = total_count
                    news_data['avg_sentiment'] = total_sentiment / total_count
                    news_data['avg_relevance'] = total_relevance / total_count
                
                return news_data
                
        except Exception as e:
            self.logger.error(f"❌ Error getting news data: {e}")
            return {'symbol': symbol, 'timeframe_hours': hours, 'news_by_source': {}, 'total_news_count': 0}
    
    async def _get_recent_social_data(self, symbol: str, hours: int) -> Dict[str, Any]:
        """Get recent social media data for SDE analysis"""
        try:
            async with self.db_service.db_pool.acquire() as conn:
                query = """
                    SELECT platform, COUNT(*) as post_count, AVG(sentiment_score) as avg_sentiment,
                           AVG(influence_score) as avg_influence, MAX(timestamp) as last_updated
                    FROM free_api_social_data
                    WHERE symbol = $1 AND timestamp >= NOW() - INTERVAL '%s hours'
                    GROUP BY platform
                """ % hours
                
                rows = await conn.fetch(query, symbol)
                
                social_data = {
                    'symbol': symbol,
                    'timeframe_hours': hours,
                    'social_by_platform': {},
                    'total_post_count': 0,
                    'avg_sentiment': 0.0,
                    'avg_influence': 0.0,
                    'last_updated': None
                }
                
                total_sentiment = 0.0
                total_influence = 0.0
                total_count = 0
                
                for row in rows:
                    platform = row['platform']
                    count = int(row['post_count'])
                    avg_sentiment = float(row['avg_sentiment']) if row['avg_sentiment'] else 0.0
                    avg_influence = float(row['avg_influence']) if row['avg_influence'] else 0.0
                    
                    social_data['social_by_platform'][platform] = {
                        'count': count,
                        'avg_sentiment': avg_sentiment,
                        'avg_influence': avg_influence,
                        'last_updated': row['last_updated']
                    }
                    
                    total_sentiment += avg_sentiment * count
                    total_influence += avg_influence * count
                    total_count += count
                    
                    if not social_data['last_updated'] or row['last_updated'] > social_data['last_updated']:
                        social_data['last_updated'] = row['last_updated']
                
                if total_count > 0:
                    social_data['total_post_count'] = total_count
                    social_data['avg_sentiment'] = total_sentiment / total_count
                    social_data['avg_influence'] = total_influence / total_count
                
                return social_data
                
        except Exception as e:
            self.logger.error(f"❌ Error getting social data: {e}")
            return {'symbol': symbol, 'timeframe_hours': hours, 'social_by_platform': {}, 'total_post_count': 0}
    
    def _calculate_data_quality_score(self, market_data: Dict, sentiment_data: Dict, 
                                     news_data: Dict, social_data: Dict, 
                                     liquidation_events: List) -> float:
        """Calculate overall data quality score"""
        try:
            scores = []
            
            # Market data quality
            if market_data.get('total_data_points', 0) > 0:
                market_score = min(1.0, market_data['total_data_points'] / 10.0)  # Normalize to 0-1
                scores.append(market_score * self.sde_weights['market_data'])
            
            # Sentiment data quality
            if sentiment_data.get('total_sentiment_count', 0) > 0:
                sentiment_score = min(1.0, sentiment_data['total_sentiment_count'] / 20.0)  # Normalize to 0-1
                scores.append(sentiment_score * self.sde_weights['sentiment_data'])
            
            # News data quality
            if news_data.get('total_news_count', 0) > 0:
                news_score = min(1.0, news_data['total_news_count'] / 15.0)  # Normalize to 0-1
                scores.append(news_score * self.sde_weights['news_data'])
            
            # Social data quality
            if social_data.get('total_post_count', 0) > 0:
                social_score = min(1.0, social_data['total_post_count'] / 25.0)  # Normalize to 0-1
                scores.append(social_score * self.sde_weights['social_data'])
            
            # Liquidation events quality
            if liquidation_events:
                liquidation_score = min(1.0, len(liquidation_events) / 5.0)  # Normalize to 0-1
                scores.append(liquidation_score * self.sde_weights['liquidation_events'])
            
            return sum(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating data quality score: {e}")
            return 0.0
    
    def _calculate_confidence_score(self, market_data: Dict, sentiment_data: Dict, 
                                  news_data: Dict, social_data: Dict, 
                                  liquidation_events: List) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_scores = []
            
            # Market data confidence
            if market_data.get('market_data_by_source'):
                market_confidence = len(market_data['market_data_by_source']) / 3.0  # Normalize by expected sources
                confidence_scores.append(market_confidence * self.sde_weights['market_data'])
            
            # Sentiment data confidence
            if sentiment_data.get('overall_confidence', 0) > 0:
                confidence_scores.append(sentiment_data['overall_confidence'] * self.sde_weights['sentiment_data'])
            
            # News data confidence (based on relevance)
            if news_data.get('avg_relevance', 0) > 0:
                confidence_scores.append(news_data['avg_relevance'] * self.sde_weights['news_data'])
            
            # Social data confidence (based on influence)
            if social_data.get('avg_influence', 0) > 0:
                confidence_scores.append(social_data['avg_influence'] * self.sde_weights['social_data'])
            
            # Liquidation events confidence (presence indicates market activity)
            if liquidation_events:
                liquidation_confidence = min(1.0, len(liquidation_events) / 10.0)
                confidence_scores.append(liquidation_confidence * self.sde_weights['liquidation_events'])
            
            return sum(confidence_scores) if confidence_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence score: {e}")
            return 0.0
    
    async def analyze_with_sde_framework(self, sde_input: FreeAPISDEInput) -> FreeAPISDEResult:
        """Analyze data using SDE framework principles"""
        try:
            # Determine market regime
            market_regime = self._determine_market_regime(sde_input.market_data)
            
            # Determine sentiment regime
            sentiment_regime = self._determine_sentiment_regime(sde_input.sentiment_data)
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(sde_input)
            
            # Determine risk level
            risk_level = self._determine_risk_level(sde_input)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(sde_input, confluence_score)
            
            # Calculate SDE confidence
            sde_confidence = self._calculate_sde_confidence(sde_input, confluence_score, signal_strength)
            
            # Generate final recommendation
            final_recommendation = self._generate_final_recommendation(
                market_regime, sentiment_regime, risk_level, signal_strength, sde_confidence
            )
            
            # Calculate risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(sde_input, signal_strength)
            
            # Prepare free API contributions
            free_api_contributions = self._prepare_free_api_contributions(sde_input)
            
            return FreeAPISDEResult(
                symbol=sde_input.symbol,
                timestamp=sde_input.timestamp,
                sde_confidence=sde_confidence,
                market_regime=market_regime,
                sentiment_regime=sentiment_regime,
                risk_level=risk_level,
                signal_strength=signal_strength,
                confluence_score=confluence_score,
                free_api_contributions=free_api_contributions,
                final_recommendation=final_recommendation,
                risk_reward_ratio=risk_reward_ratio
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing with SDE framework: {e}")
            return self._get_default_sde_result(sde_input.symbol)
    
    def _determine_market_regime(self, market_data: Dict) -> str:
        """Determine market regime based on price changes and volume"""
        try:
            price_change = market_data.get('consensus_price_change', 0.0)
            volume = market_data.get('consensus_volume', 0.0)
            fear_greed = market_data.get('consensus_fear_greed', 50.0)
            
            # Combine price change and fear/greed index
            regime_score = (price_change * 0.7) + ((fear_greed - 50) / 50.0 * 0.3)
            
            if regime_score > self.market_regime_thresholds['bull']:
                return 'bull'
            elif regime_score < self.market_regime_thresholds['bear']:
                return 'bear'
            else:
                return 'sideways'
                
        except Exception as e:
            self.logger.error(f"❌ Error determining market regime: {e}")
            return 'sideways'
    
    def _determine_sentiment_regime(self, sentiment_data: Dict) -> str:
        """Determine sentiment regime based on sentiment scores"""
        try:
            overall_sentiment = sentiment_data.get('overall_sentiment', 0.0)
            confidence = sentiment_data.get('overall_confidence', 0.0)
            
            # Weight sentiment by confidence
            weighted_sentiment = overall_sentiment * confidence
            
            if weighted_sentiment > self.sentiment_regime_thresholds['bullish']:
                return 'bullish'
            elif weighted_sentiment < self.sentiment_regime_thresholds['bearish']:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            self.logger.error(f"❌ Error determining sentiment regime: {e}")
            return 'neutral'
    
    def _calculate_confluence_score(self, sde_input: FreeAPISDEInput) -> float:
        """Calculate confluence score from multiple data sources"""
        try:
            confluence_factors = []
            
            # Market data confluence
            market_sources = len(sde_input.market_data.get('market_data_by_source', {}))
            if market_sources > 0:
                market_confluence = min(1.0, market_sources / 3.0)  # Normalize by expected sources
                confluence_factors.append(market_confluence * self.sde_weights['market_data'])
            
            # Sentiment data confluence
            sentiment_sources = len(sde_input.sentiment_data.get('sentiment_by_type', {}))
            if sentiment_sources > 0:
                sentiment_confluence = min(1.0, sentiment_sources / 3.0)  # Normalize by expected types
                confluence_factors.append(sentiment_confluence * self.sde_weights['sentiment_data'])
            
            # News data confluence
            news_sources = len(sde_input.news_data.get('news_by_source', {}))
            if news_sources > 0:
                news_confluence = min(1.0, news_sources / 2.0)  # Normalize by expected sources
                confluence_factors.append(news_confluence * self.sde_weights['news_data'])
            
            # Social data confluence
            social_platforms = len(sde_input.social_data.get('social_by_platform', {}))
            if social_platforms > 0:
                social_confluence = min(1.0, social_platforms / 3.0)  # Normalize by expected platforms
                confluence_factors.append(social_confluence * self.sde_weights['social_data'])
            
            # Liquidation events confluence
            if sde_input.liquidation_events:
                liquidation_confluence = min(1.0, len(sde_input.liquidation_events) / 5.0)
                confluence_factors.append(liquidation_confluence * self.sde_weights['liquidation_events'])
            
            return sum(confluence_factors) if confluence_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confluence score: {e}")
            return 0.0
    
    def _determine_risk_level(self, sde_input: FreeAPISDEInput) -> str:
        """Determine risk level based on data quality and market conditions"""
        try:
            risk_factors = []
            
            # Data quality risk
            if sde_input.data_quality_score < 0.5:
                risk_factors.append(0.3)  # High risk due to poor data quality
            
            # Market volatility risk
            price_change = abs(sde_input.market_data.get('consensus_price_change', 0.0))
            if price_change > 0.1:  # >10% price change
                risk_factors.append(0.4)  # High risk due to volatility
            
            # Liquidation events risk
            if sde_input.liquidation_events:
                total_liquidation_value = sum(event.get('value_usd', 0) for event in sde_input.liquidation_events)
                if total_liquidation_value > 1000000:  # >$1M liquidations
                    risk_factors.append(0.3)  # High risk due to liquidations
            
            # Sentiment divergence risk
            market_sentiment = sde_input.market_data.get('consensus_fear_greed', 50.0)
            social_sentiment = sde_input.sentiment_data.get('overall_sentiment', 0.0)
            if abs(market_sentiment - 50) > 20 and abs(social_sentiment) > 0.3:
                risk_factors.append(0.2)  # Medium risk due to sentiment divergence
            
            total_risk = sum(risk_factors)
            
            if total_risk > 0.6:
                return 'high'
            elif total_risk > 0.3:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"❌ Error determining risk level: {e}")
            return 'medium'
    
    def _calculate_signal_strength(self, sde_input: FreeAPISDEInput, confluence_score: float) -> float:
        """Calculate signal strength based on confluence and data quality"""
        try:
            # Base signal strength from confluence
            base_strength = confluence_score
            
            # Adjust for data quality
            quality_adjustment = sde_input.data_quality_score * 0.3
            
            # Adjust for confidence
            confidence_adjustment = sde_input.confidence_score * 0.2
            
            # Calculate final signal strength
            signal_strength = base_strength + quality_adjustment + confidence_adjustment
            
            return min(1.0, max(0.0, signal_strength))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating signal strength: {e}")
            return 0.0
    
    def _calculate_sde_confidence(self, sde_input: FreeAPISDEInput, confluence_score: float, signal_strength: float) -> float:
        """Calculate SDE confidence score"""
        try:
            # Base confidence from confluence
            confluence_confidence = confluence_score * 0.4
            
            # Data quality confidence
            quality_confidence = sde_input.data_quality_score * 0.3
            
            # Signal strength confidence
            strength_confidence = signal_strength * 0.3
            
            # Calculate final SDE confidence
            sde_confidence = confluence_confidence + quality_confidence + strength_confidence
            
            return min(1.0, max(0.0, sde_confidence))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating SDE confidence: {e}")
            return 0.0
    
    def _generate_final_recommendation(self, market_regime: str, sentiment_regime: str, 
                                     risk_level: str, signal_strength: float, sde_confidence: float) -> str:
        """Generate final trading recommendation"""
        try:
            # Determine base recommendation
            if market_regime == 'bull' and sentiment_regime == 'bullish':
                base_recommendation = 'strong_buy'
            elif market_regime == 'bear' and sentiment_regime == 'bearish':
                base_recommendation = 'strong_sell'
            elif market_regime == 'bull' and sentiment_regime == 'neutral':
                base_recommendation = 'buy'
            elif market_regime == 'bear' and sentiment_regime == 'neutral':
                base_recommendation = 'sell'
            elif market_regime == 'sideways' and sentiment_regime == 'bullish':
                base_recommendation = 'weak_buy'
            elif market_regime == 'sideways' and sentiment_regime == 'bearish':
                base_recommendation = 'weak_sell'
            else:
                base_recommendation = 'hold'
            
            # Adjust for risk level
            if risk_level == 'high':
                if 'strong' in base_recommendation:
                    base_recommendation = base_recommendation.replace('strong_', '')
                elif base_recommendation in ['buy', 'sell']:
                    base_recommendation = f'weak_{base_recommendation}'
            
            # Adjust for signal strength and confidence
            if signal_strength < 0.3 or sde_confidence < 0.3:
                if base_recommendation in ['buy', 'sell']:
                    base_recommendation = f'weak_{base_recommendation}'
                elif base_recommendation in ['strong_buy', 'strong_sell']:
                    base_recommendation = base_recommendation.replace('strong_', '')
            
            return base_recommendation
            
        except Exception as e:
            self.logger.error(f"❌ Error generating final recommendation: {e}")
            return 'hold'
    
    def _calculate_risk_reward_ratio(self, sde_input: FreeAPISDEInput, signal_strength: float) -> float:
        """Calculate risk-reward ratio"""
        try:
            # Base risk-reward from signal strength
            base_ratio = 1.0 + (signal_strength * 2.0)  # 1.0 to 3.0
            
            # Adjust for market volatility
            price_change = abs(sde_input.market_data.get('consensus_price_change', 0.0))
            if price_change > 0.05:  # >5% volatility
                base_ratio *= 0.8  # Reduce ratio for high volatility
            
            # Adjust for data quality
            if sde_input.data_quality_score < 0.7:
                base_ratio *= 0.9  # Reduce ratio for poor data quality
            
            return max(0.5, min(3.0, base_ratio))
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating risk-reward ratio: {e}")
            return 1.0
    
    def _prepare_free_api_contributions(self, sde_input: FreeAPISDEInput) -> Dict[str, Any]:
        """Prepare free API contributions summary"""
        try:
            return {
                'market_data_sources': len(sde_input.market_data.get('market_data_by_source', {})),
                'sentiment_sources': len(sde_input.sentiment_data.get('sentiment_by_type', {})),
                'news_sources': len(sde_input.news_data.get('news_by_source', {})),
                'social_platforms': len(sde_input.social_data.get('social_by_platform', {})),
                'liquidation_events_count': len(sde_input.liquidation_events),
                'data_quality_score': sde_input.data_quality_score,
                'confidence_score': sde_input.confidence_score,
                'total_data_points': (
                    sde_input.market_data.get('total_data_points', 0) +
                    sde_input.sentiment_data.get('total_sentiment_count', 0) +
                    sde_input.news_data.get('total_news_count', 0) +
                    sde_input.social_data.get('total_post_count', 0) +
                    len(sde_input.liquidation_events)
                )
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error preparing free API contributions: {e}")
            return {}
    
    def _get_default_sde_result(self, symbol: str) -> FreeAPISDEResult:
        """Get default SDE result when analysis fails"""
        return FreeAPISDEResult(
            symbol=symbol,
            timestamp=datetime.now(),
            sde_confidence=0.0,
            market_regime='sideways',
            sentiment_regime='neutral',
            risk_level='medium',
            signal_strength=0.0,
            confluence_score=0.0,
            free_api_contributions={},
            final_recommendation='hold',
            risk_reward_ratio=1.0
        )
