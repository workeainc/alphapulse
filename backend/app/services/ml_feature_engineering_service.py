#!/usr/bin/env python3
"""
ML Feature Engineering Service
Extends existing volume analysis with comprehensive ML features
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FeatureCategory(Enum):
    VOLUME = "volume"
    PRICE = "price"
    TECHNICAL = "technical"
    ORDERBOOK = "orderbook"
    TIME = "time"
    MULTI_TIMEFRAME = "multi_timeframe"

@dataclass
class MLFeatures:
    """Comprehensive ML features for model training"""
    # Basic features
    volume_ratio: float
    volume_positioning_score: float
    order_book_imbalance: float
    vwap: float
    cumulative_volume_delta: float
    relative_volume: float
    volume_flow_imbalance: float
    
    # Technical indicators
    ema_20: float
    ema_50: float
    ema_200: float
    atr_14: float
    obv: float
    rsi_14: float
    macd: float
    macd_signal: float
    macd_histogram: float
    
    # Order book features
    bid_depth_0_5: float
    bid_depth_1_0: float
    bid_depth_2_0: float
    ask_depth_0_5: float
    ask_depth_1_0: float
    ask_depth_2_0: float
    bid_ask_ratio: float
    spread_bps: float
    liquidity_score: float
    
    # Time features
    minute_of_day: int
    hour_of_day: int
    day_of_week: int
    is_session_open: bool
    session_volatility: float
    
    # Multi-timeframe features
    h1_return: float
    h4_return: float
    d1_return: float
    h1_volume_ratio: float
    h4_volume_ratio: float
    d1_volume_ratio: float
    
    # Market regime features
    market_regime: str  # 'bull', 'bear', 'sideways'
    volatility_regime: str  # 'low', 'medium', 'high'
    
    # Volume pattern features
    volume_pattern_type: Optional[str]
    volume_pattern_confidence: float
    volume_breakout: bool
    
    # Support/Resistance features
    distance_to_support: float
    distance_to_resistance: float
    nearest_volume_node: float
    volume_node_strength: float

class MLFeatureEngineeringService:
    """Service for generating comprehensive ML features"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Technical indicator parameters
        self.ema_periods = [20, 50, 200]
        self.atr_period = 14
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Market regime thresholds
        self.volatility_thresholds = {
            'low': 0.01,    # 1% daily volatility
            'medium': 0.03, # 3% daily volatility
            'high': 0.05    # 5% daily volatility
        }
        
        self.logger.info("🔧 ML Feature Engineering Service initialized")
    
    async def generate_comprehensive_features(self, symbol: str, timeframe: str, ohlcv_data: List[Dict]) -> MLFeatures:
        """Generate comprehensive ML features from OHLCV data"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            if df.empty:
                raise ValueError("Empty OHLCV data provided")
            
            # Get latest data point
            latest = df.iloc[-1]
            
            # Basic volume features (from existing volume analysis)
            volume_features = await self._extract_volume_features(symbol, timeframe, df)
            
            # Technical indicators
            technical_features = self._calculate_technical_indicators(df)
            
            # Order book features (simulated for now)
            orderbook_features = await self._extract_orderbook_features(symbol, latest['timestamp'])
            
            # Time features
            time_features = self._calculate_time_features(latest['timestamp'])
            
            # Multi-timeframe features
            multi_tf_features = await self._extract_multi_timeframe_features(symbol, timeframe, latest['timestamp'])
            
            # Market regime features
            market_regime_features = self._calculate_market_regime_features(df)
            
            # Volume pattern features
            pattern_features = await self._extract_pattern_features(symbol, timeframe, latest['timestamp'])
            
            # Support/Resistance features
            support_resistance_features = await self._extract_support_resistance_features(symbol, timeframe, latest['close'])
            
            # Combine all features
            ml_features = MLFeatures(
                # Volume features
                volume_ratio=volume_features['volume_ratio'],
                volume_positioning_score=volume_features['volume_positioning_score'],
                order_book_imbalance=volume_features['order_book_imbalance'],
                vwap=volume_features['vwap'],
                cumulative_volume_delta=volume_features['cumulative_volume_delta'],
                relative_volume=volume_features['relative_volume'],
                volume_flow_imbalance=volume_features['volume_flow_imbalance'],
                
                # Technical features
                ema_20=technical_features['ema_20'],
                ema_50=technical_features['ema_50'],
                ema_200=technical_features['ema_200'],
                atr_14=technical_features['atr_14'],
                obv=technical_features['obv'],
                rsi_14=technical_features['rsi_14'],
                macd=technical_features['macd'],
                macd_signal=technical_features['macd_signal'],
                macd_histogram=technical_features['macd_histogram'],
                
                # Order book features
                bid_depth_0_5=orderbook_features['bid_depth_0_5'],
                bid_depth_1_0=orderbook_features['bid_depth_1_0'],
                bid_depth_2_0=orderbook_features['bid_depth_2_0'],
                ask_depth_0_5=orderbook_features['ask_depth_0_5'],
                ask_depth_1_0=orderbook_features['ask_depth_1_0'],
                ask_depth_2_0=orderbook_features['ask_depth_2_0'],
                bid_ask_ratio=orderbook_features['bid_ask_ratio'],
                spread_bps=orderbook_features['spread_bps'],
                liquidity_score=orderbook_features['liquidity_score'],
                
                # Time features
                minute_of_day=time_features['minute_of_day'],
                hour_of_day=time_features['hour_of_day'],
                day_of_week=time_features['day_of_week'],
                is_session_open=time_features['is_session_open'],
                session_volatility=time_features['session_volatility'],
                
                # Multi-timeframe features
                h1_return=multi_tf_features['h1_return'],
                h4_return=multi_tf_features['h4_return'],
                d1_return=multi_tf_features['d1_return'],
                h1_volume_ratio=multi_tf_features['h1_volume_ratio'],
                h4_volume_ratio=multi_tf_features['h4_volume_ratio'],
                d1_volume_ratio=multi_tf_features['d1_volume_ratio'],
                
                # Market regime features
                market_regime=market_regime_features['market_regime'],
                volatility_regime=market_regime_features['volatility_regime'],
                
                # Pattern features
                volume_pattern_type=pattern_features['volume_pattern_type'],
                volume_pattern_confidence=pattern_features['volume_pattern_confidence'],
                volume_breakout=pattern_features['volume_breakout'],
                
                # Support/Resistance features
                distance_to_support=support_resistance_features['distance_to_support'],
                distance_to_resistance=support_resistance_features['distance_to_resistance'],
                nearest_volume_node=support_resistance_features['nearest_volume_node'],
                volume_node_strength=support_resistance_features['volume_node_strength']
            )
            
            self.logger.info(f"✅ Generated comprehensive ML features for {symbol}")
            return ml_features
            
        except Exception as e:
            self.logger.error(f"❌ Error generating ML features: {e}")
            raise
    
    async def _extract_volume_features(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict:
        """Extract volume features from existing volume analysis"""
        try:
            # Get latest volume analysis from database
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT volume_ratio, volume_positioning_score, order_book_imbalance,
                       vwap, cumulative_volume_delta, relative_volume, volume_flow_imbalance
                FROM volume_analysis
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp DESC
                LIMIT 1
                """
                row = await conn.fetchrow(query, symbol, timeframe)
                
                if row:
                    return {
                        'volume_ratio': float(row['volume_ratio']),
                        'volume_positioning_score': float(row['volume_positioning_score']),
                        'order_book_imbalance': float(row['order_book_imbalance']),
                        'vwap': float(row['vwap']) if row['vwap'] else 0.0,
                        'cumulative_volume_delta': float(row['cumulative_volume_delta']) if row['cumulative_volume_delta'] else 0.0,
                        'relative_volume': float(row['relative_volume']) if row['relative_volume'] else 0.0,
                        'volume_flow_imbalance': float(row['volume_flow_imbalance']) if row['volume_flow_imbalance'] else 0.0
                    }
                else:
                    # Fallback to calculated values
                    return self._calculate_basic_volume_features(df)
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting volume features: {e}")
            return self._calculate_basic_volume_features(df)
    
    def _calculate_basic_volume_features(self, df: pd.DataFrame) -> Dict:
        """Calculate basic volume features as fallback"""
        latest = df.iloc[-1]
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        return {
            'volume_ratio': float(latest['volume'] / avg_volume if avg_volume > 0 else 1.0),
            'volume_positioning_score': 0.5,  # Default neutral
            'order_book_imbalance': 0.0,  # Default neutral
            'vwap': float(latest['close']),  # Use close as fallback
            'cumulative_volume_delta': 0.0,
            'relative_volume': 1.0,
            'volume_flow_imbalance': 0.0
        }
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        try:
            # EMAs
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr_14 = true_range.rolling(14).mean().iloc[-1]
            
            # OBV (On-Balance Volume)
            obv = 0
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv += df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv -= df['volume'].iloc[i]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            return {
                'ema_20': float(ema_20),
                'ema_50': float(ema_50),
                'ema_200': float(ema_200),
                'atr_14': float(atr_14),
                'obv': float(obv),
                'rsi_14': float(rsi_14),
                'macd': float(macd.iloc[-1]),
                'macd_signal': float(macd_signal.iloc[-1]),
                'macd_histogram': float(macd_histogram.iloc[-1])
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating technical indicators: {e}")
            return {
                'ema_20': 0.0, 'ema_50': 0.0, 'ema_200': 0.0,
                'atr_14': 0.0, 'obv': 0.0, 'rsi_14': 50.0,
                'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0
            }
    
    async def _extract_orderbook_features(self, symbol: str, timestamp: datetime) -> Dict:
        """Extract order book features from database"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT bid_depth_0_5, bid_depth_1_0, bid_depth_2_0,
                       ask_depth_0_5, ask_depth_1_0, ask_depth_2_0,
                       bid_ask_ratio, spread_bps, liquidity_score
                FROM liquidity_snapshots
                WHERE symbol = $1 AND timestamp <= $2
                ORDER BY timestamp DESC
                LIMIT 1
                """
                row = await conn.fetchrow(query, symbol, timestamp)
                
                if row:
                    return {
                        'bid_depth_0_5': float(row['bid_depth_0_5']),
                        'bid_depth_1_0': float(row['bid_depth_1_0']),
                        'bid_depth_2_0': float(row['bid_depth_2_0']),
                        'ask_depth_0_5': float(row['ask_depth_0_5']),
                        'ask_depth_1_0': float(row['ask_depth_1_0']),
                        'ask_depth_2_0': float(row['ask_depth_2_0']),
                        'bid_ask_ratio': float(row['bid_ask_ratio']),
                        'spread_bps': float(row['spread_bps']),
                        'liquidity_score': float(row['liquidity_score'])
                    }
                else:
                    # Return default values
                    return {
                        'bid_depth_0_5': 1000.0, 'bid_depth_1_0': 2000.0, 'bid_depth_2_0': 5000.0,
                        'ask_depth_0_5': 1000.0, 'ask_depth_1_0': 2000.0, 'ask_depth_2_0': 5000.0,
                        'bid_ask_ratio': 1.0, 'spread_bps': 10.0, 'liquidity_score': 0.5
                    }
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting orderbook features: {e}")
            return {
                'bid_depth_0_5': 1000.0, 'bid_depth_1_0': 2000.0, 'bid_depth_2_0': 5000.0,
                'ask_depth_0_5': 1000.0, 'ask_depth_1_0': 2000.0, 'ask_depth_2_0': 5000.0,
                'bid_ask_ratio': 1.0, 'spread_bps': 10.0, 'liquidity_score': 0.5
            }
    
    def _calculate_time_features(self, timestamp: datetime) -> Dict:
        """Calculate time-based features"""
        return {
            'minute_of_day': timestamp.hour * 60 + timestamp.minute,
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_session_open': self._is_trading_session_open(timestamp),
            'session_volatility': self._calculate_session_volatility(timestamp)
        }
    
    def _is_trading_session_open(self, timestamp: datetime) -> bool:
        """Check if trading session is open (24/7 for crypto)"""
        return True  # Crypto trades 24/7
    
    def _calculate_session_volatility(self, timestamp: datetime) -> float:
        """Calculate session volatility (placeholder)"""
        return 0.02  # Default 2% volatility
    
    async def _extract_multi_timeframe_features(self, symbol: str, timeframe: str, timestamp: datetime) -> Dict:
        """Extract multi-timeframe features from continuous aggregates"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get 1h, 4h, 1d returns and volume ratios
                query = """
                SELECT 
                    (SELECT AVG(avg_volume_ratio) FROM volume_stats_1h 
                     WHERE symbol = $1 AND timestamp >= $2 - INTERVAL '1 hour' LIMIT 1) as h1_volume_ratio,
                    (SELECT AVG(avg_volume_ratio) FROM volume_stats_1h 
                     WHERE symbol = $1 AND timestamp >= $2 - INTERVAL '4 hours' LIMIT 1) as h4_volume_ratio,
                    (SELECT AVG(avg_volume_ratio) FROM volume_stats_1d 
                     WHERE symbol = $1 AND timestamp >= $2 - INTERVAL '1 day' LIMIT 1) as d1_volume_ratio
                """
                row = await conn.fetchrow(query, symbol, timestamp)
                
                # Calculate returns (simplified)
                h1_return = 0.001  # Placeholder
                h4_return = 0.002  # Placeholder
                d1_return = 0.005  # Placeholder
                
                return {
                    'h1_return': h1_return,
                    'h4_return': h4_return,
                    'd1_return': d1_return,
                    'h1_volume_ratio': float(row['h1_volume_ratio']) if row['h1_volume_ratio'] else 1.0,
                    'h4_volume_ratio': float(row['h4_volume_ratio']) if row['h4_volume_ratio'] else 1.0,
                    'd1_volume_ratio': float(row['d1_volume_ratio']) if row['d1_volume_ratio'] else 1.0
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting multi-timeframe features: {e}")
            return {
                'h1_return': 0.001, 'h4_return': 0.002, 'd1_return': 0.005,
                'h1_volume_ratio': 1.0, 'h4_volume_ratio': 1.0, 'd1_volume_ratio': 1.0
            }
    
    def _calculate_market_regime_features(self, df: pd.DataFrame) -> Dict:
        """Calculate market regime features"""
        try:
            # Calculate daily volatility
            daily_returns = df['close'].pct_change().dropna()
            daily_volatility = daily_returns.std()
            
            # Determine volatility regime
            if daily_volatility < self.volatility_thresholds['low']:
                volatility_regime = 'low'
            elif daily_volatility < self.volatility_thresholds['medium']:
                volatility_regime = 'medium'
            else:
                volatility_regime = 'high'
            
            # Determine market regime based on EMAs
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
            
            if ema_20 > ema_50 > ema_200:
                market_regime = 'bull'
            elif ema_20 < ema_50 < ema_200:
                market_regime = 'bear'
            else:
                market_regime = 'sideways'
            
            return {
                'market_regime': market_regime,
                'volatility_regime': volatility_regime
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating market regime features: {e}")
            return {
                'market_regime': 'sideways',
                'volatility_regime': 'medium'
            }
    
    async def _extract_pattern_features(self, symbol: str, timeframe: str, timestamp: datetime) -> Dict:
        """Extract volume pattern features"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                SELECT volume_pattern_type, volume_pattern_confidence, volume_breakout
                FROM volume_analysis
                WHERE symbol = $1 AND timeframe = $2 AND timestamp <= $3
                ORDER BY timestamp DESC
                LIMIT 1
                """
                row = await conn.fetchrow(query, symbol, timeframe, timestamp)
                
                if row:
                    return {
                        'volume_pattern_type': row['volume_pattern_type'],
                        'volume_pattern_confidence': float(row['volume_pattern_confidence']) if row['volume_pattern_confidence'] else 0.0,
                        'volume_breakout': bool(row['volume_breakout'])
                    }
                else:
                    return {
                        'volume_pattern_type': None,
                        'volume_pattern_confidence': 0.0,
                        'volume_breakout': False
                    }
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting pattern features: {e}")
            return {
                'volume_pattern_type': None,
                'volume_pattern_confidence': 0.0,
                'volume_breakout': False
            }
    
    async def _extract_support_resistance_features(self, symbol: str, timeframe: str, current_price: float) -> Dict:
        """Extract support/resistance features"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get nearest support and resistance levels
                query = """
                SELECT price_level, node_strength
                FROM volume_nodes
                WHERE symbol = $1 AND timeframe = $2 AND is_active = TRUE
                ORDER BY ABS(price_level - $3)
                LIMIT 2
                """
                rows = await conn.fetch(query, symbol, timeframe, current_price)
                
                if len(rows) >= 2:
                    nearest_node = float(rows[0]['price_level'])
                    nearest_strength = float(rows[0]['node_strength'])
                    
                    # Determine if it's support or resistance
                    if nearest_node < current_price:
                        distance_to_support = current_price - nearest_node
                        distance_to_resistance = nearest_node - current_price if len(rows) > 1 else 1000.0
                    else:
                        distance_to_support = nearest_node - current_price if len(rows) > 1 else 1000.0
                        distance_to_resistance = nearest_node - current_price
                else:
                    distance_to_support = 1000.0
                    distance_to_resistance = 1000.0
                    nearest_node = current_price
                    nearest_strength = 0.0
                
                return {
                    'distance_to_support': distance_to_support,
                    'distance_to_resistance': distance_to_resistance,
                    'nearest_volume_node': nearest_node,
                    'volume_node_strength': nearest_strength
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting support/resistance features: {e}")
            return {
                'distance_to_support': 1000.0,
                'distance_to_resistance': 1000.0,
                'nearest_volume_node': current_price,
                'volume_node_strength': 0.0
            }
    
    async def store_ml_features(self, symbol: str, timeframe: str, timestamp: datetime, features: MLFeatures) -> bool:
        """Store ML features in the database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Convert features to JSONB format
                technical_features = {
                    'ema_20': features.ema_20, 'ema_50': features.ema_50, 'ema_200': features.ema_200,
                    'atr_14': features.atr_14, 'obv': features.obv, 'rsi_14': features.rsi_14,
                    'macd': features.macd, 'macd_signal': features.macd_signal, 'macd_histogram': features.macd_histogram
                }
                
                orderbook_features = {
                    'bid_depth_0_5': features.bid_depth_0_5, 'bid_depth_1_0': features.bid_depth_1_0, 'bid_depth_2_0': features.bid_depth_2_0,
                    'ask_depth_0_5': features.ask_depth_0_5, 'ask_depth_1_0': features.ask_depth_1_0, 'ask_depth_2_0': features.ask_depth_2_0,
                    'bid_ask_ratio': features.bid_ask_ratio, 'spread_bps': features.spread_bps, 'liquidity_score': features.liquidity_score
                }
                
                time_features = {
                    'minute_of_day': features.minute_of_day, 'hour_of_day': features.hour_of_day,
                    'day_of_week': features.day_of_week, 'is_session_open': features.is_session_open,
                    'session_volatility': features.session_volatility
                }
                
                multi_timeframe_features = {
                    'h1_return': features.h1_return, 'h4_return': features.h4_return, 'd1_return': features.d1_return,
                    'h1_volume_ratio': features.h1_volume_ratio, 'h4_volume_ratio': features.h4_volume_ratio, 'd1_volume_ratio': features.d1_volume_ratio
                }
                
                # Insert into ML dataset table
                query = """
                INSERT INTO volume_analysis_ml_dataset 
                (symbol, timeframe, timestamp, features, targets, metadata, technical_features, 
                 order_book_features, time_features, multi_timeframe_features, market_regime, volatility_regime)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (timestamp, id) DO UPDATE SET
                features = EXCLUDED.features,
                technical_features = EXCLUDED.technical_features,
                order_book_features = EXCLUDED.order_book_features,
                time_features = EXCLUDED.time_features,
                multi_timeframe_features = EXCLUDED.multi_timeframe_features,
                market_regime = EXCLUDED.market_regime,
                volatility_regime = EXCLUDED.volatility_regime
                """
                
                # Create features dict
                features_dict = {
                    'volume_ratio': features.volume_ratio,
                    'volume_positioning_score': features.volume_positioning_score,
                    'order_book_imbalance': features.order_book_imbalance,
                    'vwap': features.vwap,
                    'cumulative_volume_delta': features.cumulative_volume_delta,
                    'relative_volume': features.relative_volume,
                    'volume_flow_imbalance': features.volume_flow_imbalance,
                    'volume_pattern_type': features.volume_pattern_type,
                    'volume_pattern_confidence': features.volume_pattern_confidence,
                    'volume_breakout': features.volume_breakout,
                    'distance_to_support': features.distance_to_support,
                    'distance_to_resistance': features.distance_to_resistance,
                    'nearest_volume_node': features.nearest_volume_node,
                    'volume_node_strength': features.volume_node_strength
                }
                
                await conn.execute(query, symbol, timeframe, timestamp, features_dict, {}, {},
                                 technical_features, orderbook_features, time_features, 
                                 multi_timeframe_features, features.market_regime, features.volatility_regime)
                
                self.logger.info(f"✅ Stored ML features for {symbol} {timeframe}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing ML features: {e}")
            return False
