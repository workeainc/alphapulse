#!/usr/bin/env python3
"""
Noise Filter Engine for Advanced Pattern Recognition
Filters out low-quality patterns based on volume, volatility, time, and spread conditions
"""

import logging
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

def safe_json_dumps(data):
    """Safely convert dictionary to JSON, handling NaN values"""
    try:
        # Convert all NaN values to None before JSON serialization
        def convert_nan(obj):
            if isinstance(obj, dict):
                return {k: convert_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan(v) for v in obj]
            elif pd.isna(obj) if hasattr(pd, 'isna') else obj != obj:  # Check for NaN
                return None
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj) if not pd.isna(obj) else None
            else:
                return obj
        
        cleaned_data = convert_nan(data)
        return json.dumps(cleaned_data)
    except Exception as e:
        logger.warning(f"Failed to serialize data to JSON: {e}")
        return json.dumps({})

class NoiseFilterEngine:
    """Engine for filtering out noise and low-quality patterns"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.filter_settings = {}
        self.initialized = False
        
        logger.info("🔧 Noise Filter Engine initialized")
    
    async def initialize(self):
        """Initialize the noise filter engine"""
        try:
            logger.info("🔧 Initializing Noise Filter Engine...")
            
            # Connect to database
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            
            # Load filter settings
            await self._load_filter_settings()
            
            self.initialized = True
            logger.info("✅ Noise Filter Engine ready")
            
        except Exception as e:
            logger.error(f"❌ Noise Filter Engine initialization failed: {e}")
            raise
    
    async def _load_filter_settings(self):
        """Load noise filter settings from database"""
        try:
            self.cursor.execute("""
                SELECT filter_type, filter_name, filter_parameters, priority, is_active
                FROM noise_filter_settings 
                WHERE is_active = TRUE 
                ORDER BY priority
            """)
            
            settings = self.cursor.fetchall()
            
            for setting in settings:
                filter_type, filter_name, filter_params, priority, is_active = setting
                
                if filter_type not in self.filter_settings:
                    self.filter_settings[filter_type] = {}
                
                self.filter_settings[filter_type][filter_name] = {
                    'parameters': filter_params,
                    'priority': priority,
                    'is_active': is_active
                }
            
            logger.info(f"📋 Loaded {len(settings)} noise filter settings")
            
        except Exception as e:
            logger.error(f"❌ Failed to load filter settings: {e}")
            raise
    
    async def filter_pattern(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Filter a pattern based on noise filtering rules
        
        Args:
            pattern_data: Pattern data to filter
            market_data: Market data DataFrame with OHLCV
            
        Returns:
            Tuple of (passed_filter, filter_results)
        """
        try:
            if not self.initialized:
                await self.initialize()
            
            filter_results = {
                'passed': True,
                'overall_score': 1.0,
                'filter_scores': {},
                'filter_reasons': [],
                'noise_level': 0.0
            }
            
            # Apply volume-based filtering
            volume_result = await self._apply_volume_filter(pattern_data, market_data)
            filter_results['filter_scores']['volume'] = float(volume_result['score'])
            if not volume_result['passed']:
                filter_results['passed'] = False
                filter_results['filter_reasons'].append(volume_result['reason'])
            
            # Apply volatility-based filtering
            volatility_result = await self._apply_volatility_filter(pattern_data, market_data)
            filter_results['filter_scores']['volatility'] = float(volatility_result['score'])
            if not volatility_result['passed']:
                filter_results['passed'] = False
                filter_results['filter_reasons'].append(volatility_result['reason'])
            
            # Apply time-based filtering
            time_result = await self._apply_time_filter(pattern_data, market_data)
            filter_results['filter_scores']['time'] = float(time_result['score'])
            if not time_result['passed']:
                filter_results['passed'] = False
                filter_results['filter_reasons'].append(time_result['reason'])
            
            # Apply spread-based filtering
            spread_result = await self._apply_spread_filter(pattern_data, market_data)
            filter_results['filter_scores']['spread'] = float(spread_result['score'])
            if not spread_result['passed']:
                filter_results['passed'] = False
                filter_results['filter_reasons'].append(spread_result['reason'])
            
            # Calculate overall score and noise level
            scores = list(filter_results['filter_scores'].values())
            filter_results['overall_score'] = float(sum(scores) / len(scores) if scores else 1.0)
            filter_results['noise_level'] = float(1.0 - filter_results['overall_score'])
            
            # Store quality metrics
            await self._store_quality_metrics(pattern_data, filter_results)
            
            return filter_results['passed'], filter_results
            
        except Exception as e:
            logger.error(f"❌ Pattern filtering failed: {e}")
            return True, {'passed': True, 'overall_score': 1.0, 'error': str(e)}
    
    async def _apply_volume_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply volume-based filtering"""
        try:
            if 'volume' not in self.filter_settings:
                return {'passed': True, 'score': 1.0, 'reason': 'No volume filter configured'}
            
            volume_filters = self.filter_settings['volume']
            
            for filter_name, filter_config in volume_filters.items():
                if not filter_config['is_active']:
                    continue
                
                params = filter_config['parameters']
                
                if filter_name == 'low_volume_filter':
                    return await self._apply_low_volume_filter(pattern_data, market_data, params)
                elif filter_name == 'volume_spike_filter':
                    return await self._apply_volume_spike_filter(pattern_data, market_data, params)
            
            return {'passed': True, 'score': 1.0, 'reason': 'Volume filter passed'}
            
        except Exception as e:
            logger.error(f"❌ Volume filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'Volume filter error: {e}'}
    
    async def _apply_low_volume_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Apply low volume filter"""
        try:
            min_volume_ratio = params.get('min_volume_ratio', 0.5)
            volume_period = params.get('volume_period', 20)
            
            if len(market_data) < volume_period:
                return {'passed': True, 'score': 1.0, 'reason': 'Insufficient data for volume filter'}
            
            # Calculate current volume ratio
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].tail(volume_period).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate score (1.0 = good, 0.0 = bad)
            score = float(min(volume_ratio / min_volume_ratio, 1.0))
            
            if volume_ratio < min_volume_ratio:
                return {
                    'passed': False,
                    'score': score,
                    'reason': f'Low volume: {volume_ratio:.3f} < {min_volume_ratio}'
                }
            
            return {
                'passed': True,
                'score': score,
                'reason': f'Volume OK: {volume_ratio:.3f}'
            }
            
        except Exception as e:
            logger.error(f"❌ Low volume filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'Low volume filter error: {e}'}
    
    async def _apply_volatility_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply volatility-based filtering"""
        try:
            if 'volatility' not in self.filter_settings:
                return {'passed': True, 'score': 1.0, 'reason': 'No volatility filter configured'}
            
            volatility_filters = self.filter_settings['volatility']
            
            for filter_name, filter_config in volatility_filters.items():
                if not filter_config['is_active']:
                    continue
                
                params = filter_config['parameters']
                
                if filter_name == 'low_volatility_filter':
                    return await self._apply_low_volatility_filter(pattern_data, market_data, params)
            
            return {'passed': True, 'score': 1.0, 'reason': 'Volatility filter passed'}
            
        except Exception as e:
            logger.error(f"❌ Volatility filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'Volatility filter error: {e}'}
    
    async def _apply_low_volatility_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Apply low volatility filter"""
        try:
            min_atr_ratio = params.get('min_atr_ratio', 0.005)
            atr_period = params.get('atr_period', 14)
            
            if len(market_data) < atr_period:
                return {'passed': True, 'score': 1.0, 'reason': 'Insufficient data for volatility filter'}
            
            # Calculate ATR
            high_low = market_data['high'] - market_data['low']
            high_close = np.abs(market_data['high'] - market_data['close'].shift())
            low_close = np.abs(market_data['low'] - market_data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=atr_period).mean().iloc[-1]
            
            # Convert to float to avoid numpy type issues
            atr = float(atr)
            
            current_price = market_data['close'].iloc[-1]
            atr_ratio = atr / current_price if current_price > 0 else 0.0
            
            # Calculate score
            score = float(min(atr_ratio / min_atr_ratio, 1.0))
            
            if atr_ratio < min_atr_ratio:
                return {
                    'passed': False,
                    'score': score,
                    'reason': f'Low volatility: {atr_ratio:.4f} < {min_atr_ratio}'
                }
            
            return {
                'passed': True,
                'score': score,
                'reason': f'Volatility OK: {atr_ratio:.4f}'
            }
            
        except Exception as e:
            logger.error(f"❌ Low volatility filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'Low volatility filter error: {e}'}
    
    async def _apply_time_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply time-based filtering"""
        try:
            if 'time' not in self.filter_settings:
                return {'passed': True, 'score': 1.0, 'reason': 'No time filter configured'}
            
            time_filters = self.filter_settings['time']
            
            for filter_name, filter_config in time_filters.items():
                if not filter_config['is_active']:
                    continue
                
                params = filter_config['parameters']
                
                if filter_name == 'low_liquidity_hours_filter':
                    return await self._apply_low_liquidity_hours_filter(pattern_data, market_data, params)
            
            return {'passed': True, 'score': 1.0, 'reason': 'Time filter passed'}
            
        except Exception as e:
            logger.error(f"❌ Time filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'Time filter error: {e}'}
    
    async def _apply_low_liquidity_hours_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Apply low liquidity hours filter"""
        try:
            start_time_str = params.get('low_liquidity_start', '02:00')
            end_time_str = params.get('low_liquidity_end', '06:00')
            
            # Parse times
            start_time = datetime.strptime(start_time_str, '%H:%M').time()
            end_time = datetime.strptime(end_time_str, '%H:%M').time()
            
            # Get current time (assuming UTC)
            current_time = datetime.now().time()
            
            # Check if current time is in low liquidity period
            if start_time <= end_time:
                in_low_liquidity = start_time <= current_time <= end_time
            else:  # Crosses midnight
                in_low_liquidity = current_time >= start_time or current_time <= end_time
            
            if in_low_liquidity:
                return {
                    'passed': False,
                    'score': 0.3,  # Reduce confidence during low liquidity
                    'reason': f'Low liquidity hours: {start_time_str}-{end_time_str}'
                }
            
            return {
                'passed': True,
                'score': 1.0,
                'reason': 'Normal liquidity hours'
            }
            
        except Exception as e:
            logger.error(f"❌ Low liquidity hours filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'Low liquidity hours filter error: {e}'}
    
    async def _apply_spread_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply spread-based filtering"""
        try:
            if 'spread' not in self.filter_settings:
                return {'passed': True, 'score': 1.0, 'reason': 'No spread filter configured'}
            
            spread_filters = self.filter_settings['spread']
            
            for filter_name, filter_config in spread_filters.items():
                if not filter_config['is_active']:
                    continue
                
                params = filter_config['parameters']
                
                if filter_name == 'high_spread_filter':
                    return await self._apply_high_spread_filter(pattern_data, market_data, params)
            
            return {'passed': True, 'score': 1.0, 'reason': 'Spread filter passed'}
            
        except Exception as e:
            logger.error(f"❌ Spread filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'Spread filter error: {e}'}
    
    async def _apply_high_spread_filter(self, pattern_data: Dict[str, Any], market_data: pd.DataFrame, params: Dict) -> Dict[str, Any]:
        """Apply high spread filter"""
        try:
            max_spread_ratio = params.get('max_spread_ratio', 0.001)
            
            # For now, use a simplified spread calculation
            # In a real implementation, you would get actual bid/ask data
            current_price = market_data['close'].iloc[-1]
            
            # Simulate spread (0.1% of price)
            simulated_spread = current_price * 0.001
            spread_ratio = simulated_spread / current_price
            
            # Calculate score
            score = float(max(0, 1.0 - (spread_ratio / max_spread_ratio)))
            
            if spread_ratio > max_spread_ratio:
                return {
                    'passed': False,
                    'score': score,
                    'reason': f'High spread: {spread_ratio:.4f} > {max_spread_ratio}'
                }
            
            return {
                'passed': True,
                'score': score,
                'reason': f'Spread OK: {spread_ratio:.4f}'
            }
            
        except Exception as e:
            logger.error(f"❌ High spread filter failed: {e}")
            return {'passed': True, 'score': 1.0, 'reason': f'High spread filter error: {e}'}
    
    async def _store_quality_metrics(self, pattern_data: Dict[str, Any], filter_results: Dict[str, Any]):
        """Store pattern quality metrics in database"""
        try:
            metric_id = f"quality_{pattern_data.get('pattern_id', 'unknown')}_{int(datetime.now().timestamp())}"
            
            self.cursor.execute("""
                INSERT INTO pattern_quality_metrics (
                    timestamp, metric_id, pattern_id, symbol, pattern_name, timeframe,
                    quality_score, volume_quality, volatility_quality, spread_quality,
                    time_quality, candlestick_quality, market_context_quality, noise_level,
                    filter_reasons, quality_factors
                ) VALUES (
                    NOW(), %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb
                )
            """, (
                metric_id,
                pattern_data.get('pattern_id', 'unknown'),
                pattern_data.get('symbol', 'unknown'),
                pattern_data.get('pattern_name', 'unknown'),
                pattern_data.get('timeframe', 'unknown'),
                float(filter_results['overall_score']),
                float(filter_results['filter_scores'].get('volume', 1.0)),
                float(filter_results['filter_scores'].get('volatility', 1.0)),
                float(filter_results['filter_scores'].get('spread', 1.0)),
                float(filter_results['filter_scores'].get('time', 1.0)),
                1.0,  # candlestick_quality (placeholder)
                1.0,  # market_context_quality (placeholder)
                float(filter_results['noise_level']),
                safe_json_dumps(filter_results['filter_reasons']),
                safe_json_dumps(filter_results['filter_scores'])
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Failed to store quality metrics: {e}")
            self.conn.rollback()
    
    async def get_filter_settings(self) -> Dict[str, Any]:
        """Get current filter settings"""
        return self.filter_settings
    
    async def update_filter_setting(self, filter_type: str, filter_name: str, parameters: Dict[str, Any]):
        """Update a filter setting"""
        try:
            self.cursor.execute("""
                UPDATE noise_filter_settings 
                SET filter_parameters = %s::jsonb, updated_at = NOW()
                WHERE filter_type = %s AND filter_name = %s
            """, (json.dumps(parameters), filter_type, filter_name))
            
            self.conn.commit()
            
            # Reload settings
            await self._load_filter_settings()
            
            logger.info(f"✅ Updated filter setting: {filter_type}.{filter_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update filter setting: {e}")
            self.conn.rollback()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.info("✅ Noise Filter Engine cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Example usage
async def test_noise_filter_engine():
    """Test the noise filter engine"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',
        'user': 'postgres',
        'password': 'Emon_@17711'
    }
    
    # Create noise filter engine
    noise_filter = NoiseFilterEngine(db_config)
    
    try:
        # Initialize engine
        await noise_filter.initialize()
        
        # Sample pattern data
        pattern_data = {
            'pattern_id': 'test_pattern_123',
            'symbol': 'BTCUSDT',
            'pattern_name': 'doji',
            'timeframe': '1h',
            'confidence': 0.75
        }
        
        # Sample market data
        market_data = pd.DataFrame({
            'open': [50000, 50100, 50200, 50300, 50400],
            'high': [50100, 50200, 50300, 50400, 50500],
            'low': [49900, 50000, 50100, 50200, 50300],
            'close': [50100, 50200, 50300, 50400, 50500],
            'volume': [1000, 800, 1200, 900, 1100]  # Low volume for testing
        })
        
        # Test filtering
        passed, results = await noise_filter.filter_pattern(pattern_data, market_data)
        
        print(f"🎯 Pattern Filter Results:")
        print(f"   Passed: {passed}")
        print(f"   Overall Score: {results['overall_score']:.3f}")
        print(f"   Noise Level: {results['noise_level']:.3f}")
        print(f"   Filter Scores: {results['filter_scores']}")
        print(f"   Filter Reasons: {results['filter_reasons']}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    
    finally:
        await noise_filter.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_noise_filter_engine())
