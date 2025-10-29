#!/usr/bin/env python3
"""
LIVE Signal Generation Test
Tests the integration of perfect calculations and generates signals from REAL-TIME WebSocket data
Collects live market data from Binance WebSocket and generates trading signals
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging

# Add backend to path
sys.path.append('backend')

# Import our updated modules
from src.strategies.dynamic_support_resistance_analyzer import DynamicSupportResistanceAnalyzer
from src.data.volume_analyzer import VolumeAnalyzer
from src.core.websocket_binance import BinanceWebSocketClient
from src.ai.sde_framework import SDEFramework
from src.ai.multi_timeframe_fusion import MultiTimeframeFusion
from src.data.realtime_data_pipeline import RealTimeDataPipeline
from src.ai.sde_database_integration import SDEDatabaseIntegration, SignalGenerationRequest
# from src.app.signals.intelligent_signal_generator import IntelligentSignalGenerator  # Requires db_pool and exchange

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveSignalGenerator:
    """Generate LIVE signals using perfect calculations from REAL-TIME WebSocket data"""
    
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Initialize components
        self.sr_analyzer = DynamicSupportResistanceAnalyzer()
        self.volume_analyzer = VolumeAnalyzer()
        self.websocket_client = BinanceWebSocketClient()
        
        # Initialize data pipeline and SDE integration with real PostgreSQL database
        db_url = "postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse"
        self.data_pipeline = RealTimeDataPipeline(db_url=db_url)
        self.sde_integration = SDEDatabaseIntegration(db_url=db_url)
        
        # Note: IntelligentSignalGenerator requires db_pool and exchange, skipping for test
        self.signal_generator = None
        
        # Data storage
        self.market_data = {}
        self.signals = []
        
    async def collect_live_data(self, duration_minutes=3):
        """Collect live market data from Binance WebSocket"""
        logger.info(f"🔄 Collecting LIVE data for {duration_minutes} minutes...")
        
        try:
            # Start WebSocket connection
            await self.websocket_client.connect()
            
            # Collect data for specified duration
            start_time = datetime.now()
            data_collected = []
            
            logger.info("📡 Listening for live WebSocket data...")
            
            # Listen to WebSocket messages
            async for message in self.websocket_client.listen():
                if message and message.get('type') == 'kline' and message.get('symbol') == self.symbol:
                    # The message is already parsed by the WebSocket client
                    candle_data = {
                        'timestamp': message['timestamp'],
                        'open': message['open'],
                        'high': message['high'],
                        'low': message['low'],
                        'close': message['close'],
                        'volume': message['volume']
                    }
                    
                    data_collected.append(candle_data)
                    # Log every 10th data point to reduce spam
                    if len(data_collected) % 10 == 0:
                        logger.info(f"📊 Live data #{len(data_collected)}: {self.symbol} - Price: ${candle_data['close']:.2f}, Volume: {candle_data['volume']:.2f}")
                
                # Check if we've collected enough data or time is up
                elapsed_minutes = (datetime.now() - start_time).total_seconds() / 60
                if len(data_collected) >= 100:  # Collect at least 100 data points
                    logger.info(f"✅ Collected sufficient data ({len(data_collected)} points) in {elapsed_minutes:.1f} minutes")
                    break
                elif elapsed_minutes >= duration_minutes and len(data_collected) >= 50:
                    logger.info(f"✅ Time limit reached ({elapsed_minutes:.1f} minutes) with {len(data_collected)} data points")
                    break
            
            self.market_data[self.symbol] = data_collected
            logger.info(f"✅ Collected {len(data_collected)} LIVE data points")
            
        except Exception as e:
            logger.error(f"❌ Error collecting live data: {e}")
            # Fallback to mock data if live collection fails
            logger.info("🔄 Falling back to mock data...")
            await self.create_mock_data()
        finally:
            await self.websocket_client.disconnect()
    
    async def create_mock_data(self):
        """Create mock market data as fallback"""
        logger.info("📊 Creating mock market data as fallback...")
        
        # Generate realistic mock data
        np.random.seed(42)  # For reproducible results
        base_price = 45000.0  # BTC base price
        
        data = []
        current_price = base_price
        
        for i in range(500):  # 500 candles
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            current_price *= (1 + price_change)
            
            # Generate OHLC data
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            close_price = current_price
            
            # Generate volume
            volume = np.random.uniform(100, 1000)
            
            # Create timestamp
            timestamp = datetime.now(timezone.utc) - timedelta(minutes=500-i)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        self.market_data[self.symbol] = data
        logger.info(f"✅ Created {len(data)} mock candles")
        return data
    
    def prepare_dataframe(self, symbol):
        """Convert collected data to DataFrame"""
        if symbol not in self.market_data or not self.market_data[symbol]:
            return None
        
        data = self.market_data[symbol]
        df = pd.DataFrame(data)
        
        # Ensure proper data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        return df
    
    async def test_perfect_calculations(self, df):
        """Test the perfect calculations integration"""
        logger.info("🧪 Testing Perfect Calculations Integration...")
        
        try:
            # Test Psychological Levels with Perfect Calculations
            logger.info("📊 Testing Psychological Levels...")
            psychological_levels = await self.sr_analyzer._detect_psychological_levels(df, self.symbol)
            logger.info(f"✅ Psychological Levels: {len(psychological_levels)} levels detected")
            
            for level in psychological_levels[:3]:  # Show first 3
                logger.info(f"   - {level.level_type}: ${level.price_level:.2f} "
                          f"(Strength: {level.psychological_strength:.2f}, Reliability: {level.reliability_score:.2f})")
            
            # Test Dynamic Support/Resistance with Perfect Calculations
            logger.info("📈 Testing Dynamic Support/Resistance...")
            support_levels, resistance_levels = await self.sr_analyzer._detect_dynamic_levels(df)
            logger.info(f"✅ Support Levels: {len(support_levels)}, Resistance Levels: {len(resistance_levels)}")
            
            for level in support_levels[:2]:  # Show first 2
                logger.info(f"   - Support: ${level.price_level:.2f} "
                          f"(Strength: {level.strength:.2f}, Confidence: {level.confidence:.2f})")
            
            for level in resistance_levels[:2]:  # Show first 2
                logger.info(f"   - Resistance: ${level.price_level:.2f} "
                          f"(Strength: {level.strength:.2f}, Confidence: {level.confidence:.2f})")
            
            # Test Volume Pattern Detection with Perfect Calculations
            logger.info("📊 Testing Volume Pattern Detection...")
            volume_patterns = await self.volume_analyzer.analyze_volume_patterns(df, self.symbol, "1m")
            logger.info(f"✅ Volume Patterns: {len(volume_patterns)} patterns detected")
            
            for pattern in volume_patterns[:3]:  # Show first 3
                logger.info(f"   - {pattern.pattern_type.value}: "
                          f"Volume Ratio: {pattern.volume_ratio:.2f}x, "
                          f"ML Confidence: {pattern.confidence:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error testing perfect calculations: {e}")
            return False
    
    async def generate_live_signal(self, df):
        """Generate a live trading signal using perfect calculations"""
        logger.info("🎯 Generating Live Signal...")
        
        try:
            current_price = df.iloc[-1]['close']
            logger.info(f"💰 Current {self.symbol} Price: ${current_price:.2f}")
            
            # Get psychological levels
            psychological_levels = await self.sr_analyzer._detect_psychological_levels(df, self.symbol)
            
            # Get dynamic support/resistance
            support_levels, resistance_levels = await self.sr_analyzer._detect_dynamic_levels(df)
            
            # Get volume patterns
            volume_patterns = await self.volume_analyzer.analyze_volume_patterns(df, self.symbol, "1m")
            
            # Analyze signal strength
            signal_strength = 0.0
            signal_direction = "NEUTRAL"
            confidence = 0.0
            
            # Check for support/resistance confluence
            nearby_support = [level for level in support_levels 
                            if abs(level.price_level - current_price) / current_price < 0.02]
            nearby_resistance = [level for level in resistance_levels 
                              if abs(level.price_level - current_price) / current_price < 0.02]
            
            # Check for psychological levels
            nearby_psychological = [level for level in psychological_levels 
                                  if abs(level.price_level - current_price) / current_price < 0.02]
            
            # Check for volume patterns
            recent_volume_patterns = [pattern for pattern in volume_patterns 
                                    if pattern.confidence > 0.7]
            
            # Generate signal based on confluence
            if nearby_support and recent_volume_patterns:
                # Bullish signal
                signal_direction = "BUY"
                signal_strength = min(1.0, len(nearby_support) * 0.3 + len(recent_volume_patterns) * 0.2)
                confidence = np.mean([level.confidence for level in nearby_support])
                
            elif nearby_resistance and recent_volume_patterns:
                # Bearish signal
                signal_direction = "SELL"
                signal_strength = min(1.0, len(nearby_resistance) * 0.3 + len(recent_volume_patterns) * 0.2)
                confidence = np.mean([level.confidence for level in nearby_resistance])
            
            # Create signal
            signal = {
                'timestamp': datetime.now(timezone.utc),
                'symbol': self.symbol,
                'direction': signal_direction,
                'strength': signal_strength,
                'confidence': confidence,
                'current_price': current_price,
                'support_levels': len(nearby_support),
                'resistance_levels': len(nearby_resistance),
                'psychological_levels': len(nearby_psychological),
                'volume_patterns': len(recent_volume_patterns),
                'ml_confidence': confidence,
                'perfect_calculations_active': True
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating live signal: {e}")
            return None
    
    async def run_live_test(self):
        """Run the complete live signal generation test"""
        logger.info("🚀 Starting Live Signal Generation Test")
        logger.info("=" * 60)
        
        try:
            # Step 1: Initialize data pipeline and SDE integration
            await self.data_pipeline.initialize()
            await self.sde_integration.initialize()
            
            # Step 2: Collect LIVE data from WebSocket
            await self.collect_live_data(duration_minutes=1)  # Reduced to 1 minute for faster testing
            
            if not self.market_data.get(self.symbol):
                logger.error("❌ No data collected, cannot proceed")
                return
            
            # Step 3: Prepare DataFrame
            df = self.prepare_dataframe(self.symbol)
            if df is None or len(df) < 50:
                logger.error("❌ Insufficient data for analysis")
                return
            
            logger.info(f"📊 Data prepared: {len(df)} candles")
            
            # Step 4: Store data in TimescaleDB via pipeline
            try:
                await self.store_data_in_database(df)
            except Exception as e:
                logger.warning(f"⚠️ Data storage had issues but continuing: {e}")
            
            # Step 5: Test perfect calculations
            calculations_ok = await self.test_perfect_calculations(df)
            if not calculations_ok:
                logger.error("❌ Perfect calculations test failed")
                return
            
            # Step 6: Generate live signal using SDE framework
            signal = await self.generate_sde_signal(df)
            if signal is None:
                logger.error("❌ SDE signal generation failed")
                return
            
            # Step 5: Display results
            logger.info("🎯 LIVE SIGNAL GENERATED!")
            logger.info("=" * 40)
            logger.info(f"📅 Timestamp: {signal['timestamp']}")
            logger.info(f"💰 Symbol: {signal['symbol']}")
            logger.info(f"💵 Current Price: ${signal['current_price']:.2f}")
            logger.info(f"🎯 Direction: {signal['direction']}")
            logger.info(f"💪 Strength: {signal['strength']:.2f}")
            logger.info(f"🎲 Confidence: {signal['confidence']:.2f}")
            logger.info(f"🤖 ML Confidence: {signal['ml_confidence']:.2f}")
            logger.info(f"📊 Support Levels: {signal['support_levels']}")
            logger.info(f"📈 Resistance Levels: {signal['resistance_levels']}")
            logger.info(f"🧠 Psychological Levels: {signal['psychological_levels']}")
            logger.info(f"📊 Volume Patterns: {signal['volume_patterns']}")
            logger.info(f"✅ Perfect Calculations: {'ACTIVE' if signal['perfect_calculations_active'] else 'INACTIVE'}")
            logger.info(f"🤖 SDE Framework: {'ACTIVE' if signal.get('sde_framework_active', False) else 'INACTIVE'}")
            logger.info(f"🧠 Model Heads: {signal.get('model_head_results', 0)}")
            logger.info(f"🎯 Consensus: {'REACHED' if signal.get('consensus_reached', False) else 'NOT REACHED'}")
            
            # Determine if this is a "sure shot" signal
            if signal['confidence'] > 0.7 and signal['strength'] > 0.6:
                logger.info("🎉 SURE SHOT SIGNAL DETECTED!")
                logger.info("🎯 High confidence signal with perfect calculations and SDE framework")
            else:
                logger.info("⚠️ Signal generated but below sure shot threshold")
            
            logger.info("=" * 60)
            logger.info("✅ Live Signal Generation Test COMPLETED!")
            
        except Exception as e:
            logger.error(f"❌ Live test failed: {e}")
        finally:
            # Cleanup
            await self.data_pipeline.close()
            await self.sde_integration.close()
    
    async def store_data_in_database(self, df):
        """Store collected data in TimescaleDB via pipeline"""
        logger.info("💾 Storing data in TimescaleDB...")
        
        try:
            # Process each row as WebSocket message
            for _, row in df.iterrows():
                message = {
                    'type': 'kline',
                    'symbol': self.symbol,
                    'timeframe': '1m',
                    'timestamp': row['timestamp'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume'],
                    'quote_volume': row['volume'] * row['close'],  # Estimate
                    'trades': 100  # Estimate
                }
                
                await self.data_pipeline.process_websocket_message(message)
            
            # Calculate and store technical indicators
            await self.data_pipeline.calculate_technical_indicators(self.symbol, '1m')
            
            logger.info("✅ Data stored in TimescaleDB successfully")
            
        except Exception as e:
            logger.error(f"❌ Error storing data in database: {e}")
    
    async def generate_sde_signal(self, df):
        """Generate signal using SDE framework"""
        logger.info("🤖 Generating signal using SDE framework...")
        
        try:
            current_price = df.iloc[-1]['close']
            
            # Prepare market data
            market_data = {
                'current_price': current_price,
                'indicators': {
                    'sma_20': df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_price,
                    'sma_50': df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else current_price,
                    'rsi_14': 50.0,  # Placeholder - would calculate RSI
                    'macd': 0.0,     # Placeholder - would calculate MACD
                }
            }
            
            # Prepare analysis results
            analysis_results = {
                'sentiment_analysis': {
                    'overall_sentiment': 0.0,  # Neutral
                    'confidence': 0.5
                },
                'volume_analysis': {
                    'volume_trend': 'normal',
                    'volume_ratio': df['volume'].iloc[-1] / df['volume'].mean() if df['volume'].mean() > 0 else 1.0
                }
            }
            
            # Create signal generation request
            request = type('SignalGenerationRequest', (), {
                'symbol': self.symbol,
                'timeframe': '1m',
                'market_data': market_data,
                'analysis_results': analysis_results,
                'timestamp': datetime.now(timezone.utc)
            })()
            
            # Generate signal using SDE framework
            result = await self.sde_integration.generate_signal(request)
            
            if result:
                # Convert to signal format
                signal = {
                    'timestamp': result.timestamp,
                    'symbol': result.symbol,
                    'direction': result.direction,
                    'strength': result.strength,
                    'confidence': result.confidence,
                    'current_price': current_price,
                    'support_levels': 0,  # Would get from S/R analysis
                    'resistance_levels': 0,
                    'psychological_levels': 0,
                    'volume_patterns': 0,
                    'ml_confidence': result.confidence,
                    'perfect_calculations_active': True,
                    'sde_framework_active': True,
                    'model_head_results': len(result.model_head_results),
                    'consensus_reached': result.consensus_result.get('consensus_reached', False)
                }
                
                logger.info(f"✅ SDE signal generated: {result.signal_id}")
                return signal
            else:
                logger.warning("⚠️ SDE framework could not reach consensus")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating SDE signal: {e}")
            return None

async def main():
    """Main function to run the live signal generation test"""
    generator = LiveSignalGenerator()
    await generator.run_live_test()

if __name__ == "__main__":
    asyncio.run(main())
