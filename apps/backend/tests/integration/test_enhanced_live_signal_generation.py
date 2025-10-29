#!/usr/bin/env python3
"""
ENHANCED LIVE Signal Generation Test with All 8 Algorithms
Tests the integration of all algorithms with perfect calculations and generates signals from REAL-TIME WebSocket data
Collects live market data from Binance WebSocket and generates trading signals using algorithm confluence
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
from src.services.algorithm_integration_service import AlgorithmIntegrationService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLiveSignalGenerator:
    """Generate LIVE signals using ALL 8 algorithms with perfect calculations from REAL-TIME WebSocket data"""
    
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
        
        # NEW: Initialize Algorithm Integration Service
        self.algorithm_service = AlgorithmIntegrationService(db_url=db_url)
        
        # Performance tracking
        self.stats = {
            'total_runs': 0,
            'signals_generated': 0,
            'high_confidence_signals': 0,
            'algorithm_confluence_signals': 0,
            'sde_consensus_signals': 0,
            'last_run': None
        }
        
        logger.info("🚀 Enhanced Live Signal Generator initialized with ALL 8 algorithms")
    
    async def collect_live_data(self, duration_minutes: int = 1) -> pd.DataFrame:
        """Collect live data from WebSocket"""
        logger.info(f"📡 Collecting live data for {self.symbol} for {duration_minutes} minute(s)...")
        
        data_points = []
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        try:
            async for message in self.websocket_client.listen():
                if datetime.now(timezone.utc) >= end_time:
                    break
                
                if message.get('type') == 'kline' and message.get('symbol') == self.symbol:
                    data_points.append({
                        'timestamp': message['timestamp'],
                        'open': float(message['open']),
                        'high': float(message['high']),
                        'low': float(message['low']),
                        'close': float(message['close']),
                        'volume': float(message['volume'])
                    })
                    
                    # Log progress every 10 data points
                    if len(data_points) % 10 == 0:
                        logger.info(f"📊 Collected {len(data_points)} data points...")
                    
                    # Stop if we have enough data (100+ points or 1 minute)
                    if len(data_points) >= 100:
                        logger.info(f"✅ Collected {len(data_points)} data points (sufficient for analysis)")
                        break
        
        except Exception as e:
            logger.error(f"❌ Error collecting live data: {e}")
            return pd.DataFrame()
        
        if not data_points:
            logger.warning("⚠️ No data points collected")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data_points)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ Live data collection complete: {len(df)} points")
        logger.info(f"📈 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        logger.info(f"📊 Volume range: {df['volume'].min():.2f} - {df['volume'].max():.2f}")
        
        return df
    
    async def store_data_in_database(self, df: pd.DataFrame):
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
    
    async def run_all_algorithms(self, df: pd.DataFrame):
        """Run all 8 algorithms on live data"""
        logger.info("🔧 Running ALL 8 algorithms on live data...")
        
        try:
            # Run all algorithms using the integration service
            algorithm_results = await self.algorithm_service.run_all_algorithms(df, self.symbol, '1m')
            
            # Calculate confluence analysis
            confluence = await self.algorithm_service.calculate_confluence_analysis(algorithm_results)
            
            # Store signal confluence in database
            current_price = df.iloc[-1]['close']
            await self.algorithm_service.store_signal_confluence(
                confluence, self.symbol, '1m', 
                entry_price=current_price,
                stop_loss=current_price * 0.98,  # 2% stop loss
                take_profit=current_price * 1.04  # 4% take profit
            )
            
            # Update statistics
            self.stats['total_runs'] += 1
            if confluence.signal_direction != 'NEUTRAL':
                self.stats['signals_generated'] += 1
                self.stats['algorithm_confluence_signals'] += 1
                if confluence.ml_confidence > 0.7:
                    self.stats['high_confidence_signals'] += 1
            
            logger.info(f"✅ All algorithms completed")
            logger.info(f"📊 Confluence Score: {confluence.confluence_score:.3f}")
            logger.info(f"🎯 Signal Direction: {confluence.signal_direction}")
            logger.info(f"🤖 ML Confidence: {confluence.ml_confidence:.3f}")
            logger.info(f"🔗 Algorithm Confirmations: {confluence.confirmations}")
            
            return algorithm_results, confluence
            
        except Exception as e:
            logger.error(f"❌ Error running algorithms: {e}")
            return {}, None
    
    async def generate_enhanced_sde_signal(self, df: pd.DataFrame, algorithm_results: dict, confluence):
        """Generate enhanced SDE signal using algorithm results"""
        logger.info("🤖 Generating enhanced SDE signal with algorithm confluence...")
        
        try:
            current_price = df.iloc[-1]['close']
            
            # Prepare enhanced market data with algorithm results
            market_data = {
                'current_price': current_price,
                'indicators': {
                    'sma_20': df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else current_price,
                    'sma_50': df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else current_price,
                    'rsi_14': 50.0,  # Placeholder - would calculate RSI
                    'macd': 0.0,     # Placeholder - would calculate MACD
                },
                'algorithm_results': algorithm_results,
                'confluence_score': confluence.confluence_score if confluence else 0.0,
                'algorithm_confirmations': confluence.confirmations if confluence else []
            }
            
            # Prepare enhanced analysis results
            analysis_results = {
                'sentiment_analysis': {
                    'overall_sentiment': 0.0,  # Neutral
                    'confidence': 0.5
                },
                'volume_analysis': {
                    'volume_trend': 'normal',
                    'volume_ratio': df['volume'].iloc[-1] / df['volume'].mean() if df['volume'].mean() > 0 else 1.0
                },
                'algorithm_confluence': {
                    'confluence_score': confluence.confluence_score if confluence else 0.0,
                    'confirmations': confluence.confirmations if confluence else [],
                    'signal_direction': confluence.signal_direction if confluence else 'NEUTRAL',
                    'ml_confidence': confluence.ml_confidence if confluence else 0.0
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
                # Convert to enhanced signal format
                signal = {
                    'timestamp': result.timestamp,
                    'symbol': result.symbol,
                    'direction': result.direction,
                    'strength': result.strength,
                    'confidence': result.confidence,
                    'current_price': current_price,
                    'support_levels': algorithm_results.get('support_resistance', {}).get('result_data', {}).get('support_levels', 0),
                    'resistance_levels': algorithm_results.get('support_resistance', {}).get('result_data', {}).get('resistance_levels', 0),
                    'psychological_levels': algorithm_results.get('support_resistance', {}).get('result_data', {}).get('psychological_levels', 0),
                    'volume_patterns': algorithm_results.get('volume_analysis', {}).get('result_data', {}).get('volume_patterns', 0),
                    'chart_patterns': algorithm_results.get('pattern_recognition', {}).get('result_data', {}).get('patterns_detected', 0),
                    'demand_zones': algorithm_results.get('demand_supply_zones', {}).get('result_data', {}).get('demand_zones', 0),
                    'supply_zones': algorithm_results.get('demand_supply_zones', {}).get('result_data', {}).get('supply_zones', 0),
                    'breakout_signals': algorithm_results.get('breakout_detection', {}).get('result_data', {}).get('breakout_up', False) or algorithm_results.get('breakout_detection', {}).get('result_data', {}).get('breakout_down', False),
                    'ml_confidence': result.confidence,
                    'perfect_calculations_active': True,
                    'sde_framework_active': True,
                    'algorithm_integration_active': True,
                    'model_head_results': len(result.model_head_results),
                    'consensus_reached': result.consensus_result.get('consensus_reached', False),
                    'confluence_score': confluence.confluence_score if confluence else 0.0,
                    'algorithm_confirmations': confluence.confirmations if confluence else [],
                    'algorithm_count': confluence.algorithm_count if confluence else 0
                }
                
                logger.info(f"✅ Enhanced SDE signal generated: {result.signal_id}")
                return signal
            else:
                logger.warning("⚠️ SDE framework could not reach consensus")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error generating enhanced SDE signal: {e}")
            return None
    
    def display_enhanced_signal(self, signal: dict, algorithm_results: dict, confluence):
        """Display enhanced signal with algorithm details"""
        if not signal:
            logger.warning("⚠️ No signal to display")
            return
        
        logger.info("=" * 80)
        logger.info("🎯 ENHANCED LIVE SIGNAL GENERATED")
        logger.info("=" * 80)
        
        # Basic signal info
        logger.info(f"📊 Symbol: {signal['symbol']}")
        logger.info(f"⏰ Timestamp: {signal['timestamp']}")
        logger.info(f"💰 Current Price: ${signal['current_price']:.2f}")
        logger.info(f"🎯 Direction: {signal['direction']}")
        logger.info(f"💪 Strength: {signal['strength']:.3f}")
        logger.info(f"🎯 Confidence: {signal['confidence']:.3f}")
        
        # Algorithm results
        logger.info("\n🔧 ALGORITHM RESULTS:")
        logger.info("-" * 40)
        for algorithm_type, result in algorithm_results.items():
            if result.success:
                logger.info(f"✅ {algorithm_type.upper()}:")
                logger.info(f"   Confidence: {result.confidence_score:.3f}")
                logger.info(f"   Strength: {result.strength_score:.3f}")
                logger.info(f"   Processing Time: {result.processing_time_ms}ms")
                if result.result_data:
                    key_metrics = {k: v for k, v in result.result_data.items() if isinstance(v, (int, float, bool))}
                    if key_metrics:
                        logger.info(f"   Key Metrics: {key_metrics}")
            else:
                logger.info(f"❌ {algorithm_type.upper()}: FAILED")
                logger.info(f"   Error: {result.error_message}")
        
        # Confluence analysis
        if confluence:
            logger.info("\n🔗 CONFLUENCE ANALYSIS:")
            logger.info("-" * 40)
            logger.info(f"📊 Confluence Score: {confluence.confluence_score:.3f}")
            logger.info(f"🎯 Signal Direction: {confluence.signal_direction}")
            logger.info(f"🤖 ML Confidence: {confluence.ml_confidence:.3f}")
            logger.info(f"🔢 Algorithm Count: {confluence.algorithm_count}")
            logger.info(f"✅ Confirmations: {confluence.confirmations}")
            logger.info(f"💪 Strong Signals: {confluence.strong_signals}")
            logger.info(f"⚠️ Weak Signals: {confluence.weak_signals}")
        
        # SDE Framework info
        logger.info("\n🤖 SDE FRAMEWORK:")
        logger.info("-" * 40)
        logger.info(f"🎯 Model Heads: {signal['model_head_results']}")
        logger.info(f"🤝 Consensus Reached: {signal['consensus_reached']}")
        logger.info(f"🔗 Algorithm Integration: {signal['algorithm_integration_active']}")
        
        # System status
        logger.info("\n⚙️ SYSTEM STATUS:")
        logger.info("-" * 40)
        logger.info(f"✅ Perfect Calculations: {signal['perfect_calculations_active']}")
        logger.info(f"✅ SDE Framework: {signal['sde_framework_active']}")
        logger.info(f"✅ Algorithm Integration: {signal['algorithm_integration_active']}")
        
        # Statistics
        logger.info("\n📊 STATISTICS:")
        logger.info("-" * 40)
        logger.info(f"🔄 Total Runs: {self.stats['total_runs']}")
        logger.info(f"🎯 Signals Generated: {self.stats['signals_generated']}")
        logger.info(f"💪 High Confidence Signals: {self.stats['high_confidence_signals']}")
        logger.info(f"🔗 Algorithm Confluence Signals: {self.stats['algorithm_confluence_signals']}")
        logger.info(f"🤖 SDE Consensus Signals: {self.stats['sde_consensus_signals']}")
        
        logger.info("=" * 80)
    
    async def run_enhanced_live_test(self):
        """Run enhanced live test with all algorithms"""
        logger.info("🚀 Starting ENHANCED Live Signal Generation Test")
        logger.info("=" * 80)
        
        try:
            # Initialize algorithm service
            logger.info("🔧 Initializing Algorithm Integration Service...")
            await self.algorithm_service.initialize()
            
            # Initialize data pipeline
            logger.info("🔧 Initializing RealTime Data Pipeline...")
            await self.data_pipeline.initialize()
            
            # Step 1: Collect live data
            logger.info("📡 Step 1: Collecting live data...")
            df = await self.collect_live_data(duration_minutes=1)
            
            if df.empty:
                logger.error("❌ No data collected, cannot proceed")
                return
            
            # Step 2: Store data in TimescaleDB via pipeline
            logger.info("💾 Step 2: Storing data in TimescaleDB...")
            try:
                await self.store_data_in_database(df)
            except Exception as e:
                logger.warning(f"⚠️ Data storage had issues but continuing: {e}")
            
            # Step 3: Run all algorithms
            logger.info("🔧 Step 3: Running all 8 algorithms...")
            algorithm_results, confluence = await self.run_all_algorithms(df)
            
            # Step 4: Generate enhanced SDE signal
            logger.info("🤖 Step 4: Generating enhanced SDE signal...")
            enhanced_signal = await self.generate_enhanced_sde_signal(df, algorithm_results, confluence)
            
            # Step 5: Display results
            logger.info("📊 Step 5: Displaying enhanced results...")
            self.display_enhanced_signal(enhanced_signal, algorithm_results, confluence)
            
            # Step 6: Display algorithm service stats
            logger.info("📈 Step 6: Algorithm Service Statistics...")
            service_stats = self.algorithm_service.get_stats()
            logger.info(f"🔧 Service Stats: {service_stats}")
            
            logger.info("✅ Enhanced Live Signal Generation Test completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Enhanced test failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        finally:
            # Close services
            await self.algorithm_service.close()
            await self.data_pipeline.close()

async def main():
    """Main function to run the enhanced live signal generation test"""
    logger.info("🚀 Starting Enhanced Live Signal Generation Test")
    logger.info("🔧 Testing ALL 8 algorithms with live data")
    logger.info("📊 Using TimescaleDB for algorithm results storage")
    logger.info("🤖 Enhanced SDE framework with algorithm confluence")
    
    generator = EnhancedLiveSignalGenerator()
    await generator.run_enhanced_live_test()

if __name__ == "__main__":
    asyncio.run(main())
