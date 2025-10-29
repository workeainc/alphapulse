#!/usr/bin/env python3
"""
Advanced Pattern Recognition Integration Example
Shows how to integrate with existing AlphaPlus analyzing engine
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any

# Import your existing components
from src.strategies.vectorized_pattern_detector import VectorizedPatternDetector
from src.database.connection import TimescaleDBConnection

# Import new advanced pattern components
from src.ai.multi_timeframe_pattern_engine import MultiTimeframePatternEngine
from src.ai.pattern_failure_analyzer import PatternFailureAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPatternIntegration:
    """Integration layer between existing and advanced pattern recognition"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.db_connection = None
        
        # Existing components
        self.pattern_detector = VectorizedPatternDetector()
        
        # New advanced components
        self.multi_timeframe_engine = MultiTimeframePatternEngine(db_config)
        self.failure_analyzer = PatternFailureAnalyzer(db_config)
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize existing database connection
            self.db_connection = TimescaleDBConnection(self.db_config)
            await self.db_connection.initialize()
            
            # Initialize advanced pattern engines
            await self.multi_timeframe_engine.initialize()
            await self.failure_analyzer.initialize()
            
            logger.info("✅ Advanced Pattern Integration initialized")
            
        except Exception as e:
            logger.error(f"❌ Integration initialization failed: {e}")
            raise
    
    async def enhanced_pattern_analysis(self, symbol: str, candlestick_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced pattern analysis combining existing and advanced features"""
        try:
            # Step 1: Run existing pattern detection
            logger.info(f"🔍 Running existing pattern detection for {symbol}")
            
            # Your existing pattern detection logic here
            existing_patterns = await self.pattern_detector.detect_patterns_vectorized(
                candlestick_data, use_talib=True, use_incremental=True
            )
            
            if not existing_patterns:
                return {"status": "no_patterns", "patterns": []}
            
            # Step 2: Enhanced analysis for each pattern
            enhanced_results = []
            
            for pattern in existing_patterns:
                enhanced_pattern = await self._enhance_pattern_analysis(symbol, pattern, candlestick_data)
                enhanced_results.append(enhanced_pattern)
            
            return {
                "status": "success",
                "patterns": enhanced_results,
                "total_patterns": len(enhanced_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced pattern analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _enhance_pattern_analysis(self, symbol: str, pattern: Any, candlestick_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single pattern with advanced analysis"""
        try:
            # Step 1: Multi-timeframe confirmation
            logger.info(f"🔄 Running multi-timeframe analysis for {pattern.pattern_name}")
            
            multi_timeframe_result = await self.multi_timeframe_engine.detect_multi_timeframe_patterns(
                symbol, "1h", candlestick_data
            )
            
            # Step 2: Pattern failure prediction
            logger.info(f"⚠️ Running failure prediction for {pattern.pattern_name}")
            
            failure_prediction = await self.failure_analyzer.predict_pattern_failure(
                {
                    "pattern_id": pattern.pattern_id,
                    "symbol": symbol,
                    "pattern_name": pattern.pattern_name,
                    "confidence": pattern.confidence,
                    "strength": pattern.strength
                },
                {"ohlcv": candlestick_data}
            )
            
            # Step 3: Combine results
            enhanced_pattern = {
                # Original pattern data
                "original_pattern": {
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.pattern_name,
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "strength": pattern.strength,
                    "timestamp": pattern.timestamp,
                    "price_level": pattern.price_level
                },
                
                # Multi-timeframe confirmation
                "multi_timeframe": {
                    "confirmation_score": multi_timeframe_result[0].confirmation_score if multi_timeframe_result else 0.0,
                    "trend_alignment": multi_timeframe_result[0].trend_alignment if multi_timeframe_result else "neutral",
                    "timeframe_confirmations": len(multi_timeframe_result[0].confirmation_timeframes) if multi_timeframe_result else 0
                } if multi_timeframe_result else None,
                
                # Failure prediction
                "failure_prediction": {
                    "failure_probability": failure_prediction.failure_probability if failure_prediction else 0.5,
                    "failure_confidence": failure_prediction.failure_confidence if failure_prediction else 0.5,
                    "failure_reasons": failure_prediction.failure_reasons if failure_prediction else []
                } if failure_prediction else None,
                
                # Enhanced confidence score
                "enhanced_confidence": self._calculate_enhanced_confidence(
                    pattern, multi_timeframe_result, failure_prediction
                ),
                
                # Analysis metadata
                "analysis_timestamp": datetime.now(),
                "analysis_method": "enhanced_integration"
            }
            
            # Store enhanced results in database
            await self._store_enhanced_analysis(enhanced_pattern)
            
            return enhanced_pattern
            
        except Exception as e:
            logger.error(f"❌ Pattern enhancement failed: {e}")
            return {
                "original_pattern": pattern,
                "error": str(e),
                "analysis_timestamp": datetime.now()
            }
    
    def _calculate_enhanced_confidence(self, pattern: Any, multi_timeframe_result: List, failure_prediction: Any) -> float:
        """Calculate enhanced confidence score"""
        try:
            base_confidence = pattern.confidence
            
            # Multi-timeframe bonus
            mtf_bonus = 0.0
            if multi_timeframe_result:
                mtf_pattern = multi_timeframe_result[0]
                mtf_bonus = (mtf_pattern.confirmation_score / 100) * 0.3  # Up to 30% bonus
            
            # Failure prediction penalty
            failure_penalty = 0.0
            if failure_prediction:
                failure_penalty = failure_prediction.failure_probability * 0.2  # Up to 20% penalty
            
            # Calculate enhanced confidence
            enhanced_confidence = base_confidence + mtf_bonus - failure_penalty
            
            return max(0.0, min(1.0, enhanced_confidence))
            
        except Exception as e:
            logger.error(f"❌ Enhanced confidence calculation failed: {e}")
            return pattern.confidence
    
    async def _store_enhanced_analysis(self, enhanced_pattern: Dict[str, Any]):
        """Store enhanced analysis results"""
        try:
            # Store in existing signals table (if compatible)
            if self.db_connection:
                async with self.db_connection.get_async_session() as session:
                    from sqlalchemy import text
                    
                    # Store in your existing signals table with enhanced data
                    query = text("""
                        INSERT INTO signals (
                            symbol, pattern_name, confidence, strength, timestamp, 
                            price_level, metadata, created_at
                        ) VALUES (
                            :symbol, :pattern_name, :confidence, :strength, :timestamp,
                            :price_level, :metadata, NOW()
                        )
                    """)
                    
                    await session.execute(query, {
                        "symbol": enhanced_pattern["original_pattern"]["symbol"],
                        "pattern_name": enhanced_pattern["original_pattern"]["pattern_name"],
                        "confidence": enhanced_pattern["enhanced_confidence"],
                        "strength": enhanced_pattern["original_pattern"]["strength"],
                        "timestamp": enhanced_pattern["original_pattern"]["timestamp"],
                        "price_level": enhanced_pattern["original_pattern"]["price_level"],
                        "metadata": enhanced_pattern
                    })
                    
                    await session.commit()
                    logger.info("✅ Enhanced analysis stored in database")
                    
        except Exception as e:
            logger.error(f"❌ Failed to store enhanced analysis: {e}")
    
    async def get_enhanced_signals(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get enhanced signals for a symbol"""
        try:
            if not self.db_connection:
                return []
            
            async with self.db_connection.get_async_session() as session:
                from sqlalchemy import text
                
                # Query existing signals with enhanced metadata
                query = text("""
                    SELECT * FROM signals 
                    WHERE symbol = :symbol 
                    AND metadata IS NOT NULL
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """)
                
                result = await session.execute(query, {"symbol": symbol, "limit": limit})
                rows = result.fetchall()
                
                enhanced_signals = []
                for row in rows:
                    signal = {
                        "signal_id": row.signal_id,
                        "symbol": row.symbol,
                        "pattern_name": row.pattern_name,
                        "confidence": row.confidence,
                        "strength": row.strength,
                        "timestamp": row.timestamp,
                        "price_level": row.price_level,
                        "enhanced_data": row.metadata
                    }
                    enhanced_signals.append(signal)
                
                return enhanced_signals
                
        except Exception as e:
            logger.error(f"❌ Failed to get enhanced signals: {e}")
            return []
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.multi_timeframe_engine:
                await self.multi_timeframe_engine.cleanup()
            if self.failure_analyzer:
                await self.failure_analyzer.cleanup()
            if self.db_connection:
                await self.db_connection.close()
            logger.info("✅ Advanced Pattern Integration cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

# Usage example
async def main():
    """Example usage of the integration"""
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'alphapulse',  # Your existing database
        'user': 'postgres',
        'password': 'Emon_@17711'
    }
    
    integration = AdvancedPatternIntegration(db_config)
    
    try:
        await integration.initialize()
        
        # Example candlestick data (your existing format)
        candlestick_data = {
            "timestamp": [datetime.now()],
            "open": [50000.0],
            "high": [51000.0],
            "low": [49000.0],
            "close": [50500.0],
            "volume": [1000.0]
        }
        
        # Run enhanced analysis
        result = await integration.enhanced_pattern_analysis("BTCUSDT", candlestick_data)
        print(f"Enhanced Analysis Result: {result}")
        
        # Get enhanced signals
        signals = await integration.get_enhanced_signals("BTCUSDT", 10)
        print(f"Enhanced Signals: {len(signals)} found")
        
    finally:
        await integration.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
