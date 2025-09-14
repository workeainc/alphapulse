"""
SDE Integration Manager - Connects SDE Framework with Intelligent Signal Generator
Handles model consensus, confluence scoring, execution quality, and explainability
"""

import asyncio
import logging
import uuid
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncpg

from .sde_framework import SDEFramework, ModelHeadResult, ConsensusResult, ConfluenceScore, ExecutionQuality, SignalDirection, SDEOutput
from .sde_calibration import SDECalibrationSystem
from .divergence_analyzer import AdvancedDivergenceAnalyzer, DivergenceAnalysis

logger = logging.getLogger(__name__)

@dataclass
class SDEIntegrationResult:
    """Result of SDE integration process"""
    signal_id: str
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Core SDE Results
    consensus_result: ConsensusResult
    confluence_result: ConfluenceScore
    execution_result: ExecutionQuality
    
    # Divergence Analysis
    divergence_analysis: Optional[DivergenceAnalysis] = None
    
    # Enhanced Results
    news_blackout_result: Optional[Any] = None
    signal_limits_result: Optional[Any] = None
    tp_structure_result: Optional[Any] = None
    
    # Calibration Results
    calibrated_confidence: float = 0.0
    calibration_method: str = ""
    
    # Integration Metadata
    processing_time_ms: int = 0
    all_gates_passed: bool = False
    final_confidence: float = 0.0
    integration_reason: str = ""

class SDEIntegrationManager:
    """Manages integration between SDE Framework and Signal Generator"""
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.sde_framework = SDEFramework(db_pool)
        self.calibration_system = SDECalibrationSystem(db_pool)
        self.divergence_analyzer = AdvancedDivergenceAnalyzer(db_pool)
        self.config = {}
        self.performance_stats = {
            'total_integrations': 0,
            'successful_integrations': 0,
            'failed_integrations': 0,
            'avg_processing_time_ms': 0.0
        }
        
    async def load_integration_config(self):
        """Load SDE integration configuration"""
        try:
            async with self.db_pool.acquire() as conn:
                configs = await conn.fetch("""
                    SELECT config_name, config_data 
                    FROM sde_integration_config 
                    WHERE is_active = true
                """)
                
                for config in configs:
                    config_name = config['config_name']
                    config_data = config['config_data']
                    
                    if isinstance(config_data, str):
                        config_data = json.loads(config_data)
                    
                    self.config[config_name] = config_data
                
                logger.info(f"✅ Loaded {len(configs)} SDE integration configurations")
                
        except Exception as e:
            logger.error(f"❌ Failed to load integration config: {e}")
    
    async def integrate_sde_with_signal(self, 
                                      signal_id: str,
                                      symbol: str, 
                                      timeframe: str,
                                      analysis_results: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      account_id: str = "default") -> SDEIntegrationResult:
        """Integrate SDE framework with signal generation"""
        
        start_time = datetime.now()
        
        try:
            await self.load_integration_config()
            integration_config = self.config.get('sde_integration_default', {})
            
            # 1. Create Model Head Results
            head_results = await self.sde_framework.create_enhanced_model_head_results(
                analysis_results, market_data, symbol, timeframe
            )
            
            # 2. Check Model Consensus
            consensus_result = await self.sde_framework.check_model_consensus(head_results)
            
            # 3. Calculate Confluence Score
            market_data_df = market_data.get('market_data_df', pd.DataFrame())
            confluence_result = await self.sde_framework.calculate_confluence_score(market_data_df, symbol, timeframe)
            
            # 4. Analyze Divergences
            divergence_analysis = None
            if 'market_data_df' in market_data:
                divergence_analysis = await self.divergence_analyzer.analyze_divergences(
                    market_data['market_data_df'], symbol, timeframe
                )
            
            # 5. Assess Execution Quality
            execution_result = await self.sde_framework.assess_execution_quality(market_data)
            
            # 6. Check News Blackout
            news_blackout_result = await self.sde_framework.check_news_blackout(symbol, datetime.now())
            
            # 7. Check Signal Limits
            signal_limits_result = await self.sde_framework.check_signal_limits(account_id, symbol, timeframe)
            
            # 8. Calculate TP Structure (if consensus achieved)
            tp_structure_result = None
            if consensus_result.achieved and consensus_result.direction != SignalDirection.FLAT:
                entry_price = market_data.get('current_price', 0.0)
                stop_loss = market_data.get('stop_loss', 0.0)
                atr_value = market_data.get('atr_value', 0.0)
                direction = consensus_result.direction.value
                
                tp_structure_result = await self.sde_framework.calculate_tp_structure(
                    entry_price, stop_loss, atr_value, direction
                )
            
            # 9. Calibrate Confidence
            calibrated_confidence = 0.0
            calibration_method = "none"
            
            if consensus_result.achieved and consensus_result.probability > 0:
                calibration_result = await self.calibration_system.calibrate_probability(
                    consensus_result.probability,
                    method='isotonic',
                    model_name='head_a',
                    symbol=symbol,
                    timeframe=timeframe
                )
                calibrated_confidence = calibration_result.calibrated_probability
                calibration_method = calibration_result.method
            
            # 10. Apply Integration Gates
            all_gates_passed = self._apply_integration_gates(
                consensus_result, confluence_result, execution_result,
                news_blackout_result, signal_limits_result, divergence_analysis, integration_config
            )
            
            # 11. Calculate Final Confidence
            final_confidence = self._calculate_final_confidence(
                consensus_result, confluence_result, execution_result,
                calibrated_confidence, divergence_analysis, integration_config
            )
            
            # 12. Store Integration Results
            await self._store_integration_results(
                signal_id, symbol, timeframe, consensus_result, confluence_result,
                execution_result, news_blackout_result, signal_limits_result,
                tp_structure_result, calibrated_confidence, final_confidence, divergence_analysis
            )
            
            # 13. Update Performance Stats
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(processing_time, all_gates_passed)
            
            return SDEIntegrationResult(
                signal_id=signal_id,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                consensus_result=consensus_result,
                confluence_result=confluence_result,
                execution_result=execution_result,
                divergence_analysis=divergence_analysis,
                news_blackout_result=news_blackout_result,
                signal_limits_result=signal_limits_result,
                tp_structure_result=tp_structure_result,
                calibrated_confidence=calibrated_confidence,
                calibration_method=calibration_method,
                processing_time_ms=int(processing_time),
                all_gates_passed=all_gates_passed,
                final_confidence=final_confidence,
                integration_reason=self._generate_integration_reason(
                    consensus_result, confluence_result, execution_result,
                    news_blackout_result, signal_limits_result, divergence_analysis
                )
            )
            
        except Exception as e:
            logger.error(f"❌ SDE integration failed for {symbol}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(processing_time, False)
            
            # Return failed result
            return SDEIntegrationResult(
                signal_id=signal_id,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                consensus_result=ConsensusResult(
                    consensus_achieved=False, 
                    consensus_direction=None, 
                    consensus_score=0.0, 
                    agreeing_heads=[], 
                    disagreeing_heads=[], 
                    confidence_threshold=0.85
                ),
                confluence_result=ConfluenceScore(
                    zone_score=0.0, htf_bias=0.0, trigger_quality=0.0, 
                    fvg_confluence=0.0, ob_alignment=0.0, total_score=0.0
                ),
                execution_result=ExecutionQuality(
                    spread_ok=False, impact_cost=0.0, volatility_regime="unknown",
                    liquidity_score=0.0, execution_score=0.0
                ),
                processing_time_ms=int(processing_time),
                all_gates_passed=False,
                final_confidence=0.0,
                integration_reason=f"Integration failed: {e}"
            )
    
    def _create_model_head_results(self, analysis_results: Dict[str, Any]) -> List[ModelHeadResult]:
        """Create model head results from analysis"""
        try:
            # Head A - Technical Analysis
            technical_confidence = analysis_results.get('technical_confidence', 0.5)
            technical_direction = analysis_results.get('technical_direction', 'neutral')
            head_a = ModelHeadResult(
                head_type=ModelHead.HEAD_A,
                direction=SignalDirection.LONG if technical_direction == 'long' else 
                         SignalDirection.SHORT if technical_direction == 'short' else SignalDirection.FLAT,
                probability=technical_confidence,
                confidence=technical_confidence,
                features_used=['rsi', 'macd', 'volume'],
                reasoning='Technical analysis based on multiple indicators'
            )
            
            # Head B - Sentiment Analysis
            sentiment_score = analysis_results.get('sentiment_score', 0.5)
            sentiment_direction = 'long' if sentiment_score > 0.6 else 'short' if sentiment_score < 0.4 else 'neutral'
            head_b = ModelHeadResult(
                head_type=ModelHead.HEAD_B,
                direction=SignalDirection.LONG if sentiment_direction == 'long' else 
                         SignalDirection.SHORT if sentiment_direction == 'short' else SignalDirection.FLAT,
                probability=abs(sentiment_score - 0.5) * 2,  # Convert to 0-1 range
                confidence=abs(sentiment_score - 0.5) * 2,
                features_used=['sentiment', 'news'],
                reasoning='Sentiment analysis based on news and social media'
            )
            
            # Head C - Volume Analysis
            volume_confidence = analysis_results.get('volume_confidence', 0.5)
            volume_direction = analysis_results.get('volume_direction', 'neutral')
            head_c = ModelHeadResult(
                head_type=ModelHead.HEAD_C,
                direction=SignalDirection.LONG if volume_direction == 'long' else 
                         SignalDirection.SHORT if volume_direction == 'short' else SignalDirection.FLAT,
                probability=volume_confidence,
                confidence=volume_confidence,
                features_used=['volume', 'orderbook'],
                reasoning='Volume and orderbook analysis'
            )
            
            # Head D - Market Regime Analysis
            regime_confidence = analysis_results.get('market_regime_confidence', 0.5)
            regime_direction = analysis_results.get('market_regime_direction', 'neutral')
            head_d = ModelHeadResult(
                head_type=ModelHead.HEAD_D,
                direction=SignalDirection.LONG if regime_direction == 'long' else 
                         SignalDirection.SHORT if regime_direction == 'short' else SignalDirection.FLAT,
                probability=regime_confidence,
                confidence=regime_confidence,
                features_used=['market_regime', 'volatility'],
                reasoning='Market regime and volatility analysis'
            )
            
            return [head_a, head_b, head_c, head_d]
            
        except Exception as e:
            logger.error(f"❌ Failed to create model head results: {e}")
            # Return default results
            default_head = ModelHeadResult(
                head_type=ModelHead.HEAD_A,
                direction=SignalDirection.FLAT,
                probability=0.5,
                confidence=0.5,
                features_used=['default'],
                reasoning='Default analysis'
            )
            return [default_head, default_head, default_head, default_head]
    
    def _apply_integration_gates(self, 
                                consensus_result: ConsensusResult,
                                confluence_result: ConfluenceScore,
                                execution_result: ExecutionQuality,
                                news_blackout_result: Any,
                                signal_limits_result: Any,
                                divergence_analysis: Optional[DivergenceAnalysis],
                                config: Dict[str, Any]) -> bool:
        """Apply integration gates"""
        try:
            # Consensus gate
            if not consensus_result.achieved:
                return False
            
            # Confluence gate
            min_confluence = config.get('confluence_threshold', 8.0)
            if confluence_result.total_score < min_confluence:
                return False
            
            # Execution quality gate
            min_execution = config.get('execution_quality_threshold', 7.0)
            if execution_result.execution_score < min_execution:
                return False
            
            # Divergence analysis gate (optional boost)
            if divergence_analysis and divergence_analysis.overall_confidence > 0.7:
                # Divergence provides confidence boost, but doesn't block
                pass
            
            # News blackout gate
            if news_blackout_result and hasattr(news_blackout_result, 'blackout_active'):
                if news_blackout_result.blackout_active:
                    return False
            
            # Signal limits gate
            if signal_limits_result and hasattr(signal_limits_result, 'limit_exceeded'):
                if signal_limits_result.limit_exceeded:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply integration gates: {e}")
            return False
    
    def _calculate_final_confidence(self,
                                   consensus_result: ConsensusResult,
                                   confluence_result: ConfluenceScore,
                                   execution_result: ExecutionQuality,
                                   calibrated_confidence: float,
                                   divergence_analysis: Optional[DivergenceAnalysis],
                                   config: Dict[str, Any]) -> float:
        """Calculate final confidence score"""
        try:
            # Base confidence from consensus
            base_confidence = consensus_result.probability if consensus_result.achieved else 0.0
            
            # Apply confluence multiplier
            confluence_multiplier = min(confluence_result.total_score / 10.0, 1.2)
            
            # Apply execution quality multiplier
            execution_multiplier = min(execution_result.execution_score / 10.0, 1.1)
            
            # Apply divergence analysis multiplier (if available)
            divergence_multiplier = 1.0
            if divergence_analysis and divergence_analysis.overall_confidence > 0.5:
                divergence_multiplier = 1.0 + (divergence_analysis.overall_confidence * 0.2)  # Up to 20% boost
            
            # Use calibrated confidence if available
            if calibrated_confidence > 0:
                final_confidence = calibrated_confidence * confluence_multiplier * execution_multiplier * divergence_multiplier
            else:
                final_confidence = base_confidence * confluence_multiplier * execution_multiplier * divergence_multiplier
            
            # Apply confidence threshold
            min_confidence = config.get('confidence_threshold', 0.85)
            if final_confidence < min_confidence:
                final_confidence = 0.0
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate final confidence: {e}")
            return 0.0
    
    async def _store_integration_results(self, signal_id: str, symbol: str, timeframe: str,
                                       consensus_result: ConsensusResult, confluence_result: ConfluenceScore,
                                       execution_result: ExecutionQuality, news_blackout_result: Any,
                                       signal_limits_result: Any, tp_structure_result: Any,
                                       calibrated_confidence: float, final_confidence: float,
                                       divergence_analysis: Optional[DivergenceAnalysis]):
        """Store integration results in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store consensus tracking
                await conn.execute("""
                    INSERT INTO sde_model_consensus_tracking (
                        signal_id, symbol, timeframe, consensus_achieved, consensus_direction,
                        consensus_probability, agreeing_heads_count, min_agreeing_heads, min_head_probability
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, signal_id, symbol, timeframe, consensus_result.achieved,
                     consensus_result.direction.value if consensus_result.direction else None,
                     consensus_result.probability, consensus_result.agreeing_heads_count, 3, 0.70)
                
                # Store signal validation
                await conn.execute("""
                    INSERT INTO sde_signal_validation (
                        signal_id, symbol, timeframe, consensus_passed, confluence_passed,
                        execution_passed, news_blackout_passed, signal_limits_passed,
                        consensus_score, confluence_score, execution_quality_score,
                        final_confidence, confidence_threshold, confidence_passed
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, signal_id, symbol, timeframe, consensus_result.consensus_achieved, confluence_result.total_score >= 8.0,
                     execution_result.execution_score >= 7.0, 
                     not (news_blackout_result and hasattr(news_blackout_result, 'blackout_active') and news_blackout_result.blackout_active),
                     not (signal_limits_result and hasattr(signal_limits_result, 'limit_exceeded') and signal_limits_result.limit_exceeded),
                     consensus_result.consensus_score, confluence_result.total_score, execution_result.execution_score,
                     final_confidence, 0.85, final_confidence >= 0.85)
                
                logger.info(f"✅ Stored SDE integration results for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Failed to store integration results: {e}")
    
    def _generate_integration_reason(self, consensus_result: ConsensusResult,
                                   confluence_result: ConfluenceScore,
                                   execution_result: ExecutionQuality,
                                   news_blackout_result: Any,
                                   signal_limits_result: Any,
                                   divergence_analysis: Optional[DivergenceAnalysis]) -> str:
        """Generate human-readable integration reason"""
        reasons = []
        
        if consensus_result.achieved:
            reasons.append(f"Consensus achieved ({consensus_result.agreeing_heads_count}/4 heads)")
        else:
            reasons.append("No consensus achieved")
        
        if confluence_result.total_score >= 8.0:
            reasons.append(f"Strong confluence score: {confluence_result.total_score:.1f}/10")
        else:
            reasons.append(f"Low confluence: {confluence_result.total_score:.1f}/10")
        
        if execution_result.execution_score >= 7.0:
            reasons.append(f"Good execution quality: {execution_result.execution_score:.1f}/10")
        else:
            reasons.append(f"Poor execution: {execution_result.execution_score:.1f}/10")
        
        if divergence_analysis and divergence_analysis.overall_confidence > 0.5:
            reasons.append(f"Divergence confidence: {divergence_analysis.overall_confidence:.2f}")
        
        if news_blackout_result and hasattr(news_blackout_result, 'blackout_active') and news_blackout_result.blackout_active:
            reasons.append("News blackout active")
        
        if signal_limits_result and hasattr(signal_limits_result, 'limit_exceeded') and signal_limits_result.limit_exceeded:
            reasons.append("Signal limits exceeded")
        
        return " | ".join(reasons)
    
    def _update_performance_stats(self, processing_time_ms: float, success: bool):
        """Update performance statistics"""
        self.performance_stats['total_integrations'] += 1
        
        if success:
            self.performance_stats['successful_integrations'] += 1
        else:
            self.performance_stats['failed_integrations'] += 1
        
        # Update average processing time
        total_time = self.performance_stats['avg_processing_time_ms'] * (self.performance_stats['total_integrations'] - 1)
        self.performance_stats['avg_processing_time_ms'] = (total_time + processing_time_ms) / self.performance_stats['total_integrations']
    
    async def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics"""
        return {
            'performance_stats': self.performance_stats,
            'success_rate': (
                self.performance_stats['successful_integrations'] / 
                max(self.performance_stats['total_integrations'], 1)
            ),
            'config_loaded': len(self.config) > 0
        }
