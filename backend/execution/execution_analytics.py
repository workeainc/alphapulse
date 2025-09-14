#!/usr/bin/env python3
"""
Execution Analytics Manager for AlphaPulse
Tracks execution performance, quality metrics, and provides optimization insights
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import json

logger = logging.getLogger(__name__)

class ExecutionMetric(Enum):
    """Execution performance metrics"""
    FILL_RATE = "fill_rate"
    SLIPPAGE = "slippage"
    EXECUTION_SPEED = "execution_speed"
    SUCCESS_RATE = "success_rate"
    COST_IMPACT = "cost_impact"
    TIMING_QUALITY = "timing_quality"

class ExecutionQuality(Enum):
    """Execution quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"

@dataclass
class ExecutionRecord:
    """Individual execution record"""
    execution_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    target_price: float
    executed_price: float
    execution_time: datetime
    signal_time: datetime
    exchange: str
    slippage: float
    commission: float
    market_impact: float
    execution_quality: ExecutionQuality
    cost_impact: float = None  # Total cost impact (commission + market impact)
    notes: Optional[str] = None

@dataclass
class ExecutionSummary:
    """Execution performance summary"""
    symbol: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_slippage: float
    avg_execution_speed_ms: float
    fill_rate: float
    avg_cost_impact: float
    execution_quality_score: float
    period_start: datetime
    period_end: datetime

@dataclass
class OptimizationRecommendation:
    """Execution optimization recommendation"""
    category: str
    description: str
    impact: str  # "high", "medium", "low"
    priority: int  # 1-5, where 1 is highest
    suggested_action: str
    expected_improvement: str
    confidence: float  # 0.0-1.0

class ExecutionAnalyticsManager:
    """Manages execution analytics and performance tracking"""
    
    def __init__(self):
        # Execution records storage
        self.execution_records: List[ExecutionRecord] = []
        
        # Performance thresholds
        self.quality_thresholds = {
            ExecutionQuality.EXCELLENT: {
                "slippage": 0.001,  # 0.1%
                "execution_speed": 100,  # 100ms
                "fill_rate": 0.95,  # 95%
                "cost_impact": 0.002  # 0.2%
            },
            ExecutionQuality.GOOD: {
                "slippage": 0.003,  # 0.3%
                "execution_speed": 250,  # 250ms
                "fill_rate": 0.90,  # 90%
                "cost_impact": 0.005  # 0.5%
            },
            ExecutionQuality.AVERAGE: {
                "slippage": 0.005,  # 0.5%
                "execution_speed": 500,  # 500ms
                "fill_rate": 0.85,  # 85%
                "cost_impact": 0.008  # 0.8%
            },
            ExecutionQuality.POOR: {
                "slippage": 0.010,  # 1.0%
                "execution_speed": 1000,  # 1s
                "fill_rate": 0.80,  # 80%
                "cost_impact": 0.015  # 1.5%
            }
        }
        
        # Analytics configuration
        self.analysis_periods = [1, 7, 30, 90]  # days
        self.max_records = 10000
        
        logger.info("Execution Analytics Manager initialized")
    
    def record_execution(self, execution_record: ExecutionRecord):
        """Record a new execution"""
        self.execution_records.append(execution_record)
        
        # Maintain record limit
        if len(self.execution_records) > self.max_records:
            self.execution_records = self.execution_records[-self.max_records:]
        
        logger.debug(f"Recorded execution: {execution_record.execution_id}")
    
    def calculate_execution_quality(self, slippage: float, execution_speed: float,
                                  fill_rate: float, cost_impact: float) -> ExecutionQuality:
        """Calculate execution quality based on metrics"""
        
        # Score each metric
        slippage_score = self._score_slippage(slippage)
        speed_score = self._score_execution_speed(execution_speed)
        fill_score = self._score_fill_rate(fill_rate)
        cost_score = self._score_cost_impact(cost_impact)
        
        # Calculate overall score (weighted average)
        overall_score = (
            slippage_score * 0.3 +
            speed_score * 0.25 +
            fill_score * 0.25 +
            cost_score * 0.2
        )
        
        # Determine quality level
        if overall_score >= 0.9:
            return ExecutionQuality.EXCELLENT
        elif overall_score >= 0.8:
            return ExecutionQuality.GOOD
        elif overall_score >= 0.7:
            return ExecutionQuality.AVERAGE
        elif overall_score >= 0.6:
            return ExecutionQuality.POOR
        else:
            return ExecutionQuality.UNACCEPTABLE
    
    def _score_slippage(self, slippage: float) -> float:
        """Score slippage performance (0.0-1.0)"""
        if slippage <= 0.001:  # 0.1%
            return 1.0
        elif slippage <= 0.003:  # 0.3%
            return 0.9
        elif slippage <= 0.005:  # 0.5%
            return 0.8
        elif slippage <= 0.010:  # 1.0%
            return 0.7
        elif slippage <= 0.020:  # 2.0%
            return 0.5
        else:
            return 0.3
    
    def _score_execution_speed(self, speed_ms: float) -> float:
        """Score execution speed performance (0.0-1.0)"""
        if speed_ms <= 100:  # 100ms
            return 1.0
        elif speed_ms <= 250:  # 250ms
            return 0.9
        elif speed_ms <= 500:  # 500ms
            return 0.8
        elif speed_ms <= 1000:  # 1s
            return 0.7
        elif speed_ms <= 2000:  # 2s
            return 0.5
        else:
            return 0.3
    
    def _score_fill_rate(self, fill_rate: float) -> float:
        """Score fill rate performance (0.0-1.0)"""
        if fill_rate >= 0.95:  # 95%
            return 1.0
        elif fill_rate >= 0.90:  # 90%
            return 0.9
        elif fill_rate >= 0.85:  # 85%
            return 0.8
        elif fill_rate >= 0.80:  # 80%
            return 0.7
        elif fill_rate >= 0.70:  # 70%
            return 0.5
        else:
            return 0.3
    
    def _score_cost_impact(self, cost_impact: float) -> float:
        """Score cost impact performance (0.0-1.0)"""
        if cost_impact <= 0.002:  # 0.2%
            return 1.0
        elif cost_impact <= 0.005:  # 0.5%
            return 0.9
        elif cost_impact <= 0.008:  # 0.8%
            return 0.8
        elif cost_impact <= 0.015:  # 1.5%
            return 0.7
        elif cost_impact <= 0.025:  # 2.5%
            return 0.5
        else:
            return 0.3
    
    def get_execution_summary(self, symbol: Optional[str] = None,
                            period_days: int = 30) -> ExecutionSummary:
        """Get execution performance summary"""
        
        # Filter records by period and symbol
        cutoff_date = datetime.now() - timedelta(days=period_days)
        filtered_records = [
            record for record in self.execution_records
            if record.execution_time >= cutoff_date and
            (symbol is None or record.symbol == symbol)
        ]
        
        if not filtered_records:
            return ExecutionSummary(
                symbol=symbol or "ALL",
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                avg_slippage=0.0,
                avg_execution_speed_ms=0.0,
                fill_rate=0.0,
                avg_cost_impact=0.0,
                execution_quality_score=0.0,
                period_start=cutoff_date,
                period_end=datetime.now()
            )
        
        # Calculate metrics
        total_executions = len(filtered_records)
        successful_executions = len([r for r in filtered_records if r.execution_quality != ExecutionQuality.UNACCEPTABLE])
        
        avg_slippage = np.mean([r.slippage for r in filtered_records])
        avg_execution_speed = np.mean([self._calculate_execution_speed(r) for r in filtered_records])
        fill_rate = successful_executions / total_executions
        avg_cost_impact = np.mean([r.cost_impact for r in filtered_records])
        
        # Calculate quality score
        quality_scores = [self._quality_to_score(r.execution_quality) for r in filtered_records]
        execution_quality_score = np.mean(quality_scores)
        
        return ExecutionSummary(
            symbol=symbol or "ALL",
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=total_executions - successful_executions,
            avg_slippage=avg_slippage,
            avg_execution_speed_ms=avg_execution_speed,
            fill_rate=fill_rate,
            avg_cost_impact=avg_cost_impact,
            execution_quality_score=execution_quality_score,
            period_start=cutoff_date,
            period_end=datetime.now()
        )
    
    def _calculate_execution_speed(self, record: ExecutionRecord) -> float:
        """Calculate execution speed in milliseconds"""
        time_diff = record.execution_time - record.signal_time
        return time_diff.total_seconds() * 1000
    
    def _quality_to_score(self, quality: ExecutionQuality) -> float:
        """Convert quality enum to numeric score"""
        quality_scores = {
            ExecutionQuality.EXCELLENT: 1.0,
            ExecutionQuality.GOOD: 0.8,
            ExecutionQuality.AVERAGE: 0.6,
            ExecutionQuality.POOR: 0.4,
            ExecutionQuality.UNACCEPTABLE: 0.0
        }
        return quality_scores.get(quality, 0.0)
    
    def analyze_execution_patterns(self, symbol: Optional[str] = None,
                                 period_days: int = 30) -> Dict:
        """Analyze execution patterns and identify trends"""
        
        summary = self.get_execution_summary(symbol, period_days)
        
        if summary.total_executions == 0:
            return {"error": "No execution data available"}
        
        # Filter records for analysis
        cutoff_date = datetime.now() - timedelta(days=period_days)
        filtered_records = [
            record for record in self.execution_records
            if record.execution_time >= cutoff_date and
            (symbol is None or record.symbol == symbol)
        ]
        
        # Time-based analysis
        hourly_performance = self._analyze_hourly_performance(filtered_records)
        daily_performance = self._analyze_daily_performance(filtered_records)
        
        # Order type analysis
        order_type_performance = self._analyze_order_type_performance(filtered_records)
        
        # Side analysis (buy vs sell)
        side_performance = self._analyze_side_performance(filtered_records)
        
        # Exchange analysis
        exchange_performance = self._analyze_exchange_performance(filtered_records)
        
        return {
            "summary": summary,
            "hourly_performance": hourly_performance,
            "daily_performance": daily_performance,
            "order_type_performance": order_type_performance,
            "side_performance": side_performance,
            "exchange_performance": exchange_performance
        }
    
    def _analyze_hourly_performance(self, records: List[ExecutionRecord]) -> Dict:
        """Analyze performance by hour of day"""
        hourly_data = {}
        
        for record in records:
            hour = record.execution_time.hour
            if hour not in hourly_data:
                hourly_data[hour] = {
                    "count": 0,
                    "slippage": [],
                    "speed": [],
                    "quality_scores": []
                }
            
            hourly_data[hour]["count"] += 1
            hourly_data[hour]["slippage"].append(record.slippage)
            hourly_data[hour]["speed"].append(self._calculate_execution_speed(record))
            hourly_data[hour]["quality_scores"].append(self._quality_to_score(record.execution_quality))
        
        # Calculate averages
        for hour in hourly_data:
            data = hourly_data[hour]
            data["avg_slippage"] = np.mean(data["slippage"])
            data["avg_speed"] = np.mean(data["speed"])
            data["avg_quality"] = np.mean(data["quality_scores"])
            
            # Remove raw data
            del data["slippage"]
            del data["speed"]
            del data["quality_scores"]
        
        return hourly_data
    
    def _analyze_daily_performance(self, records: List[ExecutionRecord]) -> Dict:
        """Analyze performance by day of week"""
        daily_data = {}
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for day in days:
            daily_data[day] = {
                "count": 0,
                "avg_slippage": 0.0,
                "avg_speed": 0.0,
                "avg_quality": 0.0
            }
        
        for record in records:
            day = record.execution_time.strftime("%A")
            if day in daily_data:
                daily_data[day]["count"] += 1
        
        return daily_data
    
    def _analyze_order_type_performance(self, records: List[ExecutionRecord]) -> Dict:
        """Analyze performance by order type"""
        order_type_data = {}
        
        for record in records:
            order_type = record.order_type
            if order_type not in order_type_data:
                order_type_data[order_type] = {
                    "count": 0,
                    "slippage": [],
                    "speed": [],
                    "quality_scores": []
                }
            
            order_type_data[order_type]["count"] += 1
            order_type_data[order_type]["slippage"].append(record.slippage)
            order_type_data[order_type]["speed"].append(self._calculate_execution_speed(record))
            order_type_data[order_type]["quality_scores"].append(self._quality_to_score(record.execution_quality))
        
        # Calculate averages
        for order_type in order_type_data:
            data = order_type_data[order_type]
            data["avg_slippage"] = np.mean(data["slippage"])
            data["avg_speed"] = np.mean(data["speed"])
            data["avg_quality"] = np.mean(data["quality_scores"])
            
            # Remove raw data
            del data["slippage"]
            del data["speed"]
            del data["quality_scores"]
        
        return order_type_data
    
    def _analyze_side_performance(self, records: List[ExecutionRecord]) -> Dict:
        """Analyze performance by trade side (buy/sell)"""
        side_data = {"buy": [], "sell": []}
        
        for record in records:
            side = record.side.lower()
            if side in side_data:
                side_data[side].append(record)
        
        result = {}
        for side, side_records in side_data.items():
            if side_records:
                result[side] = {
                    "count": len(side_records),
                    "avg_slippage": np.mean([r.slippage for r in side_records]),
                    "avg_speed": np.mean([self._calculate_execution_speed(r) for r in side_records]),
                    "avg_quality": np.mean([self._quality_to_score(r.execution_quality) for r in side_records])
                }
            else:
                result[side] = {"count": 0, "avg_slippage": 0.0, "avg_speed": 0.0, "avg_quality": 0.0}
        
        return result
    
    def _analyze_exchange_performance(self, records: List[ExecutionRecord]) -> Dict:
        """Analyze performance by exchange"""
        exchange_data = {}
        
        for record in records:
            exchange = record.exchange
            if exchange not in exchange_data:
                exchange_data[exchange] = {
                    "count": 0,
                    "slippage": [],
                    "speed": [],
                    "quality_scores": []
                }
            
            exchange_data[exchange]["count"] += 1
            exchange_data[exchange]["slippage"].append(record.slippage)
            exchange_data[exchange]["speed"].append(self._calculate_execution_speed(record))
            exchange_data[exchange]["quality_scores"].append(self._quality_to_score(record.execution_quality))
        
        # Calculate averages
        for exchange in exchange_data:
            data = exchange_data[exchange]
            data["avg_slippage"] = np.mean(data["slippage"])
            data["avg_speed"] = np.mean(data["speed"])
            data["avg_quality"] = np.mean(data["quality_scores"])
            
            # Remove raw data
            del data["slippage"]
            del data["speed"]
            del data["quality_scores"]
        
        return exchange_data
    
    def generate_optimization_recommendations(self, symbol: Optional[str] = None,
                                           period_days: int = 30) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on execution analysis"""
        
        analysis = self.analyze_execution_patterns(symbol, period_days)
        recommendations = []
        
        if "error" in analysis:
            return recommendations
        
        summary = analysis["summary"]
        
        # Slippage recommendations
        if summary.avg_slippage > 0.005:  # > 0.5%
            recommendations.append(OptimizationRecommendation(
                category="Slippage",
                description=f"High average slippage of {summary.avg_slippage:.3%}",
                impact="high",
                priority=1,
                suggested_action="Consider using limit orders or implementing smart order routing",
                expected_improvement="Reduce slippage by 30-50%",
                confidence=0.8
            ))
        
        # Execution speed recommendations
        if summary.avg_execution_speed_ms > 500:  # > 500ms
            recommendations.append(OptimizationRecommendation(
                category="Execution Speed",
                description=f"Slow execution speed of {summary.avg_execution_speed_ms:.0f}ms",
                impact="medium",
                priority=2,
                suggested_action="Optimize order routing and reduce network latency",
                expected_improvement="Improve speed by 40-60%",
                confidence=0.7
            ))
        
        # Fill rate recommendations
        if summary.fill_rate < 0.90:  # < 90%
            recommendations.append(OptimizationRecommendation(
                category="Fill Rate",
                description=f"Low fill rate of {summary.fill_rate:.1%}",
                impact="high",
                priority=1,
                suggested_action="Review order placement timing and market conditions",
                expected_improvement="Improve fill rate to 90%+",
                confidence=0.8
            ))
        
        # Quality score recommendations
        if summary.execution_quality_score < 0.7:
            recommendations.append(OptimizationRecommendation(
                category="Overall Quality",
                description=f"Low execution quality score of {summary.execution_quality_score:.2f}",
                impact="high",
                priority=1,
                suggested_action="Comprehensive review of execution strategy and infrastructure",
                expected_improvement="Improve quality score to 0.8+",
                confidence=0.9
            ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority)
        
        return recommendations
    
    def export_analytics_data(self, symbol: Optional[str] = None,
                            period_days: int = 30,
                            format: str = "json") -> str:
        """Export analytics data in specified format"""
        
        analysis = self.analyze_execution_patterns(symbol, period_days)
        
        if format.lower() == "json":
            return json.dumps(analysis, default=str, indent=2)
        elif format.lower() == "csv":
            # Convert to CSV format
            return self._convert_to_csv(analysis)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _convert_to_csv(self, analysis: Dict) -> str:
        """Convert analysis data to CSV format"""
        # This is a simplified CSV conversion
        # In a real implementation, you'd want more sophisticated CSV handling
        csv_lines = []
        
        # Summary
        summary = analysis["summary"]
        csv_lines.append("Category,Metric,Value")
        csv_lines.append(f"Summary,Total Executions,{summary.total_executions}")
        csv_lines.append(f"Summary,Success Rate,{summary.fill_rate:.3f}")
        csv_lines.append(f"Summary,Avg Slippage,{summary.avg_slippage:.6f}")
        csv_lines.append(f"Summary,Avg Speed (ms),{summary.avg_execution_speed_ms:.0f}")
        
        return "\n".join(csv_lines)

def test_execution_analytics():
    """Test the execution analytics manager"""
    manager = ExecutionAnalyticsManager()
    
    # Create sample execution records
    base_time = datetime.now()
    
    for i in range(10):
        record = ExecutionRecord(
            execution_id=f"exec_{i}",
            symbol="BTCUSDT",
            side="buy" if i % 2 == 0 else "sell",
            order_type="market" if i % 3 == 0 else "limit",
            quantity=100.0,
            target_price=50000.0,
            executed_price=50000.0 + (i * 10),
            execution_time=base_time + timedelta(hours=i),
            signal_time=base_time + timedelta(hours=i, minutes=-5),
            exchange="binance",
            slippage=0.001 + (i * 0.0005),
            commission=0.001,
            market_impact=0.0005,
            execution_quality=ExecutionQuality.GOOD
        )
        manager.record_execution(record)
    
    # Get summary
    summary = manager.get_execution_summary("BTCUSDT", 1)
    print(f"Execution Summary: {summary}")
    
    # Analyze patterns
    patterns = manager.analyze_execution_patterns("BTCUSDT", 1)
    print(f"Pattern Analysis: {patterns}")
    
    # Get recommendations
    recommendations = manager.generate_optimization_recommendations("BTCUSDT", 1)
    print(f"Optimization Recommendations: {len(recommendations)}")
    
    for rec in recommendations:
        print(f"- {rec.category}: {rec.description}")

if __name__ == "__main__":
    test_execution_analytics()
