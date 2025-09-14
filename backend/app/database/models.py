"""
Database Models for AlphaPlus
Defines database models and schemas
"""

from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class Trade(BaseModel):
    """Trade model"""
    id: Optional[int] = None
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    status: str = 'pending'  # 'pending', 'executed', 'cancelled'
    
    class Config:
        from_attributes = True

class Strategy(BaseModel):
    """Strategy model"""
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True

class MarketData(BaseModel):
    """Market data model"""
    id: Optional[int] = None
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = '1h'
    
    class Config:
        from_attributes = True

class Signal(BaseModel):
    """Trading signal model"""
    id: Optional[int] = None
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    strength: float
    timestamp: datetime = Field(default_factory=datetime.now)
    price: Optional[float] = None
    reason: Optional[str] = None
    indicators: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True

class Pattern(BaseModel):
    """Pattern detection model"""
    id: Optional[int] = None
    symbol: str
    pattern_type: str
    confidence: float
    strength: float
    timestamp: datetime = Field(default_factory=datetime.now)
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True

class SentimentAnalysis(BaseModel):
    """Sentiment analysis model"""
    id: Optional[int] = None
    symbol: str
    sentiment_score: float
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)
    source: Optional[str] = None
    text: Optional[str] = None
    
    class Config:
        from_attributes = True

class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    id: Optional[int] = None
    strategy_id: Optional[int] = None
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True

class SystemHealth(BaseModel):
    """System health metrics model"""
    id: Optional[int] = None
    component: str
    status: str  # 'healthy', 'warning', 'error'
    message: Optional[str] = None
    metrics: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True

class WebSocketConnection(BaseModel):
    """WebSocket connection model"""
    id: Optional[int] = None
    client_id: str
    status: str  # 'connected', 'disconnected', 'error'
    connected_at: datetime = Field(default_factory=datetime.now)
    disconnected_at: Optional[datetime] = None
    messages_sent: int = 0
    messages_received: int = 0
    
    class Config:
        from_attributes = True

class WebSocketPerformance(BaseModel):
    """WebSocket performance metrics model"""
    id: Optional[int] = None
    client_id: str
    latency_ms: float
    throughput_mbps: float
    error_rate: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True
