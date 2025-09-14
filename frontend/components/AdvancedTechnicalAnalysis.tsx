import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Activity, 
  Target, 
  Zap,
  Gauge,
  Volume2,
  ArrowUpRight,
  ArrowDownRight,
  Clock,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Layers,
  Settings,
  Maximize2,
  Minimize2
} from 'lucide-react';

interface TechnicalIndicator {
  name: string;
  value: number;
  signal: 'buy' | 'sell' | 'neutral';
  color: string;
  description: string;
  strength: number;
}

interface MarketData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  marketCap: number;
  high24h: number;
  low24h: number;
}

interface AdvancedTechnicalAnalysisProps {
  symbol?: string;
  timeframe?: string;
  className?: string;
}

export default function AdvancedTechnicalAnalysis({ 
  symbol = 'BTC/USDT', 
  timeframe = '1H',
  className = ''
}: AdvancedTechnicalAnalysisProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedTimeframe, setSelectedTimeframe] = useState(timeframe);
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(true);

  // Mock market data
  const marketData: MarketData = useMemo(() => ({
    symbol: 'BTC/USDT',
    price: 46000,
    change24h: 1250,
    changePercent24h: 2.8,
    volume24h: 28500000000,
    marketCap: 900000000000,
    high24h: 46500,
    low24h: 44500
  }), []);

  // Advanced technical indicators
  const technicalIndicators: TechnicalIndicator[] = useMemo(() => [
    {
      name: 'RSI (14)',
      value: 72.5,
      signal: 'sell',
      color: '#EF4444',
      description: 'Overbought condition - potential reversal',
      strength: 0.85
    },
    {
      name: 'MACD',
      value: 145.3,
      signal: 'buy',
      color: '#10B981',
      description: 'Bullish crossover - momentum building',
      strength: 0.78
    },
    {
      name: 'Bollinger Bands',
      value: 46000,
      signal: 'neutral',
      color: '#6B7280',
      description: 'Price within bands - consolidation',
      strength: 0.45
    },
    {
      name: 'Volume',
      value: 28500000000,
      signal: 'buy',
      color: '#10B981',
      description: 'High volume - strong confirmation',
      strength: 0.92
    },
    {
      name: 'ATR',
      value: 480,
      signal: 'neutral',
      color: '#6B7280',
      description: 'Normal volatility - stable conditions',
      strength: 0.55
    },
    {
      name: 'OBV',
      value: 1328000000,
      signal: 'buy',
      color: '#10B981',
      description: 'Accumulation pattern - bullish',
      strength: 0.88
    }
  ], []);

  // Market sentiment analysis
  const sentimentAnalysis = useMemo(() => ({
    overall: 'bullish',
    score: 0.75,
    confidence: 0.82,
    factors: [
      { name: 'Technical Analysis', score: 0.85, weight: 0.4 },
      { name: 'Volume Analysis', score: 0.92, weight: 0.25 },
      { name: 'Market Structure', score: 0.78, weight: 0.2 },
      { name: 'Sentiment Data', score: 0.65, weight: 0.15 }
    ]
  }), []);

  // Support and resistance levels
  const supportResistance = useMemo(() => ({
    support: [
      { level: 45500, strength: 0.85, type: 'strong' },
      { level: 44800, strength: 0.72, type: 'medium' },
      { level: 44000, strength: 0.65, type: 'weak' }
    ],
    resistance: [
      { level: 46500, strength: 0.78, type: 'medium' },
      { level: 47000, strength: 0.85, type: 'strong' },
      { level: 48000, strength: 0.92, type: 'very_strong' }
    ]
  }), []);

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'buy':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'sell':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      default:
        return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStrengthColor = (strength: number) => {
    if (strength >= 0.8) return 'text-green-600';
    if (strength >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className={`space-y-6 ${className} ${isFullscreen ? 'fixed inset-0 z-50 bg-white p-6 overflow-auto' : ''}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h2 className="text-xl font-bold text-gray-900">Advanced Technical Analysis</h2>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">{symbol}</span>
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="px-2 py-1 text-sm border border-gray-300 rounded-md bg-white"
            >
              <option value="1m">1m</option>
              <option value="5m">5m</option>
              <option value="15m">15m</option>
              <option value="1H">1H</option>
              <option value="4H">4H</option>
              <option value="1D">1D</option>
            </select>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowAdvancedMetrics(!showAdvancedMetrics)}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md"
          >
            <Layers className="h-4 w-4" />
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md"
          >
            {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Market Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <DollarSign className="h-5 w-5 text-green-600" />
            <span>Market Overview</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-gray-900">${marketData.price.toLocaleString()}</p>
              <p className="text-sm text-gray-600">Current Price</p>
            </div>
            <div className="text-center">
              <p className={`text-lg font-semibold ${marketData.changePercent24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {marketData.changePercent24h >= 0 ? '+' : ''}{marketData.changePercent24h.toFixed(2)}%
              </p>
              <p className="text-sm text-gray-600">24h Change</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-semibold text-gray-900">${(marketData.volume24h / 1000000000).toFixed(1)}B</p>
              <p className="text-sm text-gray-600">24h Volume</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-semibold text-gray-900">${(marketData.marketCap / 1000000000).toFixed(1)}B</p>
              <p className="text-sm text-gray-600">Market Cap</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technical Indicators */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5 text-blue-600" />
            <span>Technical Indicators</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {technicalIndicators.map((indicator, index) => (
              <div key={index} className="p-4 border border-gray-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{indicator.name}</span>
                  {getSignalIcon(indicator.signal)}
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-lg font-bold" style={{ color: indicator.color }}>
                    {indicator.name === 'Volume' 
                      ? `$${(indicator.value / 1000000000).toFixed(1)}B`
                      : indicator.name === 'ATR' || indicator.name === 'OBV'
                      ? indicator.value.toLocaleString()
                      : indicator.value.toFixed(1)
                    }
                  </span>
                  <span className={`text-sm font-medium ${getStrengthColor(indicator.strength)}`}>
                    {(indicator.strength * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-sm text-gray-600">{indicator.description}</p>
                <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="h-2 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${indicator.strength * 100}%`,
                      backgroundColor: indicator.color
                    }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Sentiment Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Gauge className="h-5 w-5 text-purple-600" />
            <span>Market Sentiment Analysis</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex items-center justify-between mb-4">
                <span className="text-lg font-semibold text-gray-900">Overall Sentiment</span>
                <div className="flex items-center space-x-2">
                  <span className={`text-lg font-bold ${
                    sentimentAnalysis.overall === 'bullish' ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {sentimentAnalysis.overall.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-600">
                    {(sentimentAnalysis.score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                <div 
                  className="h-3 rounded-full transition-all duration-300"
                  style={{ 
                    width: `${sentimentAnalysis.score * 100}%`,
                    backgroundColor: sentimentAnalysis.overall === 'bullish' ? '#10B981' : '#EF4444'
                  }}
                ></div>
              </div>
              <p className="text-sm text-gray-600">
                Confidence: {(sentimentAnalysis.confidence * 100).toFixed(0)}%
              </p>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-3">Sentiment Factors</h4>
              <div className="space-y-3">
                {sentimentAnalysis.factors.map((factor, index) => (
                  <div key={index}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm text-gray-700">{factor.name}</span>
                      <span className="text-sm font-medium text-gray-900">
                        {(factor.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="h-2 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${factor.score * 100}%`,
                          backgroundColor: factor.score >= 0.7 ? '#10B981' : factor.score >= 0.5 ? '#F59E0B' : '#EF4444'
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Support & Resistance Levels */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5 text-orange-600" />
            <span>Support & Resistance Levels</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-red-600 mb-3">Resistance Levels</h4>
              <div className="space-y-2">
                {supportResistance.resistance.map((level, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-red-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-red-700">${level.level.toLocaleString()}</span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        level.type === 'very_strong' ? 'bg-red-200 text-red-800' :
                        level.type === 'strong' ? 'bg-red-100 text-red-700' :
                        'bg-red-50 text-red-600'
                      }`}>
                        {level.type.replace('_', ' ')}
                      </span>
                    </div>
                    <span className="text-sm text-red-600">{(level.strength * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-medium text-green-600 mb-3">Support Levels</h4>
              <div className="space-y-2">
                {supportResistance.support.map((level, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-green-50 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-green-700">${level.level.toLocaleString()}</span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        level.type === 'strong' ? 'bg-green-200 text-green-800' :
                        level.type === 'medium' ? 'bg-green-100 text-green-700' :
                        'bg-green-50 text-green-600'
                      }`}>
                        {level.type}
                      </span>
                    </div>
                    <span className="text-sm text-green-600">{(level.strength * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trading Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="h-5 w-5 text-yellow-600" />
            <span>Trading Recommendations</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <span className="font-medium text-green-800">Strong Buy Signal</span>
              </div>
              <p className="text-sm text-green-700 mb-2">
                Multiple technical indicators align for a bullish move. RSI shows momentum, 
                MACD confirms trend, and volume supports the move.
              </p>
              <div className="flex items-center space-x-4 text-sm text-green-600">
                <span>Confidence: 85%</span>
                <span>Target: $47,500</span>
                <span>Stop Loss: $44,800</span>
              </div>
            </div>
            
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <div className="flex items-center space-x-2 mb-2">
                <AlertTriangle className="h-5 w-5 text-yellow-600" />
                <span className="font-medium text-yellow-800">Caution: Overbought RSI</span>
              </div>
              <p className="text-sm text-yellow-700">
                RSI above 70 indicates overbought conditions. Consider taking profits 
                or waiting for a pullback before new positions.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
