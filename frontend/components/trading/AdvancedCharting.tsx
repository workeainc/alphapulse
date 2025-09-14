/**
 * Advanced Charting Component
 * Sophisticated trading charts with real-time data integration
 * Phase 5: Advanced Features & Performance Optimization
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { 
  BarChart3, 
  TrendingUp, 
  TrendingDown, 
  Activity,
  Settings,
  Maximize2,
  Minimize2,
  RefreshCw,
  Play,
  Pause,
  Volume2,
  Target,
  Layers,
  Zap
} from 'lucide-react';
import { useRealTimeAnalysisSimulation } from '../../lib/hooks_single_pair';

interface AdvancedChartingProps {
  selectedPair: string;
  selectedTimeframe: string;
  className?: string;
  showIndicators?: boolean;
  showVolume?: boolean;
  showTPLevels?: boolean;
  autoRefresh?: boolean;
}

interface ChartData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface IndicatorData {
  rsi: number[];
  macd: { macd: number[]; signal: number[]; histogram: number[] };
  sma20: number[];
  sma50: number[];
  ema12: number[];
  ema26: number[];
}

export const AdvancedCharting: React.FC<AdvancedChartingProps> = ({
  selectedPair,
  selectedTimeframe,
  className = '',
  showIndicators = true,
  showVolume = true,
  showTPLevels = true,
  autoRefresh = true
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [indicatorData, setIndicatorData] = useState<IndicatorData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(true);
  const [chartType, setChartType] = useState<'candlestick' | 'line' | 'area'>('candlestick');
  const [timeRange, setTimeRange] = useState<'1h' | '4h' | '1d' | '1w'>('1h');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);

  // Get real-time analysis data
  const { analysisData, isUpdating } = useRealTimeAnalysisSimulation(selectedPair, selectedTimeframe);

  // Generate mock chart data
  const generateChartData = useCallback((pair: string, timeframe: string, count: number = 100): ChartData[] => {
    const data: ChartData[] = [];
    const basePrice = pair.includes('BTC') ? 50000 : pair.includes('ETH') ? 3000 : 100;
    const volatility = 0.02; // 2% volatility
    
    let currentPrice = basePrice;
    const now = Date.now();
    const intervalMs = timeframe === '15m' ? 15 * 60 * 1000 : 
                      timeframe === '1h' ? 60 * 60 * 1000 :
                      timeframe === '4h' ? 4 * 60 * 60 * 1000 :
                      timeframe === '1d' ? 24 * 60 * 60 * 1000 : 60 * 60 * 1000;

    for (let i = count - 1; i >= 0; i--) {
      const timestamp = now - (i * intervalMs);
      const priceChange = (Math.random() - 0.5) * volatility * currentPrice;
      const open = currentPrice;
      const close = currentPrice + priceChange;
      const high = Math.max(open, close) + Math.random() * volatility * currentPrice * 0.5;
      const low = Math.min(open, close) - Math.random() * volatility * currentPrice * 0.5;
      const volume = 1000000 + Math.random() * 500000;

      data.push({
        timestamp,
        open,
        high,
        low,
        close,
        volume
      });

      currentPrice = close;
    }

    return data;
  }, []);

  // Calculate technical indicators
  const calculateIndicators = useCallback((data: ChartData[]): IndicatorData => {
    const closes = data.map(d => d.close);
    const volumes = data.map(d => d.volume);

    // RSI calculation
    const rsi = calculateRSI(closes, 14);
    
    // MACD calculation
    const macd = calculateMACD(closes, 12, 26, 9);
    
    // Moving averages
    const sma20 = calculateSMA(closes, 20);
    const sma50 = calculateSMA(closes, 50);
    const ema12 = calculateEMA(closes, 12);
    const ema26 = calculateEMA(closes, 26);

    return {
      rsi,
      macd,
      sma20,
      sma50,
      ema12,
      ema26
    };
  }, []);

  // Initialize chart data
  useEffect(() => {
    setIsLoading(true);
    const data = generateChartData(selectedPair, selectedTimeframe);
    setChartData(data);
    
    const indicators = calculateIndicators(data);
    setIndicatorData(indicators);
    
    setIsLoading(false);
  }, [selectedPair, selectedTimeframe, generateChartData, calculateIndicators]);

  // Auto-refresh data
  useEffect(() => {
    if (!autoRefresh || !isPlaying) return;

    const interval = setInterval(() => {
      setChartData(prev => {
        const newData = [...prev];
        const lastCandle = newData[newData.length - 1];
        const newCandle = {
          timestamp: Date.now(),
          open: lastCandle.close,
          close: lastCandle.close + (Math.random() - 0.5) * lastCandle.close * 0.01,
          high: lastCandle.close + Math.random() * lastCandle.close * 0.005,
          low: lastCandle.close - Math.random() * lastCandle.close * 0.005,
          volume: 1000000 + Math.random() * 500000
        };
        
        newCandle.high = Math.max(newCandle.open, newCandle.close, newCandle.high);
        newCandle.low = Math.min(newCandle.open, newCandle.close, newCandle.low);
        
        newData.push(newCandle);
        return newData.slice(-100); // Keep only last 100 candles
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [autoRefresh, isPlaying]);

  // Draw chart
  const drawChart = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !chartData.length) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    // Chart dimensions
    const padding = 40;
    const chartWidth = width - padding * 2;
    const chartHeight = height - padding * 2;

    // Find price range
    const prices = chartData.flatMap(d => [d.high, d.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    // Draw candlesticks
    const candleWidth = chartWidth / chartData.length;
    
    chartData.forEach((candle, index) => {
      const x = padding + index * candleWidth;
      const highY = padding + ((maxPrice - candle.high) / priceRange) * chartHeight;
      const lowY = padding + ((maxPrice - candle.low) / priceRange) * chartHeight;
      const openY = padding + ((maxPrice - candle.open) / priceRange) * chartHeight;
      const closeY = padding + ((maxPrice - candle.close) / priceRange) * chartHeight;

      // Draw wick
      ctx.strokeStyle = candle.close >= candle.open ? '#10b981' : '#ef4444';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, highY);
      ctx.lineTo(x + candleWidth / 2, lowY);
      ctx.stroke();

      // Draw body
      const bodyHeight = Math.abs(closeY - openY);
      const bodyTop = Math.min(openY, closeY);
      
      ctx.fillStyle = candle.close >= candle.open ? '#10b981' : '#ef4444';
      ctx.fillRect(x + candleWidth * 0.1, bodyTop, candleWidth * 0.8, bodyHeight);
    });

    // Draw moving averages
    if (indicatorData && showIndicators) {
      // SMA 20
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      indicatorData.sma20.forEach((value, index) => {
        if (value) {
          const x = padding + index * candleWidth;
          const y = padding + ((maxPrice - value) / priceRange) * chartHeight;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // SMA 50
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.beginPath();
      indicatorData.sma50.forEach((value, index) => {
        if (value) {
          const x = padding + index * candleWidth;
          const y = padding + ((maxPrice - value) / priceRange) * chartHeight;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    }

    // Draw price levels
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + (priceRange * i / 5);
      const y = padding + (i / 5) * chartHeight;
      
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
      
      // Price labels
      ctx.fillStyle = '#9ca3af';
      ctx.font = '12px Arial';
      ctx.fillText(price.toFixed(2), 5, y + 4);
    }
    
    ctx.setLineDash([]);
  }, [chartData, indicatorData, showIndicators]);

  // Draw chart when data changes
  useEffect(() => {
    drawChart();
  }, [drawChart]);

  // Handle canvas resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      }
      
      drawChart();
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, [drawChart]);

  return (
    <Card className={`bg-gray-900 border-gray-800 ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <BarChart3 className="h-6 w-6 text-blue-500" />
            <div>
              <CardTitle className="text-white text-lg">
                Advanced Charting - {selectedPair}
              </CardTitle>
              <p className="text-gray-400 text-sm">
                {selectedTimeframe} • {chartData.length} candles
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Chart Controls */}
            <div className="flex items-center space-x-1 bg-gray-800 rounded-lg p-1">
              <Button
                variant={chartType === 'candlestick' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setChartType('candlestick')}
                className="h-8 w-8 p-0"
              >
                <BarChart3 className="h-4 w-4" />
              </Button>
              <Button
                variant={chartType === 'line' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setChartType('line')}
                className="h-8 w-8 p-0"
              >
                <TrendingUp className="h-4 w-4" />
              </Button>
            </div>
            
            {/* Play/Pause */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsPlaying(!isPlaying)}
              className="border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            </Button>
            
            {/* Refresh */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                const data = generateChartData(selectedPair, selectedTimeframe);
                setChartData(data);
                const indicators = calculateIndicators(data);
                setIndicatorData(indicators);
              }}
              className="border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              <RefreshCw className={`h-4 w-4 ${isUpdating ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-0">
        {/* Chart Canvas */}
        <div className="relative h-96 bg-gray-950">
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ background: '#0a0a0a' }}
          />
          
          {/* Loading Overlay */}
          {isLoading && (
            <div className="absolute inset-0 bg-gray-950/80 flex items-center justify-center">
              <div className="flex items-center space-x-2">
                <RefreshCw className="h-5 w-5 animate-spin text-blue-500" />
                <span className="text-white">Loading chart data...</span>
              </div>
            </div>
          )}
          
          {/* Live Indicator */}
          {isPlaying && (
            <div className="absolute top-4 right-4">
              <Badge className="bg-green-500/20 text-green-400 border-green-500/30">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse mr-2" />
                LIVE
              </Badge>
            </div>
          )}
        </div>
        
        {/* Technical Indicators */}
        {showIndicators && indicatorData && (
          <div className="p-4 border-t border-gray-800">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* RSI */}
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">RSI (14)</span>
                  <span className={`text-sm font-medium ${
                    indicatorData.rsi[indicatorData.rsi.length - 1] > 70 ? 'text-red-400' :
                    indicatorData.rsi[indicatorData.rsi.length - 1] < 30 ? 'text-green-400' :
                    'text-yellow-400'
                  }`}>
                    {indicatorData.rsi[indicatorData.rsi.length - 1]?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-red-500 to-yellow-500 to-green-500 h-2 rounded-full"
                    style={{ width: `${indicatorData.rsi[indicatorData.rsi.length - 1] || 0}%` }}
                  />
                </div>
              </div>
              
              {/* MACD */}
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">MACD</span>
                  <span className={`text-sm font-medium ${
                    (indicatorData.macd.macd[indicatorData.macd.macd.length - 1] || 0) > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(indicatorData.macd.macd[indicatorData.macd.macd.length - 1] || 0).toFixed(4)}
                  </span>
                </div>
                <div className="text-xs text-gray-500">
                  Signal: {(indicatorData.macd.signal[indicatorData.macd.signal.length - 1] || 0).toFixed(4)}
                </div>
              </div>
              
              {/* Moving Averages */}
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">MA Cross</span>
                  <span className={`text-sm font-medium ${
                    (indicatorData.sma20[indicatorData.sma20.length - 1] || 0) > (indicatorData.sma50[indicatorData.sma50.length - 1] || 0) 
                      ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(indicatorData.sma20[indicatorData.sma20.length - 1] || 0) > (indicatorData.sma50[indicatorData.sma50.length - 1] || 0) ? 'Bullish' : 'Bearish'}
                  </span>
                </div>
                <div className="text-xs text-gray-500">
                  SMA20: {(indicatorData.sma20[indicatorData.sma20.length - 1] || 0).toFixed(2)}
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

// Technical indicator calculation functions
function calculateRSI(prices: number[], period: number = 14): number[] {
  const rsi: number[] = [];
  
  for (let i = period; i < prices.length; i++) {
    const gains: number[] = [];
    const losses: number[] = [];
    
    for (let j = i - period + 1; j <= i; j++) {
      const change = prices[j] - prices[j - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? -change : 0);
    }
    
    const avgGain = gains.reduce((sum, gain) => sum + gain, 0) / period;
    const avgLoss = losses.reduce((sum, loss) => sum + loss, 0) / period;
    
    const rs = avgGain / (avgLoss || 0.0001);
    const rsiValue = 100 - (100 / (1 + rs));
    
    rsi.push(rsiValue);
  }
  
  return rsi;
}

function calculateMACD(prices: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) {
  const emaFast = calculateEMA(prices, fastPeriod);
  const emaSlow = calculateEMA(prices, slowPeriod);
  
  const macd: number[] = [];
  for (let i = 0; i < emaFast.length; i++) {
    macd.push(emaFast[i] - emaSlow[i]);
  }
  
  const signal = calculateEMA(macd, signalPeriod);
  const histogram: number[] = [];
  
  for (let i = 0; i < macd.length; i++) {
    histogram.push(macd[i] - (signal[i] || 0));
  }
  
  return { macd, signal, histogram };
}

function calculateSMA(prices: number[], period: number): number[] {
  const sma: number[] = [];
  
  for (let i = period - 1; i < prices.length; i++) {
    const sum = prices.slice(i - period + 1, i + 1).reduce((sum, price) => sum + price, 0);
    sma.push(sum / period);
  }
  
  return sma;
}

function calculateEMA(prices: number[], period: number): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);
  
  // First EMA is SMA
  const firstSMA = prices.slice(0, period).reduce((sum, price) => sum + price, 0) / period;
  ema.push(firstSMA);
  
  for (let i = period; i < prices.length; i++) {
    const emaValue = (prices[i] * multiplier) + (ema[ema.length - 1] * (1 - multiplier));
    ema.push(emaValue);
  }
  
  return ema;
}

export default AdvancedCharting;
