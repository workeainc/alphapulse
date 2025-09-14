/**
 * Pair and Timeframe Selectors Component
 * Sophisticated selectors for trading pair and timeframe selection
 */

import React, { useState } from 'react';
import { 
  ChevronDown, 
  TrendingUp, 
  Clock, 
  Star,
  RefreshCw,
  Settings,
  Zap
} from 'lucide-react';

interface PairTimeframeSelectorsProps {
  selectedPair: string;
  selectedTimeframe: string;
  onPairChange: (pair: string) => void;
  onTimeframeChange: (timeframe: string) => void;
  className?: string;
  showFavorites?: boolean;
  onRefresh?: () => void;
  onSettings?: () => void;
}

export const PairTimeframeSelectors: React.FC<PairTimeframeSelectorsProps> = ({
  selectedPair,
  selectedTimeframe,
  onPairChange,
  onTimeframeChange,
  className = '',
  showFavorites = true,
  onRefresh,
  onSettings
}) => {
  const [isPairOpen, setIsPairOpen] = useState(false);
  const [isTimeframeOpen, setIsTimeframeOpen] = useState(false);

  // Popular trading pairs
  const tradingPairs = [
    { symbol: 'BTCUSDT', name: 'BTC/USDT', icon: '₿', favorite: true },
    { symbol: 'ETHUSDT', name: 'ETH/USDT', icon: 'Ξ', favorite: true },
    { symbol: 'BNBUSDT', name: 'BNB/USDT', icon: 'B', favorite: true },
    { symbol: 'ADAUSDT', name: 'ADA/USDT', icon: 'A', favorite: false },
    { symbol: 'SOLUSDT', name: 'SOL/USDT', icon: 'S', favorite: false },
    { symbol: 'XRPUSDT', name: 'XRP/USDT', icon: 'X', favorite: false },
    { symbol: 'DOTUSDT', name: 'DOT/USDT', icon: 'D', favorite: false },
    { symbol: 'AVAXUSDT', name: 'AVAX/USDT', icon: 'A', favorite: false },
    { symbol: 'MATICUSDT', name: 'MATIC/USDT', icon: 'M', favorite: false },
    { symbol: 'LINKUSDT', name: 'LINK/USDT', icon: 'L', favorite: false }
  ];

  // Timeframe options
  const timeframes = [
    { value: '15m', label: '15 Minutes', description: 'Scalping', icon: '⚡' },
    { value: '1h', label: '1 Hour', description: 'Short-term', icon: '📈' },
    { value: '4h', label: '4 Hours', description: 'Swing', icon: '🎯' },
    { value: '1d', label: '1 Day', description: 'Position', icon: '📊' },
    { value: '1w', label: '1 Week', description: 'Long-term', icon: '🌟' }
  ];

  const selectedPairData = tradingPairs.find(pair => pair.symbol === selectedPair);
  const selectedTimeframeData = timeframes.find(tf => tf.value === selectedTimeframe);

  const getTimeframeColor = (timeframe: string) => {
    switch (timeframe) {
      case '15m': return 'text-yellow-400';
      case '1h': return 'text-blue-400';
      case '4h': return 'text-green-400';
      case '1d': return 'text-purple-400';
      case '1w': return 'text-pink-400';
      default: return 'text-gray-400';
    }
  };

  const getTimeframeBgColor = (timeframe: string) => {
    switch (timeframe) {
      case '15m': return 'bg-yellow-500/20';
      case '1h': return 'bg-blue-500/20';
      case '4h': return 'bg-green-500/20';
      case '1d': return 'bg-purple-500/20';
      case '1w': return 'bg-pink-500/20';
      default: return 'bg-gray-500/20';
    }
  };

  return (
    <div className={`flex items-center space-x-4 ${className}`}>
      {/* Pair Selector */}
      <div className="relative">
        <button
          onClick={() => setIsPairOpen(!isPairOpen)}
          className="flex items-center space-x-2 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white hover:bg-gray-700 transition-colors min-w-[140px]"
        >
          <div className="flex items-center space-x-2">
            <span className="text-lg">{selectedPairData?.icon || '₿'}</span>
            <span className="font-medium">{selectedPairData?.name || 'BTC/USDT'}</span>
          </div>
          <ChevronDown className="h-4 w-4 text-gray-400" />
        </button>

        {isPairOpen && (
          <div className="absolute top-full left-0 mt-1 w-64 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50">
            <div className="p-2">
              <div className="text-xs text-gray-400 mb-2 px-2">Select Trading Pair</div>
              {tradingPairs.map((pair) => (
                <button
                  key={pair.symbol}
                  onClick={() => {
                    onPairChange(pair.symbol);
                    setIsPairOpen(false);
                  }}
                  className={`w-full flex items-center justify-between px-2 py-2 rounded text-sm hover:bg-gray-700 transition-colors ${
                    selectedPair === pair.symbol ? 'bg-blue-600/20 text-blue-400' : 'text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">{pair.icon}</span>
                    <span>{pair.name}</span>
                  </div>
                  {pair.favorite && showFavorites && (
                    <Star className="h-3 w-3 text-yellow-400 fill-current" />
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Timeframe Selector */}
      <div className="relative">
        <button
          onClick={() => setIsTimeframeOpen(!isTimeframeOpen)}
          className={`flex items-center space-x-2 border rounded-lg px-3 py-2 text-white hover:bg-gray-700 transition-colors min-w-[120px] ${getTimeframeBgColor(selectedTimeframe)} border-gray-700`}
        >
          <div className="flex items-center space-x-2">
            <span className="text-sm">{selectedTimeframeData?.icon || '📈'}</span>
            <span className={`font-medium ${getTimeframeColor(selectedTimeframe)}`}>
              {selectedTimeframeData?.label || '1 Hour'}
            </span>
          </div>
          <ChevronDown className="h-4 w-4 text-gray-400" />
        </button>

        {isTimeframeOpen && (
          <div className="absolute top-full left-0 mt-1 w-56 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-50">
            <div className="p-2">
              <div className="text-xs text-gray-400 mb-2 px-2">Select Timeframe</div>
              {timeframes.map((timeframe) => (
                <button
                  key={timeframe.value}
                  onClick={() => {
                    onTimeframeChange(timeframe.value);
                    setIsTimeframeOpen(false);
                  }}
                  className={`w-full flex items-center justify-between px-2 py-2 rounded text-sm hover:bg-gray-700 transition-colors ${
                    selectedTimeframe === timeframe.value ? 'bg-blue-600/20 text-blue-400' : 'text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <span className="text-sm">{timeframe.icon}</span>
                    <div>
                      <div className="font-medium">{timeframe.label}</div>
                      <div className="text-xs text-gray-400">{timeframe.description}</div>
                    </div>
                  </div>
                  <Clock className="h-3 w-3 text-gray-400" />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-2">
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="p-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
            title="Refresh Data"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        )}
        
        {onSettings && (
          <button
            onClick={onSettings}
            className="p-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
            title="Settings"
          >
            <Settings className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Current Selection Display */}
      <div className="hidden md:flex items-center space-x-2 text-sm text-gray-400">
        <Zap className="h-3 w-3" />
        <span>Analyzing {selectedPairData?.name} on {selectedTimeframeData?.label}</span>
      </div>
    </div>
  );
};

export default PairTimeframeSelectors;
