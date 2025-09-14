/**
 * Real-Time Analysis Panels Component
 * Displays FA, TA, and Sentiment analysis in real-time
 */

import React, { useState, useEffect, memo } from 'react';
import { 
  BarChart3, 
  TrendingUp, 
  Brain, 
  Activity,
  Zap,
  AlertTriangle,
  CheckCircle,
  Clock,
  RefreshCw
} from 'lucide-react';
import { useRealTimeAnalysisSimulation } from '../../lib/hooks_single_pair';
import { usePerformanceMonitor, useDebouncedValue } from '../../lib/performance';

interface AnalysisData {
  fundamental: {
    marketRegime: string;
    newsImpact: number;
    macroFactors: string;
    confidence: number;
    lastUpdate: string;
  };
  technical: {
    rsi: number;
    macd: string;
    pattern: string;
    confidence: number;
    lastUpdate: string;
  };
  sentiment: {
    socialSentiment: number;
    fearGreed: string;
    volume: number;
    confidence: number;
    lastUpdate: string;
  };
}

interface AnalysisPanelsProps {
  selectedPair: string;
  selectedTimeframe: string;
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const AnalysisPanels: React.FC<AnalysisPanelsProps> = memo(({
  selectedPair,
  selectedTimeframe,
  className = '',
  autoRefresh = true,
  refreshInterval = 5000
}) => {
  // Performance monitoring
  usePerformanceMonitor('AnalysisPanels');

  // Use the hook for real-time analysis data
  const {
    analysisData,
    isUpdating,
    isLoading,
    error
  } = useRealTimeAnalysisSimulation(selectedPair, selectedTimeframe);

  // Debounce analysis data to prevent excessive re-renders
  const debouncedAnalysisData = useDebouncedValue(analysisData, 200);

  const [lastUpdate, setLastUpdate] = useState<string>('');

  // Simulate real-time data updates
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      setIsRefreshing(true);
      
      // Simulate data updates
      setTimeout(() => {
        setAnalysisData(prev => ({
          fundamental: {
            ...prev.fundamental,
            newsImpact: prev.fundamental.newsImpact + (Math.random() - 0.5) * 0.5,
            confidence: Math.min(0.95, prev.fundamental.confidence + (Math.random() - 0.5) * 0.02),
            lastUpdate: new Date().toISOString()
          },
          technical: {
            ...prev.technical,
            rsi: Math.max(0, Math.min(100, prev.technical.rsi + (Math.random() - 0.5) * 2)),
            confidence: Math.min(0.95, prev.technical.confidence + (Math.random() - 0.5) * 0.02),
            lastUpdate: new Date().toISOString()
          },
          sentiment: {
            ...prev.sentiment,
            socialSentiment: prev.sentiment.socialSentiment + (Math.random() - 0.5) * 2,
            volume: Math.max(0.5, prev.sentiment.volume + (Math.random() - 0.5) * 0.2),
            confidence: Math.min(0.95, prev.sentiment.confidence + (Math.random() - 0.5) * 0.02),
            lastUpdate: new Date().toISOString()
          }
        }));
        setIsRefreshing(false);
      }, 1000);
    }, refreshInterval);

    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceBgColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-500/20';
    if (confidence >= 0.6) return 'bg-yellow-500/20';
    return 'bg-red-500/20';
  };

  const getMarketRegimeColor = (regime: string) => {
    switch (regime.toLowerCase()) {
      case 'bullish': return 'text-green-400';
      case 'bearish': return 'text-red-400';
      case 'neutral': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getFearGreedColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'extreme greed': return 'text-red-400';
      case 'greed': return 'text-yellow-400';
      case 'neutral': return 'text-gray-400';
      case 'fear': return 'text-blue-400';
      case 'extreme fear': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  };

  const formatTimeAgo = (timestamp: string) => {
    const now = new Date();
    const updateTime = new Date(timestamp);
    const diffMs = now.getTime() - updateTime.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    
    if (diffSecs < 60) return `${diffSecs}s ago`;
    if (diffSecs < 3600) return `${Math.floor(diffSecs / 60)}m ago`;
    return `${Math.floor(diffSecs / 3600)}h ago`;
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Activity className="h-5 w-5 text-blue-500" />
          <h2 className="text-xl font-bold text-white">Real-Time Analysis</h2>
          <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white text-xs px-2 py-1 rounded-full">
            LIVE
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isRefreshing ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
          <span className="text-sm text-gray-400">
            {isRefreshing ? 'Updating...' : 'Live'}
          </span>
        </div>
      </div>

      {/* Analysis Panels Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Fundamental Analysis Panel */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5 text-blue-500" />
              <h3 className="text-white font-semibold">📊 Fundamental</h3>
            </div>
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceBgColor(analysisData.fundamental.confidence)} ${getConfidenceColor(analysisData.fundamental.confidence)}`}>
              {(analysisData.fundamental.confidence * 100).toFixed(0)}%
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Market Regime:</span>
              <span className={`text-sm font-medium ${getMarketRegimeColor(analysisData.fundamental.marketRegime)}`}>
                {analysisData.fundamental.marketRegime}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">News Impact:</span>
              <span className={`text-sm font-medium ${analysisData.fundamental.newsImpact >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {analysisData.fundamental.newsImpact >= 0 ? '+' : ''}{analysisData.fundamental.newsImpact.toFixed(1)}%
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Macro Factors:</span>
              <span className="text-white text-sm font-medium">{analysisData.fundamental.macroFactors}</span>
            </div>
            
            <div className="pt-2 border-t border-gray-800">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Last update:</span>
                <span>{formatTimeAgo(analysisData.fundamental.lastUpdate)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Technical Analysis Panel */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5 text-green-500" />
              <h3 className="text-white font-semibold">📈 Technical</h3>
            </div>
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceBgColor(analysisData.technical.confidence)} ${getConfidenceColor(analysisData.technical.confidence)}`}>
              {(analysisData.technical.confidence * 100).toFixed(0)}%
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">RSI:</span>
              <span className={`text-sm font-medium ${
                analysisData.technical.rsi > 70 ? 'text-red-400' :
                analysisData.technical.rsi < 30 ? 'text-green-400' : 'text-yellow-400'
              }`}>
                {analysisData.technical.rsi.toFixed(1)}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">MACD:</span>
              <span className={`text-sm font-medium ${
                analysisData.technical.macd.toLowerCase() === 'bullish' ? 'text-green-400' : 'text-red-400'
              }`}>
                {analysisData.technical.macd}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Pattern:</span>
              <span className="text-white text-sm font-medium">{analysisData.technical.pattern}</span>
            </div>
            
            <div className="pt-2 border-t border-gray-800">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Last update:</span>
                <span>{formatTimeAgo(analysisData.technical.lastUpdate)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Sentiment Analysis Panel */}
        <div className="bg-gray-900 rounded-lg p-4 border border-gray-800 hover:border-gray-700 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-purple-500" />
              <h3 className="text-white font-semibold">💭 Sentiment</h3>
            </div>
            <div className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceBgColor(analysisData.sentiment.confidence)} ${getConfidenceColor(analysisData.sentiment.confidence)}`}>
              {(analysisData.sentiment.confidence * 100).toFixed(0)}%
            </div>
          </div>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Social Sentiment:</span>
              <span className={`text-sm font-medium ${analysisData.sentiment.socialSentiment >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {analysisData.sentiment.socialSentiment >= 0 ? '+' : ''}{analysisData.sentiment.socialSentiment.toFixed(1)}%
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Fear & Greed:</span>
              <span className={`text-sm font-medium ${getFearGreedColor(analysisData.sentiment.fearGreed)}`}>
                {analysisData.sentiment.fearGreed}
              </span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Volume:</span>
              <span className="text-white text-sm font-medium">{analysisData.sentiment.volume.toFixed(1)}x</span>
            </div>
            
            <div className="pt-2 border-t border-gray-800">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Last update:</span>
                <span>{formatTimeAgo(analysisData.sentiment.lastUpdate)}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Overall Analysis Summary */}
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-white font-semibold flex items-center space-x-2">
            <Zap className="h-4 w-4 text-yellow-400" />
            <span>Overall Analysis Summary</span>
          </h3>
          <div className="text-sm text-gray-400">
            {selectedPair} • {selectedTimeframe}
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">
              {((analysisData.fundamental.confidence + analysisData.technical.confidence + analysisData.sentiment.confidence) / 3 * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-400">Overall Confidence</div>
          </div>
          
          <div className="text-center">
            <div className="text-lg font-semibold text-blue-400">
              {analysisData.fundamental.marketRegime}
            </div>
            <div className="text-xs text-gray-400">Market Regime</div>
          </div>
          
          <div className="text-center">
            <div className="text-lg font-semibold text-purple-400">
              {analysisData.sentiment.fearGreed}
            </div>
            <div className="text-xs text-gray-400">Market Sentiment</div>
          </div>
        </div>
      </div>
    </div>
  );
});

AnalysisPanels.displayName = 'AnalysisPanels';

export default AnalysisPanels;
