/**
 * Intelligent Signal Feed Component
 * Displays intelligent trading signals with comprehensive analysis
 */

import React, { useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Clock, 
  Target, 
  Shield,
  BarChart3,
  Activity,
  Zap,
  Eye,
  EyeOff
} from 'lucide-react';
import { 
  useLatestIntelligentSignals, 
  useHighConfidenceSignals,
  useEntrySignals,
  useNoSafeEntrySignals 
} from '../../lib/hooks_intelligent';
import { 
  IntelligentSignal,
  getSignalIcon, 
  getSignalColor, 
  getRiskLevelColor,
  getSignalStrengthColor,
  formatConfidence,
  formatPrice,
  formatRiskReward,
  formatPercentage
} from '../../lib/api_intelligent';

/**
 * Sophisticated Single-Pair Signal Feed Component
 * Displays intelligent trading signals focused on single-pair analysis with 4-TP system
 */

import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Clock, 
  Target, 
  Shield,
  BarChart3,
  Activity,
  Zap,
  Eye,
  EyeOff,
  Brain,
  Gauge,
  Layers,
  Play,
  Square,
  Filter,
  RefreshCw
} from 'lucide-react';
import { 
  useLatestIntelligentSignals, 
  useHighConfidenceSignals,
  useEntrySignals,
  useNoSafeEntrySignals 
} from '../../lib/hooks_intelligent';
import { 
  IntelligentSignal,
  getSignalIcon, 
  getSignalColor, 
  getRiskLevelColor,
  getSignalStrengthColor,
  formatConfidence,
  formatPrice,
  formatRiskReward,
  formatPercentage
} from '../../lib/api_intelligent';

interface SophisticatedSignalFeedProps {
  selectedPair: string;
  selectedTimeframe: string;
  maxSignals?: number;
  showNoSafeEntry?: boolean;
  autoRefresh?: boolean;
  onSignalSelect?: (signal: IntelligentSignal) => void;
}

export const SophisticatedSignalFeed: React.FC<SophisticatedSignalFeedProps> = ({
  selectedPair,
  selectedTimeframe,
  maxSignals = 5,
  showNoSafeEntry = true,
  autoRefresh = true,
  onSignalSelect
}) => {
  const [filter, setFilter] = useState<'sure_shot' | 'building' | 'all' | 'entry' | 'no_safe_entry'>('sure_shot');
  const [showDetails, setShowDetails] = useState<Record<string, boolean>>({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);

  // Get signals based on filter
  const { data: allSignalsData, isLoading: isLoadingAll } = useLatestIntelligentSignals(maxSignals);
  const { signals: highConfidenceSignals, isLoading: isLoadingHigh } = useHighConfidenceSignals();
  const { signals: entrySignals, isLoading: isLoadingEntry } = useEntrySignals();
  const { signals: noSafeEntrySignals, isLoading: isLoadingNoSafe } = useNoSafeEntrySignals();

  // Filter signals for selected pair and timeframe
  const getFilteredSignals = () => {
    let baseSignals: IntelligentSignal[] = [];
    
    switch (filter) {
      case 'sure_shot':
        baseSignals = highConfidenceSignals.filter(s => s.confidence_score >= 0.85);
        break;
      case 'building':
        baseSignals = allSignalsData?.signals?.filter(s => s.confidence_score < 0.85 && s.confidence_score >= 0.6) || [];
        break;
      case 'entry':
        baseSignals = entrySignals;
        break;
      case 'no_safe_entry':
        baseSignals = noSafeEntrySignals;
        break;
      default:
        baseSignals = allSignalsData?.signals || [];
    }
    
    // Filter by selected pair and timeframe
    return baseSignals
      .filter(signal => signal.symbol === selectedPair && signal.timeframe === selectedTimeframe)
      .slice(0, maxSignals);
  };

  const signals = getFilteredSignals();
  
  // Simulate analysis progress for building signals
  useEffect(() => {
    if (filter === 'building' && signals.length > 0) {
      setIsAnalyzing(true);
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 100) {
            setIsAnalyzing(false);
            clearInterval(interval);
            return 100;
          }
          return prev + Math.random() * 10;
        });
      }, 500);
      
      return () => clearInterval(interval);
    } else {
      setIsAnalyzing(false);
      setAnalysisProgress(0);
    }
  }, [filter, signals.length]);
  const isLoading = isLoadingAll || isLoadingHigh || isLoadingEntry || isLoadingNoSafe;

  const toggleDetails = (signalId: string) => {
    setShowDetails(prev => ({
      ...prev,
      [signalId]: !prev[signalId]
    }));
  };

  const getFilterCount = (filterType: string) => {
    const allPairSignals = (allSignalsData?.signals || [])
      .filter(signal => signal.symbol === selectedPair && signal.timeframe === selectedTimeframe);
    
    switch (filterType) {
      case 'sure_shot':
        return allPairSignals.filter(s => s.confidence_score >= 0.85).length;
      case 'building':
        return allPairSignals.filter(s => s.confidence_score < 0.85 && s.confidence_score >= 0.6).length;
      case 'entry':
        return entrySignals.filter(s => s.symbol === selectedPair && s.timeframe === selectedTimeframe).length;
      case 'no_safe_entry':
        return noSafeEntrySignals.filter(s => s.symbol === selectedPair && s.timeframe === selectedTimeframe).length;
      default:
        return allPairSignals.length;
    }
  };

  const handleSignalSelect = (signal: IntelligentSignal) => {
    if (onSignalSelect) {
      onSignalSelect(signal);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-300">Analyzing {selectedPair}...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-6">
      {/* Sophisticated Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Brain className="h-6 w-6 text-blue-500" />
          <div>
            <h2 className="text-xl font-bold text-white">Single-Pair Analysis</h2>
            <p className="text-gray-400 text-sm">{selectedPair} - {selectedTimeframe}</p>
          </div>
          <div className="bg-gradient-to-r from-blue-500 to-purple-500 text-white text-xs px-3 py-1 rounded-full">
            🎯 SURE SHOT FOCUS
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-green-400 font-medium">LIVE ANALYSIS</span>
        </div>
      </div>

      {/* Sophisticated Filter Tabs */}
      <div className="flex space-x-2 mb-6">
        {[
          { key: 'sure_shot', label: '🎯 Sure Shot', icon: Target, color: 'green' },
          { key: 'building', label: '⚡ Building', icon: Gauge, color: 'yellow' },
          { key: 'entry', label: '📈 Entry Ready', icon: TrendingUp, color: 'blue' },
          { key: 'no_safe_entry', label: '⚠️ No Safe Entry', icon: AlertTriangle, color: 'red' }
        ].map(({ key, label, icon: Icon, color }) => (
          <button
            key={key}
            onClick={() => setFilter(key as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              filter === key
                ? `bg-${color}-600 text-white shadow-lg`
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
            <span className={`text-xs px-2 py-1 rounded-full ${
              filter === key ? 'bg-white/20' : 'bg-gray-700'
            }`}>
              {getFilterCount(key)}
            </span>
          </button>
        ))}
      </div>

      {/* Analysis Progress Indicator */}
      {isAnalyzing && (
        <div className="mb-6 bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <Gauge className="h-4 w-4 text-yellow-400" />
              <span className="text-yellow-400 text-sm font-medium">Building Confidence...</span>
            </div>
            <span className="text-yellow-400 text-sm font-medium">{Math.round(analysisProgress)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className="bg-gradient-to-r from-yellow-500 to-orange-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${analysisProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Signals List */}
      <div className="space-y-4">
        {signals.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
              <Target className="h-8 w-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">
              {filter === 'sure_shot' ? 'No Sure Shot Signals' : 
               filter === 'building' ? 'No Building Signals' :
               filter === 'entry' ? 'No Entry Signals' : 'No Signals'}
            </h3>
            <p className="text-gray-400 text-sm">
              {filter === 'sure_shot' ? 'Waiting for 85%+ confidence signals...' :
               filter === 'building' ? 'No signals building confidence...' :
               'No signals available for this filter'}
            </p>
          </div>
        ) : (
          signals.map((signal) => (
            <SophisticatedSignalCard
              key={signal.signal_id}
              signal={signal}
              showDetails={showDetails[signal.signal_id] || false}
              onToggleDetails={() => toggleDetails(signal.signal_id)}
              onSignalSelect={() => handleSignalSelect(signal)}
              isSureShot={signal.confidence_score >= 0.85}
            />
          ))
        )}
      </div>

      {/* Sophisticated Footer */}
      <div className="mt-6 pt-4 border-t border-gray-800">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center space-x-4">
            <span>Last updated: {new Date().toLocaleTimeString()}</span>
            <span>•</span>
            <span>{signals.length} signals for {selectedPair}</span>
          </div>
          <div className="flex items-center space-x-2">
            <RefreshCw className="h-3 w-3" />
            <span>Auto-refresh enabled</span>
          </div>
        </div>
      </div>
    </div>
  );
};

interface SophisticatedSignalCardProps {
  signal: IntelligentSignal;
  showDetails: boolean;
  onToggleDetails: () => void;
  onSignalSelect: () => void;
  isSureShot: boolean;
}

const SophisticatedSignalCard: React.FC<SophisticatedSignalCardProps> = ({
  signal,
  showDetails,
  onToggleDetails,
  onSignalSelect,
  isSureShot
}) => {
  const getSignalBadge = () => {
    if (isSureShot) {
      return (
        <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white text-xs px-3 py-1 rounded-full flex items-center space-x-1">
          <Target className="h-3 w-3" />
          <span>🎯 SURE SHOT</span>
        </div>
      );
    }
    
    if (signal.signal_type === 'no_safe_entry') {
      return (
        <div className="bg-yellow-500/20 text-yellow-400 text-xs px-2 py-1 rounded-full flex items-center space-x-1">
          <AlertTriangle className="h-3 w-3" />
          <span>No Safe Entry</span>
        </div>
      );
    }
    
    return (
      <div className={`text-xs px-2 py-1 rounded-full flex items-center space-x-1 ${
        signal.signal_direction === 'long' 
          ? 'bg-green-500/20 text-green-400' 
          : 'bg-red-500/20 text-red-400'
      }`}>
        {signal.signal_direction === 'long' ? (
          <TrendingUp className="h-3 w-3" />
        ) : (
          <TrendingDown className="h-3 w-3" />
        )}
        <span className="uppercase">{signal.signal_direction}</span>
      </div>
    );
  };

  return (
    <div className={`rounded-lg p-4 border transition-all duration-200 hover:shadow-lg ${
      isSureShot 
        ? 'bg-gradient-to-r from-green-900/30 to-blue-900/30 border-green-500/30' 
        : 'bg-gray-800 border-gray-700 hover:border-gray-600'
    }`}>
      {/* Signal Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${
            isSureShot ? 'bg-green-500/20' : 'bg-gray-700'
          }`}>
            <span className="text-2xl">{getSignalIcon(signal.signal_type, signal.signal_direction)}</span>
          </div>
          <div>
            <h3 className="text-white font-semibold text-lg">{signal.symbol}</h3>
            <p className="text-gray-400 text-sm">{signal.timeframe}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          {getSignalBadge()}
          <div className="text-right">
            <div className={`text-2xl font-bold ${
              isSureShot ? 'text-green-400' : 'text-blue-400'
            }`}>
              {formatConfidence(signal.confidence_score)}
            </div>
            <div className="text-xs text-gray-400">Confidence</div>
          </div>
          <button
            onClick={onToggleDetails}
            className="text-gray-400 hover:text-white transition-colors"
          >
            {showDetails ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* 4-TP System Visualization */}
      {signal.signal_type === 'entry' && signal.entry_price && (
        <div className="mb-4">
          <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
            <Layers className="h-4 w-4 text-blue-400" />
            <span>4-TP System</span>
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { level: 1, price: signal.take_profit_1, percentage: 25, color: 'green-400' },
              { level: 2, price: signal.take_profit_2, percentage: 25, color: 'green-500' },
              { level: 3, price: signal.take_profit_3, percentage: 25, color: 'green-600' },
              { level: 4, price: signal.take_profit_4, percentage: 25, color: 'green-700' }
            ].map((tp) => (
              <div key={tp.level} className={`bg-gray-700 rounded-lg p-3 border border-gray-600 ${
                isSureShot ? 'ring-1 ring-green-500/30' : ''
              }`}>
                <div className="text-center">
                  <div className="text-sm text-gray-400">TP{tp.level}</div>
                  <div className="text-lg font-bold text-white">
                    ${tp.price?.toFixed(4) || 'N/A'}
                  </div>
                  <div className="text-xs text-gray-500">{tp.percentage}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Entry and Risk Management */}
      {signal.signal_type === 'entry' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-gray-700 rounded-lg p-3 border border-gray-600">
            <div className="text-center">
              <div className="text-sm text-gray-400">Entry Price</div>
              <div className="text-xl font-bold text-white">
                ${signal.entry_price?.toFixed(4) || 'N/A'}
              </div>
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3 border border-gray-600">
            <div className="text-center">
              <div className="text-sm text-gray-400">Stop Loss</div>
              <div className="text-xl font-bold text-red-400">
                ${signal.stop_loss?.toFixed(4) || 'N/A'}
              </div>
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3 border border-gray-600">
            <div className="text-center">
              <div className="text-sm text-gray-400">Risk/Reward</div>
              <div className="text-xl font-bold text-blue-400">
                {formatRiskReward(signal.risk_reward_ratio)}:1
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Action Button */}
      {isSureShot && (
        <div className="mb-4">
          <button
            onClick={onSignalSelect}
            className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-bold py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
          >
            <Play className="h-4 w-4" />
            <span>Execute Sure Shot Trade</span>
          </button>
        </div>
      )}

      {/* Detailed Analysis */}
      {showDetails && (
        <div className="mt-4 pt-4 border-t border-gray-700 space-y-3">
          <h4 className="text-white font-medium text-sm flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Real-Time Analysis</span>
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            <div className="bg-gray-700/30 rounded p-3 border border-gray-600">
              <p className="text-gray-400 text-xs mb-1 font-medium">📊 Pattern Analysis</p>
              <p className="text-white text-xs leading-relaxed">{signal.pattern_analysis || "Pattern analysis unavailable"}</p>
            </div>
            
            <div className="bg-gray-700/30 rounded p-3 border border-gray-600">
              <p className="text-gray-400 text-xs mb-1 font-medium">📈 Technical Analysis</p>
              <p className="text-white text-xs leading-relaxed">{signal.technical_analysis || "Technical analysis unavailable"}</p>
            </div>
            
            <div className="bg-gray-700/30 rounded p-3 border border-gray-600">
              <p className="text-gray-400 text-xs mb-1 font-medium">💭 Sentiment Analysis</p>
              <p className="text-white text-xs leading-relaxed">{signal.sentiment_analysis || "Sentiment analysis unavailable"}</p>
            </div>
            
            <div className="bg-gray-700/30 rounded p-3 border border-gray-600">
              <p className="text-gray-400 text-xs mb-1 font-medium">📊 Volume Analysis</p>
              <p className="text-white text-xs leading-relaxed">{signal.volume_analysis || "Volume analysis unavailable"}</p>
            </div>
          </div>
          
          {signal.entry_reasoning && (
            <div className="bg-blue-500/10 border border-blue-500/20 rounded p-3">
              <p className="text-blue-400 text-xs mb-1 font-medium">🎯 Entry Reasoning</p>
              <p className="text-blue-300 text-xs leading-relaxed">{signal.entry_reasoning}</p>
            </div>
          )}
        </div>
      )}

      {/* Signal Footer */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-700">
        <div className="flex items-center space-x-2 text-gray-400 text-xs">
          <Clock className="h-3 w-3" />
          <span>{new Date(signal.timestamp).toLocaleString()}</span>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-gray-400 text-xs">ID:</span>
          <span className="text-gray-300 text-xs font-mono">{signal.signal_id.slice(0, 8)}</span>
        </div>
      </div>
    </div>
  );
};

export default SophisticatedSignalFeed;
