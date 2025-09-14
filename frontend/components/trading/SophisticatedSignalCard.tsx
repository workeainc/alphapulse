/**
 * Sophisticated Signal Card Component
 * Advanced signal display with 4-TP system, confidence visualization, and real-time analysis
 */

import React, { useState } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Target, 
  Clock, 
  BarChart3,
  Eye,
  EyeOff,
  Play,
  Square,
  Layers,
  Shield,
  AlertTriangle,
  Zap,
  Brain,
  Activity
} from 'lucide-react';
import { IntelligentSignal } from '../../lib/api_intelligent';
import { useSinglePairSignalSimulation } from '../../lib/hooks_single_pair';

interface SophisticatedSignalCardProps {
  signal: IntelligentSignal;
  isSureShot?: boolean;
  showDetails?: boolean;
  onToggleDetails?: () => void;
  onExecute?: () => void;
  onAddToWatchlist?: () => void;
  compact?: boolean;
  className?: string;
}

export const SophisticatedSignalCard: React.FC<SophisticatedSignalCardProps> = ({
  signal,
  isSureShot = false,
  showDetails = false,
  onToggleDetails,
  onExecute,
  onAddToWatchlist,
  compact = false,
  className = ''
}) => {
  const [isHovered, setIsHovered] = useState(false);

  // Determine if this is a sure shot signal
  const sureShot = isSureShot || signal.confidence_score >= 0.85;

  const getSignalIcon = () => {
    if (signal.signal_type === 'no_safe_entry') {
      return <AlertTriangle className="h-6 w-6 text-yellow-400" />;
    }
    
    return signal.signal_direction === 'long' ? (
      <TrendingUp className="h-6 w-6 text-green-400" />
    ) : (
      <TrendingDown className="h-6 w-6 text-red-400" />
    );
  };

  const getSignalBadge = () => {
    if (sureShot) {
      return (
        <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white text-xs px-3 py-1 rounded-full flex items-center space-x-1 animate-pulse">
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

  const formatPrice = (price: number | undefined) => {
    if (!price) return 'N/A';
    return `$${price.toFixed(4)}`;
  };

  const formatConfidence = (confidence: number) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const formatRiskReward = (ratio: number | undefined) => {
    if (!ratio) return 'N/A';
    return ratio.toFixed(2);
  };

  return (
    <div 
      className={`rounded-lg p-4 border transition-all duration-300 hover:shadow-lg ${
        sureShot 
          ? 'bg-gradient-to-r from-green-900/30 to-blue-900/30 border-green-500/30 hover:border-green-400/50' 
          : 'bg-gray-800 border-gray-700 hover:border-gray-600'
      } ${className}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Signal Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg transition-all duration-200 ${
            sureShot ? 'bg-green-500/20' : 'bg-gray-700'
          } ${isHovered ? 'scale-105' : ''}`}>
            {getSignalIcon()}
          </div>
          <div>
            <h3 className="text-white font-semibold text-lg">{signal.symbol}</h3>
            <p className="text-gray-400 text-sm">{signal.timeframe}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          {getSignalBadge()}
          <div className="text-right">
            <div className={`text-2xl font-bold transition-colors duration-200 ${
              sureShot ? 'text-green-400' : 'text-blue-400'
            }`}>
              {formatConfidence(signal.confidence_score)}
            </div>
            <div className="text-xs text-gray-400">Confidence</div>
          </div>
          {onToggleDetails && (
            <button
              onClick={onToggleDetails}
              className="text-gray-400 hover:text-white transition-colors p-1 rounded hover:bg-gray-700"
            >
              {showDetails ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          )}
        </div>
      </div>

      {/* 4-TP System Visualization */}
      {signal.signal_type === 'entry' && signal.entry_price && !compact && (
        <div className="mb-4">
          <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
            <Layers className="h-4 w-4 text-blue-400" />
            <span>4-TP System</span>
            {sureShot && <Zap className="h-4 w-4 text-yellow-400 animate-pulse" />}
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { level: 1, price: signal.take_profit_1, percentage: 25, color: 'green-400' },
              { level: 2, price: signal.take_profit_2, percentage: 25, color: 'green-500' },
              { level: 3, price: signal.take_profit_3, percentage: 25, color: 'green-600' },
              { level: 4, price: signal.take_profit_4, percentage: 25, color: 'green-700' }
            ].map((tp) => (
              <div key={tp.level} className={`bg-gray-700 rounded-lg p-3 border border-gray-600 transition-all duration-200 ${
                sureShot ? 'ring-1 ring-green-500/30 hover:ring-green-400/50' : 'hover:border-gray-500'
              }`}>
                <div className="text-center">
                  <div className="text-sm text-gray-400">TP{tp.level}</div>
                  <div className="text-lg font-bold text-white">
                    {formatPrice(tp.price)}
                  </div>
                  <div className="text-xs text-gray-500">{tp.percentage}%</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Entry and Risk Management */}
      {signal.signal_type === 'entry' && !compact && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-gray-700 rounded-lg p-3 border border-gray-600 hover:border-gray-500 transition-colors">
            <div className="text-center">
              <div className="text-sm text-gray-400">Entry Price</div>
              <div className="text-xl font-bold text-white">
                {formatPrice(signal.entry_price)}
              </div>
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3 border border-gray-600 hover:border-gray-500 transition-colors">
            <div className="text-center">
              <div className="text-sm text-gray-400">Stop Loss</div>
              <div className="text-xl font-bold text-red-400">
                {formatPrice(signal.stop_loss)}
              </div>
            </div>
          </div>
          <div className="bg-gray-700 rounded-lg p-3 border border-gray-600 hover:border-gray-500 transition-colors">
            <div className="text-center">
              <div className="text-sm text-gray-400">Risk/Reward</div>
              <div className="text-xl font-bold text-blue-400">
                {formatRiskReward(signal.risk_reward_ratio)}:1
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Compact View for Entry Signals */}
      {signal.signal_type === 'entry' && compact && (
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="bg-gray-700 rounded p-2">
            <div className="text-xs text-gray-400">Entry</div>
            <div className="text-sm font-bold text-white">{formatPrice(signal.entry_price)}</div>
          </div>
          <div className="bg-gray-700 rounded p-2">
            <div className="text-xs text-gray-400">Stop Loss</div>
            <div className="text-sm font-bold text-red-400">{formatPrice(signal.stop_loss)}</div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      {sureShot && signal.signal_type === 'entry' && (
        <div className="mb-4">
          <div className="flex space-x-3">
            {onExecute && (
              <button
                onClick={onExecute}
                className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-bold py-3 px-4 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2 hover:scale-105 transform"
              >
                <Play className="h-4 w-4" />
                <span>Execute Sure Shot Trade</span>
              </button>
            )}
            {onAddToWatchlist && (
              <button
                onClick={onAddToWatchlist}
                className="px-4 py-3 border border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 rounded-lg transition-all duration-200 hover:bg-gray-700"
              >
                <Square className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Detailed Analysis */}
      {showDetails && !compact && (
        <div className="mt-4 pt-4 border-t border-gray-700 space-y-3">
          <h4 className="text-white font-medium text-sm flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Real-Time Analysis</span>
            <Activity className="h-3 w-3 text-green-400 animate-pulse" />
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

          {/* Confidence Breakdown */}
          <div className="bg-gray-700/30 rounded p-3 border border-gray-600">
            <p className="text-gray-400 text-xs mb-2 font-medium">🎯 Confidence Breakdown</p>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span className="text-gray-400">Overall Confidence:</span>
                <span className="text-white ml-1 font-medium">{formatConfidence(signal.confidence_score)}</span>
              </div>
              <div>
                <span className="text-gray-400">Signal Strength:</span>
                <span className="text-white ml-1 font-medium">{signal.signal_strength?.replace('_', ' ').toUpperCase() || 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-400">Risk Level:</span>
                <span className="text-white ml-1 font-medium">{signal.risk_level?.toUpperCase() || 'N/A'}</span>
              </div>
              <div>
                <span className="text-gray-400">R:R Ratio:</span>
                <span className="text-white ml-1 font-medium">{formatRiskReward(signal.risk_reward_ratio)}:1</span>
              </div>
            </div>
          </div>
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

export default SophisticatedSignalCard;
