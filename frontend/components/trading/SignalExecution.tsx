/**
 * Signal Execution Component
 * Handles signal execution with risk management and position sizing
 */

import React, { useState } from 'react';
import { 
  Play, 
  Square, 
  AlertTriangle, 
  CheckCircle, 
  Shield,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Calculator,
  Settings,
  Zap
} from 'lucide-react';
import { IntelligentSignal } from '../../lib/api_intelligent';
import { useSignalExecution } from '../../lib/hooks_single_pair';

interface SignalExecutionProps {
  signal: IntelligentSignal;
  onExecute: (executionData: ExecutionData) => void;
  onCancel: () => void;
  className?: string;
}

interface ExecutionData {
  signalId: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  stopLoss: number;
  takeProfits: number[];
  positionSize: number;
  riskAmount: number;
  maxRisk: number;
  executionType: 'market' | 'limit';
  notes?: string;
}

export const SignalExecution: React.FC<SignalExecutionProps> = ({
  signal,
  onExecute,
  onCancel,
  className = ''
}) => {
  const [positionSize, setPositionSize] = useState(1000); // Default $1000
  const [maxRisk, setMaxRisk] = useState(2); // Default 2% risk
  const [executionType, setExecutionType] = useState<'market' | 'limit'>('market');
  const [notes, setNotes] = useState('');
  const [isCalculating, setIsCalculating] = useState(false);

  // Calculate risk metrics
  const calculateRiskMetrics = () => {
    if (!signal.entry_price || !signal.stop_loss) return null;

    const entryPrice = signal.entry_price;
    const stopLoss = signal.stop_loss;
    const riskPerUnit = Math.abs(entryPrice - stopLoss);
    const riskAmount = (positionSize * maxRisk) / 100;
    const maxPositionSize = riskAmount / riskPerUnit;
    const actualPositionSize = Math.min(positionSize, maxPositionSize);
    const actualRiskAmount = actualPositionSize * riskPerUnit;

    return {
      riskPerUnit,
      riskAmount: actualRiskAmount,
      maxPositionSize,
      actualPositionSize,
      riskPercentage: (actualRiskAmount / positionSize) * 100
    };
  };

  const riskMetrics = calculateRiskMetrics();

  const getTakeProfits = () => {
    return [
      signal.take_profit_1,
      signal.take_profit_2,
      signal.take_profit_3,
      signal.take_profit_4
    ].filter(Boolean);
  };

  const calculatePotentialProfit = (tpLevel: number) => {
    if (!signal.entry_price || !riskMetrics) return 0;
    
    const profitPerUnit = Math.abs(tpLevel - signal.entry_price);
    return riskMetrics.actualPositionSize * profitPerUnit;
  };

  const getRiskRewardRatio = (tpLevel: number) => {
    if (!riskMetrics) return 0;
    const profit = calculatePotentialProfit(tpLevel);
    return profit / riskMetrics.riskAmount;
  };

  const handleExecute = () => {
    if (!riskMetrics) return;

    const executionData: ExecutionData = {
      signalId: signal.signal_id,
      symbol: signal.symbol,
      direction: signal.signal_direction as 'long' | 'short',
      entryPrice: signal.entry_price || 0,
      stopLoss: signal.stop_loss || 0,
      takeProfits: getTakeProfits(),
      positionSize: riskMetrics.actualPositionSize,
      riskAmount: riskMetrics.riskAmount,
      maxRisk: maxRisk,
      executionType: executionType,
      notes: notes
    };

    onExecute(executionData);
  };

  const isSureShot = signal.confidence_score >= 0.85;

  return (
    <div className={`bg-gray-900 rounded-lg p-6 border border-gray-800 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${isSureShot ? 'bg-green-500/20' : 'bg-blue-500/20'}`}>
            {signal.signal_direction === 'long' ? (
              <TrendingUp className="h-6 w-6 text-green-400" />
            ) : (
              <TrendingDown className="h-6 w-6 text-red-400" />
            )}
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Execute Signal</h2>
            <p className="text-gray-400 text-sm">{signal.symbol} • {signal.timeframe}</p>
          </div>
        </div>
        
        {isSureShot && (
          <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white text-xs px-3 py-1 rounded-full flex items-center space-x-1">
            <Zap className="h-3 w-3" />
            <span>🎯 SURE SHOT</span>
          </div>
        )}
      </div>

      {/* Signal Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-sm text-gray-400">Entry Price</div>
          <div className="text-lg font-bold text-white">
            ${signal.entry_price?.toFixed(4) || 'N/A'}
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-sm text-gray-400">Stop Loss</div>
          <div className="text-lg font-bold text-red-400">
            ${signal.stop_loss?.toFixed(4) || 'N/A'}
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-sm text-gray-400">Confidence</div>
          <div className={`text-lg font-bold ${isSureShot ? 'text-green-400' : 'text-blue-400'}`}>
            {(signal.confidence_score * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-3 text-center">
          <div className="text-sm text-gray-400">R:R Ratio</div>
          <div className="text-lg font-bold text-blue-400">
            {signal.risk_reward_ratio?.toFixed(2) || 'N/A'}:1
          </div>
        </div>
      </div>

      {/* Position Sizing */}
      <div className="mb-6">
        <h3 className="text-white font-semibold mb-4 flex items-center space-x-2">
          <Calculator className="h-4 w-4 text-blue-400" />
          <span>Position Sizing & Risk Management</span>
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Position Size ($)</label>
            <input
              type="number"
              value={positionSize}
              onChange={(e) => setPositionSize(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
              min="100"
              max="100000"
              step="100"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-400 mb-2">Max Risk (%)</label>
            <input
              type="number"
              value={maxRisk}
              onChange={(e) => setMaxRisk(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
              min="0.5"
              max="10"
              step="0.5"
            />
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      {riskMetrics && (
        <div className="mb-6">
          <h3 className="text-white font-semibold mb-4 flex items-center space-x-2">
            <Shield className="h-4 w-4 text-yellow-400" />
            <span>Risk Metrics</span>
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <div className="text-sm text-gray-400">Risk Amount</div>
              <div className="text-lg font-bold text-red-400">
                ${riskMetrics.riskAmount.toFixed(2)}
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <div className="text-sm text-gray-400">Actual Position</div>
              <div className="text-lg font-bold text-white">
                ${riskMetrics.actualPositionSize.toFixed(2)}
              </div>
            </div>
            
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <div className="text-sm text-gray-400">Risk %</div>
              <div className="text-lg font-bold text-yellow-400">
                {riskMetrics.riskPercentage.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Take Profit Analysis */}
      <div className="mb-6">
        <h3 className="text-white font-semibold mb-4">Take Profit Analysis</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {getTakeProfits().map((tp, index) => (
            <div key={index} className="bg-gray-800 rounded-lg p-3 text-center">
              <div className="text-sm text-gray-400">TP{index + 1}</div>
              <div className="text-lg font-bold text-green-400">
                ${tp?.toFixed(4) || 'N/A'}
              </div>
              <div className="text-xs text-gray-500">
                Profit: ${calculatePotentialProfit(tp || 0).toFixed(2)}
              </div>
              <div className="text-xs text-blue-400">
                R:R: {getRiskRewardRatio(tp || 0).toFixed(2)}:1
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Execution Settings */}
      <div className="mb-6">
        <h3 className="text-white font-semibold mb-4 flex items-center space-x-2">
          <Settings className="h-4 w-4 text-gray-400" />
          <span>Execution Settings</span>
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Execution Type</label>
            <select
              value={executionType}
              onChange={(e) => setExecutionType(e.target.value as 'market' | 'limit')}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
            >
              <option value="market">Market Order</option>
              <option value="limit">Limit Order</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-400 mb-2">Notes (Optional)</label>
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add execution notes..."
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-4">
        <button
          onClick={handleExecute}
          disabled={!riskMetrics || isCalculating}
          className={`flex-1 py-3 px-4 rounded-lg font-bold transition-all duration-200 flex items-center justify-center space-x-2 ${
            isSureShot 
              ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white hover:scale-105 transform' 
              : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white hover:scale-105 transform'
          } disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100`}
        >
          {isCalculating ? (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          <span>{isSureShot ? 'Execute Sure Shot' : 'Execute Signal'}</span>
        </button>
        
        <button
          onClick={onCancel}
          className="px-6 py-3 border border-gray-600 text-gray-300 hover:text-white hover:border-gray-500 rounded-lg transition-all duration-200 hover:bg-gray-700"
        >
          <Square className="h-4 w-4" />
        </button>
      </div>

      {/* Risk Warning */}
      <div className="mt-4 p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
        <div className="flex items-center space-x-2">
          <AlertTriangle className="h-4 w-4 text-yellow-400" />
          <span className="text-yellow-400 text-sm font-medium">Risk Warning</span>
        </div>
        <p className="text-yellow-300 text-xs mt-1">
          Trading involves risk. Only trade with money you can afford to lose. 
          This signal is for educational purposes and not financial advice.
        </p>
      </div>
    </div>
  );
};

export default SignalExecution;
