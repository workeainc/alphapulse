import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Wallet, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target,
  BarChart3,
  Trophy,
  Users,
  Zap,
  Play,
  Pause,
  RotateCcw
} from 'lucide-react';

interface PaperTrade {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  status: 'open' | 'closed' | 'hit_sl' | 'hit_tp';
  entryTime: string;
  exitTime?: string;
  pnl?: number;
  pnlPercentage?: number;
}

interface PaperTradingProps {
  signals: any[];
  onExecutePaperTrade: (trade: Partial<PaperTrade>) => void;
}

export const PaperTrading: React.FC<PaperTradingProps> = ({
  signals,
  onExecutePaperTrade
}) => {
  const [paperTrades, setPaperTrades] = useState<PaperTrade[]>([]);
  const [virtualBalance, setVirtualBalance] = useState(10000);
  const [isActive, setIsActive] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'trades' | 'leaderboard'>('overview');

  const getTotalPnL = () => {
    return paperTrades
      .filter(trade => trade.pnl !== undefined)
      .reduce((sum, trade) => sum + (trade.pnl || 0), 0);
  };

  const getWinRate = () => {
    const closedTrades = paperTrades.filter(trade => trade.status !== 'open');
    const winningTrades = closedTrades.filter(trade => (trade.pnl || 0) > 0);
    return closedTrades.length > 0 ? (winningTrades.length / closedTrades.length) * 100 : 0;
  };

  const getROI = () => {
    const totalPnL = getTotalPnL();
    return ((totalPnL / 10000) * 100).toFixed(2);
  };

  const handleExecuteTrade = (signal: any) => {
    const trade: Partial<PaperTrade> = {
      id: Date.now().toString(),
      symbol: signal.symbol,
      direction: signal.direction,
      entryPrice: signal.entry_price || 0,
      quantity: 100, // Default quantity
      status: 'open',
      entryTime: new Date().toISOString(),
    };
    
    setPaperTrades([trade as PaperTrade, ...paperTrades]);
    onExecutePaperTrade(trade);
  };

  const resetPaperTrading = () => {
    setPaperTrades([]);
    setVirtualBalance(10000);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Wallet className="w-6 h-6 text-green-500" />
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Paper Trading</h2>
              <p className="text-sm text-gray-600">Risk-free signal testing</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsActive(!isActive)}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                isActive 
                  ? 'bg-green-100 text-green-700' 
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              {isActive ? (
                <>
                  <Pause className="w-4 h-4 inline mr-1" />
                  Active
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 inline mr-1" />
                  Paused
                </>
              )}
            </button>
            <button
              onClick={resetPaperTrading}
              className="p-2 text-gray-400 hover:text-gray-600 transition-colors"
              title="Reset Paper Trading"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        <button
          onClick={() => setActiveTab('overview')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'overview'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => setActiveTab('trades')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'trades'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Trades
        </button>
        <button
          onClick={() => setActiveTab('leaderboard')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'leaderboard'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Leaderboard
        </button>
      </div>

      {/* Content */}
      <div className="p-4">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Virtual Wallet */}
            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-4 border border-green-200">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Virtual Wallet</h3>
                  <p className="text-sm text-gray-600">Risk-free trading balance</p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-gray-900">
                    ${virtualBalance.toLocaleString()}
                  </div>
                  <div className={`text-sm font-medium ${getTotalPnL() >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {getTotalPnL() >= 0 ? '+' : ''}${getTotalPnL().toFixed(2)} ({getROI()}%)
                  </div>
                </div>
              </div>
            </div>

            {/* Performance Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <BarChart3 className="w-5 h-5 text-blue-500" />
                  <span className="text-sm font-medium text-gray-700">Win Rate</span>
                </div>
                <div className="text-2xl font-bold text-gray-900">{getWinRate().toFixed(1)}%</div>
                <div className="text-xs text-gray-500">
                  {paperTrades.filter(t => t.status !== 'open').length} closed trades
                </div>
              </div>
              
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  <span className="text-sm font-medium text-gray-700">Total Trades</span>
                </div>
                <div className="text-2xl font-bold text-gray-900">{paperTrades.length}</div>
                <div className="text-xs text-gray-500">
                  {paperTrades.filter(t => t.status === 'open').length} open
                </div>
              </div>
              
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-sm font-medium text-gray-700">Avg Return</span>
                </div>
                <div className="text-2xl font-bold text-gray-900">{getROI()}%</div>
                <div className="text-xs text-gray-500">Since start</div>
              </div>
            </div>

            {/* Recent Signals */}
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Recent Signals</h3>
              <div className="space-y-2">
                {signals.slice(0, 3).map((signal, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      {signal.direction === 'long' ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                      )}
                      <div>
                        <div className="font-medium text-gray-900">{signal.symbol}</div>
                        <div className="text-sm text-gray-600">{signal.pattern_type}</div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="text-sm text-gray-600">
                        ${signal.entry_price?.toFixed(4) || 'N/A'}
                      </div>
                      <button
                        onClick={() => handleExecuteTrade(signal)}
                        disabled={!isActive}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white text-sm rounded-md transition-colors"
                      >
                        Execute
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'trades' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Paper Trading History</h3>
            {paperTrades.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Wallet className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>No paper trades yet</p>
                <p className="text-sm">Execute your first signal to start paper trading</p>
              </div>
            ) : (
              <div className="space-y-2">
                {paperTrades.map((trade) => (
                  <div key={trade.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      {trade.direction === 'long' ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                      )}
                      <div>
                        <div className="font-medium text-gray-900">{trade.symbol}</div>
                        <div className="text-sm text-gray-600">
                          {trade.direction.toUpperCase()} • {trade.quantity} units
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-600">
                        Entry: ${trade.entryPrice.toFixed(4)}
                      </div>
                      {trade.pnl !== undefined && (
                        <div className={`text-sm font-medium ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'leaderboard' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-900">Paper Trading Leaderboard</h3>
            <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-4 border border-yellow-200">
              <div className="flex items-center space-x-2 mb-2">
                <Trophy className="w-5 h-5 text-yellow-600" />
                <span className="font-medium text-gray-900">Coming Soon!</span>
              </div>
              <p className="text-sm text-gray-600">
                Compete with other traders in our paper trading leaderboard. 
                Track performance, share strategies, and climb the ranks.
              </p>
            </div>
            
            {/* Mock Leaderboard */}
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6 bg-yellow-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
                    1
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">TraderPro_2024</div>
                    <div className="text-sm text-gray-600">Win Rate: 78%</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-gray-900">+$2,450</div>
                  <div className="text-sm text-gray-600">+24.5% ROI</div>
                </div>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className="w-6 h-6 bg-gray-400 rounded-full flex items-center justify-center text-white text-xs font-bold">
                    2
                  </div>
                  <div>
                    <div className="font-medium text-gray-900">CryptoMaster</div>
                    <div className="text-sm text-gray-600">Win Rate: 72%</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-lg font-bold text-gray-900">+$1,890</div>
                  <div className="text-sm text-gray-600">+18.9% ROI</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
