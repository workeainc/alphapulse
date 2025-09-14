import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { 
  Activity, 
  Bell, 
  Settings, 
  BarChart3, 
  TrendingUp,
  RefreshCw,
  Zap
} from 'lucide-react';
import { SignalFeed } from '../components/trading/SignalFeed';
import { SignalAnalysis } from '../components/trading/SignalAnalysis';
import { TradeLog } from '../components/trading/TradeLog';
import { HistoricalData } from '../components/trading/HistoricalData';
import { 
  SophisticatedSignalCard,
  ConfidenceThermometer,
  PairTimeframeSelectors,
  AnalysisPanels,
  SignalExecution
} from '../components/trading';
import { useLatestSignals } from '../lib/hooks';
import { Signal } from '../lib/api';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
});

const TradingDashboardContent: React.FC = () => {
  const { data: signalsData, isLoading, error, refetch } = useLatestSignals();
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  
  // Sophisticated component states
  const [selectedPair, setSelectedPair] = useState('BTCUSDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [currentConfidence, setCurrentConfidence] = useState(0.0);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(true);
  const [showSignalExecution, setShowSignalExecution] = useState(false);
  const [executionSignal, setExecutionSignal] = useState(null);

  const signals = signalsData || [];

  const handleSignalClick = (signal: Signal) => {
    setSelectedSignal(signal);
  };

  const handleCloseAnalysis = () => {
    setSelectedSignal(null);
  };

  const handleAddTrade = (tradeData: any) => {
    const newTrade = {
      id: Date.now().toString(),
      ...tradeData,
      status: 'open',
      entryTime: new Date().toISOString(),
    };
    setTrades([newTrade, ...trades]);
  };

  // Sophisticated component handlers
  const handleSignalSelect = (signal) => {
    setExecutionSignal(signal);
    setShowSignalExecution(true);
  };

  const handleSignalExecute = (executionData) => {
    console.log('Executing signal:', executionData);
    // In real implementation, this would call the trading API
    setShowSignalExecution(false);
    setExecutionSignal(null);
    // Add to trades
    handleAddTrade({
      symbol: executionData.symbol,
      direction: executionData.direction,
      entryPrice: executionData.entryPrice,
      stopLoss: executionData.stopLoss,
      takeProfits: executionData.takeProfits,
      positionSize: executionData.positionSize,
      riskAmount: executionData.riskAmount
    });
  };

  const handleSignalCancel = () => {
    setShowSignalExecution(false);
    setExecutionSignal(null);
  };

  const handleConfidenceThresholdReached = () => {
    console.log('Confidence threshold reached! Sure shot signal available.');
    // In real implementation, this would trigger notifications
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Zap className="w-8 h-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">AlphaPulse Trading</h1>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-gray-600">Live Trading</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => refetch()}
                className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
                title="Refresh Signals"
              >
                <RefreshCw className="w-5 h-5" />
              </button>
              
              <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors" title="Notifications">
                <Bell className="w-5 h-5" />
              </button>
              
              <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors" title="Settings">
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Loading State */}
        {isLoading && (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="mt-4 text-gray-600">Loading trading signals...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
            <h3 className="text-lg font-medium text-red-800 mb-2">Connection Error</h3>
            <p className="text-red-700 mb-4">
              Failed to load trading signals. Please check your connection.
            </p>
            <button
              onClick={() => refetch()}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Sophisticated Components Section */}
        {!isLoading && !error && (
          <div className="mb-8">
            {/* Sophisticated Pair and Timeframe Selectors */}
            <div className="mb-6">
              <PairTimeframeSelectors
                selectedPair={selectedPair}
                selectedTimeframe={selectedTimeframe}
                onPairChange={setSelectedPair}
                onTimeframeChange={setSelectedTimeframe}
                onRefresh={() => refetch()}
              />
            </div>

            {/* Sophisticated Confidence Thermometer */}
            <div className="mb-6">
              <ConfidenceThermometer
                currentConfidence={currentConfidence}
                targetConfidence={0.85}
                isBuilding={isAnalysisRunning}
                showThreshold={true}
                size="lg"
                onThresholdReached={handleConfidenceThresholdReached}
              />
            </div>

            {/* Sophisticated Analysis Panels */}
            <div className="mb-6">
              <AnalysisPanels
                selectedPair={selectedPair}
                selectedTimeframe={selectedTimeframe}
                autoRefresh={true}
                refreshInterval={5000}
              />
            </div>
          </div>
        )}

        {/* Dashboard Content */}
        {!isLoading && !error && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Column - Signal Feed */}
              <div className="lg:col-span-1">
                <SignalFeed
                  signals={signals}
                  onSignalClick={handleSignalClick}
                  selectedSignal={selectedSignal || undefined}
                />
              </div>

              {/* Middle Column - Signal Analysis */}
              <div className="lg:col-span-1">
                <SignalAnalysis
                  signal={selectedSignal}
                  onClose={handleCloseAnalysis}
                />
              </div>

              {/* Right Column - Trade Log */}
              <div className="lg:col-span-1">
                <TradeLog
                  trades={trades}
                  onAddTrade={handleAddTrade}
                />
              </div>
            </div>

            {/* Historical Data Section */}
            <div className="mt-8">
              <HistoricalData />
            </div>
          </>
        )}

        {/* Quick Stats */}
        {!isLoading && !error && (
          <div className="mt-8">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Activity className="w-5 h-5 text-blue-500" />
                  <span className="text-sm font-medium text-gray-700">Total Signals</span>
                </div>
                <p className="text-2xl font-bold text-gray-900 mt-1">{signals.length}</p>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <TrendingUp className="w-5 h-5 text-green-500" />
                  <span className="text-sm font-medium text-gray-700">Long Signals</span>
                </div>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {signals.filter(s => s.direction === 'long').length}
                </p>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-purple-500" />
                  <span className="text-sm font-medium text-gray-700">Avg Confidence</span>
                </div>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {signals.length > 0 
                    ? ((signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length) * 100).toFixed(1)
                    : '0'
                  }%
                </p>
              </div>
              
              <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-sm font-medium text-gray-700">Active Trades</span>
                </div>
                <p className="text-2xl font-bold text-gray-900 mt-1">
                  {trades.filter(t => t.status === 'open').length}
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Toast Notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />

      {/* Signal Execution Modal */}
      {showSignalExecution && executionSignal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <SignalExecution
              signal={executionSignal}
              onExecute={handleSignalExecute}
              onCancel={handleSignalCancel}
            />
          </div>
        </div>
      )}
    </div>
  );
};

const TradingDashboard: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <TradingDashboardContent />
    </QueryClientProvider>
  );
};

export default TradingDashboard;
