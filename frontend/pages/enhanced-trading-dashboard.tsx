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
  Zap,
  Home,
  Clock,
  History,
  Wallet,
  Target,
  Shield,
  Users,
  Award
} from 'lucide-react';
import { EnhancedSignalCard } from '../components/trading/EnhancedSignalCard';
import { SignalAnalysis } from '../components/trading/SignalAnalysis';
import { TradeLog } from '../components/trading/TradeLog';
import { PaperTrading } from '../components/trading/PaperTrading';
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

const EnhancedTradingDashboardContent: React.FC = () => {
  const { data: signalsData, isLoading, error, refetch } = useLatestSignals();
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
  const [trades, setTrades] = useState<any[]>([]);
  const [activeView, setActiveView] = useState<'signals' | 'paper-trading' | 'history' | 'analytics'>('signals');

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

  const handleExecutePaperTrade = (tradeData: any) => {
    // Handle paper trade execution
    console.log('Paper trade executed:', tradeData);
  };

  const sidebarItems = [
    { id: 'signals', label: 'Live Signals', icon: Activity, active: activeView === 'signals' },
    { id: 'paper-trading', label: 'Paper Trading', icon: Wallet, active: activeView === 'paper-trading' },
    { id: 'history', label: 'Trade History', icon: History, active: activeView === 'history' },
    { id: 'analytics', label: 'Analytics', icon: BarChart3, active: activeView === 'analytics' },
  ];

  const renderMainContent = () => {
    switch (activeView) {
      case 'signals':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Enhanced Signal Feed */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200">
                <div className="p-4 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-semibold text-gray-900">Live Signal Feed</h2>
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                      <span className="text-sm text-gray-600">Real-time</span>
                    </div>
                  </div>
                </div>
                <div className="p-4">
                  {signals.length === 0 ? (
                    <div className="text-center py-12 text-gray-500">
                      <Activity className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                      <p>No signals available</p>
                      <p className="text-sm">New signals will appear here in real-time</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      {signals.map((signal, index) => (
                        <EnhancedSignalCard
                          key={`${signal.symbol}-${signal.timestamp}`}
                          signal={signal}
                          onSignalClick={handleSignalClick}
                          isSelected={selectedSignal?.symbol === signal.symbol && selectedSignal?.timestamp === signal.timestamp}
                          historicalAccuracy={0.75 + (Math.random() * 0.2)} // Mock data
                          similarTradesCount={Math.floor(Math.random() * 20) + 5} // Mock data
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column - Signal Analysis */}
            <div className="lg:col-span-1">
              <SignalAnalysis
                signal={selectedSignal}
                onClose={handleCloseAnalysis}
              />
            </div>
          </div>
        );

      case 'paper-trading':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PaperTrading
              signals={signals}
              onExecutePaperTrade={handleExecutePaperTrade}
            />
            <TradeLog
              trades={trades}
              onAddTrade={handleAddTrade}
            />
          </div>
        );

      case 'history':
        return (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Trade History</h2>
              <TradeLog
                trades={trades}
                onAddTrade={handleAddTrade}
              />
            </div>
          </div>
        );

      case 'analytics':
        return (
          <div className="space-y-6">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Trading Analytics</h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <TrendingUp className="w-5 h-5 text-blue-500" />
                    <span className="text-sm font-medium text-gray-700">Total Signals</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">{signals.length}</div>
                </div>
                
                <div className="bg-green-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Target className="w-5 h-5 text-green-500" />
                    <span className="text-sm font-medium text-gray-700">Success Rate</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">78.5%</div>
                </div>
                
                <div className="bg-purple-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <BarChart3 className="w-5 h-5 text-purple-500" />
                    <span className="text-sm font-medium text-gray-700">Avg ROI</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">+12.3%</div>
                </div>
                
                <div className="bg-yellow-50 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <Award className="w-5 h-5 text-yellow-500" />
                    <span className="text-sm font-medium text-gray-700">Best Trade</span>
                  </div>
                  <div className="text-2xl font-bold text-gray-900">+45.2%</div>
                </div>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
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
                <h1 className="text-2xl font-bold text-gray-900">AlphaPulse Pro</h1>
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

      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
          <div className="p-4">
            <nav className="space-y-2">
              {sidebarItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActiveView(item.id as any)}
                  className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    item.active
                      ? 'bg-blue-50 text-blue-700 border border-blue-200'
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                  }`}
                >
                  <item.icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 p-6">
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

          {/* Dashboard Content */}
          {!isLoading && !error && renderMainContent()}
        </div>
      </div>

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
    </div>
  );
};

const EnhancedTradingDashboard: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <EnhancedTradingDashboardContent />
    </QueryClientProvider>
  );
};

export default EnhancedTradingDashboard;
