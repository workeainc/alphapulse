/**
 * Intelligent AlphaPulse Dashboard
 * Advanced AI-powered trading signal dashboard with 85% confidence threshold
 */

import React, { useState } from 'react';
import { NextPage } from 'next';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import { 
  Brain, 
  Activity, 
  Shield, 
  Target, 
  BarChart3, 
  Settings,
  Play,
  Square,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Zap,
  Database,
  Cpu
} from 'lucide-react';
import { IntelligentSignalFeed } from '../components/intelligent/IntelligentSignalFeed';
import { 
  SophisticatedSignalCard,
  ConfidenceThermometer,
  PairTimeframeSelectors,
  AnalysisPanels,
  SignalExecution
} from '../components/trading';
import { 
  useIntelligentDashboardData,
  useIntelligentSystemStatus,
  useStartIntelligentSystem,
  useStopIntelligentSystem,
  useGenerateManualSignal
} from '../lib/hooks_intelligent';
import { formatConfidence, formatPercentage } from '../lib/api_intelligent';

const IntelligentDashboard: NextPage = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC/USDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  
  // Sophisticated component states
  const [showSignalExecution, setShowSignalExecution] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState(null);
  const [currentConfidence, setCurrentConfidence] = useState(0.0);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(true);
  
  // Data hooks
  const dashboardData = useIntelligentDashboardData();
  const systemStatus = useIntelligentSystemStatus();
  
  // Control hooks
  const startSystem = useStartIntelligentSystem();
  const stopSystem = useStopIntelligentSystem();
  const generateSignal = useGenerateManualSignal();

  const handleGenerateSignal = async () => {
    try {
      await generateSignal.mutateAsync({ 
        symbol: selectedSymbol, 
        timeframe: selectedTimeframe 
      });
    } catch (error) {
      console.error('Error generating signal:', error);
    }
  };

  // Sophisticated component handlers
  const handleSignalSelect = (signal) => {
    setSelectedSignal(signal);
    setShowSignalExecution(true);
  };

  const handleSignalExecute = (executionData) => {
    console.log('Executing signal:', executionData);
    // In real implementation, this would call the trading API
    setShowSignalExecution(false);
    setSelectedSignal(null);
  };

  const handleSignalCancel = () => {
    setShowSignalExecution(false);
    setSelectedSignal(null);
  };

  const handleConfidenceThresholdReached = () => {
    console.log('Confidence threshold reached! Sure shot signal available.');
    // In real implementation, this would trigger notifications
  };

  const getSystemStatusColor = () => {
    if (!systemStatus.data?.system?.status) return 'text-gray-400';
    return systemStatus.data.system.status === 'running' ? 'text-green-500' : 'text-red-500';
  };

  const getSystemStatusIcon = () => {
    if (!systemStatus.data?.system?.status) return <Square className="h-4 w-4" />;
    return systemStatus.data.system.status === 'running' ? 
      <Activity className="h-4 w-4" /> : <Square className="h-4 w-4" />;
  };

  return (
    <>
      <Head>
        <title>Intelligent AlphaPulse Dashboard</title>
        <meta name="description" content="AI-powered trading signals with 85% confidence threshold" />
      </Head>

      <div className="min-h-screen bg-gray-950 text-white">
        {/* Header */}
        <header className="bg-gray-900 border-b border-gray-800">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <Brain className="h-8 w-8 text-blue-500" />
                  <h1 className="text-xl font-bold">AlphaPulse Intelligent</h1>
                  <div className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                    AI-Powered
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                {/* System Status */}
                <div className="flex items-center space-x-2">
                  {getSystemStatusIcon()}
                  <span className={`text-sm font-medium ${getSystemStatusColor()}`}>
                    {systemStatus.data?.system?.status || 'Unknown'}
                  </span>
                </div>

                {/* Control Buttons */}
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => startSystem.mutate()}
                    disabled={startSystem.isPending || systemStatus.data?.system?.status === 'running'}
                    className="flex items-center space-x-1 px-3 py-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded text-sm font-medium transition-colors"
                  >
                    <Play className="h-3 w-3" />
                    <span>Start</span>
                  </button>
                  
                  <button
                    onClick={() => stopSystem.mutate()}
                    disabled={stopSystem.isPending || systemStatus.data?.system?.status !== 'running'}
                    className="flex items-center space-x-1 px-3 py-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded text-sm font-medium transition-colors"
                  >
                    <Square className="h-3 w-3" />
                    <span>Stop</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* System Overview Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {/* System Status Card */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">System Status</p>
                  <p className="text-2xl font-bold text-white">
                    {systemStatus.data?.system?.status || 'Unknown'}
                  </p>
                  <p className="text-gray-400 text-xs">
                    Version {systemStatus.data?.system?.version || 'N/A'}
                  </p>
                </div>
                <div className={`p-3 rounded-full ${getSystemStatusColor().replace('text-', 'bg-').replace('-500', '-500/20')}`}>
                  <Cpu className="h-6 w-6" />
                </div>
              </div>
            </div>

            {/* Signal Statistics Card */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Signals</p>
                  <p className="text-2xl font-bold text-white">
                    {dashboardData.signalStats?.total_signals || 0}
                  </p>
                  <p className="text-gray-400 text-xs">
                    {dashboardData.signalStats?.entry_signals || 0} Entry, {dashboardData.signalStats?.no_safe_entry_signals || 0} No Safe Entry
                  </p>
                </div>
                <div className="p-3 rounded-full bg-blue-500/20">
                  <BarChart3 className="h-6 w-6 text-blue-500" />
                </div>
              </div>
            </div>

            {/* High Confidence Signals Card */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">High Confidence</p>
                  <p className="text-2xl font-bold text-green-500">
                    {dashboardData.signalStats?.high_confidence_signals || 0}
                  </p>
                  <p className="text-gray-400 text-xs">
                    ≥85% Confidence
                  </p>
                </div>
                <div className="p-3 rounded-full bg-green-500/20">
                  <Shield className="h-6 w-6 text-green-500" />
                </div>
              </div>
            </div>

            {/* Average Confidence Card */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Avg Confidence</p>
                  <p className="text-2xl font-bold text-blue-500">
                    {dashboardData.signalStats?.average_confidence ? 
                      formatConfidence(dashboardData.signalStats.average_confidence) : 'N/A'}
                  </p>
                  <p className="text-gray-400 text-xs">
                    Overall System
                  </p>
                </div>
                <div className="p-3 rounded-full bg-blue-500/20">
                  <Target className="h-6 w-6 text-blue-500" />
                </div>
              </div>
            </div>
          </div>

          {/* Data Collection Status */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Data Collection Status */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center space-x-3 mb-4">
                <Database className="h-5 w-5 text-blue-500" />
                <h3 className="text-lg font-semibold">Data Collection</h3>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Status</span>
                  <span className={`text-sm font-medium ${
                    dashboardData.dataCollectionStatus?.is_running ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {dashboardData.dataCollectionStatus?.is_running ? 'Running' : 'Stopped'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Market Intelligence</span>
                  <span className="text-white text-sm">
                    {dashboardData.dataCollectionStatus?.market_intelligence_count || 0} records
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Volume Analysis</span>
                  <span className="text-white text-sm">
                    {dashboardData.dataCollectionStatus?.volume_analysis_count || 0} records
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Last Update</span>
                  <span className="text-white text-sm">
                    {dashboardData.dataCollectionStatus?.last_update ? 
                      new Date(dashboardData.dataCollectionStatus.last_update).toLocaleTimeString() : 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* Signal Generation Status */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center space-x-3 mb-4">
                <Zap className="h-5 w-5 text-yellow-500" />
                <h3 className="text-lg font-semibold">Signal Generation</h3>
              </div>
              
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Status</span>
                  <span className={`text-sm font-medium ${
                    dashboardData.signalStats?.is_running ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {dashboardData.signalStats?.is_running ? 'Running' : 'Stopped'}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Entry Signals</span>
                  <span className="text-green-500 text-sm">
                    {dashboardData.signalStats?.entry_signals || 0}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">No Safe Entry</span>
                  <span className="text-yellow-500 text-sm">
                    {dashboardData.signalStats?.no_safe_entry_signals || 0}
                  </span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Monitored Symbols</span>
                  <span className="text-white text-sm">
                    {dashboardData.signalStats?.monitored_symbols?.length || 0}
                  </span>
                </div>
              </div>
            </div>

            {/* Manual Signal Generation */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="flex items-center space-x-3 mb-4">
                <Target className="h-5 w-5 text-purple-500" />
                <h3 className="text-lg font-semibold">Manual Generation</h3>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-gray-400 text-sm mb-2">Symbol</label>
                  <select
                    value={selectedSymbol}
                    onChange={(e) => setSelectedSymbol(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm"
                  >
                    <option value="BTC/USDT">BTC/USDT</option>
                    <option value="ETH/USDT">ETH/USDT</option>
                    <option value="ADA/USDT">ADA/USDT</option>
                    <option value="SOL/USDT">SOL/USDT</option>
                    <option value="BNB/USDT">BNB/USDT</option>
                    <option value="XRP/USDT">XRP/USDT</option>
                    <option value="DOT/USDT">DOT/USDT</option>
                    <option value="LINK/USDT">LINK/USDT</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-gray-400 text-sm mb-2">Timeframe</label>
                  <select
                    value={selectedTimeframe}
                    onChange={(e) => setSelectedTimeframe(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm"
                  >
                    <option value="15m">15m</option>
                    <option value="1h">1h</option>
                    <option value="4h">4h</option>
                  </select>
                </div>
                
                <button
                  onClick={handleGenerateSignal}
                  disabled={generateSignal.isPending}
                  className="w-full flex items-center justify-center space-x-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                >
                  {generateSignal.isPending ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Zap className="h-4 w-4" />
                  )}
                  <span>{generateSignal.isPending ? 'Generating...' : 'Generate Signal'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* High Confidence Signals Feature Card */}
          <div className="mb-8">
            <div className="bg-gradient-to-r from-green-900/50 to-blue-900/50 rounded-lg p-6 border border-green-500/30">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-green-500/20 rounded-lg">
                    <Shield className="h-6 w-6 text-green-400" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white">High Confidence Signals</h2>
                    <p className="text-green-300 text-sm">85%+ Confidence Threshold - Premium Trading Opportunities</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm font-medium">LIVE</span>
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* High Confidence Signals List */}
                <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                    <Target className="h-5 w-5 text-green-400" />
                    <span>Available Signals</span>
                  </h3>
                  
                                    {(dashboardData.signalStats?.high_confidence_signals || 0) > 0 ? (
                    <div className="space-y-3">
                      {/* Show high confidence signals from the main signal feed */}
                      <div className="text-center py-4">
                        <p className="text-green-400 font-medium">
                          {dashboardData.signalStats?.high_confidence_signals || 0} High Confidence Signals Available
                        </p>
                        <p className="text-gray-400 text-sm mt-1">
                          Check the signal feed below for detailed analysis
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="w-16 h-16 bg-gray-800/50 rounded-full flex items-center justify-center mx-auto mb-4">
                        <Target className="h-8 w-8 text-gray-400" />
                      </div>
                      <p className="text-gray-400 mb-2">No High Confidence Signals Available</p>
                      <p className="text-gray-500 text-sm">System is monitoring for 85%+ confidence opportunities</p>
                    </div>
                  )}
                </div>
                
                {/* Signal Statistics */}
                <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5 text-blue-400" />
                    <span>Signal Statistics</span>
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Total Signals Generated</span>
                      <span className="text-white font-medium">
                        {dashboardData.signalStats?.total_signals || 0}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">High Confidence (85%+)</span>
                      <span className="text-green-400 font-medium">
                        {dashboardData.signalStats?.high_confidence_signals || 0}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Average Confidence</span>
                      <span className="text-blue-400 font-medium">
                        {dashboardData.signalStats?.average_confidence ? 
                          `${(dashboardData.signalStats.average_confidence * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Long Signals</span>
                      <span className="text-green-400 font-medium">
                        {dashboardData.signalStats?.long_signals || 0}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-gray-400">Short Signals</span>
                      <span className="text-red-400 font-medium">
                        {dashboardData.signalStats?.short_signals || 0}
                      </span>
                    </div>
                  </div>
                  
                  <div className="mt-6 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <p className="text-blue-300 text-sm">
                      <strong>💡 Tip:</strong> High confidence signals are generated when the system detects 
                      strong pattern formation, volume confirmation, and favorable market conditions.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Sophisticated Components Section */}
          <div className="mb-8">
            {/* Sophisticated Pair and Timeframe Selectors */}
            <div className="mb-6">
              <PairTimeframeSelectors
                selectedPair={selectedSymbol.replace('/', '')}
                selectedTimeframe={selectedTimeframe}
                onPairChange={(pair) => setSelectedSymbol(pair.replace('USDT', '/USDT'))}
                onTimeframeChange={setSelectedTimeframe}
                onRefresh={() => window.location.reload()}
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
                selectedPair={selectedSymbol.replace('/', '')}
                selectedTimeframe={selectedTimeframe}
                autoRefresh={true}
                refreshInterval={5000}
              />
            </div>
          </div>
          <div className="mb-8">
            <IntelligentSignalFeed 
              maxSignals={10}
              showNoSafeEntry={true}
              autoRefresh={true}
            />
          </div>

          {/* System Information */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="flex items-center space-x-3 mb-4">
              <Settings className="h-5 w-5 text-gray-400" />
              <h3 className="text-lg font-semibold">System Information</h3>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-400">Version</p>
                <p className="text-white font-medium">{systemStatus.data?.system?.version || 'N/A'}</p>
              </div>
              
              <div>
                <p className="text-gray-400">Startup Time</p>
                <p className="text-white font-medium">
                  {systemStatus.data?.system?.startup_time ? 
                    new Date(systemStatus.data.system.startup_time).toLocaleString() : 'N/A'}
                </p>
              </div>
              
              <div>
                <p className="text-gray-400">Database Status</p>
                <p className="text-white font-medium">{systemStatus.data?.database?.status || 'N/A'}</p>
              </div>
              
              <div>
                <p className="text-gray-400">Connection Pool</p>
                <p className="text-white font-medium">{systemStatus.data?.database?.pool_size || 0} connections</p>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Signal Execution Modal */}
      {showSignalExecution && selectedSignal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <SignalExecution
              signal={selectedSignal}
              onExecute={handleSignalExecute}
              onCancel={handleSignalCancel}
            />
          </div>
        </div>
      )}
    </>
  );
};

export default dynamic(() => Promise.resolve(IntelligentDashboard), {
  ssr: false,
});
