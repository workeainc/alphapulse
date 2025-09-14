import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity, 
  Target, 
  Zap,
  BarChart3,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Clock,
  Volume2,
  ArrowUpRight,
  ArrowDownRight,
  Eye,
  EyeOff,
  Settings,
  Maximize2,
  Minimize2,
  Brain,
  Shield,
  Gauge,
  Layers,
  Play,
  Square
} from 'lucide-react';
import AdvancedTechnicalAnalysis from '../components/AdvancedTechnicalAnalysis';
import { useSignals, useMarketData, usePerformanceMetrics } from '../lib/hooks';
import { 
  SophisticatedSignalCard,
  ConfidenceThermometer,
  PairTimeframeSelectors,
  AnalysisPanels,
  SignalExecution
} from '../components/trading';

export default function Dashboard() {
  // Single-pair focused state
  const [selectedPair, setSelectedPair] = useState('BTCUSDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [currentConfidence, setCurrentConfidence] = useState(0.0);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // New sophisticated component states
  const [showSignalExecution, setShowSignalExecution] = useState(false);
  const [selectedSignal, setSelectedSignal] = useState(null);
  const [showAnalysisDetails, setShowAnalysisDetails] = useState(false);

  // Custom hooks for data fetching
  const { signals, isLoading: signalsLoading, error: signalsError } = useSignals();
  const { marketData, isLoading: marketLoading, error: marketError } = useMarketData();
  const { performanceMetrics, isLoading: performanceLoading, error: performanceError } = usePerformanceMetrics();

  // Simulate confidence building for demo
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentConfidence(prev => {
        const newConfidence = Math.min(prev + Math.random() * 0.05, 0.95);
        return newConfidence;
      });
    }, 2000);

    return () => clearInterval(interval);
  }, [selectedPair]);

  // Filter signals for selected pair
  const pairSignals = signals?.filter(signal => signal.symbol === selectedPair) || [];
  const highConfidenceSignals = pairSignals.filter(signal => signal.confidence >= 0.85);
  const activeSignal = pairSignals.find(signal => signal.status === 'active') || null;

  const handleRefresh = () => {
    // Trigger refresh for all data
    window.location.reload();
  };

  // New sophisticated component handlers
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

  const getSignalColor = (type: string) => {
    return type === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.8) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className={`min-h-screen bg-gray-950 text-white ${isFullscreen ? 'fixed inset-0 z-50' : ''}`}>
      <Head>
        <title>AlphaPulse - Sophisticated Trading Interface</title>
        <meta name="description" content="AI-powered single-pair trading with 85% confidence threshold and 4-TP system" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="container mx-auto px-4 py-6">
        {/* Sophisticated Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <Brain className="h-8 w-8 text-blue-500" />
              <div>
                <h1 className="text-2xl font-bold text-white">AlphaPulse</h1>
                <p className="text-gray-400 text-sm">Sophisticated Trading Interface</p>
              </div>
            </div>
            
            {/* Sophisticated Pair and Timeframe Selectors */}
            <PairTimeframeSelectors
              selectedPair={selectedPair}
              selectedTimeframe={selectedTimeframe}
              onPairChange={setSelectedPair}
              onTimeframeChange={setSelectedTimeframe}
              onRefresh={handleRefresh}
              className="ml-8"
            />
          </div>
          
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={signalsLoading || marketLoading}
              className="border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              <RefreshCw className={`h-4 w-4 ${signalsLoading || marketLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="border-gray-700 text-gray-300 hover:bg-gray-800"
            >
              {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </Button>
          </div>
        </div>

        {/* Sophisticated Confidence Thermometer */}
        <div className="mb-8">
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
        <div className="mb-8">
          <AnalysisPanels
            selectedPair={selectedPair}
            selectedTimeframe={selectedTimeframe}
            autoRefresh={true}
            refreshInterval={5000}
          />
        </div>
        {/* Sophisticated Single Signal Display */}
        <div className="mb-8">
          {activeSignal && activeSignal.confidence >= 0.85 ? (
            <SophisticatedSignalCard
              signal={activeSignal}
              isSureShot={true}
              showDetails={showAnalysisDetails}
              onToggleDetails={() => setShowAnalysisDetails(!showAnalysisDetails)}
              onExecute={() => handleSignalSelect(activeSignal)}
              onAddToWatchlist={() => console.log('Added to watchlist:', activeSignal)}
            />
          ) : (
            <Card className="bg-gray-900 border-gray-800">
              <CardContent className="p-8 text-center">
                <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Target className="h-8 w-8 text-gray-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">Waiting for Sure Shot</h3>
                <p className="text-gray-400 mb-4">
                  System is analyzing {selectedPair} on {selectedTimeframe} timeframe
                </p>
                <p className="text-gray-500 text-sm">
                  Current confidence: {(currentConfidence * 100).toFixed(1)}% (Target: 85%+)
                </p>
              </CardContent>
            </Card>
          )}
        </div>
        {/* Technical Analysis Chart */}
        <div className="mb-8">
          <Card className="bg-gray-900 border-gray-800">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2 text-white">
                <BarChart3 className="h-5 w-5 text-blue-500" />
                <span>Technical Analysis - {selectedPair}</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <AdvancedTechnicalAnalysis symbol={selectedPair} />
            </CardContent>
          </Card>
        </div>
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
    </div>
  );
}
