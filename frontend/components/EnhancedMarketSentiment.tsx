import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { TrendingUp, TrendingDown, Minus, RefreshCw, AlertCircle } from 'lucide-react';
import { phase4Api, type SentimentSummary, type PricePrediction, type CrossAssetAnalysis, type ModelPerformance } from '../lib/phase4Api';

interface EnhancedMarketSentimentProps {
  symbol: string;
  timeHorizon?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export default function EnhancedMarketSentiment({ 
  symbol, 
  timeHorizon = '4h', 
  autoRefresh = true, 
  refreshInterval = 30000 
}: EnhancedMarketSentimentProps) {
  const [data, setData] = useState<{
    summary: SentimentSummary | null;
    prediction: PricePrediction | null;
    crossAsset: CrossAssetAnalysis | null;
    modelPerformance: ModelPerformance | null;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const comprehensiveData = await phase4Api.getComprehensiveSentiment(symbol, timeHorizon);
      setData(comprehensiveData);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch sentiment data');
      console.error('Error fetching sentiment data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();

    if (autoRefresh) {
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [symbol, timeHorizon, autoRefresh, refreshInterval]);

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
        return <TrendingUp className="h-5 w-5 text-green-600" />;
      case 'bearish':
        return <TrendingDown className="h-5 w-5 text-red-600" />;
      default:
        return <Minus className="h-5 w-5 text-gray-600" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish':
        return 'text-green-600';
      case 'bearish':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  if (loading && !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Market Sentiment - {symbol}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
            <span className="ml-2">Loading sentiment data...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Market Sentiment - {symbol}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8 text-red-600">
            <AlertCircle className="h-8 w-8 mr-2" />
            <div>
              <div className="font-medium">Error loading data</div>
              <div className="text-sm">{error}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const summary = data?.summary;
  const prediction = data?.prediction;
  const crossAsset = data?.crossAsset;
  const modelPerformance = data?.modelPerformance;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            Market Sentiment - {symbol}
            {summary && getSentimentIcon(summary.overall_sentiment)}
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </CardTitle>
        {lastUpdate && (
          <div className="text-xs text-muted-foreground">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Overall Sentiment */}
          {summary && (
            <div className="text-center p-4 border rounded-lg">
              <div className={`text-2xl font-bold ${getSentimentColor(summary.overall_sentiment)}`}>
                {summary.overall_sentiment.toUpperCase()}
              </div>
              <div className="text-sm text-muted-foreground">
                Overall Score: {formatPercentage(summary.sentiment_score)}
              </div>
              <div className="text-xs text-muted-foreground">
                Confidence: {formatPercentage(summary.confidence)}
              </div>
            </div>
          )}

          {/* Sentiment Sources */}
          {summary && (
            <div className="space-y-3">
              <h4 className="text-sm font-medium">Sentiment Sources</h4>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Social Media</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${summary.sources.social_media * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium">{formatPercentage(summary.sources.social_media)}</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">News Analysis</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ width: `${summary.sources.news * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium">{formatPercentage(summary.sources.news)}</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Technical Indicators</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-purple-600 h-2 rounded-full" 
                      style={{ width: `${summary.sources.technical * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium">{formatPercentage(summary.sources.technical)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Phase 4: Price Prediction */}
          {prediction && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium mb-3">Price Prediction ({prediction.time_horizon})</h4>
              <div className="text-center p-3 border rounded-lg bg-gradient-to-r from-blue-50 to-purple-50">
                <div className={`text-xl font-bold ${getSentimentColor(prediction.direction)}`}>
                  {prediction.direction.toUpperCase()} ({prediction.strength})
                </div>
                <div className="text-sm text-muted-foreground">
                  Probability: {formatPercentage(prediction.prediction_probability)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Confidence: {formatPercentage(prediction.confidence)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Sentiment Score: {prediction.sentiment_score.toFixed(3)}
                </div>
              </div>
            </div>
          )}

          {/* Phase 4: Cross-Asset Correlation */}
          {crossAsset && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium mb-3">Cross-Asset Correlation</h4>
              <div className="space-y-2">
                <div className="text-center p-2 border rounded bg-gray-50">
                  <div className="text-sm font-medium">Market Mood: {crossAsset.market_sentiment.market_mood}</div>
                  <div className="text-xs text-muted-foreground">
                    Overall Sentiment: {crossAsset.market_sentiment.overall_sentiment.toFixed(3)}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(crossAsset.correlations).map(([asset, correlation]) => (
                    <div key={asset} className="text-xs p-2 border rounded">
                      <div className="font-medium">{asset}</div>
                      <div className="text-muted-foreground">{correlation.toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Phase 4: Model Performance */}
          {modelPerformance && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium mb-3">Model Performance</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Total Predictions</span>
                  <span className="text-xs font-medium">{modelPerformance.total_predictions}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Accuracy</span>
                  <span className="text-xs font-medium">{formatPercentage(modelPerformance.accuracy)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Average Confidence</span>
                  <span className="text-xs font-medium">{formatPercentage(modelPerformance.average_confidence)}</span>
                </div>
                
                {/* Model Breakdown */}
                <div className="mt-3">
                  <div className="text-xs font-medium mb-2">Model Breakdown:</div>
                  {Object.entries(modelPerformance.model_breakdown).map(([model, metrics]) => (
                    <div key={model} className="text-xs p-1 border rounded mb-1">
                      <div className="font-medium">{model}</div>
                      <div className="text-muted-foreground">
                        Acc: {formatPercentage(metrics.accuracy)} | 
                        Conf: {formatPercentage(metrics.confidence)} | 
                        Used: {metrics.usage_count}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
