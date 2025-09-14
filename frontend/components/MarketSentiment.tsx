import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface MarketSentimentProps {
  data?: {
    overall_sentiment: 'bullish' | 'bearish' | 'neutral';
    sentiment_score: number;
    social_media_sentiment: number;
    news_sentiment: number;
    technical_indicators: number;
    // Phase 4 additions
    prediction?: {
      direction: 'bullish' | 'bearish' | 'neutral';
      probability: number;
      confidence: number;
      time_horizon: string;
    };
    cross_asset_correlation?: {
      correlations: Record<string, number>;
      market_mood: string;
    };
    model_performance?: {
      accuracy: number;
      confidence: number;
    };
  };
}

export default function MarketSentiment({ data }: MarketSentimentProps) {
  // Mock data for demonstration
  const mockData = {
    overall_sentiment: 'bullish' as const,
    sentiment_score: 0.75,
    social_media_sentiment: 0.68,
    news_sentiment: 0.82,
    technical_indicators: 0.71,
    prediction: {
      direction: 'bullish' as const,
      probability: 0.78,
      confidence: 0.82,
      time_horizon: '4h'
    },
    cross_asset_correlation: {
      correlations: {
        'BTC/USDT': 0.85,
        'ETH/USDT': 0.72,
        'ADA/USDT': 0.68
      },
      market_mood: 'bullish'
    },
    model_performance: {
      accuracy: 0.87,
      confidence: 0.79
    }
  };

  const sentimentData = data || mockData;

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

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Market Sentiment
          {getSentimentIcon(sentimentData.overall_sentiment)}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Overall Sentiment */}
          <div className="text-center p-4 border rounded-lg">
            <div className={`text-2xl font-bold ${getSentimentColor(sentimentData.overall_sentiment)}`}>
              {sentimentData.overall_sentiment.toUpperCase()}
            </div>
            <div className="text-sm text-muted-foreground">
              Overall Score: {formatPercentage(sentimentData.sentiment_score)}
            </div>
          </div>

          {/* Sentiment Breakdown */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Social Media</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full" 
                    style={{ width: `${sentimentData.social_media_sentiment * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">{formatPercentage(sentimentData.social_media_sentiment)}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">News Analysis</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-600 h-2 rounded-full" 
                    style={{ width: `${sentimentData.news_sentiment * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">{formatPercentage(sentimentData.news_sentiment)}</span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Technical Indicators</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-purple-600 h-2 rounded-full" 
                    style={{ width: `${sentimentData.technical_indicators * 100}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium">{formatPercentage(sentimentData.technical_indicators)}</span>
              </div>
            </div>
          </div>

          {/* Phase 4: Price Prediction */}
          {sentimentData.prediction && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium mb-3">Price Prediction ({sentimentData.prediction.time_horizon})</h4>
              <div className="text-center p-3 border rounded-lg bg-gradient-to-r from-blue-50 to-purple-50">
                <div className={`text-xl font-bold ${getSentimentColor(sentimentData.prediction.direction)}`}>
                  {sentimentData.prediction.direction.toUpperCase()}
                </div>
                <div className="text-sm text-muted-foreground">
                  Probability: {formatPercentage(sentimentData.prediction.probability)}
                </div>
                <div className="text-xs text-muted-foreground">
                  Confidence: {formatPercentage(sentimentData.prediction.confidence)}
                </div>
              </div>
            </div>
          )}

          {/* Phase 4: Cross-Asset Correlation */}
          {sentimentData.cross_asset_correlation && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium mb-3">Cross-Asset Correlation</h4>
              <div className="space-y-2">
                <div className="text-center p-2 border rounded bg-gray-50">
                  <div className="text-sm font-medium">Market Mood: {sentimentData.cross_asset_correlation.market_mood}</div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(sentimentData.cross_asset_correlation.correlations).map(([asset, correlation]) => (
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
          {sentimentData.model_performance && (
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium mb-3">Model Performance</h4>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Accuracy</span>
                  <span className="text-xs font-medium">{formatPercentage(sentimentData.model_performance.accuracy)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Confidence</span>
                  <span className="text-xs font-medium">{formatPercentage(sentimentData.model_performance.confidence)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
