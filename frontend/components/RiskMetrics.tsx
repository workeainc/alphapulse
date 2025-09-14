import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { AlertTriangle, Shield, TrendingDown, Gauge, Activity, Layers, BarChart3 } from 'lucide-react';

interface RiskMetricsProps {
  data?: {
    var_95: number;
    max_drawdown: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    current_risk: 'low' | 'medium' | 'high';
    daily_loss_limit: number;
    position_size_limit: number;
    // Enhanced risk metrics
    liquidation_risk_score: number;
    margin_utilization: number;
    leverage_ratio: number;
    correlation_risk: number;
    volatility_risk: number;
    liquidity_risk: number;
    // Advanced analytics
    stress_test_results: {
      scenario_1: number;
      scenario_2: number;
      scenario_3: number;
    };
    risk_decomposition: {
      market_risk: number;
      leverage_risk: number;
      liquidity_risk: number;
      correlation_risk: number;
    };
  };
}

export default function RiskMetrics({ data }: RiskMetricsProps) {
  // Mock data for demonstration
  const mockData = {
    var_95: 2.5,
    max_drawdown: 8.2,
    sharpe_ratio: 1.85,
    sortino_ratio: 2.1,
    current_risk: 'medium' as const,
    daily_loss_limit: 500,
    position_size_limit: 1000,
    // Enhanced risk metrics
    liquidation_risk_score: 35,
    margin_utilization: 65.5,
    leverage_ratio: 2.8,
    correlation_risk: 0.45,
    volatility_risk: 0.28,
    liquidity_risk: 0.15,
    // Advanced analytics
    stress_test_results: {
      scenario_1: -12.5,
      scenario_2: -8.3,
      scenario_3: -15.7,
    },
    risk_decomposition: {
      market_risk: 45.2,
      leverage_risk: 28.7,
      liquidity_risk: 15.3,
      correlation_risk: 10.8,
    },
  };

  const riskData = data || mockData;

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low':
        return 'text-green-600';
      case 'medium':
        return 'text-yellow-600';
      case 'high':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case 'low':
        return <Shield className="h-5 w-5 text-green-600" />;
      case 'medium':
        return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
      case 'high':
        return <TrendingDown className="h-5 w-5 text-red-600" />;
      default:
        return <Shield className="h-5 w-5 text-gray-600" />;
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
    }).format(value);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Risk Metrics
          {getRiskIcon(riskData.current_risk)}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Current Risk Level */}
          <div className="text-center p-4 border rounded-lg">
            <div className={`text-2xl font-bold ${getRiskColor(riskData.current_risk)}`}>
              {riskData.current_risk.toUpperCase()} RISK
            </div>
            <div className="text-sm text-muted-foreground">
              Current Risk Assessment
            </div>
          </div>

          {/* Risk Metrics Grid */}
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 border rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{riskData.var_95}%</div>
              <div className="text-sm text-muted-foreground">VaR (95%)</div>
            </div>
            
            <div className="text-center p-3 border rounded-lg">
              <div className="text-2xl font-bold text-red-600">{riskData.max_drawdown}%</div>
              <div className="text-sm text-muted-foreground">Max Drawdown</div>
            </div>
            
            <div className="text-center p-3 border rounded-lg">
              <div className="text-2xl font-bold text-green-600">{riskData.sharpe_ratio}</div>
              <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
            </div>
            
            <div className="text-center p-3 border rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{riskData.sortino_ratio}</div>
              <div className="text-sm text-muted-foreground">Sortino Ratio</div>
            </div>
          </div>

          {/* Enhanced Risk Metrics */}
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Gauge className="h-4 w-4" />
              Advanced Risk Analytics
            </h4>
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center p-2 border rounded-lg">
                <div className={`text-lg font-bold ${
                  riskData.liquidation_risk_score > 70 ? 'text-red-600' : 
                  riskData.liquidation_risk_score > 40 ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {riskData.liquidation_risk_score}
                </div>
                <div className="text-xs text-muted-foreground">Liquidation Risk</div>
              </div>
              
              <div className="text-center p-2 border rounded-lg">
                <div className={`text-lg font-bold ${
                  riskData.margin_utilization > 80 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {riskData.margin_utilization.toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Margin Util.</div>
              </div>
              
              <div className="text-center p-2 border rounded-lg">
                <div className="text-lg font-bold text-blue-600">{riskData.leverage_ratio}x</div>
                <div className="text-xs text-muted-foreground">Leverage</div>
              </div>
              
              <div className="text-center p-2 border rounded-lg">
                <div className={`text-lg font-bold ${
                  riskData.correlation_risk > 0.7 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {(riskData.correlation_risk * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Correlation</div>
              </div>
              
              <div className="text-center p-2 border rounded-lg">
                <div className={`text-lg font-bold ${
                  riskData.volatility_risk > 0.3 ? 'text-orange-600' : 'text-green-600'
                }`}>
                  {(riskData.volatility_risk * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Volatility</div>
              </div>
              
              <div className="text-center p-2 border rounded-lg">
                <div className={`text-lg font-bold ${
                  riskData.liquidity_risk > 0.2 ? 'text-red-600' : 'text-green-600'
                }`}>
                  {(riskData.liquidity_risk * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-muted-foreground">Liquidity</div>
              </div>
            </div>
          </div>

          {/* Risk Decomposition */}
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Risk Decomposition
            </h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Market Risk</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-600 h-2 rounded-full" style={{width: `${riskData.risk_decomposition.market_risk}%`}}></div>
                  </div>
                  <span className="text-sm font-medium">{riskData.risk_decomposition.market_risk.toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Leverage Risk</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div className="bg-orange-600 h-2 rounded-full" style={{width: `${riskData.risk_decomposition.leverage_risk}%`}}></div>
                  </div>
                  <span className="text-sm font-medium">{riskData.risk_decomposition.leverage_risk.toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Liquidity Risk</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div className="bg-green-600 h-2 rounded-full" style={{width: `${riskData.risk_decomposition.liquidity_risk}%`}}></div>
                  </div>
                  <span className="text-sm font-medium">{riskData.risk_decomposition.liquidity_risk.toFixed(1)}%</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Correlation Risk</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div className="bg-purple-600 h-2 rounded-full" style={{width: `${riskData.risk_decomposition.correlation_risk}%`}}></div>
                  </div>
                  <span className="text-sm font-medium">{riskData.risk_decomposition.correlation_risk.toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Stress Test Results */}
          <div className="border-t pt-4">
            <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Stress Test Results
            </h4>
            <div className="grid grid-cols-3 gap-3">
              <div className="text-center p-2 border rounded-lg">
                <div className="text-lg font-bold text-red-600">{riskData.stress_test_results.scenario_1.toFixed(1)}%</div>
                <div className="text-xs text-muted-foreground">Scenario 1</div>
              </div>
              
              <div className="text-center p-2 border rounded-lg">
                <div className="text-lg font-bold text-orange-600">{riskData.stress_test_results.scenario_2.toFixed(1)}%</div>
                <div className="text-xs text-muted-foreground">Scenario 2</div>
              </div>
              
              <div className="text-center p-2 border rounded-lg">
                <div className="text-lg font-bold text-red-600">{riskData.stress_test_results.scenario_3.toFixed(1)}%</div>
                <div className="text-xs text-muted-foreground">Scenario 3</div>
              </div>
            </div>
          </div>

          {/* Risk Limits */}
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <span className="text-sm text-muted-foreground">Daily Loss Limit</span>
              <span className="font-medium text-red-600">{formatCurrency(riskData.daily_loss_limit)}</span>
            </div>
            
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <span className="text-sm text-muted-foreground">Position Size Limit</span>
              <span className="font-medium text-blue-600">{formatCurrency(riskData.position_size_limit)}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
