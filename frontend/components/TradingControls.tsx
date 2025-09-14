import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Play, Square, Settings, AlertTriangle } from 'lucide-react';

interface TradingControlsProps {
  isRunning?: boolean;
  onStartTrading?: () => void;
  onStopTrading?: () => void;
  onSettings?: () => void;
}

export default function TradingControls({ 
  isRunning = false, 
  onStartTrading, 
  onStopTrading, 
  onSettings 
}: TradingControlsProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Trading Controls
          {isRunning && <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Status Display */}
          <div className="text-center p-4 border rounded-lg">
            <div className={`text-lg font-semibold ${isRunning ? 'text-green-600' : 'text-red-600'}`}>
              {isRunning ? 'TRADING ACTIVE' : 'TRADING STOPPED'}
            </div>
            <div className="text-sm text-muted-foreground">
              {isRunning ? 'Bot is actively trading' : 'Bot is not trading'}
            </div>
          </div>

          {/* Control Buttons */}
          <div className="grid grid-cols-2 gap-3">
            {!isRunning ? (
              <button
                onClick={onStartTrading}
                className="flex items-center justify-center gap-2 p-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                <Play className="h-4 w-4" />
                Start Trading
              </button>
            ) : (
              <button
                onClick={onStopTrading}
                className="flex items-center justify-center gap-2 p-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                <Square className="h-4 w-4" />
                Stop Trading
              </button>
            )}
            
            <button
              onClick={onSettings}
              className="flex items-center justify-center gap-2 p-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Settings className="h-4 w-4" />
              Settings
            </button>
          </div>

          {/* Warning */}
          <div className="flex items-center gap-2 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
            <AlertTriangle className="h-4 w-4 text-yellow-600" />
            <span className="text-sm text-yellow-800">
              Trading involves risk. Ensure proper risk management is configured.
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
