/**
 * Confidence Thermometer Component
 * Visual representation of confidence building with 85% threshold
 */

import React, { useEffect, useState, memo } from 'react';
import { Gauge, Target, Zap, TrendingUp, AlertTriangle } from 'lucide-react';
import { useConfidenceBuildingSimulation } from '../../lib/hooks_single_pair';
import { usePerformanceMonitor, useDebouncedValue } from '../../lib/performance';
import { ConfidenceThermometerAnimation, ProgressBarAnimation, CounterAnimation } from '../../lib/animations';

interface ConfidenceThermometerProps {
  selectedPair: string;
  selectedTimeframe: string;
  currentConfidence?: number;
  targetConfidence?: number;
  isBuilding?: boolean;
  showThreshold?: boolean;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  onThresholdReached?: () => void;
}

export const ConfidenceThermometer: React.FC<ConfidenceThermometerProps> = memo(({
  selectedPair,
  selectedTimeframe,
  currentConfidence: propCurrentConfidence,
  targetConfidence = 0.85,
  isBuilding: propIsBuilding,
  showThreshold = true,
  size = 'md',
  className = '',
  onThresholdReached
}) => {
  // Performance monitoring
  usePerformanceMonitor('ConfidenceThermometer');

  // Use the hook for real-time confidence data
  const {
    currentConfidence: hookCurrentConfidence,
    isBuilding: hookIsBuilding,
    thresholdReached,
    isLoading,
    error
  } = useConfidenceBuildingSimulation(selectedPair, selectedTimeframe);

  // Use prop values if provided, otherwise use hook values
  const currentConfidence = propCurrentConfidence ?? hookCurrentConfidence;
  const isBuilding = propIsBuilding ?? hookIsBuilding;

  // Debounce confidence value to prevent excessive re-renders
  const debouncedConfidence = useDebouncedValue(currentConfidence, 100);

  const [animatedConfidence, setAnimatedConfidence] = useState(0);
  const [hasReachedThreshold, setHasReachedThreshold] = useState(false);

  // Size configurations
  const sizeConfig = {
    sm: {
      height: 'h-4',
      textSize: 'text-sm',
      iconSize: 'h-3 w-3',
      thresholdText: 'text-xs'
    },
    md: {
      height: 'h-6',
      textSize: 'text-lg',
      iconSize: 'h-4 w-4',
      thresholdText: 'text-sm'
    },
    lg: {
      height: 'h-8',
      textSize: 'text-2xl',
      iconSize: 'h-6 w-6',
      thresholdText: 'text-base'
    }
  };

  const config = sizeConfig[size];

  // Animate confidence changes
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedConfidence(currentConfidence);
      
      // Check if threshold is reached
      if (currentConfidence >= targetConfidence && !hasReachedThreshold) {
        setHasReachedThreshold(true);
        if (onThresholdReached) {
          onThresholdReached();
        }
      } else if (currentConfidence < targetConfidence) {
        setHasReachedThreshold(false);
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [currentConfidence, targetConfidence, hasReachedThreshold, onThresholdReached]);

  const getConfidenceColor = () => {
    if (animatedConfidence >= targetConfidence) {
      return 'from-green-500 to-emerald-500';
    } else if (animatedConfidence >= 0.7) {
      return 'from-yellow-500 to-orange-500';
    } else {
      return 'from-red-500 to-pink-500';
    }
  };

  const getConfidenceIcon = () => {
    if (animatedConfidence >= targetConfidence) {
      return <Target className={`${config.iconSize} text-green-400`} />;
    } else if (animatedConfidence >= 0.7) {
      return <TrendingUp className={`${config.iconSize} text-yellow-400`} />;
    } else {
      return <AlertTriangle className={`${config.iconSize} text-red-400`} />;
    }
  };

  const getConfidenceText = () => {
    if (animatedConfidence >= targetConfidence) {
      return '🎯 SURE SHOT READY!';
    } else if (animatedConfidence >= 0.7) {
      return '⚡ Building Confidence';
    } else {
      return '📊 Analyzing...';
    }
  };

  return (
    <ConfidenceThermometerAnimation 
      confidence={debouncedConfidence} 
      threshold={targetConfidence}
    >
      <div className={`bg-gray-900 rounded-lg p-4 border border-gray-800 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <Gauge className={`${config.iconSize} text-blue-500`} />
          <div>
            <h3 className="text-white font-semibold">Confidence Building</h3>
            <p className="text-gray-400 text-sm">{getConfidenceText()}</p>
          </div>
        </div>
        
        <div className="text-right">
          <div className={`${config.textSize} font-bold text-white`}>
            {(animatedConfidence * 100).toFixed(1)}%
          </div>
          <div className="text-xs text-gray-400">Current</div>
        </div>
      </div>

      {/* Thermometer */}
      <div className="relative mb-4">
        <div className={`w-full ${config.height} bg-gray-800 rounded-full overflow-hidden`}>
          <div 
            className={`h-full bg-gradient-to-r ${getConfidenceColor()} transition-all duration-1000 ease-out`}
            style={{ width: `${Math.min(animatedConfidence * 100, 100)}%` }}
          />
          
          {/* Threshold Line */}
          {showThreshold && (
            <div 
              className="absolute top-0 w-0.5 h-full bg-white opacity-50"
              style={{ left: `${targetConfidence * 100}%` }}
            />
          )}
        </div>
        
        {/* Threshold Label */}
        {showThreshold && (
          <div 
            className="absolute top-0 text-white text-xs font-medium"
            style={{ left: `${targetConfidence * 100}%`, transform: 'translateX(-50%)' }}
          >
            {targetConfidence * 100}%
          </div>
        )}
      </div>

      {/* Status Indicators */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {getConfidenceIcon()}
          <span className={`text-sm font-medium ${
            animatedConfidence >= targetConfidence ? 'text-green-400' :
            animatedConfidence >= 0.7 ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {animatedConfidence >= targetConfidence ? 'Ready' :
             animatedConfidence >= 0.7 ? 'Building' : 'Analyzing'}
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            isBuilding ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
          }`} />
          <span className="text-xs text-gray-400">
            {isBuilding ? 'LIVE' : 'PAUSED'}
          </span>
        </div>
      </div>

      {/* Sure Shot Alert */}
      {hasReachedThreshold && (
        <div className="mt-4 p-3 bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 rounded-lg">
          <div className="flex items-center space-x-2">
            <Zap className="h-4 w-4 text-green-400" />
            <span className="text-green-400 font-medium text-sm">
              🎯 SURE SHOT SIGNAL AVAILABLE!
            </span>
          </div>
          <p className="text-green-300 text-xs mt-1">
            Confidence has reached {targetConfidence * 100}% threshold. Ready for execution.
          </p>
        </div>
      )}

      {/* Progress Details */}
      <div className="mt-4 pt-3 border-t border-gray-800">
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className="text-gray-400">Target:</span>
            <span className="text-white ml-1 font-medium">{targetConfidence * 100}%</span>
          </div>
          <div>
            <span className="text-gray-400">Progress:</span>
            <span className="text-white ml-1 font-medium">
              {((animatedConfidence / targetConfidence) * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
    </ConfidenceThermometerAnimation>
  );
});

ConfidenceThermometer.displayName = 'ConfidenceThermometer';

export default ConfidenceThermometer;
