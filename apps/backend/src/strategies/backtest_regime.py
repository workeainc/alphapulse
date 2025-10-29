#!/usr/bin/env python3
"""
Market Regime Detection Backtesting Module
Backtesting, threshold optimization, and performance evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeMetrics, RegimeState

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtesting result"""
    accuracy: float
    stability_score: float
    avg_regime_duration: float
    regime_changes: int
    signal_filter_rate: float
    win_rate: float
    latency_ms: float
    thresholds: Dict[str, float]
    regime_distribution: Dict[str, int]

@dataclass
class OptimizationResult:
    """Optimization result"""
    best_thresholds: Dict[str, float]
    best_accuracy: float
    best_stability: float
    optimization_history: List[Dict[str, Any]]
    model_performance: Dict[str, float]

class RegimeBacktester:
    """
    Market Regime Detection Backtester
    Comprehensive backtesting with threshold optimization
    """
    
    def __init__(self, 
                 data_path: str,
                 symbol: str = 'BTC/USDT',
                 timeframe: str = '15m',
                 train_ratio: float = 0.7,
                 random_state: int = 42):
        """
        Initialize Regime Backtester
        
        Args:
            data_path: Path to historical CSV data
            symbol: Trading symbol
            timeframe: Timeframe
            train_ratio: Training data ratio
            random_state: Random seed
        """
        self.data_path = data_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_ratio = train_ratio
        self.random_state = random_state
        
        # Load and prepare data
        self.data = None
        self.train_data = None
        self.test_data = None
        self.indicators_data = None
        
        # Results storage
        self.backtest_results = []
        self.optimization_results = []
        
        logger.info(f"Regime Backtester initialized for {symbol} {timeframe}")
    
    def load_data(self) -> bool:
        """Load and prepare historical data"""
        try:
            # Load CSV data
            self.data = pd.read_csv(self.data_path)
            
            # Ensure required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Convert timestamp
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            self.data = self.data.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate technical indicators
            self.calculate_indicators()
            
            # Split data
            split_idx = int(len(self.data) * self.train_ratio)
            self.train_data = self.data.iloc[:split_idx]
            self.test_data = self.data.iloc[split_idx:]
            
            logger.info(f"Data loaded: {len(self.data)} candles, train: {len(self.train_data)}, test: {len(self.test_data)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def calculate_indicators(self):
        """Calculate technical indicators for regime detection"""
        try:
            # RSI
            delta = self.data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = self.data['close'].ewm(span=8).mean()
            ema26 = self.data['close'].ewm(span=24).mean()
            self.data['macd'] = ema12 - ema26
            self.data['macd_signal'] = self.data['macd'].ewm(span=9).mean()
            self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            self.data['bb_middle'] = self.data['close'].rolling(window=bb_period).mean()
            bb_std_dev = self.data['close'].rolling(window=bb_period).std()
            self.data['bb_upper'] = self.data['bb_middle'] + (bb_std_dev * bb_std)
            self.data['bb_lower'] = self.data['bb_middle'] - (bb_std_dev * bb_std)
            
            # ADX
            self.calculate_adx()
            
            # ATR
            self.calculate_atr()
            
            # Volume SMA
            self.data['volume_sma'] = self.data['volume'].rolling(window=20).mean()
            
            # Pivot Points
            self.calculate_pivot_points()
            
            # Fill NaN values
            self.data = self.data.fillna(method='bfill').fillna(0)
            
            logger.info("Technical indicators calculated")
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
    
    def calculate_adx(self):
        """Calculate ADX (Average Directional Index)"""
        try:
            # True Range
            self.data['tr1'] = self.data['high'] - self.data['low']
            self.data['tr2'] = abs(self.data['high'] - self.data['close'].shift(1))
            self.data['tr3'] = abs(self.data['low'] - self.data['close'].shift(1))
            self.data['tr'] = self.data[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Directional Movement
            self.data['dm_plus'] = np.where(
                (self.data['high'] - self.data['high'].shift(1)) > (self.data['low'].shift(1) - self.data['low']),
                np.maximum(self.data['high'] - self.data['high'].shift(1), 0),
                0
            )
            self.data['dm_minus'] = np.where(
                (self.data['low'].shift(1) - self.data['low']) > (self.data['high'] - self.data['high'].shift(1)),
                np.maximum(self.data['low'].shift(1) - self.data['low'], 0),
                0
            )
            
            # Smoothed values
            period = 14
            self.data['tr_smooth'] = self.data['tr'].rolling(window=period).mean()
            self.data['dm_plus_smooth'] = self.data['dm_plus'].rolling(window=period).mean()
            self.data['dm_minus_smooth'] = self.data['dm_minus'].rolling(window=period).mean()
            
            # DI+ and DI-
            self.data['di_plus'] = 100 * (self.data['dm_plus_smooth'] / self.data['tr_smooth'])
            self.data['di_minus'] = 100 * (self.data['dm_minus_smooth'] / self.data['tr_smooth'])
            
            # DX and ADX
            self.data['dx'] = 100 * abs(self.data['di_plus'] - self.data['di_minus']) / (self.data['di_plus'] + self.data['di_minus'])
            self.data['adx'] = self.data['dx'].rolling(window=period).mean()
            
        except Exception as e:
            logger.error(f"Failed to calculate ADX: {e}")
    
    def calculate_atr(self):
        """Calculate ATR (Average True Range)"""
        try:
            period = 14
            self.data['atr'] = self.data['tr'].rolling(window=period).mean()
        except Exception as e:
            logger.error(f"Failed to calculate ATR: {e}")
    
    def calculate_pivot_points(self):
        """Calculate pivot points"""
        try:
            # Pivot Point
            self.data['pivot'] = (self.data['high'] + self.data['low'] + self.data['close']) / 3
            
            # Support and Resistance
            self.data['r1'] = 2 * self.data['pivot'] - self.data['low']
            self.data['s1'] = 2 * self.data['pivot'] - self.data['high']
            
        except Exception as e:
            logger.error(f"Failed to calculate pivot points: {e}")
    
    def generate_synthetic_labels(self) -> pd.Series:
        """Generate synthetic regime labels for training"""
        labels = []
        
        for i in range(len(self.data)):
            adx = self.data.iloc[i]['adx']
            rsi = self.data.iloc[i]['rsi']
            bb_width = (self.data.iloc[i]['bb_upper'] - self.data.iloc[i]['bb_lower']) / self.data.iloc[i]['bb_middle']
            volume_ratio = self.data.iloc[i]['volume'] / self.data.iloc[i]['volume_sma']
            
            # Rule-based labeling
            if adx > 35 and rsi > 60:
                label = 0  # STRONG_TREND_BULL
            elif adx > 35 and rsi < 40:
                label = 1  # STRONG_TREND_BEAR
            elif adx > 25 and adx <= 35:
                label = 2  # WEAK_TREND
            elif adx < 25 and bb_width < 0.05:
                label = 3  # RANGING
            elif bb_width > 0.07 and volume_ratio > 1.5:
                label = 4  # VOLATILE_BREAKOUT
            else:
                label = 5  # CHOPPY
            
            labels.append(label)
        
        return pd.Series(labels, index=self.data.index)
    
    def backtest_regime_detector(self, 
                                thresholds: Dict[str, float],
                                enable_ml: bool = False) -> BacktestResult:
        """Backtest regime detector with given thresholds"""
        try:
            # Initialize detector
            detector = MarketRegimeDetector(
                symbol=self.symbol,
                timeframe=self.timeframe,
                lookback_period=10,
                min_regime_duration=5,
                hysteresis_threshold=0.2,
                enable_ml=enable_ml
            )
            
            # Update thresholds
            detector.thresholds.update(thresholds)
            
            # Run backtest
            regime_predictions = []
            regime_states = []
            latencies = []
            
            for i in range(len(self.test_data)):
                row = self.test_data.iloc[i]
                
                # Prepare indicators
                indicators = {
                    'adx': row['adx'],
                    'bb_upper': row['bb_upper'],
                    'bb_lower': row['bb_lower'],
                    'bb_middle': row['bb_middle'],
                    'atr': row['atr'],
                    'rsi': row['rsi'],
                    'volume_sma': row['volume_sma']
                }
                
                # Prepare candlestick
                candlestick = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                
                # Update regime
                start_time = datetime.now()
                regime_state = detector.update_regime(indicators, candlestick)
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                regime_predictions.append(regime_state.regime.value)
                regime_states.append(regime_state)
                latencies.append(latency)
            
            # Generate ground truth labels
            ground_truth = self.generate_synthetic_labels().iloc[len(self.train_data):]
            
            # Calculate metrics
            accuracy = self.calculate_accuracy(regime_predictions, ground_truth)
            stability_score = np.mean([state.stability_score for state in regime_states])
            avg_regime_duration = np.mean([state.duration_candles for state in regime_states])
            regime_changes = len([i for i in range(1, len(regime_predictions)) if regime_predictions[i] != regime_predictions[i-1]])
            
            # Calculate signal filtering metrics
            signal_filter_rate = self.calculate_signal_filter_rate(regime_states)
            win_rate = self.calculate_win_rate(regime_states)
            
            # Regime distribution
            regime_distribution = {}
            for regime in MarketRegime:
                regime_distribution[regime.value] = regime_predictions.count(regime.value)
            
            result = BacktestResult(
                accuracy=accuracy,
                stability_score=stability_score,
                avg_regime_duration=avg_regime_duration,
                regime_changes=regime_changes,
                signal_filter_rate=signal_filter_rate,
                win_rate=win_rate,
                latency_ms=np.mean(latencies),
                thresholds=thresholds,
                regime_distribution=regime_distribution
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None
    
    def calculate_accuracy(self, predictions: List[str], ground_truth: pd.Series) -> float:
        """Calculate regime classification accuracy"""
        try:
            # Convert predictions to numeric labels
            regime_map = {
                'strong_trend_bull': 0,
                'strong_trend_bear': 1,
                'weak_trend': 2,
                'ranging': 3,
                'volatile_breakout': 4,
                'choppy': 5
            }
            
            pred_labels = [regime_map.get(pred, 5) for pred in predictions]
            true_labels = ground_truth.values
            
            return accuracy_score(true_labels, pred_labels)
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy: {e}")
            return 0.0
    
    def calculate_signal_filter_rate(self, regime_states: List[RegimeState]) -> float:
        """Calculate signal filtering rate based on regimes"""
        try:
            filtered_signals = 0
            total_signals = 0
            
            for state in regime_states:
                # Simulate signal generation with random confidence
                signal_confidence = np.random.uniform(0.5, 0.95)
                total_signals += 1
                
                if state.regime == MarketRegime.CHOPPY and signal_confidence < 0.85:
                    filtered_signals += 1
                elif state.regime == MarketRegime.VOLATILE_BREAKOUT and signal_confidence < 0.75:
                    filtered_signals += 1
                elif state.regime in [MarketRegime.STRONG_TREND_BULL, MarketRegime.STRONG_TREND_BEAR] and signal_confidence < 0.65:
                    filtered_signals += 1
                elif signal_confidence < 0.70:
                    filtered_signals += 1
            
            return filtered_signals / total_signals if total_signals > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate signal filter rate: {e}")
            return 0.0
    
    def calculate_win_rate(self, regime_states: List[RegimeState]) -> float:
        """Calculate simulated win rate based on regime quality"""
        try:
            wins = 0
            total_trades = 0
            
            for state in regime_states:
                # Simulate trade outcome based on regime quality
                if state.regime in [MarketRegime.STRONG_TREND_BULL, MarketRegime.STRONG_TREND_BEAR]:
                    win_prob = 0.85
                elif state.regime == MarketRegime.WEAK_TREND:
                    win_prob = 0.75
                elif state.regime == MarketRegime.RANGING:
                    win_prob = 0.70
                elif state.regime == MarketRegime.VOLATILE_BREAKOUT:
                    win_prob = 0.80
                else:  # CHOPPY
                    win_prob = 0.60
                
                # Simulate trade
                if np.random.random() < 0.3:  # 30% chance of trade
                    total_trades += 1
                    if np.random.random() < win_prob:
                        wins += 1
            
            return wins / total_trades if total_trades > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate win rate: {e}")
            return 0.0
    
    def optimize_thresholds(self, n_trials: int = 100) -> OptimizationResult:
        """Optimize thresholds using Optuna"""
        try:
            def objective(trial):
                # Define parameter ranges
                thresholds = {
                    'adx_trend': trial.suggest_float('adx_trend', 20.0, 30.0),
                    'adx_strong_trend': trial.suggest_float('adx_strong_trend', 30.0, 40.0),
                    'ma_slope_bull': trial.suggest_float('ma_slope_bull', 0.00005, 0.0002),
                    'ma_slope_bear': trial.suggest_float('ma_slope_bear', -0.0002, -0.00005),
                    'bb_width_volatile': trial.suggest_float('bb_width_volatile', 0.03, 0.07),
                    'bb_width_breakout': trial.suggest_float('bb_width_breakout', 0.05, 0.10),
                    'rsi_overbought': trial.suggest_float('rsi_overbought', 55.0, 65.0),
                    'rsi_oversold': trial.suggest_float('rsi_oversold', 35.0, 45.0),
                    'volume_ratio_high': trial.suggest_float('volume_ratio_high', 1.2, 2.0),
                    'breakout_strength_high': trial.suggest_float('breakout_strength_high', 60.0, 80.0)
                }
                
                # Run backtest
                result = self.backtest_regime_detector(thresholds, enable_ml=False)
                
                if result is None:
                    return 0.0
                
                # Multi-objective optimization
                score = (result.accuracy * 0.4 + 
                        result.stability_score * 0.3 + 
                        result.win_rate * 0.3)
                
                return score
            
            # Create study
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Get best result
            best_thresholds = study.best_params
            best_result = self.backtest_regime_detector(best_thresholds, enable_ml=False)
            
            # Store optimization history
            optimization_history = []
            for trial in study.trials:
                optimization_history.append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
            
            optimization_result = OptimizationResult(
                best_thresholds=best_thresholds,
                best_accuracy=best_result.accuracy if best_result else 0.0,
                best_stability=best_result.stability_score if best_result else 0.0,
                optimization_history=optimization_history,
                model_performance={
                    'accuracy': best_result.accuracy if best_result else 0.0,
                    'stability': best_result.stability_score if best_result else 0.0,
                    'win_rate': best_result.win_rate if best_result else 0.0,
                    'latency_ms': best_result.latency_ms if best_result else 0.0
                }
            )
            
            logger.info(f"Threshold optimization completed. Best accuracy: {optimization_result.best_accuracy:.3f}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Threshold optimization failed: {e}")
            return None
    
    def train_ml_model(self) -> bool:
        """Train ML model for regime classification"""
        try:
            # Prepare features
            feature_columns = ['adx', 'rsi', 'volume_sma', 'atr']
            self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
            self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma']
            self.data['price_momentum'] = (self.data['close'] - self.data['open']) / self.data['open']
            self.data['volatility_score'] = self.data['atr'] / self.data['close']
            
            # Calculate MA slope
            self.data['ma_slope'] = self.data['close'].diff(10) / self.data['close'].shift(10)
            
            # Calculate breakout strength
            self.data['breakout_strength'] = (
                self.data['volume_ratio'] * 0.6 + 
                (self.data['atr'] / 1000) * 0.3 + 
                (self.data['adx'] > 25).astype(int) * 0.1
            ) * 100
            
            # Add all features
            feature_columns.extend(['bb_width', 'volume_ratio', 'price_momentum', 'volatility_score', 'ma_slope', 'breakout_strength'])
            
            # Generate labels
            labels = self.generate_synthetic_labels()
            
            # Prepare training data
            X = self.data[feature_columns].fillna(0)
            y = labels
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            model_path = f"models/regime_detector_{self.symbol.replace('/', '_')}_{self.timeframe}"
            Path("models").mkdir(exist_ok=True)
            
            joblib.dump(model, f"{model_path}_model.pkl")
            joblib.dump(scaler, f"{model_path}_scaler.pkl")
            
            logger.info(f"ML model trained and saved. Accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"ML model training failed: {e}")
            return False
    
    def generate_report(self, output_path: str = "regime_backtest_report.html"):
        """Generate comprehensive backtest report"""
        try:
            # Create report content
            report_content = f"""
            <html>
            <head>
                <title>Market Regime Detection Backtest Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ margin: 10px 0; padding: 10px; background-color: #f5f5f5; }}
                    .highlight {{ background-color: #e8f4fd; }}
                </style>
            </head>
            <body>
                <h1>Market Regime Detection Backtest Report</h1>
                <p><strong>Symbol:</strong> {self.symbol}</p>
                <p><strong>Timeframe:</strong> {self.timeframe}</p>
                <p><strong>Data Period:</strong> {self.data['timestamp'].min()} to {self.data['timestamp'].max()}</p>
                
                <h2>Performance Metrics</h2>
                <div class="metric highlight">
                    <h3>Best Optimization Result</h3>
                    <p><strong>Accuracy:</strong> {self.optimization_results[-1].best_accuracy:.3f}</p>
                    <p><strong>Stability Score:</strong> {self.optimization_results[-1].best_stability:.3f}</p>
                    <p><strong>Win Rate:</strong> {self.optimization_results[-1].model_performance['win_rate']:.3f}</p>
                    <p><strong>Average Latency:</strong> {self.optimization_results[-1].model_performance['latency_ms']:.2f} ms</p>
                </div>
                
                <h2>Optimal Thresholds</h2>
                <div class="metric">
                    <pre>{json.dumps(self.optimization_results[-1].best_thresholds, indent=2)}</pre>
                </div>
                
                <h2>Regime Distribution</h2>
                <div class="metric">
                    <p>Analysis of regime distribution across the test period</p>
                </div>
            </body>
            </html>
            """
            
            # Save report
            with open(output_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Backtest report generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return False

def main():
    """Main function for running regime backtesting"""
    # Example usage
    backtester = RegimeBacktester(
        data_path="test_data/sample_historical_data.csv",
        symbol="BTC/USDT",
        timeframe="15m"
    )
    
    # Load data
    if not backtester.load_data():
        logger.error("Failed to load data")
        return
    
    # Train ML model
    backtester.train_ml_model()
    
    # Optimize thresholds
    optimization_result = backtester.optimize_thresholds(n_trials=50)
    
    if optimization_result:
        backtester.optimization_results.append(optimization_result)
        
        # Generate report
        backtester.generate_report()
        
        logger.info("Regime backtesting completed successfully")

if __name__ == "__main__":
    main()
