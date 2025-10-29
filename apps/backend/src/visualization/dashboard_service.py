"""
Real-Time Dashboard Service for AlphaPulse
Week 8: Real-Time Dashboards & Reporting

Features:
- Interactive Plotly charts for funding rates, anomalies, and predictions
- Real-time data updates from TimescaleDB
- Performance metrics visualization
- Multi-symbol support with dropdown selection
- No external API dependencies (local Plotly + TimescaleDB)
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly")

# Dash imports for web interface
try:
    from dash import Dash, dcc, html, Input, Output, callback_context
    from dash.exceptions import PreventUpdate
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logging.warning("Dash not available. Install with: pip install dash")

logger = logging.getLogger(__name__)

class DashboardService:
    """Real-time dashboard service for AlphaPulse"""
    
    def __init__(self, db_connection, host: str = 'localhost', port: int = 8050):
        self.db = db_connection
        self.host = host
        self.port = port
        self.logger = logger
        
        # Dashboard configuration
        self.update_interval = 10  # seconds
        self.history_hours = 24
        self.max_data_points = 1000
        
        # Available symbols (will be populated from database)
        self.available_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        
        # Initialize Dash app if available
        if DASH_AVAILABLE:
            self.app = Dash(__name__)
            self.setup_layout()
            self.setup_callbacks()
        else:
            self.app = None
            self.logger.warning("Dash not available - dashboard will run in console mode")
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        if not self.app:
            return
            
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🚀 AlphaPulse Real-Time Dashboard", 
                        style={'textAlign': 'center', 'color': '#1f77b4', 'marginBottom': '20px'}),
                html.P("Real-time crypto market insights and performance monitoring", 
                       style={'textAlign': 'center', 'color': '#666', 'marginBottom': '30px'})
            ]),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Symbol:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[{'label': sym, 'value': sym} for sym in self.available_symbols],
                        value='BTC/USDT',
                        style={'width': '200px'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '20px'}),
                
                html.Div([
                    html.Label("Time Range:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='time-range-dropdown',
                        options=[
                            {'label': '1 Hour', 'value': 1},
                            {'label': '6 Hours', 'value': 6},
                            {'label': '24 Hours', 'value': 24},
                            {'label': '7 Days', 'value': 168}
                        ],
                        value=24,
                        style={'width': '150px'}
                    )
                ], style={'display': 'inline-block'})
            ], style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # Charts Grid
            html.Div([
                # Row 1: Funding Rates and Anomalies
                html.Div([
                    html.Div([
                        dcc.Graph(id='funding-rate-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(id='anomaly-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Row 2: Performance and Predictions
                html.Div([
                    html.Div([
                        dcc.Graph(id='performance-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(id='prediction-chart', style={'height': '400px'})
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Row 3: System Metrics
                html.Div([
                    html.Div([
                        dcc.Graph(id='system-metrics-chart', style={'height': '300px'})
                    ], style={'width': '100%'})
                ])
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # milliseconds
                n_intervals=0
            ),
            
            # Status indicator
            html.Div([
                html.Div(id='status-indicator', style={'textAlign': 'center', 'marginTop': '20px'})
            ])
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for real-time updates"""
        if not self.app:
            return
            
        @self.app.callback(
            [Output('funding-rate-chart', 'figure'),
             Output('anomaly-chart', 'figure'),
             Output('performance-chart', 'figure'),
             Output('prediction-chart', 'figure'),
             Output('system-metrics-chart', 'figure'),
             Output('status-indicator', 'children')],
            [Input('symbol-dropdown', 'value'),
             Input('time-range-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        async def update_charts(symbol, time_range, n_intervals):
            """Update all charts with real-time data"""
            try:
                if not symbol:
                    raise PreventUpdate
                
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=time_range)
                
                # Fetch data from database
                funding_fig = await self.create_funding_rate_chart(symbol, start_time, end_time)
                anomaly_fig = await self.create_anomaly_chart(symbol, start_time, end_time)
                performance_fig = await self.create_performance_chart(symbol, start_time, end_time)
                prediction_fig = await self.create_prediction_chart(symbol, start_time, end_time)
                system_fig = await self.create_system_metrics_chart(start_time, end_time)
                
                # Status message
                status_msg = f"✅ Last updated: {datetime.now().strftime('%H:%M:%S')} | Symbol: {symbol} | Time Range: {time_range}h"
                
                return funding_fig, anomaly_fig, performance_fig, prediction_fig, system_fig, status_msg
                
            except Exception as e:
                self.logger.error(f"Error updating charts: {e}")
                # Return empty figures on error
                empty_fig = go.Figure().add_annotation(text="Error loading data", xref="paper", yref="paper", x=0.5, y=0.5)
                error_status = f"❌ Error: {str(e)}"
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, error_status
    
    async def create_funding_rate_chart(self, symbol: str, start_time: datetime, end_time: datetime) -> go.Figure:
        """Create funding rate chart"""
        try:
            # Fetch funding rate data
            query = """
                SELECT timestamp, funding_rate, exchange
                FROM funding_rates 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """
            data = await self.db.fetch(query, symbol, start_time, end_time)
            
            if not data:
                return self._create_empty_chart("No funding rate data available")
            
            df = pd.DataFrame(data, columns=['timestamp', 'funding_rate', 'exchange'])
            
            # Create chart
            fig = go.Figure()
            
            # Add funding rate line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['funding_rate'],
                mode='lines+markers',
                name='Funding Rate',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            # Add zero line for reference
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Line")
            
            fig.update_layout(
                title=f'{symbol} Funding Rates',
                xaxis_title='Time',
                yaxis_title='Funding Rate (%)',
                template='plotly_white',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating funding rate chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    async def create_anomaly_chart(self, symbol: str, start_time: datetime, end_time: datetime) -> go.Figure:
        """Create anomaly detection chart"""
        try:
            # Fetch anomaly data
            query = """
                SELECT timestamp, value, z_score, data_type
                FROM anomalies 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """
            data = await self.db.fetch(query, symbol, start_time, end_time)
            
            if not data:
                return self._create_empty_chart("No anomaly data available")
            
            df = pd.DataFrame(data, columns=['timestamp', 'value', 'z_score', 'data_type'])
            
            # Create chart
            fig = go.Figure()
            
            # Add anomaly points with size based on z-score
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['value'],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=df['z_score'].abs() * 10,  # Size based on z-score
                    color=df['z_score'].abs(),
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Z-Score")
                ),
                text=df['data_type'],
                hovertemplate='<b>%{text}</b><br>Value: %{y}<br>Z-Score: %{marker.color}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'{symbol} Anomaly Detection',
                xaxis_title='Time',
                yaxis_title='Value',
                template='plotly_white',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating anomaly chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    async def create_performance_chart(self, symbol: str, start_time: datetime, end_time: datetime) -> go.Figure:
        """Create performance metrics chart"""
        try:
            # Fetch performance data
            query = """
                SELECT timestamp, pnl, win_rate, drawdown
                FROM performance_metrics 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """
            data = await self.db.fetch(query, symbol, start_time, end_time)
            
            if not data:
                return self._create_empty_chart("No performance data available")
            
            df = pd.DataFrame(data, columns=['timestamp', 'pnl', 'win_rate', 'drawdown'])
            
            # Create subplot for multiple metrics
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('PnL Over Time', 'Win Rate & Drawdown'),
                vertical_spacing=0.1
            )
            
            # PnL chart
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['pnl'], name='PnL', line=dict(color='#2ca02c')),
                row=1, col=1
            )
            
            # Win rate and drawdown
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['win_rate'], name='Win Rate', line=dict(color='#1f77b4')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['drawdown'], name='Drawdown', line=dict(color='#ff7f0e')),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f'{symbol} Performance Metrics',
                template='plotly_white',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating performance chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    async def create_prediction_chart(self, symbol: str, start_time: datetime, end_time: datetime) -> go.Figure:
        """Create predictive signal chart"""
        try:
            # Fetch prediction data
            query = """
                SELECT timestamp, confidence, predicted_pnl, signal_type
                FROM signal_predictions 
                WHERE symbol = $1 AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """
            data = await self.db.fetch(query, symbol, start_time, end_time)
            
            if not data:
                return self._create_empty_chart("No prediction data available")
            
            df = pd.DataFrame(data, columns=['timestamp', 'confidence', 'predicted_pnl', 'signal_type'])
            
            # Create chart
            fig = go.Figure()
            
            # Add confidence line
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['confidence'],
                mode='lines+markers',
                name='Signal Confidence',
                line=dict(color='#9467bd', width=2),
                marker=dict(size=6)
            ))
            
            # Add predicted PnL
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['predicted_pnl'],
                mode='lines+markers',
                name='Predicted PnL',
                line=dict(color='#8c564b', width=2),
                marker=dict(size=6),
                yaxis='y2'
            ))
            
            # Add confidence threshold line
            fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="High Confidence (70%)")
            
            fig.update_layout(
                title=f'{symbol} Predictive Signals',
                xaxis_title='Time',
                yaxis_title='Confidence',
                yaxis2=dict(title='Predicted PnL', overlaying='y', side='right'),
                template='plotly_white',
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating prediction chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    async def create_system_metrics_chart(self, start_time: datetime, end_time: datetime) -> go.Figure:
        """Create system metrics chart"""
        try:
            # Fetch system metrics (latency, cache hits, etc.)
            query = """
                SELECT timestamp, metric_name, metric_value
                FROM system_metrics 
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp
            """
            data = await self.db.fetch(query, start_time, end_time)
            
            if not data:
                return self._create_empty_chart("No system metrics available")
            
            df = pd.DataFrame(data, columns=['timestamp', 'metric_name', 'metric_value'])
            
            # Create chart
            fig = go.Figure()
            
            # Group by metric name and plot each
            for metric_name in df['metric_name'].unique():
                metric_data = df[df['metric_name'] == metric_name]
                fig.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data['metric_value'],
                    mode='lines+markers',
                    name=metric_name
                ))
            
            fig.update_layout(
                title='System Performance Metrics',
                xaxis_title='Time',
                yaxis_title='Metric Value',
                template='plotly_white',
                height=300,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating system metrics chart: {e}")
            return self._create_empty_chart(f"Error: {str(e)}")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig
    
    def run(self):
        """Run the dashboard server"""
        if self.app:
            self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
            self.app.run_server(host=self.host, port=self.port, debug=False)
        else:
            self.logger.warning("Dashboard app not available - running in console mode")
            self._run_console_mode()
    
    def _run_console_mode(self):
        """Run dashboard in console mode for testing"""
        self.logger.info("Running dashboard in console mode")
        try:
            while True:
                # Simulate real-time updates
                asyncio.run(self._console_update())
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            self.logger.info("Dashboard stopped by user")
    
    async def _console_update(self):
        """Console mode update function"""
        try:
            # Fetch and display summary data
            summary = await self._get_summary_data()
            self.logger.info(f"Dashboard Update: {summary}")
        except Exception as e:
            self.logger.error(f"Console update error: {e}")
    
    async def _get_summary_data(self) -> Dict[str, Any]:
        """Get summary data for console mode"""
        try:
            # Get latest data for each metric
            summary = {}
            
            # Funding rates
            funding_query = "SELECT COUNT(*) FROM funding_rates WHERE timestamp > NOW() - INTERVAL '1 hour'"
            funding_count = await self.db.fetch(funding_query)
            summary['funding_rates_1h'] = funding_count[0][0] if funding_count else 0
            
            # Anomalies
            anomaly_query = "SELECT COUNT(*) FROM anomalies WHERE timestamp > NOW() - INTERVAL '1 hour'"
            anomaly_count = await self.db.fetch(anomaly_query)
            summary['anomalies_1h'] = anomaly_count[0][0] if anomaly_count else 0
            
            # Performance
            perf_query = "SELECT COUNT(*) FROM performance_metrics WHERE timestamp > NOW() - INTERVAL '1 hour'"
            perf_count = await self.db.fetch(perf_query)
            summary['performance_1h'] = perf_count[0][0] if perf_count else 0
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting summary data: {e}")
            return {'error': str(e)}
