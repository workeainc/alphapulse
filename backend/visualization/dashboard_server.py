"""
Scalable Dashboard Server for AlphaPulse
Week 8: Real-Time Dashboards & Reporting

Features:
- Flask-based web server for dashboard hosting
- Gunicorn support for production deployment
- Multi-user access with load balancing
- Health checks and monitoring
- No external API dependencies
"""

import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from flask import Flask, render_template, jsonify, request, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask not available. Install with: pip install flask flask-cors")

try:
    from backend.database.connection import TimescaleDBConnection
    from backend.visualization.dashboard_service import DashboardService
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logging.warning("Backend modules not available")

logger = logging.getLogger(__name__)

class DashboardServer:
    """Scalable dashboard server for AlphaPulse"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logger
        
        # Server configuration
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 8050)
        self.debug = self.config.get('debug', False)
        
        # Database configuration
        self.db_config = self.config.get('db_config', {
            'host': 'localhost',
            'port': 5432,
            'database': 'alphapulse',
            'user': 'postgres',
            'password': 'password'
        })
        
        # Initialize components
        self.db_connection = None
        self.dashboard_service = None
        self.flask_app = None
        
        # Health monitoring
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
    async def initialize(self):
        """Initialize the dashboard server"""
        try:
            self.logger.info("Initializing Dashboard Server...")
            
            # Initialize database connection
            if BACKEND_AVAILABLE:
                self.db_connection = TimescaleDBConnection(
                    f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
                    f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
                )
                await self.db_connection.connect()
                self.logger.info("Database connection established")
                
                # Initialize dashboard service
                self.dashboard_service = DashboardService(
                    self.db_connection,
                    host=self.host,
                    port=self.port
                )
                self.logger.info("Dashboard service initialized")
            
            # Initialize Flask app
            if FLASK_AVAILABLE:
                self.flask_app = self._create_flask_app()
                self.logger.info("Flask app created")
            
            self.logger.info("Dashboard Server initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Dashboard Server: {e}")
            return False
    
    def _create_flask_app(self) -> Flask:
        """Create and configure Flask application"""
        app = Flask(__name__)
        
        # Enable CORS for cross-origin requests
        CORS(app)
        
        # Configure logging
        if not self.debug:
            app.logger.setLevel(logging.INFO)
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    def _register_routes(self, app: Flask):
        """Register Flask routes"""
        
        @app.route('/')
        def index():
            """Main dashboard page"""
            self.request_count += 1
            try:
                return render_template('dashboard.html')
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Error rendering dashboard: {e}")
                return f"<h1>AlphaPulse Dashboard</h1><p>Error: {str(e)}</p>"
        
        @app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            try:
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'uptime': str(datetime.now() - self.start_time),
                    'requests': self.request_count,
                    'errors': self.error_count,
                    'database': 'connected' if self.db_connection and self.db_connection.connected else 'disconnected'
                }
                return jsonify(health_status)
            except Exception as e:
                self.error_count += 1
                return jsonify({'status': 'unhealthy', 'error': str(e)}), 500
        
        @app.route('/api/metrics')
        def get_metrics():
            """Get system metrics"""
            try:
                if not self.db_connection:
                    return jsonify({'error': 'Database not connected'}), 500
                
                # Get basic metrics
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'system': {
                        'uptime': str(datetime.now() - self.start_time),
                        'requests': self.request_count,
                        'errors': self.error_count
                    }
                }
                
                return jsonify(metrics)
                
            except Exception as e:
                self.error_count += 1
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/symbols')
        def get_symbols():
            """Get available symbols"""
            try:
                if not self.db_connection:
                    return jsonify({'error': 'Database not connected'}), 500
                
                # Get symbols from database
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
                
                return jsonify({'symbols': symbols})
                
            except Exception as e:
                self.error_count += 1
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/data/<symbol>')
        def get_symbol_data(symbol):
            """Get data for a specific symbol"""
            try:
                if not self.db_connection:
                    return jsonify({'error': 'Database not connected'}), 500
                
                # Get time range from query params
                hours = request.args.get('hours', 24, type=int)
                
                # Mock data for now (replace with actual database queries)
                data = {
                    'symbol': symbol,
                    'hours': hours,
                    'funding_rates': [],
                    'anomalies': [],
                    'performance': [],
                    'predictions': []
                }
                
                return jsonify(data)
                
            except Exception as e:
                self.error_count += 1
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/chart/<chart_type>')
        def get_chart_data(chart_type):
            """Get chart data for specific chart type"""
            try:
                if not self.db_connection:
                    return jsonify({'error': 'Database not connected'}), 500
                
                symbol = request.args.get('symbol', 'BTC/USDT')
                hours = request.args.get('hours', 24, type=int)
                
                # Mock chart data (replace with actual database queries)
                chart_data = {
                    'chart_type': chart_type,
                    'symbol': symbol,
                    'hours': hours,
                    'data': []
                }
                
                return jsonify(chart_data)
                
            except Exception as e:
                self.error_count += 1
                return jsonify({'error': str(e)}), 500
        
        @app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors"""
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors"""
            self.error_count += 1
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self):
        """Run the dashboard server"""
        if not self.flask_app:
            self.logger.error("Flask app not available")
            return
        
        try:
            self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
            self.flask_app.run(
                host=self.host,
                port=self.port,
                debug=self.debug,
                threaded=True
            )
        except Exception as e:
            self.logger.error(f"Error running dashboard server: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_connection:
                await self.db_connection.close()
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def create_app():
    """Factory function for creating Flask app (for Gunicorn)"""
    server = DashboardServer()
    
    # Initialize asynchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.initialize())
    
    return server.flask_app

if __name__ == "__main__":
    # Configuration
    config = {
        'host': os.getenv('DASHBOARD_HOST', '0.0.0.0'),
        'port': int(os.getenv('DASHBOARD_PORT', 8050)),
        'debug': os.getenv('DASHBOARD_DEBUG', 'false').lower() == 'true',
        'db_config': {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'alphapulse'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password')
        }
    }
    
    # Create and run server
    server = DashboardServer(config)
    
    try:
        # Initialize server
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        success = loop.run_until_complete(server.initialize())
        
        if success:
            # Run server
            server.run()
        else:
            logger.error("Failed to initialize dashboard server")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped by user")
    except Exception as e:
        logger.error(f"Dashboard server error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if 'server' in locals():
            loop.run_until_complete(server.cleanup())
