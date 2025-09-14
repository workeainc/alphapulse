# Backend Directory Organization Plan

## Current Issues in Backend Directory:
1. **Scattered Files**: JSON reports, PNG images, DB files, logs mixed with source code
2. **Duplicate Test Directories**: `tests/`, `test/`, `test_data/`, `test_models/`
3. **Mixed File Types**: Performance reports, images, databases, and source code all mixed
4. **Unorganized Structure**: No clear separation of concerns

## Proposed Clean Backend Structure:

```
backend/
├── app/                           # Main application code
│   ├── __init__.py
│   ├── main_unified.py
│   ├── core/                      # Core configurations
│   ├── services/                  # Business services
│   ├── strategies/                # Trading strategies
│   ├── routes/                    # API endpoints
│   └── database/                  # Database models and connections
├── ai/                            # AI/ML components
│   ├── ml_models/                 # ML model implementations
│   ├── feature_engineering/       # Feature engineering
│   ├── retraining/                # Model retraining
│   └── deployment/                # Model deployment
├── data/                          # Data processing
│   ├── collectors/                # Data collection services
│   ├── processors/                # Data processing services
│   └── normalization/             # Data normalization
├── execution/                     # Trading execution
│   ├── connectors/                # Exchange connectors
│   ├── order_management/          # Order management
│   └── analytics/                 # Execution analytics
├── monitoring/                    # System monitoring
│   ├── metrics/                   # Performance metrics
│   ├── alerts/                    # Alerting system
│   └── dashboards/                # Monitoring dashboards
├── config/                        # Configuration files
│   ├── exchange_config.py
│   ├── alembic.ini
│   └── requirements.txt
├── deployment/                    # Deployment files
│   ├── docker/                    # Docker configuration
│   ├── k8s/                       # Kubernetes configuration
│   ├── nginx.conf
│   └── deploy scripts
├── reports/                       # Generated reports
│   ├── performance/               # Performance reports
│   ├── test_reports/              # Test reports
│   └── benchmarks/                # Benchmark results
├── logs/                          # Application logs
├── cache/                         # Cache storage
├── models/                        # Trained ML models
├── static/                        # Static assets
└── scripts/                       # Utility scripts
```

## Files to Move/Organize:

### Move to reports/
- `alphapulse_test_report_*.json` → `reports/test_reports/`
- `performance_baseline_*.json` → `reports/performance/`
- `system_metrics.png`, `performance_*.png` → `reports/performance/`
- `reorganization_*.json` → `reports/`

### Move to config/
- `alembic.ini` → `config/`
- `requirements*.txt` → `config/`
- `docker-compose.yml` → `deployment/docker/`

### Move to deployment/
- `deploy.bat`, `deploy.sh` → `deployment/`
- `Dockerfile` → `deployment/docker/`
- `nginx.conf` → `deployment/`

### Clean up test files
- Remove duplicate test directories
- Move test files to root `tests/` directory
- Remove test database files

### Clean up temporary files
- Remove `__pycache__/`
- Remove `.pytest_cache/`
- Remove `.benchmarks/`
- Remove `.coverage`

## Benefits:
- Clean, organized source code structure
- Clear separation of concerns
- Easy to navigate and maintain
- Professional appearance
- Better scalability
