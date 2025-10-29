# AlphaPulse Backend

**Version:** 2.0.0  
**Type:** Python FastAPI Application  
**Purpose:** Signal Analysis & Recommendation Engine

---

## Structure

```
apps/backend/
├── src/                   # All source code
│   ├── app/              # Main FastAPI application
│   ├── ai/               # AI/ML pipeline
│   ├── strategies/       # Trading strategies
│   ├── database/         # Database models & migrations
│   ├── data/             # Data pipelines
│   ├── services/         # Business services
│   ├── core/             # Core functionality
│   ├── routes/           # API routes
│   ├── monitoring/       # Monitoring
│   ├── streaming/        # Streaming services
│   ├── tracking/         # Signal tracking
│   └── utils/            # Utilities
├── tests/                # All tests
│   ├── integration/      # Integration tests
│   ├── unit/             # Unit tests
│   └── performance/      # Performance tests
├── scripts/              # Development scripts
├── migrations/           # Database migrations
├── config/               # Configuration files
├── docs/                 # Documentation
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── package.json          # Package scripts
└── pyproject.toml        # Python package config
```

---

## Development

### Start Server

```bash
# From root
pnpm dev:backend

# OR from this directory
python main.py
```

### Run Tests

```bash
# From root
pnpm test:backend

# OR from this directory
pytest tests/ -v
```

### Linting

```bash
# From root
pnpm lint:backend

# OR from this directory
flake8 src/
black src/
isort src/
```

---

## API Endpoints

- **Health:** GET `/health`
- **Signals:** GET `/api/signals/latest`
- **Patterns:** GET `/api/patterns/latest`
- **Market Status:** GET `/api/market/status`
- **WebSocket:** WS `/ws`

**Full API docs:** http://localhost:8000/docs (when running)

---

## Database

### Connection

Uses `DATABASE_URL` environment variable:
```
postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse
```

### Migrations

```bash
# From root
pnpm db:migrate

# OR from this directory
python scripts/run_migration.py
```

---

## Import Paths

All imports use the `src.` prefix:

```python
from src.app.core.config import settings
from src.database.models import Signal, SignalRecommendation
from src.ai.signal_generator import SignalGenerator
from src.strategies.pattern_detector import PatternDetector
```

**PYTHONPATH is set to `apps/backend` by:**
- Docker: ENV in Dockerfile
- Local dev: npm scripts
- main.py: Sets up path automatically

---

## Dependencies

Main dependencies:
- FastAPI 0.104+
- SQLAlchemy 2.0+
- TimescaleDB client
- Pandas, NumPy
- XGBoost, LightGBM, CatBoost
- TensorFlow, PyTorch

See `requirements.txt` for complete list.

---

## Environment Variables

Required:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string (optional)

Optional:
- `DEBUG` - Enable debug mode
- `LOG_LEVEL` - Logging level
- API keys for data sources

See `/ENV.md` for complete list.

---

**Part of AlphaPulse Monorepo**  
**See:** `/MONOREPO_README.md` for full monorepo guide

