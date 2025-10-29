# AlphaPulse Monorepo Import Changes Guide

**Date:** October 26, 2025  
**Purpose:** Guide for updating imports after monorepo migration

---

## Python Import Changes

### Good News: Minimal Changes Needed! ✅

Thanks to `PYTHONPATH=apps/backend`, most Python imports **remain exactly the same**.

### What Stays the Same

```python
# These all still work without changes:
from app.core.config import settings
from app.services.market_data_service import MarketDataService
from database.models import Signal, SignalRecommendation
from ai.signal_generator import SignalGenerator
from strategies.pattern_detector import PatternDetector
```

### What to Clean Up (Optional)

**Files with manual path manipulation (20 files):**

Remove these lines (they still work but are no longer needed):

```python
# REMOVE THESE (optional cleanup):
import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))
sys.path.append(str(backend_path))

# Just use the imports directly - PYTHONPATH handles it
from app.core.config import settings
```

**Files to clean up:**
1. `apps/backend/database/migrations/init_db.py`
2. `apps/backend/database/migrations/init_db_simple.py`
3. `apps/backend/app/main_ai_system_simple.py`
4. `apps/backend/test_enhanced_integration.py`
5. `apps/backend/data/real_time_processor.py`
6. And 15 more test/migration files

---

## TypeScript/Frontend Import Changes

### NEW: Use Shared Types

**Before (duplicate definitions):**
```typescript
// In multiple files, duplicated:
interface Signal {
  signal_id: string;
  symbol: string;
  direction: 'long' | 'short';
  confidence: number;
  // ...
}
```

**After (shared types):**
```typescript
import { Signal, SignalRecommendation } from '@alphapulse/shared-types';

// No need to define - already imported!
```

### NEW: Use Shared Config

**Before:**
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const endpoints = {
  signals: '/api/v1/signals',
  recommendations: '/api/v1/recommendations',
  // ...
};
```

**After:**
```typescript
import { API_CONFIG } from '@alphapulse/config';

const API_URL = API_CONFIG.backend.url;
const signalsEndpoint = API_CONFIG.endpoints.signals;
```

### Files to Update

**Frontend API files:**
- `apps/web/lib/api.ts`
- `apps/web/lib/api_intelligent.ts`
- `apps/web/lib/phase4Api.ts`
- `apps/web/lib/hooks.ts`
- `apps/web/lib/websocket_single_pair.ts`

**Component files (optional):**
- Any component with duplicate type definitions
- Can gradually migrate to use shared types

---

## Docker Path Changes

### docker-compose.yml

**Before:**
```yaml
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
```

**After:**
```yaml
services:
  backend:
    build:
      context: ../../
      dockerfile: infrastructure/docker/Dockerfile.backend
    working_dir: /app/apps/backend
```

### Dockerfile Changes

**Before (backend/Dockerfile):**
```dockerfile
COPY . .
WORKDIR /app
CMD ["uvicorn", "main:app"]
```

**After (infrastructure/docker/Dockerfile.backend):**
```dockerfile
COPY apps/backend ./apps/backend
ENV PYTHONPATH=/app/apps/backend
WORKDIR /app/apps/backend
CMD ["uvicorn", "main:app"]
```

---

## NPM Script Changes

### Running Commands

**Before:**
```bash
cd backend && python main.py
cd frontend && npm run dev
```

**After:**
```bash
pnpm dev                    # Starts both!
# OR
pnpm dev:backend           # Backend only
pnpm dev:web               # Frontend only
```

### Testing

**Before:**
```bash
cd backend && pytest tests/
cd frontend && npm test
```

**After:**
```bash
pnpm test                   # All tests
pnpm test:backend          # Backend only
pnpm test:web              # Frontend only
```

---

## Database Connection Changes

### Connection String Update

**Before:**
```
postgresql://alpha_emon:Emon_%4017711@localhost:5432/alphapulse
```

**After:**
```
postgresql://alpha_emon:Emon_%4017711@localhost:55433/alphapulse
```

**Note:** Port changed from 5432 to 55433 to avoid conflicts

### In Code

```python
# No changes needed in Python code
# Uses DATABASE_URL environment variable

# In TypeScript (if you add DB access):
import { DATABASE_CONFIG, buildDatabaseUrl } from '@alphapulse/config';
const dbUrl = buildDatabaseUrl('alpha_emon', 'Emon_@17711');
```

---

## Import Summary Table

| Location | What Changed | Action Required |
|----------|--------------|-----------------|
| **Python files** | Paths stay same | Optional: Remove sys.path hacks |
| **Frontend types** | Use shared types | Update imports to @alphapulse/shared-types |
| **Frontend config** | Use shared config | Update to use @alphapulse/config |
| **Docker** | New paths | Already updated in new Dockerfiles |
| **Scripts** | New location | Use `pnpm` commands instead |
| **Database** | New port | Already updated (55433) |

---

## Migration Checklist

### Python Files (Optional Cleanup)

- [ ] Remove sys.path hacks from `apps/backend/database/migrations/init_db.py`
- [ ] Remove sys.path hacks from `apps/backend/app/main_ai_system_simple.py`
- [ ] Remove sys.path hacks from test files in `apps/backend/tests/integration/`

### Frontend Files (Recommended)

- [ ] Update `apps/web/lib/api.ts` to use shared types
- [ ] Update component props to use shared types
- [ ] Use `@alphapulse/config` for API URLs

### Configuration Files (Done)

- [x] Updated `vercel.json`
- [x] Updated `docker-compose.yml`
- [x] Created new Dockerfiles
- [x] Updated package.json files

---

## Testing After Changes

### 1. Test Python Imports

```bash
cd apps/backend
python -c "from app.core.config import settings; print('✓ Imports work')"
python -c "from database.models import Signal; print('✓ Models work')"
```

### 2. Test Frontend Builds

```bash
pnpm --filter=web build
```

### 3. Test Full Stack

```bash
pnpm dev
# Should start both backend and frontend
```

### 4. Test Docker

```bash
pnpm docker:build
pnpm docker:up
```

---

## Common Issues & Solutions

### Issue: Python module not found

**Solution:** Ensure PYTHONPATH is set:
```bash
export PYTHONPATH=apps/backend
# OR
cd apps/backend && python main.py
```

### Issue: Workspace package not found

**Solution:** Install from root:
```bash
pnpm install
```

### Issue: Types not resolving in IDE

**Solution:** Restart TypeScript server or reload VS Code

---

## Quick Reference

### Project Structure
```
Root → apps/ → backend/     (Python)
              → web/        (Next.js)
     → packages/ → shared-types/  (Types)
                → config/         (Config)
     → infrastructure/ (Docker, K8s)
     → tools/ (Scripts)
```

### Common Commands
```bash
pnpm dev            # Start everything
pnpm test           # Test everything
pnpm build          # Build everything
pnpm docker:up      # Start Docker
cd apps/backend     # Go to backend
cd apps/web         # Go to frontend
```

---

**Status:** ✅ Import paths documented  
**Action Required:** Minimal (Python paths unchanged)  
**Optional Cleanup:** Remove sys.path hacks (20 files)  
**Frontend Updates:** Recommended (use shared types)

