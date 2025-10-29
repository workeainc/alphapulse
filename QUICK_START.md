# AlphaPulse Quick Start

## Install & Run

```bash
# 1. Install dependencies
pnpm install
cd apps/backend && pip install -r requirements.txt && cd ../..

# 2. Start database
pnpm db:up

# 3. Start everything
pnpm dev
```

**Access:**
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## Structure

```
apps/backend/   - Python FastAPI backend
apps/web/       - Next.js frontend
packages/       - Shared types & config
infrastructure/ - Docker, K8s
tools/scripts/  - Utility scripts
```

## Commands

```bash
pnpm dev           # Start all
pnpm test          # Test all
pnpm docker:up     # Start Docker
```

## Environment

Copy `env.template` to `.env` and fill in your values.

See `ENV.md` for all environment variables.

