# AlphaPulse - Clean Monorepo ✅

**Version:** 2.0.0  
**Status:** Ready  
**Date:** October 26, 2025

## What It Is

**AlphaPulse** = AI Signal Analysis & Recommendation Engine  
- Analyzes markets → Generates signals → Recommends trades  
- **You** execute manually (no automated trading)

## Quick Start

```bash
pnpm install                    # Install deps
cd apps/backend && pip install -r requirements.txt
pnpm db:up                      # Start database
pnpm dev                        # Start everything
```

**Access:** Backend (8000) | Frontend (3000)

## Structure

```
apps/backend/    Python FastAPI backend
apps/web/        Next.js frontend
packages/        Shared types & config
infrastructure/  Docker, K8s, monitoring
tools/scripts/   Utility scripts
docs/            Documentation (30 files, archived 190+)
```

## Commands

```bash
pnpm dev          # Start all
pnpm test         # Test all  
pnpm docker:up    # Docker all
```

## What Changed

✅ Removed execution code (~4,000 lines)  
✅ Renamed Trade → SignalRecommendation  
✅ Created clean monorepo structure  
✅ Fixed all import issues  
✅ Reduced docs from 220+ to 30  
✅ Database optimized (15 tables)

## Status

Root: 5 folders + 10 files ✅ Minimal!  
Imports: All fixed ✅  
Database: Running on port 55433 ✅  
Ready: YES ✅

See `README.md` for full docs.

