# Zeapl AI Analytics

Schema-driven natural-language analytics with deterministic execution, validated LLM planning, forecasting, conversation context, and interactive visualization.

> **Core principle:** The dataset defines the schema, the schema defines what the system can safely understand, and the LLM translates business language into a validated plan. Authoritative calculations are performed by the local backend.

## Overview

Zeapl AI Analytics allows a user to ask business questions about an approved CSV or Excel dataset in natural language. The application profiles the active dataset, resolves the requested analytical operation, validates all referenced fields and conditions, executes supported calculations locally, and renders the result in a React interface.

The reference implementation uses:

| Area | Technology |
| --- | --- |
| Frontend | React, Vite, JavaScript, Chart.js |
| Backend | FastAPI, Python, pandas |
| LLM integration | Anthropic Claude |
| Forecasting and ML utilities | scikit-learn |
| Testing | pytest and frontend test tooling configured in the repository |

## Key Capabilities

- Dynamic CSV/Excel dataset profiling
- Schema-driven metric and dimension discovery
- Totals, averages, minima, maxima, and distinct counts
- Filters, date ranges, membership conditions, and supported AND/OR logic
- Grouped breakdowns, trends, comparisons, and correlations
- Top-N and bottom-N rankings
- Two-level nested rankings
- Multi-intent questions
- Conversation follow-ups using compact canonical request context
- Daily, weekly, and monthly forecast paths for eligible metrics
- Cards, tables, charts, hierarchical ranking trees, and forecast panels
- Validated LLM error recovery
- Safe errors for unsupported metrics, dimensions, filters, and ranking structures

## Architecture

```text
Business user
    |
React + Vite frontend
    |
FastAPI API and conversation orchestration
    |
Dataset loader -> profiler -> runtime schema/configuration
    |
Deterministic resolver and/or Claude semantic planner
    |
Canonical-plan validation
    |
Local filters and deterministic pandas/scikit-learn execution
    |
Normalized API response
    |
Cards, tables, charts, ranking trees, explanations, or forecasts
```

Claude proposes a structured interpretation or explanation. It does not execute arbitrary Python or SQL, and it is not the source of numerical truth.

## Repository Layout

```text
Zeapl-MVP/
├── backend/
│   ├── app.py                    # FastAPI application and dynamic endpoints
│   ├── dynamic_analysis.py       # Orchestration, validation, and execution flow
│   ├── semantic_planner.py       # Schema-grounded planning and recovery
│   ├── LLM.py                    # Anthropic client and explanation paths
│   ├── configs/                  # Generated runtime configurations
│   ├── schemas/                  # Generated dataset profiles
│   ├── models/                   # Generated forecast/model artifacts
│   └── test_*.py                 # Backend test suites
├── frontend/
│   ├── src/
│   │   ├── components/           # Query, profile, results, and chart UI
│   │   └── lib/adapters.js       # Backend-response normalization
│   ├── package.json
│   └── vite.config.*
└── README.md
```

Generated configs, schemas, and models are runtime artifacts. Review them before committing; local paths, dataset fingerprints, or derived metadata may not belong in source control.

## Prerequisites

- Python 3.12
- Node.js with npm (use the version expected by the frontend toolchain; an active LTS release is recommended)
- An Anthropic API key for semantic planning, recovery, and explanation features
- An approved CSV or Excel dataset

## Environment Variables

Create a backend environment file:

```bash
cd backend
touch .env
```

Add the confirmed required variable:

```dotenv
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Important:**

- Never commit `.env`.
- Never paste the real key into documentation, screenshots, logs, or issues.
- Restart the backend after changing the key; an already-running process will not automatically inherit a newly edited environment file.
- The current documentation confirms `ANTHROPIC_API_KEY`. Do not add model, timeout, proxy, or frontend environment-variable names unless the corresponding code explicitly reads them.

Ensure `.gitignore` contains at least:

```gitignore
.env
backend/.env
.venv/
backend/.venv/
node_modules/
frontend/node_modules/
*.log
```

## Backend Setup

From the repository root:

```bash
cd backend

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Verify that the backend is running:

```bash
curl --max-time 5 http://127.0.0.1:8000/
```

Expected response includes a message indicating that the Zeapl AI Dashboard API is running.

## Backend Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Health check and endpoint discovery |
| GET | `/api/analyze` | Legacy analysis path retained for compatibility |
| GET | `/api/v2/profile` | Profile a dataset and return its discovered analytical schema |
| GET | `/api/v2/analyze` | Run natural-language analysis |
| GET | `/api/v2/forecast` | Run the dynamic forecasting pipeline |

### Example Analysis Request

Run this from `backend/` so a relative dataset filename resolves there:

```bash
curl --max-time 120 -sG \
  'http://127.0.0.1:8000/api/v2/analyze' \
  --data-urlencode 'question=Show campaign count by category' \
  --data-urlencode 'dataset=your_approved_dataset.csv' \
  --data-urlencode 'conversation_id=readme-example-01' \
  --data-urlencode 'use_llm_fallback=false'
```

Use only approved datasets. Do not commit company or customer data to the repository.

## Frontend Setup

Open a second terminal and run from the repository root:

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Vite normally prints a local URL such as:

```text
http://127.0.0.1:5173
```

Open that URL in a browser. Keep the backend running at `http://127.0.0.1:8000`.

For a production build check:

```bash
npm --prefix frontend run build
```

Do not commit `frontend/node_modules/` or local Node backup directories.

## Running Tests

### Complete Backend Suite

From the repository root:

```bash
backend/.venv/bin/python -m pytest backend -v
```

### Targeted Semantic and Ranking Regression Suites

```bash
backend/.venv/bin/python -m pytest \
  backend/test_nested_ranking.py \
  backend/test_llm_recovery.py \
  backend/test_multi_intent.py \
  -v
```

Run any additional repository-specific regression suite before merging or handing off a build.

### Frontend Checks

Use the scripts defined in `frontend/package.json`. At minimum:

```bash
npm --prefix frontend run build
```

If test scripts are present, run them using their exact package-script names.

## Dataset Lifecycle

For each selected dataset, the backend:

1. Loads the CSV or Excel file into a pandas DataFrame.
2. Profiles columns, nulls, cardinality, identifiers, categories, numeric fields, dates, and sample values.
3. Builds a dataset-specific runtime configuration.
4. Generates a fingerprint to isolate schemas, configurations, and model artifacts.
5. Resolves the question only against fields supported by the active schema.
6. Applies validated filters before aggregation, ranking, forecasting, or explanation.

The current build supports one active dataset per request. It is not yet a relational multi-table query engine or a complete SQL replacement.

## LLM Usage and Deterministic Boundary

Claude is used for defined interpretation and explanation responsibilities:

| Stage | LLM use | Output |
| --- | --- | --- |
| Initial semantic planning | Common for natural-language resolution | Structured analytical plan |
| Error recovery | Conditional when fallback is enabled | Corrected plan subject to validation |
| Dataset explanation | Conditional | Grounded reasoning and insights |
| JSON repair | Conditional | Repaired structured response |
| Filtering and analytics | No | Deterministic result |
| Forecast calculation | No LLM required | Local prediction, intervals, and quality |
| Frontend rendering | No | Cards, tables, charts, and trees |

Every proposed plan must be validated against the runtime schema before execution. Recovery must not invent fields, remove valid filters, or silently discard unresolved conditions.

## Data Privacy

The complete source dataset remains local for authoritative calculations, but the current implementation is not a zero-data-to-LLM design.

Depending on the request path, Claude may receive:

- The user question
- Schema and date context
- Metric and dimension names
- Aliases and relevant sample values
- Failed-plan and validation context
- Computed statistics
- For dataset explanations, a bounded sample of raw rows

The original dataset file is not uploaded as a file, but sampled row-level business data may leave the backend in explanation mode. Before production use with sensitive data, implement or verify:

- Field-level allowlists
- Identifier masking or hashing
- Email, phone, URL, secret, and free-text redaction
- Question-relevant column selection
- Statistics-first explanations
- Smaller bounded samples
- Explicit cloud-planning, recovery, and sample-sharing controls
- Outbound-call audit logs
- Approved provider retention, region, and data-processing policies

The existing LLM-fallback option should not be treated as a universal privacy switch: normal semantic planning may still invoke Claude.

## Reliability and Safe-Failure Behavior

The implementation includes safeguards for:

- Unsupported or invented fields
- Removed filters during recovery
- Invalid ranking structures
- No-matching-row conditions
- Failed conversation turns
- Cross-dataset artifact contamination
- Corrupt model artifacts
- Low forecast quality
- Malformed LLM JSON
- Long-running external LLM calls

Anthropic client behavior should use finite timeouts and controlled retries. Where required by the deployment environment, disable unintended environment-proxy inheritance rather than allowing requests to hang indefinitely.

## Troubleshooting

### Port 8000 Is Already in Use

Identify the existing process:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

If it is the correct backend, keep it running. Otherwise stop the specific process after confirming its PID, or start this application on another port:

```bash
uvicorn app:app --reload --port 8001
```

### Backend Health Works but Analysis Hangs

- Test the health endpoint.
- Confirm `ANTHROPIC_API_KEY` is loaded by the backend process.
- Test provider connectivity separately.
- Inspect semantic-planner, recovery, and explanation calls.
- Verify finite client timeouts and controlled retry behavior.
- Restart the backend after `.env` changes.

### Frontend Fails with a Node `ETIMEDOUT` Read Error

- Use a supported Node LTS version.
- Confirm the repository is available locally rather than being lazily downloaded from cloud storage.
- Reinstall dependencies only after preserving any intentional lockfile changes.
- Do not commit `node_modules` or a local backup of it.

### Generated Files Dominate `git status`

Runtime profiling and forecasting can modify `backend/configs`, `backend/schemas`, and `backend/models`. Review these artifacts separately from intentional source changes. Never use `git add .` without inspecting the staged file list.

## Git and Security Hygiene

Before committing:

```bash
git status --short
git diff
git diff --cached
```

Stage intentional source files explicitly. Do not commit:

- `.env` or API keys
- Real company/customer datasets
- Credentials, tokens, cookies, or internal URLs
- `node_modules` or `.venv`
- QA logs containing sensitive queries or responses
- Generated models/configurations/schemas unless intentionally approved

If a secret was ever committed, removing it from the latest file is not sufficient. Rotate the secret and follow the repository owner's approved history-cleaning procedure.

## Current Limitations

- One active dataset per request
- No validated relational join layer across multiple datasets
- LLM explanation mode can transmit sampled row-level data
- Existing UI controls do not expose every cloud-call category separately
- Multi-intent, recovery, explanation, and JSON repair can multiply LLM calls
- Forecast quality depends on dataset history and must be interpreted with quality warnings
- LLM responses still require schema validation, JSON validation, timeout handling, and safe errors

## Handoff Principle

The system should continue to follow this boundary:

> **Claude interprets or explains. The backend validates and calculates.**

Future work should prioritize deterministic resolution before cloud planning, explicit user controls for every outbound-data path, statistics-first explanations, audited token/data movement, and a separately designed relational layer if multi-dataset joins are introduced.

## Confidentiality

This repository may contain proprietary company logic and generated metadata. Keep the repository private, use only approved datasets, and obtain authorization before sharing source code or internal documentation outside Zeapl.ai.
