# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (requires Python 3.11+)
pip install -e ".[dev]"

# Start infrastructure (PostgreSQL, Redis, Neo4j, ChromaDB)
docker compose up -d postgres redis neo4j chromadb

# Run dev server (hot reload)
uvicorn app.main:app --reload --port 8000

# Run all tests
pytest tests/ -v --cov=app

# Run a single test file
pytest tests/test_workflow.py -v

# Lint
ruff check app/ tests/
ruff format app/ tests/
```

API docs at http://localhost:8000/docs (only available when `DEBUG=True`).

## Architecture

This is a multi-agent health advisory system. The core request path is:

**REST/WebSocket → Chat API → LangGraph Workflow → Specialized Agent → LLM**

### LangGraph Workflow (`app/core/`)

The central orchestration is a `StateGraph` compiled in `workflow.py`. Every chat request flows through these nodes in order:

```
load_memory → classify_intent → [mental_agent | nutrition_agent | general_agent] → save_memory → [consolidate_memory?] → END
```

All nodes share a single `AgentState` TypedDict (defined in `state.py`). Nodes communicate exclusively by reading and writing fields on this state — never by calling each other directly.

**Intent routing** (`intent_router.py`): Uses LLM JSON output to classify each message into `mental`, `nutrition`, or `general`. Confidence below 0.7 falls back to `general`.

### Agents (`app/agents/`)

`BaseAgent` defines the interface: `run(state) -> AgentState` (blocking) and `stream(state) -> AsyncGenerator[str, None]` (token streaming). Each subclass calls the LLM client and writes its reply to `state["response"]`.

### LLM Client (`app/llm/client.py`)

`LLMClient` wraps two `AsyncOpenAI` backends:
- `general` — OpenAI-compatible API (intent classification, general/nutrition agents, memory consolidation)
- `vllm` — locally deployed fine-tuned model (mental agent; defaults to `deepseek-8b-qlora`)

Use `get_llm_client()` singleton. `complete_json()` enforces `response_format: json_object` and parses the result.

### Database Layer (`app/db/`)

- **PostgreSQL** — SQLAlchemy async engine (`postgres.py`). `get_db_session()` is the FastAPI dependency. `init_db()` creates tables via `Base.metadata.create_all` (dev only; use Alembic in production).
- **Redis** — session caching via `redis_client.py`.
- **Neo4j + ChromaDB** — stubs for Phase 3 (KG-based RAG).

### Configuration (`app/config.py`)

All settings loaded from `.env` via pydantic-settings. Call `get_settings()` (cached with `lru_cache`) — never instantiate `Settings` directly. Copy `.env.example` to `.env` before first run.

## Development Phases

The project is being built iteratively. Currently completed:
- **Phase 1**: FastAPI skeleton, DB layer, auth, health-calc tools
- **Phase 2**: LangGraph workflow + intent router + agent stubs (in progress)

Pending: Phase 3 (RAG), Phase 4 (fine-tuned mental model), Phase 5 (hierarchical memory), Phase 6 (optimization).

Stub nodes in `workflow.py` (`load_memory_node`, `save_memory_node`, `consolidate_memory_node`) have `# TODO: Phase 5` comments marking where full memory logic will live.

## Key Conventions

- Logging uses `structlog` via `get_logger(name)` from `app/utils/logger.py`. Use `await logger.ainfo(...)` in async contexts.
- Custom exceptions live in `app/utils/exceptions.py`; they are registered globally in `app/api/middleware.py`.
- Pydantic schemas (request/response models) are in `app/schemas/`; ORM models are in `app/db/models.py`.
- Prompts are separated into `app/prompts/` — one file per domain (intent, mental, nutrition, system, consolidation).
- Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).
