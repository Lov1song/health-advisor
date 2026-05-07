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

# Lint / format
ruff check app/ tests/
ruff format app/ tests/
```

API docs at http://localhost:8000/docs (only available when `DEBUG=True`).

**Setup**: No `.env.example` exists — create `.env` manually using `app/config.py` as the field reference. Minimum required: `DEEPSEEK_API_KEY`, `POSTGRES_URL`, `JWT_SECRET`.

## Architecture

This is a multi-agent health advisory system. The core request path is:

**REST/WebSocket → Chat API → LangGraph Workflow → Specialized Agent → LLM**

All API routes are prefixed `/api/v1/`. Auth uses Bearer JWT for REST; WebSocket authenticates via `?token=<jwt>` query param (FastAPI deps don't work in WebSocket handlers).

### LangGraph Workflow (`app/core/`)

The central orchestration is a `StateGraph` compiled in `workflow.py`. Every chat request flows through these nodes in order:

```
load_memory → classify_intent → [mental_agent | nutrition_agent | general_agent] → save_memory → [consolidate_memory?] → END
```

All nodes share a single `AgentState` TypedDict (defined in `state.py`). Nodes communicate exclusively by reading and writing fields on this state — never by calling each other directly.

`get_workflow()` is a lazy singleton that compiles the graph once; `run_workflow()` is the REST chat entry point.

**Intent routing** (`intent_router.py`): Uses LLM JSON output to classify each message into `mental`, `nutrition`, or `general`. Confidence below 0.7 falls back to `general`.

### Agents (`app/agents/`)

`BaseAgent` defines the interface: `run(state) -> AgentState` (blocking) and `stream(state) -> AsyncGenerator[str, None]` (token streaming). Each subclass writes its reply to `state["response"]`.

| Agent | LLM backend | Pattern |
|---|---|---|
| `MentalAgent` | `vllm` (local fine-tuned `deepseek-8b-qlora`) | direct chat |
| `NutritionAgent` | `general` (DeepSeek API) | Plan-Execute: LLM plans tasks → parallel execution (health calc / recipe search / KG query) → synthesize |
| `GeneralAgent` | `general` (DeepSeek API) | direct chat |

`NutritionAgent` uses `asyncio.gather` to run sub-tasks in parallel. Tool results populate `state["tool_calls"]`, `state["rag_context"]`, and `state["kg_context"]` before the synthesize step.

Note: `workflow.py` and `chat.py` each instantiate their own agent singletons — workflow agents handle REST, `chat.py` agents handle WebSocket streaming.

### Memory System (`app/memory/`)

Both `MemoryManager` and `MemoryConsolidator` are fully implemented (not Phase 5 stubs).

- `MemoryManager.load_short_term()` — fetches the last `SHORT_TERM_WINDOW` (default: 10) messages from the current session.
- `MemoryManager.load_long_term()` — fetches the latest `LONG_TERM_TOP_K` (default: 5) summarized memories for the user.
- `MemoryConsolidator.consolidate()` — calls `complete_json()` to compress a conversation into `{summary, key_topics, emotional_state, profile_updates}`. Triggered every `CONSOLIDATION_INTERVAL` (default: 10) turns.

Stub nodes in `workflow.py` (`load_memory_node`, `save_memory_node`, `consolidate_memory_node`) are marked `# TODO: Phase 5` — they initialize/count state but do not call `MemoryManager`. The actual memory load/save happens in `chat.py` before and after `run_workflow()`.

### LLM Client (`app/llm/client.py`)

`LLMClient` wraps two `AsyncOpenAI` backends:
- `general` — DeepSeek OpenAI-compatible API (intent classification, general/nutrition agents, memory consolidation)
- `vllm` — locally deployed fine-tuned model (mental agent; defaults to `deepseek-8b-qlora`)

Use `get_llm_client()` singleton. `complete_json()` enforces `response_format: json_object` and parses the result (with markdown code block fallback).

### RAG Layer (`app/rag/`)

Phase 3 stubs. Embedder uses `BAAI/bge-small-zh-v1.5` (512-dim Chinese model, auto-downloaded on first use via `sentence-transformers`). `recipe_retriever.py` queries ChromaDB; `kg_retriever.py` queries Neo4j via Cypher. `scripts/seed_recipes.py` populates the recipe vector store.

### Health Tools (`app/tools/health_calc.py`)

Pure functions for BMI (Chinese standard), BMR (Mifflin-St Jeor), and TDEE. Exposed as REST endpoints in `app/api/v1/health.py`. Also called directly by `NutritionAgent` during task execution.

### Database Layer (`app/db/`)

- **PostgreSQL** — SQLAlchemy async engine (`postgres.py`). `get_db_session()` is the FastAPI dependency; `async_session_factory` is used directly in WebSocket handlers. `init_db()` creates tables via `Base.metadata.create_all` (dev only; use Alembic in production).
- **Redis** — session caching via `redis_client.py`.
- **Neo4j + ChromaDB** — Phase 3 stubs.

ORM note: the `Message.metadata_` column maps to the DB column `metadata` (renamed to avoid shadowing the SQLAlchemy built-in).

### Configuration (`app/config.py`)

All settings loaded from `.env` via pydantic-settings. Call `get_settings()` (cached with `lru_cache`) — never instantiate `Settings` directly.

## Development Phases

- **Phase 1** ✅: FastAPI skeleton, DB layer, auth, health-calc tools
- **Phase 2** ✅: LangGraph workflow + intent router + all three agents
- **Phase 3** 🔲: RAG (ChromaDB recipe search + Neo4j KG) — stubs in `app/rag/`
- **Phase 4** 🔲: Fine-tuned mental health model (vLLM deployment)
- **Phase 5** 🔲: Hierarchical memory wired into workflow nodes
- **Phase 6** 🔲: Optimization

## Key Conventions

- Logging uses `structlog` via `get_logger(name)` from `app/utils/logger.py`. Use `await logger.ainfo(...)` in async contexts.
- Custom exceptions live in `app/utils/exceptions.py`; they are registered globally in `app/api/middleware.py`.
- Pydantic schemas (request/response models) are in `app/schemas/`; ORM models are in `app/db/models.py`.
- Prompts are separated into `app/prompts/` — one file per domain (intent, mental, nutrition, system, consolidation).
- Tests use `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).
