# Build Log — 多智能体健康咨询系统

## Phase 1 — FastAPI 骨架 + 基础设施
**状态**: ✅ 完成

### 完成内容
- FastAPI 应用入口 (`app/main.py`)
- Pydantic-Settings 配置 (`app/config.py`)
- PostgreSQL 异步引擎 + ORM 模型 (User, UserProfile, ChatSession, Message)
- Redis 客户端
- JWT 认证 (注册/登录/依赖注入)
- 健康指标工具: BMI、BMR、TDEE (`app/tools/health_calc.py`)
- REST API: `/api/v1/users`, `/api/v1/health`
- 自定义异常 + 全局中间件
- Docker Compose (PostgreSQL, Redis, Neo4j, ChromaDB)
- 基础测试 (`tests/test_health_calc.py`)

---

## Phase 2 — 多智能体 LangGraph 工作流
**状态**: ✅ 完成

### 完成内容
- **LLM 客户端** (`app/llm/client.py`): 封装 DeepSeek/OpenAI-compatible + VLLM 双后端，支持同步/流式/JSON 调用
- **AgentState** (`app/core/state.py`): LangGraph 共享状态 TypedDict
- **意图路由** (`app/core/intent_router.py`): LLM JSON 三分类 (mental/nutrition/general)，低置信度自动 fallback
- **LangGraph 工作流** (`app/core/workflow.py`):
  ```
  load_memory → classify_intent → [mental|nutrition|general] → save_memory → [consolidate?] → END
  ```
- **Agent 三件套**:
  - `MentalAgent`: VLLM 微调模型 + general fallback + 危机关键词检测 (7×24 热线)
  - `NutritionAgent`: Plan-Execute 模式 (规划 → 并行执行 → 综合回复)
  - `GeneralAgent`: 通用健康问答
- **Prompt 模板** (`app/prompts/`): 意图分类、心理、营养、通用、记忆整理
- **聊天 API** (`app/api/v1/chat.py`): REST POST `/chat` + WebSocket `/ws/chat`
- **测试**: `test_workflow.py`、`test_intent_router.py`

### 关键设计决策
- DeepSeek API (OpenAI-compatible) 替换 OpenAI，配置项 `DEEPSEEK_API_KEY` + `DEEPSEEK_BASE_URL`
- VLLM 后端为 MentalAgent 专用，不可用时自动降级到通用模型

---

## Phase 3 — 营养师 RAG 系统
**状态**: ✅ 完成  
**完成日期**: 2026-05-06

### 新增文件
| 文件 | 说明 |
|------|------|
| `app/db/neo4j_client.py` | Neo4j 异步驱动封装，单例 + 错误处理 |
| `app/rag/kg_retriever.py` | 营养学知识图谱 Cypher 查询，支持 1-2 跳，含 KG 种子数据 |
| `scripts/seed_recipes.py` | 15 条中餐菜谱写入 ChromaDB（早餐/减脂/增肌/糖尿病友好等） |
| `tests/test_rag.py` | RAG 组件单元测试（ChromaDB 检索、过敏原过滤、KG 三元组） |

### 修改文件
| 文件 | 变更 |
|------|------|
| `app/agents/nutrition_agent.py` | `_execute_recipe_search` → 调用 `search_recipes()`；`_execute_kg_query` → 调用 `query_nutrition_knowledge()` |

### 知识图谱 Schema
```
(:Food)-[:CONTAINS {amount, unit}]->(:Nutrient)
(:Food)-[:BENEFITS]->(:Condition)
(:Food)-[:CONTRAINDICATED_FOR]->(:Condition)
(:Nutrient)-[:TREATS]->(:Condition)
```
内置中文关系翻译: CONTAINS→含有、BENEFITS→有益于、CONTRAINDICATED_FOR→禁忌于

### 菜谱数据分类
- 早餐: 燕麦粥、全麦三明治、蒸蛋羹、小米南瓜粥
- 减脂: 凉拌黄瓜、蒸红薯、苦瓜炒蛋、杂粮饭
- 增肌: 西兰花炒鸡胸、水煮牛肉、全麦三明治
- 补铁/补血: 菠菜猪肝汤
- 心血管健康: 清蒸鲈鱼、三文鱼牛油果沙拉
- 糖尿病友好: 燕麦粥、苦瓜炒蛋、蒸红薯、杂粮饭

---

## Phase 4 — 心理咨询模型集成
**状态**: ✅ 基础架构完成 (模型训练为外部流程)

### 说明
VLLM 后端集成已完整实现于 Phase 2：
- MentalAgent 优先调用 `backend="vllm"` (VLLM 部署的 deepseek-8b-qlora)
- VLLM 不可用时自动 fallback 到 DeepSeek general 模型
- docker-compose.yml 预留 VLLM 服务配置位置

### 待完成 (需外部资源)
- QLora 微调数据集准备
- 模型训练脚本
- VLLM 服务 Docker 配置

---

## Phase 5 — 分层记忆系统
**状态**: ✅ 完成  
**完成日期**: 2026-05-06

### 新增文件
| 文件 | 说明 |
|------|------|
| `app/memory/manager.py` | MemoryManager: 短期(DB消息窗口) + 长期(摘要)加载/保存 |
| `app/memory/consolidator.py` | MemoryConsolidator: LLM 整理对话 → 结构化摘要 |
| `app/memory/__init__.py` | 导出 MemoryManager、MemoryConsolidator |
| `tests/test_memory.py` | 记忆系统单元测试 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `app/db/models.py` | 新增 `LongTermMemory` ORM 表；User 添加 `long_term_memories` 关系 |
| `app/agents/base.py` | 添加 `_get_long_term_memory(state)` 辅助方法 |
| `app/agents/general_agent.py` | `_build_messages` 注入长期记忆摘要 |
| `app/core/workflow.py` | `run_workflow` 新增 `long_term_memory` 参数并传入 initial_state |
| `app/api/v1/chat.py` | 集成 MemoryManager + MemoryConsolidator，REST 端点自动触发整理 |

### 分层记忆架构
```
短期记忆 (short_term_memory)
  ├─ 存储: PostgreSQL messages 表，最近 SHORT_TERM_WINDOW (=10) 条
  └─ 用途: 上下文连续性，直接注入 Agent prompt

长期记忆 (long_term_memory)
  ├─ 存储: PostgreSQL long_term_memories 表
  ├─ 触发: 每 CONSOLIDATION_INTERVAL (=10) 轮 LLM 整理一次
  └─ 用途: 跨会话用户画像更新、历史摘要注入
```

### 整理流程
1. 每 10 轮对话 `save_memory_node` 设置 `should_consolidate=True`
2. `chat.py` 检测到标志后调用 `MemoryConsolidator.consolidate()`
3. LLM 输出: `{summary, profile_updates, key_topics, emotional_state}`
4. `MemoryManager.save_long_term()` 持久化到 DB

---

## Phase 6 — 优化与测试
**状态**: ✅ 基础完成  
**完成日期**: 2026-05-06

### 测试覆盖
| 测试文件 | 覆盖内容 |
|----------|----------|
| `tests/test_health_calc.py` | BMI/BMR/TDEE 计算正确性 |
| `tests/test_intent_router.py` | 意图分类、低置信度 fallback、LLM 错误 fallback |
| `tests/test_workflow.py` | 节点状态流转、路由函数、危机检测 |
| `tests/test_rag.py` | ChromaDB 检索、过敏原过滤、Neo4j KG 查询 |
| `tests/test_memory.py` | 记忆整理、短/长期加载保存、工作流集成 |

### 待优化项
- [ ] 添加 Alembic 数据库迁移脚本
- [ ] NutritionAgent 菜谱精排 (reranker)
- [ ] WebSocket 端点集成 MemoryManager
- [ ] 添加请求速率限制
- [ ] 生产环境监控 (Prometheus metrics)

---

## 依赖汇总

```toml
# 新增依赖 (已在 pyproject.toml 中)
neo4j>=5.25.0           # Neo4j 异步驱动
chromadb>=0.5.0         # 向量数据库
sentence-transformers>=3.1.0  # BAAI/bge-small-zh-v1.5 中文嵌入
```

## 配置项

```env
# .env 必填
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
GENERAL_MODEL=deepseek-chat

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIR=./data/chroma

SHORT_TERM_WINDOW=10
CONSOLIDATION_INTERVAL=10
LONG_TERM_TOP_K=5
```
