# 多智能体健康咨询系统 (Health Advisor)

基于 FastAPI + LangGraph 构建的多智能体健康咨询系统，集成 DeepSeek LLM、ChromaDB 语义检索、Neo4j 知识图谱和分层记忆管理。

## 系统架构

```
REST/WebSocket → Chat API → LangGraph Workflow → Specialized Agent → LLM (DeepSeek)
                                                         ↓
                                          ChromaDB (菜谱 RAG) / Neo4j (知识图谱)
                                                         ↓
                                          分层记忆 (短期消息窗口 + 长期摘要)
```

### 工作流节点

```
load_memory → classify_intent → [mental_agent | nutrition_agent | general_agent]
                                          → save_memory → [consolidate_memory?] → END
```

### 三类 Agent

| Agent | 职责 | LLM 后端 |
|-------|------|----------|
| **MentalAgent** | 心理咨询、情绪支持，内置危机关键词检测 + 求助热线附加 | VLLM (自动 fallback → DeepSeek) |
| **NutritionAgent** | Plan-Execute 模式：规划 → 并行执行健康计算/菜谱检索/知识图谱 → 综合回复 | DeepSeek |
| **GeneralAgent** | 运动、疾病科普、养生等通用健康问答 | DeepSeek |

---

## 快速启动

### 1. 环境准备

```bash
# 需要 Python 3.11+
pip install -e ".[dev]"
```

创建 `.env` 文件（无 `.env.example`，按以下模板手动创建）：

```env
# 必填
DEEPSEEK_API_KEY=sk-你的key
JWT_SECRET=your-secret-key-change-in-prod

# 可选（以下均为默认值）
DEEPSEEK_BASE_URL=https://api.deepseek.com
GENERAL_MODEL=deepseek-chat
POSTGRES_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/health_advisor
REDIS_URL=redis://localhost:6379/0
VLLM_BASE_URL=http://localhost:8001/v1
DEBUG=true
```

> ⚠️ **依赖注意**：`passlib 1.7.4` 与 `bcrypt 4.x` 不兼容，如遇报错请固定 `pip install "bcrypt<4.0"`。

### 2. 启动基础设施

```bash
# 最小启动（仅 PostgreSQL + Redis）
docker compose up -d postgres redis

# 全量启动（含 ChromaDB + Neo4j RAG 组件）
docker compose up -d
```

数据库表在应用首次启动时由 `init_db()` 自动创建，无需手动执行迁移（生产环境请改用 Alembic）。

### 3. 初始化数据（首次运行）

```bash
# ChromaDB 写入 15 条示例中餐菜谱
python scripts/seed_recipes.py
```

```python
# Neo4j 写入营养知识图谱（7 种食材 + 营养素 + 疾病关系，应用内调用）
from app.rag.kg_retriever import seed_nutrition_kg
await seed_nutrition_kg()  # 仅在图谱为空时写入
```

### 4. 启动应用

```bash
uvicorn app.main:app --reload --port 8000
```

- Swagger UI：http://localhost:8000/docs （需 `DEBUG=true`）
- 健康检查：http://localhost:8000/healthz

---

## API 快速测试

### 用户认证

```bash
# 注册
curl -X POST http://localhost:8000/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "123456"}'

# 登录，获取 JWT
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "123456"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 更新健康档案（影响营养 Agent 的个性化回复）
curl -X PUT http://localhost:8000/api/v1/users/profile \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "gender": "male", "height_cm": 175, "weight_kg": 70, "allergies": ["花生"]}'
```

### 聊天（REST）

```bash
# 心理咨询（新会话）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "我最近压力很大，总是失眠"}'

# 继续同一会话（替换 SESSION_ID）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "message": "有什么缓解的方法吗"}'

# 营养咨询（触发 Plan-Execute：健康计算 + 菜谱检索 + 知识图谱）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "推荐几道适合减脂的低热量晚餐"}'
```

### 健康指标计算（无需认证）

```bash
# BMI
curl -X POST http://localhost:8000/api/v1/health/bmi \
  -H "Content-Type: application/json" \
  -d '{"weight_kg": 70, "height_cm": 175}'

# BMR（基础代谢率，Mifflin-St Jeor 公式）
curl -X POST http://localhost:8000/api/v1/health/bmr \
  -H "Content-Type: application/json" \
  -d '{"weight_kg": 70, "height_cm": 175, "age": 30, "gender": "male"}'

# TDEE（每日总能量消耗）
curl -X POST http://localhost:8000/api/v1/health/tdee \
  -H "Content-Type: application/json" \
  -d '{"weight_kg": 70, "height_cm": 175, "age": 30, "gender": "male", "activity_level": "moderate"}'
```

`activity_level` 可选值：`sedentary` / `light` / `moderate` / `active` / `very_active`

### WebSocket 流式聊天

WebSocket 通过 `?token=` 查询参数认证（不支持 Header 方式）：

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/chat?token=YOUR_TOKEN');

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  // msg.type: "intent" | "tool" | "token" | "done" | "error"
  if (msg.type === 'token') process.stdout.write(msg.data.content);
  if (msg.type === 'done')  console.log('\n延迟:', msg.data.latency_ms, 'ms');
};

ws.send(JSON.stringify({
  content: "推荐适合高血压患者的饮食方案",
  session_id: "可选，传入复用历史会话"
}));
```

**消息类型说明：**

| type | 触发时机 | data 内容 |
|------|----------|-----------|
| `intent` | 意图识别完成 | `{intent, confidence}` |
| `tool` | 营养 Agent 执行工具前 | `{tool, status}` |
| `token` | 每个流式 token | `{content}` |
| `done` | 本轮完成 | `{session_id, latency_ms, intent, tool_calls}` |
| `error` | 出错 | `{message}` |

---

## 项目结构

```
app/
├── main.py              # FastAPI 入口 + lifespan (自动建表、初始化 Redis)
├── config.py            # 配置 (pydantic-settings，从 .env 加载，用 get_settings() 获取)
├── api/v1/              # 路由前缀 /api/v1
│   ├── chat.py          #   POST /chat, WS /ws/chat
│   ├── users.py         #   /users/register, /login, /me, /profile
│   └── health.py        #   /health/bmi, /bmr, /tdee（无需认证）
├── core/
│   ├── workflow.py      # LangGraph StateGraph 编排 + run_workflow() 入口
│   ├── intent_router.py # LLM 意图三分类 (mental/nutrition/general，置信度<0.7 fallback)
│   └── state.py         # AgentState TypedDict（所有节点共享）
├── agents/
│   ├── base.py          # BaseAgent：run() + stream() 接口
│   ├── mental_agent.py  # 心理咨询 + 危机关键词检测
│   ├── nutrition_agent.py # Plan-Execute：asyncio.gather 并行执行子任务
│   └── general_agent.py # 通用健康顾问
├── llm/
│   └── client.py        # LLMClient：complete() / stream() / complete_json()
├── rag/
│   ├── embedder.py      # SentenceTransformer (BAAI/bge-small-zh-v1.5，首次自动下载)
│   ├── recipe_retriever.py # ChromaDB 语义检索 + 过敏原过滤
│   └── kg_retriever.py  # Neo4j 1-2 跳 Cypher 查询 + seed_nutrition_kg()
├── memory/
│   ├── manager.py       # 短期消息窗口加载/保存 + 长期摘要持久化
│   └── consolidator.py  # LLM 将对话压缩为 {summary, key_topics, emotional_state}
├── db/
│   ├── models.py        # ORM: User / UserProfile / ChatSession / Message / LongTermMemory
│   ├── postgres.py      # SQLAlchemy async engine + async_session_factory
│   ├── redis_client.py  # Redis 初始化
│   └── neo4j_client.py  # Neo4j 异步驱动 + run_query()
├── tools/
│   └── health_calc.py   # BMI / BMR / TDEE 纯函数（中国 BMI 标准）
├── prompts/             # Prompt 模板，按 Agent 分文件
└── schemas/             # Pydantic 请求/响应模型

scripts/
├── seed_recipes.py      # ChromaDB 写入 15 条示例中餐菜谱
└── data_soulchat.py     # SoulChat 数据集转换为 LlamaFactory ShareGPT 格式

train/
├── deepseek_8b_qlora.yaml  # QLoRA 训练配置（Qwen2.5-3B-Instruct，4-bit）
└── merge_lora.py           # 训练后 LoRA 权重合并脚本

deploy/
├── docker-compose.vllm.yml # vLLM Docker 部署（端口 8001）
├── start_vllm.sh           # vLLM 启动脚本
└── test_vllm.py            # vLLM 集成测试（5 个用例）

run_train.py               # Phase 4 训练入口（解决 Windows CLI 静默退出问题）

tests/
├── test_health_calc.py
├── test_intent_router.py
├── test_workflow.py
├── test_rag.py
└── test_memory.py
```

---

## 分层记忆机制

```
短期记忆 ── PostgreSQL messages 表，最近 SHORT_TERM_WINDOW(=10) 条
              ↓ 每 CONSOLIDATION_INTERVAL(=10) 轮触发
长期记忆 ── LLM 整理 → {summary, key_topics, emotional_state}
              ↓ 持久化到 long_term_memories 表，供下次对话加载
```

每次请求进入 `chat.py` 时，先通过 `MemoryManager` 加载短期/长期记忆注入 AgentState，响应后再触发整理。Agent 接收到的上下文层次：

1. 系统 Prompt（角色设定）
2. 用户健康档案（UserProfile）
3. **长期记忆摘要**（跨会话历史，最近 5 条）
4. **短期记忆**（当前会话最近 10 条）
5. RAG 检索结果（仅营养 Agent）
6. 用户当前消息

---

## 营养知识图谱 Schema

```cypher
(:Food {name, aliases, calories_per_100g})
(:Nutrient {name, unit, daily_recommended})
(:Condition {name, description})

(:Food)-[:CONTAINS {amount, unit}]->(:Nutrient)
(:Food)-[:BENEFITS]->(:Condition)
(:Food)-[:CONTRAINDICATED_FOR]->(:Condition)
(:Nutrient)-[:TREATS]->(:Condition)
```

种子数据包含苹果、菠菜、鸡胸肉、豆腐、三文鱼、燕麦、苦瓜等 7 种食材，以及糖尿病、高血压、贫血、肥胖、心血管疾病等关联关系。

---

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | **必填**，DeepSeek API Key |
| `JWT_SECRET` | `change-me-in-production` | **必填**，生产环境须替换 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `GENERAL_MODEL` | `deepseek-chat` | 通用对话模型名称 |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | 本地微调模型地址（MentalAgent） |
| `VLLM_MODEL` | `deepseek-8b-qlora` | vLLM `--served-model-name` 须与此一致 |
| `SHORT_TERM_WINDOW` | `10` | 短期记忆消息条数 |
| `CONSOLIDATION_INTERVAL` | `10` | 长期记忆整理间隔（轮次） |
| `LONG_TERM_TOP_K` | `5` | 加载长期记忆条数 |
| `RECIPE_TOP_K` | `20` | 菜谱检索召回数量 |
| `RERANK_TOP_K` | `10` | 重排序后保留数量 |
| `KG_MAX_HOPS` | `2` | 知识图谱最大查询跳数 |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB 本地存储路径 |

---

## 运行测试

```bash
pytest tests/ -v --cov=app
```

---

## 开发路线图

- [x] Phase 1：项目骨架、PostgreSQL/Redis、JWT 认证、BMI/BMR/TDEE 工具
- [x] Phase 2：LangGraph 工作流、意图路由、三类 Agent、DeepSeek LLM 客户端
- [x] Phase 3：ChromaDB 菜谱 RAG + Neo4j 营养知识图谱
- [x] Phase 5：分层记忆系统（MemoryManager + MemoryConsolidator，在 chat.py 中生效）
- [🔄] Phase 4：心理咨询 QLoRA 微调模型训练中（Qwen2.5-3B-Instruct，RTX 3060 6GB）
- [ ] Phase 4 部署：LoRA 合并 → vLLM Docker 部署（端口 8001）
- [ ] Phase 5 完整集成：将 MemoryManager 接入 LangGraph workflow 节点
- [ ] Phase 6：性能优化（Alembic 迁移、RERANK、缓存）

### Phase 4 本地模型训练流程

```bash
# 1. 数据准备
python scripts/data_soulchat.py

# 2. QLoRA 微调（需 NVIDIA GPU，显存 ≥ 6GB）
python run_train.py
# 训练配置：train/deepseek_8b_qlora.yaml（4-bit, rank=32, ~3-4 小时）

# 3. LoRA 权重合并
python train/merge_lora.py
# 合并后模型输出到 models/mental_health_merged/

# 4. 启动 vLLM 推理服务
docker compose -f deploy/docker-compose.vllm.yml up -d

# 5. 验证部署
python deploy/test_vllm.py
```

> 训练完成前，MentalAgent 自动 fallback 到 DeepSeek API，系统功能不受影响。
