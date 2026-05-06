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

| Agent | 职责 | 后端 |
|-------|------|------|
| **MentalAgent** | 心理咨询、情绪支持，内置危机关键词检测 | VLLM (fallback → DeepSeek) |
| **NutritionAgent** | Plan-Execute 模式：规划 → 并行执行健康计算/菜谱检索/知识图谱 → 综合回复 | DeepSeek |
| **GeneralAgent** | 运动、疾病科普、养生等通用健康问答 | DeepSeek |

---

## 快速启动

### 1. 环境准备

```bash
# 需要 Python 3.11+
cd health-advisor
pip install -e ".[dev]"

# 复制并编辑配置
cp .env.example .env
```

`.env` 必填项：
```env
DEEPSEEK_API_KEY=sk-你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
GENERAL_MODEL=deepseek-chat
```

### 2. 启动基础设施

```bash
# 最小启动（仅 PostgreSQL + Redis）
docker compose up -d postgres redis

# 全量启动（含 RAG 组件）
docker compose up -d
```

### 3. 写入菜谱数据（首次运行）

```bash
python scripts/seed_recipes.py
```

### 4. 启动应用

```bash
uvicorn app.main:app --reload --port 8000
```

- Swagger UI：http://localhost:8000/docs
- 健康检查：http://localhost:8000/healthz

---

## API 快速测试

```bash
# 注册
curl -X POST http://localhost:8000/api/v1/users/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "123456"}'

# 登录
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "123456"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 聊天（新会话）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "我最近压力很大，总是失眠"}'

# 继续对话（同一会话，替换 SESSION_ID）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "message": "有什么缓解的方法吗"}'

# 营养咨询（触发 RAG + 健康计算）
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "推荐几道适合减脂的低热量晚餐"}'

# BMI 计算
curl -X POST http://localhost:8000/api/v1/health/bmi \
  -H "Content-Type: application/json" \
  -d '{"weight_kg": 70, "height_cm": 175}'
```

### WebSocket 流式聊天

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

---

## 项目结构

```
app/
├── main.py              # FastAPI 入口 + lifespan
├── config.py            # 配置 (pydantic-settings, 从 .env 加载)
├── api/v1/              # REST + WebSocket 端点
│   ├── chat.py          #   聊天 (POST /chat, WS /ws/chat)
│   ├── users.py         #   用户注册/登录/档案
│   └── health.py        #   BMI/BMR/TDEE 计算
├── core/
│   ├── workflow.py      # LangGraph StateGraph 编排
│   ├── intent_router.py # LLM 意图三分类 (mental/nutrition/general)
│   └── state.py         # AgentState TypedDict
├── agents/
│   ├── base.py          # BaseAgent 抽象基类
│   ├── mental_agent.py  # 心理咨询 + 危机检测
│   ├── nutrition_agent.py # Plan-Execute 营养师
│   └── general_agent.py # 通用健康顾问
├── llm/
│   └── client.py        # LLMClient (DeepSeek + VLLM 双后端)
├── rag/
│   ├── embedder.py      # SentenceTransformer (BAAI/bge-small-zh-v1.5)
│   ├── recipe_retriever.py # ChromaDB 菜谱语义检索
│   └── kg_retriever.py  # Neo4j 营养知识图谱 (1-2 跳 Cypher)
├── memory/
│   ├── manager.py       # 短期消息窗口 + 长期摘要 加载/保存
│   └── consolidator.py  # LLM 记忆整理 → 结构化摘要
├── db/
│   ├── models.py        # ORM: User/ChatSession/Message/LongTermMemory
│   ├── postgres.py      # SQLAlchemy async engine
│   ├── redis_client.py  # Redis 会话缓存
│   └── neo4j_client.py  # Neo4j 异步驱动
├── tools/
│   └── health_calc.py   # BMI / BMR / TDEE 计算工具
├── prompts/             # Prompt 模板 (按 Agent 分文件)
└── schemas/             # Pydantic 请求/响应模型

scripts/
└── seed_recipes.py      # ChromaDB 菜谱数据初始化 (15 条)

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

Agent 接收到的上下文层次：
1. 系统 Prompt（角色设定）
2. 用户健康档案（UserProfile）
3. **长期记忆摘要**（跨会话历史）
4. **短期记忆**（近期对话）
5. RAG 检索结果（仅营养 Agent）
6. 用户当前消息

---

## 营养知识图谱 Schema

```cypher
(:Food)-[:CONTAINS {amount, unit}]->(:Nutrient)
(:Food)-[:BENEFITS]->(:Condition)
(:Food)-[:CONTRAINDICATED_FOR]->(:Condition)
(:Nutrient)-[:TREATS]->(:Condition)
```

首次启动后可运行种子数据（苹果、菠菜、鸡胸肉等 7 种食材 + 疾病关系）：

```python
# 在应用内部调用
from app.rag.kg_retriever import seed_nutrition_kg
await seed_nutrition_kg()  # 仅在 Neo4j 为空时写入
```

---

## 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | API 基础地址 |
| `GENERAL_MODEL` | `deepseek-chat` | 通用对话模型 |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | VLLM 微调模型地址 |
| `SHORT_TERM_WINDOW` | `10` | 短期记忆消息条数 |
| `CONSOLIDATION_INTERVAL` | `10` | 长期记忆整理间隔（轮） |
| `LONG_TERM_TOP_K` | `5` | 加载长期记忆条数 |
| `RECIPE_TOP_K` | `20` | 菜谱检索召回数量 |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB 本地存储路径 |

---

## 运行测试

```bash
pytest tests/ -v --cov=app
```

覆盖范围：健康计算、意图路由、工作流节点、RAG 检索、记忆系统（共 43 个测试用例）

---

## 依赖项

```toml
fastapi, uvicorn          # Web 框架
langgraph, langchain-core # 多智能体编排
openai                    # DeepSeek / VLLM 调用 (OpenAI-compatible)
sqlalchemy, asyncpg       # PostgreSQL async ORM
chromadb                  # 向量数据库
sentence-transformers     # 中文嵌入 (BAAI/bge-small-zh-v1.5)
neo4j                     # 知识图谱
redis                     # 会话缓存
passlib[bcrypt]           # 密码哈希 (需 bcrypt<4.0)
python-jose               # JWT 认证
structlog                 # 结构化日志
pydantic-settings         # 配置管理
```

> ⚠️ **注意**：passlib 1.7.4 与 bcrypt 4.x 不兼容，安装时请固定 `bcrypt<4.0`。

---

## 开发路线图

- [x] Phase 1: 项目骨架、PostgreSQL/Redis、JWT 认证、健康计算工具
- [x] Phase 2: LangGraph 工作流、意图路由、三类 Agent、DeepSeek LLM 客户端
- [x] Phase 3: ChromaDB 菜谱 RAG + Neo4j 营养知识图谱
- [x] Phase 5: 分层记忆系统（短期消息窗口 + LLM 长期摘要整理）
- [ ] Phase 4: 心理咨询 QLora 微调模型（VLLM 服务）
- [ ] Alembic 数据库迁移脚本
