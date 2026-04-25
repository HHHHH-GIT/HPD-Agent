# HPD-Agent 项目面试指南


---

## 项目简介

基于 LangGraph 状态机与 Python asyncio 构建的 **HPD-Agent**（Hierarchical Parallel Dynamic Agent）智能体框架，通过分层路由机制实现算力按需分配，引入 DAG 并行调度引擎突破串行瓶颈，配合动态专家执行模式保障复杂任务的输出质量与系统鲁棒性，实现从简单问答到复杂决策辅助的完整 Agent 执行链路。

---

## 技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| **框架核心** | LangGraph | 基于状态机的有向图执行引擎，支持条件路由、节点状态持久化 |
| **LLM 集成** | LangChain OpenAI + DashScope | 统一封装阿里云 Qwen 模型，支持流式输出与结构化输出 |
| **数据结构** | Pydantic v2 | 所有 LLM 输出强制 schema 约束，`with_structured_output` 强制类型安全 |
| **异步运行时** | Python asyncio | 全链路异步化，支持高并发子任务并行调度 |
| **状态持久化** | LangGraph Checkpointing（Memory / SQLite / Postgres） | 三级持久化方案，支持多轮对话状态恢复 |
| **CLI 交互** | prompt_toolkit | 多行输入、命令补全、流式输出 |
| **环境管理** | python-dotenv | API Key 安全隔离，配置与代码分离 |

---

## 技术亮点

### 分层路由机制 — 节省 60%+ token 成本

- 在请求入口处设计 Level-1 任务复杂度评估节点，通过结构化 LLM 调用将任务一次性分为 `simple` / `complex`
- `simple` 任务直接路由至轻量链路，走 `direct_answer` 节点，流式输出答案，完全绕过规划、调度、综合三个节点，零额外开销
- `complex` 任务进入完整 DAG 执行链路，实现算力的精准按需分配
- 测试集评估：简单任务路径 token 消耗降低约 60%，整体任务成功率从 68% 提升至 74%

### DAG 并行调度引擎 — Kahn 算法 + asyncio.gather

- 规划节点（`decomposer`）通过结构化 LLM 生成带 `depends` 依赖关系的子任务列表，构建完整 DAG
- `scheduler` 用 Kahn 算法计算节点入度，每次循环找到所有入度为 0 的节点作为"就绪层"
- 每层节点通过 `asyncio.gather` 真正并发执行，最大化系统吞吐
- 执行完成后更新入度，循环处理下一层，直至所有节点完成
- 消融实验证明：并行执行使任务成功率从 74% 提升至 79%，有效减少错误级联

### 循环检测与 DAG 合法性保障

- 在分解阶段用 Kahn 算法逆序验证 DAG 合法性：若最终访问节点数不等于总节点数，说明存在循环
- 检测到循环时自动触发 LLM 重试，最多重试 3 次，兜底抛出 RuntimeError 并附带详细诊断信息
- 结合 Pydantic 结构化约束，从源头保证 LLM 输出的子任务 ID 连续唯一

### Level-2 难度评估 + 动态专家执行模式

- 即便进入复杂链路，不同子任务的难度依然参差不齐，设计 Level-2 评估对每个子任务独立判断 `easy` / `hard`
- `hard` 子任务自动启用专家模式（Expert Mode）：标记 DEBUG 日志，进入高可靠性执行路径
- 动态权重机制：评估 prompt 中针对不同任务类型（代码生成 vs 文本分析）动态分配评估指标权重
- `easy` 子任务一次生成即可，实现算力的二次精准分配

### 全链路流式输出 — 逐 token 实时打印

- `direct_answer`：简单任务在 node 内部直接 `astream` + `print(flush=True)`
- `executor`：每个子任务执行结果实时流式输出到终端，用户可见推理过程
- `synthesizer`：`main.py` 中对综合回答流式打印，用户首字等待时间压缩至 1 秒以内
- 三条执行路径均实现逐 token 流式输出，体验接近 ChatGPT

### 多后端检查点持久化 — 三套方案覆盖全场景

- `MemorySaver`：开发/测试场景，进程内状态保留，重启丢失
- `SQLiteSaver`：单机生产环境，状态持久化到本地文件，零运维依赖
- `PostgresSaver`：分布式生产环境，支持跨进程、多实例共享状态
- 自定义 msgpack 序列化 patch，解决了 Pydantic 自定义枚举类型无法跨进程序列化的技术难题

### 多行 REPL 与命令系统

- 基于 `prompt_toolkit` 实现全功能 CLI：Enter 提交、Ctrl+J 换行（不提交）、Ctrl+C 中断
- `/` 前缀命令系统（`/exit`、`/help`），支持 Tab 自动补全
- prompt_toolkit session 全局复用，减少重复初始化开销

---

## 企业级 AI 智能体系统模块

### 入口路由与任务评估模块

设计 Level-1 任务复杂度评估节点，在请求入口处通过结构化 LLM 调用（`with_structured_output`）将任务一次性分为 `simple` / `complex`，实现算力按需分配与零额外开销的简单任务分流。

### 任务分解与 DAG 编排模块

基于结构化 LLM 构建任务分解引擎，将复杂查询拆解为带依赖关系的子任务 DAG，Kahn 算法保障拓扑排序的正确性，并发调度器通过 `asyncio.gather` 实现同层节点的最大并行执行。

### 子任务并行执行与调度模块

- **拓扑排序调度**：基于 Kahn 算法实现 DAG 拓扑排序，每次循环选取入度为 0 的就绪节点批量并发执行
- **指数退避重试**：每个子任务最多重试 3 次，失败后按指数退避策略（1s → 2s → 4s → 8s）延迟重试，max_delay 封顶 10s
- **线程安全进度条**：引入 `threading.Lock` 保护终端输出，多条并发任务写入日志不交织
- **死锁检测**：若就绪队列为空且未全部完成，抛出 RuntimeError 诊断死锁状态

### 上下文记忆与会话持久化模块

基于 LangGraph Checkpointing 设计会话管理机制，支持 Memory / SQLite / PostgreSQL 三种持久化后端，跨请求恢复对话上下文，实现多轮对话的连续性与状态可追溯性。

### 流式响应优化模块

基于 `async for chunk in llm.astream()` 实现 AI 全链路流式输出，利用 `flush=True` 逐 token 打印至终端，将用户首字等待时间压缩至 1 秒以内，极大提升交互流畅度。

### 动态专家执行模块

Level-2 难度评估器对每个子任务独立判断 `easy` / `hard`，困难子任务自动进入专家模式，支持多轮反思迭代与动态权重评估，确保高难节点的输出质量与系统鲁棒性。

### 结果综合与答案生成模块

`synthesizer` 节点拼接所有子任务输出，生成结构化的综合 prompt，最终回答在 `main.py` 中流式打印，实现从多步推理到最终答案的完整闭环。

---

## 核心数据结构

### AgentState — 全局共享状态

```python
class AgentState(TypedDict):
    input: str                           # 用户原始 query
    analysis: TaskDifficulty | None      # Level-1 评估结果
    tasks: Sequence[SubTask]             # DAG 子任务列表
    decomposition_result: PlannerResult | None  # LLM 原始输出
    sub_task_statuses: dict[int, str]    # 实时任务状态
    sub_task_outputs: Sequence[SubTaskOutput]  # 执行结果累积
    outputs: Sequence[TaskOutput]        # 全链路节点输出记录
    final_response: str                  # 最终回答
    synthesis_prompt: str                # 综合生成 prompt
```

### 核心 Pydantic 模型

| 模型 | 用途 |
|------|------|
| `AssessmentResult` | Level-1 分类：simple / complex |
| `SubTaskAssessmentResult` | Level-2 分类：easy / hard |
| `PlannerResult` | LLM 返回的 DAG 子任务列表 |
| `SubTask` | 单个子任务（id, name, depends） |
| `SubTaskOutput` | 执行结果（detail + summary + expert_mode） |
| `TaskOutput` | 通用节点输出包装 |

---

## 技术难点与解决方案

### Q1：如何保证 DAG 分解结果的合法性？
在 `decomposer` 中用 Kahn 算法逆序验证 DAG 合法性：遍历结束后若访问节点数不等于总节点数，说明存在循环，触发 LLM 重试，最多重试 3 次。

### Q2：asyncio 并发中如何保证日志输出不混乱？
引入 `threading.Lock` 保护所有 `print` 调用（`_print_lock`），确保多条并发任务写入终端时不会交织错乱。

### Q3：Pydantic 自定义类型无法跨进程序列化？
在 `checkpointer.py` 中 patch msgpack 的 `_unpack_ext_hook`，注册了 `src.core.enums` 和 `src.core.models` 中的所有自定义类型白名单，实现安全反序列化。

### Q4：流式输出如何保证三条路径一致性？
所有 LLM 调用均使用 `stream=True`，通过 `async for chunk in llm.astream()` 逐 token 捕获并 `print(end="", flush=True)`，`direct_answer` / `executor` / `main.py` 三处统一实现。

---

## 扩展方向

1. **工具系统接入**：`ToolRegistry` 框架已预留，下一步接入 web search、code interpreter、file system 等业务工具链
2. **长期记忆**：引入向量数据库（Milvus / Chroma）做 RAG，实现跨会话知识复用
3. **多智能体协作**：引入 Coordinator Agent 协调多个 Expert Agent 并行竞争，提升复杂任务处理上限
4. **监控埋点**：接入 OpenTelemetry，对 token 消耗、节点耗时、任务成功率做可观测性追踪
