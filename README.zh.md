# HPD-Agent

基于 LangGraph 和 Python asyncio 构建的**分层并行动态**智能体框架。

HPD-Agent 将每个传入的查询经过两级评估：简单任务走快速直答路径，复杂任务则被分解为 DAG 并行执行——只在真正需要的地方才动用重计算。

---

## 快速开始

```bash
# pip install .
pip install git+https://github.com/HHHHH-GIT/HPD-Agent.git
cp .env.example .env        # 设置 DEEPSEEK_API_KEY
hpd
```

```
> 你好
# ... 流式响应输出中 ...

> /model create            # 管理 LLM 模型配置
> /context                 # 查看上下文窗口
> /new                     # 开始新会话
> /exit                    # 退出
```

---

## 架构

```
用户查询
    │
    ▼
┌──────────────────────────┐
│  first_level_assessment  │  ← Level-1: simple / complex
└──────────┬───────────────┘
           │
      ┌────┴────┐
      ▼         ▼
  simple     complex
      │         │
      ▼         ▼
direct_    coordinator
answer     (DAG 规划)
      │         │
      ▼         ▼
    END      scheduler_node
             (Kahn 并行执行)
                  │
                  ▼
              synthesizer
              (流式综合)
                  │
                  ▼
                 END
```

**执行路径**

| 路径 | 触发条件 | 步骤 |
|---|---|---|
| 直答 | Level-1 判定为 `simple` | assessment → direct_answer → END |
| 完整 | Level-1 判定为 `complex` | assessment → coordinator → scheduler_node → synthesizer → END |

---

## 功能清单

基于 [HPD-Agent 论文](./HPD-Agent.md)。

| # | 论文功能 | 状态 | 备注 |
|---|---|---|---|
| **2.1 分层路由** | | | |
| 1 | Level-1 评估：simple / complex | **已实现** | `src/nodes/assessment.py` |
| 2 | Level-2 评估：easy / hard（逐子任务） | **已实现** | `src/nodes/execution.py` |
| 3 | 针对 hard 子任务的动态专家模式 | **部分实现** | 状态中记录了 expert_mode 标记，但自我反思迭代循环尚未接入 |
| **2.2 并行执行** | | | |
| 4 | 通过 LLM planner 进行 DAG 分解 | **已实现** | `src/nodes/planning.py` |
| 5 | Kahn 拓扑排序 | **已实现** | `src/nodes/scheduler.py` |
| 6 | 基于入度的就绪层批处理 | **已实现** | `src/nodes/scheduler.py` |
| 7 | `asyncio.gather` 并行执行 | **已实现** | `src/nodes/scheduler.py` |
| 8 | 指数退避重试（1s→2s→4s，上限 10s） | **已实现** | `src/nodes/scheduler.py` |
| 9 | 死锁检测（就绪队列为空但未全部完成） | **已实现** | `src/nodes/scheduler.py` |
| 10 | DAG 环检测 + LLM 重试（最多 3 次） | **已实现** | `src/nodes/planning.py` |
| 11 | 线程安全的进度输出（`threading.Lock`） | **已实现** | `src/nodes/scheduler.py` |
| **2.3 动态专家模式** | | | |
| 12 | 多路候选生成（高温度参数） | 未实现 | 计划中 |
| 13 | 动态权重评估（任务类型感知指标） | 未实现 | 计划中 |
| 14 | 多维评分筛选（评估智能体） | 未实现 | 计划中 |
| 15 | 自我反思与迭代优化 | 未实现 | 计划中 |
| 16 | 收敛失败时回退至缓存最优结果 | 未实现 | 计划中 |
| **3+ 跨领域** | | | |
| 17 | 全链路流式输出（全部三条路径） | **已实现** | `direct_answer`、`executor`、`main.py` |
| 18 | 按项目隔离会话（SHA256 路径哈希） | **已实现** | `session_store.py` — `~/.hpagent/sessions/{hash}/` |
| 19 | 多会话管理（创建、列表、切换） | **已实现** | `/new`、`/sessions`、`/sessions delete` |
| 20 | 项目知识扫描与 HPD.MD 生成 | **已实现** | `/skim`、`project_scanner.py` |
| 21 | HPD.MD 自动注入 boot prompt | **已实现** | `system_info.py`、`build_boot_prompt()` |
| 22 | 上下文摘要 | **已实现** | `/summary` |
| 23 | Token 用量追踪（tiktoken） | **已实现** | `/tokens` |
| 24 | 多后端检查点持久化（Memory / SQLite / Postgres） | **已实现** | LangGraph Checkpointing |
| 25 | 模型配置管理（JSON + CLI） | **已实现** | `/model` + `src/models/` |
| 26 | 多智能体协作（Coordinator + Expert agents） | **部分实现** | Coordinator 已存在；Expert agents 尚未独立拆分 |
| 27 | 工具注册与工具调用 | 未实现 | 存根已预留，尚未接入 |
| 28 | 长期记忆（向量数据库 RAG） | 未实现 | 计划中 |
| 29 | OpenTelemetry 可观测性 | 未实现 | 计划中 |

---

## 项目结构

```
src/
├── __init__.py
├── main.py                      # CLI 入口（REPL 循环）
├── run.py                       # 直接运行入口（备选入口点）
├── agents/
│   ├── __init__.py
│   ├── coordinator_agent.py     # coordinator 节点（LLM 规划）
│   ├── expert_agent.py          # 针对 hard 子任务的专家智能体
│   └── query_agent.py           # 查询智能体
├── commands/
│   ├── __init__.py              # 命令注册与分发
│   ├── details.py               # 所有命令的帮助文本
│   └── handlers/
│       ├── __init__.py
│       ├── context_cmd.py        # /context
│       ├── exit.py              # /exit
│       ├── help.py              # /help
│       ├── love.py              # 彩蛋
│       ├── model_cmd.py         # /model（列表/创建/切换模型配置）
│       ├── new_session.py       # /new
│       ├── sessions.py          # /sessions
│       ├── summary.py           # /summary
│       └── tokens.py            # /tokens
├── core/
│   ├── __init__.py
│   ├── enums.py                 # TaskDifficulty 枚举
│   ├── models.py                # 所有 Pydantic 数据模型
│   └── state.py                 # AgentState TypedDict
├── llm/
│   ├── __init__.py
│   ├── client.py                # ChatOpenAI 工厂（读取当前模型配置）
│   └── prompts.py               # 各节点的 system prompt
├── memory/
│   ├── __init__.py
│   ├── checkpointer.py          # LangGraph 检查点（Memory / SQLite / Postgres）
│   ├── context.py               # ConversationContext 和消息历史
│   └── session_store.py         # 按项目分目录的会话持久化（~/.hpagent/sessions/{hash}/）
├── models/
│   ├── __init__.py
│   └── store.py                 # ModelProfile JSON 存储 + 单例模式
├── nodes/
│   ├── __init__.py
│   ├── assessment.py            # first_level_assessment（simple / complex）
│   ├── direct_answer.py         # direct_answer（简单任务流式输出）
│   ├── execution.py             # 子任务执行器（Level-2 评估 + LLM）
│   ├── planning.py              # DAG 分解（LLM + 环检测）
│   ├── scheduler.py             # Kahn 算法 + asyncio.gather 并行运行器
│   ├── scheduler_node.py        # scheduler 的图节点封装
│   └── synthesizer.py            # 综合 prompt 构建器
├── tools/
│   ├── __init__.py
│   ├── project_scanner.py      # 项目结构和技术栈扫描器
│   ├── read_file.py            # 读取文件内容
│   ├── write_file.py           # 写入/编辑文件内容
│   ├── terminal.py             # Shell 命令执行
│   └── registry.py             # 工具注册与工具调用
└── workflow/
    ├── __init__.py
    └── builder.py               # LangGraph StateGraph 组装
```

---

## 命令

| 命令 | 说明 |
|---|---|
| `/model` | 列出所有已保存的 LLM 模型配置 |
| `/model create` | 交互式创建新模型配置 |
| `/model <name>` | 切换到指定模型配置 |
| `/context [-d] [-N \| *]` | 查看上下文窗口（`-d`：完整内容，`*`：全部，`-N`：最近 N 条消息） |
| `/new` | 开始新会话 |
| `/sessions [id]` | 列出当前项目的所有会话或切换到指定会话 |
| `/sessions delete <id>` | 删除当前项目的指定会话 |
| `/summary` | 对上下文窗口进行摘要并清空消息 |
| `/skim [path]` | 扫描项目并生成 `HPD.MD` 项目知识摘要 |
| `/tokens` | 显示当前上下文的 token 用量 |
| `/exit` | 退出 |
| `/help` | 显示所有命令 |

---

## 配置

**模型配置** 存储在 `~/.hpagent/models.json`（首次运行自动创建，旧版 `~/.evo_agent/models.json` 会被自动迁移）。

默认配置使用：

| 字段 | 值 |
|---|---|
| 模型 | `deepseek-v4-flash` |
| Base URL | `https://api.deepseek.com` |
| API Key | `DEEPSEEK_API_KEY` 环境变量 |

可通过 `/model create` 添加更多配置。在 `get_llm()` 中任何显式传入的参数都会覆盖当前激活的配置。

---

## 开源协议

MIT — 详见 [LICENSE](./LICENSE)。
