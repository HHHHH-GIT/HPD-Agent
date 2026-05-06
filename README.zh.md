# HPD-Agent

**Hierarchical Parallel Dynamic Agent** — 层次化并行动态 Agent，一个能智能路由任务并并行执行子任务的多智能体 AI 编程助手。

基于 [LangGraph](https://langchain-ai.github.io/langgraph/) 构建，HPD-Agent 实现了**两级层次化路由系统**，对任务复杂度进行分类，并通过 Kahn 拓扑排序算法并发执行独立子任务。

---

## 架构设计

```
用户查询
    │
    ▼
一级评估（简单 / 复杂）
    │
    ├── 简单 ──► 直接回答（流式，快路径）
    │
    └── 复杂 ──► 协调智能体（DAG 分解）
                         │
                         ▼
                   二级评估（difficulty + requires_tools）
                         │
                         ▼
                   调度器（Kahn 并行算法执行）
                         │
                         ▼
                   审查器（质量评估）
                         │
                   ┌─────┼─────────────┐
                   │     │             │
                通过   重执行       追加任务
                   │     │             │
                   ▼     ▼             ▼
             综合器   调度器       协调智能体
                   │   (循环)      (重新规划)
                   ▼
             流式输出最终答案
```

### 多智能体系统

| 智能体               | 职责                                                                |
| -------------------- | ------------------------------------------------------------------- |
| **QueryAgent**       | 对外接口。管理会话，注入系统信息，启动 REPL。                       |
| **CoordinatorAgent** | 将复杂查询分解为带循环检测的 DAG 子任务图。                         |
| **ExpertAgent**      | 执行各个子任务，通过二级评估路由。                                  |
| **Reviewer**         | 评估子任务质量。可要求重做或追加新子任务（最多 2 轮反馈循环）。     |

### Agent 工具

| 工具                         | 说明                                                     |
| ---------------------------- | -------------------------------------------------------- |
| `read_file(path, lines=100)` | 读取文件内容，支持行数限制                               |
| `apply_patch(...)`           | 仓库写入的唯一正式路径，支持 dry-run 和冲突提示          |
| `terminal(cmd)`              | 执行 Shell 命令；在 CLI 中会先进行确认                   |

### 路由层级

| 层级     | 分类            | 路径                                                                 |
| -------- | --------------- | -------------------------------------------------------------------- |
| **一级** | `简单` / `复杂` | 简单 → 直接回答；复杂 → 协调智能体                                   |
| **二级** | `difficulty` + `requires_tools` | `easy + no-tools` → 单次调用；`easy + tools` → 单次工具执行；`hard + no-tools` → TOT；`hard + tools` → 工具型专家循环 |
| **审查** | `通过` / `重执行` / `追加任务` | 执行后质量门控。可重做弱任务或追加新子任务（最多 2 轮）。 |

### CLI 体验

REPL 现在使用 `rich` 渲染，把输出区和输入区视觉上分开。近期主要更新包括：

- 统一的彩色状态输出和底部输入区
- `/help`、`/sessions`、`/model`、`/summary`、`/trace`、`/tokens` 的结构化面板
- 基于树的 trace 渲染，展示 span 状态、耗时、token 和 metadata
- `terminal` 命令确认提示统一走 CLI 渲染层

---

## 命令

在 REPL 提示符下输入以下命令。

| 命令                    | 说明                                 |
| ----------------------- | ------------------------------------ |
| `/help`                 | 显示所有可用命令                     |
| `/context [-c N] [-d]`  | 查看当前对话上下文窗口               |
| `/context clear`        | 强制清空当前会话上下文               |
| `/exit`                 | 退出 Agent                           |
| `/model list`           | 列出所有已保存的 LLM 模型配置        |
| `/model create`         | 交互式创建新的模型配置               |
| `/model switch <name>`  | 切换到指定的模型配置                 |
| `/sessions list`        | 列出当前项目的所有会话               |
| `/sessions create`      | 创建新会话                           |
| `/sessions switch <id>` | 切换到指定会话                       |
| `/sessions delete <id>` | 删除指定会话                         |
| `/skim [path]`          | 扫描项目并生成 `HPD.MD` 知识摘要文件 |
| `/summary`              | 总结上下文并重置窗口（节省 tokens）  |
| `/tokens`               | 显示上下文窗口占用和下一次请求估算   |
| `/trace [on\|half\|off]`| 链路追踪：on（控制台+文件）、half（仅控制台）、off（禁用）。跨重启持久化。 |

---

## 安装

```bash
pip install -e .
```

或不安装直接运行：

```bash
python src/run.py
python -m src.main -p /path/to/project
```

---

## 配置

### 环境变量

在项目根目录创建 `.env` 文件：

```bash
# 方式一：DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 方式二：DashScope（OpenAI 兼容）
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# 方式三：自定义 OpenAI 兼容端点
CUSTOM_API_KEY=your_api_key_here
```

### 模型配置

模型配置存储在 `~/.hpagent/models.json`。默认配置：

| 字段          | 默认值                       | 说明              |
| ------------- | ---------------------------- | ----------------- |
| `name`        | `"default"`                  | 配置名称          |
| `model`       | `"deepseek-v4-flash"`        | 模型名称          |
| `base_url`    | `"https://api.deepseek.com"` | API 端点          |
| `api_key`     | （来自环境变量）             | API 密钥          |
| `temperature` | `0.0`                        | 采样温度          |
| `thinking`    | `"disabled"`                 | 启用/禁用模型思考 |

使用 `/model create` 命令创建更多配置。

---

## 会话管理

会话按项目路径的 SHA256 哈希值隔离存储在 `~/.hpagent/sessions/` 下。每个会话保存：

- 完整对话历史
- LangGraph 检查点状态
- 模型配置

---

## 项目知识（`/skim`）

运行 `/skim` 会扫描项目并生成 `HPD.MD` 文件，包含：

- 项目结构概览
- 检测到的技术栈（Python、Node.js、Rust、Go、Java、C++、Unity、Godot）
- Web 框架（Vite、Webpack、Next.js、Astro）
- 构建工具和包管理器
- Docker 和 CI/CD 配置

该文件会自动注入启动提示词作为项目上下文。

---

## Token 管理

HPD-Agent 实时追踪 token 使用量。可用以下命令管理上下文窗口：

- `/tokens` — 查看当前常驻上下文占用、剩余窗口和下一次调用估算
- `/summary` — 压缩对话历史以节省 token
- `/context` — 查看并裁剪上下文窗口

`/tokens` 的主数字现在只表示“下一轮真正会注入模型的常驻上下文”，这更接近 Codex / Claude Code 的 context window used。工具 schema 开销和分析缓存会单独展示，不会混进主占用。

## 规划与执行说明

- 对于“至少给出三种不同解法”这类多方案请求，planner 现在会优先拆成多个无依赖子任务并行执行，而不是合并成一个粗粒度专家任务。
- 工具执行现在有显式的轮次与调用预算；预算耗尽时，会基于已收集证据做一次最终无工具总结，而不是只返回半截工具日志。
- simple path 的 `direct_answer()` 已与复杂任务共用同一套工具预算和确认逻辑。

---

## LangGraph 检查点

状态在会话之间持久化。支持以下后端：

- **Memory**（默认，不持久化）
- **SQLite** — `LANGGRAPH_CHECKPOINT/sqlite`
- **PostgreSQL** — `LANGGRAPH_CHECKPOINT/postgres`

---

## 依赖

```
langchain-openai>=0.1.0
langgraph>=0.2.0
langgraph-checkpoint>=2.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
tiktoken>=0.7.0
dacite>=0.8.0
prompt_toolkit>=3.0.0
rich>=13.0.0
```

---

## 相关文档

- [HPD-Agent 论文](HPD-Agent.md) — 详细技术论文
- [HPD.MD 模板](HPD.MD) — 项目知识摘要格式
