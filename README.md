# HPD-Agent

**Hierarchical Parallel Dynamic Agent** — A multi-agent AI coding assistant that routes tasks intelligently and executes sub-tasks in parallel.

Built on [LangGraph](https://langchain-ai.github.io/langgraph/), HPD-Agent implements a two-level hierarchical routing system that classifies tasks by complexity and executes independent sub-tasks concurrently using Kahn's topological algorithm.

---

## Architecture

```
User Query
    │
    ▼
Level-1 Assessment (simple / complex)
    │
    ├── simple ──► Direct Answer (streaming, fast path)
    │
    └── complex ──► Coordinator Agent (DAG decomposition)
                          │
                          ▼
                    Level-2 Assessment (difficulty + requires_tools)
                          │
                          ▼
                    Scheduler (parallel execution via Kahn's algorithm)
                          │
                          ▼
                    Reviewer (quality assessment)
                          │
                    ┌─────┼─────────────┐
                    │     │             │
                 proceed  re-execute  add_tasks
                    │     │             │
                    ▼     ▼             ▼
              Synthesizer Scheduler  Coordinator
                    │    (loop)      (re-plan)
                    ▼
              Streaming Final Answer
```

### Multi-Agent System

| Agent                | Role                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------ |
| **QueryAgent**       | Public facade. Manages sessions, boots with system info, exposes the REPL.           |
| **CoordinatorAgent** | Decomposes complex queries into a DAG of sub-tasks with cycle detection.             |
| **ExpertAgent**      | Executes individual sub-tasks, routing each through Level-2 assessment.              |
| **Reviewer**         | Evaluates sub-task quality. Can request re-execution or suggest new sub-tasks (max 2 rounds). |

### Agent Tools

| Tool                         | Description                                                                           |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| `read_file(path, lines=100)` | Read file contents with optional line limit                                           |
| `apply_patch(...)`           | The only supported write path for repository edits; includes dry-run and conflict hints |
| `terminal(cmd)`              | Execute shell commands; terminal commands require confirmation in the CLI             |

### Routing Levels

| Level       | Classification       | Path                                                                                  |
| ----------- | -------------------- | ------------------------------------------------------------------------------------- |
| **Level 1** | `simple` / `complex` | Simple → direct answer; Complex → Coordinator                                         |
| **Level 2** | `difficulty` + `requires_tools` | `easy + no-tools` → single call; `easy + tools` → tool-backed single pass; `hard + no-tools` → TOT; `hard + tools` → tool-backed expert loop |
| **Review**  | `proceed` / `re-execute` / `add_tasks` | Quality gate after execution. Re-execute weak tasks or add new sub-tasks (max 2 rounds). |

### CLI Experience

The REPL now uses a `rich`-rendered layout that separates output from input. Recent updates include:

- colored command/status output and a fixed bottom prompt area
- styled `/help`, `/sessions`, `/model`, `/summary`, `/trace`, and `/tokens` views
- tree-based trace rendering with span status, timing, tokens, and metadata
- terminal command confirmation prompts rendered consistently inside the CLI

---

## Commands

All commands are entered at the REPL prompt.

| Command                 | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `/help`                 | Show all available commands                                   |
| `/context [-c N] [-d]`  | View the conversation context window                          |
| `/context clear`        | Force clear the current session context                       |
| `/exit`                 | Exit the agent                                                |
| `/model list`           | List all saved LLM model configurations                       |
| `/model create`         | Interactively create a new model profile                      |
| `/model switch <name>`  | Switch to a different model configuration                     |
| `/sessions list`        | List all sessions for the current project                     |
| `/sessions create`      | Create a new session                                          |
| `/sessions switch <id>` | Switch to a different session                                 |
| `/sessions delete <id>` | Delete a session                                              |
| `/skim [path]`          | Scan the project and generate `HPD.MD` knowledge summary      |
| `/summary`              | Summarize context and reset the context window (saves tokens) |
| `/tokens`               | Show context-window occupancy and next-request estimates      |
| `/trace [on\|half\|off]`| Toggle tracing: on (console+file), half (console only), off. Persisted across restarts. |

---

## Installation

```bash
pip install -e .
```

Or run directly without installation:

```bash
python src/run.py
python -m src.main -p /path/to/project
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Option 1: DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Option 2: DashScope (OpenAI-compatible)
DASHSCOPE_API_KEY=your_dashscope_api_key_here

# Option 3: Custom OpenAI-compatible endpoint
CUSTOM_API_KEY=your_api_key_here
```

### Model Profiles

Model configurations are stored in `~/.hpagent/models.json`. The default profile uses:

| Field         | Default                      | Description                   |
| ------------- | ---------------------------- | ----------------------------- |
| `name`        | `"default"`                  | Profile identifier            |
| `model`       | `"deepseek-v4-flash"`        | Model name                    |
| `base_url`    | `"https://api.deepseek.com"` | API endpoint                  |
| `api_key`     | (from env)                   | API key                       |
| `temperature` | `0.0`                        | Sampling temperature          |
| `thinking`    | `"disabled"`                 | Enable/disable model thinking |

Create additional profiles with `/model create`.

---

## Session Management

Sessions are isolated per project using SHA256 path hashing and stored in `~/.hpagent/sessions/`. Each session persists:

- Full conversation history
- LangGraph checkpoint state
- Model configuration

---

## Project Knowledge (`/skim`)

Running `/skim` scans your project and generates a `HPD.MD` file containing:

- Project structure overview
- Detected tech stack (Python, Node.js, Rust, Go, Java, C++, Unity, Godot)
- Web frameworks (Vite, Webpack, Next.js, Astro)
- Build tools and package managers
- Docker and CI/CD configurations

This file is automatically injected into the boot prompt as project context.

---

## Token Management

HPD-Agent tracks token usage in real-time. Use these commands to manage your context window:

- `/tokens` — View resident context usage, remaining window, and rough next-call estimates
- `/summary` — Compress conversation history to save tokens
- `/context` — Inspect and prune the context window

`/tokens` is intentionally focused on the resident context that will be injected on the next turn, which is the closest analogue to Codex/Claude Code style “context window used”. It also shows tool-schema overhead and analysis cache separately so the headline number is not inflated by non-resident data.

## Planning and Execution Notes

- Multi-solution prompts such as “give me at least three different approaches” are now expanded into independent parallel subtasks instead of a single coarse expert task.
- Tool-enabled execution has explicit round and tool-call budgets. When a budget is exhausted, the agent does a final no-tool synthesis from the evidence already collected instead of returning only partial tool traces.
- Simple-path `direct_answer()` now uses the same tool budget and confirmation flow as complex-task execution.

---

## LangGraph Checkpointing

State is persisted across sessions. Supported backends:

- **Memory** (default, no persistence)
- **SQLite** — `LANGGRAPH_CHECKPOINT/sqlite`
- **PostgreSQL** — `LANGGRAPH_CHECKPOINT/postgres`

---

## Dependencies

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

## Related Documentation

- [HPD-Agent Paper](HPD-Agent.md) — Detailed technical paper
- [HPD.MD Template](HPD.MD) — Project knowledge summary format
