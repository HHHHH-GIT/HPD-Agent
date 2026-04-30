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
                    Level-2 Assessment (easy / hard)
                          │
                          ▼
                    Scheduler (parallel execution via Kahn's algorithm)
                          │
                          ▼
                    Synthesizer (streaming final answer)
```

### Multi-Agent System

| Agent                | Role                                                                       |
| -------------------- | -------------------------------------------------------------------------- |
| **QueryAgent**       | Public facade. Manages sessions, boots with system info, exposes the REPL. |
| **CoordinatorAgent** | Decomposes complex queries into a DAG of sub-tasks with cycle detection.   |
| **ExpertAgent**      | Executes individual sub-tasks, routing each through Level-2 assessment.    |

### Agent Tools

| Tool                                         | Description                                                   |
| -------------------------------------------- | ------------------------------------------------------------- |
| `read_file(path, lines=100)`                 | Read file contents with optional line limit                   |
| `write_file(filename, content, append=True)` | Write or append to files                                      |
| `terminal(cmd)`                              | Execute shell commands; read-only commands are always allowed |
| `project_scanner()`                          | Scan the current project structure and tech stack             |

### Routing Levels

| Level       | Classification       | Path                                                 |
| ----------- | -------------------- | ---------------------------------------------------- |
| **Level 1** | `simple` / `complex` | Simple → direct answer; Complex → Coordinator        |
| **Level 2** | `easy` / `hard`      | Easy → single tool call; Hard → multi-step reasoning |

---

## Commands

All commands are entered at the REPL prompt.

| Command                 | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `/help`                 | Show all available commands                                   |
| `/context [-c N] [-d]`  | View the conversation context window                          |
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
| `/tokens`               | Show current token usage                                      |

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

- `/tokens` — View current token count and limits
- `/summary` — Compress conversation history to save tokens
- `/context` — Inspect and prune the context window

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
```

---

## Related Documentation

- [HPD-Agent Paper](HPD-Agent.md) — Detailed technical paper
- [HPD.MD Template](HPD.MD) — Project knowledge summary format
