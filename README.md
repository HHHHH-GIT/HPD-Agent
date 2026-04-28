# HPD-Agent

A **Hierarchical Parallel Dynamic** agent framework for complex task execution, built with LangGraph and Python asyncio.

HPD-Agent routes every incoming query through a two-tier assessment: simple tasks take a fast direct-answer path, while complex tasks are decomposed into a DAG and executed in parallel — reserving heavyweight computation for where it actually matters.

---

## Quick Start

```bash
# pip install .
pip install git+https://github.com/HHHHH-GIT/HPD-Agent.git
cp .env.example .env        # set DEEPSEEK_API_KEY
hpd
```

```
> 你好
# ... streamed response ...

> /model create            # manage LLM profiles
> /context                 # view context window
> /new                     # start a fresh session
> /exit                    # quit
```

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  first_level_assessment  │  ← Level-1: simple / complex
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 simple    complex
    │         │
    ▼         ▼
direct_   coordinator
answer    (DAG planning)
    │         │
    ▼         ▼
   END      scheduler_node
             (Kahn parallel execution)
                  │
                  ▼
              synthesizer
              (streaming synthesis)
                  │
                  ▼
                 END
```

**Execution paths**


| Path   | Trigger              | Steps                                                         |
| ------ | -------------------- | ------------------------------------------------------------- |
| Direct | `simple` at Level-1  | assessment → direct_answer → END                              |
| Full   | `complex` at Level-1 | assessment → coordinator → scheduler_node → synthesizer → END |


---

## Feature Checklist

Based on the [HPD-Agent paper](./HPD-Agent.md).


| #                            | Paper Feature                                            | Status          | Notes                                                                                 |
| ---------------------------- | -------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------- |
| **2.1 Hierarchical Routing** |                                                          |                 |                                                                                       |
| 1                            | Level-1 assessment: simple / complex                     | **Implemented** | `src/nodes/assessment.py`                                                             |
| 2                            | Level-2 assessment: easy / hard (per sub-task)           | **Implemented** | `src/nodes/execution.py`                                                              |
| 3                            | Dynamic expert mode for hard sub-tasks                   | **Partial**     | Expert mode flag is tracked in state; self-reflection iteration loop is not yet wired |
| **2.2 Parallel Execution**   |                                                          |                 |                                                                                       |
| 4                            | DAG decomposition via LLM planner                        | **Implemented** | `src/nodes/planning.py`                                                               |
| 5                            | Kahn topological sort                                    | **Implemented** | `src/nodes/scheduler.py`                                                              |
| 6                            | In-degree-based ready-layer batching                     | **Implemented** | `src/nodes/scheduler.py`                                                              |
| 7                            | `asyncio.gather` parallel execution                      | **Implemented** | `src/nodes/scheduler.py`                                                              |
| 8                            | Exponential back-off retry (1s→2s→4s, cap 10s)           | **Implemented** | `src/nodes/scheduler.py`                                                              |
| 9                            | Deadlock detection (empty ready queue, unfinished)       | **Implemented** | `src/nodes/scheduler.py`                                                              |
| 10                           | DAG cycle detection + LLM retry (up to 3×)               | **Implemented** | `src/nodes/planning.py`                                                               |
| 11                           | Thread-safe progress output (`threading.Lock`)           | **Implemented** | `src/nodes/scheduler.py`                                                              |
| **2.3 Dynamic Expert Mode**  |                                                          |                 |                                                                                       |
| 12                           | Multi-path generation (high-temperature candidates)      | Not yet         | Planned                                                                               |
| 13                           | Dynamic weight assessment (task-type-aware metrics)      | Not yet         | Planned                                                                               |
| 14                           | Multi-dimensional scoring (evaluation agent)             | Not yet         | Planned                                                                               |
| 15                           | Self-reflection & iteration                              | Not yet         | Planned                                                                               |
| 16                           | Fallback to cached best result on convergence failure    | Not yet         | Planned                                                                               |
| **3+ Cross-cutting**         |                                                          |                 |                                                                                       |
| 17                           | Full streaming output (all three paths)                  | **Implemented** | `direct_answer`, `executor`, `main.py`                                                |
| 18                           | Per-project session isolation (SHA256 path hash)           | **Implemented** | `session_store.py` — `~/.hpagent/sessions/{hash}/`                                       |
| 19                           | Multi-session management (create, list, switch)            | **Implemented** | `/new`, `/sessions`, `/sessions delete`                                                 |
| 20                           | Project knowledge scan & HPD.MD generation                 | **Implemented** | `/skim`, `project_scanner.py`                                                          |
| 21                           | HPD.MD auto-injection into boot prompt                    | **Implemented** | `system_info.py`, `build_boot_prompt()`                                                |
| 22                           | Context summarization                                     | **Implemented** | `/summary`                                                                            |
| 23                           | Token-usage tracking (tiktoken)                          | **Implemented** | `/tokens`                                                                             |
| 24                           | Multi-backend checkpointing (Memory / SQLite / Postgres) | **Implemented** | LangGraph Checkpointing                                                               |
| 25                           | Model profile management (JSON, CLI)                     | **Implemented** | `/model` + `src/models/`                                                              |
| 26                           | Multi-agent coordination (Coordinator + Expert agents)   | **Partial**     | Coordinator exists; Expert agents not yet separate                                    |
| 27                           | Tool registry & tool-calling                             | Not yet         | Stub exists, not wired                                                                |
| 28                           | Long-term memory (vector DB RAG)                         | Not yet         | Planned                                                                               |
| 29                           | OpenTelemetry observability                              | Not yet         | Planned                                                                               |


---

## Project Structure

```
src/
├── __init__.py
├── main.py                      # CLI entry point (REPL loop)
├── run.py                       # direct runner (alternative entry)
├── agents/
│   ├── __init__.py
│   ├── coordinator_agent.py     # coordinator node (LLM planning)
│   ├── expert_agent.py          # expert agent for hard sub-tasks
│   └── query_agent.py           # query agent
├── commands/
│   ├── __init__.py              # command registry & dispatcher
│   ├── details.py               # help text for all commands
│   └── handlers/
│       ├── __init__.py
│       ├── context_cmd.py        # /context
│       ├── exit.py              # /exit
│       ├── help.py              # /help
│       ├── love.py              # easter egg
│       ├── model_cmd.py         # /model (list/create/switch profiles)
│       ├── new_session.py       # /new
│       ├── sessions.py          # /sessions
│       ├── summary.py           # /summary
│       └── tokens.py            # /tokens
├── core/
│   ├── __init__.py
│   ├── enums.py                 # TaskDifficulty enum
│   ├── models.py                # all Pydantic data models
│   └── state.py                 # AgentState TypedDict
├── llm/
│   ├── __init__.py
│   ├── client.py                # ChatOpenAI factory (reads active model profile)
│   └── prompts.py               # system prompts for each node
├── memory/
│   ├── __init__.py
│   ├── checkpointer.py          # LangGraph checkpointer (Memory / SQLite / Postgres)
│   ├── context.py               # ConversationContext & message history
│   └── session_store.py         # per-project session persistence (~/.hpagent/sessions/{hash}/)
├── models/
│   ├── __init__.py
│   └── store.py                 # ModelProfile JSON store + singleton
├── nodes/
│   ├── __init__.py
│   ├── assessment.py            # first_level_assessment (simple / complex)
│   ├── direct_answer.py         # direct_answer (streaming for simple tasks)
│   ├── execution.py             # sub-task executor (Level-2 assessment + LLM)
│   ├── planning.py              # DAG decomposition (LLM + cycle check)
│   ├── scheduler.py             # Kahn's algorithm + asyncio.gather parallel runner
│   ├── scheduler_node.py        # graph node wrapper around scheduler
│   └── synthesizer.py           # synthesis prompt builder
├── tools/
│   ├── __init__.py
│   ├── project_scanner.py      # project structure & tech-stack scanner
│   ├── read_file.py            # read file contents
│   ├── write_file.py           # write / edit file contents
│   ├── terminal.py             # shell command execution
│   └── registry.py             # tool registry & tool-calling
└── workflow/
    ├── __init__.py
    └── builder.py               # LangGraph StateGraph assembly
```

---

## Commands


| Command                  | Description                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| `/model`                 | List all saved LLM profiles                                                |
| `/model create`          | Interactively create a new model profile                                   |
| `/model <name>`          | Switch to a saved model                                                    |
| `/context [-d] [-N]`     | View context window ( `-d`: full content, `*`: all, `-N`: last N messages) |
| `/new`                   | Start a new conversation session                                           |
| `/sessions [id]`         | List sessions for the current project or switch to one                     |
| `/sessions delete <id>`  | Delete a session for the current project                                   |
| `/summary`               | Summarize context window and clear messages                                |
| `/skim [path]`           | Scan the project and generate `HPD.MD` project knowledge summary           |
| `/tokens`                | Show token usage of current context                                        |
| `/exit`                  | Exit                                                                       |
| `/help`                  | Show all commands                                                          |


---

## Configuration

**Model profiles** are stored in `~/.hpagent/models.json` (created on first run; legacy `~/.evo_agent/models.json` is migrated automatically).

The default profile uses:


| Field    | Value                      |
| -------- | -------------------------- |
| Model    | `deepseek-v4-flash`        |
| Base URL | `https://api.deepseek.com` |
| API Key  | `DEEPSEEK_API_KEY` env var |


Add more profiles via `/model create`. Any explicitly-passed parameter in `get_llm()` overrides the active profile.

---

## License

MIT — see [LICENSE](./LICENSE) for details.
