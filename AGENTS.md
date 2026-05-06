# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/`. Key areas are `src/agents/` for agent entry points, `src/nodes/` for planning/execution graph nodes, `src/tools/` for tool adapters such as `read_file` and `terminal`, and `src/memory/`, `src/llm/`, and `src/workflow/` for persistence, model access, and graph assembly. CLI startup is in `src/main.py` and `src/run.py`.

Tests live in `tests/`, grouped by behavior: `tests/expert_loop/`, `tests/feedback_loop/`, `tests/schedule_system/`, and `tests/tools/`. Treat `build/`, `*.egg-info/`, `__pycache__/`, and `.pytest_cache/` as generated output, not source.

## Build, Test, and Development Commands
Install the package and test dependency with:

```bash
pip install -e ".[dev]"
```

Run the CLI locally with either entry point:

```bash
hpd
python -m src.main -p /path/to/project
```

Run the full test suite with:

```bash
pytest
```

Run a focused test file while iterating, for example:

```bash
pytest tests/tools/test_apply_patch.py
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for modules/functions, PascalCase for classes, and explicit, short docstrings on non-trivial functions. Keep imports grouped and prefer type hints on public functions and state-bearing helpers. No formatter or linter is configured in `pyproject.toml`, so match surrounding code closely and keep diffs minimal.

## Testing Guidelines
Use `pytest` as the default runner; it already collects both `pytest` tests and `unittest.IsolatedAsyncioTestCase` suites used in this repo. Name files `test_*.py` and keep test classes and functions descriptive, mirroring the behavior under test. Add or update tests for changes in routing, scheduling, reviewer flow, or tool behavior before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit prefixes such as `fix:`, `feat:`, `refactor:`, and `docs:` with concise summaries. Keep that format, for example `fix: retry empty planning output`. PRs should describe the behavioral change, list affected modules, note any new env vars or model assumptions, and include the exact `pytest` command you ran. For CLI-facing changes, include a short transcript or screenshot when it clarifies the result.

## Configuration & Secrets
Load local credentials from `.env`; do not commit API keys. The README documents supported variables such as `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, and `CUSTOM_API_KEY`. When changing model or session behavior, document any new configuration in `README.md` alongside the code change.
