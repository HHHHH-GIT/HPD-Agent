"""Scan a project directory and produce a structured summary of its layout and configuration."""

import os
import shlex
import subprocess
from pathlib import Path


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except Exception:
        return ""


def scan_project(root: str | None = None) -> str:
    """Gather project overview information from the given directory.

    If root is None, uses the current working directory.
    Returns a raw markdown-formatted string with all findings.
    """
    root_path = Path(root) if root else Path.cwd()
    lines = [f"# 项目扫描报告 — {root_path.resolve()}"]

    # 1. Tree structure (limit depth)
    root_str = str(root_path.resolve())
    tree = ""
    try:
        raw = subprocess.check_output(
            ["find", root_str, "-maxdepth", "3", "-not", "-path", "*/\\.*"],
            text=True, stderr=subprocess.DEVNULL, timeout=15
        )
        tree = "\n".join(raw.splitlines()[:120])
    except (subprocess.TimeoutExpired, OSError):
        pass

    if tree:
        lines.append("\n## 项目结构")
        for entry in tree.splitlines():
            depth = entry.count(os.sep) - root_str.count(os.sep)
            indent = "  " * max(0, depth)
            name = os.path.basename(entry) or entry
            lines.append(f"{indent}- {name}")
        lines.append("\n(仅显示前 120 项，深度 3)")

    # 2. Tech stack detection
    tech = _detect_tech_stack(root_path)
    if tech:
        lines.append("\n## 技术栈")
        for k, v in tech.items():
            lines.append(f"- {k}: {v}")

    # 3. Config / env files
    config_files = [
        ".env", ".env.local", ".env.development", ".env.production",
        "pyproject.toml", "setup.py", "setup.cfg",
        "requirements.txt", "Pipfile", "poetry.lock", "uv.lock",
        "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "Cargo.toml", "go.mod", "go.sum",
        "docker-compose.yml", "docker-compose.yaml",
        "Dockerfile",
        "Makefile", "justfile",
        "tsconfig.json", "vite.config.ts", "webpack.config.js",
        ".prettierrc", ".eslintrc", ".editorconfig",
        "pytest.ini", "tox.ini", "conftest.py",
        "mypy.ini", ".mypy.ini",
        ".gitignore", ".gitattributes",
        "ruff.toml", ".ruff.toml",
        "config.yaml", "config.yml", "config.toml", "config.json",
        "settings.py", "settings.json",
    ]
    found = []
    for cf in config_files:
        p = root_path / cf
        if p.exists():
            found.append(cf)

    if found:
        lines.append("\n## 配置文件")
        for f in found:
            lines.append(f"- {f}")

    # 4. Read key config contents
    key_files = {
        "package.json": root_path / "package.json",
        "pyproject.toml": root_path / "pyproject.toml",
        "requirements.txt": root_path / "requirements.txt",
        "Cargo.toml": root_path / "Cargo.toml",
        "go.mod": root_path / "go.mod",
        "Dockerfile": root_path / "Dockerfile",
        "docker-compose.yml": root_path / "docker-compose.yml",
    }
    for label, path in key_files.items():
        if path.exists():
            content = _read_truncated(path, max_lines=40)
            lines.append(f"\n### {label} 内容")
            lines.append("```")
            lines.append(content)
            lines.append("```")

    # 5. .env template (no real secrets)
    for env_name in [".env", ".env.example", ".env.sample"]:
        env_path = root_path / env_name
        if env_path.exists():
            lines.append(f"\n### {env_name} 内容（示例）")
            lines.append("```")
            template = _env_template(env_path)
            lines.append(template if template else "(仅含占位符，无实际密钥)")
            lines.append("```")
            break

    # 6. Git info
    git_branch = _run(f"cd {root_path} && git rev-parse --abbrev-ref HEAD")
    git_remote = _run(f"cd {root_path} && git remote get-url origin 2>/dev/null || echo ''")
    git_tag = _run(f"cd {root_path} && git describe --tags --abbrev=0 2>/dev/null || echo ''")
    if git_branch:
        lines.append("\n## Git 信息")
        lines.append(f"- 当前分支: {git_branch}")
        if git_remote:
            lines.append(f"- 远程仓库: {git_remote}")
        if git_tag:
            lines.append(f"- 最新标签: {git_tag}")

    # 7. README preview
    for readme in ["README.md", "README.zh.md", "README_EN.md", "readme.md"]:
        rm = root_path / readme
        if rm.exists():
            content = _read_truncated(rm, max_lines=30)
            lines.append(f"\n## README ({readme}) 预览")
            lines.append("```")
            lines.append(content)
            lines.append("```")
            break

    return "\n".join(lines)


def _detect_tech_stack(root: Path) -> dict[str, str]:
    stack: dict[str, str] = {}

    checks = [
        ("Python", ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile", "setup.cfg"]),
        ("Node.js", ["package.json", "yarn.lock", "pnpm-lock.yaml"]),
        ("Rust", ["Cargo.toml"]),
        ("Go", ["go.mod"]),
        ("Java/Maven", ["pom.xml"]),
        ("Java/Gradle", ["build.gradle", "gradlew"]),
        ("C++/CMake", ["CMakeLists.txt", "CMakeCache.txt"]),
        ("Unity", ["Assembly-CSharp.csproj"]),
        ("Godot", ["project.godot"]),
    ]

    for label, files in checks:
        if any(root.joinpath(f).exists() for f in files):
            stack["语言/框架"] = label

    # Web frameworks
    if (root / "vite.config.ts").exists():
        stack["构建工具"] = "Vite"
    elif (root / "vite.config.js").exists():
        stack["构建工具"] = "Vite"
    elif (root / "webpack.config.js").exists():
        stack["构建工具"] = "Webpack"
    elif (root / "next.config.js").exists():
        stack["框架"] = "Next.js"
    elif (root / "astro.config.mjs").exists():
        stack["框架"] = "Astro"

    # Python web
    py_files = list(root.glob("*.py"))
    if (root / "app.py").exists():
        stack["Python"] = "Flask/FastAPI (app.py)"
    if (root / "manage.py").exists():
        stack["Python"] = "Django (manage.py)"

    # Docker / K8s
    if (root / "docker-compose.yml").exists() or (root / "docker-compose.yaml").exists():
        stack["容器"] = "Docker Compose"
    if (root / "Dockerfile").exists():
        stack["容器"] = "Docker"

    return stack


def _read_truncated(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n... (+{len(lines) - max_lines} lines)"
        return "\n".join(lines)
    except Exception:
        return "(读取失败)"


def _env_template(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        placeholders = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key = line.split("=", 1)[0].strip()
                placeholders.append(f"{key}=<placeholder>")
        return "\n".join(placeholders) if placeholders else ""
    except Exception:
        return ""
