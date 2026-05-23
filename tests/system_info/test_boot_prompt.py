from pathlib import Path

from src.system_info import build_boot_prompt


def test_boot_prompt_includes_current_project_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    prompt = build_boot_prompt()

    assert "当前项目目录" in prompt
    assert str(tmp_path) in prompt
