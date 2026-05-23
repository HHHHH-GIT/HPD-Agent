from pathlib import Path

from src.tools.terminal_policy import should_confirm_terminal_command


def test_read_only_commands_never_require_confirmation(tmp_path: Path) -> None:
    assert (
        should_confirm_terminal_command(
            "ls -la",
            riskless_enabled=False,
            project_dir=tmp_path,
        ).requires_confirmation
        is False
    )
    assert (
        should_confirm_terminal_command(
            "cat pyproject.toml",
            riskless_enabled=True,
            project_dir=tmp_path,
        ).requires_confirmation
        is False
    )


def test_riskless_mode_skips_confirmation_for_normal_commands(tmp_path: Path) -> None:
    off = should_confirm_terminal_command(
        "pytest tests/tools/test_terminal.py",
        riskless_enabled=False,
        project_dir=tmp_path,
    )
    on = should_confirm_terminal_command(
        "pytest tests/tools/test_terminal.py",
        riskless_enabled=True,
        project_dir=tmp_path,
    )

    assert off.requires_confirmation is True
    assert on.requires_confirmation is False


def test_git_force_always_requires_confirmation(tmp_path: Path) -> None:
    result = should_confirm_terminal_command(
        "git push --force-with-lease origin main",
        riskless_enabled=True,
        project_dir=tmp_path,
    )

    assert result.requires_confirmation is True
    assert "force" in result.reason


def test_rm_outside_project_always_requires_confirmation(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    result = should_confirm_terminal_command(
        f"rm -rf {outside}",
        riskless_enabled=True,
        project_dir=tmp_path,
    )

    assert result.requires_confirmation is True
    assert "outside project" in result.reason


def test_rm_inside_project_follows_riskless_mode(tmp_path: Path) -> None:
    inside = tmp_path / "build"
    result = should_confirm_terminal_command(
        f"rm -rf {inside}",
        riskless_enabled=True,
        project_dir=tmp_path,
    )

    assert result.requires_confirmation is False


def test_rm_project_root_always_requires_confirmation(tmp_path: Path) -> None:
    result = should_confirm_terminal_command(
        f"rm -rf {tmp_path}",
        riskless_enabled=True,
        project_dir=tmp_path,
    )

    assert result.requires_confirmation is True
