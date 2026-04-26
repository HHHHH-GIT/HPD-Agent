"""Handler for the /model command — list, create, and switch LLM profiles."""

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from src.agents import QueryAgent
from src.models import ModelStore, get_store


# -------------------------------------------------------------------------- #
# Shared prompt session (ESC cancels current input, same as Ctrl+C)            #
# -------------------------------------------------------------------------- #

_model_kb = KeyBindings()

@_model_kb.add(Keys.Escape, eager=True)
@_model_kb.add(Keys.ControlC, eager=True)
def _cancel(_):
    raise KeyboardInterrupt


_model_session = PromptSession(
    key_bindings=_model_kb,
    enable_history_search=False,
    multiline=False,
)


# -------------------------------------------------------------------------- #
# Formatting helpers                                                          #
# -------------------------------------------------------------------------- #

def _print_profile(name: str, p: object, is_active: bool, name_width: int) -> None:
    marker = "   Yes" if is_active else ""
    print(f"  {name:<{name_width}}   {' ' + getattr(p, 'model', '?'):<24}{marker}")


def _print_list(store: ModelStore) -> None:
    name_width = max((len(p.name) for p in store.all()), default=0)
    active = store.active
    profiles = store.all()
    print(f"=== Models ({len(profiles)} total) ===")
    print(f"  {'Name':<{name_width + 3}} {'Model':<24} {'Active'}")
    print(f"  {'-' * (name_width + 3)} {'-' * 24} {'-' * 6}")
    for p in profiles:
        _print_profile(p.name, p, p.name == active, name_width)
    print()


# -------------------------------------------------------------------------- #
# Interactive create                                                           #
# -------------------------------------------------------------------------- #

def _read_line(prompt_text: str, default: str = "") -> str | None:
    """Prompt with ESC / Ctrl+C returning None to signal cancellation."""
    suffix = f" [{default}]: " if default else ": "
    try:
        return _model_session.prompt(prompt_text + suffix).strip() or default
    except KeyboardInterrupt:
        return None


def _run_create(store: ModelStore) -> None:
    """Walk the user through creating a new model profile. ESC cancels."""
    print("\n=== Create New Model ===")
    print("Press ESC at any prompt to cancel.\n")
    print("Required:\n")

    name = _read_line("  name")
    if name is None:
        print("\nCancelled.\n")
        return
    model = _read_line("  model", "deepseek-v4-flash")
    if model is None:
        print("\nCancelled.\n")
        return
    base_url = _read_line("  base_url", "https://api.deepseek.com")
    if base_url is None:
        print("\nCancelled.\n")
        return

    print("\nOptional (press Enter to skip):\n")
    api_key = _read_line("  api_key") or ""
    temp_str = _read_line("  temperature") or "0.0"
    thinking = _read_line("  thinking (enabled/disabled)", "disabled") or "disabled"

    try:
        temperature = float(temp_str)
    except ValueError:
        temperature = 0.0
    if thinking not in ("enabled", "disabled"):
        thinking = "disabled"

    profile = ModelStore.create_profile(
        name=name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        thinking=thinking,
    )

    try:
        store.add(profile, set_active=True)
        print(f"\nModel '{name}' created and activated.\n")
    except ValueError as e:
        print(f"\nError: {e}\n")


# -------------------------------------------------------------------------- #
# Entry point                                                                  #
# -------------------------------------------------------------------------- #

def run(raw: str, agent: QueryAgent) -> bool:
    """
    Handle /model command.

        /model           → list all profiles
        /model create   → interactive create
        /model <name>   → switch to named profile
    """
    store = get_store()
    parts = raw.strip().split()

    # /model
    if len(parts) == 1:
        _print_list(store)
        return False

    sub = parts[1].lower()

    # /model create
    if sub == "create":
        _run_create(store)
        return False

    # /model <name>
    name = sub
    try:
        store.switch(name)
        profile = store.get(name)
        print(f"Switched to model '{name}'.")
        if profile:
            _print_profile(name, profile, is_active=True, name_width=len(name))
        print()
    except ValueError as e:
        print(f"Error: {e}")
        print("Use /model to see available models.\n")
    return False
