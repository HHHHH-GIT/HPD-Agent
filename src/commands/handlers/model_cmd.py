"""Handler for the /model command — list, create, and switch LLM profiles.

Sub-commands:
    /model list                → list all profiles
    /model create              → interactive create
    /model switch <name>      → switch to named profile
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from src.agents import QueryAgent
from src.cli import get_renderer
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

def _print_list(store: ModelStore) -> None:
    active = store.active
    rows = [(p.name, getattr(p, "model", "?"), p.name == active) for p in store.all()]
    get_renderer().render_models(rows)


# -------------------------------------------------------------------------- #
# Interactive create                                                           #
# -------------------------------------------------------------------------- #

async def _read_line(prompt_text: str, default: str = "") -> str | None:
    """Prompt with ESC / Ctrl+C returning None to signal cancellation."""
    suffix = f" [{default}]: " if default else ": "
    try:
        result = await _model_session.prompt_async(prompt_text + suffix)
        return result.strip() or default
    except KeyboardInterrupt:
        return None


def _sanitize_base_url(url: str) -> str:
    """Strip trailing /chat/completions — OpenAI SDK appends it automatically."""
    url = url.rstrip("/")
    for suffix in ("/chat/completions", "/v1/chat/completions"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


async def _run_create(store: ModelStore) -> None:
    """Walk the user through creating a new model profile. ESC cancels."""
    renderer = get_renderer()
    renderer.render_command_result("Create Model", "Press ESC at any prompt to cancel.", style="accent")
    renderer.info("Required:")

    name = await _read_line("  name")
    if name is None:
        renderer.warning("Cancelled.")
        return
    model = await _read_line("  model", "deepseek-v4-flash")
    if model is None:
        renderer.warning("Cancelled.")
        return
    base_url = await _read_line("  base_url", "https://api.deepseek.com")
    if base_url is None:
        renderer.warning("Cancelled.")
        return
    base_url = _sanitize_base_url(base_url)

    renderer.info("Optional (press Enter to skip):")
    api_key = await _read_line("  api_key") or ""
    temp_str = await _read_line("  temperature") or "0.0"
    thinking = await _read_line("  thinking (enabled/disabled)", "disabled") or "disabled"

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
        renderer.success(f"Model '{name}' created and activated.")
    except ValueError as e:
        renderer.error(f"Error: {e}")


# -------------------------------------------------------------------------- #
# Sub-command handler                                                          #
# -------------------------------------------------------------------------- #

VALID_SUBS = ("list", "create", "switch")

async def handle(raw: str, agent: QueryAgent | None = None) -> bool:
    """Dispatch to the appropriate sub-command."""
    parts = raw.strip().split()

    if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "list"):
        store = get_store()
        _print_list(store)
        return False

    sub = parts[1].lower()

    if sub not in VALID_SUBS:
        get_renderer().error(f"Unknown sub-command: '{sub}'")
        get_renderer().info(f"Valid sub-commands: {', '.join(VALID_SUBS)}")
        return False

    store = get_store()

    if sub == "create":
        await _run_create(store)
        return False

    if sub == "switch":
        if len(parts) < 3:
            get_renderer().error("Usage: /model switch <name>")
            return False
        name = parts[2]
        try:
            store.switch(name)
            profile = store.get(name)
            get_renderer().success(f"Switched to model '{name}'.")
            if profile:
                get_renderer().render_models([(profile.name, getattr(profile, "model", "?"), True)])
        except ValueError as e:
            get_renderer().error(f"Error: {e}")
            get_renderer().info("Use /model list to see available models.")
        return False

    return False


# Legacy entry point (still called by the registry as run(raw, agent))
async def run(raw: str, agent: QueryAgent) -> bool:
    return await handle(raw, agent)
