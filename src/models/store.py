"""Model profile store — JSON-backed persistence for LLM configurations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import dacite

DEFAULT_STORE_PATH = Path.home() / ".hpagent" / "models.json"
_OLD_STORE_PATH = Path.home() / ".evo_agent" / "models.json"


@dataclass
class ModelProfile:
    """A named, persisted LLM configuration."""

    name: str
    model: str
    api_key: str
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    thinking: str = "disabled"  # "enabled" | "disabled"
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> ModelProfile:
        """Safe deserialization that ignores unknown fields."""
        return dacite.from_dict(cls, data, config=dacite.Config(strict=False))

    def to_dict(self) -> dict:
        return asdict(self)


class ModelStore:
    """In-memory view over a JSON-backed model registry."""

    def __init__(self, path: Path | None = None):
        self._path: Path = path or DEFAULT_STORE_PATH
        self._profiles: dict[str, ModelProfile] = {}
        self._active: str = ""
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Load profiles from JSON. Silently create the file if missing."""
        self._migrate_legacy()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._init_defaults()
            return

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._init_defaults()
            return

        self._profiles = {
            p["name"]: ModelProfile.from_dict(p)
            for p in data.get("profiles", [])
        }
        self._active = data.get("active", "")

        if not self._profiles:
            self._init_defaults()

        if self._active not in self._profiles:
            self._active = next(iter(self._profiles), "")

    def _migrate_legacy(self) -> None:
        """Copy models.json from the legacy ~/.evo_agent/ path if it exists."""
        if self._path.exists():
            return
        if not _OLD_STORE_PATH.exists():
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            _OLD_STORE_PATH.rename(self._path)
        except OSError:
            pass

    def _save(self) -> None:
        """Write current profiles + active flag back to JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active": self._active,
            "profiles": [p.to_dict() for p in self._profiles.values()],
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _init_defaults(self) -> None:
        """Install the built-in default if the store is empty."""
        self._profiles = {
            "default": ModelProfile(
                name="default",
                model="deepseek-v4-flash",
                api_key="",          # resolved from env at call time
                base_url="https://api.deepseek.com",
                temperature=0.0,
                thinking="disabled",
                extra_body={"thinking": {"type": "disabled"}},
            )
        }
        self._active = "default"
        self._save()

    # ------------------------------------------------------------------ #
    # Query                                                               #
    # ------------------------------------------------------------------ #

    @property
    def active(self) -> str:
        """Name of the currently active model."""
        return self._active

    def get(self, name: str) -> ModelProfile | None:
        """Return a profile by name, or None if not found."""
        return self._profiles.get(name)

    def active_profile(self) -> ModelProfile | None:
        """Return the currently active profile."""
        return self._profiles.get(self._active)

    def all(self) -> list[ModelProfile]:
        """All profiles sorted by name."""
        return sorted(self._profiles.values(), key=lambda p: p.name)

    # ------------------------------------------------------------------ #
    # Mutations                                                           #
    # ------------------------------------------------------------------ #

    def add(self, profile: ModelProfile, set_active: bool = False) -> None:
        """Register a new profile. Raises ValueError on duplicate name."""
        if profile.name in self._profiles:
            raise ValueError(f"Model '{profile.name}' already exists.")
        self._profiles[profile.name] = profile
        if set_active:
            self._active = profile.name
        self._save()

    def remove(self, name: str) -> None:
        """Delete a profile. Raises ValueError if it doesn't exist or is the last one."""
        if name not in self._profiles:
            raise ValueError(f"Model '{name}' not found.")
        if len(self._profiles) == 1:
            raise ValueError("Cannot remove the last model.")
        del self._profiles[name]
        if self._active == name:
            self._active = next(iter(self._profiles))
        self._save()

    def switch(self, name: str) -> None:
        """Set the active model. Raises ValueError if not found."""
        if name not in self._profiles:
            raise ValueError(f"Model '{name}' not found.")
        self._active = name
        self._save()

    def update(self, name: str, updates: dict[str, Any]) -> None:
        """Patch an existing profile in place. Raises ValueError if not found."""
        if name not in self._profiles:
            raise ValueError(f"Model '{name}' not found.")
        raw = asdict(self._profiles[name])
        raw.update(updates)
        self._profiles[name] = ModelProfile.from_dict(raw)
        self._save()

    # ------------------------------------------------------------------ #
    # Convenience constructors                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def create_profile(
        cls,
        name: str,
        model: str,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.0,
        thinking: str = "disabled",
        extra_body: dict[str, Any] | None = None,
    ) -> ModelProfile:
        """Build a ModelProfile with sensible defaults."""
        if thinking == "disabled":
            eb = {"thinking": {"type": "disabled"}}
        else:
            eb = extra_body or {}

        return ModelProfile(
            name=name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            thinking=thinking,
            extra_body=eb,
        )


# -------------------------------------------------------------------------- #
# Module-level singleton                                                      #
# -------------------------------------------------------------------------- #

_store: ModelStore | None = None


def get_store() -> ModelStore:
    """Return the global model store singleton."""
    global _store
    if _store is None:
        _store = ModelStore()
    return _store
