"""Client for the browser worker subprocess."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


class BrowserServiceClient:
    """JSON-lines client for a long-lived browser worker process."""

    def __init__(self, profile_dir: str | None = None) -> None:
        self.profile_dir = profile_dir or os.environ.get("HPD_BROWSER_PROFILE_DIR")
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def request(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one browser action to the worker and return its JSON result."""
        async with self._lock:
            try:
                await self._ensure_started()
                assert self._proc is not None
                if self._proc.stdin is None or self._proc.stdout is None:
                    return {"ok": False, "error": "Browser worker pipes are unavailable"}
                message = json.dumps(
                    {
                        "action": action,
                        "session_id": payload.get("session_id", "default"),
                        "params": {
                            key: value
                            for key, value in payload.items()
                            if key != "session_id"
                        },
                    },
                    ensure_ascii=False,
                )
                self._proc.stdin.write((message + "\n").encode("utf-8"))
                await self._proc.stdin.drain()
                raw = await self._proc.stdout.readline()
                if not raw:
                    return {"ok": False, "error": "Browser worker exited without a response"}
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:
                return {"ok": False, "error": f"Browser worker request failed: {exc}"}

    async def close(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()
        self._proc = None

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        cmd = [sys.executable, "-m", "src.browser.worker"]
        if self.profile_dir:
            cmd.extend(["--profile-dir", str(Path(self.profile_dir).expanduser())])
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )


_browser_service: BrowserServiceClient | None = None


def get_browser_service() -> BrowserServiceClient:
    """Return the process-wide browser service client."""
    global _browser_service
    if _browser_service is None:
        _browser_service = BrowserServiceClient()
    return _browser_service
