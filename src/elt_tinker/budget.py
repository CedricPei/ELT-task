"""File-locked, fail-closed request reservations. Ambiguous requests keep their reserve."""

import fcntl
import json
import math
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class Budget:
    def __init__(self, path: Path, limit: float = 10.0):
        if not 0 < limit <= 80:
            raise ValueError("Model request allowance must be in (0, 80] USD")
        self.path, self.limit = path, limit
        path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _ledger(self):
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else {"requests": []}
            try:
                yield data
            finally:
                temp = self.path.with_suffix(".tmp")
                temp.write_text(json.dumps(data, indent=2))
                temp.replace(self.path)

    def reserve(self, kind: str, upper_usd: float) -> str:
        if not math.isfinite(upper_usd) or upper_usd <= 0:
            raise ValueError("A finite positive reservation is required")
        with self._ledger() as data:
            used = sum(item.get("charged_usd", item["reserved_usd"]) for item in data["requests"])
            if used + upper_usd > self.limit:
                raise RuntimeError("Request rejected by the model expense gate")
            ticket = uuid.uuid4().hex
            data["requests"].append(
                {
                    "id": ticket,
                    "kind": kind,
                    "reserved_usd": upper_usd,
                    "utc": datetime.now(UTC).isoformat(),
                }
            )
            return ticket

    def settle(self, ticket: str, charged_usd: float) -> None:
        with self._ledger() as data:
            item = next(item for item in data["requests"] if item["id"] == ticket)
            if not math.isfinite(charged_usd) or not 0 <= charged_usd <= item["reserved_usd"]:
                raise ValueError("Charge exceeds reservation; retain reserve and investigate")
            item["charged_usd"] = charged_usd
