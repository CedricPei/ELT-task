"""Verify Tinker credentials through metadata only; never print the API key."""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import tinker
from tinker.lib.base_url import DEFAULT_BASE_URL

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    credentials = json.loads((ROOT / "secrets/credentials.local.json").read_text())
    key = credentials.get("tinker_api_key", "")
    if not isinstance(key, str) or not key.strip():
        print("Tinker API key is missing from the private configuration.")
        return 1
    client = tinker.ServiceClient(api_key=key, base_url=DEFAULT_BASE_URL)
    status = "errored"
    try:
        capabilities = client.get_server_capabilities()
        names = sorted(model.model_name for model in capabilities.supported_models)
        report = {
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "transport": "HTTPS",
            "check": "server_capabilities_only",
            "supported_models": names,
        }
        output = ROOT / "runs/setup/tinker_connection.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(f"Tinker metadata connection OK; {len(names)} models available.")
        print("No sampling or training requests were submitted.")
        status = "success"
        return 0
    except Exception as exc:
        code = getattr(exc, "status_code", None)
        code_text = f", HTTP {code}" if isinstance(code, int) else ""
        print(f"Tinker metadata check failed ({type(exc).__name__}{code_text}).")
        return 1
    finally:
        try:
            client.close(status).result()
        except Exception:
            # Never print raw connector exceptions, which may include request data.
            pass


if __name__ == "__main__":
    sys.exit(main())
