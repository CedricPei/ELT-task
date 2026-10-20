"""Read credentials as data; never source them as shell commands."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

ACCOUNT_PATTERN = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*\Z")


def normalize_snowflake_account(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Snowflake account is missing")
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
            raise ValueError("Expected an HTTPS Snowflake account URL")
        host = (parsed.hostname or "").lower()
        if host == "app.snowflake.com":
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 2:
                raise ValueError("Expected organization/account in the Snowflake URL")
            value = f"{parts[0]}-{parts[1]}"
        elif host.endswith(".snowflakecomputing.com"):
            value = host.removesuffix(".snowflakecomputing.com")
        else:
            raise ValueError("Unexpected Snowflake account host")
    if not ACCOUNT_PATTERN.fullmatch(value):
        raise ValueError("Invalid Snowflake account identifier")
    return value


def credential_status(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {"credentials": "missing"}
    try:
        raw = json.loads(path.read_text())
    except (ValueError, OSError):
        return {"credentials": "invalid JSON or unreadable"}
    if not isinstance(raw, dict):
        return {"credentials": "expected a JSON object"}
    snowflake = raw.get("snowflake", {})
    if not isinstance(snowflake, dict):
        snowflake = {}
    result = {
        "tinker_api_key": "configured" if str(raw.get("tinker_api_key", "")).strip() else "missing",
        "snowflake.bootstrap_user": (
            "configured" if str(snowflake.get("bootstrap_user", "")).strip() else "missing"
        ),
    }
    try:
        normalize_snowflake_account(str(snowflake.get("account", "")))
        result["snowflake.account"] = "configured"
    except ValueError:
        result["snowflake.account"] = "missing or invalid"
    result["permissions"] = "private" if path.stat().st_mode & 0o077 == 0 else "too broad"
    return result
