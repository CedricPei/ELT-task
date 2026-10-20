import json

import pytest

from elt_tinker.config import credential_status, normalize_snowflake_account


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("myorg-myaccount", "myorg-myaccount"),
        ("https://app.snowflake.com/myorg/myaccount/#/home", "myorg-myaccount"),
        ("https://myorg-myaccount.snowflakecomputing.com", "myorg-myaccount"),
        ("xy12345.us-east-1", "xy12345.us-east-1"),
    ],
)
def test_accept_official_account_formats(given, expected):
    assert normalize_snowflake_account(given) == expected


@pytest.mark.parametrize(
    "value",
    ["", "wss://example.com", "https://attacker.example/account", "bad; DROP USER X", "$(id)"],
)
def test_reject_unexpected_hosts_and_executable_input(value):
    with pytest.raises(ValueError):
        normalize_snowflake_account(value)


def test_setup_report_does_not_disclose_secrets(tmp_path):
    path = tmp_path / "credentials.json"
    secret = "test-api-key-must-not-appear"
    path.write_text(json.dumps({"tinker_api_key": secret, "snowflake": {}}))
    path.chmod(0o600)
    status = credential_status(path)
    assert status["tinker_api_key"] == "configured"
    assert secret not in json.dumps(status)
    assert status["permissions"] == "private"
