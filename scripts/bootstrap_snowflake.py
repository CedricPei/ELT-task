"""One-time browser login to create credentials scoped to this project's database.

This script creates an initially suspended warehouse. It does not submit an agent
rollout or a model-training request. Never print credentials or connector errors.
"""

import argparse
import base64
import contextlib
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from elt_tinker.config import normalize_snowflake_account

ROOT = Path(__file__).resolve().parents[1]


def private_write(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)


def main() -> None:
    import snowflake.connector

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-sql", action="store_true")
    mode.add_argument("--verify-key", action="store_true")
    args = parser.parse_args()
    secret_dir = ROOT / "secrets"
    credential_path = secret_dir / "credentials.local.json"
    credentials = json.loads(credential_path.read_text())["snowflake"]
    account = normalize_snowflake_account(credentials["account"])
    user = credentials["bootstrap_user"].strip()
    if not user:
        raise ValueError("Snowflake bootstrap login name is missing")
    key_path = secret_dir / "snowflake_runner.p8"
    if key_path.exists():
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("Expected an RSA private key")
    else:
        key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
        private_write(
            key_path,
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
    public_key = base64.b64encode(
        key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    ).decode("ascii")
    statements = [
        "CREATE ROLE IF NOT EXISTS ELT_TINKER_ROLE",
        "GRANT ROLE ELT_TINKER_ROLE TO ROLE SYSADMIN",
        "CREATE RESOURCE MONITOR IF NOT EXISTS ELT_TINKER_MONITOR WITH "
        "CREDIT_QUOTA = 3 FREQUENCY = NEVER START_TIMESTAMP = IMMEDIATELY "
        "TRIGGERS ON 90 PERCENT DO SUSPEND ON 100 PERCENT DO SUSPEND_IMMEDIATE",
        "CREATE WAREHOUSE IF NOT EXISTS ELT_TINKER_WH "
        "WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE "
        "INITIALLY_SUSPENDED = TRUE RESOURCE_MONITOR = ELT_TINKER_MONITOR",
        "ALTER WAREHOUSE ELT_TINKER_WH SET RESOURCE_MONITOR = ELT_TINKER_MONITOR",
        "GRANT MONITOR ON RESOURCE MONITOR ELT_TINKER_MONITOR TO ROLE ELT_TINKER_ROLE",
        "GRANT MONITOR, OPERATE ON WAREHOUSE ELT_TINKER_WH TO ROLE ELT_TINKER_ROLE",
        "CREATE DATABASE IF NOT EXISTS ELT_TINKER",
        "GRANT USAGE, CREATE SCHEMA ON DATABASE ELT_TINKER TO ROLE ELT_TINKER_ROLE",
        "GRANT USAGE ON WAREHOUSE ELT_TINKER_WH TO ROLE ELT_TINKER_ROLE",
        "CREATE USER IF NOT EXISTS ELT_TINKER_RUNNER TYPE = SERVICE "
        "DEFAULT_ROLE = ELT_TINKER_ROLE DEFAULT_WAREHOUSE = ELT_TINKER_WH",
        f"ALTER USER ELT_TINKER_RUNNER SET RSA_PUBLIC_KEY = '{public_key}'",
        "GRANT ROLE ELT_TINKER_ROLE TO USER ELT_TINKER_RUNNER",
    ]
    runtime_config = {
        "account": account,
        "user": "ELT_TINKER_RUNNER",
        "authenticator": "SNOWFLAKE_JWT",
        "private_key_file": str(key_path),
        "role": "ELT_TINKER_ROLE",
        "warehouse": "ELT_TINKER_WH",
        "database": "ELT_TINKER",
        "session_parameters": {"QUERY_TAG": "elt-tinker"},
    }
    if args.export_sql:
        sql_path = secret_dir / "bootstrap_snowflake.sql"
        sql = (
            ";\n\n".join(
                ["USE ROLE ACCOUNTADMIN", *statements, "SELECT 'ELT_TINKER_SETUP_READY' AS STATUS"]
            )
            + ";\n"
        )
        private_write(sql_path, sql.encode())
        print(f"One-time setup SQL saved to {sql_path}.")
        print("The SQL contains the public key only; no password, private key, or Tinker key.")
        return
    if args.verify_key:
        connection = snowflake.connector.connect(**runtime_config)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT CURRENT_USER(), CURRENT_ROLE()")
                identity = cursor.fetchone()
                if identity != ("ELT_TINKER_RUNNER", "ELT_TINKER_ROLE"):
                    raise ValueError("Unexpected runtime identity")
        finally:
            connection.close()
    else:
        print(
            "Opening Snowflake browser authentication; complete login in the browser.", flush=True
        )
        log_path = secret_dir / "snowflake-bootstrap.log"
        log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(log_fd, 0o600)
        with (
            os.fdopen(log_fd, "a", buffering=1) as log,
            contextlib.redirect_stdout(log),
            contextlib.redirect_stderr(log),
        ):
            connection = snowflake.connector.connect(
                account=account,
                user=user,
                role="ACCOUNTADMIN",
                authenticator="externalbrowser",
                disable_console_login=False,
                login_timeout=180,
                session_parameters={"QUERY_TAG": "elt-tinker/bootstrap"},
            )
        print(
            "Snowflake browser authentication succeeded; preparing project resources.", flush=True
        )
        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
        finally:
            connection.close()
    private_write(
        secret_dir / "snowflake.runtime.json",
        json.dumps(runtime_config, indent=2).encode(),
    )
    print("Snowflake bootstrap completed; private runtime credentials saved locally.")
    print("Warehouse: XSMALL, auto-suspend 60s; compute monitor: 3 credits, no reset.")
    print("This monitor does not cover storage/cloud-services charges or Tinker usage.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        code = getattr(exc, "errno", None)
        code_text = f", error {code}" if isinstance(code, int) else ""
        print(
            f"Bootstrap failed ({type(exc).__name__}{code_text}). "
            "Credential values are not printed."
        )
        log_path = ROOT / "secrets/snowflake-bootstrap-error.log"
        private_write(log_path, str(exc).encode())
        sys.exit(1)
