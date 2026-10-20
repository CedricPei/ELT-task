"""Bind task data and executable comparator to the reviewed upstream snapshot."""

import hashlib
from pathlib import Path

FILES = {
    "data/trains/train.csv": "f3ed8d5d62147ae2f51accd86f4d960602deb5e5c0acb8313b084eceacb0ecf9",
    "data/trains/cars.csv": "d36ba10381d12eef1079d830f391f28c38158c0376c6a8d317c1102efcbfc3f8",
    "data/evaluator/trains.csv": "9b11c739c45c2ff131f99e06902bebe2a5ec44545d1934a534627bd0d7985996",
    "vendor/ELT-Bench/evaluation/eva_stage2.py": (
        "4b0243916d43a7d706b5efc7a059a6cc636ab6c9abce20360a2d585e03862170"
    ),
    "vendor/ELT-Bench/elt-bench/snowflake/trains/data_model.yaml": (
        "3dc07e1fca74550e7f9dfeda45c97a3198934faff31b6d685ad144bd72fdf247"
    ),
}


def verify_files(root: Path) -> None:
    for relative, expected in FILES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Unreviewed or modified task input: {relative}")
