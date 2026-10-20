"""Official trains task data, local source API, and evaluator kept outside model tools."""

from __future__ import annotations

import importlib.util
import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yaml

from elt_tinker.provenance import verify_files
from elt_tinker.warehouse import ToolError


@dataclass
class Task:
    name: str
    description: str
    sources: dict[str, pd.DataFrame]
    connectors: dict[str, str]
    expected: pd.DataFrame
    key: str
    evaluator_path: Path | None = None


def official_trains(root: Path) -> Task:
    verify_files(root)
    upstream = root / "vendor/ELT-Bench"
    description = yaml.safe_load(
        (upstream / "elt-bench/snowflake/trains/data_model.yaml").read_text()
    )
    return Task(
        "trains",
        "Final table: trains. Each record represents a train.\nRequired columns:\n"
        + "\n".join(
            c["name"] + ": " + c["description"] for c in description["models"][0]["columns"]
        ),
        {name: pd.read_csv(root / f"data/trains/{name}.csv") for name in ("train", "cars")},
        {"train": "csv", "cars": "api"},
        pd.read_csv(root / "data/evaluator/trains.csv", dtype=str),
        "train_id",
        upstream / "evaluation/eva_stage2.py",
    )


def fixture_task() -> Task:
    """Offline fixture with NULLs, both directions, and several classification edge cases."""
    trains = pd.DataFrame({"id": [1, 2, 3], "direction": ["east", "west", "east"]})
    cars = pd.DataFrame(
        {
            "id": [10, 11, 12, 20],
            "train_id": [1, 1, 1, 2],
            "position": [1, 2, 3, 1],
            "shape": ["rectangle", "ellipse", "ellipse", "ellipse"],
            "len": ["long", "long", "long", "short"],
            "sides": ["not_double", "double", "double", "double"],
            "roof": ["open", None, "closed", "open"],
        }
    )
    gold = pd.DataFrame(
        {
            "train_id": [1, 2, 3],
            "direction": ["east", "west", "east"],
            "run_east_and_has_less_than_4_cars": [1, 0, 1],
            "has_more_than_2_long_cars": [1, 0, 0],
            "has_double_sided_cars_in_3rd_position": [1, 0, 0],
            "has_rectangular_shaped_head_cars": [1, 0, 0],
            "number_of_open_roof_car": [1, 1, 0],
        }
    )
    return Task(
        "fixture",
        "Produce trains summary with the columns in the public data model.",
        {"train": trains, "cars": cars},
        {"train": "csv", "cars": "api"},
        gold,
        "train_id",
    )


class Sources:
    """Two connectors. Endpoint is a local replay of the official read-only cars API."""

    def __init__(self, task: Task):
        self.task = task
        api_data = json.loads(task.sources["cars"].to_json(orient="records"))

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/trains/cars":
                    self.send_error(404)
                    return
                payload = json.dumps(api_data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def inspect(self) -> dict:
        return {
            name: {
                "connector": self.task.connectors[name],
                "table": "raw_" + name,
                "columns": list(frame.columns),
                "sample": json.loads(frame.head(3).to_json(orient="records")),
            }
            for name, frame in self.task.sources.items()
        }

    def extract(self, config: dict) -> pd.DataFrame:
        if not isinstance(config, dict):
            raise ToolError("Load configuration must be an object")
        source = config.get("source")
        if not isinstance(source, str) or source not in self.task.sources:
            raise ToolError("Unknown source identifier")
        if config.get("table") != "raw_" + source or config.get("mode") != "replace":
            raise ToolError("Use the source's raw table with replace mode")
        if config.get("connector") != self.task.connectors[source]:
            raise ToolError("Connector does not match this source")
        if self.task.connectors[source] == "api":
            url = f"http://127.0.0.1:{self.server.server_port}/trains/cars"
            with urlopen(url, timeout=10) as response:
                return pd.DataFrame(json.load(response))
        # Exercise CSV serialization/parsing without depending on a cloud service.
        from io import StringIO

        return pd.read_csv(StringIO(self.task.sources[source].to_csv(index=False)))

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@lru_cache(maxsize=4)
def _official_comparator(path: Path):
    spec = importlib.util.spec_from_file_location("official_elt_comparator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(
    task: Task, expected: pd.DataFrame, actual: pd.DataFrame, key: str, *, per_cell=False,
) -> dict:
    expected = expected.copy()
    actual = actual.copy()
    expected.columns = expected.columns.str.lower()
    actual.columns = actual.columns.str.lower()
    if task.evaluator_path:
        module = _official_comparator(task.evaluator_path)
        expected = module.sort_by_keys(expected, [key])
        actual = module.sort_by_keys(actual, [key])
        comparator = module.check_corretness
    else:
        expected = expected.sort_values(key).reset_index(drop=True)
        actual = actual.sort_values(key).reset_index(drop=True) if key in actual else actual
        comparator = _fixture_comparator
    result = comparator(expected, actual)
    if per_cell:
        accuracy = {}
        for col in expected:
            if col == key.lower():
                continue
            if col in result["matched"] or col not in actual or not len(expected):
                accuracy[col] = float(col in result["matched"])
                continue
            if task.evaluator_path:
                left = expected[col].reset_index(drop=True)
                right = actual[col].reset_index(drop=True)
                numeric = all(
                    pd.api.types.is_numeric_dtype(module._to_numeric_if_possible(s))
                    for s in (left, right)
                )
                if not numeric:
                    # Preserve the official column-wide coercion decision. Coercing
                    # each cell separately could award 1 to an official non-pass.
                    def normalize(x):
                        return x.strip().lower() if isinstance(x, str) else x

                    matches = left.map(normalize).eq(right.map(normalize)) | (
                        left.isna() & right.isna()
                    )
                    accuracy[col] = float(matches.fillna(False).mean())
                    continue
            correct = sum(
                comparator(expected[[col]].iloc[i:i + 1], actual[[col]].iloc[i:i + 1])["match"]
                for i in range(len(expected))
            )
            accuracy[col] = correct / len(expected)
        result["cell_accuracy"] = accuracy
    return result


def _fixture_comparator(expected: pd.DataFrame, actual: pd.DataFrame) -> dict:
    matched = []
    for col in expected:
        if col not in actual or len(actual) != len(expected):
            continue
        try:
            pd.testing.assert_series_equal(
                expected[col],
                actual[col],
                check_dtype=False,
                check_names=False,
                rtol=1e-2,
                atol=1e-9,
            )
        except AssertionError:
            continue
        matched.append(col)
    return {
        "match": len(matched) == len(expected.columns),
        "matched": matched,
        "unmatched": [c for c in expected if c not in matched and c in actual],
        "missed": [c for c in expected if c not in actual],
    }
