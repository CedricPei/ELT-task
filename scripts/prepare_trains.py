"""Download official data as inert bytes, verify digests, extract exactly one ZIP member."""

import hashlib
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from elt_tinker.provenance import verify_files

ROOT = Path(__file__).resolve().parents[1]
HF_REV = "52a48541547a4340e19e54386350740acca80146"
DIGESTS = {
    "train.csv": "f3ed8d5d62147ae2f51accd86f4d960602deb5e5c0acb8313b084eceacb0ecf9",
    "cars.csv": "d36ba10381d12eef1079d830f391f28c38158c0376c6a8d317c1102efcbfc3f8",
    "reference": "9b11c739c45c2ff131f99e06902bebe2a5ec44545d1934a534627bd0d7985996",
}


def download(url: str, path: Path, limit: int) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as output:
        count = 0
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > limit:
                raise ValueError("Download exceeds the expected size limit")
            output.write(chunk)


def main() -> None:
    destination = ROOT / "data/trains"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "train.csv"
    if not csv_path.exists():
        download(
            "https://drive.google.com/uc?export=download&id=1w5_eaQI7_bxn5zQVEOwB5WsIAkAu7VCB",
            csv_path,
            10000,
        )
    cars_path = destination / "cars.csv"
    if not cars_path.exists():
        archive = ROOT / "data/data_api.zip"
        if not archive.exists():
            # This is the data_api.zip link in the pinned upstream setup script.
            initial = (
                "https://drive.google.com/uc?export=download&id=1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0"
            )
            with urllib.request.urlopen(initial, timeout=30) as response:
                page = response.read(10000).decode()
            fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', page))
            if fields.get("id") != "1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0":
                raise ValueError("Unexpected Google Drive confirmation form")
            download(
                "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(fields),
                archive,
                200_000_000,
            )
        with zipfile.ZipFile(archive) as contents:
            member = contents.getinfo("data/trains/cars.csv")
            if member.file_size > 100000:
                raise ValueError("Unexpected source file size")
            cars_path.write_bytes(contents.read(member))
    reference = ROOT / "data/evaluator/trains.csv"
    reference.parent.mkdir(exist_ok=True)
    if not reference.exists():
        download(
            f"https://huggingface.co/datasets/tttjjj/elt_bench/resolve/{HF_REV}/"
            "gt_snowflake/trains/trains.csv",
            reference,
            10000,
        )
    paths = {"train.csv": csv_path, "cars.csv": cars_path, "reference": reference}
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != DIGESTS[name]:
            raise ValueError(f"Digest mismatch: {name}")
    upstream = ROOT / "vendor/ELT-Bench"
    for relative in (
        "evaluation/eva_stage2.py",
        "evaluation/sql/trains/trains.sql",
        "elt-bench/snowflake/trains/data_model.yaml",
    ):
        if not (upstream / relative).is_file():
            raise ValueError("Clone the pinned ELT-Bench checkout under vendor/ELT-Bench")
    (ROOT / "data/provenance.json").write_text(
        json.dumps(
            {
                "dataset": "tttjjj/elt_bench",
                "dataset_revision": HF_REV,
                "digests": DIGESTS,
                "cars_archive_member": "data/trains/cars.csv",
                "official_source_rows": {"train": 20, "cars": 63},
            },
            indent=2,
        )
    )
    verify_files(ROOT)
    print("Official trains sources and private evaluator reference verified.")


if __name__ == "__main__":
    main()
