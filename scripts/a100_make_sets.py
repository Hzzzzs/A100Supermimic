"""Derive fixed development and smoke lists from a filtered BONES manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from a100mimic.sets import select_dev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("keep", "1") == "1"]
    names = [row["name"] for row in rows]
    frames = {row["name"]: int(row["frames"]) for row in rows}
    sets = {"full": sorted(names), **select_dev(names, frames=frames)}
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, clips in sets.items():
        path = args.output / f"{name}.txt"
        data = ("\n".join(clips) + "\n").encode()
        path.write_bytes(data)
        records[name] = {"count": len(clips), "sha256": hashlib.sha256(data).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
