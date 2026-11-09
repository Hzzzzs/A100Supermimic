"""Validate evaluator JSON shards against a frozen clip list and summarize."""

import argparse
import json
from pathlib import Path

from a100mimic.reporting import summarize_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_list", type=Path)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    expected = [line.strip() for line in args.clip_list.read_text().splitlines() if line.strip()]
    documents = [json.loads(path.read_text()) for path in args.shards]
    # These fields must describe the same policy and iteration across shards.
    keys = ("policy", "run_name", "checkpoint_path", "checkpoint_sha256", "iteration", "weights", "robot", "solver_iterations", "solver_ls_iterations", "njmax", "num_envs", "chunk_size")
    signatures = {tuple(doc.get("checkpoint", {}).get(key) for key in keys) for doc in documents}
    if len(signatures) != 1:
        raise ValueError("checkpoint metadata differs across evaluation shards")
    rows = [row for doc in documents for row in doc["clips"]]
    print(json.dumps(summarize_complete(rows, expected), indent=2))


if __name__ == "__main__":
    main()
