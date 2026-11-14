"""python -m kb_build"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kb_build.pipeline import DEFAULT_DATASET, DEFAULT_KB, DEFAULT_OUT, run_build


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build EnergyTool KB chunks and artifacts")
    p.add_argument("--kb", type=Path, default=DEFAULT_KB)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = p.parse_args(argv)

    report = run_build(kb_dir=args.kb, out_dir=args.out, dataset_path=args.dataset)
    print(json_dumps(report))
    return 0 if report.get("ok") else 1


def json_dumps(obj: object) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())