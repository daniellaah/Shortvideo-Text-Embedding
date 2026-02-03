#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


DEFAULT_INPUT_DIR = "/Users/daboluo/OpenSourceData/short-video-dataset/asr_en"


def natural_key(name: str) -> tuple[int, object]:
    # Numeric sort for filenames like 1.txt, 10.txt, 100.txt
    stem = os.path.splitext(name)[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-video text files into a single TSV (video_id<TAB>text)."
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Input directory containing .txt files")
    parser.add_argument(
        "--output",
        default="",
        help="Output TSV path (default: data/<input_dir_name>.tsv in this repo)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N files (0 means no limit)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "data"

    input_dir = args.input_dir
    output_path = args.output or str(default_data_dir / f"{Path(input_dir).name}.tsv")
    limit = int(args.limit or 0)

    if not os.path.isdir(input_dir):
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 1

    names = [n for n in os.listdir(input_dir) if n.lower().endswith(".txt")]
    names.sort(key=natural_key)
    if limit > 0:
        names = names[:limit]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ws = re.compile(r"\s+")

    with open(output_path, "w", encoding="utf-8", newline="") as out:
        for i, name in enumerate(names, start=1):
            video_id = os.path.splitext(name)[0]
            path = os.path.join(input_dir, name)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()

            # Force a single-line TSV: replace newlines/tabs with spaces, collapse whitespace.
            txt = txt.replace("\t", " ")
            txt = ws.sub(" ", txt).strip()

            out.write(f"{video_id}\t{txt}\n")

            if i % 10000 == 0:
                print(f"processed {i}/{len(names)}", file=sys.stderr)

    print(f"wrote {len(names)} rows to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
