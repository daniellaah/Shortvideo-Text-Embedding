#!/usr/bin/env bash
set -euo pipefail

# Merge per-video text files (ASR/title/etc.) into a single TSV:
#   video_id<TAB>asr_text
#
# Input directory:
#   defaults to /Users/daboluo/OpenSourceData/short-video-dataset/asr_en
#
# Output (repo-relative):
#   defaults to ./data/<basename(input_dir)>.txt
#
# Notes:
# - Each ASR file becomes exactly one line in the output (newlines/tabs in ASR are normalized to spaces).
# - File order is numeric by filename stem (e.g. 1.txt, 2.txt, 10.txt).
#
# Optional:
#   --limit N    Only process first N files (useful for quick testing).
#   --input_dir DIR
#   --output PATH

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash data/preprocessing/merge_asr_en_to_tsv.sh [--input_dir DIR] [--output PATH] [--limit N]

Examples:
  # ASR (default input dir, default output)
  bash data/preprocessing/merge_asr_en_to_tsv.sh

  # Title_en directory -> data/title_en.txt
  bash data/preprocessing/merge_asr_en_to_tsv.sh --input_dir /Users/daboluo/OpenSourceData/short-video-dataset/title_en/title_en

  # Quick test on first 10 files
  bash data/preprocessing/merge_asr_en_to_tsv.sh --limit 10
USAGE
}

LIMIT=0
INPUT_DIR="/Users/daboluo/OpenSourceData/short-video-dataset/asr_en"
OUTPUT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="${2:-0}"
      shift 2
      ;;
    --input_dir)
      INPUT_DIR="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$REPO_DATA_DIR/$(basename "$INPUT_DIR").txt"
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: input dir not found: $INPUT_DIR" >&2
  exit 1
fi

INPUT_DIR="$INPUT_DIR" OUTPUT_PATH="$OUTPUT_PATH" LIMIT="$LIMIT" python - <<PY
import os
import re
import sys

input_dir = os.environ["INPUT_DIR"]
output_path = os.environ["OUTPUT_PATH"]
limit = int(os.environ.get("LIMIT", "0"))

def natural_key(name: str):
    # Numeric sort for filenames like 1.txt, 10.txt, 100.txt
    stem = os.path.splitext(name)[0]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)

names = [n for n in os.listdir(input_dir) if n.lower().endswith(".txt")]
names.sort(key=natural_key)
if limit > 0:
    names = names[:limit]

os.makedirs(os.path.dirname(output_path), exist_ok=True)

ws = re.compile(r"\\s+")

with open(output_path, "w", encoding="utf-8", newline="") as out:
    for i, name in enumerate(names, start=1):
        video_id = os.path.splitext(name)[0]
        path = os.path.join(input_dir, name)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()

        # Force a single-line TSV: replace newlines/tabs with spaces, collapse whitespace.
        txt = txt.replace("\\t", " ")
        txt = ws.sub(" ", txt).strip()

        out.write(f"{video_id}\\t{txt}\\n")

        if i % 10000 == 0:
            print(f"processed {i}/{len(names)}", file=sys.stderr)

print(f"wrote {len(names)} rows to {output_path}", file=sys.stderr)
PY
