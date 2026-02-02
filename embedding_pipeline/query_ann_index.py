#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import hnswlib
import numpy as np
import pyarrow.parquet as pq


def load_manifest(index_dir: str) -> Dict[str, Any]:
    path = os.path.join(index_dir, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(index_dir: str) -> Any:
    path = os.path.join(index_dir, "metadata.parquet")
    table = pq.read_table(path)
    return table.to_pandas()


def load_embedding_from_parquet(path: str, index_id: int, *, embedding_col: str = "embedding") -> List[float]:
    table = pq.read_table(path, columns=[embedding_col])
    df = table.to_pandas()
    if index_id < 0 or index_id >= len(df):
        raise ValueError("index_id out of range")
    return [float(x) for x in df.iloc[index_id][embedding_col]]


def load_embedding_from_file(path: str) -> List[float]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return arr.astype(float).tolist()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError("Empty embedding file")
    return json.loads(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query ANN index with a single embedding.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--embedding-json", default="")
    parser.add_argument("--embedding-file", default="")
    parser.add_argument("--embedding-parquet", default="")
    parser.add_argument("--embedding-index-id", type=int, default=-1)
    parser.add_argument("--embedding-col", default="", help="Override embedding column (default: from manifest).")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not args.embedding_json and not args.embedding_file and not args.embedding_parquet:
        raise ValueError("Provide --embedding-json, --embedding-file, or --embedding-parquet")
    if args.embedding_parquet and args.embedding_index_id < 0:
        raise ValueError("--embedding-index-id is required when using --embedding-parquet")

    manifest = load_manifest(args.index_dir)
    dim = int(manifest["dimensions"])
    space = str(manifest["space"])
    embedding_col = args.embedding_col or str(manifest.get("embedding_col", "embedding"))

    if args.embedding_json:
        emb = json.loads(args.embedding_json)
    elif args.embedding_parquet:
        emb = load_embedding_from_parquet(args.embedding_parquet, args.embedding_index_id, embedding_col=embedding_col)
    else:
        emb = load_embedding_from_file(args.embedding_file)

    vec = np.array(emb, dtype=np.float32)
    if vec.ndim != 1 or int(vec.shape[0]) != dim:
        raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {vec.shape}")

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(os.path.join(args.index_dir, "index.bin"))
    index.set_ef(args.ef_search)

    labels, distances = index.knn_query(vec, k=args.topk)
    labels = labels[0].tolist()
    distances = distances[0].tolist()

    meta = load_metadata(args.index_dir)

    results = []
    for rank, (idx, dist) in enumerate(zip(labels, distances), start=1):
        row = meta.iloc[idx]
        if space == "cosine":
            score = 1.0 - float(dist)
        else:
            score = -float(dist)
        results.append(
            {
                "rank": rank,
                "index_id": int(idx),
                "score": score,
                # Best-effort passthrough fields (depends on your parquet schema)
                "video_id": str(row.get("video_id", "")),
                "video_title": str(row.get("video_title", "")),
            }
        )

    for r in results:
        print(json.dumps(r, ensure_ascii=True))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

