#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any, Dict

import hnswlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _infer_dim(col: pa.ChunkedArray) -> int:
    # pyarrow.ChunkedArray.combine_chunks() returns an Array.
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    t = arr.type
    if pa.types.is_fixed_size_list(t):
        return int(t.list_size)
    # Fallback: read the first row as python list.
    first = arr[0].as_py()
    if not isinstance(first, list):
        raise ValueError(f"Unsupported embedding type: {t}")
    return int(len(first))


def _to_numpy_2d(col: pa.ChunkedArray, *, dim: int) -> np.ndarray:
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    t = arr.type

    if pa.types.is_fixed_size_list(t):
        # FixedSizeListArray exposes underlying flat values.
        flat = arr.values.to_numpy(zero_copy_only=False)  # float32
        out = flat.reshape(-1, dim)
        return np.asarray(out, dtype=np.float32)

    # Slow fallback for variable-length list arrays.
    py = [arr[i].as_py() for i in range(len(arr))]
    out = np.asarray(py, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != dim:
        raise ValueError(f"Embeddings must be 2D with dim={dim}, got {out.shape}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ANN index (HNSW) from embeddings parquet.")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet")
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. output/ann_index/run_001)")
    parser.add_argument("--embedding-col", default="embedding")
    parser.add_argument("--space", default="cosine", choices=["cosine", "l2"], help="Distance space")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pf = pq.ParquetFile(args.input)
    n = int(pf.metadata.num_rows)
    if n <= 0:
        raise ValueError("Embeddings parquet is empty.")

    # Infer vector dim from the first row group.
    first = pf.read_row_group(0, columns=[args.embedding_col])
    if args.embedding_col not in first.column_names:
        raise ValueError(f"Missing embedding column: {args.embedding_col}")
    dim = _infer_dim(first[args.embedding_col])

    index = hnswlib.Index(space=args.space, dim=dim)
    index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.M)

    # Write metadata.parquet incrementally (all columns except embedding).
    meta_writer = None
    meta_path = os.path.join(args.output_dir, "metadata.parquet")

    offset = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        if args.embedding_col not in table.column_names:
            raise ValueError(f"Missing embedding column: {args.embedding_col}")

        emb = _to_numpy_2d(table[args.embedding_col], dim=dim)
        labels = np.arange(offset, offset + emb.shape[0], dtype=np.int64)
        index.add_items(emb, labels)
        offset += int(emb.shape[0])

        meta = table.drop([args.embedding_col])
        if meta_writer is None:
            meta_writer = pq.ParquetWriter(meta_path, meta.schema)
        meta_writer.write_table(meta)

    if meta_writer is not None:
        meta_writer.close()

    if offset != n:
        raise RuntimeError(f"Row count mismatch while building index: expected {n}, wrote {offset}")

    index.set_ef(args.ef_search)

    index_path = os.path.join(args.output_dir, "index.bin")
    index.save_index(index_path)

    manifest = {
        "input": os.path.abspath(args.input),
        "output_dir": os.path.abspath(args.output_dir),
        "embedding_col": args.embedding_col,
        "space": args.space,
        "dimensions": dim,
        "count": n,
        "M": args.M,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "index_file": "index.bin",
        "metadata_file": "metadata.parquet",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
