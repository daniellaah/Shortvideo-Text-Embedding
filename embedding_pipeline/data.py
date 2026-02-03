from __future__ import annotations

import logging
import os
import re
from typing import Dict, Iterator, List, Optional

import pandas as pd
from pandas.errors import EmptyDataError

log = logging.getLogger(__name__)

_TSV_LIKE_EXTS = {".tsv", ".txt"}


def _natural_key(s: str) -> List[object]:
    # Sort like: 1.txt, 2.txt, 10.txt (instead of 1,10,2).
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _resolve_tabular_read_kwargs(
    input_path: str,
    *,
    title_col: str,
    id_col: str,
) -> Dict[str, object]:
    """
    Resolve pandas.read_csv kwargs for file inputs.

    Supported file formats:
      - Headered CSV (default behavior)
      - Headered TSV/TXT (tab-separated)
      - Headerless 2-column TSV/TXT: id<TAB>text
    """
    ext = os.path.splitext(input_path)[1].lower()

    # For TSV-like files, support both headered and headerless 2-column data.
    if ext in _TSV_LIKE_EXTS:
        try:
            probe = pd.read_csv(input_path, sep="\t", nrows=1, keep_default_na=False)
            if title_col in probe.columns:
                return {"sep": "\t", "header": 0}
        except EmptyDataError:
            # Empty file: keep TSV parsing settings; row iteration will return 0 rows.
            pass

        # Fallback: headerless 2-column TSV/TXT (id, text).
        return {
            "sep": "\t",
            "header": None,
            "names": [id_col, title_col],
            "usecols": [0, 1],
        }

    # CSV (comma-separated) with header.
    return {}


def _iter_tabular_chunks(
    input_path: str,
    *,
    chunksize: int,
    title_col: str,
    id_col: str,
) -> Iterator[pd.DataFrame]:
    kwargs = _resolve_tabular_read_kwargs(input_path, title_col=title_col, id_col=id_col)

    try:
        reader = pd.read_csv(
            input_path,
            chunksize=chunksize,
            keep_default_na=False,
            **kwargs,
        )
        for chunk in reader:
            yield chunk
    except EmptyDataError:
        return
    except ValueError as e:
        # Improve error message for malformed headerless TSV/TXT inputs.
        if kwargs.get("header") is None and kwargs.get("sep") == "\t":
            raise ValueError(
                f"Failed to parse '{input_path}' as headerless TSV/TXT id-text format. "
                "Expected 2 columns: id<TAB>text."
            ) from e
        raise


def count_rows_csv(
    input_path: str,
    *,
    chunksize: int = 100_000,
    title_col: str = "video_title",
    id_col: str = "video_id",
) -> int:
    """
    Count data rows using streaming tabular reads.
    """
    n = 0
    for chunk in _iter_tabular_chunks(
        input_path,
        chunksize=chunksize,
        title_col=title_col,
        id_col=id_col,
    ):
        n += len(chunk)
    return int(n)


def count_rows_dir(input_dir: str, *, glob_ext: str = ".txt") -> int:
    n = 0
    for name in os.listdir(input_dir):
        if name.lower().endswith(glob_ext.lower()):
            n += 1
    return int(n)


def load_texts_from_dir(
    input_dir: str,
    *,
    chunksize: int = 10_000,
    glob_ext: str = ".txt",
    id_from: str = "stem",
    encoding: str = "utf-8",
    errors: str = "replace",
) -> Iterator[Dict[str, object]]:
    """
    Stream texts from a directory of text files.

    Yields dicts:
      - video_title: list[str]   (file contents)
      - video_id: list[str]      (derived from filename)
    """
    names = [n for n in os.listdir(input_dir) if n.lower().endswith(glob_ext.lower())]
    names.sort(key=_natural_key)

    def _make_id(filename: str) -> str:
        if id_from == "filename":
            return filename
        # stem
        base = os.path.basename(filename)
        if base.lower().endswith(glob_ext.lower()):
            return base[: -len(glob_ext)]
        return os.path.splitext(base)[0]

    batch_ids: List[str] = []
    batch_texts: List[str] = []
    for name in names:
        path = os.path.join(input_dir, name)
        with open(path, "r", encoding=encoding, errors=errors) as f:
            # Common for `.txt` corpora to end files with a newline; strip trailing newline(s) only.
            txt = f.read().rstrip("\r\n")

        batch_ids.append(_make_id(name))
        batch_texts.append(txt)

        if len(batch_texts) >= chunksize:
            yield {"video_id": batch_ids, "video_title": batch_texts}
            batch_ids, batch_texts = [], []

    if batch_texts:
        yield {"video_id": batch_ids, "video_title": batch_texts}


def load_titles(
    input_path: str,
    *,
    chunksize: int = 10_000,
    title_col: str = "video_title",
    id_col: str = "video_id",
) -> Iterator[Dict[str, object]]:
    """
    Stream input tabular file in chunks. Yields dicts:
      - video_title: list[str]
      - video_id: Optional[list[object]] (only if present)

    Does not modify text content beyond converting nulls to empty strings.
    """
    first = True
    for chunk in _iter_tabular_chunks(
        input_path,
        chunksize=chunksize,
        title_col=title_col,
        id_col=id_col,
    ):
        if first:
            if title_col not in chunk.columns:
                raise ValueError(f"Missing required column '{title_col}' in {input_path}")
            first = False

        raw_titles = chunk[title_col].tolist()
        # Preserve text as-is; only coerce null-ish values to empty strings for safety.
        titles = ["" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v) for v in raw_titles]

        out: Dict[str, object] = {"video_title": titles}
        if id_col in chunk.columns:
            # Keep original dtype as much as possible; only normalize missing values.
            ids = chunk[id_col].tolist()
            out["video_id"] = ids

        yield out


def count_rows(
    input_path: str,
    *,
    chunksize: int = 100_000,
    glob_ext: str = ".txt",
    text_col: str = "video_title",
    id_col: str = "video_id",
) -> int:
    if os.path.isdir(input_path):
        return count_rows_dir(input_path, glob_ext=glob_ext)
    return count_rows_csv(
        input_path,
        chunksize=chunksize,
        title_col=text_col,
        id_col=id_col,
    )


def load_texts(
    input_path: str,
    *,
    chunksize: int = 10_000,
    text_col: str = "video_title",
    id_col: str = "video_id",
    glob_ext: str = ".txt",
) -> Iterator[Dict[str, object]]:
    """
    Unified loader:
      - if input_path is a directory -> read *.txt files
      - else -> treat as tabular file (CSV/TSV)
    """
    if os.path.isdir(input_path):
        return load_texts_from_dir(input_path, chunksize=chunksize, glob_ext=glob_ext)
    return load_titles(input_path, chunksize=chunksize, title_col=text_col, id_col=id_col)
