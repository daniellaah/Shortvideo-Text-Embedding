#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CatRow:
    level: int
    category_id: str
    parent_id: str
    root_id: str
    name: str


def _read_categories(categories_csv: str, *, name_col: str) -> List[CatRow]:
    rows: List[CatRow] = []
    with open(categories_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"category_level", "category_id", "parent_id", "root_id", name_col}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"categories_cn_en.csv missing required columns: {sorted(missing)}")

        for row in r:
            try:
                level = int(str(row.get("category_level", "")).strip() or 0)
            except ValueError:
                continue
            cid = str(row.get("category_id", "")).strip()
            pid = str(row.get("parent_id", "")).strip()
            rid = str(row.get("root_id", "")).strip()
            name = str(row.get(name_col, "")).strip()
            if not cid or level <= 0:
                continue
            rows.append(CatRow(level=level, category_id=cid, parent_id=pid, root_id=rid, name=name))
    return rows


def _build_name_map(rows: List[CatRow]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in rows:
        if r.category_id and r.category_id not in out:
            out[r.category_id] = r.name
    return out


def _combo_from_row(row: CatRow) -> Optional[Tuple[List[str], List[str]]]:
    """
    Build an id path (list[str]) for a category row.
    Returns (ids, ids_for_display) where ids_for_display is the same.
    """
    if row.level == 1:
        # Level-1 rows are their own root.
        if not row.category_id:
            return None
        return ([row.category_id], [row.category_id])
    if row.level == 2:
        # Root -> Level2
        if not row.root_id or not row.category_id:
            return None
        return ([row.root_id, row.category_id], [row.root_id, row.category_id])
    if row.level == 3:
        # Root -> Level2 (parent) -> Level3
        if not row.root_id or not row.parent_id or not row.category_id:
            return None
        return ([row.root_id, row.parent_id, row.category_id], [row.root_id, row.parent_id, row.category_id])
    return None


def _names_for_ids(ids: List[str], name_map: Dict[str, str]) -> List[str]:
    return [name_map.get(cid, "") for cid in ids]


def generate_category_combination_files(
    categories_csv: str,
    output_file: str,
    *,
    name_col: str = "category_name_en",
) -> Dict[str, int]:
    """
    Generate a single combo table file with two columns:
      combo_id<TAB>combo_name

    Only generates Level-3 combination files:
      - combo_id:   {cat1}_{cat2}_{cat3}
      - combo_name: cat1_name > cat2_name > cat3_name

    Where:
      - cat1 = root_id (level 1)
      - cat2 = parent_id (level 2)
      - cat3 = category_id (level 3)
    """
    rows = _read_categories(categories_csv, name_col=name_col)
    name_map = _build_name_map(rows)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    written = 0
    skipped = 0
    missing_names = 0
    seen: set[str] = set()

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for row in rows:
            # Only generate 3-level combos.
            if row.level != 3:
                continue

            combo = _combo_from_row(row)
            if combo is None:
                skipped += 1
                continue
            ids, _ = combo
            if len(ids) != 3:
                skipped += 1
                continue

            combo_id = "_".join(ids)
            if combo_id in seen:
                continue

            names = _names_for_ids(ids, name_map)
            # Skip incomplete paths; output rows must contain all three names.
            if any(n == "" for n in names):
                missing_names += 1
                continue

            seen.add(combo_id)
            writer.writerow([combo_id, " > ".join(names)])
            written += 1

    return {"written": written, "skipped": skipped, "missing_names": missing_names, "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a single category combo TSV from categories_cn_en.csv."
    )
    parser.add_argument(
        "--input_file",
        "--categories_csv",
        dest="input_file",
        required=True,
        help="Path to categories_cn_en.csv",
    )
    parser.add_argument(
        "--output_file",
        "--output",
        "--output_dir",
        dest="output_file",
        required=True,
        help="Output file path; writes 2-column TSV: combo_id<TAB>combo_name",
    )
    parser.add_argument(
        "--name_col",
        default="category_name_en",
        choices=["category_name_en", "category_name_cn"],
        help="Which category name column to use for output text.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("generate_category_combinations")

    stats = generate_category_combination_files(
        args.input_file,
        args.output_file,
        name_col=args.name_col,
    )
    log.info(
        "Done. rows=%d written=%d skipped=%d missing_names=%d output_file=%s name_col=%s",
        stats["rows"],
        stats["written"],
        stats["skipped"],
        stats["missing_names"],
        args.output_file,
        args.name_col,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
