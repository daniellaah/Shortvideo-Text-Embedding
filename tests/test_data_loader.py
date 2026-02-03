from __future__ import annotations

from pathlib import Path

from embedding_pipeline.data import count_rows, load_titles


def _collect(rows):
    ids = []
    titles = []
    for batch in rows:
        titles.extend(batch["video_title"])
        if "video_id" in batch:
            ids.extend(["" if v is None else str(v) for v in batch["video_id"]])
    return ids, titles


def test_load_titles_supports_headerless_tsv_id_text(tmp_path: Path) -> None:
    path = tmp_path / "title_en.tsv"
    path.write_text("1\tFirst title\n2\tSecond title\n3\t\n", encoding="utf-8")

    ids, titles = _collect(load_titles(str(path), chunksize=2))

    assert ids == ["1", "2", "3"]
    assert titles == ["First title", "Second title", ""]
    assert count_rows(str(path), text_col="video_title", id_col="video_id") == 3


def test_load_titles_headerless_tsv_uses_custom_column_names(tmp_path: Path) -> None:
    path = tmp_path / "category_combo_cn.tsv"
    path.write_text("14_163_1256\t教育 > 职业教育 > 产品与运营\n", encoding="utf-8")

    rows = list(load_titles(str(path), chunksize=1, title_col="combo_name", id_col="combo_id"))

    assert len(rows) == 1
    assert rows[0]["video_title"] == ["教育 > 职业教育 > 产品与运营"]
    assert [str(v) for v in rows[0]["video_id"]] == ["14_163_1256"]
