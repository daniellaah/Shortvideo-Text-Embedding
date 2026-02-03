# Shortvideo-Text-Embedding

Text embedding pipeline for short-video metadata/text, with:
- local embeddings via `BAAI/bge-m3` (sentence-transformers)
- optional OpenAI embeddings (for example `text-embedding-3-small`)
- optional ANN index build/query via HNSW (`hnswlib`)

## Project Structure

```text
.
├── main.py                          # CLI entrypoint for embedding jobs
├── embedding_pipeline/              # Loaders, model backends, writers, ANN tools
├── data_preprocessing/              # Data prep scripts
├── data/                            # Sample data files
├── tests/                           # Smoke/unit tests
└── tools/                           # ANN sampling utilities
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1) Local embedding (BGE-M3)

This uses bundled sample CSV data and writes parquet embeddings:

```bash
python main.py \
  --input data/categories_cn_en.csv \
  --text_col category_name_en \
  --id_col category_id \
  --output output/embeddings/category_name_en_bge.parquet \
  --backend local \
  --model BAAI/bge-m3 \
  --batch_size 128
```

### 2) OpenAI embedding

```bash
export OPENAI_API_KEY="..."
python main.py \
  --input data/categories_cn_en.csv \
  --text_col category_name_en \
  --id_col category_id \
  --output output/embeddings/category_name_en_openai.parquet \
  --backend openai \
  --openai_model text-embedding-3-small \
  --dimensions 1024
```

### 3) Build and query ANN index

```bash
python -m embedding_pipeline.build_ann_index \
  --input output/embeddings/category_name_en_bge.parquet \
  --output-dir output/ann_index/category_name_en_bge_index

python -m embedding_pipeline.query_ann_index \
  --index-dir output/ann_index/category_name_en_bge_index \
  --embedding-parquet output/embeddings/category_name_en_bge.parquet \
  --embedding-index-id 0 \
  --topk 5
```

## Input Formats

`main.py` supports:
- CSV input with a text column (`--text_col`, default `video_title`)
- optional ID column (`--id_col`, default `video_id`)
- directory input where each `.txt` file is one row (ID from file stem)

Note: `data/title_en.txt` and `data/category_combo_cn.tsv` are TSV-like exports, not headered CSV files. Convert/reformat if you want to use them directly with `main.py`.

## Data Preprocessing

### Generate category combo table

Produces a single 2-column TSV file:
- column 1: `combo_id` (`cat1_cat2_cat3`)
- column 2: `combo_name` (`cat1 > cat2 > cat3`)

```bash
python data_preprocessing/generate_category_combinations.py \
  --input_file data/categories_cn_en.csv \
  --output_file data/category_combo_en.tsv \
  --name_col category_name_en
```

Use Chinese category names instead:

```bash
python data_preprocessing/generate_category_combinations.py \
  --input_file data/categories_cn_en.csv \
  --output_file data/category_combo_cn.tsv \
  --name_col category_name_cn
```

### Merge text files to TSV

```bash
python data_preprocessing/merge_to_tsv.py \
  --input_dir /path/to/txt_dir \
  --output data/merged_text.tsv
```

## Output Formats

- `.parquet` (recommended): columns include `video_title`, `embedding`, and optional `video_id`
- `.npy`: dense matrix `(N, dim)` only (row order matches input; metadata not stored)

## Device and Offline Notes

- `--device auto` uses MPS when available, else CPU
- local backend can run fully offline if model is already cached

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

Optional one-time model download:

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
python main.py --input ... --output ... --model models/bge-m3 --local_files_only
```

## Tests

```bash
python tests/smoke_test.py
python -m pytest -q tests/test_openai_backend.py
```
