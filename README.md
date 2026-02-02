# Text Embedding Pipeline (Local BGE-M3 + Optional OpenAI)

Batch embedding pipeline that can run:
- locally via `BAAI/bge-m3` (sentence-transformers)
- optionally via OpenAI embeddings (e.g. `text-embedding-3-small`)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### Local backend (BGE-M3)

```bash
python main.py \
  --input data/video_titles.csv \
  --output output/embeddings/title_embeddings.parquet \
  --backend local \
  --model models/bge-m3 \
  --batch_size 128
```

### OpenAI backend (API)

```bash
export OPENAI_API_KEY="..."
python main.py \
  --input data/video_titles.csv \
  --output output/embeddings/title_embeddings_openai.parquet \
  --backend openai \
  --openai_model text-embedding-3-small \
  --dimensions 1024 \
  --batch_size 128
```

### Directory input (`.txt` files)

This repo includes example directories `asr_en/` and `title_en/` (each file is one row).

```bash
python main.py \
  --input title_en \
  --output output/embeddings/title_en_embeddings.parquet \
  --backend local \
  --model models/bge-m3 \
  --local_files_only
```

## Smoke Test (No Model Download)

If you want to validate the pipeline I/O + batching without downloading a model, run:

```bash
python tests/smoke_test.py
```

## Input

- CSV with required column `video_title` (override with `--text_col`)
- Optional column `video_id` (override with `--id_col`)
- Or a directory of `.txt` files (one file = one text row)

## Output

- Preferred: `.parquet` with columns: `video_id` (if present), `video_title`, `embedding` (list[float])
- Also supported: `.npy` dense matrix of shape `(N, dim)` (row order matches input)

## Device strategy

- `--device auto` (default): use MPS if available, otherwise CPU
- If MPS hits a runtime error during encoding, the pipeline falls back to CPU and retries that batch.

## ANN index (optional)

Build an HNSW index from an embeddings parquet:

```bash
python -m embedding_pipeline.build_ann_index --input output/embeddings/title_embeddings.parquet --output-dir output/ann_index/run_001
python -m embedding_pipeline.query_ann_index --index-dir output/ann_index/run_001 --embedding-parquet output/embeddings/title_embeddings.parquet --embedding-index-id 0 --topk 5
```

## Offline notes (no external APIs)

Local backend does not call any hosted embedding APIs.

Model weights must be present locally (Hugging Face cache) for fully offline runs.
To enforce offline behavior, you can set:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

If the model is not cached yet, download it once beforehand (while online) and then rerun offline.

You can also download into a local folder and point `--model` to that path:

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
python main.py --input ... --output ... --model models/bge-m3 --local_files_only
```
