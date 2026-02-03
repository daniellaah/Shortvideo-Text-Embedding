from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local text embedding pipeline for video titles (BGE-M3).")
    p.add_argument(
        "--input",
        required=True,
        help="Input path: CSV/TSV file (headered, or TSV/TXT id<TAB>text) OR a directory of .txt files (one file per row).",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output path (.parquet preferred; .npy also supported). If omitted, auto-uses output/models/<model>/embeddings/.",
    )
    p.add_argument(
        "--output_root",
        default="output/models",
        help="Root output directory used when --output is omitted (default: output/models).",
    )
    p.add_argument(
        "--output_format",
        default="parquet",
        choices=["parquet", "npy"],
        help="Output extension used when --output is omitted (default: parquet).",
    )
    p.add_argument(
        "--dataset_name",
        default="",
        help="Optional dataset name for auto output filename when --output is omitted.",
    )
    p.add_argument(
        "--backend",
        default="local",
        choices=["local", "openai"],
        help="Embedding backend: local (sentence-transformers) or openai (API).",
    )
    p.add_argument("--model", default="BAAI/bge-m3", help="Local HF model name/path (default: BAAI/bge-m3).")
    p.add_argument("--openai_model", default="text-embedding-3-small", help="OpenAI embedding model name.")
    p.add_argument("--dimensions", type=int, default=1024, help="OpenAI embedding dimensions (default: 1024).")
    p.add_argument("--batch_size", type=int, default=128, help="Encoding batch size (default: 128).")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="Device: auto/mps/cpu (default: auto).",
    )
    p.add_argument("--text_col", default="video_title", help="Tabular text column for headered files (default: video_title).")
    p.add_argument("--id_col", default="video_id", help="Tabular id column for headered files (default: video_id).")
    p.add_argument("--glob_ext", default=".txt", help="When --input is a directory, read files ending with this ext.")
    p.add_argument(
        "--chunksize",
        type=int,
        default=10_000,
        help="Tabular streaming chunksize (rows per chunk, default: 10000).",
    )
    p.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load model files from local cache (no downloads).",
    )
    p.add_argument("--openai_api_key", default="", help="OpenAI API key (defaults to env OPENAI_API_KEY).")
    p.add_argument("--openai_base_url", default="", help="Optional OpenAI base_url (for proxies/self-hosted gateways).")
    p.add_argument("--max_retries", type=int, default=5, help="OpenAI max retries (default: 5).")
    p.add_argument("--clean", action="store_true", help="Apply optional text cleaning rules before embedding.")
    p.add_argument("--clean_rules", default="config/cleaning.json", help="Path to cleaning rules JSON.")
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize embeddings (default: true).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("bge-m3")

    t0 = time.time()

    # Lazy imports: keep `python main.py --help` working even if deps aren't installed yet.
    from tqdm import tqdm

    from embedding_pipeline.data import count_rows, load_texts
    from embedding_pipeline.io import make_writer
    from embedding_pipeline.model import build_model, resolve_device
    from embedding_pipeline.paths import default_embedding_output_path

    cleaning_rules = None
    if args.clean:
        from embedding_pipeline.cleaning import clean_text, load_rules

        cleaning_rules = load_rules(args.clean_rules)

    device = "cpu"
    if args.backend == "local":
        device = resolve_device(args.device)
        log.info("Device selected: %s (requested=%s)", device, args.device)
    else:
        log.info("Backend openai selected (device flag ignored).")

    model_name_for_path = args.model if args.backend == "local" else args.openai_model
    output_path = args.output or default_embedding_output_path(
        output_root=args.output_root,
        model_name=model_name_for_path,
        input_path=args.input,
        extension=args.output_format,
        dataset_name=args.dataset_name,
    )

    total_rows = count_rows(
        args.input,
        chunksize=args.chunksize,
        glob_ext=args.glob_ext,
        text_col=args.text_col,
        id_col=args.id_col,
    )
    log.info("Input rows: %d", total_rows)
    log.info("Output path: %s", output_path)

    writer = make_writer(output_path, total_rows=total_rows)

    model = None
    openai_backend = None
    embed_dim = None

    if args.backend == "local":
        from embedding_pipeline.encoder import encode_titles_with_fallback

        model = build_model(args.model, device=device, local_files_only=args.local_files_only)
        embed_dim = int(model.get_sentence_embedding_dimension())
    else:
        from embedding_pipeline.openai_backend import OpenAIEmbedder

        openai_backend = OpenAIEmbedder(
            model=args.openai_model,
            dimensions=args.dimensions,
            request_batch_size=args.batch_size,
            max_retries=args.max_retries,
            api_key=args.openai_api_key or None,
            base_url=args.openai_base_url or None,
            normalize=args.normalize,
        )
        embed_dim = int(openai_backend.get_sentence_embedding_dimension())

    writer_initialized = False
    written = 0

    if total_rows == 0:
        # Still create an empty output file with the correct schema/shape.
        writer.init_if_needed(embedding_dim=embed_dim, has_ids=False)
        writer.close()
        log.info("Done. rows=0 dim=%d output=%s seconds=%.2f", embed_dim, output_path, time.time() - t0)
        return 0

    with tqdm(total=total_rows, desc="Encoding", unit="rows") as pbar:
        for batch in load_texts(
            args.input,
            chunksize=args.chunksize,
            text_col=args.text_col,
            id_col=args.id_col,
            glob_ext=args.glob_ext,
        ):
            titles = batch["video_title"]
            ids = batch.get("video_id")

            if cleaning_rules is not None:
                titles = [clean_text(t, cleaning_rules) for t in titles]

            if args.backend == "local":
                embeddings, model = encode_titles_with_fallback(
                    model,
                    titles,
                    batch_size=args.batch_size,
                    embedding_dim=embed_dim,
                )
            else:
                embeddings = openai_backend.encode(titles, batch_size=args.batch_size)

            if not writer_initialized:
                writer.init_if_needed(embedding_dim=embed_dim, has_ids=ids is not None)
                writer_initialized = True

            writer.write_batch(video_titles=titles, embeddings=embeddings, video_ids=ids)

            written += len(titles)
            pbar.update(len(titles))

    writer.close()

    dt = time.time() - t0
    log.info(
        "Done. rows=%d dim=%s output=%s seconds=%.2f",
        written,
        embed_dim,
        output_path,
        dt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
