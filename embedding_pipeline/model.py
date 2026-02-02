from __future__ import annotations

import os
import logging
from contextlib import contextmanager

import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub.constants import HF_HUB_CACHE

log = logging.getLogger(__name__)


def resolve_device(device_arg: str) -> str:
    """
    Resolve user device choice to a torch device string.
    """
    device_arg = (device_arg or "auto").lower()
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return "mps"

    # auto
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def _hf_offline_mode(enabled: bool):
    """
    Force HF/Transformers offline mode to prevent any network calls.
    Restores environment variables afterward.
    """
    if not enabled:
        yield
        return

    keys = ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"]
    old = {k: os.environ.get(k) for k in keys}
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_model(model_name: str, *, device: str, local_files_only: bool = False) -> SentenceTransformer:
    """
    Load SentenceTransformer model onto the requested device.
    """
    log.info("Loading model: %s (device=%s local_files_only=%s)", model_name, device, local_files_only)
    if local_files_only and not os.path.exists(model_name) and "/" in model_name:
        # Fail fast: sentence-transformers may still attempt network HEAD calls even with local_files_only.
        org, repo = model_name.split("/", 1)
        expected_repo_cache = os.path.join(HF_HUB_CACHE, f"models--{org}--{repo}")
        if not os.path.exists(expected_repo_cache):
            raise RuntimeError(
                f"Model '{model_name}' not found in Hugging Face cache at {expected_repo_cache}. "
                "Download it first (or pass a local filesystem path via --model)."
            )
    try:
        with _hf_offline_mode(local_files_only):
            # sentence-transformers has evolved its constructor API; support both older and newer versions.
            if local_files_only:
                try:
                    return SentenceTransformer(
                        model_name,
                        device=device,
                        model_kwargs={"local_files_only": True},
                        tokenizer_kwargs={"local_files_only": True},
                    )
                except TypeError:
                    # Fallback: some versions accept local_files_only directly, others don't support it at all.
                    try:
                        return SentenceTransformer(model_name, device=device, local_files_only=True)
                    except TypeError:
                        log.warning(
                            "local_files_only not supported by this sentence-transformers version; using offline env only."
                        )

            return SentenceTransformer(model_name, device=device)
    except Exception as e:
        if local_files_only:
            raise RuntimeError(
                f"Failed to load model '{model_name}' in offline/local-only mode. "
                "Make sure the model is already downloaded into the local Hugging Face cache "
                "(or pass a local filesystem path via --model)."
            ) from e
        raise


def move_model_to_cpu(model: SentenceTransformer) -> SentenceTransformer:
    """
    Move model to CPU (used for MPS -> CPU fallback).
    """
    if getattr(model, "device", None) is None or str(model.device) != "cpu":
        model.to("cpu")
    return model
