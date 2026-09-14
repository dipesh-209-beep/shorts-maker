#!/usr/bin/env python3
"""Shared faster-whisper model loading with GPU->CPU fallback and caching."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def _bootstrap_cuda():
    """faster-whisper bundles CUDA libs inside the venv; make sure LD_LIBRARY_PATH
    points at them, re-executing once if the environment needs to change."""
    if os.environ.get("SHORTS_CUDA_SET") == "1":
        return
    env = os.environ.get("LD_LIBRARY_PATH", "")
    nvidia = PROJECT_ROOT / "venv" / "lib"
    add = []
    for p in nvidia.glob("python3*/site-packages/nvidia") if nvidia.exists() else []:
        for _dir in sorted(p.iterdir()):
            lib = _dir / "lib"
            if lib.is_dir():
                sp = str(lib)
                if sp not in env:
                    add.append(sp)
    if add:
        os.environ["LD_LIBRARY_PATH"] = ":".join(add + ([env] if env else []))
        os.environ["SHORTS_CUDA_SET"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_bootstrap_cuda()

from faster_whisper import WhisperModel

_MODELS = {}


def load_whisper_model(model_size: str = "small"):
    """Load a faster-whisper model, trying CUDA first then falling back to CPU."""
    for device, compute_type in [("cuda", "int8_float16"), ("cpu", "int8")]:
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"  using whisper on {device}")
            return model
        except Exception as e:
            print(f"  whisper on {device} unavailable ({e}); trying next option")
    raise RuntimeError(
        f"Could not load a whisper model on cuda or cpu (model size: {model_size})"
    )


def get_whisper_model(model_size: str = "small"):
    """Like load_whisper_model, but caches one model per model_size so a batch run
    loads the model once and reuses it across all clips.

    Note: the cache accumulates — one model per distinct model_size stays loaded
    for the process lifetime, so a batch that mixes sizes (e.g. some clips "tiny",
    others "large-v3") keeps every distinct size resident, and VRAM usage grows
    with the number of distinct sizes rather than staying at one model's footprint."""
    if model_size not in _MODELS:
        _MODELS[model_size] = load_whisper_model(model_size)
    return _MODELS[model_size]