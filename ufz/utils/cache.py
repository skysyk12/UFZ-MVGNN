"""Unified cache management for intermediate computation results."""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "cache"


def get_cache_dir(base: str = _DEFAULT_CACHE_DIR) -> Path:
    """Return (and create) the cache directory."""
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_key(*args: str) -> str:
    """Derive a deterministic hex key from arbitrary string tokens."""
    h = hashlib.sha256("|".join(args).encode()).hexdigest()[:16]
    return h


def cached(
    path: str,
    compute_fn: Callable[[], Any],
    force: bool = False,
) -> Any:
    """
    Load a cached result from *path*, or call *compute_fn* and save the result.

    Uses ``pickle`` for serialisation.  Pass ``force=True`` to recompute.
    """
    if not force and os.path.exists(path):
        logger.info(f"Loading cached result: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    result = compute_fn()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Cached result saved: {path}")

    return result
