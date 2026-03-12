"""Checkpoint and results saving / loading."""

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def save_results(
    output_dir: str,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    feature_df: pd.DataFrame,
    model_state: Optional[Dict] = None,
    config: Optional[Dict] = None,
    history: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Persist all training artefacts to *output_dir*.

    Returns:
        Mapping of artefact name -> file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    paths: Dict[str, str] = {}

    np.save(p := os.path.join(output_dir, f"embeddings_{ts}.npy"), embeddings)
    paths["embeddings"] = p

    np.save(p := os.path.join(output_dir, f"cluster_labels_{ts}.npy"), cluster_labels)
    paths["cluster_labels"] = p

    feature_df.to_csv(p := os.path.join(output_dir, f"features_{ts}.csv"), index=False)
    paths["features"] = p

    results_df = pd.DataFrame({
        "node_index": np.arange(len(cluster_labels)),
        "cluster_label": cluster_labels,
    })
    results_df.to_csv(p := os.path.join(output_dir, f"clustered_results_{ts}.csv"), index=False)
    paths["clustered_results"] = p

    if model_state is not None:
        torch.save(model_state, p := os.path.join(output_dir, f"model_{ts}.pt"))
        paths["model"] = p

    if config is not None:
        with open(p := os.path.join(output_dir, f"config_{ts}.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)
        paths["config"] = p

    if history is not None:
        with open(p := os.path.join(output_dir, f"history_{ts}.json"), "w") as f:
            json.dump(history, f, indent=2)
        paths["history"] = p

    logger.info(f"Results saved to: {output_dir}")
    for name, fpath in paths.items():
        logger.info(f"  {name}: {os.path.basename(fpath)}")

    return paths


def load_results(output_dir: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Load previously saved results.

    If *timestamp* is None, loads the most recent set.
    """
    files = os.listdir(output_dir)

    if timestamp is None:
        emb_files = sorted(f for f in files if f.startswith("embeddings_"))
        if not emb_files:
            raise FileNotFoundError("No results found")
        timestamp = emb_files[-1].split("_", 1)[1].replace(".npy", "")

    results: Dict[str, Any] = {}

    emb_path = os.path.join(output_dir, f"embeddings_{timestamp}.npy")
    if os.path.exists(emb_path):
        results["embeddings"] = np.load(emb_path)

    labels_path = os.path.join(output_dir, f"cluster_labels_{timestamp}.npy")
    if os.path.exists(labels_path):
        results["cluster_labels"] = np.load(labels_path)

    features_path = os.path.join(output_dir, f"features_{timestamp}.csv")
    if os.path.exists(features_path):
        results["features"] = pd.read_csv(features_path)

    config_path = os.path.join(output_dir, f"config_{timestamp}.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            results["config"] = json.load(f)

    logger.info(f"Loaded results from timestamp: {timestamp}")
    return results
