import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.logging import setup_logging
from src.data.digest_data import DataDigestion
from src.features.clustering import ClusterManager
from src.features.pca import PCAManager
from src.models.runner import run_models


def _df_to_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a dataframe into (X, y) using an explicit feature column list.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_cols (list[str]): List of feature column names
        target_col (str): Name of the target column

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features matrix X and target vector y
    """
    logger = logging.getLogger("main")
    if target_col not in df.columns:
        logger.error(f"Target '{target_col}' not found in dataframe columns.")
        raise ValueError(f"Target '{target_col}' not found in dataframe columns.")
    logger.debug(f"Converting dataframe to X, y | features={len(feature_cols)} | target={target_col}")
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1)
    return X, y


def _save_feature_cache(proc_dir: Path, name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, columns: list[str]) -> None:
    """
    Saves train/test into separate files for a given feature set:
      - features_{name}_train.npz: X_train, y_train
      - features_{name}_test.npz:  X_test,  (optional) y_test
      - features_{name}_meta.json: feature columns in order

    Args:
        proc_dir (Path): Directory to save files into
        name (str): Feature set name (e.g., "tree" or "nn")
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        columns (list[str]): List of feature column names in order

    Returns:
        None
    """
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_path = proc_dir / f"features_{name}_train.npz"
    test_path = proc_dir / f"features_{name}_test.npz"
    meta_path = proc_dir / f"features_{name}_meta.json"

    np.savez_compressed(train_path, X_train=X_train, y_train=y_train)
    np.savez_compressed(test_path, X_test=X_test, y_test=y_test)

    meta = {"feature_set": name, "columns": columns}
    meta_path.write_text(json.dumps(meta, indent=2))


def _load_feature_cache(proc_dir: Path, name: str) -> Optional[Dict[str, Optional[np.ndarray]]]:
    """
    Loads train/test from separate files for a given feature set:
      - features_{name}_train.npz
      - features_{name}_test.npz

    Args:
        proc_dir (Path): Directory to load files from
        name (str): Feature set name (e.g., "tree" or "nn")

    Returns:
        - Optional[Dict[str, Optional[np.ndarray]]]: Dictionary with keys 'X_train', 'y_train', 'X_test', 'y_test' if files exist and are valid
        - None otherwise
    """
    train_path = proc_dir / f"features_{name}_train.npz"
    test_path = proc_dir / f"features_{name}_test.npz"

    if (
        not train_path.exists()
        or train_path.stat().st_size == 0
        or not test_path.exists()
        or test_path.stat().st_size == 0
    ):
        return None

    with np.load(train_path) as train_data:
        X_train = train_data["X_train"]
        y_train = train_data["y_train"]

    with np.load(test_path) as test_data:
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]

    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


def build_or_load_feature_sets(cfg: dict) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
    """
    Builds or loads feature sets for modeling:
        - Checks for cached feature sets ("tree" and "nn") in processed data directory
        - If cached sets exist, loads and returns them
        - If not, runs data digestion, clustering, and PCA to create feature sets
        - Caches the created feature sets for future use

    Args:
        cfg (dict): Configuration dictionary

    Returns:
        Dict[str, Dict[str, Optional[np.ndarray]]]: Dictionary containing feature sets with keys "tree" and "nn", each mapping to a dict with 'X_train', 'y_train', 'X_test', 'y_test'
    """
    logger = logging.getLogger("main")
    proc_dir = Path(cfg["data"]["processed_dir"])

    cached_tree = _load_feature_cache(proc_dir, "tree")
    cached_nn = _load_feature_cache(proc_dir, "nn")
    if cached_tree is not None and cached_nn is not None:
        logger.info("Loaded cached feature sets from %s", proc_dir)
        return {"tree": cached_tree, "nn": cached_nn}

    logger.info("No complete cache found -> running preprocessing")

    # 1) Data digestion (clean + impute + encode)
    dig = DataDigestion(cfg)
    train_df, test_df = dig.run()

    target_col = cfg["data"]["target_col"]

    # Feature set: TREE
    tree_feature_cols = [c for c in train_df.columns if c != target_col]
    X_train_tree, y_train_tree = _df_to_xy(train_df, tree_feature_cols, target_col)
    X_test_tree, y_test_tree = _df_to_xy(test_df, tree_feature_cols, target_col)

    # Feature set: NN
    cluster_tool = ClusterManager(cfg)
    train_clusters, test_clusters = cluster_tool.run_clustering_pipeline(train_df, test_df)

    pca_tool = PCAManager(cfg)
    train_pca_df, test_pca_df = pca_tool.run_pca_pipeline(train_df, test_df)

    # Append cluster_id after PCA
    train_pca_df = train_pca_df.join(train_clusters)
    test_pca_df = test_pca_df.join(test_clusters)

    nn_feature_cols = [c for c in train_pca_df.columns if c != target_col]
    X_train_nn, y_train_nn = _df_to_xy(train_pca_df, nn_feature_cols, target_col)
    X_test_nn, y_test_nn = _df_to_xy(test_pca_df, nn_feature_cols, target_col)

    feature_sets = {
        "tree": {"X_train": X_train_tree, "y_train": y_train_tree, "X_test": X_test_tree, "y_test": y_test_tree},
        "nn": {"X_train": X_train_nn, "y_train": y_train_nn, "X_test": X_test_nn, "y_test": y_test_nn},
    }

    # Cache both sets
    _save_feature_cache(proc_dir, "tree", X_train_tree, y_train_tree, X_test_tree, y_test_tree, tree_feature_cols)
    _save_feature_cache(proc_dir, "nn", X_train_nn, y_train_nn, X_test_nn, y_test_nn, nn_feature_cols)
    logger.info("Saved feature caches to %s", proc_dir)

    return feature_sets


def main() -> None:
    with open("src/config/settings.json", "r") as f:
        cfg = json.load(f)

    setup_logging(
        log_dir=cfg["logging"]["log_dir"],
        console_level=getattr(logging, cfg["logging"]["console_level"]),
        max_bytes=cfg["logging"]["max_bytes"],
        backup_count=cfg["logging"]["backup_count"],
    )

    logger = logging.getLogger("main")
    logger.info("Logging is live.")

    feature_sets = build_or_load_feature_sets(cfg)

    logger.info("Starting modeling pipeline...")
    results = run_models(cfg, feature_sets)

    for model_name, res in results.items():
        logger.info("Model=%s | stage=%s | metrics=%s", model_name, res.params.get("stage"), res.metrics)


if __name__ == "__main__":
    main()
