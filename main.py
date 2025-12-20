import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config.logging import setup_logging
from src.data.digest_data import DataDigestion
from src.features.pca import PCAManager
from src.features.clustering import ClusterManager
from src.models.runner import run_models


def _dataframes_to_numpy(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert processed training and testing DataFrames to numpy arrays for modeling.

    Args:
        train_df (pd.DataFrame): Processed training DataFrame
        test_df (pd.DataFrame): Processed testing DataFrame
        target_col (str): Name of the target column

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test

    Raises:
        ValueError: If target column is missing or conversion to numeric fails
    """
    logger = logging.getLogger("main")
    if target_col not in train_df.columns:
        logger.error("Target column '%s' not found in training DataFrame columns: %s", target_col, list(train_df.columns))
        raise ValueError(f"Target column '{target_col}' not found in train_df columns: {list(train_df.columns)}")

    feature_cols = [c for c in train_df.columns if c != target_col]

    # Convert to numpy (force numeric)
    try:
        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df[target_col].to_numpy(dtype=np.float32).reshape(-1)
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df[target_col].to_numpy(dtype=np.float32).reshape(-1)
    except Exception as e:
        logger.exception("Error converting DataFrames to numpy arrays.", extras={
            "feature_columns": feature_cols,
            "train_df_head": train_df.head().to_dict(),
            "test_df_head": test_df.head().to_dict(),
            "error": str(e),
        })
        raise ValueError(
            "Failed converting processed DataFrames to numeric numpy arrays. "
            "Make sure all feature columns are numeric after PCA/encoding."
        ) from e

    return X_train, y_train, X_test, y_test


def _get_or_build_processed_arrays(cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Check for cached processed arrays; if not found, run full preprocessing pipeline.
        Steps:
            1) Data digestion
            2) Clustering and PCA
            3) Save processed CSVs for caching
            4) Convert to numpy arrays

    Args:
        cfg (dict): Configuration dictionary loaded from settings.json

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_test, y_test
    """
    logger = logging.getLogger("main")

    proc_dir = Path(cfg["data"]["processed_dir"])
    train_path = proc_dir / "train_processed.csv"
    test_path = proc_dir / "test_processed.csv"

    have_cache = (train_path.exists() and test_path.exists() and train_path.stat().st_size > 0 and test_path.stat().st_size > 0)

    target_col = cfg["data"]["target_col"]

    if have_cache:
        logger.info("Found cached processed files. Skipping digestion/PCA/clustering.")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train, X_test, y_test = _dataframes_to_numpy(train_df, test_df, target_col)
        logger.info(
            "Loaded cached arrays | X_train=%s y_train=%s X_test=%s y_test=%s",
            X_train.shape,
            y_train.shape,
            X_test.shape,
            y_test.shape,
        )
        return X_train, y_train, X_test, y_test

    logger.info("No valid cache found. Running full preprocessing flow...")

    # 1) Data digestion
    logger.info("Initializing data digestion...")
    dig = DataDigestion(cfg)
    logger.info("Starting data digestion...")
    train_df, test_df = dig.run()
    logger.info("Data digestion completed.")

    # 2) Clustering and PCA
    logger.info("Starting Clustering and PCA pipeline...")

    cluster_tool = ClusterManager(cfg)
    train_clusters, test_clusters = cluster_tool.run_clustering_pipeline(train_df, test_df)

    pca_tool = PCAManager(cfg)
    train_df, test_df = pca_tool.run_pca_pipeline(train_df, test_df)

    # Merge cluster assignments back (supports Series or 1-col DataFrame)
    logger.info("Merging cluster assignments back to main DataFrames...")
    train_df = train_df.join(train_clusters)
    test_df = test_df.join(test_clusters)

    # 3) Save checkpoint
    proc_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Feature handoff saved to %s as CSV files.", proc_dir)

    # 4) Convert to numpy arrays
    X_train, y_train, X_test, y_test = _dataframes_to_numpy(train_df, test_df, target_col)
    logger.info(
        "Built arrays | X_train=%s y_train=%s X_test=%s y_test=%s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        None if y_test is None else y_test.shape,
    )
    return X_train, y_train, X_test, y_test


def main() -> None:
    # load settings
    with open("src/config/settings.json", "r") as f:
        cfg = json.load(f)

    # set up logging
    setup_logging(
        log_dir=cfg["logging"]["log_dir"],
        console_level=getattr(logging, cfg["logging"]["console_level"]),
        max_bytes=cfg["logging"]["max_bytes"],
        backup_count=cfg["logging"]["backup_count"],
    )

    # Check that logging is set up
    logger = logging.getLogger("main")
    logger.info("Logging is live.")

    # Build/load processed arrays
    X_train, y_train, X_test, y_test = _get_or_build_processed_arrays(cfg)

    # Modeling (runner currently uses train arrays only)
    logger.info("Starting modeling pipeline...")
    logger.info(
        "Pipeline finished | X_train=%s | y_train=%s | X_test=%s | y_test=%s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )

    results = run_models(cfg, X_train, y_train)

    # Log results summary
    for model_name, result in results.items():
        logger.info("Model: %s | Metrics: %s", model_name, result.metrics)


if __name__ == "__main__":
    main()
