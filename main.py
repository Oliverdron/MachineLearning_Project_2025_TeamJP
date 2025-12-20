import json
import logging
from pathlib import Path

from src.config.logging import setup_logging
from src.data.digest_data import DataDigestion
from src.features.pca import PCAManager
from src.features.clustering import ClusterManager
from src.models.runner import run_models


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

    # Start flow
    # 1.) Data digestion
    logger.info("Initializing data digestion...")
    dig = DataDigestion(cfg)
    logger.info("Starting data digestion...")
    train, test = dig.run()
    logger.info("Data digestion completed.")

    # 2.) PCA and clustering
    logger.info("Starting Clustering and PCA pipeline...")
    # Run Clustering
    cluster_tool = ClusterManager(cfg)
    # Get cluster assignments as pd.Series to merge back later
    train_clusters, test_clusters = cluster_tool.run_clustering_pipeline(train, test)

    # Run PCA
    pca_tool = PCAManager(cfg)
    # Overwrite train/test with PCA versions (cleaned + encoded + scaled + PCA)
    train, test = pca_tool.run_pca_pipeline(train, test)
    
    # Merge cluster assignments back to PCA data
    train = train.join(train_clusters)
    test = test.join(test_clusters)

    # 3.) Checkpoint: Save to disk
    proc_dir = Path(cfg["data"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = proc_dir / "train_processed.csv"
    test_path = proc_dir / "test_processed.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info(f"Feature handoff saved to {proc_dir} as CSV files.")

    # 4.) Modeling
    logger.info("Starting modeling pipeline...")
    # results = run_models(cfg, train, test)
    # logger.info("Pipeline finished | train_shape=%s | test_shape=%s", train.shape, test.shape)
    
    # Log results summary
    # for model_name, result in results.items():
        # logger.info("Model: %s | Metrics: %s", model_name, result.metrics)


if __name__ == "__main__":
    main()
