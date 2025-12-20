import json
import logging
import os
import numpy as np
from src.features.pca import PCAManager
from src.features.clustering import ClusterManager
from src.config.logging import setup_logging
from src.data.digest_data import DataDigestion
# pca and clustering imports would go here
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
  
    logger.info("Initializing Feature Engineering (PCA + Clustering)...")
    
    # Run PCA
    pca_tool = PCAManager(cfg)
    train_pca, test_pca = pca_tool.run_pca_pipeline(train, test)
    
    # Run Clustering
    cluster_tool = ClusterManager(cfg)
    
   
    train_final_arr, test_final_arr = cluster_tool.run_clustering_pipeline(train_pca, test_pca)

    # 3.) Checkpoint: Save to disk
    
    proc_dir = cfg["data"]["postpca_dir"]
    os.makedirs(proc_dir, exist_ok=True)
    
    np.save(os.path.join(proc_dir, "train_features.npy"), train_final_arr)
    np.save(os.path.join(proc_dir, "test_features.npy"), test_final_arr)
    logger.info(f"Feature handoff saved to {proc_dir} as NumPy arrays.")

    # Update variables for the next step (Modeling)
    train, test = train_final_arr, test_final_arr

    # 3.) Modeling
    logger.info("Starting modeling pipeline...")
    results = run_models(cfg, train, test)
    logger.info("Pipeline finished | train_shape=%s | test_shape=%s", train.shape, test.shape)
    
    # Log results summary
    for model_name, result in results.items():
        logger.info("Model: %s | Metrics: %s", model_name, result.metrics)


if __name__ == "__main__":
    main()
