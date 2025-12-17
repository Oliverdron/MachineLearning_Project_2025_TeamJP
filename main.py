import json
import logging

from src.config.logging import setup_logging
from src.data.digest_data import DataDigestion


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
    # Should save the data here in np format as a checkpoint
    # Then check if these files exist, if so, skip previous steps and do modeling

    # 3.) Modeling
    # Will get the np arrays, call decision tree and feed forward neural network implementations
    
    logger.info("Pipeline finished | train_shape=%s | test_shape=%s", train.shape, test.shape)


if __name__ == "__main__":
    main()
