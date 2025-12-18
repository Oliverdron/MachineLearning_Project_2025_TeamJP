import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Best-effort global seeding for reproducibility across various libraries.

    Args:
        seed (int): The seed to set globally

    Raises:
        ValueError: If seed is None
    """
    if seed is None:
        logger.error("Seed value is None; cannot set global seed.")
        raise ValueError("seed must be an int (got None)")

    # Set seeds for various libraries
    random.seed(seed)
    np.random.seed(seed)

    # Hash seed affects dict/set iteration order in some contexts
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    except ImportError:
        logger.info("Torch not installed; seeded python & numpy only.")

    logger.info("Global seed set to %d", seed)
