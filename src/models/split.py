import logging
import numpy as np

logger = logging.getLogger(__name__)


def k_fold_indices(n_samples: int, k: int, seed: int = 42, shuffle: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate k-fold train/validation indices.

    Args:
        n_samples (int): Number of samples in the dataset
        k (int): Number of folds
        seed (int): Random seed for shuffling
        shuffle (bool): Whether to shuffle before splitting

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    # Sanity checks
    if k < 2:
        logger.error("k must be at least 2 for k-fold cross-validation.")
        raise ValueError(f"k must be >= 2, got {k}")
    if n_samples < k:
        logger.error("n_samples must be >= k.")
        raise ValueError(f"n_samples must be >= k, got n_samples={n_samples}, k={k}")

    idx = np.arange(n_samples)
    # Shuffle indices if required
    if shuffle:
        # Seed the random number generator for reproducibility
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    # Split indices into k folds (approximately equal size and without overlap)
    folds = np.array_split(idx, k)
    # Create train/val index pairs
    out = []
    # Iterate over each fold to create train and validation indices
    for i in range(k):
        # Validation indices are the current fold
        val_idx = folds[i]
        # Training indices are all other folds concatenated
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        # Append the train/val index pair to the output list
        out.append((train_idx, val_idx))

    logger.info("Created %d-fold split: n_samples=%d | shuffle=%s", k, n_samples, shuffle)
    return out
