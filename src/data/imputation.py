import logging
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer

# create logger for this module
logger = logging.getLogger(__name__)

def knn_impute_numeric(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], n_neighbors: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        KNN imputation for numeric columns for both dataset but fitted on train only.

        Args:
            train (pd.DataFrame): Training dataset
            test (pd.DataFrame): Test dataset
            cols (list[str]): List of numeric columns to impute
            n_neighbors (int): Number of neighbors for KNN imputer

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Imputed training and test datasets
    """
    logger.info("KNN numeric impute | requested_cols=%d | n_neighbors=%d | train_shape=%s | test_shape=%s",
                len(cols), n_neighbors, train.shape, test.shape)

    cols = [c for c in cols if c in train.columns]
    # Should not have any missing columns but in case we have log
    missing_cols = [c for c in cols if c not in train.columns]
    if missing_cols:
        logger.warning("KNN numeric impute | missing_cols=%s", missing_cols)

    # If we have no matching columns, skip
    if not cols:
        logger.info("KNN numeric impute skipped | reason=no_matching_cols")
        return train, test

    # Convert to numeric for imputer (KNNImputer expects numeric arrays)
    train_num = train[cols].astype("float64")
    test_num  = test[cols].astype("float64")

    # Scale using TRAIN stats only (prevents leakage + distance domination)
    mu = train_num.mean(skipna=True)
    sigma = train_num.std(skipna=True).replace(0.0, 1.0)  # avoid divide-by-zero for constant columns
    logger.debug("KNN numeric impute | train_mu=%s", mu.to_dict())
    logger.debug("KNN numeric impute | train_sigma=%s", sigma.to_dict())

    # Scale the data
    train_scaled = (train_num - mu) / sigma
    test_scaled  = (test_num  - mu) / sigma

    # Fit on train only (prevents leakage)
    logger.info("KNN numeric impute | fitting imputer on train (scaled)")
    imp = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    # Store the training data to compute neighbors and figure out which values to impute
    imp.fit(train_scaled)

    logger.info("KNN numeric impute | transforming train/test")
    # Then go row-by-row to check missing values and impute with closest neighbors
    train_imp = imp.transform(train_scaled)
    test_imp  = imp.transform(test_scaled)

    # Write back in original scale
    train.loc[:, cols] = train_imp * sigma.values + mu.values
    test.loc[:, cols]  = test_imp  * sigma.values + mu.values

    # Log missingness after imputation (we already logged the before state)
    miss_after = train[cols].isna().sum()
    total_missing_after = int(miss_after.sum())
    logger.info("KNN numeric impute | missing_after_train=%d",
                total_missing_after)

    return train, test


def mode_impute_categorical(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Mode (most-frequent) imputation for categorical columns for both dataset but fitted on train only.

        Args:
            train (pd.DataFrame): Training dataset
            test (pd.DataFrame): Test dataset
            cols (list[str]): List of categorical columns to impute

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Imputed training and test datasets
    """
    logger.info("Mode categorical impute | requested_cols=%d | train_shape=%s | test_shape=%s",
                len(cols), train.shape, test.shape)
    
    cols = [c for c in cols if c in train.columns]
    # Should not have any missing columns but in case we have log
    missing_cols = [c for c in cols if c not in train.columns]
    if missing_cols:
        logger.warning("Mode categorical impute | missing_cols=%s", missing_cols)

    # If we have no matching columns, skip
    if not cols:
        logger.info("Mode categorical impute skipped | reason=no_matching_cols")
        return train, test

    # Make copies of only the categorical columns to impute
    train_cat = train[cols].astype("object")
    test_cat  = test[cols].astype("object")

    # SimpleImputer can't mask pd.NA so convert to np.nan
    train_cat = train_cat.where(train_cat.notna(), np.nan)
    test_cat  = test_cat.where(test_cat.notna(), np.nan)

    # Set up and fit the imputer on train only (prevents leakage)
    logger.info("Mode categorical impute | initializing and fitting imputer on train")
    imp = SimpleImputer(strategy="most_frequent")
    imp.fit(train_cat)

    # Then iterate through both datasets to impute missing values
    logger.info("Mode categorical impute | transforming train/test")
    train.loc[:, cols] = imp.transform(train_cat)
    test.loc[:, cols]  = imp.transform(test_cat)

    # Since we logged the missingness before, log after state
    miss_after = train[cols].isna().sum()
    total_missing_after = int(miss_after.sum())
    logger.info("Mode categorical impute | missing_after_train=%d",
                total_missing_after)

    return train, test
