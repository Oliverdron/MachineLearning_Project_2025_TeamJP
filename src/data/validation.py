import logging
import numpy as np
import pandas as pd

# create logger for this module
logger = logging.getLogger(__name__)

def validate_and_fix_numeric(df: pd.DataFrame, target_col: str = "ClaimNb") -> pd.DataFrame:
    """
        Validates and applies safe fixes to numeric columns like:
            - Exposure
            - ClaimNb / ClaimRate
            - VehAge
            - DrivAge
            - BonusMalus
            - VehPower

        Args:
            df (pd.DataFrame): The input DataFrame to validate and fix
            target_col (str): The name of the target column to validate
        
        Returns:
            pd.DataFrame: The validated and fixed DataFrame
    """
    # Log starting state
    logger.info("Validate/fix numeric | target_col=%s | df_shape=%s", target_col, df.shape)

    # Quick check of expected numeric columns
    expected_cols = [target_col, "Exposure", "VehAge", "DrivAge", "BonusMalus", "VehPower"]
    present = [c for c in expected_cols if c in df.columns]
    missing = [c for c in expected_cols if c not in df.columns]
    # Make sure we don't try to fix missing columns
    if missing:
        logger.debug("Numeric columns present=%s | missing=%s", present, missing)
        raise ValueError(f"Missing expected numeric columns: {missing}")

    total_fixes = 0
    total_flags = 0

    # Exposure: (0, 1]: identify and fix bad values
    bad = ~df["Exposure"].between(0.0, 1.0, inclusive="right")
    # Count the number of bad values (e.g. out-of-range or non-finite)
    n = int(bad.sum())
    if n:
        logger.warning("[FIX] Exposure out of range: %d -> clip to (0,1].", n)
        # As said before, we clip to a small positive value to avoid zero and max of 1.0
        df.loc[bad, "Exposure"] = df.loc[bad, "Exposure"].clip(lower=1e-6, upper=1.0)
        # Update total fixes stats
        total_fixes += n
        # Log the updated stats
        logger.debug("Exposure stats after  | min=%s max=%s",
                     df["Exposure"].min(), df["Exposure"].max())


    # ClaimNb: integer >= 0
    if target_col == "ClaimNb":
        # Since the read-in is based on our pre-defined dtypes, we are safe to assume numeric here
        bad = (df[target_col] < 0)
        # Count the number of bad values
        n = int(bad.sum())
        if n:
            logger.warning("[DROP] %s invalid (negative) rows: %d -> dropping rows.", target_col, n)
            # since we cannot auto-fix negative ClaimNb, we drop the rows
            df = df.loc[~bad].copy()
            total_fixes += n

        # Ensures target is integer-valued and compact dtype for memory/perf
        df[target_col] = np.round(df[target_col]).astype("int32", copy=False)

        logger.debug("%s stats after  | min=%s max=%s dtype=%s",
                 target_col, df[target_col].min(), df[target_col].max(), df[target_col].dtype)

    # ClaimRate: float in [0, some_upper_bound)
    elif target_col == "ClaimRate":
        # Use a statistical upper bound to identify bad ClaimRate values
        upper_bound = df[target_col].mean() + 6 * df[target_col].std()
        bad = ~df[target_col].between(0.0, upper_bound, inclusive="both")
        n = int(bad.sum())
        if n:
            logger.warning("[CLIP] %s out of range: %d -> clip to [0, %.3f].", target_col, n, upper_bound)
            # Clip ClaimRate to valid range
            df.loc[bad, target_col] = df.loc[bad, target_col].clip(lower=0.0, upper=upper_bound)
            total_fixes += n

            logger.debug("%s stats after | min=%s max=%s",
                     target_col, df[target_col].min(), df[target_col].max())
    # Other target columns are not recognized
    else:
        logger.warning("Unknown target_col for numeric validation: %s", target_col)
        raise ValueError(f"Unknown target_col: {target_col}")
    

    # VehAge >= 0: drop rows with negative VehAge
    bad = df["VehAge"] < 0
    n = int(bad.sum())
    if n:
        logger.warning("[DROP] VehAge invalid (negative) rows: %d -> dropping rows.", n)
        # Drop rows with invalid VehAge
        df = df.loc[~bad].copy()
        total_fixes += n

        logger.debug("VehAge stats after  | min=%s max=%s",
                     df["VehAge"].min(), df["VehAge"].max())


    # DrivAge: drop rows where DrivAge < 18
    mask_invalid_age = df["DrivAge"] < 18
    n = int(mask_invalid_age.sum())
    if n:
        logger.warning("[DROP] DrivAge < 18: %d row(s) dropped.", n)
        # Drop rows with invalid DrivAge
        df = df.loc[~mask_invalid_age].copy()
        total_flags += n

        logger.debug(
            "DrivAge stats (post-clean) | min=%s max=%s",
            df["DrivAge"].min(), df["DrivAge"].max()
        )

    # BonusMalus: clip to [50, 350]
    mask_outside_bonus = ~df["BonusMalus"].between(50, 350, inclusive="both")
    n = int(mask_outside_bonus.sum())
    if n:
        logger.warning("[CLIP] BonusMalus outside [50,350]: %d value(s) clipped.", n)
        # Clip BonusMalus to the valid range
        df["BonusMalus"] = df["BonusMalus"].clip(lower=50, upper=350)
        total_flags += n

        logger.debug(
            "BonusMalus stats (post-fix) | min=%s max=%s",
            df["BonusMalus"].min(), df["BonusMalus"].max()
        )

    # VehPower: drop values <= 0
    bad = df["VehPower"] <= 0
    n = int(bad.sum())
    if n:
        logger.warning("[DROP] VehPower <= 0: %d row(s) dropped.", n)
        # Drop rows with invalid VehPower
        df = df.loc[~bad].copy()
        total_flags += n

        logger.debug("VehPower stats | min=%s max=%s",
                     df["VehPower"].min(), df["VehPower"].max())

    # Log final summary
    logger.info("Validate/fix numeric done | fixes=%d | flags=%d | df_shape=%s",
                total_fixes, total_flags, df.shape)

    return df


def clean_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
        Cleans categorical columns by:
            - Stripping leading/trailing whitespace
            - Converting to lowercase

        Args:
            df (pd.DataFrame): The input DataFrame to clean
            cols (list[str]): List of categorical columns to clean

        Returns:
            pd.DataFrame: The cleaned DataFrame
    """
    # Log starting state
    logger.info("Cleaning categoricals | requested_cols=%d | df_shape=%s", len(cols), df.shape)

    cleaned = 0
    skipped_missing = 0

    for c in cols:
        if c in df.columns:
            # Log stats before cleaning for traceability
            logger.debug("Cleaning col=%s | non_null=%d | unique_non_null=%d",
                         c, int(df[c].notna().sum()), int(df[c].dropna().nunique()))

            df[c] = df[c].astype("string").str.strip().str.lower()
            cleaned += 1
        else:
            # Column list might come from config; shouldn't have missing columns but log just in case
            logger.debug("Skipping missing categorical col=%s", c)
            skipped_missing += 1

    # Final log and return
    logger.info("Finished cleaning categoricals | cleaned=%d | missing_cols_skipped=%d", cleaned, skipped_missing)
    return df


def assert_test_categories_subset(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> None:
    """
        Asserts that the categories in the test set are a subset of those in the training set for the specified categorical columns.
        Raises an error if any unknown categories are found.

        Args:
            train (pd.DataFrame): The training dataset
            test (pd.DataFrame): The testing dataset
            cols (list[str]): The categorical columns to check

        Raises:
            ValueError: If unknown categories are found in the test set
    """
    logger.info("Validating test categories are subset of train | cols=%d | train_shape=%s | test_shape=%s",
                len(cols), train.shape, test.shape)

    checked = 0
    skipped_missing = 0
    violated = 0

    for c in cols:
        # Should not have missing columns but log and skip just in case
        if c not in train.columns:
            logger.debug("Skipping validation for missing train col=%s", c)
            skipped_missing += 1
            continue
        # Should not have missing columns but log and skip just in case
        if c not in test.columns:
            logger.debug("Skipping validation for missing test col=%s", c)
            skipped_missing += 1
            continue

        checked += 1

        # Get the unique categories in train and test
        train_set = set(train[c].dropna().unique())
        test_set  = set(test[c].dropna().unique())
        # Then compare the two sets to find unknown categories in test
        unknown = sorted(test_set - train_set)

        # Log stats for this column
        logger.debug("Category stats col=%s | train_unique=%d | test_unique=%d | unknown=%d",
                     c, len(train_set), len(test_set), len(unknown))

        # Check if we have a violation
        if unknown:
            violated += 1
            # Keep the log readable; show first N, but still fail hard
            logger.error("[STOP] Unknown categories in test for '%s': %s", c, unknown[:20])
            raise ValueError(f"Unknown test categories in column '{c}'")

    # Otherwise, all good
    logger.info("Category subset validation passed | checked=%d | skipped=%d | violations=%d",
                checked, skipped_missing, violated)
