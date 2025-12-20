import logging
from pathlib import Path
import pandas as pd

# Load data processing modules
from src.data.load_data import load_train_test
from src.data.imputation import knn_impute_numeric, mode_impute_categorical
from src.data.validation import validate_and_fix_numeric, clean_categoricals, assert_test_categories_subset
from src.data.encoding import encode_categoricals
from src.data.plotting import plot_distributions

# Create logger for this module
logger = logging.getLogger(__name__)

class DataDigestion:
    """
        Orchestrates data ingestion and cleaning pipeline.

        Attributes:
            cfg (dict): Configuration dictionary with paths, dtypes, columns, etc.

        Methods:
            __init__(): Initializes the DataDigestion with configuration
            run(): Executes the data ingestion and cleaning pipeline
            _log_missingness(): Logs missingness information for a given dataset
            _drop_duplicates(): Drops duplicate rows from train and test datasets
            _assert_alignment(): Asserts that train and test datasets have the same columns and dtypes
            _assert_no_nans(): Asserts that there are no NaN values in train and test datasets
    """
    def __init__(self, cfg: dict) -> None:
        logger.info("Initializing DataDigestion")

        self.cfg = cfg

        # Extract config values
        self.train_path = cfg["data"]["train_path"]
        self.test_path  = cfg["data"]["test_path"]
        self.processed_dir = cfg["data"]["processed_dir"]
        self.figures_dir = cfg["reports"]["figures_dir"]

        self.dtypes = cfg["dtypes"]
        self.num_cols = cfg["columns"]["numeric"]
        self.cat_cols = cfg["columns"]["categorical"]

        # ID and target columns (with defaults)
        self.id_col = cfg["data"].get("id_col", "IDpol")
        self.target_col = cfg["data"].get("target_col", "ClaimRate")

        # Categorical encoding configuration
        enc = cfg.get("encoding", {})
        self.ordinal_cols = enc.get("ordinal_cols", [])
        self.onehot_cols = enc.get("onehot_cols", [])
        self.ordinal_orders = enc.get("ordinal_orders", {})
        self.onehot_drop = enc.get("onehot_drop", None)
        self.unknown_ordinal_value = enc.get("unknown_ordinal_value", -1)

        # Validate ordinal orders are provided for all ordinal cols
        missing_orders = [c for c in self.ordinal_cols if c not in self.ordinal_orders]
        if missing_orders:
            logger.error("Missing explicit ordinal order(s) for: %s", missing_orders)
            raise ValueError(f"Missing explicit ordinal order(s) for: {missing_orders}")

        # Validate that encoding cols are in categorical cols
        bad_cols = [c for c in (self.ordinal_cols + self.onehot_cols) if c not in self.cat_cols]
        if bad_cols:
            logger.warning("Encoding cols not listed in cfg['columns']['categorical']: %s", bad_cols)

        # Can't allow overlap between ordinal and one-hot
        overlap = sorted(set(self.ordinal_cols) & set(self.onehot_cols))
        if overlap:
            logger.error("Columns cannot be both ordinal and one-hot: %s", overlap)
            raise ValueError(f"Columns cannot be both ordinal and one-hot: {overlap}")

        # All encoding columns that were provided
        enc_cols = self.ordinal_cols + self.onehot_cols

        # Validate that all categorical cols are covered by encoding
        missing_from_encoding = sorted(set(self.cat_cols) - set(enc_cols))
        extra_in_encoding = sorted(set(enc_cols) - set(self.cat_cols))
        if missing_from_encoding or extra_in_encoding:
            logger.error(
                "Encoding coverage mismatch | missing_from_encoding=%s | extra_in_encoding=%s",
                missing_from_encoding, extra_in_encoding
            )
            raise ValueError(
                "Encoding coverage mismatch | "
                f"missing_from_encoding={missing_from_encoding} | extra_in_encoding={extra_in_encoding}"
            )

        # Prepare output dirs early to avoid "directory not found" errors later
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)

        # Log configuration summary
        logger.info(
        "DataDigestion init | train_path=%s | test_path=%s | processed_dir=%s | figures_dir=%s",
        self.train_path, self.test_path, self.processed_dir, self.figures_dir
        )
        logger.debug(
            "Columns | numeric=%d | categorical=%d | id_col=%s | target_col=%s",
            len(self.num_cols), len(self.cat_cols), self.id_col, self.target_col
        )
        logger.debug(
            "Encoding | ordinal=%s | onehot=%s | onehot_drop=%s | unknown_ordinal_value=%s",
            self.ordinal_cols, self.onehot_cols, self.onehot_drop, self.unknown_ordinal_value
        )

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
            Executes the data ingestion and cleaning pipeline.
            Steps:
                - Load train/test datasets
                - Validate and fix numeric columns
                - Clean categorical columns
                - Drop duplicates
                - Log missingness
                - Impute missing values
                - Assert category subsets
                - Plot distributions
                - Encode categorical features
                - Final assertions

            Returns:
                tuple[pd.DataFrame, pd.DataFrame]: Cleaned training and testing datasets to pass on downstream
        """
        logger.info("Running DataDigestion pipeline")

        # Fetch raw data
        train, test = load_train_test(self.train_path, self.test_path, self.dtypes)

        # Basic sanity checks
        if self.id_col in train.columns and self.id_col in test.columns:
            logger.debug("ID column present | id_col=%s", self.id_col)
        else:
            logger.warning("ID column missing in train/test | id_col=%s", self.id_col)

        # Numeric validations / auto-fixes
        logger.info("Numeric validation/fixes started")
        train = validate_and_fix_numeric(train, self.target_col)
        test  = validate_and_fix_numeric(test, self.target_col)
        logger.info("Numeric validation/fixes finished")

        # Categorical cleaning (we handle 'A' and 'a' equally, strip spaces, etc.)
        logger.info("Categorical cleaning started")
        train = clean_categoricals(train, self.cat_cols)
        test  = clean_categoricals(test,  self.cat_cols)
        logger.info("Categorical cleaning finished")

        # Deduplication
        logger.info("Duplicate removal started")
        train, test = self._drop_duplicates(train, test)
        logger.info("Duplicate removal finished | train_shape=%s | test_shape=%s", train.shape, test.shape)

        # Log missingness before imputation for both datasets
        has_missing_train = self._log_missingness(train, "train")
        has_missing_test  = self._log_missingness(test,  "test")

        # Imputation only if we have missing values
        if not (has_missing_train or has_missing_test):
            logger.info("No missing values in train/test -> skipping imputation")
        else:
            # Imputation (fit on train, apply to both)
            logger.info("Imputation started")
            train, test = knn_impute_numeric(train, test, self.num_cols)
            train, test = mode_impute_categorical(train, test, self.cat_cols)
            logger.info("Imputation finished")

        # Final assertions
        logger.info("Category subset validation started")
        assert_test_categories_subset(train, test, self.cat_cols)
        logger.info("Category subset validation finished")

        # Plot distributions after cleaning/imputation
        logger.info("Plotting distributions started")
        plot_distributions(
            train=train,
            figures_dir=self.figures_dir,
            claim_col="ClaimNb",
        )
        logger.info("Plotting distributions finished")

        # Handle leakage for ClaimRate target (cant drop it earlier due to plotting)
        if self.target_col == "ClaimRate":
            logger.info("Target column is ClaimRate; dropping ClaimNb to avoid leakage")
            train.drop(columns=["ClaimNb"], inplace=True, errors="ignore")
            test.drop(columns=["ClaimNb"], inplace=True, errors="ignore")

        # Encoding
        logger.info("Encoding started")
        train_cat = train[self.cat_cols]
        test_cat  = test[self.cat_cols]
        train_enc, test_enc = encode_categoricals(
            train_cat=train_cat,
            test_cat=test_cat,
            ordinal_cols=self.ordinal_cols,
            onehot_cols=self.onehot_cols,
            ordinal_orders=self.ordinal_orders,
            onehot_drop=self.onehot_drop,
            unknown_ordinal_value=self.unknown_ordinal_value,
        )
        # Replace original categorical columns with encoded versions
        train = pd.concat([train.drop(columns=self.cat_cols), train_enc], axis=1)
        test  = pd.concat([test.drop(columns=self.cat_cols),  test_enc], axis=1)
        logger.info("Encoding finished")

        # Structural assertions: columns + dtypes + no missing values
        logger.info("Final assertions started")
        self._assert_alignment(train, test)
        self._assert_no_nans(train, test)
        logger.info("Final assertions passed")

        logger.info("DataDigestion run finished")
        return train, test

    def _log_missingness(self, df: pd.DataFrame, name: str) -> bool:
        """
            Logs missingness % per column (only columns with missing values)
            
            Args:
                df (pd.DataFrame): The DataFrame to analyze
                name (str): Name identifier for logging (e.g., "train" or "test")

            Returns:
                bool: True if there were missing values, False otherwise
        """
        logger.debug("Missingness scan started | name=%s | shape=%s", name, df.shape)

        # Check for missing values with .isna() -> returns boolean DataFrame with True for NaNs
        # then .mean() to get % missing per column
        # filter to only columns with missing values
        miss = df.isna().mean().sort_values(ascending=False) * 100
        miss = miss[miss > 0]
        if len(miss) == 0:
            logger.info("%s: no missing values", name)
            return False

        # If we reach here, there are missing values to log
        logger.info("%s missing %%:\n%s", name, miss.to_string())
        logger.debug("Missingness scan finished | name=%s | cols_with_missing=%d", name, len(miss))

        return True

    def _drop_duplicates(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
            Drops duplicate rows from train and test datasets.
            Full-row dupes are usually safe to drop; ID dupes are riskier (we keep first and log second)

            Args:
                train (pd.DataFrame): Training dataset
                test (pd.DataFrame): Test dataset

            Returns:
                tuple[pd.DataFrame, pd.DataFrame]: Deduplicated training and test datasets
        """
        logger.debug("Deduplication started | train_shape=%s | test_shape=%s", train.shape, test.shape)

        # Check for duplicates in train dataset, drop if any and log
        n = int(train.duplicated().sum())
        if n:
            logger.info("[DUPES] train full-row: drop %d", n)
            train = train.drop_duplicates()
        else:
            logger.debug("[DUPES] train full-row: none")

        # Check for duplicates in test dataset, drop if any and log
        n = int(test.duplicated().sum())
        if n:
            logger.info("[DUPES] test full-row: drop %d", n)
            test = test.drop_duplicates()
        else:
            logger.debug("[DUPES] test full-row: none")

        # Now check for ID-based duplicates if ID column is present
        if self.id_col in train.columns:
            n = int(train.duplicated(subset=[self.id_col]).sum())
            # If we have ID dupes, we keep first and log a warning
            if n:
                logger.debug("[DUPES] train %s: drop %d keep first", self.id_col, n)
                train = train.drop_duplicates(subset=[self.id_col], keep="first")
            else:
                logger.debug("[DUPES] train %s: none", self.id_col)
            # Then drop ID column from train to avoid leakage downstream
            train.drop(columns=[self.id_col], inplace=True)
        else:
            logger.warning("[DUPES] train id_col missing: %s", self.id_col)
    
        # Do the same for test dataset
        if self.id_col in test.columns:
            n = int(test.duplicated(subset=[self.id_col]).sum())
            if n:
                logger.debug("[DUPES] test %s: drop %d keep first", self.id_col, n)
                test = test.drop_duplicates(subset=[self.id_col], keep="first")
            else:
                logger.debug("[DUPES] test %s: none", self.id_col)
            # Then drop ID column from test to avoid leakage downstream
            test.drop(columns=[self.id_col], inplace=True)
        else:
            logger.warning("[DUPES] test id_col missing: %s", self.id_col)

        return train, test

    def _assert_alignment(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
            Asserts that train and test datasets have the same columns and dtypes.

            Args:
                train (pd.DataFrame): Training dataset
                test (pd.DataFrame): Test dataset

            Raises:
                ValueError: If columns or dtypes do not match between train and test
        """
        logger.debug("Alignment check started")

        # Get the columns into lists for comparison
        train_cols = list(train.columns)
        test_cols  = list(test.columns)
        if train_cols != test_cols:
            # Log the diff to reduce debugging time
            missing_in_test = [c for c in train_cols if c not in test_cols]
            missing_in_train = [c for c in test_cols if c not in train_cols]
            logger.error("[STOP] Train/test column mismatch | missing_in_test=%s | missing_in_train=%s",
                              missing_in_test, missing_in_train)
            raise ValueError("Train/test columns mismatch")

        # Then a sanity check for matching dtypes
        for c in train.columns:
            if str(train[c].dtype) != str(test[c].dtype):
                logger.error("[STOP] dtype mismatch %s: %s vs %s", c, train[c].dtype, test[c].dtype)
                raise ValueError(f"dtype mismatch on {c}")
            
        logger.debug("Alignment check passed")

    def _assert_no_nans(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
            Asserts that there are no NaN values in train and test datasets.

            Args:
                train (pd.DataFrame): Training dataset
                test (pd.DataFrame): Test dataset

            Raises:
                ValueError: If any NaN values are found in train or test
        """
        logger.debug("NaN check started")

        # Fetch the number of NaNs in both datasets
        tn = int(train.isna().sum().sum())
        sn = int(test.isna().sum().sum())
        if tn or sn:
            # Log top columns with NaNs for easier debugging
            top_train = train.isna().sum().sort_values(ascending=False).head(10)
            top_test  = test.isna().sum().sort_values(ascending=False).head(10)
            logger.error("[STOP] Residual NaNs: train=%d, test=%d", tn, sn)
            # Log top NaNs columns
            logger.error("Top NaNs train:\n%s", top_train[top_train > 0].to_string())
            logger.error("Top NaNs test:\n%s", top_test[top_test > 0].to_string())
            raise ValueError("Residual NaNs remain")

        logger.debug("NaN check passed")
