import pandas as pd
import logging
from time import perf_counter # benchmarking
from pathlib import Path

# create logger for this module
logger = logging.getLogger(__name__)

def load_csv(path: str, dtypes: dict) -> pd.DataFrame:
    """
        Load data from a CSV file into a pandas DataFrame with specified dtypes.

        Args:
            path (str): The file path to the CSV file
            dtypes (dict): A dictionary specifying the data types for each column

        Returns:
            pd.DataFrame: The loaded DataFrame
    """
    # convert string to Path object (OS correct separators)
    p = Path(path)
    logger.info("Loading CSV from path: %s", p)

    # check file existence and raise error if not found
    if not p.exists():
        logger.error("CSV not found: %s", p)
        raise FileNotFoundError(p)

    # load CSV with benchmarking
    t0 = perf_counter()
    # wrap in try/except to catch and log any errors
    try:
        # load CSV with specified dtypes
        df = pd.read_csv(p, dtype=dtypes)
    except Exception:
        # includes stack trace automatically
        logger.exception("Failed reading CSV: %s", p)
        raise

    # stop benchmarking and log time taken
    secs = perf_counter() - t0
    logger.info("Loaded CSV: %s | rows=%d cols=%d | shape=%s | seconds=%.3f",
                p, df.shape[0], df.shape[1], df.shape, secs)

    # small sanity checks
    if df.empty:
        logger.warning("CSV is empty: %s", p)

    # check the loaded dtypes
    logger.debug("Dtypes for %s:\n%s", p, df.dtypes.to_string())

    return df


def load_train_test(train_path: str, test_path: str, dtypes: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Load train and test datasets from CSV files.

        Args:
            train_path (str): The file path to the training CSV
            test_path (str): The file path to the testing CSV
            dtypes (dict): A dictionary specifying the data types for each column

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames
    """
    logger.info("Processing train dataset")
    train = load_csv(train_path, dtypes)

    logger.info("Processing test dataset")
    test = load_csv(test_path, dtypes)

    logger.info("Loaded train/test | train_shape=%s test_shape=%s", train.shape, test.shape)
    return train, test
