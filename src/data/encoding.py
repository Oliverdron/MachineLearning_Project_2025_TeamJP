import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# create logger for this module
logger = logging.getLogger(__name__)


def encode_ordinal(
    train_cat: pd.DataFrame,
    test_cat: pd.DataFrame,
    cols: list[str],
    orders: dict[str, list[str]],
    unknown_value: int = -1,
    dtype: np.dtype = np.int16,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]:
    """
        Ordinal encode cols based on provided orders. Unseen categories in test are set to unknown_value.

        Args:
            train_cat (pd.DataFrame): Training categorical DataFrame
            test_cat (pd.DataFrame): Testing categorical DataFrame
            cols (list[str]): List of columns to ordinal encode
            orders (dict[str, list[str]]): Mapping of column names to their explicit category order
            unknown_value (int): Value to assign to unseen categories in test set

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]]: Encoded training and testing DataFrames, and mapping dictionaries
    """
    train_out = pd.DataFrame(index=train_cat.index)
    test_out = pd.DataFrame(index=test_cat.index)
    # Store mapping dictionaries for each column to return
    maps: dict[str, dict[str, int]] = {}

    # Early exit if no cols to encode
    if not cols:
        logger.info("Ordinal encoding skipped | reason=no_cols")
        return train_out, test_out, maps

    # Validate that every requested col has an explicit order
    missing_orders = [c for c in cols if c not in orders]
    if missing_orders:
        logger.error("Missing explicit ordinal order(s) for: %s", missing_orders)
        raise ValueError(f"Missing explicit ordinal order(s) for: {missing_orders}")

    # Validate that the columns exist in both frames
    missing_in_train = [c for c in cols if c not in train_cat.columns]
    missing_in_test = [c for c in cols if c not in test_cat.columns]
    if missing_in_train or missing_in_test:
        logger.error("Missing columns | train_missing=%s | test_missing=%s",
                     missing_in_train, missing_in_test)
        raise ValueError(f"Missing columns | train_missing={missing_in_train} | test_missing={missing_in_test}")

    logger.info("Ordinal encoding started | cols=%s", cols)

    for col in cols:
        # Force order values to strings and check for duplicates (as order maps should not have duplicates)
        order_list = [str(x) for x in orders[col]]
        if len(order_list) != len(set(order_list)):
            logger.error("Duplicate values in ordinal order for col=%s: %s", col, order_list)
            raise ValueError(f"Duplicate values in ordinal order for col={col}: {order_list}")

        # Create mapping dict: category value -> ordinal integer
        mapping = {v: i for i, v in enumerate(order_list)}
        maps[col] = mapping

        # Force raw values to string for consistent mapping
        tr_raw = train_cat[col].astype("string")
        te_raw = test_cat[col].astype("string")

        # Hard stop, if the train set has categories not in the mapping
        train_vals = set(tr_raw.dropna().unique().tolist())
        unknown_in_train = sorted(train_vals - set(mapping.keys()))
        if unknown_in_train:
            logger.error("Train has category(ies) not in provided order | col=%s | n=%d | sample=%s",
                         col, len(unknown_in_train), unknown_in_train[:10])
            raise ValueError(
                f"Train has category(ies) not in provided order | col={col} | "
                f"n={len(unknown_in_train)} | sample={unknown_in_train[:10]}"
            )

        # Map train and test values to predefined ordinals
        tr = tr_raw.map(mapping)
        te = te_raw.map(mapping)

        # Test set can have values that are not in train; set those to unknown_value
        unseen_test = int(te_raw.notna().sum() - te.notna().sum())
        if unseen_test:
            logger.warning(
                "[ORDINAL] unseen categories in test | col=%s | n=%d -> set to %d",
                col, unseen_test, unknown_value
            )
            # te_raw.isna() gives the true missing mask (as after mapping we might introduce new Nones)
            # but we only want to replace the newly unseen values, not the original ones
            te = te.where(te_raw.isna(), te.fillna(unknown_value))

        # Keep NaNs as NaNs; output numeric
        train_out[col] = tr.astype(dtype)
        test_out[col] = te.astype(dtype)

    # Debug mapping info
    logger.info("Ordinal encoding started | cols=%s | mapping=%s", cols, maps)

    return train_out, test_out, maps


def encode_onehot(
    train_cat: pd.DataFrame,
    test_cat: pd.DataFrame,
    cols: list[str],
    drop: str = "first",
    dtype: np.dtype = np.int8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str]]:
    """
        One-hot encode specified categorical columns using sklearn OneHotEncoder.
        Unseen categories in test are ignored by the encoder.

        Args:
            train_cat (pd.DataFrame): Training categorical DataFrame
            test_cat (pd.DataFrame): Testing categorical DataFrame
            cols (list[str]): List of columns to one-hot encode
            drop (str): Drop strategy for one-hot encoding ("first", "if_binary", or None to keep all)
            dtype (np.dtype): Data type for the output one-hot encoded DataFrames

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], list[str]]:
                Encoded training and testing DataFrames, mapping of categories per column, and list of feature names
    """
    train_out = pd.DataFrame(index=train_cat.index)
    test_out = pd.DataFrame(index=test_cat.index)
    categories_map: dict[str, list[str]] = {}
    feat_names: list[str] = []

    # Early exit if no cols
    if not cols:
        logger.info("One-hot encoding skipped | reason=no_cols")
        return train_out, test_out, categories_map, feat_names

    # Validate columns exist in both frames
    missing_in_train = [c for c in cols if c not in train_cat.columns]
    missing_in_test = [c for c in cols if c not in test_cat.columns]
    if missing_in_train or missing_in_test:
        logger.error("Missing columns | train_missing=%s | test_missing=%s",
                     missing_in_train, missing_in_test)
        raise ValueError(f"Missing columns | train_missing={missing_in_train} | test_missing={missing_in_test}")

    logger.info("One-hot encoding started | cols=%s", cols)

    # Force to pandas 'string' dtype for stable category handling and logging
    tr_raw = train_cat[cols].astype("string")
    te_raw = test_cat[cols].astype("string")

    # Log unseen categories in test (not a failure; encoder ignores them)
    for col in cols:
        tr_set = set(tr_raw[col].dropna().unique().tolist())
        te_set = set(te_raw[col].dropna().unique().tolist())
        unseen = sorted(te_set - tr_set)
        if unseen:
            logger.warning("[ONEHOT] unseen categories in test | col=%s | n=%d (encoder will ignore) | sample=%s",
                           col, len(unseen), unseen[:10])

    # Initialize OneHotEncoder
    try:
        # first with sparse_output (newer sklearn)
        ohe = OneHotEncoder(handle_unknown="ignore", drop=drop, sparse_output=False)
    except TypeError:
        # fallback to sparse (older sklearn)
        ohe = OneHotEncoder(handle_unknown="ignore", drop=drop, sparse=False)

    # Fit on train only (prevents leakage)
    ohe.fit(tr_raw)

    # Then transform both train and test
    train_arr = ohe.transform(tr_raw)
    test_arr = ohe.transform(te_raw)

    # Feature names (stable)
    try:
        feat_names = ohe.get_feature_names_out(cols).tolist()
    except Exception:
        logger.debug("[ONEHOT] get_feature_names_out not available; constructing feature names manually")
        feat_names = []
        for c, cats in zip(cols, ohe.categories_):
            for cat in cats:
                feat_names.append(f"{c}__{cat}")

    # Create output DataFrames with proper indices, data type and column names
    train_out = pd.DataFrame(train_arr, index=train_cat.index, columns=feat_names, dtype=dtype)
    test_out = pd.DataFrame(test_arr, index=test_cat.index, columns=feat_names, dtype=dtype)

    # Categories learned per column (from train) - cast to string for consistency
    for c, cats in zip(cols, ohe.categories_):
        categories_map[c] = [str(x) for x in cats]

    # Log stats and return
    logger.info(
        "One-hot encoding finished | new_features=%d | train_shape=%s | test_shape=%s",
        len(feat_names), train_out.shape, test_out.shape
    )

    return train_out, test_out, categories_map, feat_names


def encode_categoricals(
    train_cat: pd.DataFrame,
    test_cat: pd.DataFrame,
    ordinal_cols: list[str],
    onehot_cols: list[str],
    ordinal_orders: dict[str, list[str]],
    onehot_drop: str | None = None,
    unknown_ordinal_value: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
        Encode categorical columns using a mix of ordinal and one-hot encoding.
        Args:
            train_cat (pd.DataFrame): Training categorical DataFrame
            test_cat (pd.DataFrame): Testing categorical DataFrame
            ordinal_cols (list[str]): List of columns to ordinal encode
            onehot_cols (list[str]): List of columns to one-hot encode
            ordinal_orders (dict[str, list[str]]): Mapping of column names to their explicit category order for ordinal encoding
            onehot_drop (str | None): Drop strategy for one-hot encoding ("first", "if_binary", or None to keep all)
            unknown_ordinal_value (int): Value to assign to unseen categories in test set for ordinal encoding

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, dict]: Encoded training and testing DataFrames
    """
    logger.info("Categorical encoding started | ordinal=%s | onehot=%s", ordinal_cols, onehot_cols)

    # Ordinal encoding
    tr_ord, te_ord, ord_maps = encode_ordinal(
        train_cat=train_cat,
        test_cat=test_cat,
        cols=ordinal_cols,
        orders=ordinal_orders,
        unknown_value=unknown_ordinal_value,
    )

    # One-hot encoding
    tr_ohe, te_ohe, ohe_cats, feat_names = encode_onehot(
        train_cat=train_cat,
        test_cat=test_cat,
        cols=onehot_cols,
        drop=onehot_drop,
    )

    # Combine both encoded parts
    train_enc = pd.concat([tr_ord, tr_ohe], axis=1)
    test_enc = pd.concat([te_ord, te_ohe], axis=1)

    # Construct artifacts dict
    artifacts = {
        "ordinal_maps": ord_maps,
        "onehot_categories": ohe_cats,
        "onehot_feature_names": feat_names,
        "ordinal_cols": ordinal_cols,
        "onehot_cols": onehot_cols,
        "onehot_drop": onehot_drop,
        "unknown_ordinal_value": unknown_ordinal_value,
    }

    # Final log and return
    logger.info("Categorical encoding finished | train_shape=%s | test_shape=%s | artifacts=%s", train_enc.shape, test_enc.shape, artifacts)
    return train_enc, test_enc
