import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config.seed import set_global_seed
from src.evaluation.metrics import METRICS, compute_metrics
from src.evaluation.performance import evaluate_and_plot
from src.models.registry import ModelSpec, init_registry
from src.models.results import ModelResult
from src.models.split import k_fold_indices

logger = logging.getLogger(__name__)


def _is_better(new: float, best: float, direction: str) -> bool:
    """
    Determine if the new score is better than the best score based on the direction.

    Args:
        new (float): New score to compare
        best (float): Best score so far
        direction (str): "min" if lower is better, "max" if higher is better

    Returns:
        bool: True if new is better than best, False otherwise
    """
    if direction == "min":
        return new < best
    if direction == "max":
        return new > best
    logger.error("Unknown direction: %s", direction)
    raise ValueError(f"Unknown direction: {direction}")


def _default_params(params: dict, param_space: dict) -> dict:
    """
    Fetch the first value for each hyperparam from param_space.
    
    Args:
        params (dict): Base parameters
        param_space (dict): Hyperparameter search space

    Returns:
        dict: Configuration dictionary with default hyperparameter values
    """
    # Get the config template
    cfg = deepcopy(params)
    # Iterate over each hyperparameter in the search space
    for k, choices in param_space.items():
        # Set the hyperparameter to its first value
        cfg[k] = choices[0]
    return cfg


def _sample_params(params: dict, param_space: dict, rng: np.random.Generator) -> dict:
    """
    Randomly sample one value per hyperparam in param_space.
    
    Args:
        params (dict): Base parameters
        param_space (dict): Hyperparameter search space
        rng (np.random.Generator): Random number generator for sampling

    Returns:
        dict: Configuration dictionary with sampled hyperparameter values
    """
    # Get the config template
    cfg = deepcopy(params)
    # Iterate over each hyperparameter in the search space
    for k, choices in param_space.items():
        # Randomly sample one value for the hyperparameter
        cfg[k] = choices[int(rng.integers(0, len(choices)))]
    return cfg


def _make_candidates(params: dict, param_space: dict, tune_hyperparams: bool, n_trials: int, seed: int) -> List[dict]:
    """
    Create a list of candidate configurations for hyperparameter tuning.

    Args:
        params (dict): Base parameters
        param_space (dict): Hyperparameter search space
        tune_hyperparams (bool): Whether to perform hyperparameter tuning
        n_trials (int): Number of hyperparameter trials to generate
        seed (int): Random seed for sampling

    Returns:
        List[dict]: List of candidate configuration dictionaries
    """
    # Get the default configuration (first values from param_space)
    default_cfg = _default_params(params, param_space)

    # If no tuning required, return just the default config
    if not tune_hyperparams:
        return [default_cfg]

    # Set up random number generator with seed
    rng = np.random.default_rng(seed)
    # Initialize candidates with the default config
    candidates: List[dict] = [default_cfg]
    # Sample additional configurations
    for _ in range(max(0, n_trials - 1)):
        # Randomly sample one configuration and add to candidates
        candidates.append(_sample_params(params, param_space, rng))
    return candidates


def _train_one_fold(spec: ModelSpec, model_cfg: dict, X_tr: np.ndarray, y_tr: np.ndarray, X_va: Optional[np.ndarray] = None, y_va: Optional[np.ndarray] = None) -> Tuple[Any, np.ndarray]:
    """
    Train one model fold and return the trained model and validation predictions.
    
    Args:
        spec (ModelSpec): Model specification containing Config and Model classes
        model_cfg (dict): Model configuration parameters
        X_tr (np.ndarray): Training features
        y_tr (np.ndarray): Training labels
        X_va (Optional[np.ndarray]): Validation features (optional: used only for NN early stopping)
        y_va (Optional[np.ndarray]): Validation labels (optional: used only for NN early stopping)
        
    Returns:
        Tuple[Any, np.ndarray]: Trained model instance and validation predictions
    """
    # Initialize model configuration
    cfg_obj = spec.Config(**model_cfg)
    # Then create the model instance with the config
    model = spec.Model(cfg_obj)
    # Fit the model
    try:
        logger.debug("Calling fit with validation data for model %s", type(model).__name__)
        model.fit(X_tr, y_tr, X_va, y_va)
    except TypeError:
        # Fallback for sklearn-like fit(X, y)
        model.fit(X_tr, y_tr)
    except Exception as e:
        logger.error("Error during model fitting: %s", e)
        raise e
    # Generate predictions on validation set
    y_pred = model.predict(X_va)
    # Return the trained model and predictions
    return model, np.asarray(y_pred).reshape(-1)


def _train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Small holdout split used only for final refit (to support early stopping)
    
    Args:
        X (np.ndarray): Full features matrix
        y (np.ndarray): Full target vector
        val_frac (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train, X_val, y_val
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n = X.shape[0]
    if n < 2:
        logger.error("Need at least 2 samples for a train/val split, got %d.", n)
        raise ValueError("Need at least 2 samples for a train/val split.")

    if val_frac <= 0.0 or val_frac >= 1.0:
        logger.error("val_frac must be in (0.0, 1.0), got %f.", val_frac)
        raise ValueError("val_frac must be in (0.0, 1.0).")

    val_n = int(round(n * float(val_frac)))

    # Set seed for reproducibility
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    # Split indices to training and validation
    va_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    # Then return the splits
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]


def run_models(cfg: dict, feature_sets: Dict[str, Dict[str, Optional[np.ndarray]]]) -> Dict[str, ModelResult]:
    """
    Run models as per configuration on the provided feature sets.
    
    Logic:
        - For each enabled model in cfg:
            - Fetch model spec from registry
            - Fetch feature set requested by model
            - Create candidate configurations for hyperparameter tuning
            - Create k-fold splits for cross-validation
            - For each candidate configuration:
                - For each fold:
                    - Train model on training split
                    - Evaluate on validation split
                - Compute mean/std of validation scores across folds
            - Select best candidate configuration based on mean validation score
            - Refit model on full training data with best hyperparameters
            - Evaluate on test data
            - Store results and metrics in ModelResult

    Args:
        cfg (dict): Configuration dictionary
        feature_sets (Dict[str, Dict[str, Optional[np.ndarray]]]): Feature sets for modeling
    
    Returns:
        Dict[str, ModelResult]: Dictionary of model results keyed by model name
    """
    seed = cfg["models"]["seed"]
    enabled = cfg["models"]["enabled"]
    k_folds = cfg["models"]["k_folds"]
    shuffle = cfg["models"]["shuffle"]
    defs = cfg["models"]["definitions"]
    artifacts_dir = cfg["models"].get("artifacts_dir", "artifacts/models")
    save_best = bool(cfg["models"].get("save_best_model", False))
    eval_cfg = cfg["evaluation"]
    base_plots_dir = eval_cfg.get("plots_dir", "reports/figures/models")

    set_global_seed(seed)
    registry = init_registry()

    results: Dict[str, ModelResult] = {}

    # Sanity checks before starting
    for model_name in enabled:
        if model_name not in registry:
            logger.error("Model '%s' not found in registry.", model_name)
            raise KeyError(f"Model '{model_name}' not found in registry.")
        if model_name not in defs:
            logger.error("Model '%s' not found in cfg['models']['definitions'].", model_name)
            raise KeyError(f"Model '{model_name}' not found in cfg['models']['definitions'].")

        # Fetch model spec and definition from registry and config
        spec = registry[model_name]
        model_def = defs[model_name]

        logger.debug("Starting training for model: %s", model_name)

        # Fetch feature set requested by this model
        feature_set_name = model_def.get("feature_set", "tree")
        # Sanity check: feature set must be provided
        if feature_set_name not in feature_sets:
            logger.error("Feature set '%s' requested by model '%s' not provided. Available: %s", feature_set_name, model_name, list(feature_sets.keys()))
            raise KeyError(
                f"feature_set '{feature_set_name}' requested by model '{model_name}' not provided. "
                f"Available: {list(feature_sets.keys())}"
            )

        # Fetch training and test data for this specific feature set
        X_train = np.asarray(feature_sets[feature_set_name]["X_train"])
        y_train = np.asarray(feature_sets[feature_set_name]["y_train"]).reshape(-1)
        X_test = np.asarray(feature_sets[feature_set_name]["X_test"])
        y_test = np.asarray(feature_sets[feature_set_name]["y_test"]).reshape(-1)

        logger.debug("Loaded data for model '%s' with feature set '%s': X_train=%s, y_train=%s, X_test=%s, y_test=%s",
                     model_name, feature_set_name, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        # Fetch non-tunable params
        params = model_def["params"]
        # Fetch hyperparam search space and if tuning is enabled (default is no tuning)
        param_space = model_def.get("param_space", {})
        tune_hyperparams = model_def.get("tune_hyperparams", False)
        # Number of hyperparam trials (default is 1)
        n_trials = model_def.get("n_trials", 1)

        # Fetch the metric and direction for optimization
        metric = spec.metric
        direction = spec.direction
        # Sanity check: metric must be known
        if metric not in METRICS:
            logger.error("Metric '%s' not in METRICS. Available: %s", metric, list(METRICS))
            raise ValueError(f"Metric '{metric}' not in METRICS. Available: {list(METRICS)}")

        # Fetch list of metrics to compute during evaluation
        metric_names = eval_cfg.get("metrics", ["rmse", "mae"])
        # Make sure our optimization metric is included
        metric_names_norm = [str(m).lower().strip() for m in metric_names]
        if metric not in metric_names_norm:
            metric_names = list(metric_names) + [metric]

        # Create candidate configurations for hyperparameter tuning (if tuning is disabled, this is just the default config)
        candidates = _make_candidates(params, param_space, tune_hyperparams, n_trials, seed)
        # Create k-fold splits indices for cross-validation
        folds = k_fold_indices(n_samples=X_train.shape[0], k=k_folds, seed=seed, shuffle=shuffle)

        logger.debug(
            "Model=%s | feature_set=%s | candidates=%d | k_folds=%d | objective=%s (%s)",
            model_name, feature_set_name, len(candidates), k_folds, metric, direction
        )

        # For plots/debug: keep best-fold predictions for the selected candidate
        best = {
            "cfg": None,
            "model": None,
            "mean": float("inf") if direction == "min" else -float("inf"),
            "std": float("inf"),
            "fold_scores": [],
            "best_fold": {
                "y_true": None,
                "y_pred": None,
                "info": {},
                "score": float("inf") if direction == "min" else -float("inf"),
            },
        }

        logger.debug("Beginning hyperparameter tuning for model '%s'...", model_name)
        # Hyperparameter tuning tryouts
        for candidate_idx, cand_cfg in enumerate(candidates):
            cand = {
                "fold_scores": [],
                "best_fold": {
                    "model": None,
                    "score": float("inf") if direction == "min" else -float("inf"),
                    "y_true": None,
                    "y_pred": None,
                    "info": {},
                },
            }
            # Cross-validation folds
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                # From the respective feature set, get train/val splits
                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_va, y_va = X_train[va_idx], y_train[va_idx]

                # Then, train one fold with the given candidate config (now we dont need the model instance)
                model, y_pred = _train_one_fold(spec, cand_cfg, X_tr, y_tr, X_va, y_va)
                # Compute metrics on validation set
                m = compute_metrics(metric_names, y_va, y_pred)
                # Metric is what we optimize for
                score = float(m[metric])
                cand["fold_scores"].append(score)

                logger.debug("Model=%s | trial=%d | fold=%d | %s=%.6f", model_name, candidate_idx, fold_idx, metric, score)

                # Keep best fold predictions for this candidate
                if _is_better(score, cand["best_fold"]["score"], direction):
                    cand["best_fold"]["model"] = model
                    cand["best_fold"]["score"] = score
                    cand["best_fold"]["y_true"] = np.asarray(y_va).reshape(-1)
                    cand["best_fold"]["y_pred"] = np.asarray(y_pred).reshape(-1)
                    # Store info about which trial/fold produced this best score
                    cand["best_fold"]["info"] = {"trial": candidate_idx, "fold": fold_idx, "objective": metric, "score": score}

            # After all folds, compute mean/std for this candidate
            mean_score = float(np.mean(cand["fold_scores"]))
            std_score = float(np.std(cand["fold_scores"], ddof=0))
            # Log the candidate summary
            logger.debug("Model=%s | trial=%d | %s_mean=%.6f | %s_std=%.6f", model_name, candidate_idx, metric, mean_score, metric, std_score)

            # Keep best candidate across all tried ones
            if _is_better(mean_score, best["mean"], direction):
                best["model"] = cand["best_fold"]["model"]
                best["cfg"] = cand_cfg
                best["mean"] = mean_score
                best["std"] = std_score
                best["fold_scores"] = cand["fold_scores"]
                best["best_fold"] = cand["best_fold"]

        # Sanity check: we must have at least 1 best candidate with predictions
        if best["cfg"] is None or best["best_fold"]["y_true"] is None or best["best_fold"]["y_pred"] is None:
            logger.error("No valid candidate produced predictions for model '%s'.", model_name)
            raise RuntimeError(f"No valid candidate produced predictions for model '{model_name}'.")


        # --- Stage A: CV plots (best fold of best candidate) ---
        cv_result = ModelResult(
            model_name=model_name,
            y_true=best["best_fold"]["y_true"],
            y_pred=best["best_fold"]["y_pred"],
            # Pass history for plotting if available for model
            history=best["model"].history if best["model"] and hasattr(best["model"], "history") else {},
            params={},
        )
        # Update model result with evaluation metrics and plots
        cv_result = evaluate_and_plot(
            cv_result,
            eval_cfg,
            # To avoid overwriting test plots, include "cv" in the path
            plots_dir=os.path.join(base_plots_dir, model_name, "cv")
        )
        # Additional CV metrics
        cv_result.metrics[f"cv_mean_{metric}"] = best["mean"]
        cv_result.metrics[f"cv_std_{metric}"] = best["std"]
        cv_result.metrics["cv_folds"] = k_folds

        logger.debug(
            "Completed CV for model=%s | feature_set=%s | cv_mean_%s=%.6f",
            model_name, feature_set_name, metric, cv_result.metrics[f"cv_mean_{metric}"]
        )

        # --- Stage B: refit best hyperparams on full train, evaluate on test ---
        cfg_obj = spec.Config(**best["cfg"])
        final_model = spec.Model(cfg_obj)

        # Fit final model on full training data
        final_model.fit(X_train, y_train)

        # Generate predictions on test set
        y_test_pred = np.asarray(final_model.predict(X_test)).reshape(-1)

        # Create test result object
        test_result = ModelResult(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_test_pred,
            history=getattr(final_model, "history", {}),
            params={
                "feature_set": feature_set_name,
                "stage": "cv",
            },
        )
        # Update test result with evaluation metrics and plots
        test_result = evaluate_and_plot(
            test_result,
            eval_cfg,
            plots_dir=os.path.join(base_plots_dir, model_name, "test")
        )

        # Attach CV summary to the final test result
        test_result.metrics[f"cv_mean_{metric}"] = best["mean"]
        test_result.metrics[f"cv_std_{metric}"] = best["std"]
        test_result.metrics["cv_folds"] = k_folds

        # Every detail about the training process
        test_result.params = {
            "feature_set": feature_set_name,
            "best_hyperparams": best["cfg"],
            "objective": metric,
            "direction": direction,
            "cv": {
                "mean": best["mean"],
                "std": best["std"],
                "fold_scores": best["fold_scores"],
                "best_fold": best["best_fold"]["info"],
            },
            "stage": "test",
        }

        if save_best:
            os.makedirs(artifacts_dir, exist_ok=True)
            if hasattr(final_model, "save"):
                path = os.path.join(artifacts_dir, f"{model_name}_final.npz")
                test_result.model_artifact_path = final_model.save(path)
            else:
                logger.warning("Model=%s | save_best_model=True but model has no save() method.", model_name)

        results[model_name] = test_result
        logger.info(
            "Finalized model=%s | feature_set=%s | test_%s=%.6f",
            model_name, feature_set_name, metric, test_result.metrics[metric]
        )

    return results