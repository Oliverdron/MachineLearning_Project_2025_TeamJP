import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config.seed import set_global_seed
from src.models.registry import init_registry
from src.models.results import ModelResult
from src.models.registry import ModelSpec
from src.models.split import k_fold_indices
from src.evaluation.metrics import compute_metrics, METRICS
from src.evaluation.performance import evaluate_and_plot

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


def _train_one_fold(spec: ModelSpec, model_cfg: dict, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray) -> Tuple[ModelResult, Any]:
    """
    Train one fold of a model and return the result.

    Args:
        spec (ModelSpec): Model specification from registry
        model_cfg (dict): Configuration dictionary for the model
        X_tr (np.ndarray): Training features
        y_tr (np.ndarray): Training labels
        X_va (np.ndarray): Validation features
        y_va (np.ndarray): Validation labels

    Returns:
        Tuple[ModelResult, Any]: Trained model result and model object
    """
    # Initialize model configuration and pass it to the model instance
    cfg_obj = spec.Config(**model_cfg)
    model = spec.Model(cfg_obj)

    # Contract: models expose train_and_return_result(X_train, y_train, X_val, y_val) -> ModelResult
    res: ModelResult = model.train_and_return_result(X_tr, y_tr, X_va, y_va)

    return res, model


def run_models(cfg: dict, X: np.ndarray, y: np.ndarray) -> Dict[str, ModelResult]:
    """
    Run modeling pipeline with k-fold cross-validation and hyperparameter tuning.

    Logic:
        - For each enabled model:
            - Generate candidate hyperparameter configurations
            - For each candidate:
                - Perform k-fold CV:
                    - Train on k-1 folds, validate on 1 fold
                    - Collect fold scores
                - Compute mean CV score for candidate
            - Select best candidate based on CV mean score
            - Evaluate and plot using best fold of best candidate

    Important variables:
        - cfg: Configuration dictionary
        - registry: Model registry with specifications
        - folds: Precomputed k-fold indices for CV
        - candidates: List of hyperparameter configurations to try
        - best_candidate_cfg: Best hyperparameter configuration found
        - best_fold_res: Best fold ModelResult for final evaluation
        - best_fold_model: Best fold model object for artifact saving
        - best_fold_info: Metadata about the best fold (trial, fold, score)
        - best_fold_score: Score of the best fold
        - final_res: Final ModelResult after evaluation and plotting
        - results: Dictionary of all model results

    Args:
        cfg (dict): Configuration dictionary
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector

    Returns:
        Dict[str, ModelResult]: Dictionary of model results keyed by model name
    """
    # Ensure inputs are numpy arrays (target is 1D column vector)
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    # Extract common modeling settings from config
    seed = cfg["models"]["seed"]
    enabled = cfg["models"]["enabled"]
    k_folds = cfg["models"]["k_folds"]
    shuffle = cfg["models"]["shuffle"]
    defs = cfg["models"]["definitions"]
    eval_cfg = cfg["evaluation"]

    # Set global random seed for reproducibility
    set_global_seed(seed)
    # Initialize model registry
    registry = init_registry()

    # Initialize results dictionary
    results: Dict[str, ModelResult] = {}

    # Same folds for all candidates (fair comparison)
    folds = k_fold_indices(
        n_samples=X.shape[0],
        k=k_folds,
        seed=seed,
        shuffle=shuffle,
    )

    # Iterate over each enabled model
    for model_name in enabled:
        # Fetch model specification and definition from registry and config
        spec = registry[model_name]
        model_def = defs[model_name]

        # Fetch model specific settings
        params = model_def["params"]
        # Hyperparameter search space (e.g options to try)
        param_space = model_def["param_space"]
        # Hyperparameter tuning needed?
        tune_hyperparams = model_def["tune_hyperparams"]
        # Number of hyperparameter trials
        n_trials = model_def["n_trials"]

        # Generate candidate configurations for hyperparameter tuning (include default)
        candidates = _make_candidates(
            params=params,
            param_space=param_space,
            tune_hyperparams=tune_hyperparams,
            n_trials=n_trials,
            seed=seed,
        )

        # Fetch metric and direction for hyperparameter tuning (e.g. "rmse" and "min" or "max")
        metric = spec.metric
        direction = spec.direction

        # Make sure metric is valid
        if metric not in METRICS:
            logger.error("Metric '%s' not in METRICS. Available: %s", metric, list(METRICS))
            raise ValueError(f"Metric '{metric}' not in METRICS. Available: {list(METRICS)}")

        # Fetch evaluation metric names for given model
        metric_names = eval_cfg.get("metrics", ["rmse", "mae"])
        # Ensure the tuning metric is included in the evaluation metrics
        if metric not in [str(m).lower().strip() for m in metric_names]:
            metric_names = list(metric_names) + [metric]

        logger.info(
            "Model=%s | candidates=%d | k_folds=%d | metric =%s (%s)",
            model_name, len(candidates), k_folds, metric, direction
        )

        # Track best candidate across all trials (for a given model and hyperparams)
        best_candidate_cfg: Optional[dict] = None
        best_candidate_mean = float("inf") if direction == "min" else -float("inf")
        best_candidate_std = float("inf")
        best_candidate_fold_scores: List[float] = []
        best_candidate_fold_metrics: List[dict] = []

        # Store the single best fold model/result for the best candidate (for history + plots)
        best_fold_res: Optional[ModelResult] = None
        best_fold_model: Any = None
        best_fold_info: dict = {}
        best_fold_score = float("inf") if direction == "min" else -float("inf")

        # Iterate over each candidate configuration
        for t, cand_cfg in enumerate(candidates):
            # Initialize tracking for this candidate
            fold_scores: List[float] = []
            fold_metrics: List[dict] = []

            # Initialize tracking for this candidate's best fold
            cand_best_fold_res: Optional[ModelResult] = None
            cand_best_fold_model: Any = None
            cand_best_fold_info: dict = {}
            cand_best_fold_score = float("inf") if direction == "min" else -float("inf")

            # Perform k-fold cross-validation
            for fold_idx, (tr_idx, va_idx) in enumerate(folds):
                # Split the data into training and validation sets
                X_tr, y_tr = X[tr_idx], y[tr_idx]
                X_va, y_va = X[va_idx], y[va_idx]

                # Train one fold and get the result
                res, model_obj = _train_one_fold(
                    spec=spec,
                    model_cfg=cand_cfg,
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_va=X_va,
                    y_va=y_va,
                )

                # Compute metrics for this fold
                m = compute_metrics(metric_names, res.y_true, res.y_pred)
                # Track fold metrics
                fold_metrics.append(m)

                # Extract the score for the tuning metric
                score = float(m[metric])
                # Track fold scores
                fold_scores.append(score)

                # Log fold result
                logger.info(
                    "Model=%s | trial=%d | fold=%d | %s=%.6f",
                    model_name, t, fold_idx, metric, score
                )

                # Check if this is the best fold for this candidate
                if _is_better(score, cand_best_fold_score, direction):
                    # Update the best fold information for this candidate
                    cand_best_fold_score = score
                    cand_best_fold_res = res
                    cand_best_fold_model = model_obj
                    cand_best_fold_info = {
                        "trial": t,
                        "fold": fold_idx,
                        "objective": metric,
                        "score": score,
                    }

            # After k folds, compute mean and std for this candidate
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores, ddof=0))

            # Log candidate result
            logger.info(
                "Model=%s | trial=%d | %s_mean=%.6f | %s_std=%.6f",
                model_name, t, metric, mean_score, metric, std_score
            )

            # Pick best hyperparams by CV mean objective
            if _is_better(mean_score, best_candidate_mean, direction):
                best_candidate_mean = mean_score
                best_candidate_std = std_score
                best_candidate_cfg = cand_cfg
                best_candidate_fold_scores = fold_scores
                best_candidate_fold_metrics = fold_metrics

                # Carry the best fold model/result for plotting later
                best_fold_res = cand_best_fold_res
                best_fold_model = cand_best_fold_model
                best_fold_info = cand_best_fold_info
                best_fold_score = cand_best_fold_score

        # Sanity check
        if best_candidate_cfg is None or best_fold_res is None:
            logger.error("No valid candidate found for model: %s", model_name)
            raise RuntimeError("No candidate produced a valid result.")

        # Save the best-fold model artifact if requested
        if cfg["models"].get("save_best_model", False):
            # Create artifacts directory if it doesn't exist
            artifacts_dir = cfg["models"]["artifacts_dir"]
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Save the model using its own save method if available
            if hasattr(best_fold_model, "save"):
                # Construct the file path
                path = os.path.join(
                    artifacts_dir,
                    f"{model_name}_best_trial{best_fold_info['trial']}_fold{best_fold_info['fold']}.npz",
                )
                best_fold_res.model_artifact_path = best_fold_model.save(path)

        # Plot & evaluate using the stored best-fold ModelResult (returns updated ModelResult)
        final_res = evaluate_and_plot(best_fold_res, eval_cfg)

        # Add CV summary after evaluate_and_plot
        final_res.metrics[f"cv_mean_{metric}"] = best_candidate_mean
        final_res.metrics[f"cv_std_{metric}"] = best_candidate_std
        final_res.metrics["cv_folds"] = float(k_folds)

        # Store CV details + best hyperparams in params (can be non-floats)
        final_res.params = {
            "best_hyperparams": best_candidate_cfg,
            "cv_objective": metric,
            "cv_direction": direction,
            "cv_fold_scores": best_candidate_fold_scores,
            "best_fold_info": best_fold_info,
            "trial_fold_metrics": best_candidate_fold_metrics,
        }
        
        # Store final result for this model
        results[model_name] = final_res

        logger.info(
            "Selected model=%s | best_cv_mean_%s=%.6f | best_fold_%s=%.6f",
            model_name, metric, best_candidate_mean, metric, best_fold_score
        )

    # At the very end, return all model results
    return results
