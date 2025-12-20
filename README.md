# Insurance ClaimRate Modeling

Config-driven ML pipeline for predicting insurance claim frequency as **ClaimRate**.

**Target**
- `ClaimRate = ClaimNb / Exposure`  
- Default target column name: `ClaimNb`

## What’s in this repo

### Pipeline stages
1. **Data ingestion + cleaning** (`DataDigestion`)
   - Loads `data/raw/claims_train.csv` and `data/raw/claims_test.csv`
   - Validates numeric columns, cleans categoricals, removes duplicates
   - Imputes missing values (numeric + categorical)
   - One-hot encodes categoricals (based on config)
   - Saves data distribution plots to `reports/figures/data`
   - If `target_col == "ClaimRate"`, drops `ClaimNb` to avoid leakage

2. **Optional feature engineering**
   - **PCA** (`PCAManager`) → saves plots to `reports/figures/pca`
   - **Clustering** (`ClusterManager`) → saves plots to `reports/figures/clustering`

3. **Model training + evaluation** (`run_models`)
   - Hyperparameter random search, for each candidate (by default first values selected from `param_space`)
   - K-fold CV (default: 5 folds)
   - Final fit on full training set, evaluate on test set
   - Metrics (default): RMSE, MAE, R²
   - Saves plots to `reports/figures/models/<model_name>/{cv,test}/`
   - Optionally saves best model artifacts to `artifacts/models/`

## Included models (enabled by default)
- `ffnn_numpy` — Feedforward NN (NumPy, from scratch)
- `ffnn_torch` — Feedforward NN (PyTorch)
- `decision_tree_scratch` — Decision Tree (from scratch)
- `decision_tree_sklearn` — Decision Tree (scikit-learn)
- `random_forest_sklearn` — Random Forest (scikit-learn)

## Configuration (single source of truth)
Everything is controlled by **`src/config/settings.json`**, including:
- data paths, dtypes, column groups (numeric/categorical)
- encoding setup (ordinal/one-hot + handling unknowns)
- PCA + clustering parameters and output dirs
- evaluation metrics + plot saving
- model definitions (params, hyperparam search space, CV, saving artifacts)

## Minimal “how to run” (Python)
The core runner expects a `feature_sets` dict keyed by feature-set name (e.g. `"tree"`, `"nn"`):
- Each feature set must provide: `X_train`, `y_train`, `X_test`, `y_test`

Typical flow:
1) load config + setup logging  
2) `train_df, test_df = DataDigestion(cfg).run()`  
3) build feature sets (e.g., raw features for `"tree"`, PCA features for `"nn"`)  
4) `results = run_models(cfg, feature_sets)`

## Output folders (defaults)
- Cleaned data: `data/processed/`
- Figures:
  - data distribution plots: `reports/figures/data/`
  - PCA: `reports/figures/pca/`
  - clustering: `reports/figures/clustering/`
  - model eval: `reports/figures/models/`
- Saved models (optional): `artifacts/models/`

---

## Updating dependencies + environment + running from CLI

### Add the dependency to `pyproject.toml`:

Then run `uv sync` to fetch the dependecies, `uv lock` to save the exact version and source.
