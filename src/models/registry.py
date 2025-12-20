# src/models/registry.py
from dataclasses import dataclass
from typing import Type

@dataclass(frozen=True)
class ModelSpec:
    Config: Type
    Model: Type
    metric: str # e.g. "poisson_nll" or "rmse"
    direction: str # "min" or "max"

def init_registry() -> dict[str, ModelSpec]:
    from src.models.neural_network.ffnn_numpy import FFNNNumpyRegressor, FFNNNumpyConfig
    from src.models.neural_network.ffnn_torch import FFNNTorchRegressor, FFNNTorchConfig
    from src.models.decision_tree.decisiontree_numpy import DecisionTreeScratchConfig, DecisionTreeRegressorFromScratch
    from src.models.decision_tree.decisiontree_sklearn import DecisionTreeSklearnConfig, SklearnDecisionTreeRegressor
    from src.models.random_forest.randomforest import RandomForestSklearnConfig, SklearnRandomForestRegressor
    return {
        "ffnn_numpy": ModelSpec(
            Config=FFNNNumpyConfig,
            Model=FFNNNumpyRegressor,
            metric="rmse",
            direction="min",
        ),
        "ffnn_torch": ModelSpec(
            Config=FFNNTorchConfig,
            Model=FFNNTorchRegressor,
            metric="rmse",
            direction="min",
        ),
        "decision_tree_scratch": ModelSpec(
            Config=DecisionTreeScratchConfig,
            Model=DecisionTreeRegressorFromScratch,
            metric="rmse",
            direction="min",
        ),
        "decision_tree_sklearn": ModelSpec(
            Config=DecisionTreeSklearnConfig,
            Model=SklearnDecisionTreeRegressor,
            metric="rmse",
            direction="min",
        ),
        "random_forest_sklearn": ModelSpec(
            Config=RandomForestSklearnConfig,
            Model=SklearnRandomForestRegressor,
            metric="rmse",
            direction="min",
        ),
    }
