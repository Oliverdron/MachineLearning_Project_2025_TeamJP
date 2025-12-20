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
    return {
        "ffnn_numpy": ModelSpec(
            Config=FFNNNumpyConfig,
            Model=FFNNNumpyRegressor,
            metric="rmse",
            direction="min",
        ),
        # Add other models here
        # Should have every model and only the ones enabled in settings.json will be used
    }
