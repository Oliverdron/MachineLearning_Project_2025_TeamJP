import logging
from dataclasses import dataclass
from typing import Callable
import numpy as np

logger = logging.getLogger(__name__)


# Freeze to make instances immutable
@dataclass(frozen=True)
class Loss:
    """
    Represents a loss function and its derivative.
    """
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    d_yhat: Callable[[np.ndarray, np.ndarray], np.ndarray]


def mse() -> Loss:
    """
    Mean Squared Error (MSE) loss function.

    Formula: L = 0.5 * mean((yhat - y)^2) - the 0.5 is for easier derivative

    Returns:
        Loss: MSE loss instance
    """
    def f(yhat, y) -> float:
        """ Compute MSE loss value. """
        # Error per sample
        e = yhat - y
        # Squared error mean
        return float(0.5 * np.mean(e * e))

    def d(yhat, y) -> np.ndarray:
        """
        Compute derivative of MSE loss w.r.t. predictions yhat.
        To have a cleaner derivative, we include the 0.5 factor in the loss definition.
        """
        # Fetch the number of samples
        n = y.shape[0]
        return (yhat - y) / n

    return Loss(name="mse", fn=f, d_yhat=d)


def poisson_nll(eps: float = 1e-10) -> Loss:
    """
    Poisson Negative Log-Likelihood loss function.
    
    Formula per sample: NLL = yhat - y * log(yhat) + log(y!) (ignoring the log(y!) term as it does not depend on yhat)
    Formula for derivative: dNLL/dyhat = 1 - y/yhat

    Returns:
        Loss: Poisson NLL loss instance
    """
    def f(yhat, y) -> float:
        # Clip predictions to avoid log(0)
        yhat_c = np.clip(yhat, eps, None)
        # Use formula and take the mean across samples
        return float(np.mean(yhat_c - y * np.log(yhat_c)))

    def d(yhat, y) -> np.ndarray:
        # Fetch the number of samples
        n = y.shape[0]
        # Clip predictions to avoid division by zero
        yhat_c = np.clip(yhat, eps, None)
        # Use formula for element-wise derivative
        return (1.0 - (y / yhat_c)) / n

    return Loss(name="poisson_nll", fn=f, d_yhat=d)


_LOSSES: dict[str, Callable[[], Loss]] = {
    "mse": mse,
    "poisson_nll": poisson_nll,
}

def get_loss(name: str) -> Loss:
    """
    Build a loss instance based on the name.

    Args:
        name (str): Name of the loss function

    Returns:
        Loss: Corresponding Loss instance
    """
    # Normalize name
    key = name.lower().strip()
    try:
        logger.info(f"Trying to get loss: {key}")
        # Call factory to build Loss instance
        return _LOSSES[key]()
    except KeyError as e:
        logger.error(f"Unknown loss: {name}")
        raise ValueError(f"Unknown loss: {name}") from e
