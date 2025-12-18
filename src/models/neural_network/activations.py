import logging
from dataclasses import dataclass
from typing import Callable
import numpy as np

logger = logging.getLogger(__name__)


# Freeze to make instances immutable
@dataclass(frozen=True)
class Activation:
    """
    Represents an activation function and its derivative.
    
    Args:
        name (str): Name of the activation function
        fn (Callable[[np.ndarray], np.ndarray]): Activation function
        dfn (Callable[[np.ndarray], np.ndarray]): Derivative of the activation function
    """
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    dfn: Callable[[np.ndarray], np.ndarray]


def relu() -> Activation:
    """
    Rectified Linear Unit (ReLU) activation function.

    Returns:
        Activation: ReLU activation instance
    """
    return Activation(
        name="relu",
        # The function is zero for negative inputs and linear for positive inputs
        fn=lambda z: np.maximum(0, z),
        # The derivative is 0 for z < 0 and 1 for z > 0
        dfn=lambda z: (z > 0).astype(z.dtype),
    )


def leaky_relu(alpha: float = 0.01) -> Activation:
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.

    Args:
        alpha (float): Slope for negative inputs
    
    Returns:
        Activation: Leaky ReLU activation instance
    """
    # Function definition
    def f(z) -> np.ndarray:
        # For values >= 0, return z; for values < 0, return alpha * z
        return np.where(z >= 0, z, alpha * z)

    # Derivative definition
    def df(z) -> np.ndarray:
        # For values >= 0, derivative is 1; for values < 0, derivative is alpha
        out = np.ones_like(z, dtype=z.dtype)
        out[z < 0] = alpha
        return out

    return Activation(name="leaky_relu", fn=f, dfn=df)


def sigmoid() -> Activation:
    """
    Sigmoid activation function. (stable)

    The classic formula is: 1 / (1 + exp(-z))
    We use that for non-negative z, but if z is large negative, then exp(-z) overflows.
    To handle this, we multiply numerator and denominator by exp(z) and obtain the formula: exp(z) / (1 + exp(z))
    In this case the exp(z) term is rather small (decays to 0), so no overflow occurs.

    Returns:
        Activation: Sigmoid activation instance
    """
    def f(z: np.ndarray) -> np.ndarray:
        # define output array with same shape and float64 for stability
        out = np.empty_like(z, dtype=np.float64)
        # select positive and negative indices
        pos = z >= 0
        neg = ~pos
        # compute the positive part first with the classic formula
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        # e^z for negative part
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        # ensure output is same dtype as input
        return out.astype(z.dtype, copy=False)

    def df(z: np.ndarray) -> np.ndarray:
        """
        Derivative of the sigmoid function.

        Args:
            z (np.ndarray): Input array

        Returns:
            np.ndarray: Derivative values
        """
        return f(z) * (1 - f(z))

    return Activation(name="sigmoid", fn=f, dfn=df)


def tanh() -> Activation:
    """
    Hyperbolic tangent (tanh) activation function.

    The classic formula is: (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    For large z, tanh(z) approaches 1, and for large negative z, it approaches -1.
    Centered around 0.

    Returns:
        Activation: Tanh activation instance
    """
    return Activation(name="tanh", fn=np.tanh, dfn=lambda z: 1 - np.tanh(z) ** 2)


def softplus() -> Activation:
    """
    Softplus activation function. (stable)

    The classic formula is: log(1 + exp(z))
    For large z, exp(z) overflows, so we factor out exp(z) to get:  z + log(1 + exp(-z))
    For small z, we use the classic formula.

    Combined formula: log1p(exp(-|z|)) + max(z, 0)
        + max(z, 0) adds z for large positive z and zero for negative z
        + exp(-|z|) so the exponent is always negative
        + log1p(u) to compute log(1 + u) in a stable way for small u

    Returns:
        Activation: Softplus activation instance
    """
    def f(z: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)

    # fetch sigmoid activation for derivative of softplus
    return Activation(name="softplus", fn=f, dfn=lambda z: sigmoid().fn(z))


def linear() -> Activation:
    """
    Most basic activation: identity function.

    Returns:
        Activation: Linear activation instance
    """
    return Activation(name="linear", fn=lambda z: z, dfn=lambda z: np.ones_like(z, dtype=z.dtype))


_ACTIVATIONS: dict[str, Callable[[], Activation]] = {
    "relu": relu,
    "leaky_relu": leaky_relu, # default alpha=0.01
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softplus": softplus,
    "linear": linear,
    "none": linear,
}

def get_activation(name: str) -> Activation:
    key = name.lower().strip()
    try:
        logger.info(f"Trying to get activation: {key}")
        return _ACTIVATIONS[key]()
    except KeyError as e:
        logger.error(f"Activation {key} not found.")
        raise ValueError(f"Unknown activation: {name}") from e
