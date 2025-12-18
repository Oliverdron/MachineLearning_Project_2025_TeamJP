import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Optimizer:
    """
    Parent class for optimizers.
    Args:
        weights (List[np.ndarray]): List of weight matrices
        biases (List[np.ndarray]): List of bias vectors
        dW (List[np.ndarray]): List of gradients for weights
        db (List[np.ndarray]): List of gradients for biases

    Make sure to implement the update method in subclasses.
    """
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray],
               dW: List[np.ndarray], db: List[np.ndarray]) -> None:
        raise NotImplementedError


@dataclass
class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Args:
        lr (float): Learning rate
        weight_decay (float): L2 regularization factor
    
    Methods:
        update: Update weights and biases using gradients

    Returns:
        None
    """
    lr: float
    weight_decay: float = 0.0  # L2 regularization

    def update(self, weights, biases, dW, db) -> None:
        # Update weights and biases using SGD rule
        for i in range(len(weights)):
            # Apply weight decay if specified
            if self.weight_decay:
                dW[i] = dW[i] + self.weight_decay * weights[i]
            # The new weights and biases are old minus learning rate times gradient
            weights[i] -= self.lr * dW[i]
            biases[i] -= self.lr * db[i]


@dataclass
class Momentum(Optimizer):
    """
    Momentum optimizer.

    Args:
        lr (float): Learning rate
        beta (float): Momentum factor
        weight_decay (float): L2 regularization factor

    Methods:
        update: Update weights and biases using gradients with momentum

    Returns:
        None
    """
    lr: float
    beta: float = 0.9
    weight_decay: float = 0.0 # controls the strength of regularization

    vW: Optional[List[np.ndarray]] = None
    vb: Optional[List[np.ndarray]] = None

    def _init(self, weights, biases) -> None:
        """
        Initialize velocity terms for weights and biases with same shapes on each layer.
        """
        self.vW = [np.zeros_like(w) for w in weights]
        self.vb = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, dW, db) -> None:
        """
        Update weights and biases using Momentum optimization rule.

        Args:
            weights (List[np.ndarray]): List of weight matrices
            biases (List[np.ndarray]): List of bias vectors
            dW (List[np.ndarray]): List of gradients for weights
            db (List[np.ndarray]): List of gradients for biases

        Returns:
            None
        """
        # Initialize velocity terms if not already done
        if self.vW is None or self.vb is None:
            self._init(weights, biases)

        # Update weights and biases using Momentum rule
        for i in range(len(weights)):
            # Apply weight decay if specified
            if self.weight_decay:
                dW[i] = dW[i] + self.weight_decay * weights[i]

            # Update velocity terms
            self.vW[i] = self.beta * self.vW[i] + dW[i]
            self.vb[i] = self.beta * self.vb[i] + db[i]

            # Then update weights and biases
            weights[i] -= self.lr * self.vW[i]
            biases[i] -= self.lr * self.vb[i]


def build_optimizer(cfg: dict) -> Optimizer:
    """
        Build an optimizer instance based on the configuration dictionary.
    """
    # Fetch common parameters (defaults if missing)
    name = str(cfg.get("name", "sgd")).lower().strip()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    # Select optimizer type
    if name == "sgd":
        logger.info(f"Building SGD optimizer with lr={lr}, weight_decay={wd}")
        return SGD(lr=lr, weight_decay=wd)
    if name in ("momentum", "sgd_momentum"):
        beta = float(cfg.get("beta", 0.9))
        logger.info(f"Building Momentum optimizer with lr={lr}, beta={beta}, weight_decay={wd}")
        return Momentum(lr=lr, beta=beta, weight_decay=wd)
    
    # Unknown optimizer
    logger.error(f"Unknown optimizer: {name}")
    raise ValueError(f"Unknown optimizer: {name}")
