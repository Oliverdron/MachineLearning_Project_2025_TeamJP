import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models.neural_network.activations import Activation, get_activation
from src.models.neural_network.losses import Loss, get_loss
from src.models.neural_network.optimizers import build_optimizer

logger = logging.getLogger(__name__)


# Weight initialization to make sure signals don't blow up or vanish as they pass through layers
# Simple variance sketch: z_i = sum_i x_i * w_i
# Var(z) = Var(x) * Var(w) * n  (n = number of inputs)
# To keep Var(z) = Var(x), we want Var(w) = 1/n
def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """
    He initialization for weight matrices.

    Args:
        fan_in (int): Number of input units
        fan_out (int): Number of output units
        rng (np.random.Generator): Numpy random number generator (for seed control)

    Returns:
        np.ndarray: Initialized weight matrix
    """
    # standard deviation = sqrt of variance
    # 2 because with RELU about half of inputs are zeroed out, which shrinks variance by 2
    std = np.sqrt(2.0 / fan_in)
    # then sample from a normal distribution with this std and shape (fan_in, fan_out)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """
    Xavier initialization for weight matrices.

    Args:
        fan_in (int): Number of input units
        fan_out (int): Number of output units
        rng (np.random.Generator): Numpy random number generator (for seed control)

    Returns:
        np.ndarray: Initialized weight matrix
    """
    # standard deviation = sqrt of variance
    # 1 because with symmetric activations (e.g. tanh) the variance is preserved on average
    std = np.sqrt(1.0 / fan_in)
    # then sample from a normal distribution with this std and shape (fan_in, fan_out)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


@dataclass
class FFNNNumpyConfig:
    """
    Feed-forward neural network configuration.

    Args:
        hidden_layers (List[int]): List with number of units in each hidden layer
        activations (List[str]): List of activation function names for each hidden layer
        output_activation (str): Activation function name for output layer
        loss (str): Loss function name
        optimizer (Dict): Optimizer configuration dictionary
        batch_size (int): Mini-batch size
        epochs (int): Number of training epochs
        seed (int): Random seed for reproducibility
        weight_init (str): Weight initialization method ("he" or "xavier")
        early_stopping (bool): Whether to use early stopping
        patience (int): Patience for early stopping
        min_delta (float): Minimum change to qualify as improvement for early stopping
        grad_clip (Optional[float]): Gradient clipping threshold (None to disable)
    """
    hidden_layers: List[int]
    activations: List[str]
    output_activation: str
    loss: str

    optimizer: Dict  # {"name": "momentum", "lr": 1e-3, "beta": 0.9, "weight_decay": 0.0}
    batch_size: int = 1024
    epochs: int = 50
    seed: int = 42
    weight_init: str = "he"  # "he" or "xavier"

    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-5
    grad_clip: Optional[float] = None  # e.g. 5.0


class FFNNNumpyRegressor:
    """
    Feed-Forward Neural Network Regressor implementation from scratch using NumPy.

    Args:
        cfg (FFNNNumpyConfig): Configuration for the FFNN model

    Methods:
        __init__: Initialize the FFNN model with given configuration
        _init_params: Initialize weights and biases
        _forward: Perform forward pass through the network
        _backward: Perform backward pass to compute gradients
        fit: Train the model on training data
        predict: Make predictions on new data
        save: Save the model parameters to a file
        train_and_return_result: Train the model and return a ModelResult object

    """
    def __init__(self, cfg: FFNNNumpyConfig):
        self.cfg = cfg

        # Validate hidden layers match number of activations
        if len(cfg.hidden_layers) != len(cfg.activations):
            logger.error("Length of hidden_layers (%d) does not match length of activations (%d)",
                         len(cfg.hidden_layers), len(cfg.activations))
            raise ValueError("hidden_layers and activations must have same length")
        
        # Should be set before training
        self.input_dim = None
        # Set random seed
        self.rng = np.random.default_rng(cfg.seed)

        # Fetch activations for hidden layers
        self.hidden_acts: List[Activation] = [get_activation(a) for a in cfg.activations]
        # Fetch activation for output layer (none = linear)
        self.out_act: Activation = get_activation(cfg.output_activation)
        # Fetch loss function
        self.loss: Loss = get_loss(cfg.loss)
        # Build optimizer
        self.optimizer = build_optimizer(cfg.optimizer)

        # Declare weights and biases as empty lists then fill
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # Training history for evaluation and plotting
        self.history: Dict[str, list] = {"loss": [], "val_loss": []}

    def _init_params(self) -> None:
        # Layer sizes are input + hidden + output
        layer_sizes = [self.input_dim] + list(self.cfg.hidden_layers) + [1]
        # Fetch weight initialization method
        init = self.cfg.weight_init.lower().strip()

        self.weights, self.biases = [], []
        # Initialize weights and biases for each layer
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if init == "he":
                logger.debug("Using He initialization for layer %d -> %d", fan_in, fan_out)
                W = _he_init(fan_in, fan_out, self.rng)
            elif init == "xavier":
                logger.debug("Using Xavier initialization for layer %d -> %d", fan_in, fan_out)
                W = _xavier_init(fan_in, fan_out, self.rng)
            else:
                logger.error(f"Unknown weight_init: {self.cfg.weight_init}")
                raise ValueError(f"Unknown weight_init: {self.cfg.weight_init}")
            # Biases can be initialized to zero
            b = np.zeros((fan_out,), dtype=float)
            # Then store them
            self.weights.append(W.astype(float))
            self.biases.append(b)

        logger.debug("Initialized FFNN parameters: %s", [w.shape for w in self.weights])

    @staticmethod
    def _ensure_y(y: np.ndarray) -> np.ndarray:
        """
        Ensure that y is a 2D numpy array with shape (n_samples, 1).
        """
        y = np.asarray(y)
        # If its a row vector, convert to column vector
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        # Ensure float type
        return y.astype(float)

    def _forward(self, X: np.ndarray, cache: bool) -> Tuple[np.ndarray, Optional[Dict[str, List[np.ndarray]]]]:
        """
        Forward pass through the network.

        Args:
            X (np.ndarray): Input data
            cache (bool): Whether to cache intermediate values for backpropagation

        Returns:
            Tuple[np.ndarray, Optional[Dict[str, List[np.ndarray]]]]: Predicted outputs and caches if requested
        """
        # Input layer
        A = X.astype(float)
        # Save layer activations and pre-activations if caching is enabled
        caches = {"A": [A], "Z": []} if cache else None

        # Iterate through hidden layers with id and activation function
        for i, act in enumerate(self.hidden_acts):
            # Compute pre-activation based on formula
            Z = A @ self.weights[i] + self.biases[i]
            # Then activation
            A = act.fn(Z)
            # Save caches if enabled
            if cache:
                caches["Z"].append(Z)
                caches["A"].append(A)

        # Output layer is calculated based on last hidden layer
        L = len(self.weights) - 1
        # Compute pre-activation
        ZL = A @ self.weights[L] + self.biases[L]
        # Apply output activation function
        yhat = self.out_act.fn(ZL)
        # Then lastly, cache output layer if enabled
        if cache:
            caches["Z"].append(ZL)
            caches["A"].append(yhat)

        # Return predictions and caches if requested
        return yhat, caches

    def _backward(self, caches: Dict[str, List[np.ndarray]], y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass through the network.

        Args:
            caches (Dict[str, List[np.ndarray]]): Cached intermediate values from forward pass
            y (np.ndarray): True labels

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray]]: Gradients for weights and biases
        """
        # caches["A"]: [A0, A1, ..., AL] - (AL = yhat)
        # caches["Z"]: [Z1, Z2, ..., ZL] - (ZL = pre-activation of output layer)
        # Fetch the list from caches
        A_list = caches["A"]
        Z_list = caches["Z"]
        # Prediction is last activation
        yhat = A_list[-1]

        # Compute gradient of the loss w.r.t. predictions
        dYhat = self.loss.d_yhat(yhat, y)

        # Compute gradient at output layer (before activation), so we need to undo output activation
        dZ = dYhat * self.out_act.dfn(Z_list[-1])

        # Initialize gradient lists for weights and biases (same size)
        dW: List[np.ndarray] = [None] * len(self.weights)
        db: List[np.ndarray] = [None] * len(self.biases)

        # Fetch activation of last hidden layer
        A_prev = A_list[-2]
        # Weight gradient for output layer depends on last hidden layer activations and dZ
        dW[-1] = A_prev.T @ dZ
        # Bias gradient is sum over samples, because bias is added to each sample
        db[-1] = np.sum(dZ, axis=0)

        # Backpropagate through hidden layers in reverse order
        for layer in reversed(range(len(self.hidden_acts))):
            # The previous activation gradient is computed by propagating dZ through respective weights (their contribution)
            dA_prev = dZ @ self.weights[layer + 1].T
            # Then we need to undo the activation function at this layer
            dZ = dA_prev * self.hidden_acts[layer].dfn(Z_list[layer])

            # Compute gradients for weights and biases at this layer
            A_prev = A_list[layer]
            # Weight gradient for hidden layer depends on previous layer activations and dZ
            dW[layer] = A_prev.T @ dZ
            # Bias gradient is sum over samples, because bias is added to each sample
            db[layer] = np.sum(dZ, axis=0)

        # Apply gradient clipping if enabled
        if self.cfg.grad_clip is not None:
            # Fetch clip value
            clip = float(self.cfg.grad_clip)
            # Clip each gradient vector in place
            for i in range(len(dW)):
                np.clip(dW[i], -clip, clip, out=dW[i])
                np.clip(db[i], -clip, clip, out=db[i])

        # Return the gradients
        return dW, db

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "FFNNNumpyRegressor":
        """
        Train the model on the provided training data.

        Args:
            X_train (np.ndarray): Training input data
            y_train (np.ndarray): Training true labels
            X_val (Optional[np.ndarray]): Validation input data
            y_val (Optional[np.ndarray]): Validation true labels

        Returns:
            FFNNNumpyRegressor: Trained model instance
        """
        # Before training, set input dimension and initialize parameters
        self.input_dim = X_train.shape[1]
        self._init_params()
        
        # Sanitize y inputs to be 2D arrays
        y_train = self._ensure_y(y_train)
        # Do the same for validation data if provided
        if X_val is not None and y_val is not None:
            y_val = self._ensure_y(y_val)

        # Number of training samples
        n = X_train.shape[0]
        # Batch size
        bs = int(self.cfg.batch_size)
        
        # Early stopping variables
        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        # Training loop with epochs defined in config
        for epoch in range(1, int(self.cfg.epochs) + 1):
            # Shuffle training data at the start of each epoch (seeding for reproducibility)
            idx = self.rng.permutation(n)
            # Get shuffled data
            Xs = X_train[idx]
            ys = y_train[idx]

            # Losses history for this epoch for monitoring
            losses = []
            # Mini-batch training
            for start in range(0, n, bs):
                # Get end of the batch
                end = min(start + bs, n)
                # Then fetch the mini-batch
                Xb = Xs[start:end]
                yb = ys[start:end]

                # Do forward pass with caching
                yhat, caches = self._forward(Xb, cache=True)
                # Compute and save loss for monitoring
                loss = self.loss.fn(yhat, yb)
                losses.append(loss)

                # Then do backward pass to compute gradients
                dW, db = self._backward(caches, yb)
                # Update weights and biases using optimizer
                self.optimizer.update(self.weights, self.biases, dW, db)

            # Compute average training loss for this epoch
            train_loss = float(np.mean(losses))

            # Compute validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                # Forward pass on validation data without caching
                yhat_val, _ = self._forward(X_val, cache=False)
                # Compute validation loss
                val_loss = self.loss.fn(yhat_val, y_val)
            else:
                val_loss = float("nan")

            # Save losses to history
            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            logger.info("FFNN(Numpy) epoch=%d loss=%.6f val_loss=%.6f", epoch, train_loss, val_loss)

            # Early stopping check
            if self.cfg.early_stopping and X_val is not None and y_val is not None:
                # Check if validation loss improved enough based on min_delta
                improved = (best_val - val_loss) > float(self.cfg.min_delta)
                if improved:
                    # Update best validation loss and model state
                    best_val = val_loss
                    best_state = ([w.copy() for w in self.weights], [b.copy() for b in self.biases])
                    bad_epochs = 0
                # If not improved, increase bad epoch count
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(self.cfg.patience):
                        logger.info("Early stopping triggered at epoch=%d (best val_loss=%.6f)", epoch, best_val)
                        break
        
        # After training, if early stopping was used, restore best model state
        if best_state is not None:
            self.weights, self.biases = best_state

        # Return the trained model instance
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for the given input data.

        Logic is just a forward pass through the network without caching.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Predicted values
        """
        yhat, _ = self._forward(X, cache=False)
        return self._ensure_y(yhat)

    def save(self, path: str) -> str:
        """
        Save the model parameters to a compressed .npz file

        Args:
            path (str): Path to save the model

        Returns:
            str: Path where the model was saved
        """
        np.savez_compressed(
            path,
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object),
            cfg=self.cfg.__dict__,
        )
        logger.info(f"Model saved to {path}")
        return path
