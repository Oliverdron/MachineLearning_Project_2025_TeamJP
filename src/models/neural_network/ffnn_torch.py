import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
@dataclass
class FFNNTorchConfig:
    # Architecture
    hidden_layers: List[int]
    activations: List[str]                    # one per hidden layer
    dropout: float = 0.0
    output_activation: str = "linear"          # "linear" recommended for regression

    # Training objective
    loss: str = "mse"                         # "mse", "mae", "huber"

    # Optimizer (keeps same schema style as your numpy config)
    optimizer: Dict[str, Any] = None           # {"name": "adam"/"sgd"/"momentum", "lr":..., "beta":..., "weight_decay":...}

    # Training loop
    batch_size: int = 1024
    epochs: int = 10
    seed: int = 42
    weight_init: str = "he"                   # "he", "xavier", "normal", "default"

    # Regularization / stability
    grad_clip: Optional[float] = 5.0          # None disables clipping

    # Early stopping (only used if X_val/y_val provided)
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-5

    # Device
    device: str = "auto"                      # "auto", "cpu", "cuda"


# -----------------------------
# Helpers
# -----------------------------
def _set_seeds(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("device='cuda' requested but CUDA not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _as_float32(X) -> np.ndarray:
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def _as_float32_target(y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)
    return y


def _get_activation(name: str):
    import torch.nn as nn

    key = name.lower()
    if key in ("relu",):
        return nn.ReLU()
    if key in ("tanh",):
        return nn.Tanh()
    if key in ("sigmoid",):
        return nn.Sigmoid()
    if key in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU()
    if key in ("gelu",):
        return nn.GELU()
    if key in ("elu",):
        return nn.ELU()
    if key in ("linear", "identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def _get_loss(name: str):
    import torch.nn as nn

    key = name.lower()
    if key in ("mse", "mse_loss"):
        return nn.MSELoss()
    if key in ("mae", "l1", "l1_loss"):
        return nn.L1Loss()
    if key in ("huber", "smooth_l1"):
        return nn.SmoothL1Loss()
    raise ValueError(f"Unknown loss: {name}")


def _build_optimizer(params, opt_cfg: Dict[str, Any]):
    import torch.optim as optim

    if opt_cfg is None:
        # sensible default
        return optim.Adam(params, lr=1e-3, weight_decay=0.0)

    name = str(opt_cfg.get("name", "adam")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 0.0))
    beta = opt_cfg.get("beta", 0.9)

    if name in ("adam", "adamw"):
        cls = optim.AdamW if name == "adamw" else optim.Adam
        return cls(params, lr=lr, weight_decay=wd)

    if name in ("sgd",):
        return optim.SGD(params, lr=lr, momentum=0.0, weight_decay=wd)

    if name in ("momentum", "sgd_momentum"):
        return optim.SGD(params, lr=lr, momentum=float(beta), weight_decay=wd)

    raise ValueError(f"Unknown optimizer name: {name}")


def _init_weights(module, scheme: str) -> None:
    import torch.nn as nn
    import torch.nn.init as init

    key = scheme.lower()
    if isinstance(module, nn.Linear):
        if key in ("he", "kaiming"):
            init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif key in ("xavier", "glorot"):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif key in ("normal",):
            init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif key in ("default",):
            # leave torch defaults
            pass
        else:
            raise ValueError(f"Unknown weight_init scheme: {scheme}")


# -----------------------------
# Model wrapper (runner-compatible)
# -----------------------------
class FFNNTorchRegressor:
    """
    Runner expects:
      - __init__(cfg_obj)
      - fit(X_train, y_train, X_val=None, y_val=None)
      - predict(X)
      - optional save(path)
      - optional history dict
    """

    def __init__(self, cfg: FFNNTorchConfig):
        self.cfg = cfg
        self.input_dim: Optional[int] = None
        self.device = _resolve_device(cfg.device)

        self.model = None
        self._optimizer = None
        self._loss_fn = _get_loss(cfg.loss)

        # history keys aligned with your plotting needs
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}

    def _build_model(self, input_dim: int):
        import torch.nn as nn

        if len(self.cfg.hidden_layers) != len(self.cfg.activations):
            raise ValueError(
                f"hidden_layers length ({len(self.cfg.hidden_layers)}) must match "
                f"activations length ({len(self.cfg.activations)})"
            )

        layers: List[nn.Module] = []
        prev = input_dim

        for h, act_name in zip(self.cfg.hidden_layers, self.cfg.activations):
            layers.append(nn.Linear(prev, int(h)))
            layers.append(_get_activation(act_name))
            if self.cfg.dropout and self.cfg.dropout > 0:
                layers.append(nn.Dropout(float(self.cfg.dropout)))
            prev = int(h)

        layers.append(nn.Linear(prev, 1))
        layers.append(_get_activation(self.cfg.output_activation))

        net = nn.Sequential(*layers)

        # init weights if requested
        if self.cfg.weight_init and self.cfg.weight_init.lower() != "default":
            net.apply(lambda m: _init_weights(m, self.cfg.weight_init))

        return net

    def _maybe_init(self, X_train: np.ndarray) -> None:
        inferred = int(X_train.shape[1])
        if self.model is None or self.input_dim != inferred:
            self.input_dim = inferred
            self.model = self._build_model(inferred).to(self.device)
            self._optimizer = _build_optimizer(self.model.parameters(), self.cfg.optimizer)
            logger.info("Initialized FFNN(Torch) with input_dim=%d on device=%s", inferred, self.device)

    def fit(
        self,
        X_train,
        y_train,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "FFNNTorchRegressor":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        _set_seeds(int(self.cfg.seed))

        X_train = _as_float32(X_train)
        y_train = _as_float32_target(y_train)

        self._maybe_init(X_train)

        # Datasets/loaders
        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            drop_last=False,
        )

        # Validation tensors (optional)
        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val = _as_float32(X_val)
            y_val = _as_float32_target(y_val)
            Xv = torch.from_numpy(X_val).to(self.device)
            yv = torch.from_numpy(y_val).unsqueeze(1).to(self.device)
        else:
            Xv = yv = None

        # Reset history each fit (optional; change if you want cumulative)
        self.history = {"loss": [], "val_loss": []}

        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, int(self.cfg.epochs) + 1):
            self.model.train()
            batch_losses: List[float] = []

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self._optimizer.zero_grad(set_to_none=True)
                pred = self.model(xb)
                loss = self._loss_fn(pred, yb)
                loss.backward()

                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))

                self._optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

            # Validation
            if use_val:
                self.model.eval()
                with torch.no_grad():
                    vpred = self.model(Xv)
                    vloss = float(self._loss_fn(vpred, yv).detach().cpu().item())
            else:
                vloss = float("nan")

            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(vloss)

            logger.info(
                "FFNN(Torch) epoch=%d/%d loss=%.6f val_loss=%.6f",
                epoch, int(self.cfg.epochs), train_loss, vloss
            )

            # Early stopping (only meaningful with validation)
            if use_val and self.cfg.early_stopping:
                improved = (best_val - vloss) > float(self.cfg.min_delta)
                if improved:
                    best_val = vloss
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(self.cfg.patience):
                        logger.info("Early stopping triggered (best val_loss=%.6f).", best_val)
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("Model is not initialized. Call fit() before predict().")

        X = _as_float32(X)
        if int(X.shape[1]) != int(self.input_dim):
            raise ValueError(f"X has {X.shape[1]} columns but model expects {self.input_dim}.")

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.from_numpy(X).to(self.device)).detach().cpu().numpy().reshape(-1)
        return preds

    def save(self, path: str) -> str:
        """
        Runner might pass a *.npz path. We still save torch-native.
        """
        import torch
        import os

        # If runner gives "something.npz", we rewrite to "something.pt"
        root, ext = os.path.splitext(path)
        if ext.lower() == ".npz":
            path = root + ".pt"

        payload = {
            "state_dict": self.model.state_dict() if self.model is not None else None,
            "cfg": asdict(self.cfg),
            "input_dim": self.input_dim,
        }
        torch.save(payload, path)
        logger.info("Saved FFNN(Torch) to %s", path)
        return path