"""
pnn_cpu.py — Probabilistic Neural Network, CPU backend (NumPy)
"""

import numpy as np
from typing import Callable

# A kernel takes (X: (b,d), patterns: (n_k,d), sigma: float) and returns (b,)
KernelFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


# ===========================================================================
# Built-in kernels
# ===========================================================================

def gaussian_kernel(X: np.ndarray, patterns: np.ndarray, sigma: float) -> np.ndarray:
    """Standard Gaussian (RBF) kernel — smooth, infinite support."""
    diff    = X[:, None, :] - patterns[None, :, :]      # (b, n_k, d)
    sq_dist = np.sum(diff * diff, axis=2)               # (b, n_k)
    d       = X.shape[1]
    norm    = (2.0 * np.pi * sigma ** 2) ** (d / 2.0)
    return np.mean(np.exp(-sq_dist / (2.0 * sigma ** 2)) / norm, axis=1)


def epanechnikov_kernel(X: np.ndarray, patterns: np.ndarray, sigma: float) -> np.ndarray:
    """Epanechnikov kernel — zero outside bandwidth, MSE-optimal."""
    diff    = X[:, None, :] - patterns[None, :, :]      # (b, n_k, d)
    sq_dist = np.sum(diff * diff, axis=2)               # (b, n_k)
    u       = sq_dist / (sigma ** 2)
    inside  = np.where(u <= 1.0, 0.75 * (1.0 - u), 0.0)
    return np.mean(inside, axis=1)


def laplacian_kernel(X: np.ndarray, patterns: np.ndarray, sigma: float) -> np.ndarray:
    """Laplacian kernel — heavier tails than Gaussian, robust to outliers."""
    diff = X[:, None, :] - patterns[None, :, :]         # (b, n_k, d)
    l1   = np.sum(np.abs(diff), axis=2)                 # (b, n_k)
    norm = 2.0 * sigma
    return np.mean(np.exp(-l1 / sigma) / norm, axis=1)


# ===========================================================================
# Neurons
# ===========================================================================

class SummationNeuron:
    """
    One node in the summation layer, representing a single class.

    Stores all training patterns for that class as a contiguous matrix
    and estimates p(x | class) via Parzen window density estimation
    using the provided kernel function.
    """

    def __init__(self, label, sigma: float, kernel: KernelFn):
        self.label  = label
        self.sigma  = sigma
        self.kernel = kernel
        self._patterns: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None   # (n_k, d) — compiled at fit-end

    def add_pattern(self, x: np.ndarray) -> None:
        self._patterns.append(x)

    def compile(self) -> None:
        """Freeze patterns into a single contiguous matrix. Call once after fit."""
        self._matrix = np.stack(self._patterns).astype(np.float32)

    @property
    def n_patterns(self) -> int:
        return len(self._patterns)

    def estimate(self, X: np.ndarray) -> np.ndarray:
        """
        Parzen window density estimate for a batch of query vectors.

            p̂(x | c) = (1 / n_k) Σᵢ K(x, xᵢ)

        Parameters
        ----------
        X : (b, d)

        Returns
        -------
        density : (b,)
        """
        assert self._matrix is not None, "Call compile() before estimate()."
        return self.kernel(X, self._matrix, self.sigma)


# ===========================================================================
# PNN
# ===========================================================================

class PNN:
    """
    Probabilistic Neural Network — NumPy / CPU backend.

    Parameters
    ----------
    sigma : float
        Kernel bandwidth.
    kernel : KernelFn
        Any callable with signature (X, patterns, sigma) -> density.
        Defaults to gaussian_kernel. Use epanechnikov_kernel or
        laplacian_kernel from this module, or supply your own.
    batch_size : int
        Rows of X processed per call to SummationNeuron.estimate().
        Tune to balance memory and throughput.

    Examples
    --------
    >>> pnn = PNN(sigma=0.5)                                  # Gaussian
    >>> pnn = PNN(sigma=0.5, kernel=epanechnikov_kernel)      # Epanechnikov
    >>> pnn = PNN(sigma=0.5, kernel=laplacian_kernel)         # Laplacian
    >>> pnn = PNN(sigma=0.5, kernel=my_kernel)                # custom
    """

    def __init__(
        self,
        sigma: float = 1.0,
        kernel: KernelFn = gaussian_kernel,
        batch_size: int = 512,
    ):
        self.sigma      = sigma
        self.kernel     = kernel
        self.batch_size = batch_size

        self._summation_layer: dict[any, SummationNeuron] = {}
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PNN":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self._summation_layer = {}

        for xi, yi in zip(X, y):
            if yi not in self._summation_layer:
                self._summation_layer[yi] = SummationNeuron(
                    label=yi, sigma=self.sigma, kernel=self.kernel
                )
            self._summation_layer[yi].add_pattern(xi)

        for neuron in self._summation_layer.values():
            neuron.compile()

        self.classes_ = np.array(sorted(self._summation_layer.keys()))
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""
        X = np.asarray(X, dtype=np.float32)
        n         = len(X)
        n_classes = len(self.classes_)
        scores    = np.zeros((n, n_classes), dtype=np.float32)

        for j, cls in enumerate(self.classes_):
            neuron = self._summation_layer[cls]
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                scores[start:end, j] = neuron.estimate(X[start:end])

        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return scores / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        breakdown = {c: self._summation_layer[c].n_patterns for c in self.classes_}
        return (
            f"PNN(sigma={self.sigma}, kernel={self.kernel.__name__}, backend=numpy)\n"
            f"  Summation layer : {len(self.classes_)} neurons  {breakdown}"
        )
