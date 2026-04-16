from typing import Callable, Optional

import numpy as np


class PNN:
    """
    Probabilistic Neural Network (PNN) implementation for classification tasks.
    Train with fit() method and predict with predict() method. Uses a kernel function to compute similarity between input and training patterns.
    """

    def __init__(self, kernel: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None, sigma=0.1) -> None:
        """
        Initialize the PNN model with an optional kernel function and sigma parameter.
        :param kernel: The kernel function to use to compute similarity between input and training patterns. Default is the Gaussian kernel.
        :param sigma: The sigma parameter for the kernel function (default is 0.1).
        """
        self.patterns = None
        self.sigma: float = sigma
        self.kernel = kernel if kernel is not None else self.gaussian_kernel

    @staticmethod
    def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """
        Gaussian kernel function to compute similarity between two vectors.
        :param x: Vector representing the input pattern.
        :param y: Vector representing the output pattern.
        :param sigma: The sigma parameter for the kernel function.
        :return: The similarity score between the two vectors.
        """
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the PNN model to the training data by storing the patterns for each class.
        :param x: Training data features
        :param y: Training data labels corresponding to the features in x.
        :return: None
        """
        self.patterns = {c: x[y == c] for c in np.unique(y)}

    def predict_single(self, x: np.ndarray) -> float:
        """
        Predict the class label for a single input
        :param x: Input vector
        :return: Predicted class label
        """
        if self.patterns is None:
            raise ValueError("Model has not been fitted yet.")
        best_class = None
        best_score = -np.inf
        for c, patterns in self.patterns.items():
            score = sum(self.kernel(x, p, self.sigma) for p in patterns)
            if score > best_score:
                best_score = score
                best_class = c

        return best_class

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for a set of input vectors.
        :param x: Array of input vectors to predict class labels for.
        :return: Array of predicted class labels corresponding to the input vectors.
        """
        return np.array([self.predict_single(xi) for xi in x])


pnn = PNN(kernel=PNN.gaussian_kernel)
