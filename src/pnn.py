from typing import Callable, Optional
import numpy as np
from src.kernels import PnnKernels

class PNN:
    """
    Probabilistic Neural Network (PNN) implementation for classification tasks.
    Train with fit() method and predict with predict() method. Uses a kernel function to compute similarity between input and training patterns.
    """

    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = PnnKernels.gaussian_kernel, sigma=0.1) -> None:
        """
        Initialize the PNN model with an optional kernel function and sigma parameter.
        :param kernel: The kernel function to use to compute similarity between input and training patterns. Default is the Gaussian kernel.
        :param sigma: The sigma parameter for the kernel function (default is 0.1).
        """
        self.patterns : Optional[dict] = None
        self.sigma: float = sigma
        self.kernel = kernel

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit (Train) the PNN model to the training data by storing the patterns for each class.
        :param x: Training data features
        :param y: Training data labels corresponding to the features in x.
        :return: None
        """
        self.patterns = {c: x[y == c] for c in np.unique(y)}

    def predict_single(self, x: np.ndarray) -> Optional[float]:
        """
        Predict the class label for a single input
        :param x: Input vector
        :return: Predicted class label
        """
        if self.patterns is None:
            raise ValueError("Model has not been fitted yet.")
        best_class : Optional[float] = None
        best_score = -np.inf
        for cls, patterns in self.patterns.items():
            score = np.sum(self.kernel(x, patterns, self.sigma))
            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for a set of input vectors.
        :param x: Array of input vectors to predict class labels for.
        :return: Array of predicted class labels corresponding to the input vectors.
        """
        return np.array([self.predict_single(xi) for xi in x])

    def summation_layer(self, x: np.ndarray) -> dict:
        """
        Compute the summation layer output for a given input vector.
        :param x: Input vector
        :return: A dictionary where keys are class labels and values are the summation layer outputs for each class. {class_label: summation_output}
        """
        if self.patterns is None:
            raise ValueError("Model has not been fitted yet.")
        summation_outputs = {}
        for cls, patterns in self.patterns.items():
            summation_outputs[cls] = np.sum(self.kernel(x, patterns, self.sigma))
        return summation_outputs