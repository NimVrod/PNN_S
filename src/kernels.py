import numpy as np


class PnnKernels:
    @staticmethod
    def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
        """
        Gaussian kernel function to compute similarity between two vectors.
        :param x: Vector representing the input pattern.
        :param y: Vector representing the output pattern.
        :param sigma: The sigma parameter for the kernel function.
        :return: The similarity score between the two vectors.
        """
        diff = x - y
        sq_dist = np.sum(diff ** 2, axis=1)
        return np.exp(-sq_dist / (2 * sigma ** 2))

    @staticmethod
    def laplacian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        """
        Laplacian kernel: exp(-||x - y||_1 / sigma).
        :param x: Input vector of shape (n_features,) or batch (n_samples, n_features).
        :param y: Reference vectors of shape (n_samples, n_features) or a single vector.
        :param sigma: Bandwidth parameter (must be > 0).
        :return: Similarity scores, shape (n_samples,).
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        l1_dist = np.sum(np.abs(x - y), axis=1)
        return np.exp(-l1_dist / sigma)

    @staticmethod
    def cauchy_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        """
        Cauchy kernel: 1 / (1 + ||x - y||^2 / sigma^2).
        :param x: Input vector of shape (n_features,) or batch (n_samples, n_features).
        :param y: Reference vectors of shape (n_samples, n_features) or a single vector.
        :param sigma: Scale parameter (must be > 0).
        :return: Similarity scores, shape (n_samples,).
        """
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        sq_dist = np.sum((x - y) ** 2, axis=1)
        return 1.0 / (1.0 + sq_dist / (sigma ** 2))

    @staticmethod
    def inverse_multiquadric_kernel(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
        """
        Inverse multiquadric kernel: 1 / sqrt(||x - y||^2 + c^2).
        :param x: Input vector of shape (n_features,) or batch (n_samples, n_features).
        :param y: Reference vectors of shape (n_samples, n_features) or a single vector.
        :param c: Positive constant controlling smoothness (must be > 0).
        :return: Similarity scores, shape (n_samples,).
        """
        if c <= 0:
            raise ValueError("c must be > 0")

        sq_dist = np.sum((x - y) ** 2, axis=1)
        return 1.0 / np.sqrt(sq_dist + c ** 2)

    @staticmethod
    def epanechnikov_kernel(x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
        """
        Epanechnikov kernel: max(0, 1 - ||x - y||^2 / h^2).
        :param x: Input vector of shape (n_features,) or batch (n_samples, n_features).
        :param y: Reference vectors of shape (n_samples, n_features) or a single vector.
        :param h: Bandwidth parameter (must be > 0).
        :return: Similarity scores, shape (n_samples,).
        """
        if h <= 0:
            raise ValueError("h must be > 0")

        sq_dist = np.sum((x - y) ** 2, axis=1)
        return np.maximum(0.0, 1.0 - sq_dist / (h ** 2))

    @staticmethod
    def cosine_kernel(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """
        Cosine similarity kernel: (x · y) / (||x|| * ||y|| + eps).
        :param x: Input vector of shape (n_features,) or batch (n_samples, n_features).
        :param y: Reference vectors of shape (n_samples, n_features) or a single vector.
        :param eps: Numerical stability constant.
        :return: Similarity scores, shape (n_samples,).\
        """

        dot = np.sum(x * y, axis=1)
        x_norm = np.linalg.norm(x, axis=1)
        y_norm = np.linalg.norm(y, axis=1)
        return dot / (x_norm * y_norm + eps)