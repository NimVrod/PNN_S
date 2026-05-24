import numpy as np


class PnnKernels:
    @staticmethod
    def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        diff = x - y
        sq_dist = np.sum(diff ** 2, axis=1)
        norm_const = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
        return norm_const * np.exp(-sq_dist / (2 * sigma ** 2))

    @staticmethod
    def laplacian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        l1_dist = np.sum(np.abs(x - y), axis=1)
        norm_const = 1.0 / (2.0 * sigma)
        return norm_const * np.exp(-l1_dist / sigma)

    @staticmethod
    def cauchy_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        sq_dist = np.sum((x - y) ** 2, axis=1)
        norm_const = 1.0 / (np.pi * sigma)
        return norm_const / (1.0 + sq_dist / (sigma ** 2))

    @staticmethod
    def inverse_multiquadric_kernel(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
        if c <= 0:
            raise ValueError("c must be > 0")

        sq_dist = np.sum((x - y) ** 2, axis=1)
        norm_const = 1.0 / c
        return norm_const / np.sqrt(sq_dist + c ** 2)

    @staticmethod
    def epanechnikov_kernel(x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
        if h <= 0:
            raise ValueError("h must be > 0")

        sq_dist = np.sum((x - y) ** 2, axis=1)
        norm_const = 3.0 / (4.0 * h)
        return norm_const * np.maximum(0.0, 1.0 - sq_dist / (h ** 2))
    
    @staticmethod
    def triangular_kernel(x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
        if h <= 0:
            raise ValueError("h must be > 0")

        l2_dist = np.linalg.norm(x - y, axis=1)
        norm_const = 1.0 / h
        return norm_const * np.maximum(0.0, 1.0 - l2_dist / h)
    
    @staticmethod
    def uniform_kernel(x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
        if h <= 0:
            raise ValueError("h must be > 0")

        l2_dist = np.linalg.norm(x - y, axis=1)
        return np.where(l2_dist <= h, 0.5, 0.0)