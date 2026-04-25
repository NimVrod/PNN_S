import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from src.kernels import PnnKernels
from src.pnn import PNN


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from a CSV file. The last column is assumed to be the class label.
    :param path: Path to the CSV file containing the data.
    :return: A tuple containing the features (X) and labels (y) as numpy arrays.
    """
    data = np.loadtxt(path, delimiter=",")
    x = data[:, :-1]  # All columns except the last one are features
    y = data[:, -1]  # The last column is the class label
    return x, y


def normalize_data(x: np.ndarray) -> np.ndarray:
    """
    Normalize the features in the dataset to have zero mean and unit variance.
    :param x: The input feature matrix to be normalized.
    :return: The normalized feature matrix.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def main():
    data = load_data("spambase.data")
    x = normalize_data(data[0])
    y = data[1]  # Class labels are not normalized

    silverman = 1.06 * np.std(x) * (len(x) ** (-1 / 5))
    print(f"Silverman's rule of thumb sigma: {silverman:.4f}")
    p = PNN(kernel=PnnKernels.laplacian_kernel, sigma=silverman)
    kerns = [
        PnnKernels.gaussian_kernel,
        PnnKernels.laplacian_kernel,
        PnnKernels.cauchy_kernel,
        PnnKernels.inverse_multiquadric_kernel,
        PnnKernels.epanechnikov_kernel,
        PnnKernels.triangular_kernel,
        PnnKernels.uniform_kernel,
    ]
    silverman_accuracy: dict[str, float] = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=67)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        p.fit(x_train, y_train)
        for kernel in kerns:
            p.kernel = kernel
            y_pred = np.array([p.predict_single(x) for x in x_test])
            accuracy = np.mean(y_pred == y_test)
            silverman_accuracy[kernel.__name__] = accuracy
    print("Silverman's rule of thumb accuracies:")
    for kernel_name, acc in silverman_accuracy.items():
        print(f"Kernel: {kernel_name}, Accuracy: {acc:.4f}")
    # Best sigma and kernel search using K-Fold Cross-Validation
    accuracies = kFold_search(x, y, diff_sigma=0.05, kerns=kerns)

    # Find the best kernel and sigma based on average accuracy in (min, max, avg)
    best_kernel, best_sigma, best_stats = max(
        (
            (k, sigma, stats)
            for k, v in accuracies.items()
            if v
            for sigma, stats in v.items()
        ),
        default=(None, None, (0.0, 0.0, -1.0)),
        key=lambda t: t[2][2],
    )
    if best_kernel is not None:
        best_min, best_max, best_avg = best_stats
        print(
            f"Best configuration: {best_kernel}, sigma {best_sigma:.3f} "
            f"with avg {best_avg:.4f} (min {best_min:.4f}, max {best_max:.4f})"
        )
    else:
        print("No configurations evaluated.")

    # Visualize kernel and sigma experiments: avg line with min/max as error bars
    plt.figure(figsize=(10, 6))
    for kernel_name, accs in accuracies.items():
        if not accs:
            continue

        sigmas = np.array(sorted(accs.keys()))
        stats = np.array([accs[s] for s in sigmas], dtype=float)
        avgs = stats[:, 2]

        plt.plot(sigmas, avgs, label=kernel_name)
        plt.fill_between(sigmas, stats[:, 0], stats[:, 1], alpha=0.2)

    plt.xlabel("Sigma")
    plt.ylabel("Accuracy")
    plt.title("Performance of Different Configurations")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def kFold_search(
    x: np.ndarray,
    y: np.ndarray,
    min_sigma=0.001,
    max_sigma=2,
    diff_sigma=0.005,
    kerns=[
        PnnKernels.gaussian_kernel,
        PnnKernels.laplacian_kernel,
        PnnKernels.cauchy_kernel,
        PnnKernels.inverse_multiquadric_kernel,
        PnnKernels.epanechnikov_kernel,
        PnnKernels.triangular_kernel,
        PnnKernels.uniform_kernel,
    ],
) -> dict[str, dict[float, tuple[np.floating, np.floating, np.floating]]]:
    """
    Perform k-fold cross-validation to search for the best sigma and kernel configuration.
    :param x: The input feature matrix.
    :param y: The target labels.
    :param min_sigma: The minimum value for sigma.
    :param max_sigma: The maximum value for sigma.
    :param diff_sigma: The step size for sigma.
    :return: A dictionary containing the accuracies for each kernel and sigma combination.
    """
    accuracies: dict[
        str, dict[float, tuple[np.floating, np.floating, np.floating]]
    ] = {}  # {kernel_name: {sigma_value: (min_acc, max_acc, avg_acc)}}

    kf = KFold(n_splits=5, shuffle=True, random_state=67)
    p = PNN()
    for kernel in kerns:
        kernel_name = kernel.__name__
        accuracies[kernel_name] = {}
        for sigma in np.arange(min_sigma, max_sigma + diff_sigma, diff_sigma):
            p.kernel = kernel
            p.sigma = sigma
            fold_accuracies = []
            for train_index, test_index in kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                p.fit(x_train, y_train)
                y_pred = np.array([p.predict_single(x) for x in x_test])
                fold_accuracy = np.mean(y_pred == y_test)
                fold_accuracies.append(fold_accuracy)
            avg_accuracy = np.mean(fold_accuracies)
            min_accuracy = np.min(fold_accuracies)
            max_accuracy = np.max(fold_accuracies)
            accuracies[kernel_name][sigma] = (min_accuracy, max_accuracy, avg_accuracy)
            print(
                f"Kernel: {kernel_name}, Sigma: {sigma:.3f}, Accuracy: {avg_accuracy:.4f} (min: {min_accuracy:.4f}, max: {max_accuracy:.4f})"
            )
    return accuracies


if __name__ == "__main__":
    main()
