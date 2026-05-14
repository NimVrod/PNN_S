import matplotlib
import platform
if platform.system() == "Linux":
    matplotlib.use("Agg")

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
    silverman_accuracy: dict[
        str, float
    ] = {}  # {kernel_name: accuracy} for Silverman's rule of thumb evaluation
    kf = KFold(n_splits=5, shuffle=True, random_state=67)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        p.fit(x_train, y_train)
        for kernel in kerns:
            p.kernel = kernel
            y_pred = p.predict(x_test) 
            accuracy = np.mean(y_pred == y_test)
            silverman_accuracy[kernel.__name__] = accuracy
    print("Silverman's rule of thumb accuracies:")
    for kernel_name, acc in silverman_accuracy.items():
        print(f"Kernel: {kernel_name}, Accuracy: {acc:.4f}")
    # Best sigma and kernel search using K-Fold Cross-Validation
    accuracies = kFold_search(x, y, kerns=kerns, max_sigma=1, diff_sigma=0.5)

    # Find best kernel/sigma by mean fold accuracy
    best_kernel, best_sigma, best_stats = max(
        (
            (kernel_name, sigma, stats)
            for kernel_name, accs in accuracies.items()
            for sigma, stats in accs.items()
        ),
        key=lambda item: np.mean(item[2]),
    )

    print(
        f"Best configuration: Kernel: {best_kernel}, "
        f"Sigma: {best_sigma:.3f}, Average Accuracy: {np.mean(best_stats):.4f}"
    )

    # save_results(accuracies, filename="results.csv")

    # Visualize kernel and sigma experiments: avg line with min/max as error bars
    plt.figure(figsize=(10, 6))
    for kernel_name, accs in accuracies.items():
        if not accs:
            continue

        sigmas = np.array(sorted(accs.keys()))
        stats = np.array([accs[s] for s in sigmas], dtype=float)
        avgs = np.mean(stats, axis=1)

        plt.plot(sigmas, avgs, label=kernel_name)
        # plt.fill_between(sigmas, mins, maxs, alpha=0.2)

    plt.xlabel("Sigma")
    plt.ylabel("Średnia dokładność")
    plt.title("Wpływ parametru sigma na dokładność PNN dla różnych funkcji jądra")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("kernel_sigma_comparison.png")
    plt.show()

    # Check if you need normalization in kernels
    kernels = [
        PnnKernels.gaussian_kernel,
        lambda x, y, sigma: np.exp(-np.sum((x - y) ** 2, axis=1) / (2 * sigma ** 2)),  # Gaussian without normalization
    ]

    accs = kFold_search(x, y, kerns=kernels, max_sigma=1, diff_sigma=0.1)
    gauss_acc = accs[kernels[0].__name__]
    no_norm_acc = accs[kernels[1].__name__]
    # Compare dictionaries of sigma -> fold-accuracies safely.
    # First ensure the same set of sigma keys, then compare arrays per key using allclose.
    same_keys = set(gauss_acc.keys()) == set(no_norm_acc.keys())
    if not same_keys:
        is_same = False
    else:
        is_same = all(
            np.allclose(gauss_acc[sigma], no_norm_acc[sigma])
            for sigma in gauss_acc.keys()
        )
    print(f"Gaussian kernel (with normalization) equals no-norm Gaussian? {is_same}")
    
    



def kFold_search(
    x: np.ndarray,
    y: np.ndarray,
    min_sigma: float = 0.001,
    max_sigma: float = 2,
    diff_sigma: float = 0.005,
    splits: int = 5,
    kerns=[PnnKernels.gaussian_kernel],
) -> dict[str, dict[float, np.ndarray]]:
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
        str, dict[float, np.ndarray]
    ] = {}  # {kernel_name: {sigma_value: (min_acc, max_acc, avg_acc)}}

    kf = KFold(n_splits=splits, shuffle=True, random_state=67)
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
                y_pred = p.predict(x_test)
                fold_accuracy = np.mean(y_pred == y_test)
                fold_accuracies.append(fold_accuracy)
            accuracies[kernel_name][sigma] = np.array(fold_accuracies)
            print(
                f"Kernel: {kernel_name}, Sigma: {sigma:.3f}, Accuracy: {np.mean(accuracies[kernel_name][sigma]):.4f} (min: {np.min(accuracies[kernel_name][sigma]):.4f}, max: {np.max(accuracies[kernel_name][sigma]):.4f})"
            )
    return accuracies


def save_results(
    accuracies: dict[str, dict[float, np.ndarray]], filename: str = "results.csv"
):
    """
    Save the accuracies for each kernel and sigma combination to a CSV file.
    :param accuracies: A dictionary containing the accuracies for each kernel and sigma combination.
    :param filename: The name of the CSV file to save the results to.
    """
    with open(filename, "w") as f:
        f.write("Kernel,Sigma,Min Accuracy,Max Accuracy,Average Accuracy\n")
        for kernel_name, sigma_dict in accuracies.items():
            for sigma, stats in sigma_dict.items():
                min_acc = np.min(stats)
                max_acc = np.max(stats)
                avg_acc = np.mean(stats)
                f.write(
                    f"{kernel_name},{sigma:.3f},{min_acc:.4f},{max_acc:.4f},{avg_acc:.4f}\n"
                )


if __name__ == "__main__":
    main()
