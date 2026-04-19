import numpy as np
from src.kernels import PnnKernels
from src.pnn import PNN
import timeit
import matplotlib.pyplot as plt

def load_data(path : str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from a CSV file. The last column is assumed to be the class label.
    :param path: Path to the CSV file containing the data.
    :return: A tuple containing the features (X) and labels (y) as numpy arrays.
    """
    data = np.loadtxt(path, delimiter=',')
    x = data[:, :-1]  # All columns except the last one are features
    y = data[:, -1]   # The last column is the class label
    return x, y


def normalize_data(X: np.ndarray) -> np.ndarray:
    """
    Normalize the features in the dataset to have zero mean and unit variance.
    :param X: The input feature matrix to be normalized.
    :return: The normalized feature matrix.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def main():
    p = PNN(sigma=0.02)
    data = load_data("spambase.data")

    #Split into training (80%) and testing (20%) sets
    np.random.seed(67)  # For reproducibility
    indices = np.random.permutation(data[0].shape[0])
    train_size = int(0.8 * data[0].shape[0])
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    normalized_x = normalize_data(data[0])
    y = data[1]


    x_train, y_train = normalized_x[train_indices], y[train_indices]
    x_test, y_test = normalized_x[test_indices], y[test_indices]


    p.fit(x_train, y_train)
    print(f"Time for fitting {len(x_train)} samples: {timeit.timeit(lambda: p.fit(x_train, y_train), number=1):.4f} seconds")
    predictions = p.predict(x_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Time for predicting {len(x_test)} samples: {timeit.timeit(lambda: p.predict(x_test), number=1):.4f} seconds")

    # Best sigma and kernel search
    sigma = 0.001
    accuracies: dict[str, dict[float, np.floating]] = {} # {kernel_name: {sigma_value: accuracy_value}}
    kerns = [PnnKernels.gaussian_kernel, PnnKernels.laplacian_kernel, PnnKernels.cauchy_kernel, PnnKernels.inverse_multiquadric_kernel, PnnKernels.epanechnikov_kernel, PnnKernels.triangular_kernel]
    for kernel in kerns:
        p.kernel = kernel
        accuracies.setdefault(f"{kernel.__name__}", {})
        while sigma < 1:
            p.sigma = sigma
            predictions = p.predict(x_test[:100])
            accuracy = np.mean(predictions == y_test[:100])
            print(f"Test Accuracy: {accuracy:.2f}, Sigma: {sigma:.3f}, Kernel: {kernel.__name__}")
            accuracies[f"{kernel.__name__}"][sigma] = float(accuracy)
            sigma += 0.005
        sigma = 0.001


    # Find the best kernel and sigma based on the recorded accuracies
    best_kernel, best_sigma, best_accuracy = max(
        ((k, max(v.items(), key=lambda iv: iv[1])[0], max(v.items(), key=lambda iv: iv[1])[1]) for k, v in accuracies.items() if v),
        default=(None, None, -1.0),
        key=lambda t: t[2]
    )
    if best_kernel is not None:
        print(f"Best configuration: {best_kernel}, sigma {best_sigma:.3f} with accuracy {best_accuracy:.2f}")
    else:
        print("No configurations evaluated.")

    # Visualize kernel and sigma experiments
    for kernel_name, accs in accuracies.items():
        plt.plot(list(accs.keys()), list(accs.values()), label=kernel_name)
        plt.xlabel("Sigma")
        plt.ylabel("Accuracy")
        plt.title("Performance of Different Configurations")
        plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
