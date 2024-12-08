from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from scipy.stats import multivariate_normal


def generate_gaussian_data(mean: np.ndarray, cov: np.ndarray, num_samples: int = 1000) -> np.ndarray:
    """
    Generate samples from a Gaussian distribution in N-dimensions.

    Parameters:
        mean (np.ndarray): Mean vector of shape (D,)
        cov (np.ndarray): Covariance matrix of shape (D, D)
        num_samples (int): Number of samples to generate

    Returns:
        np.ndarray: Array of shape (num_samples, D) containing the generated samples.
    """
    return np.random.multivariate_normal(mean, cov, num_samples)


def visualize_2d_projection(data: np.ndarray, title: str = "2D Projection"):
    """
    Visualize high-dimensional data projected into 2D.

    Parameters:
        data (np.ndarray): Data of shape (N, D)
        title (str): Plot title
    """
    # Use PCA to project the data into 2 dimensions
    pca = PCA(n_components=2)
    projected = pca.fit_transform(data)

    plt.figure(figsize=(6, 5))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()


def pairwise_projections(data: np.ndarray, dims: List[int] = [0, 1, 2]):
    """
    Create pairwise scatter plots of given dimensions to visualize the shape of the distribution.

    Parameters:
        data (np.ndarray): Data of shape (N, D)
        dims (List[int]): Dimensions to plot pairwise scatter plots for.
    """
    # We'll plot pairwise scatter plots for the specified dimensions.
    # E.g., if dims = [0,1,2], we plot (0 vs 1), (0 vs 2), and (1 vs 2)
    pairs = [(i, j) for i in range(len(dims)) for j in range(i+1, len(dims))]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]

    for (ax, (di, dj)) in zip(axes, pairs):
        ax.scatter(data[:, dims[di]], data[:, dims[dj]], alpha=0.5)
        ax.set_xlabel(f"Dimension {dims[di]}")
        ax.set_ylabel(f"Dimension {dims[dj]}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def estimate_gaussian_parameters(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mean, cov


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist = fetch_openml('mnist_784', version=1, as_frame=True)
    X = mnist.data.to_numpy(dtype=np.float32)
    y = mnist.target.to_numpy(dtype=np.int64)

    # Normalize data to [0,1]
    X /= 255.0

    # Simple train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def build_class_gaussians(X_train: np.ndarray, y_train: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    class_gaussians = []
    for digit in range(10):
        digit_data = X_train[y_train == digit]
        mean, cov = estimate_gaussian_parameters(digit_data)
        # Regularize covariance
        cov += 1e-5 * np.eye(cov.shape[0])
        class_gaussians.append((mean, cov))
    return class_gaussians


def classify_image_with_gaussians(image: np.ndarray, class_gaussians: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    scores = []
    for mean, cov in class_gaussians:
        rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        score = rv.logpdf(image)
        scores.append(score)
    return int(np.argmax(scores))


if __name__ == "__main__":
    # Load and prepare MNIST data
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Build Gaussian models for each digit class
    class_gaussians = build_class_gaussians(X_train, y_train)

    # Test on a single test image
    test_image = X_test[0]  # This will now work as expected
    test_label = y_test[0]

    pred_class = classify_image_with_gaussians(test_image, class_gaussians)
    print(f"True label: {test_label}, Predicted label by Gaussian model: {pred_class}")

    # Evaluate the accuracy on a subset of the test set
    subset_size = 1000
    test_subset = X_test[:subset_size]
    labels_subset = y_test[:subset_size]

    correct = 0
    for i in range(subset_size):
        if i % 10 == 0:
            print(f"Evaluating image {i} of {subset_size}")
        pred = classify_image_with_gaussians(test_subset[i], class_gaussians)
        if pred == labels_subset[i]:
            correct += 1

    accuracy = correct / subset_size
    print(f"Accuracy on a subset of size {subset_size}: {accuracy * 100:.2f}%")