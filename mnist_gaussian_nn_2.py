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
    """
    Estimate the mean and covariance of given data to represent it as a Gaussian.

    Parameters:
        data (np.ndarray): Data of shape (N, D)

    Returns:
        (mean, cov): mean is of shape (D,), cov is of shape (D, D)
    """
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    return mean, cov


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset from openml and split into train/test sets.

    Returns:
        X_train, y_train, X_test, y_test: MNIST images and labels.
        X shapes: (N, 784), y shapes: (N,)
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    # Normalize data to [0,1]
    X /= 255.0

    # Simple train/test split:
    # We'll take the first 60,000 as train and last 10,000 as test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test


def build_class_gaussians(X_train: np.ndarray, y_train: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    For each digit class (0-9), estimate a Gaussian distribution in pixel space.

    Parameters:
        X_train (np.ndarray): Training images, shape (N, 784)
        y_train (np.ndarray): Training labels, shape (N,)

    Returns:
        List of (mean, cov) for each class digit from 0 to 9.
    """
    class_gaussians = []
    for digit in range(10):
        digit_data = X_train[y_train == digit]
        mean, cov = estimate_gaussian_parameters(digit_data)
        # Add small regularization to covariance to avoid singularity
        cov += 1e-5 * np.eye(cov.shape[0])
        class_gaussians.append((mean, cov))
    return class_gaussians


def classify_image_with_gaussians(image: np.ndarray, class_gaussians: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """
    Classify a single image by choosing the Gaussian class that gives the highest likelihood.

    Parameters:
        image (np.ndarray): Single image of shape (784,)
        class_gaussians (List): List of (mean, cov) tuples for each class.

    Returns:
        int: Predicted digit class.
    """
    # Evaluate the probability density for each class and pick the argmax
    scores = []
    for mean, cov in class_gaussians:
        rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        score = rv.logpdf(image)
        scores.append(score)
    return int(np.argmax(scores))


if __name__ == "__main__":
    # --------------------------------------------
    # Part 1: Illustrate Gaussian Data Simulation
    # --------------------------------------------
    # Let's create a random 3D Gaussian distribution and visualize it.
    mean_3d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    cov_3d = np.array([[1.0, 0.8, 0.3],
                       [0.8, 1.5, 0.4],
                       [0.3, 0.4, 1.0]], dtype=np.float64)
    data_3d = generate_gaussian_data(mean_3d, cov_3d, 2000)

    # Visualize using a 2D PCA projection
    visualize_2d_projection(data_3d, title="2D PCA Projection of 3D Gaussian Data")

    # Visualize pairwise projections (0-1), (0-2), (1-2)
    pairwise_projections(data_3d, dims=[0, 1, 2])

    # --------------------------------------------
    # Part 2: Representing MNIST Images as Gaussians
    # --------------------------------------------
    # Load MNIST
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Each image is 784-dimensional. Let's pick a single class (e.g., digit '0') and estimate its Gaussian parameters.
    zero_data = X_train[y_train == 0]
    zero_mean, zero_cov = estimate_gaussian_parameters(zero_data)

    # Visualize the zero_data distribution (784D) via PCA projection to 2D
    # We'll take a subset for speed in plotting
    subset = zero_data[:2000]
    visualize_2d_projection(subset, title="2D PCA of MNIST Zeros")

    # --------------------------------------------
    # Part 3: Gaussian Classification on MNIST
    # --------------------------------------------
    # Estimate Gaussian parameters for all digit classes
    class_gaussians = build_class_gaussians(X_train, y_train)

    # Let's pick a test image and predict its class:
    test_image = X_test[0]
    test_label = y_test[0]

    pred_class = classify_image_with_gaussians(test_image, class_gaussians)
    print(f"True label: {test_label}, Predicted label by Gaussian model: {pred_class}")


    # Notes:
    # In a real scenario, these Gaussian models are very simplistic and ignore correlations if the covariance is not well-structured.
    # PCA and pairwise plots help build intuition about the shape and distribution of the data.
    # For MNIST, the distribution of pixel intensities per class is complex, and these simple Gaussian approximations are quite rough.
