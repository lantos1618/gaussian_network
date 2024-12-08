from typing import Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from itertools import combinations

def gaussian_1d(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    # Strictly typed: mu and sigma as floats, x as np.ndarray
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gaussian_2d(
    X: np.ndarray,
    Y: np.ndarray,
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float
) -> np.ndarray:
    # Strictly typed inputs
    # X and Y are np.ndarray grids from meshgrid
    return (1.0 / (2.0 * np.pi * sigma_x * sigma_y)) * np.exp(
        -0.5 * (((X - mu_x) ** 2 / sigma_x**2) + ((Y - mu_y) ** 2 / sigma_y**2))
    )

def visualize_1d_gaussian(mu: float = 0.0, sigma: float = 1.0) -> None:
    x: np.ndarray = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    y: np.ndarray = gaussian_1d(x, mu, sigma)

    plt.figure(figsize=(6,4))
    plt.plot(x, y, label=f"1D Gaussian (μ={mu}, σ={sigma})")
    plt.title("1D Gaussian")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_2d_gaussian(
    mu_x: float = 0.0,
    mu_y: float = 0.0,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0
) -> None:
    x: np.ndarray = np.linspace(mu_x - 3*sigma_x, mu_x + 3*sigma_x, 100)
    y: np.ndarray = np.linspace(mu_y - 3*sigma_y, mu_y + 3*sigma_y, 100)
    X, Y = np.meshgrid(x, y)
    Z: np.ndarray = gaussian_2d(X, Y, mu_x, mu_y, sigma_x, sigma_y)

    fig = plt.figure(figsize=(12,5))

    # Contour plot
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X, Y, Z, cmap=cm.viridis)
    fig.colorbar(contour, ax=ax1)
    ax1.set_title("2D Gaussian - Contour")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Surface plot
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none')
    ax2.set_title("2D Gaussian - Surface")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("Probability Density")

    plt.tight_layout()
    plt.show()



def generate_multidim_gaussian(
    dim: int,
    mean: np.ndarray,
    cov: np.ndarray,
    num_samples: int = 2000
) -> np.ndarray:
    # Generate samples from an n-dimensional Gaussian
    # mean: shape (dim,)
    # cov: shape (dim, dim)
    return np.random.multivariate_normal(mean, cov, size=num_samples)

def visualize_highdim_gaussian(samples: np.ndarray) -> None:
    # samples shape: (num_samples, dim)

    dim = samples.shape[1]
    # Dimensionality reduction with PCA to 2D
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(samples)

    plt.figure(figsize=(6,4))
    plt.scatter(reduced[:,0], reduced[:,1], alpha=0.3)
    plt.title(f"PCA Projection of {dim}-D Gaussian to 2D")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

    # Additionally, show pairwise projections of a few dimension pairs:
    # For higher dim, just pick a few pairs for illustration
    pairs_to_plot = list(combinations(range(dim), 2))
    pairs_to_plot = pairs_to_plot[:min(len(pairs_to_plot), 6)]  # limit plots
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (d1, d2) in zip(axes, pairs_to_plot):
        ax.scatter(samples[:, d1], samples[:, d2], alpha=0.3)
        ax.set_title(f"Dimensions {d1} vs {d2}")
        ax.set_xlabel(f"Dim {d1}")
        ax.set_ylabel(f"Dim {d2}")
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Visualize a standard 1D Gaussian
    visualize_1d_gaussian(mu=0.0, sigma=1.0)

    # Visualize a 2D Gaussian
    visualize_2d_gaussian(mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0)

      # Example for a 5-D Gaussian
    dim: int = 5
    mean: np.ndarray = np.zeros(dim, dtype=np.float64)  # Mean vector
    # Create a covariance matrix with some correlations
    # For simplicity, start with identity and add small correlations:
    cov: np.ndarray = np.eye(dim, dtype=np.float64)
    cov[0,1] = cov[1,0] = 0.5
    cov[3,4] = cov[4,3] = -0.3

    samples: np.ndarray = generate_multidim_gaussian(dim, mean, cov, num_samples=2000)
    visualize_highdim_gaussian(samples)
