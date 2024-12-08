import time
import psutil
import os
import numpy as np
from typing import Tuple

def gaussian_2d(
    x: np.ndarray, 
    y: np.ndarray, 
    px: float, 
    py: float, 
    sigma: float, 
    amplitude: float
) -> np.ndarray:
    """
    Evaluate a 2D isotropic Gaussian at points (x, y).

    Parameters:
        x (np.ndarray): 2D array of x-coordinates.
        y (np.ndarray): 2D array of y-coordinates.
        px (float): x-coordinate of the Gaussian center.
        py (float): y-coordinate of the Gaussian center.
        sigma (float): Standard deviation of the Gaussian (isotropic).
        amplitude (float): Scaling factor for the Gaussian peak.

    Returns:
        np.ndarray: 2D array of Gaussian values corresponding to each point in (x, y).
    """
    dx = x - px
    dy = y - py
    dist_sq = dx*dx + dy*dy
    return amplitude * np.exp(-dist_sq / (2 * sigma * sigma))

def main() -> None:
    # Number of weights - let's increase this significantly to simulate a "larger" tensor
    # For example, go from 10k to 100k weights.
    N: int = 100000
    weights: np.ndarray = np.random.randn(N).astype(np.float32)

    # Measure memory at start of main for baseline
    process = psutil.Process(os.getpid())
    mem_start: float = process.memory_info().rss / (1024 * 1024)  # in MB
    print(f"Memory at start: {mem_start:.2f} MB")

    # Simple 2D embedding
    side: int = int(np.ceil(np.sqrt(N)))
    indices: np.ndarray = np.arange(N)
    px: np.ndarray = (indices % side).astype(np.float32) / side
    py: np.ndarray = (indices // side).astype(np.float32) / side

    sigma: float = 0.01

    # Increase resolution to simulate more complex tensor field
    res: int = 512  # doubling resolution to increase workload
    x_lin: np.ndarray = np.linspace(0, 1, res, dtype=np.float32)
    y_lin: np.ndarray = np.linspace(0, 1, res, dtype=np.float32)
    X, Y = np.meshgrid(x_lin, y_lin, indexing='ij')

    mem_before: float = process.memory_info().rss / (1024 * 1024)  # in MB

    start: float = time.time()

    field: np.ndarray = np.zeros((res, res), dtype=np.float32)
    # Summation of all Gaussian contributions
    for i in range(N):
        field += gaussian_2d(X, Y, float(px[i]), float(py[i]), sigma, float(weights[i]))

    end: float = time.time()
    mem_after: float = process.memory_info().rss / (1024 * 1024)  # in MB

    print(f"Computation Time: {end - start:.4f}s")
    print(f"Memory Before Field Computation: {mem_before:.2f} MB")
    print(f"Memory After Field Computation: {mem_after:.2f} MB")

    center_x: int = res // 2
    center_y: int = res // 2
    print("Field sample (center 5x5 block):")
    print(field[center_x:center_x+5, center_y:center_y+5])

if __name__ == "__main__":
    main()
