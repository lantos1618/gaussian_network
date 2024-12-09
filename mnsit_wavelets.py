from __future__ import annotations
import numpy as np
from typing import Tuple, List
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pywt

def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int64)

    # Normalize data to [0,1]
    X /= 255.0

    # Simple train/test split
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test

def wavelet_transform_image(image: np.ndarray, wavelet: str = "db1") -> np.ndarray:
    """
    Apply a wavelet transform to a single MNIST image (28x28).
    We will use a discrete wavelet transform (DWT) to represent the image as "waves".
    
    Parameters:
        image (np.ndarray): Flattened image of shape (784,)
        wavelet (str): Wavelet type (e.g., 'db1', 'haar', 'sym2', etc.)
    
    Returns:
        np.ndarray: Wavelet-transformed coefficients flattened.
    """
    # Reshape image into 2D
    img_2d = image.reshape(28, 28)
    
    # Perform a single-level 2D wavelet decomposition
    # This returns approximation and detail coefficients
    coeffs = pywt.dwt2(img_2d, wavelet)
    # coeffs = (cA, (cH, cV, cD))
    cA, (cH, cV, cD) = coeffs
    
    # Flatten all coefficients into one vector
    # We include approximation and detail coefficients
    transformed = np.concatenate([
        cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()
    ])
    return transformed

def wavelet_transform_dataset(X: np.ndarray, wavelet: str = "db1") -> np.ndarray:
    """
    Transform the entire dataset using the wavelet transform.
    
    Parameters:
        X (np.ndarray): Dataset of shape (N, 784)
        wavelet (str): Wavelet type
    
    Returns:
        np.ndarray: Wavelet-transformed dataset of shape (N, num_features)
    """
    transformed_data = []
    for i in range(X.shape[0]):
        transformed_data.append(wavelet_transform_image(X[i], wavelet=wavelet))
    return np.array(transformed_data, dtype=np.float32)

if __name__ == "__main__":
    # Load MNIST
    X_train, y_train, X_test, y_test = load_mnist_data()

    # Transform data into a wavelet domain representation
    # Using 'db1' (Daubechies 1) wavelet for simplicity
    print("Transforming training data...")
    X_train_wave = wavelet_transform_dataset(X_train, wavelet="db1")
    print("Transforming test data...")
    X_test_wave = wavelet_transform_dataset(X_test, wavelet="db1")

    # Train a simple classifier on the wavelet-transformed features
    print("Training classifier...")
    clf = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train_wave, y_train)

    # Evaluate on the test set
    print("Evaluating...")
    y_pred = clf.predict(X_test_wave)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy with wavelet-based representation: {acc * 100:.2f}%")
