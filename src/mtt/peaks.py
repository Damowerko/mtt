from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from mtt.utils import gaussian


rng = np.random.default_rng()


def find_peaks(image: np.ndarray, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in the `image` by fitting a GMM.
    To fit the mixture we randomly sample points in the image weighted by the intensity.

    Args:
        image: (H, W) the image to find peaks in.
            We assume that the sum off al pixels is approximately the number of peaks.
    Returns:
        means: (n_peaks, 2) the mean of each peak.
        covariances: (n_peaks, 2, 2) the covariance of each peak.
    """

    # set negative pixels to zero
    image = np.maximum(image, 0)

    # assuming number of peaks is approximately the sum of all pixels
    n_components = int(np.round(image.sum()))

    # Sample image based on pixel values.
    samples = sample_image(image, width)

    # Fit gaussian mixture model to find peaks.
    means, covariances = fit_gmm(samples, n_components=n_components)

    return means, covariances


def sample_image(img: np.ndarray, width: float) -> np.ndarray:
    """
    Sample `img` based on pixel values.
    """
    X, Y = np.meshgrid(
        np.linspace(-width / 2, width / 2, img.shape[1]),
        np.linspace(-width / 2, width / 2, img.shape[0]),
    )
    XY = np.stack([X, Y], axis=-1)

    # add epsilon to avoid division by zero
    img = img + 1e-8
    idx = rng.choice(
        img.size, size=img.size, p=img.reshape(-1) / img.sum(), shuffle=False
    )
    return XY.reshape(-1, 2)[idx]


def fit_gmm(samples: np.ndarray, n_components: int):
    """
    1. Sample `img` based on pixel values.
    2. Fit gaussian mixture model to find peaks.
    """
    # if n_components is zero, return empty arrays
    if n_components == 0:
        return np.empty((0, 2)), np.empty((0, 2, 2))
    if n_components < 0:
        raise ValueError(f"n_components must be non-negative, got {n_components}.")

    # Fit gaussian mixture model to find peaks.
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(samples)
    means = gmm.means_
    covariances = gmm.covariances_
    return means, covariances


def main():
    n = rng.poisson(10)
    width = 1000
    img_size = 256
    positions = rng.uniform(low=-width / 2, high=width / 2, size=(n, 2))

    XY = np.stack(
        np.meshgrid(
            np.linspace(-width / 2, width / 2, img_size),
            np.linspace(-width / 2, width / 2, img_size),
        ),
        axis=-1,
    )

    img = np.zeros((img_size, img_size))
    for p in positions:
        cov = [
            [rng.uniform(50, 100), rng.uniform(0, 50)],
            [rng.uniform(0, 50), rng.uniform(50, 100)],
        ]
        z = gaussian(XY, p, cov)
        img += z / np.sum(z)

    samples = sample_image(img, width)
    n_components = int(np.round(img.sum()))
    means, covariances = fit_gmm(samples, n_components)

    # make a plot of the model
    img_hat = np.zeros((img_size, img_size))
    for i in range(means.shape[0]):
        img_hat += gaussian(XY, means[i], covariances[i])

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # make plots of original, samples and GMM
    ax[0].imshow(img, cmap="gray_r", origin="lower", extent=[-width / 2, width / 2] * 2)
    # ax[0].scatter(positions[:, 0], positions[:, 1], s=1)
    ax[0].set_title("Original")
    ax[1].set_xlim(-width / 2, width / 2)
    ax[1].set_ylim(-width / 2, width / 2)

    ax[1].scatter(samples[:, 0], samples[:, 1], s=1)
    ax[1].set_title("Samples")
    ax[1].set_xlim(-width / 2, width / 2)
    ax[1].set_ylim(-width / 2, width / 2)

    ax[2].imshow(img_hat, cmap="gray_r", origin="lower", extent=[-width / 2, width / 2] * 2)
    ax[2].set_title("GMM")
    ax[1].set_xlim(-width / 2, width / 2)
    ax[1].set_ylim(-width / 2, width / 2)

    plt.savefig("figures/peaks.png")


if __name__ == "__main__":
    main()
