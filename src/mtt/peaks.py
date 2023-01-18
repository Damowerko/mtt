from typing import Tuple, NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.cluster import KMeans
import torch

from mtt.utils import gaussian, make_grid


rng = np.random.default_rng()


class GMM(NamedTuple):
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray


class GaussianMixtureImage:
    def __init__(self, n_components, device=None) -> None:
        self.n_components = n_components
        self.mu = torch.zeros(n_components, 2, device=device)
        self.cov = torch.zeros(n_components, 2, 2, device=device)


def find_peaks(image: np.ndarray, width: float, n_peaks=None, n_peaks_scale=1.0) -> GMM:
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

    if n_peaks is None:
        # assuming number of peaks is approximately the sum of all pixels
        n_components = int(np.round(image.sum() * n_peaks_scale))
    else:
        n_components = n_peaks

    # Sample image based on pixel values.
    samples = sample_image(image, width)

    # Fit gaussian mixture model to find peaks.
    return fit_gmm(samples, n_components=n_components)


def sample_image(img: np.ndarray, width: float) -> np.ndarray:
    """
    Sample `img` based on pixel values.
    """
    XY = make_grid(img.shape, width)
    # add epsilon to avoid division by zero
    img += 1e-8
    idx = rng.choice(img.size, size=1000, p=img.reshape(-1) / img.sum(), shuffle=False)
    return XY.reshape(-1, 2)[idx]


def fit_gmm(samples: np.ndarray, n_components: int) -> GMM:
    """
    1. Sample `img` based on pixel values.
    2. Fit gaussian mixture model to find peaks.

    Returns:
        A GMM object with the means, covariances and weights of the fitted gaussian mixture model.
    """
    # if n_components is zero, return empty arrays
    if n_components == 0:
        return GMM(np.empty((0, 2)), np.empty((0, 2, 2)), np.empty(0))
    if n_components < 0:
        raise ValueError(f"n_components must be non-negative, got {n_components}.")

    # knn = KMeans(n_clusters=n_components)
    # knn.fit(samples)
    # means = knn.cluster_centers_
    # return means, None

    # fit kmeans
    gmm = GaussianMixture(
        n_components=n_components,
        n_init=10,
    )
    gmm.fit(samples)
    # get only the large weights
    idx = gmm.weights_ > 0.5 / n_components
    return GMM(gmm.means_[idx], gmm.covariances_[idx], gmm.weights_[idx])


def main():
    n = rng.poisson(10)
    width = 1000
    img_size = 256
    positions = rng.uniform(low=-width / 2, high=width / 2, size=(n, 2))
    XY = make_grid(img_size, width)
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
    means, covariances, _ = fit_gmm(samples, n_components)

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

    ax[2].imshow(
        img_hat, cmap="gray_r", origin="lower", extent=[-width / 2, width / 2] * 2
    )
    ax[2].set_title("GMM")
    ax[1].set_xlim(-width / 2, width / 2)
    ax[1].set_ylim(-width / 2, width / 2)

    plt.savefig("figures/peaks.png")


if __name__ == "__main__":
    main()
