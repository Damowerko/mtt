from math import floor
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

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


def find_peaks(
    image: np.ndarray,
    width: float,
    n_peaks=None,
    n_peaks_scale=1.0,
    center=(0, 0),
    method="gmm",
) -> GMM:
    """
    Find peaks in the `image` by fitting a GMM.
    To fit the mixture we randomly sample points in the image weighted by the intensity.

    Args:
        image: (H, W) the image to find peaks in.
            We assume that the sum off al pixels is approximately the number of peaks.
        width: the width in meters of the image.
        n_peaks: the number of peaks to find.
            If None, we assume that the number of peaks is approximately the sum of all pixels.
        n_peaks_scale: the scale of the number of peaks to find.
        center: the center of the image in meters.
        model: the model to use to fit the peaks.
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
    samples = sample_image(image, width, center=center)

    # Fit gaussian mixture model to find peaks.
    if method == "gmm":
        return fit_gmm(samples, n_components=n_components)
    elif method == "kmeans":
        return fit_kmeans(
            samples,
            n_components=n_components,
            n_components_range=2,
        )
    else:
        raise ValueError(f"Unknown model: {method}")


def sample_image(img: np.ndarray, width: float, center=(0, 0)) -> np.ndarray:
    """
    Sample `img` based on pixel values.
    """
    n_samples = int(1000 * img.size / 128**2)
    XY = make_grid(img.shape, width, center=center)
    # add epsilon to avoid division by zero
    img += 1e-8
    idx = rng.choice(
        img.size, size=n_samples, p=img.reshape(-1) / img.sum(), shuffle=False
    )
    return XY.reshape(-1, 2)[idx]


def sample_rkhs(
    mu: np.ndarray, weights: np.ndarray, sigma: float, n_samples: int
) -> np.ndarray:
    """
    Sample from a RKHS defined by the means `mu`, weights `weights` and has gaussian kernel with width `sigma`.
    """
    n_components = mu.shape[0]
    n_samples_per_component = np.random.multinomial(n_samples, weights / weights.sum())
    sample_idx = np.repeat(np.arange(n_components), n_samples_per_component)
    mu_sampled = mu[sample_idx]
    samples = np.random.normal(mu_sampled, sigma)
    return samples


def fit_kmeans(samples: np.ndarray, n_components: int, n_components_range=0):
    if n_components == 0:
        return GMM(np.empty((0, 2)), np.empty((0, 2, 2)), np.empty(0))

    best_means = np.zeros((0, 2))
    best_labels = np.zeros(samples.shape[0], dtype=int)
    best_itertia = np.inf
    for n in range(
        n_components - n_components_range, n_components + n_components_range + 1
    ):
        if n <= 0:
            continue
        knn = KMeans(n_clusters=n, n_init="auto")
        knn.fit(samples)
        if knn.inertia_ is not None and knn.inertia_ < best_itertia:
            best_itertia = knn.inertia_
            best_means = knn.cluster_centers_
            best_labels = knn.labels_
            assert best_labels is not None

    # compute weight use the relative number of points in each cluster
    weights = np.bincount(best_labels) / best_labels.size
    covariances = np.zeros((best_means.shape[0], 2, 2))
    gmm = GMM(best_means, covariances, weights)
    return reweigh(gmm, n_components)


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
    gmm_sklearn = GaussianMixture(
        n_components=n_components,
        n_init=10,
    )
    gmm_sklearn.fit(samples)
    gmm = GMM(gmm_sklearn.means_, gmm_sklearn.covariances_, gmm_sklearn.weights_)
    return reweigh(gmm, n_components)


def reweigh(gmm: GMM, n_components):
    """
    1. Rescale the gmm weights so that the sum of the weights is equal to n_components.
    2. Split components with a weight larger than 1.0 into multiple components of weight at most 1.0.
    For example, if a component has a weight of 2.6, it will be split into 2 components of weight 2 and one component of weight 0.6.
    3. Remove components with a weight smaller than 0.5.
    """
    weights = gmm.weights * n_components / gmm.weights.sum()

    _idx = []
    _weights = []
    # split components with a weight larger than 1.0 into multiple components of weight at most 1.0
    for i in range(int(np.round(np.max(weights)))):
        # discard weights smaller than 0.5
        _idx.append(np.where(weights > i)[0])
        if i == 0:
            # the first component will have the current weight minus the integer part
            _weights.append(weights[_idx[-1]] - np.floor(weights[_idx[-1]]))
        else:
            # the rest of the components will have a weight of 1
            _weights.append(np.ones(_idx[-1].shape[0]))

    # if there are no components with a weight larger than 1.0, return the original gmm
    if len(_idx) == 0:
        return GMM(np.empty((0, 2)), np.empty((0, 2, 2)), np.empty(0))

    idx = np.concatenate(_idx)
    weights = np.concatenate(_weights)
    # get the n_components largest weights
    n_components = floor(np.round(n_components))
    idx = idx[np.argsort(weights)[::-1][:n_components]]
    weights = weights[np.argsort(weights)[::-1][:n_components]]
    return GMM(gmm.means[idx], gmm.covariances[idx], weights)


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
