from typing import List

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from mtt.simulator import Simulator

rng = np.random.default_rng()


def cv_transition_matrix(Ts: float = 1.0) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [1, 0, Ts, 0],
            [0, 1, 0, Ts],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def cv_noise_matrix(Ts: float = 1.0) -> npt.NDArray[np.float64]:
    return np.array(
        [
            [Ts**2 / 2, 0],
            [0, Ts**2 / 2],
            [Ts, 0],
            [0, Ts],
        ]
    )


class SMCPHD:
    def __init__(self, simulator: Simulator, particles_per_target: int = 1000):
        self.simulator = simulator
        self.particles_per_target = particles_per_target

        self.transition_matrix = cv_transition_matrix(simulator.dt)
        self.noise_matrix = cv_noise_matrix(simulator.dt)

        ndim = self.transition_matrix.shape[1]
        self.states = np.zeros((0, ndim))
        self.weights = np.zeros(0)

    def step(self, measurements: List[npt.NDArray[np.float64]]):
        predicted_states, predicted_weights = self.prediction_step(
            self.states, self.weights
        )
        updated_weights = self.update_step(
            predicted_states, predicted_weights, measurements
        )
        self.states, self.weights = self.resample(predicted_states, updated_weights)

    def extract_states(self, cardinality_search_range: int = 3):
        estimated_states: List[npt.NDArray] = []
        if self.weights.sum() < 0.5:
            return estimated_states

        best_n_clusters = 0
        best_inertia = np.inf

        # find best number of clusters withint the cardinality search range
        n_targets = int(np.round(self.weights.sum()))
        for n_clusters in range(
            n_targets - cardinality_search_range, n_targets + cardinality_search_range
        ):
            if n_clusters <= 0:
                continue
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init="auto",
                max_iter=50,
            )
            kmeans.fit(self.states, sample_weight=self.weights)
            inertia: float = kmeans.inertia_  # type: ignore
            if inertia < best_inertia:
                best_inertia = inertia
                best_n_clusters = n_clusters

        kmeans = KMeans(
            n_clusters=best_n_clusters,
            init="k-means++",
            n_init="auto",
            max_iter=300,
        )
        labels = kmeans.fit_predict(self.states, sample_weight=self.weights)
        for cluster_index in range(best_n_clusters):
            if np.sum(self.weights[labels == cluster_index]) > 0.5:
                estimated_states += [kmeans.cluster_centers_[cluster_index]]
        return estimated_states

    def prediction_step(
        self, states: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
    ):
        # surviving particles
        survival_states = states @ self.transition_matrix.T
        survival_states += (
            rng.normal(size=(states.shape[0], 2), scale=self.simulator.sigma_motion)
            @ self.noise_matrix.T
        )
        survival_weights = weights * self.simulator.p_survival

        # particle birth
        n_born = int(
            np.round(
                self.particles_per_target * self.simulator.p_birth * self.simulator.area
            )
        )
        birth_states = [
            rng.uniform(
                low=-self.simulator.width / 2,
                high=self.simulator.width / 2,
                size=(n_born, 2),
            )
        ]
        for sigma in self.simulator.sigma_initial_state:
            birth_states += [
                rng.normal(
                    scale=sigma,
                    size=(n_born, 2),
                )
            ]
        birth_states = np.concatenate(birth_states, axis=1)
        birth_weights = (  # the weights should add up to avg number of targets born
            self.simulator.p_birth * self.simulator.area * np.full(n_born, 1 / n_born)
        )

        predicted_states = np.concatenate([survival_states, birth_states])
        predicted_weights = np.concatenate([survival_weights, birth_weights])
        return predicted_states, predicted_weights

    def update_step(
        self,
        predicted_states,
        predicted_weights,
        measurements: List[npt.NDArray[np.float64]],
    ):
        p_not_detection = np.zeros(predicted_states.shape[0])
        for sensor in self.simulator.sensors:
            in_range = (
                np.linalg.norm(predicted_states[:, :2] - sensor.position[None, :])
                <= sensor.range_max
            )
            p_not_detection *= (~in_range) * 1.0 + in_range * (1 - sensor.p_detection)
        p_detection = 1 - p_not_detection

        likelyhood = np.zeros(predicted_states.shape[0])
        for sensor, sensor_measurements in zip(self.simulator.sensors, measurements):
            clutter_intensity = self.simulator.n_clutter / (
                np.pi * sensor.range_max**2
            )
            # N, M where N is the number of particles and M is the number of measurements (for this sensor)
            measurement_likelyhood = sensor.measurement_density(
                predicted_states[..., :2],
                sensor_measurements,
                sum=False,
                jacobian=False,
            )
            # scale by the number of detection per target on average (ie overlap between sensors)
            measurement_likelyhood /= (
                np.pi * sensor.range_max**2 * self.simulator.n_sensors / 1000**2
            )

            measurement_likelyhood = measurement_likelyhood / (
                clutter_intensity
                + (predicted_weights[None, :] @ measurement_likelyhood)
            )
            likelyhood += measurement_likelyhood.sum(axis=1)

        updated_weights = (
            (1 - p_detection) + p_detection * likelyhood
        ) * predicted_weights
        return updated_weights

    def resample(self, states, weights):
        n_particles = int(np.round(weights.sum() * self.particles_per_target))
        idx = rng.choice(len(weights), size=n_particles, p=weights / weights.sum())
        resampled_states = states[idx]
        resampled_weights = weights.sum() * np.full(n_particles, 1 / n_particles)
        return resampled_states, resampled_weights
