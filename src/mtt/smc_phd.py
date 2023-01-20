from typing import List

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans

from mtt.simulator import Simulator
from mtt.utils import to_cartesian, to_polar

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
    def __init__(
        self,
        simulator: Simulator,
        particles_per_target: int = 1000,
        adaptive_birth=False,
    ):
        self.simulator = simulator
        self.particles_per_target = particles_per_target
        self.adaptive_birth = adaptive_birth

        self.transition_matrix = cv_transition_matrix(simulator.dt)
        self.noise_matrix = cv_noise_matrix(simulator.dt)

        ndim = self.transition_matrix.shape[1]
        self.states = np.zeros((0, ndim))
        self.weights = np.zeros(0)

    def step(self, measurements: List[npt.NDArray[np.float64]]):
        predicted_states, predicted_weights = self.prediction_step(
            self.states, self.weights, measurements
        )
        updated_weights = self.update_step(
            predicted_states, predicted_weights, measurements
        )
        self.states, self.weights = self.resample(predicted_states, updated_weights)

    def extract_states(
        self, cardinality_search_range: int = 3
    ) -> npt.NDArray[np.floating]:
        """
        Extract the estimated states from the particles.
        Args:
            cardinality_search_range: Will run k-means a number of times and return the best to extract the states from the particle distribution.
        Returns:
            estimated_states: (n,d) array of n estimated states each with dimension d.
        """
        weights = self.weights

        estimated_states: List[npt.NDArray] = []
        if weights.sum() < 0.5:
            return np.zeros((0, self.states.shape[1]))

        best_n_clusters = 0
        best_inertia = np.inf

        n_targets = int(np.round(weights.sum()))
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
            kmeans.fit(self.states, sample_weight=weights)
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
        labels = kmeans.fit_predict(self.states, sample_weight=weights)
        for cluster_index in range(best_n_clusters):
            if np.sum(weights[labels == cluster_index]) > 0.5:
                estimated_states += [kmeans.cluster_centers_[cluster_index]]
        return (
            np.stack(estimated_states)
            if len(estimated_states) > 0
            else np.zeros((0, self.states.shape[1]))
        )

    def birth_adaptive(self, measurements: List[npt.NDArray[np.float64]]):
        birth_states = []
        n_measurements = sum([len(m) for m in measurements])
        n_born_per_measurement = int(
            np.ceil(
                self.particles_per_target
                * self.simulator.p_birth
                * (self.simulator.window**2 / 1000**2)
                / n_measurements
            )
        )
        # particle birth based on measurements
        for sensor, sensor_measurements in zip(self.simulator.sensors, measurements):
            # verify the number of dimensions of the measurements
            assert sensor_measurements.shape[1] == 2
            # convert measurements to polar coordinates around sensor
            sensor_measurements = to_polar(sensor_measurements - sensor.position)
            birth_positions = rng.normal(
                loc=sensor_measurements[None, ...],
                scale=sensor.noise[None, None, ...],
                size=(n_born_per_measurement,) + sensor_measurements.shape,
            ).reshape(-1, 2)
            # convert back to cartesian coordinates
            birth_positions = to_cartesian(birth_positions) + sensor.position
            birth_states += [
                np.concatenate(
                    [birth_positions]
                    + [
                        rng.normal(
                            scale=sigma,
                            size=(birth_positions.shape[0], 2),
                        )
                        for sigma in self.simulator.sigma_initial_state
                    ],
                    axis=1,
                )
            ]
        # birth_states is a list of (n_i, n_state_dims) arrays
        return np.concatenate(birth_states, axis=0)

    def birth_uniform(self, measurements: List[npt.NDArray[np.float64]]):
        # particle birth
        area = self.simulator.window**2 / 1000**2
        n_born = int(
            np.round(self.particles_per_target * self.simulator.p_birth * area)
        )
        birth_states = [
            rng.uniform(
                low=-self.simulator.window / 2,
                high=self.simulator.window / 2,
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
        return np.concatenate(birth_states, axis=1)

    def prediction_step(
        self,
        states: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        measurements: List[npt.NDArray[np.float64]],
    ):
        # surviving particles
        survival_states = states @ self.transition_matrix.T
        survival_states += (
            rng.normal(size=(states.shape[0], 2), scale=self.simulator.sigma_motion)
            @ self.noise_matrix.T
        )
        survival_weights = weights * self.simulator.p_survival

        if self.adaptive_birth:
            birth_states = self.birth_adaptive(measurements)
        else:
            birth_states = self.birth_uniform(measurements)

        # the weights should add up to avg number of targets born
        area = self.simulator.window**2 / 1000**2
        birth_weights = (
            self.simulator.p_birth
            * area
            * np.full(birth_states.shape[0], 1 / birth_states.shape[0])
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
        n_in_range = np.zeros(predicted_states.shape[0])
        for sensor in self.simulator.sensors:
            in_range = (
                np.linalg.norm(predicted_states[:, :2] - sensor.position[None, :])
                <= sensor.range_max
            )
            n_in_range += 1 * in_range
            p_not_detection *= (~in_range) * 1.0 + in_range * (1 - sensor.p_detection)

        likelyhood = np.zeros(predicted_states.shape[0])
        for sensor, sensor_measurements in zip(self.simulator.sensors, measurements):
            clutter_intensity = self.simulator.n_clutter / (
                np.pi * sensor.range_max**2
            )
            detections_per_target = (
                self.simulator.n_sensors * np.pi * sensor.range_max**2 / 1000**2
            )
            # phi is N, M where N is the number of particles and M is the number of measurements (for this sensor)
            phi = sensor.p_detection * sensor.measurement_density(
                predicted_states[..., :2],
                sensor_measurements,
                sum=False,
                jacobian=False,
            )
            C = predicted_weights @ phi
            # rescale based on number of expected detections per target
            detections_per_target = (
                self.simulator.n_sensors * np.pi * sensor.range_max**2 / 1000**2
            )
            # phi is NxM, C is M while clutter_intensity is a scaler
            likelyhood += (phi / (clutter_intensity + C)[None, :]).sum(
                axis=1
            ) / detections_per_target
        updated_weights = (p_not_detection + likelyhood) * predicted_weights
        return updated_weights

    def resample(self, states, weights):
        n_particles = int(np.round(weights.sum() * self.particles_per_target))
        idx = rng.choice(len(weights), size=n_particles, p=weights / weights.sum())
        resampled_states = states[idx]
        resampled_weights = weights.sum() * np.full(n_particles, 1 / n_particles)
        return resampled_states, resampled_weights
