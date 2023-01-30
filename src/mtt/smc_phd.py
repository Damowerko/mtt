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
            [Ts**2 / 2, 0.0],
            [0.0, Ts**2 / 2],
            [Ts, 0.0],
            [0.0, Ts],
        ]
    )


class SMCPHD:
    """
    SMC-PHD filter as described by [1]. Particles are born adaptively based on sensor measurements.

    [1] B. Ristic, D. Clark and B. -N. Vo,
        "Improved SMC implementation of the PHD filter,"
        2010 13th International Conference on Information Fusion,
        Edinburgh, UK, 2010, pp. 1-8, doi: 10.1109/ICIF.2010.5711922.
    """

    def __init__(
        self,
        simulator: Simulator,
        particles_per_measurement: int = 100,
    ):
        self.simulator = simulator
        self.particles_per_measurement = particles_per_measurement

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
        n_born_expected = 0
        newborn_states_list: List[npt.NDArray[np.floating]] = []
        # particle birth based on measurements
        for sensor, sensor_measurements in zip(self.simulator.sensors, measurements):
            assert sensor_measurements.shape[1] == 2
            # add noise in polar coordinates
            sensor_measurements = to_polar(sensor_measurements - sensor.position)
            birth_positions = rng.normal(
                loc=sensor_measurements[None, ...],
                scale=sensor.noise[None, None, ...],
                size=(self.particles_per_measurement,) + sensor_measurements.shape,
            ).reshape(-1, 2)
            birth_positions = to_cartesian(birth_positions) + sensor.position

            n_born_per_meter = self.simulator.p_birth / 1000**2
            n_born_expected += n_born_per_meter * sensor.measurement_density(
                birth_positions, sensor_measurements, sum=True, jacobian=False
            )

            newborn_states_list += [
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
        # birth_states is a list of (N, n_state_dims)
        newborn_states = np.concatenate(newborn_states_list, axis=0)
        newborn_weights = (n_born_expected / newborn_states.shape[0]) * np.ones(
            newborn_states.shape[0]
        )
        return newborn_states, newborn_weights

    def prediction_step(
        self,
        states: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        measurements: List[npt.NDArray[np.float64]],
    ):
        persistent_states = states @ self.transition_matrix.T
        persistent_states += (
            rng.normal(size=(states.shape[0], 2), scale=self.simulator.sigma_motion)
            @ self.noise_matrix.T
        )
        persistent_weights = weights * self.simulator.p_survival
        newborn_states, newborn_weights = self.birth_adaptive(measurements)
        return persistent_states, persistent_weights, newborn_states, newborn_weights

    def update_step(
        self,
        persistent_states: npt.NDArray[np.float64],
        persistent_weights: npt.NDArray[np.float64],
        newborn_states: npt.NDArray[np.float64],
        newborn_weights: npt.NDArray[np.float64],
        measurements: List[npt.NDArray[np.float64]],
    ):
        newborn_denominator = np.zeros(newborn_states.shape[0])
        for sensor, sensor_measurements in zip(self.simulator.sensors, measurements):

            target_in_range: npt.NDArray[np.floating] = (
                np.linalg.norm(persistent_states[..., :2] - sensor.position, axis=1)
                < sensor.range_max
            )
            detection_probability = sensor.p_detection * target_in_range

            measurement_in_range: npt.NDArray[np.floating] = (
                np.linalg.norm(sensor_measurements - sensor.position, axis=1)
                < sensor.range_max
            )
            clutter_intensity = (
                measurement_in_range
                * self.simulator.n_clutter
                / (np.pi * sensor.range_max**2)
            )

            # (N, M) where N is the number of particles and M is the number of measurements (for this sensor)
            measurement_likelyhood = sensor.p_detection * sensor.measurement_density(
                persistent_states[..., :2],
                sensor_measurements,
                sum=False,
                jacobian=False,
            )
            likelyhood = detection_probability * measurement_likelyhood

            # (1, M)
            L = (
                clutter_intensity[None, ...]
                + newborn_weights.sum()
                + persistent_weights[None, :] @ likelyhood
            )

            updated_newborn_weights += (newborn_weights[:, None] / L).sum(axis=1)

            # phi is NxM, C is M while clutter_intensity is a scaler
            likelyhood += (phi / (clutter_intensity + C)[None, :]).sum(
                axis=1
            ) / detections_per_target

            # denominator = clutter_intensity + newborn_weights.sum() + phi.sum()

        n_sensors_in_range = np.zeros(persistent_states.shape[0])
        for sensor in self.simulator.sensors:
            n_sensors_in_range += (
                np.linalg.norm(persistent_states[:, :2] - sensor.position[None, :])
                <= sensor.range_max
            )

        updated_weights = (p_not_detection + likelyhood) * predicted_weights
        return updated_weights

    def resample(self, states, weights):
        n_particles = int(np.round(weights.sum() * self.particles_per_target))
        idx = rng.choice(len(weights), size=n_particles, p=weights / weights.sum())
        resampled_states = states[idx]
        resampled_weights = weights.sum() * np.full(n_particles, 1 / n_particles)
        return resampled_states, resampled_weights
