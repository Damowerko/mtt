from copy import copy
from datetime import datetime
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.detection import Clutter, Detection
from stonesoup.types.state import State, StateVector, TaggedWeightedGaussianState
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.updater.pointprocess import PHDUpdater

from mtt.data import VectorData
from mtt.simulator import Simulator


def gmphd_filter(data: List[VectorData]):
    # parameters
    state_threshold = 0.5
    merge_threshold = 10
    prune_threshold = 1e-3

    simulator: Simulator = data[0].simulator
    assert simulator.model == "CV"

    transition_model = CombinedLinearGaussianTransitionModel(
        (
            ConstantVelocity(simulator.sigma_motion),
            ConstantVelocity(simulator.sigma_motion),
        )
    )

    # Initialize the Multi-Target Multi-Sensor GM-PHD Tracker
    measurement_models: List[CartesianToBearingRange] = []
    for sensor in simulator.sensors:
        sigma_range, sigma_bearing = sensor.noise
        measurement_models.append(
            CartesianToBearingRange(
                ndim_state=2,
                mapping=(0, 1),
                noise_covar=np.diag([sigma_bearing**2, sigma_range**2]),
                translation_offset=sensor.position.reshape(2, 1),
            )
        )
    # n_clutter and n_sensors are per km^2
    clutter_spatial_density = simulator.n_clutter * simulator.n_sensors / 1000**2

    kalman_updater = ExtendedKalmanUpdater()
    updater = PHDUpdater(
        kalman_updater,
        clutter_spatial_density=clutter_spatial_density,
        prob_detection=simulator.p_detection,
        prob_survival=simulator.p_survival,
    )

    kalman_predictor = KalmanPredictor(transition_model)
    base_hypothesiser = DistanceHypothesiser(
        kalman_predictor, kalman_updater, Mahalanobis(), missed_distance=100
    )
    hypothesiser = GaussianMixtureHypothesiser(
        base_hypothesiser, order_by_detection=True
    )

    # Initialise a Gaussian Mixture reducer
    reducer = GaussianMixtureReducer(
        prune_threshold=prune_threshold,
        pruning=True,
        merge_threshold=merge_threshold,
        merging=True,
    )

    # tracks are initially empty because we don't know positions
    reduced_states = set()

    birth_covar_diag = [simulator.window_width**2] * 2
    for sigma in simulator.sigma_initial_state:
        birth_covar_diag += [sigma**2] * 2
    birth_covar = np.diag(birth_covar_diag)
    birth_component = TaggedWeightedGaussianState(
        state_vector=np.array([0, 0, 0, 0]),
        covar=birth_covar,
        weight=simulator.p_birth,
        tag="birth",
        timestamp=datetime.fromtimestamp(0),
    )

    all_gaussians = []
    tracks_by_time: List[List[TaggedWeightedGaussianState]] = []
    for n, d in enumerate(data):
        _, sensor_positions, measurements, clutter, _ = d
        tracks_by_time.append([])
        all_gaussians.append([])
        time = datetime.fromtimestamp(n * simulator.dt)

        # The hypothesiser takes in the current state of the Gaussian mixture. This is equal to the list of
        # reduced states from the previous iteration. If this is the first iteration, then we use the priors
        # defined above.
        current_state = reduced_states

        measurement_set = set()
        for sensor, model, sensor_measurements, sensor_clutter in zip(
            sensor_positions, measurement_models, measurements, clutter
        ):
            for m in sensor_measurements:
                measurement_set.add(
                    Detection(
                        state_vector=model.function(State(StateVector(m)), noise=False),
                        timestamp=time,
                        measurement_model=model,
                    )
                )
            for c in sensor_clutter:
                measurement_set.add(
                    Clutter(
                        state_vector=model.function(State(StateVector(c)), noise=False),
                        timestamp=time,
                        measurement_model=model,
                    )
                )

        # At every time step we must add the birth component to the current state
        birth_component.timestamp = time
        current_state.add(birth_component)

        # Generate the set of hypotheses
        hypotheses = hypothesiser.hypothesise(
            current_state,
            measurement_set,
            timestamp=time,
            # keep our hypotheses ordered by detection, not by track
            order_by_detection=True,
        )

        # Turn the hypotheses into a GaussianMixture object holding a list of states
        updated_states = updater.update(hypotheses)

        # Prune and merge the updated states into a list of reduced states
        reduced_states = set(reducer.reduce(updated_states))

        # Add the reduced states to the track list. Each reduced state has a unique tag. If this tag matches the tag of a
        # state from a live track, we add the state to that track. Otherwise, we generate a new track if the reduced
        # state's weight is high enough (i.e. we are sufficiently certain that it is a new track).
        for reduced_state in reduced_states:
            # Add the reduced state to the list of Gaussians that we will plot later. Have a low threshold to eliminate some
            # clutter that would make the graph busy and hard to understand
            if reduced_state.weight > 0.05:
                all_gaussians[n].append(reduced_state)
            # Here we check to see if the state has a sufficiently high weight to consider being added.
            if reduced_state.weight > state_threshold:
                tracks_by_time[n].append(reduced_state)
    return tracks_by_time


def positions_from_gmphd(
    predictions: List[TaggedWeightedGaussianState], n_detections: float
) -> npt.NDArray[np.floating]:
    """Extract the positions from the PHD predictions.
    Args:
        predictions: The predictions from the PHD filter.
        n_detections: The expected number of detections.
    """
    # don't mutate the original list
    predictions = copy(predictions)
    # sort mixture by weight
    predictions.sort(key=lambda x: x.weight, reverse=True)
    # since there are multiple detections per sensor
    # the PHD intensity L1 norm should be will n_detection times the cardinality
    cardinality = int(
        np.round(np.sum([p.weight.real for p in predictions]) / n_detections)
    )
    # get the most likely states
    mu = np.array([p.state_vector[:2, 0] for p in predictions[:cardinality]])
    return mu.reshape(-1, 2)
