# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Spin system model for flocking behaviours."""
from __future__ import annotations

import math
import numpy as np
from random import Random
from models.utility_functions import normalize_angle

_PI = math.pi

class SpinModule:
    """Spin module specialized for flocking behaviors."""
    def __init__(
        self,
        random_generator: Random,
        num_groups: int,
        num_spins_per_group: int,
        T: float,
        J: float,
        nu: float,
        global_inhibition: float = 0.0,
        p_spin_up: float = 0.5,
        time_delay: int = 1,
        dynamics: str = 'metropolis'
    ):
        """Initialize the flocking spin system instance."""
        self.random_generator = random_generator
        self.num_groups = num_groups
        self.num_spins_per_group = num_spins_per_group
        self.T = T
        self.J = J
        self.nu = nu
        self.p_spin_up = p_spin_up
        self.spins = self._random_spins()
        self.spins_history = [self.spins.copy()]
        self.history_length = time_delay
        self.dynamics = dynamics
        self.global_inhibition = global_inhibition
        group_angles = np.linspace(0, 2 * _PI, num_groups, endpoint=False)
        self.angles = np.repeat(group_angles, self.num_spins_per_group)
        self._unit_angle_vectors = np.exp(1j * self.angles)
        self.external_field = np.zeros(self.num_groups * self.num_spins_per_group, dtype=np.float32)
        self.avg_direction = None
        self.J_matrix = self._precompute_j_matrix()
        self._J_upper = np.triu(self.J_matrix, 1)
        self._interaction_factor = -(self.J / (self.num_spins_per_group * self.num_groups))

    def _random_spins(self):
        """Generate random spins."""
        rng_random = self.random_generator.random
        rand_vals = np.fromiter(
            (rng_random() for _ in range(self.num_groups * self.num_spins_per_group)),
            dtype=np.float32,
            count=self.num_groups * self.num_spins_per_group
        )
        spins = (rand_vals < self.p_spin_up).astype(np.uint8)
        return spins.reshape(self.num_groups, self.num_spins_per_group)

    def _to_pm1(self, state):
        """
        Convert internal state 0/1 to ±1 for Hamiltonian calculation.
        0 → -1 (inactive)
        1 → +1 (active)
        This conversion is required because the Hamiltonian uses σ ∈ {-1, +1},
        not {0, 1}.
        """
        return 2 * state.astype(np.float32) - 1

    def _precompute_j_matrix(self):
        """Precompute coupling matrix."""
        angle_diff_matrix = np.abs(np.subtract.outer(self.angles, self.angles))
        angle_diff_matrix = np.minimum(angle_diff_matrix, 2 * _PI - angle_diff_matrix)
        return np.cos(_PI * ((angle_diff_matrix / _PI) ** self.nu))

    def set_p_spin_up(self, p_spin_up: float):
        """Set the probability of spin up."""
        self.p_spin_up = p_spin_up

    def calculate_hamiltonian(self, state):
        """
        Calculate the Hamiltonian H = -[1/Ns * Σ Jij σi σj + Σ hi σi - hb Σ σi]
        with σ ∈ {-1, +1} as in the Salahshour & Couzin model.
        The internal state is 0/1 (uint8); conversion to ±1 happens here.
        """
        state_pm1 = self._to_pm1(state)
        state_pm1_flat = state_pm1.ravel()

        # interaction term: -J/Ns * Σ_ij Jij σi σj
        interaction = state_pm1_flat[:, None] * state_pm1_flat[None, :]
        H_spin_interactions = self._interaction_factor * np.sum(self._J_upper * interaction)

        # external field term: -Σ hi σi
        external_field_contribution = -np.dot(self.external_field, state_pm1_flat)

        # global inhibition term: -hb Σ σi
        global_inhibition_contribution = self.global_inhibition * np.sum(state_pm1_flat)

        return H_spin_interactions + external_field_contribution - global_inhibition_contribution

    def step(self, timedelay=True, dt=0.1, tau=33):
        """Execute one simulation step of the spin dynamics."""
        rng_randint = self.random_generator.randint
        i = rng_randint(0, self.num_groups - 1)
        j = rng_randint(0, self.num_spins_per_group - 1)

        state_to_use = self.spins
        if timedelay:
            hybrid_state = self.spins_history[0].copy()
            hybrid_state[i, j] = self.spins[i, j]
            state_to_use = hybrid_state

        current_hamiltonian = self.calculate_hamiltonian(state_to_use)
        state_to_use[i, j] ^= 1
        new_hamiltonian = self.calculate_hamiltonian(state_to_use)
        delta_h = new_hamiltonian - current_hamiltonian

        if self.dynamics == 'metropolis':
            self._metropolis_acceptance(i, j, delta_h)
        elif self.dynamics == 'glauber':
            self._glauber_acceptance(i, j, delta_h, dt, tau)
        else:
            raise ValueError(f"Unknown dynamics type: {self.dynamics}")

        self.spins_history.append(self.spins.copy())
        if len(self.spins_history) > self.history_length:
            self.spins_history.pop(0)

    def _metropolis_acceptance(self, i, j, delta_h):
        """Metropolis acceptance criterion."""
        if delta_h <= 0 or self.random_generator.random() < math.exp(-delta_h / self.T):
            self.spins[i, j] ^= 1

    def _glauber_acceptance(self, i, j, delta_h, dt, tau):
        """Glauber acceptance criterion."""
        G = self.num_groups
        N = self.num_spins_per_group
        acceptance_prob = (G * N * dt) / tau * (1 / (1 + math.exp(delta_h / self.T)))
        acceptance_prob = min(acceptance_prob, 1.0)
        if self.random_generator.random() < acceptance_prob:
            self.spins[i, j] ^= 1

    def run_spins(self, steps=1, dt=0.1, tau=33):
        """Run the spin system for multiple steps."""
        for _ in range(steps):
            self.step(dt=dt, tau=tau)
        return self.spins

    def average_direction_of_activity(self):
        """
        Calculate the average direction of active spins (σ = +1).
        Returns angle in radians, or None if no active spins or all spins active.
        """
        flattened_spins = self.spins.ravel()
        if np.all(flattened_spins == 1):
            self.avg_direction = None
            return None

        active_mask = flattened_spins == 1
        if not np.any(active_mask):
            self.avg_direction = None
            return None

        sum_vector = np.sum(self._unit_angle_vectors[active_mask])
        self.avg_direction = None
        if sum_vector != 0:
            self.avg_direction = math.atan2(sum_vector.imag, sum_vector.real)

        return self.avg_direction

    def get_avg_direction_of_activity(self):
        """Return the average direction of activity."""
        return self.avg_direction

    def get_inverse_magnitude_of_activity(self):
        """Return the inverse magnitude of activity."""
        flattened_spins = self.spins.ravel()
        active_mask = flattened_spins == 1
        if not np.any(active_mask):
            return float('inf')

        sum_vector = np.sum(self._unit_angle_vectors[active_mask])
        magnitude = abs(sum_vector)
        return 1 / magnitude if magnitude != 0 else float('inf')

    def get_width_of_activity(self):
        """Return the width of activity distribution."""
        flattened_spins = self.spins.ravel()
        active_mask = flattened_spins == 1
        active_angles = self.angles[active_mask]

        if len(active_angles) > 1:
            unit_vectors = np.exp(1j * active_angles)
            R = abs(np.mean(unit_vectors))
            if R > 0:
                circ_std = math.sqrt(-2 * math.log(R))
                return circ_std
        return None

    def reset_spins(self):
        """Reset the spins to random initial state."""
        self.spins = self._random_spins()
        self.spins_history = [self.spins.copy()]

    def update_external_field(self, perceptual_outputs):
        """Update the external field from perception."""
        self.external_field = np.asarray(perceptual_outputs, dtype=np.float32)

    def update_edge_field(self, edge_counts, edge_weight=0.7):
        """
        Add a linear contribution proportional to the edge count to the
        external field already set by update_external_field.

        Final field = external_field + edge_weight * (edge_counts / max_edges)

        Args:
            edge_counts:  int32 array (channel_size,) from visual.py
            edge_weight:  weight of the contribution (configurable from JSON)
        """
        edge_field = np.asarray(edge_counts, dtype=np.float32)
        max_edges = edge_field.max()
        if max_edges > 0:
            edge_field = edge_field / max_edges
        self.external_field = self.external_field + edge_weight * edge_field

    '''def update_repulsion_field(self, agent_metadata, repulsion_weight=0.5, repulsion_range=0.3):
        """
        Add a negative contribution to the external field, inversely proportional
        to distance: the closer the neighbours, the more the field is inhibited.

        Contribution per neighbour i:
            field -= repulsion_weight * (1 / distance_i) * gaussian(αi, angle_i, sigma)

        The contribution becomes negligible for distances > repulsion_range.

        Args:
            agent_metadata:   list of dicts from visual.py (angle, distance, ...)
            repulsion_weight: repulsion intensity
            repulsion_range:  distance beyond which repulsion is negligible
        """
        TWO_PI = 2.0 * math.pi
        sigma = 2 * math.pi / self.num_groups  # Gaussian width = one group

        for ag in agent_metadata:
            distance = ag["distance"]
            angle = ag["angle"]

            if distance > repulsion_range:
                continue

            # contribution inversely proportional to distance
            strength = repulsion_weight / max(distance, 1e-6)

            # Gaussian centred on the neighbour's angle
            diff = self.angles - angle
            diff = (diff + math.pi) % TWO_PI - math.pi  # wrap to [-π, π]
            gaussian = np.exp(-0.5 * (diff / sigma) ** 2)

            self.external_field = self.external_field - strength * gaussian
        '''
    def get_states(self):
        """Return the current spin states."""
        return self.spins

    def get_external_field(self):
        """Return the external field."""
        return self.external_field

    def get_angles(self):
        """Return group angles information."""
        return (self.angles, self.num_groups, self.num_spins_per_group)

    def set_states(self, states):
        """Set the spin states."""
        if states.shape != self.spins.shape:
            raise ValueError(
                f"Invalid shape for spin states. Expected {self.spins.shape}, got {states.shape}."
            )
        self.spins = states.copy()
        self.spins_history.append(self.spins.copy())
        if len(self.spins_history) > self.history_length:
            self.spins_history.pop(0)

    def sense_other_ring(self, other_ring_states, gain=1.0):
        """Sense another ring and update perception based on coupling."""
        self.external_field = gain * np.asarray(other_ring_states, dtype=np.float32).ravel()