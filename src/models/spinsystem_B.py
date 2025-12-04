"""
SpinSystemB - drop-in compatible spin system for the new visual input.

Compatibility notes:
- constructor arguments and public methods mirror the original SpinSystem
  used by the project (states, angles, update_external_field, step, etc.).
- update_external_field accepts:
    * flattened vector of length num_groups * num_spins_per_group (preferred),
    * or a per-group vector of length num_groups (it will be repeated per spin).
"""

import math
import numpy as np
from random import Random

_PI = math.pi

class SpinSystemB:
    """Spin system (Version B) compatible with the simulator ring-attractor."""

    def __init__(self, random_generator: Random,num_groups: int,num_spins_per_group: int,T: float,J: float,nu: float,p_spin_up: float = 0.5,time_delay: int = 1,dynamics: str = "metropolis",):
        
        self.random_generator = random_generator
        self.num_groups = int(num_groups)
        self.num_spins_per_group = int(num_spins_per_group)
        self.T = float(T)
        self.J = float(J)
        self.nu = float(nu)
        self.p_spin_up = float(p_spin_up)
        self.history_length = int(time_delay)
        self.dynamics = str(dynamics)

        # initialize spins matrix: shape (num_groups, num_spins_per_group)
        self.spins = self._random_spins()
        self.spins_history = [self.spins.copy()]

        # precompute angles: one angle per spin (flattened order)
        group_angles = np.linspace(0.0, 2.0 * _PI, self.num_groups, endpoint=False)
        self.angles = np.repeat(group_angles, self.num_spins_per_group)
        # external field stored in flattened format (length G * N)
        self.external_field = np.zeros(self.num_groups * self.num_spins_per_group, dtype=np.float32)

        # auxiliary precomputed interaction matrix (depends on nu)
        self.J_matrix = self._precompute_j_matrix()

        # cached average direction
        self.avg_direction = None

    # -----------------------
    # Initialization helpers
    # -----------------------
    def _random_spins(self):
        size = self.num_groups * self.num_spins_per_group
        rng = self.random_generator
        rand_vals = np.array([Random.uniform(rng, 0.0, 1.0) for _ in range(size)])
        spins = (rand_vals < self.p_spin_up).astype(np.uint8)
        return spins.reshape(self.num_groups, self.num_spins_per_group)

    def _precompute_j_matrix(self):
        angle_diff_matrix = np.abs(np.subtract.outer(self.angles, self.angles))
        angle_diff_matrix = np.minimum(angle_diff_matrix, 2 * _PI - angle_diff_matrix)
        # same functional form as original system
        return np.cos(_PI * ((angle_diff_matrix / _PI) ** self.nu))

    # -----------------------
    # External field handling
    # -----------------------
    def update_external_field(self, perceptual_outputs):
        """
        Accepts:
          - flattened vector length G * N (preferred)
          - or per-group vector length G (will be repeated per spin)
        Stores internal external_field flattened array.
        """
        arr = np.asarray(perceptual_outputs, dtype=np.float32)

        expected_flat = self.num_groups * self.num_spins_per_group
        if arr.size == expected_flat:
            self.external_field = arr.ravel().astype(np.float32)
            return

        if arr.size == self.num_groups:
            # repeat each group value for the spins inside the group
            self.external_field = np.repeat(arr.ravel(), self.num_spins_per_group).astype(np.float32)
            return

        raise ValueError(
            f"external_field size mismatch: got {arr.size}, expected {expected_flat} (flat) or {self.num_groups} (per-group)"
        )
    # -----------------------
    # Hamiltonian & dynamics
    # -----------------------
    def calculate_hamiltonian(self, state):
        """
        state: 2D array shape (num_groups, num_spins_per_group) with values 0/1
        Hamiltonian: interaction term + external field term
        """
        state_flat = state.ravel()
        interaction = state_flat[:, None] * state_flat[None, :]
        H_spin_interactions = -(self.J / (self.num_spins_per_group * self.num_groups)) * np.sum(
            np.triu(self.J_matrix * interaction, 1)
        )
        external_field_contribution = -np.dot(self.external_field, state_flat)
        return H_spin_interactions + external_field_contribution

    def _metropolis_acceptance(self, i_group, j_spin, delta_h):
        if delta_h <= 0 or Random.uniform(self.random_generator, 0.0, 1.0) < math.exp(-delta_h / self.T):
            self.spins[i_group, j_spin] ^= 1

    def _glauber_acceptance(self, i_group, j_spin, delta_h, dt=0.1, tau=33):
        G = self.num_groups
        N = self.num_spins_per_group
        acceptance_prob = (G * N * dt) / tau * (1.0 / (1.0 + math.exp(delta_h / self.T)))
        acceptance_prob = min(acceptance_prob, 1.0)
        if Random.uniform(self.random_generator, 0.0, 1.0) < acceptance_prob:
            self.spins[i_group, j_spin] ^= 1

    def step(self, timedelay: bool = True, dt: float = 0.1, tau: float = 33.0):
        """Perform a single spin update (same selection protocol as original)."""
        i = Random.randint(self.random_generator, 0, self.num_groups - 1)
        j = Random.randint(self.random_generator, 0, self.num_spins_per_group - 1)

        state_to_use = self.spins
        if timedelay and self.spins_history:
            hybrid_state = self.spins_history[0].copy()
            hybrid_state[i, j] = self.spins[i, j]
            state_to_use = hybrid_state

        current_h = self.calculate_hamiltonian(state_to_use)
        # flip candidate in state_to_use for energy calc
        state_to_use[i, j] ^= 1
        new_h = self.calculate_hamiltonian(state_to_use)
        delta_h = new_h - current_h

        if self.dynamics == "metropolis":
            self._metropolis_acceptance(i, j, delta_h)
        elif self.dynamics == "glauber":
            self._glauber_acceptance(i, j, delta_h, dt, tau)
        else:
            raise ValueError(f"Unknown dynamics: {self.dynamics}")

        # update history
        self.spins_history.append(self.spins.copy())
        if len(self.spins_history) > self.history_length:
            self.spins_history.pop(0)

    def run_spins(self, steps: int = 1, dt: float = 0.1, tau: float = 33.0):
        for _ in range(steps):
            self.step(timedelay=True, dt=dt, tau=tau)
        return self.spins

    # -----------------------
    # Observables / utilities
    # -----------------------
    def get_states(self):
        return self.spins

    def get_external_field(self):
        return self.external_field

    def get_angles(self):
        return (self.angles, self.num_groups, self.num_spins_per_group)

    def set_states(self, states):
        states = np.asarray(states)
        if states.shape != self.spins.shape:
            raise ValueError(f"Invalid shape for spin states: expected {self.spins.shape}, got {states.shape}")
        self.spins = states.copy()
        self.spins_history.append(self.spins.copy())
        if len(self.spins_history) > self.history_length:
            self.spins_history.pop(0)

    def reset_spins(self):
        self.spins = self._random_spins()
        self.spins_history = [self.spins.copy()]

    def average_direction_of_activity(self):
        flattened_spins = self.spins.ravel()
        # active defined as 1
        active_mask = flattened_spins == 1
        if not np.any(active_mask):
            self.avg_direction = None
            return None
        unit_vectors = np.exp(1j * self.angles)
        sum_vector = np.sum(unit_vectors[active_mask])
        if sum_vector == 0:
            self.avg_direction = None
            return None
        self.avg_direction = math.atan2(sum_vector.imag, sum_vector.real)
        return self.avg_direction

    def get_avg_direction_of_activity(self):
        return self.avg_direction

    def get_width_of_activity(self):
        flattened_spins = self.spins.ravel()
        active_mask = flattened_spins == 1
        active_angles = self.angles[active_mask]
        if len(active_angles) > 1:
            unit_vectors = np.exp(1j * active_angles)
            R = abs(np.mean(unit_vectors))
            if R > 0:
                circ_std = math.sqrt(-2.0 * math.log(R))
                return circ_std
        return None

    def get_inverse_magnitude_of_activity(self):
        flattened_spins = self.spins.ravel()
        active_mask = flattened_spins == 1
        if not np.any(active_mask):
            return float("inf")
        unit_vectors = np.exp(1j * self.angles)
        sum_vector = np.sum(unit_vectors[active_mask])
        mag = abs(sum_vector)
        return 1.0 / mag if mag != 0 else float("inf")

    def sense_other_ring(self, other_ring_states, gain=1.0):
        self.external_field = gain * np.asarray(other_ring_states, dtype=np.float32).ravel()