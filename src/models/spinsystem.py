# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Spin system model used by spin-based movement behaviours."""
import math
import numpy as np
from random import Random

_PI = math.pi

class SpinSystem:
    """Spin system."""
    
    def __init__(self, random_generator:Random, num_groups:int, num_spins_per_group:int, T:float, J:float, nu:float, p_spin_up:float=0.5, time_delay:int=1, dynamics:str='metropolis'):
        """Initialize the instance."""
        self.random_generator = random_generator
        self.num_groups = num_groups
        self.num_spins_per_group = num_spins_per_group
        # parametri interazione spin
        self.T = T
        # matrice interazione spin
        self.J = J
        self.nu = nu
        self.p_spin_up = p_spin_up
        # matrice spin
        self.spins = self._random_spins()
        self.spins_history = [self.spins.copy()]
        self.history_length = time_delay
        self.dynamics = dynamics
        group_angles = np.linspace(0, 2 * _PI, num_groups, endpoint=False)
        # angoli per ciascun spin
        self.angles = np.repeat(group_angles, self.num_spins_per_group)
        self.external_field = np.zeros(self.num_groups * self.num_spins_per_group, dtype=np.float32)
        self.avg_direction = None
        self.J_matrix = self._precompute_j_matrix()

    # dice quali spin sono attivi in base alla loro probabilità 
    def _random_spins(self):
        """Generate the spins."""
        rand_vals = np.array([Random.uniform(self.random_generator, 0, 1)
                              for _ in range(self.num_groups * self.num_spins_per_group)])
        spins = (rand_vals < self.p_spin_up).astype(np.uint8)
        return spins.reshape(self.num_groups, self.num_spins_per_group) # è una matrice

    # precomputa la matrice interazione
    def _precompute_j_matrix(self):
        """Precompute j matrix."""
        angle_diff_matrix = np.abs(np.subtract.outer(self.angles, self.angles))
        angle_diff_matrix = np.minimum(angle_diff_matrix, 2 * _PI - angle_diff_matrix)
        return np.cos(_PI * ((angle_diff_matrix / _PI) ** self.nu))

    def set_p_spin_up(self, p_spin_up:float):
        """Set the p spin up."""
        self.p_spin_up = p_spin_up

    # calcola l'energia totale del sistema tra i contributi di interazioni di spin e del campo esterno
    def calculate_hamiltonian(self, state):
        """Calculate hamiltonian."""
        state_flat = state.ravel()
        interaction = state_flat[:, None] * state_flat[None, :]
        # interazioni di spin
        H_spin_interactions = -(self.J / (self.num_spins_per_group * self.num_groups)) * np.sum(np.triu(self.J_matrix * interaction, 1))
        # contributo campo esterno
        external_field_contribution = -np.dot(self.external_field, state_flat)
        return H_spin_interactions + external_field_contribution

    # step di simulatione
    # 1 seleziona uno spin casuale
    # 2 aggiorna lo spin secondo la dinamica scelta: metropolis -> accetta il cambiamento con probabilità e^deltaH/T; glauber -> probabilità calcolata in funzione di dt e tau
    # 3 aggiorna la storia degli spin
    def step(self,timedelay=True, dt=0.1, tau=33):
        """Execute the simulation step."""
        i = Random.randint(self.random_generator, 0, self.num_groups - 1)
        j = Random.randint(self.random_generator, 0, self.num_spins_per_group - 1)
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

    # implementa la probabilità di accettazione dello spin secondo la dinamica scelta
    def _metropolis_acceptance(self, i, j, delta_h):
        """Metropolis acceptance."""
        if delta_h <= 0 or Random.uniform(self.random_generator, 0, 1) < math.exp(-delta_h / self.T):
            self.spins[i, j] ^= 1
    # implementa la probabilità di accettazione dello spin secondo la dinamica scelta
    def _glauber_acceptance(self, i, j, delta_h, dt, tau):
        """Glauber acceptance."""
        G = self.num_groups
        N = self.num_spins_per_group
        acceptance_prob = (G * N * dt) / tau * (1 / (1 + math.exp(delta_h / self.T)))
        acceptance_prob = min(acceptance_prob, 1.0)
        if Random.uniform(self.random_generator, 0, 1) < acceptance_prob:
            self.spins[i, j] ^= 1


    def run_spins(self, steps=1, dt=0.1, tau=33):
        """Run the spins."""
        for _ in range(steps):
            self.step(dt=dt, tau=tau)
        return self.spins

    # calcola la direzione media dei spin attivi usando vettori unitari complessi per orientamento collettivo
    def average_direction_of_activity(self):
        """Average direction of activity."""
        flattened_spins = self.spins.ravel()
        if np.all(flattened_spins == 1):
            self.avg_direction = None
            return None
        unit_vectors = np.exp(1j * self.angles)
        active_mask = flattened_spins == 1
        if not np.any(active_mask):
            self.avg_direction = None
            return None
        sum_vector = np.sum(unit_vectors[active_mask])
        self.avg_direction = None
        if sum_vector != 0: self.avg_direction = math.atan2(sum_vector.imag, sum_vector.real)
        return self.avg_direction

    
    def get_avg_direction_of_activity(self):
        """Return the avg direction of activity."""
        return self.avg_direction

    # misura quanto sono allineati di spin
    def get_inverse_magnitude_of_activity(self):
        """Return the inverse magnitude of activity."""
        flattened_spins = self.spins.ravel()
        unit_vectors = np.exp(1j * self.angles)
        active_mask = flattened_spins == 1
        if not np.any(active_mask):
            return float('inf')
        sum_vector = np.sum(unit_vectors[active_mask])
        magnitude = abs(sum_vector)
        return 1 / magnitude if magnitude != 0 else float('inf')

    # deviazione circolare delle direzioni attive(quantifica direzione) di spin
    def get_width_of_activity(self):
        """Return the width of activity."""
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

    # rigenera spin casuali
    def reset_spins(self):
        """Reset the spins."""
        self.spins = self._random_spins()
        self.spins_history = [self.spins.copy()]

    def update_external_field(self, perceptual_outputs):
        """Update external field."""
        self.external_field = np.asarray(perceptual_outputs, dtype=np.float32)

    # lettura degli stati
    def get_states(self):
        """Return the states."""
        return self.spins

    def get_external_field(self):
        """Return the external field."""
        return self.external_field

    # riporta gli angoli associati agli spin
    def get_angles(self):
        """Return the angles."""
        return (self.angles, self.num_groups, self.num_spins_per_group)

    # scrittura degli stati
    def set_states(self, states):
        """Set the states."""
        if states.shape != self.spins.shape:
            raise ValueError(f"Invalid shape for spin states. Expected {self.spins.shape}, but got {states.shape}.")
        self.spins = states.copy()
        self.spins_history.append(self.spins.copy())
        if len(self.spins_history) > self.history_length:
            self.spins_history.pop(0)

    # permette l'influenza di un altro gruppo di spin
    def sense_other_ring(self, other_ring_states, gain=1.0):
        """Sense the environment and update perception."""
        self.external_field = gain * np.asarray(other_ring_states, dtype=np.float32).ravel()
