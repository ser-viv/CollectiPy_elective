# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectiPy, released under the BSD 3-Clause License.
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
        dynamics: str = 'metropolis',
        max_perceived_agents: int = 4
    ):
        """Initialize the flocking spin system instance."""
        self.random_generator = random_generator
        self.num_groups = num_groups
        self.num_spins_per_group = num_spins_per_group
        self.max_perceived_agents = max_perceived_agents
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
        external_field_contribution = -np.dot( self.external_field, state_pm1_flat)

        #print("H_spin_interactions,external_field_contribution",H_spin_interactions,external_field_contribution)
        # global inhibition term: -hb Σ σi
        global_inhibition_contribution = self.global_inhibition * np.sum(state_pm1_flat)

        print("H_spin_interactions,external_field_contribution",H_spin_interactions,external_field_contribution)
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

    def _get_agent_sectors(self, agent_channel):
        """
        Given the binary agent channel (length = num_groups * num_spins_per_group),
        aggregates per group and returns a list of contiguous blobs, each blob being
        a list of consecutive group indices where at least one agent is perceived.
        Handles circular wrap-around (e.g. a blob spanning groups 28,29,0,1).

        Example with num_groups=15:
            channel → [0,1,0,0,1,1,0,0,0,1,1,1,1,0,0]  (one value per group)
            returns  → [[1], [4,5], [9,10,11,12]]
        """
        n = self.num_groups

        # Step 1: collapse num_spins_per_group entries into one boolean per group
        active = []
        for g in range(n):
            start = g * self.num_spins_per_group
            end   = start + self.num_spins_per_group
            if np.any(np.asarray(agent_channel[start:end]) > 0):
                active.append(g)

        if not active:
            return []

        # Step 2: find contiguous runs (linear)
        groups = []
        current = [active[0]]
        for i in range(1, len(active)):
            if active[i] == active[i - 1] + 1:
                current.append(active[i])
            else:
                groups.append(current)
                current = [active[i]]
        groups.append(current)

        # Step 3: merge wrap-around (last group ends at n-1, first group starts at 0)
        if len(groups) > 1 and groups[-1][-1] == n - 1 and groups[0][0] == 0:
            merged = groups[-1] + groups[0]   # angularly contiguous across 0
            groups = [merged] + groups[1:-1]

        return groups  # list of lists of group indices

    def update_edge_field(self, agent_channel, edge_weight):
        """
        Positive contribution toward the two edge sectors of each perceived agent blob.
        Edge sectors are the first and last group index of each contiguous blob.
        Contribution is proportional to the blob's angular width (blob_length / num_groups),
        scaled by J so that edge_weight lives on the same scale as the spin coupling.
        """
        if agent_channel is None:
            return

        blobs = self._get_agent_sectors(agent_channel)
        if not blobs:
            return
        #print("lunghezza blobs attrazione",len(blobs))

        TWO_PI       = 2.0 * math.pi
        sector_width = TWO_PI / self.num_groups          # angular size of one group
        
        eff_weight   = edge_weight#* self.J           # scale relative to J
        edge_field = np.zeros(self.num_groups * self.num_spins_per_group, dtype=np.float32)
        print("edge_weight,eff_weight,J",edge_weight,eff_weight,self.J)
        for blob in clusters:
            n_body = len(blob)
            blob_angular_width = len(blob) * sector_width          # in radians
            contribution = eff_weight * blob_angular_width / TWO_PI# / 2  # = eff_weight * len(blob)/num_groups

            
            
            edge_sectors = [blob[0]] if len(blob) == 1 else [blob[0], blob[-1]]
            for g in edge_sectors:
                start = g * self.num_spins_per_group
                end   = start + self.num_spins_per_group
                edge_field[start:end] += contribution
            print("edge attraction contribution", edge_field)

        #print("edge attraction contribution", edge_field)
        self.external_field = self.external_field + edge_field

    def update_body_repulsion_field(self, agent_channel, repulsion_weight):
        """
        Negative contribution from the inner body sectors of each perceived agent blob.
        Body sectors are all group indices between the two edges (excluded).
        A blob must occupy >= 3 groups to have any body.
        Contribution is per-blob (not per-sector), distributed evenly across body sectors,
        so that the total repulsion per agent doesn't grow with the number of sectors.
        Scaled by J for consistency with edge_weight.
        """
        if agent_channel is None:
            return

        blobs = self._get_agent_sectors(agent_channel)
        if not blobs:
            return

        print("lunghezza blobs repulsione",len(blobs))
        TWO_PI       = 2.0 * math.pi
        sector_width = TWO_PI / self.num_groups
        eff_weight   = repulsion_weight#* self.J
        print("edge_weight,eff_weight,J",repulsion_weight,eff_weight,self.J)
        repulsion_field = np.zeros(self.num_groups * self.num_spins_per_group, dtype=np.float32)

        for blob in blobs:
            if len(blob) <= 2:
                continue                                   # no inner body
            
            print("blobs repulsione",blob)
            body_sectors = blob[1:-1]
            #body_sectors = blob
            n_body       = len(body_sectors)
            print("blobs repulsione dopo aver levato esterni",body_sectors)
            blob_angular_width = len(blob) * sector_width
            # Total contribution for this blob, divided evenly across body sectors
            print("eff_weight, blob_angular_width, n_body",eff_weight,blob_angular_width/TWO_PI,n_body)

            contribution = eff_weight * blob_angular_width / TWO_PI# / n_body
            
            for g in body_sectors:
                start = g * self.num_spins_per_group
                end   = start + self.num_spins_per_group
                repulsion_field[start:end] += contribution
            print("body repulsion contribution", repulsion_field)

        #print("body repulsion contribution", repulsion_field)
        self.external_field = self.external_field - repulsion_field


    def _arena_step_repulsion_level(self,d_ratio):
        """
        Step function: distance expressed as multiples of agent diameter.
        Returns the repulsion magnitude level (0 to 10^5).
        """
        if d_ratio > 6:
            return 0
        elif d_ratio > 5:
            return 10
        elif d_ratio > 4:
            return 100
        elif d_ratio > 3:
            return 1000
        elif d_ratio > 2:
            return 10000
        else:  # d_ratio <= 2
            return 100000


    def update_arena_repulsion_field(self, arena_metadata,
                                   agent_diameter):
        """
        Add a negative contribution to the external field for each spin whose
        angular sector overlaps a visible segment of the arena boundary.
        The contribution magnitude follows a step function based on distance,
        expressed in multiples of agent_diameter:
            > 6 diam  -> 0
            5-6 diam  -> 10
            4-5 diam  -> 100
            3-4 diam  -> 1000
            2-3 diam  -> 10000
            <= 2 diam -> 100000
        The angular_width weighting is kept to distribute the contribution
        correctly across overlapping sectors.
        Args:
            arena_metadata:          list of dicts from visual.py _collect_arena_boundary.
                                      Required keys: "angle" (rad), "angular_width" (rad), "distance"
            arena_repulsion_weight:  scaling factor applied on top of the step level (configurable from JSON)
            agent_diameter:          diameter of the agent, used to express distance in body-lengths
        """
        if not arena_metadata:
            return
        TWO_PI = 2.0 * math.pi
        repulsion_field = np.zeros(self.num_groups * self.num_spins_per_group,
                                   dtype=np.float32)
        sector_hw = _PI / self.num_groups

        for seg in arena_metadata:
            seg_center = seg["angle"] % TWO_PI
            seg_hw     = seg["angular_width"] / 2.0
            dist       = max(seg.get("distance", 0.0), 0.0)

            d_ratio = dist / agent_diameter
            level = self._arena_step_repulsion_level(d_ratio)

            seg_contribution = (level
                                 * (seg["angular_width"] / TWO_PI))

            seg_min = (seg_center - seg_hw) % TWO_PI
            seg_max = (seg_center + seg_hw) % TWO_PI
            seg_wraps = seg_min > seg_max
            for g in range(self.num_groups):
                sec_center = self.angles[g * self.num_spins_per_group]
                sec_min = (sec_center - sector_hw) % TWO_PI
                sec_max = (sec_center + sector_hw) % TWO_PI
                sec_wraps = sec_min > sec_max
                if not seg_wraps and not sec_wraps:
                    intersects = not (sec_max < seg_min or sec_min > seg_max)
                elif seg_wraps and not sec_wraps:
                    intersects = (sec_max >= seg_min) or (sec_min <= seg_max)
                elif not seg_wraps and sec_wraps:
                    intersects = (seg_max >= sec_min) or (seg_min <= sec_max)
                else:
                    intersects = True
                if intersects:
                    start = g * self.num_spins_per_group
                    end   = start + self.num_spins_per_group
                    repulsion_field[start:end] += seg_contribution

        #print("repulsion_field arena", repulsion_field)
        #print("repulsion_field arena",repulsion_field)
        self.external_field = self.external_field - repulsion_field

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