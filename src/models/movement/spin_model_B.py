"""
SpinMovementModelB - movement model wrapper that uses SpinSystemB and
implements the velocity / heading update following Bastien & Romanczuk (vision-based model).
The continuous equations used as reference are Eq.3 (speed) and Eq.4 (heading) of the paper.
(See vision_based_collective_behavior.pdf)
"""

import logging
import math
import numpy as np
from plugin_base import MovementModel
from plugin_registry import register_movement_model, get_detection_model
from models.utils import normalize_angle
from models.spinsystem_B import SpinSystemB

logger = logging.getLogger("sim.spin_B")

class SpinMovementModelB(MovementModel):
    """Spin movement model B: uses SpinSystemB and implements paper's motor update."""

    def __init__(self, agent):
        self.agent = agent
        self.spin_model_params = agent.config_elem.get("spin_model", {})

        # timing / spin execution
        self.spin_pre_run_steps = self.spin_model_params.get("spin_pre_run_steps", 0)
        self.spin_per_tick = self.spin_model_params.get("spin_per_tick", 10)
        self.perception_width = self.spin_model_params.get("perception_width", 0.5)

        # ring / discretization
        self.num_groups = self.spin_model_params.get("num_groups", 16)
        self.num_spins_per_group = self.spin_model_params.get("num_spins_per_group", 8)
        self.perception_global_inhibition = self.spin_model_params.get("perception_global_inhibition", 0)

        # task & reference
        agent_task = agent.get_task() if hasattr(agent, "get_task") else None
        spin_task = self.spin_model_params.get("task")
        self.task = agent_task or spin_task or "selection"
        if agent_task is None and hasattr(agent, "set_task"):
            agent.set_task(self.task)
        self.reference = self.spin_model_params.get("reference", "egocentric")
        self.fallback_behavior = agent.config_elem.get("fallback_moving_behavior", "none")

        # angles used to approximate the integrals
        self.group_angles = np.linspace(0.0, 2.0 * math.pi, self.num_groups, endpoint=False)

        # state
        self.perception = None
        self._active_perception_channel = "objects"
        self.perception_range = self._resolve_detection_range()
        self.spin_system = None
        self._fallback_model = None
        self.detection_model = self._create_detection_model()



        # --- parameters coming from the paper / defaults ---
        self.gamma = float(self.spin_model_params.get("gamma", 0.5))         # relaxation for speed
        self.v0 = float(self.spin_model_params.get("v0", 0.03))  # preferred speed REMOVED  getattr(self.agent, "max_absolute_velocity",
        self.alpha0 = float(self.spin_model_params.get("alpha0", 0.01))      # coeff for acceleration integral
        self.beta0 = float(self.spin_model_params.get("beta0", 0.01))        # coeff for turning integral
        self.alpha1 = float(self.spin_model_params.get("alpha1", 0.5))      # coeff for acceleration integral
        self.beta1 = float(self.spin_model_params.get("beta1", 0.5))        # coeff for turning integral
        self.dt = float(self.spin_model_params.get("dt", 0.1))              # discrete timestep for Euler update (per tick)
        self.angular_noise_std = float(self.spin_model_params.get("angular_noise_std", 0.05))  # optional noise (rad)
        # ----------------------------------------------------

        # internal continuous states (we store them between ticks)
        # initialize speed/state with agent current if present ???
        self._v = float(getattr(self.agent, "linear_velocity_cmd", 0.0))
        self._psi = math.radians(getattr(self.agent, "orientation", getattr(self.agent, "orientation", 0.0)).z) if hasattr(self.agent, "orientation") else 0.0

        self.reset()

    def _create_detection_model(self):
        context = {
            "num_groups": self.num_groups,
            "num_spins_per_group": self.num_spins_per_group,
            "perception_width": self.perception_width,
            "group_angles": self.group_angles,
            "reference": self.reference,
            "perception_global_inhibition": self.perception_global_inhibition,
            "max_detection_distance": self.perception_range,
            "detection_config": getattr(self.agent, "detection_config", {}),
        }
        detection_name = getattr(self.agent, "detection", None)
        if not detection_name:
            detection_name = self.agent.config_elem.get("detection", "VISUAL")
        return get_detection_model(detection_name, self.agent, context)

    def reset(self) -> None:
        self.perception = None
        self.spin_system = SpinSystemB(
            self.agent.random_generator,
            self.num_groups,
            self.num_spins_per_group,
            float(self.spin_model_params.get("T", 0.5)),
            float(self.spin_model_params.get("J", 1.0)),
            float(self.spin_model_params.get("nu", 0.0)),
            float(self.spin_model_params.get("p_spin_up", 0.5)),
            int(self.spin_model_params.get("time_delay", 1)),
            self.spin_model_params.get("dynamics", "metropolis"),
        )

        # initialize continuous states from agent if possible ???
        self._v = float(getattr(self.agent, "linear_velocity_cmd", 0.0))
        # ensure psi is in radians
        if hasattr(self.agent, "orientation"):
            try:
                self._psi = math.radians(self.agent.orientation.z)
            except Exception:
                self._psi = 0.0 

    def pre_run(self, objects: dict, agents: dict) -> None:
        if self.spin_pre_run_steps <= 0:
            return
        self._update_perception(objects, agents, None, None)
        if self.perception is None:
            return
        for _ in range(self.spin_pre_run_steps):
            # warm up without delay
            self.spin_system.step(timedelay=False)
        # keep behaviour similar to original: tune p_spin_up
        self.spin_system.set_p_spin_up(np.mean(self.spin_system.get_states()))
        self.spin_system.reset_spins()
        logger.debug("%s spin pre-run completed (%d steps)", self.agent.get_name(), self.spin_pre_run_steps)

    def step(self, agent, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        """
        Main step: implementazione Eq. 3 e 4 del paper con scalamento temporale e fisico.
        """
        
        self._update_perception(objects, agents, tick, arena_shape)
        if self.perception is None:
            self._run_fallback(tick, arena_shape, objects, agents)
            return

        
        V_raw = np.asarray(self.perception, dtype=float)
        if V_raw.size == self.num_groups * self.num_spins_per_group:
            V = V_raw.reshape(self.num_groups, self.num_spins_per_group).mean(axis=1)
        elif V_raw.size == self.num_groups:
            V = V_raw
        else:
            
            V = np.linspace(V_raw.min(), V_raw.max(), self.num_groups)

        V = np.clip(V, 0.0, 1.0) 

       
        dphi = 2.0 * math.pi / self.num_groups
        
        edges = (np.roll(V, -1) + V + np.roll(V, 1)) / 3.0
        
        dV_energy = edges / (dphi)

        # Calcolo Integrali
        accel_integrand = -V + (self.alpha1 * dV_energy)
        turn_integrand = -V + (self.beta1 * dV_energy)

        print("accel_integrand")
        print(accel_integrand)
        print("dphi")
        print(dphi)
        
        # For allocentric reference, angles are in world frame
        # We need to integrate relative to agent's heading
        if self.reference == "allocentric":
            # Get agent heading in radians
            psi = math.radians(self.agent.orientation.z)
            # Integrate with cos/sin(φ - ψ) to get relative angles
            accel_integral = np.sum(np.cos(self.group_angles - psi) * accel_integrand) * dphi
            turn_integral = np.sum(np.sin(self.group_angles - psi) * turn_integrand) * dphi
        else:
            # Egocentric: angles already relative to agent, integrate normally
            accel_integral = np.sum(np.cos(self.group_angles) * accel_integrand) * dphi
            turn_integral = np.sum(np.sin(self.group_angles) * turn_integrand) * dphi


        print("accel_integral")
        print(accel_integral)

        turning_slowdown = 1.0 - (abs(turn_integral) * 0.2) 
        self._v *= np.clip(turning_slowdown, 0.5, 1.0)


        social_force_scale = 1.0 
        
        print("correction")
        print(self.gamma * (self.v0 - self._v))
        print("paper velocity correction")
        print(self.alpha0 * accel_integral * social_force_scale)
        dv = self.gamma * (self.v0 - self._v) + (self.alpha0 * accel_integral * social_force_scale)
        #dv = self.gamma * (self.v0 - self._v) + self.alpha0 * accel_integral
        self._v += dv * self.dt # Integrazione di Eulero
        print("self._v ")
        print(self._v)
        
        self._v = np.clip(self._v, 0.001, 0.06) 
        print("self._v AFTER CLIP")
        print(self._v)

        
        dpsi_rad_sec = self.beta0 * turn_integral*social_force_scale
        # turn_velocity_deg è in gradi/tick. 
        turn_velocity_deg = math.degrees(dpsi_rad_sec) * self.dt


        # Calcoliamo la distanza dal centro dell'arena (assumendo centro in 0,0)
        arena_center = arena_shape.center 
        dx_c = arena_center.x - self.agent.position.x
        dy_c = arena_center.y - self.agent.position.y
        dist_from_center = math.hypot(dx_c, dy_c)
        
        # Calcoliamo il raggio dell'arena dai suoi vertici
        arena_radius = (arena_shape.max_vert().x - arena_shape.min_vert().x) / 2.0

        if dist_from_center > (arena_radius * 0.75): # Inizia a girare al 75% del raggio
            # Calcola l'angolo verso il centro
            angle_to_center = math.degrees(math.atan2(dy_c, dx_c))
            # Invertiamo il segno di dy_c se il simulatore ha Y invertita (CollectiPy la ha)
            angle_to_center = math.degrees(math.atan2(-dy_c, dx_c))
            
            diff_angle = normalize_angle(angle_to_center - self.agent.orientation.z)
            
            # Forza la sterzata verso il centro (peso crescente man mano che ci si avvicina al bordo)
            danger = (dist_from_center - arena_radius * 0.75) / (arena_radius * 0.25)**2
            danger = np.clip(danger, 0, 1)
            turn_velocity_deg = (turn_velocity_deg * (1 - danger)) + (diff_angle * danger)
            self._v *= (1.0 - danger * 0.5) # Rallenta per curvare meglio


        # Applicazione comandi
        self.agent.linear_velocity_cmd = self._v
        self.agent.angular_velocity_cmd = np.clip(turn_velocity_deg, 
                                                 -self.agent.max_angular_velocity, 
                                                 self.agent.max_angular_velocity)

        
        self.spin_system.update_external_field(V)
        self.spin_system.run_spins(steps=1)

    # --- perception helpers (copied logic from original model) ---
    def _update_perception(self, objects: dict, agents: dict, tick: int | None = None, arena_shape=None) -> None:
        if self.detection_model is None:
            self.perception = None
            return
        if tick is not None and hasattr(self.agent, "should_sample_detection"):
            if not self.agent.should_sample_detection(tick):
                return
        snapshot = self.detection_model.sense(self.agent, objects, agents, arena_shape)
        if snapshot is None:
            self.perception = None
            return
        if isinstance(snapshot, dict):
            selected, channel_name = self._select_perception_channel(snapshot)
        else:
            selected, channel_name = snapshot, "raw"
        self.perception = selected
        self._active_perception_channel = channel_name
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s perception snapshot channel=%s mean=%.3f",
                self.agent.get_name(),
                channel_name,
                float(np.mean(self.perception)) if self.perception is not None else 0.0,
            )

    def _select_perception_channel(self, snapshot: dict):
        task_name = (self.agent.get_task() or self.task or "selection").lower()
        objects_channel = snapshot.get("objects")
        agents_channel = snapshot.get("agents")
        combined_channel = snapshot.get("combined")
        if task_name in ("selection", "objects"):
            return self._channel_with_fallback(
                (objects_channel, "objects"),
                (combined_channel, "combined"),
                (agents_channel, "agents"),
            )
        if task_name in ("flocking", "agents"):
            return self._channel_with_fallback(
                (agents_channel, "agents"),
                (combined_channel, "combined"),
                (objects_channel, "objects"),
            )
        if task_name in ("all", "combined", "hybrid", "group", "group_hunt"):
            return self._channel_with_fallback(
                (combined_channel, "combined"),
                (objects_channel, "objects"),
                (agents_channel, "agents"),
            )
        logger.warning("%s unknown task '%s', using combined perception", self.agent.get_name(), task_name)
        return self._channel_with_fallback(
            (combined_channel, "combined"),
            (objects_channel, "objects"),
            (agents_channel, "agents"),
        )

    def _channel_with_fallback(self, primary, secondary, tertiary):
        for channel, name in (primary, secondary, tertiary):
            if channel is not None:
                return channel, name
        raise ValueError("Detection model did not provide any perception channels")

    def _resolve_detection_range(self) -> float:
        if hasattr(self.agent, "get_detection_range"):
            try:
                return float(self.agent.get_detection_range())
            except (TypeError, ValueError):
                logger.warning("%s provided invalid detection range; falling back", self.agent.get_name())
        config_elem = getattr(self.agent, "config_elem", {})
        settings = {}
        if isinstance(config_elem, dict):
            settings = config_elem.get("detection_settings", {}) or {}
        range_candidate = None
        if isinstance(settings, dict):
            range_candidate = settings.get("range", settings.get("distance"))
        if range_candidate is None and isinstance(config_elem, dict):
            range_candidate = config_elem.get("perception_distance")
        if range_candidate is None and hasattr(self.agent, "perception_distance"):
            range_candidate = self.agent.perception_distance
        if range_candidate is None:
            return math.inf
        try:
            return float(range_candidate)
        except (TypeError, ValueError):
            logger.warning("%s invalid detection range '%s', using infinite distance", self.agent.get_name(), range_candidate)
            return math.inf

    def _run_fallback(self, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        if self.fallback_behavior in ("spin_model", "none"):
            return
        behavior = self.fallback_behavior
        if self._fallback_model is None:
            from plugin_registry import get_movement_model
            self._fallback_model = get_movement_model(self.fallback_behavior, self.agent)
            if self._fallback_model is None and self.fallback_behavior != "random_walk":
                self._fallback_model = get_movement_model("random_walk", self.agent)
                behavior = "random_walk"
        if self._fallback_model is None:
            logger.warning("%s has no fallback movement model configured", self.agent.get_name())
            return
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("%s fallback to %s", self.agent.get_name(), behavior)
        self._fallback_model.step(self.agent, tick, arena_shape, objects, agents)

    def get_spin_system_data(self):
        if not self.spin_system:
            return None
        return (
            self.spin_system.get_states(),
            self.spin_system.get_angles(),
            self.spin_system.get_external_field(),
            self.spin_system.get_avg_direction_of_activity(),
        )

# register model
register_movement_model("spin_model_B", lambda agent: SpinMovementModelB(agent))
