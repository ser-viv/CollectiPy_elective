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
        self.gamma = float(self.spin_model_params.get("gamma", 1.0))         # relaxation for speed
        self.v0 = float(self.spin_model_params.get("v0", 1.0))  # preferred speed REMOVED  getattr(self.agent, "max_absolute_velocity",
        self.alpha0 = float(self.spin_model_params.get("alpha0", 1.0))      # coeff for acceleration integral
        self.beta0 = float(self.spin_model_params.get("beta0", 1.0))        # coeff for turning integral
        self.alpha1 = float(self.spin_model_params.get("alpha1", 1.0))      # coeff for acceleration integral
        self.beta1 = float(self.spin_model_params.get("beta1", 1.0))        # coeff for turning integral
        self.dt = float(self.spin_model_params.get("dt", 1.0))              # discrete timestep for Euler update (per tick)
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
        Main step:
         - update perception (selects channel as in original)
         - update spin_system external field and run spins
         - compute integrals approximating the paper's formulas and update v and psi (Euler)
         - publish linear_velocity_cmd and angular_velocity_cmd
        """
        print(">>> spin_model_B STEP CALLED FOR:", agent.get_name())

        print("STEP for agent:", self.agent.get_name())
        self._update_perception(objects, agents, tick, arena_shape)
                # costruzione del campo visivo per gruppi (φ)
        V = np.asarray(self.perception, dtype=float)
        
        
        #solo una verifica
        if V.size == self.num_groups * self.num_spins_per_group:
            V = V.reshape(self.num_groups, self.num_spins_per_group).mean(axis=1)
        elif V.size == self.num_groups:
            pass
        else:
            raise ValueError("Dimensione inattesa di perception")
        
        if self.perception is None:
            self._run_fallback(tick, arena_shape, objects, agents)
            return
        print("PERCEPTION MEAN =", np.mean(self.perception))

        # read bump angle from the ring 
        bump_angle = self.spin_system.average_direction_of_activity()  # radians or None 

        if bump_angle is not None:
            psi_error = (bump_angle - self._psi)% (2 * math.pi) - math.pi #differenza angolare minima normalizzato
            self._psi += self.dt * self.beta0 * psi_error # angolo corrente
        
        #---abbiamo angolo ring attractor---
        
        #V = np.clip(V, 0.0, None)   # visual field binario / somma oggetti
        
        dphi = 2.0 * math.pi / self.num_groups

        # crea un array grande quanto Visual field e applica le differenze finite centrali
        dV_dphi = np.zeros_like(V)
        dV_dphi[1:-1] = (V[2:] - V[:-2]) / (2 * dphi)
        dV_dphi[0] = (V[1] - V[-1]) / (2 * dphi)
        dV_dphi[-1] = (V[0] - V[-2]) / (2 * dphi)



        #---QUESTA SEZIONE SERVE PER DEFINIRE VK PER AGGIUGNERE SMOOTHING ESPONENZIALE, MODIFICA DI CONSEGUENZA V CON VK ------
        #Vk = V.copy()    # visual field binario copia
        # OPTIONAL: smoothing esponenziale (component-wise) per evitare scatti tick-to-tick
        #ema_alpha = 1.0
        #if not hasattr(self, "_vk_ema") or self._vk_ema is None:
        #    self._vk_ema = Vk.copy()
        #else:
        #    self._vk_ema = ema_alpha * Vk + (1.0 - ema_alpha) * self._vk_ema
        #Vk = self._vk_ema.copy()

        # non è scritto bene ma memorizza l'external field
        #self.spin_system.update_external_field(Vk)
        self.spin_system.run_spins(steps=1)
        
        # --- derivative (dV/dphi) su Vk normalizzato ---
        
        #dV_dphi = np.zeros_like(Vk)
        #if Vk.size >= 3:
        #    dV_dphi[1:-1] = (Vk[2:] - Vk[:-2]) / (2 * dphi)
        #    dV_dphi[0] = (Vk[1] - Vk[-1]) / (2 * dphi)
        #    dV_dphi[-1] = (Vk[0] - Vk[-2]) / (2 * dphi)
        #else:
        #    dV_dphi[:] = 0.0
        
        # integrand e integrali (uso Vk normalizzato)
        
        integrand = -V + self.alpha1 * (dV_dphi ** 2)

        accel_integral = self.alpha0 * dphi * np.sum(np.cos(self.group_angles) * (-V + self.alpha1 * dV_dphi**2))
        turn_integral = self.beta0 * dphi * np.sum(np.sin(self.group_angles) * (-V + self.beta1 * dV_dphi**2))
        
        # --- SAFETY: limit dell'accelerazione per tick (evita esplosioni istantanee) ---
        #max_accel = float(self.spin_model_params.get("max_accel_per_tick", 0.5))  # scala consigliata 0.05..0.5
        # compute raw dv then clamp
        #raw_dv = self.gamma * (self.v0 - self._v) + accel_integral
        #dv = float(np.clip(raw_dv, -abs(max_accel), abs(max_accel)))
        
        dv = self.gamma * (self.v0 - self._v) + accel_integral

        # Debug utile: mostra raw vs normalizzato

        # ---- Euler update for v (Eq.3 approx) ----
        # dv/dt = gamma * (v0 - v) + accel_integral
        # dv = self.gamma * (self.v0 - self._v) + accel_integral
        
        self._v += self.dt * dv
        
        print("v_pre_clip=", self._v)

        # clip speed to [0, v0] as safety (paper assumes speed relaxes to v0)
        self._v = float(np.clip(self._v, -self.v0, max(self.v0, getattr(self.agent, "max_absolute_velocity", self.v0))))

        print("DEBUG_AFTER:",
            "v_post_clip=", self._v,
            "dv_applied=", dv * self.dt if self.dt is not None else None)
        # ---- Euler update for heading psi (Eq.4 approx) ----
        # dpsi/dt = beta0 * turn_integral   (+ optional noise)
        #dpsi = self.beta0 * turn_integral
        #if self.angular_noise_std and self.angular_noise_std > 0.0:
        #    dpsi += float(self.agent.random_generator.normalvariate(0.0, self.angular_noise_std))
        #self._psi += self.dt * dpsi

        # Optionally, prefer to use ring bump as heading target (comment/uncomment as needed)
        # if bump_angle is not None:
        #     # direct coupling toward bump: proportional correction
        #     psi_error = (bump_angle - self._psi + math.pi) % (2*math.pi) - math.pi
        #     k_psi = float(self.spin_model_params.get("k_psi", 1.0))
        #     self._psi += self.dt * k_psi * psi_error

        # write outputs to agent (agent expects degrees for angular_velocity_cmd in the old code)
        # angular velocity command: use dpsi/dt converted into degrees and clamped by agent limits
        
        #ang_vel_deg = normalize_angle(math.degrees(dpsi))
        #ang_vel_deg = max(min(ang_vel_deg, self.agent.max_angular_velocity), -self.agent.max_angular_velocity)

        # linear velocity command is the current _v
        #self.agent.linear_velocity_cmd = float(self._v)
        #if self._v<0:
        #    self.agent.linear_velocity_cmd = -float(self._v)
        #    self.agent.angular_velocity_cmd = float(ang_vel_deg+180)
        #else:

        self.agent.linear_velocity_cmd = float(self._v)
        # !!!! self.agent.angular_velocity_cmd = float(ang_vel_deg)   

        
        # self.agent.linear_velocity_cmd = max(0.0, self.agent.linear_velocity_cmd)
        

        

        # log debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s spin_B update -> v=%.3f dv=%.3f accel_int=%.3f | dpsi=%.4f bump=%.4f ang_deg=%.2f",
                self.agent.get_name(),
                self._v,
                dv,
                accel_integral,
                dpsi,
                (bump_angle if bump_angle is not None else float("nan")),
                ang_vel_deg,
            )

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