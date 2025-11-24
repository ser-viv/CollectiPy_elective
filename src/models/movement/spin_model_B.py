import logging
import math
import numpy as np

from models.spinsystem_B import SpinSystemB
from plugin_base import MovementModel
from plugin_registry import (
    get_detection_model,
    get_movement_model,
    register_movement_model,
)
from models.utils import normalize_angle

logger = logging.getLogger("sim.spin_B")

class SpinMovementModelB(MovementModel):
    """Spin movement model B: uses SpinSystemB and expects visual detection providing V and dV."""

    def __init__(self, agent):
        self.agent = agent
        self.spin_model_params = agent.config_elem.get("spin_model", {})
        # keep many parameters consistent with original
        self.spin_pre_run_steps = self.spin_model_params.get("spin_pre_run_steps", 0)
        self.spin_per_tick = self.spin_model_params.get("spin_per_tick", 10)
        self.num_groups = self.spin_model_params.get("num_groups", 16)
        self.num_spins_per_group = self.spin_model_params.get("num_spins_per_group", 8)
        self.reference = self.spin_model_params.get("reference", "egocentric")
        self.fallback_behavior = agent.config_elem.get("fallback_moving_behavior", "none")

        # vision coefficients passed to spin system (defaults can be overridden by config)
        self.alpha0 = float(self.spin_model_params.get("alpha0", 1.0))
        self.alpha1 = float(self.spin_model_params.get("alpha1", 0.0))
        self.alpha2 = float(self.spin_model_params.get("alpha2", 0.0))
        self.beta0  = float(self.spin_model_params.get("beta0", 1.0))
        self.beta1  = float(self.spin_model_params.get("beta1", 0.0))
        self.beta2  = float(self.spin_model_params.get("beta2", 0.0))

        self.perception = None
        self._active_perception_channel = "visual"  # we expect visual detection
        self.perception_range = self._resolve_detection_range()
        self._fallback_model = None

        # create detection model (likely "VISUAL")
        self.detection_model = self._create_detection_model()

        # instantiate spin system B
        self.spin_system = SpinSystemB(
            self.agent.random_generator,
            self.num_groups,
            self.num_spins_per_group,
            float(self.spin_model_params.get("T", 0.5)),
            float(self.spin_model_params.get("J", 1)),
            float(self.spin_model_params.get("nu", 0)),
            float(self.spin_model_params.get("p_spin_up", 0.5)),
            int(self.spin_model_params.get("time_delay", 1)),
            self.spin_model_params.get("dynamics", "metropolis"),
            alpha0=self.alpha0,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            beta0=self.beta0,
            beta1=self.beta1,
            beta2=self.beta2,
        )

    def _create_detection_model(self):
        context = {
            "num_groups": self.num_groups,
            "num_spins_per_group": self.num_spins_per_group,
            "max_detection_distance": self.perception_range,
        }
        detection_name = getattr(self.agent, "detection", None)
        if not detection_name:
            detection_name = self.agent.config_elem.get("detection", "VISUAL")
        return get_detection_model(detection_name, self.agent, context)

    def step(self, agent, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        # sample detection
        snapshot = None
        if self.detection_model is not None:
            snapshot = self.detection_model.sense(self.agent, objects, agents, arena_shape)

        if snapshot is None:
            self._run_fallback(tick, arena_shape, objects, agents)
            return

        # Expect snapshot either raw array or dict produced by VisualDetectionModel
        if isinstance(snapshot, dict):
            # try to pick visual channel keys
            V = snapshot.get("V")
            dV = snapshot.get("dV")
            angles = snapshot.get("angles", None)
            # fallback: if detection returns channels objects/agents, try combined
            if V is None and "combined" in snapshot:
                V = snapshot["combined"]
                dV = np.zeros_like(V)
        else:
            # snapshot is raw array -> treat as V
            V = np.asarray(snapshot)
            dV = np.roll(V, -1) - V

        if V is None:
            self._run_fallback(tick, arena_shape, objects, agents)
            return

        # Normalize/shape V and dV to spin_system expectations:
        # expected length prefer num_groups or groups*num_spins
        if V.size == self.num_groups:
            # expand to spins
            V_exp = np.repeat(V, self.num_spins_per_group)
            dV_exp = np.repeat(dV, self.num_spins_per_group)
        else:
            V_exp = V.ravel()
            dV_exp = dV.ravel()

        # build external field following Eq. 3/4 expansion (we focus on turning-like input for ring)
        # Here we use beta-like coefficients to set the stimulus for the ring (turning).
        # field = beta0 * ( -V + beta1 * (dV^2) )
        field = self.beta0 * ( - V_exp + self.beta1 * (dV_exp ** 2) )

        # feed spin system and run spins
        self.spin_system.update_external_field(field)
        self.spin_system.run_spins(steps=self.spin_per_tick)

        # compute average direction from spin system
        angle_rad = self.spin_system.average_direction_of_activity()
        if angle_rad is None:
            self._run_fallback(tick, arena_shape, objects, agents)
            return

        # convert to agent reference if needed
        if self.reference == "allocentric":
            angle_rad = angle_rad - math.radians(self.agent.orientation.z)
        angle_deg = normalize_angle(math.degrees(angle_rad))
        angle_deg = max(min(angle_deg, self.agent.max_angular_velocity), -self.agent.max_angular_velocity)

        width = self.spin_system.get_width_of_activity()
        scaling_factor = 1.0 / width if width and width > 0 else 0.0
        scaling_factor = np.clip(scaling_factor, 0.0, 1.0)

        # set movement commands on agent: linear depends on concentration, angular on angle_deg
        self.agent.linear_velocity_cmd = self.agent.max_absolute_velocity * scaling_factor
        self.agent.angular_velocity_cmd = angle_deg

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "%s (B) spin updated -> angle=%.2f width=%s scaling=%.3f",
                self.agent.get_name(),
                angle_deg,
                str(width),
                scaling_factor
            )

    def _run_fallback(self, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        if self.fallback_behavior in ("spin_model","none"):
            return
        behavior = self.fallback_behavior
        if self._fallback_model is None:
            self._fallback_model = get_movement_model(self.fallback_behavior, self.agent)
            if self._fallback_model is None and self.fallback_behavior != "random_walk":
                self._fallback_model = get_movement_model("random_walk", self.agent)
                behavior = "random_walk"
        if self._fallback_model is None:
            logger.warning("%s has no fallback movement model configured", self.agent.get_name())
            return
        self._fallback_model.step(self.agent, tick, arena_shape, objects, agents)


register_movement_model("spin_model_B", lambda agent: SpinMovementModelB(agent))