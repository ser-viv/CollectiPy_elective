import logging
import math
import numpy as np
from plugin_base import DetectionModel
from plugin_registry import register_detection_model
from models.utils import normalize_angle

logger = logging.getLogger("sim.detection.visual")


class VisualDetectionModel(DetectionModel):
    """
    Modello visivo compatibile con il ring attractor.
    Restituisce gli stessi canali del GPS:
        - agents
        - objects
        - combined

    Differenza chiave:
        al posto della gaussiana del GPS, usa l'occlusione angolare (subtended angle)
        e la sua intersezione con ogni settore del ring.
    """

    def __init__(self, agent, context=None):
        self.agent = agent
        context = context or {}

        self.num_groups = context.get("num_groups", 1)
        self.num_spins_per_group = context.get("num_spins_per_group", 1)
        self.perception_width = context.get("perception_width", 0.5)

        # angoli dei gruppi (radianti) → come richiesto dal ring system
        self.group_angles = context.get(
            "group_angles",
            np.linspace(0, 2 * math.pi, self.num_groups, endpoint=False)
        )

        self.reference = context.get("reference", "egocentric")
        self.perception_global_inhibition = context.get("perception_global_inhibition", 0)

        self.max_detection_distance = float(
            context.get("max_detection_distance",
                getattr(self.agent, "perception_distance", math.inf))
        )

    # =====================================================================
    # SENSE (uguale interfaccia GPS)
    # =====================================================================
    def sense(self, agent, objects: dict, agents: dict, arena_shape=None):
        """
        Restituisce un dict con:
            objects:  num_groups*num_spins_per_group
            agents:   num_groups*num_spins_per_group
            combined: somma dei due
        """
        '''print(
        "[VISUAL DEBUG] sense() agents_keys=%s objects_keys=%s",
        list(agents.keys()) if isinstance(agents, dict) else type(agents),
        list(objects.keys()) if isinstance(objects, dict) else type(objects),
        )
        for k, v in agents.items():
            print(
                "[VISUAL DEBUG] agents[%s] contains %d shapes",
                k, len(v)
            )
        for k, v in objects.items():
            shapes = v[0] if isinstance(v, tuple) else v
            print(
                "[VISUAL DEBUG] objects[%s] contains %d shapes",
                k, len(shapes)
            )
            '''
        

        channel_size = self.num_groups * self.num_spins_per_group

        agent_channel = np.zeros(channel_size)
        object_channel = np.zeros(channel_size)

        # stessa gerarchia del GPS
        hierarchy = self._resolve_hierarchy(agent, arena_shape)

        # raccoglie agenti → occlusione angolare
        self._collect_agent_targets(agent_channel, agents, hierarchy)

        # raccoglie oggetti → occlusione angolare
        self._collect_object_targets(object_channel, objects)

        # global inhibition come nel GPS
        self._apply_global_inhibition(agent_channel)
        self._apply_global_inhibition(object_channel)

        object_channel[:] = 0.0

        combined = np.maximum(agent_channel, object_channel)
        combined = np.clip(combined, 0.0, 1.0)

        '''
        print("[VISUAL DEBUG] perception min=%.3f max=%.3f mean=%.3f",
        combined.min(),
        combined.max(),
        combined.mean())
        '''
        return {
            "objects": object_channel,
            "agents": agent_channel,
            "combined": combined,
        }

    # =====================================================================
    # RACCOLTA TARGET (AGENTI)
    # =====================================================================
    def _collect_agent_targets(self, perception, agents, hierarchy):
        for club, agent_shapes in agents.items():
            for n, shape in enumerate(agent_shapes):
                meta = getattr(shape, "metadata", {}) if hasattr(shape, "metadata") else {}
                target_name = meta.get("entity_name")

                # ignorare sé stessi
                if target_name:
                    if target_name == self.agent.get_name():
                        continue
                elif f"{club}_{n}" == self.agent.get_name():
                    continue

                target_node = meta.get("hierarchy_node")
                if not self._hierarchy_allows_agent(target_node, hierarchy):
                    continue

                agent_pos = shape.center_of_mass()
                dx = agent_pos.x - self.agent.position.x
                dy = agent_pos.y - self.agent.position.y
                dz = agent_pos.z - self.agent.position.z

                radius = getattr(shape, "bounding_radius", 0.025)

                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if distance > self.max_detection_distance:
                    return

                # angolo relativo
                angle = math.degrees(math.atan2(-dy, dx))

                angle = normalize_angle(angle)  # ritorna in [-180,180]

                angle_rad = math.radians(angle)

                # subtended angle
                half_subt = math.atan(radius / max(distance, 1e-6))

                obj_min = angle_rad - half_subt
                obj_max = angle_rad + half_subt
                
                # diverso da gps
                # print("[VISUAL DEBUG] target_name=%s radius=%.3f pos=(%.2f, %.2f, %.2f)",target_name,radius,dx, dy, dz)
                #self._accumulate_occlusion(perception, dx, dy, dz, radius, strength=5.0)
                self._accumulate_occlusion_interval(
                    perception,
                    obj_min,
                    obj_max,
                    strength=1.0
                )
    # =====================================================================
    # RACCOLTA OGGETTI
    # =====================================================================
    def _collect_object_targets(self, perception, objects):
        for _, (shapes, positions, strengths, uncertainties) in objects.items():
            for i in range(len(shapes)):
                dx = positions[i].x - self.agent.position.x
                dy = positions[i].y - self.agent.position.y
                dz = positions[i].z - self.agent.position.z

                # diverso da gps
                radius = getattr(shapes[i], "bounding_radius", 0.025)
                strength = strengths[i]

                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                if distance > self.max_detection_distance:
                    return

                # angolo relativo
                angle = math.degrees(math.atan2(-dy, dx))

                angle = normalize_angle(angle)  # ritorna in [-180,180]

                angle_rad = math.radians(angle)

                # subtended angle
                half_subt = math.atan(radius / max(distance, 1e-6))

                obj_min = angle_rad - half_subt
                obj_max = angle_rad + half_subt

                #self._accumulate_occlusion(perception, dx, dy, dz, radius, strength)
                self._accumulate_occlusion_interval(
                    perception,
                    obj_min,
                    obj_max,
                    strength=1.0
                )

    # =====================================================================
    # GLOBAL INHIBITION
    # =====================================================================
    def _apply_global_inhibition(self, perception_channel):
        if self.perception_global_inhibition == 0:
            return
        perception_channel -= self.perception_global_inhibition

    # =====================================================================
    # ACCUMULO BASATO SU OCCLUSIONE ANGOLARE (subtended angle)
    # =====================================================================
    def _accumulate_occlusion_interval(self, perception, obj_min, obj_max, strength=1.0):
        """
        Accumula un visual field binario V(φ):
        tutti i settori del ring che intersecano l'intervallo angolare
        [obj_min, obj_max] ricevono un contributo costante.
        """
    
        TWO_PI = 2.0 * math.pi
    
        # normalizzazione in [0, 2π)
        obj_min = obj_min % TWO_PI
        obj_max = obj_max % TWO_PI
    
        for g, center in enumerate(self.group_angles):
        
            sec_min = (center - self.perception_width / 2) % TWO_PI
            sec_max = (center + self.perception_width / 2) % TWO_PI
    
            # verifica intersezione tra due intervalli circolari
            if obj_min <= obj_max:
                obj_intersects = not (sec_max < obj_min or sec_min > obj_max)
            else:
                # intervallo oggetto attraversa 2π
                obj_intersects = (sec_min <= obj_max) or (sec_max >= obj_min)
    
            if obj_intersects:
                start = g * self.num_spins_per_group
                end = start + self.num_spins_per_group
                perception[start:end] = 1.0
    
    

    # =====================================================================
    # UTILS
    # =====================================================================
    @staticmethod
    def _interval_intersection(a1, a2, b1, b2):
        left = max(a1, b1)
        right = min(a2, b2)
        return max(0.0, right - left)

    @staticmethod
    def _resolve_hierarchy(agent, arena_shape):
        if arena_shape is not None:
            metadata = getattr(arena_shape, "metadata", None)
            if metadata:
                hierarchy = metadata.get("hierarchy")
                if hierarchy:
                    return hierarchy
        return getattr(agent, "hierarchy_context", None)

    
    def _hierarchy_allows_agent(self, target_node, hierarchy) -> bool:
        """Return True if the observer can interact with the target based on hierarchy."""
        checker = getattr(self.agent, "allows_hierarchical_link", None)
        if not callable(checker):
            return True
        return checker(target_node, "detection", hierarchy)


register_detection_model("VISUAL", lambda agent, context=None: VisualDetectionModel(agent, context))