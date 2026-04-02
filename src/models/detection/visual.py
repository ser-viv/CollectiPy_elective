import logging
import math
import numpy as np
from core.configuration.plugin_base import DetectionModel
from core.configuration.plugin_registry import register_detection_model
from models.utility_functions import normalize_angle

logger = logging.getLogger("sim.detection.visual")


class VisualDetectionModel(DetectionModel):
    """
    Modello visivo compatibile con il ring attractor.

    Restituisce:
        agents:         campo binario agenti (invariato)
        objects:        campo binario oggetti — ora include i bordi dell'arena circolare
        combined:       max(agents, objects)
        edge_counts:    array (channel_size,) int32 — per ogni spin, quanti bordi
                        angolari cadono nel suo settore. Bordo sx e dx contano separatamente.
        agent_metadata: lista di dict con angle, distance, edge_left, edge_right,
                        angular_width per ogni agente percepito
        arena_metadata: lista di dict con angle, distance, angular_width per ogni
                        segmento di bordo dell'arena percepito
    """

    def __init__(self, agent, context=None):
        self.agent = agent
        context = context or {}

        self.num_groups = context.get("num_groups", 1)
        self.num_spins_per_group = context.get("num_spins_per_group", 1)
        self.perception_width = context.get("perception_width", 0.5)

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

        # Numero di punti campionati sul bordo circolare dell'arena.
        # Più alto = rappresentazione più fine ma più costosa.
        # Configurabile via context: "num_boundary_samples"
        self.num_boundary_samples = int(context.get("num_boundary_samples", 36))

        min_width = 2 * math.pi / self.num_groups
        if self.perception_width < min_width:
            logger.warning(
                "perception_width (%.3f) < angular spacing (%.3f): "
                "blind spots exist between groups",
                self.perception_width, min_width
            )

    def sense(self, agent, objects: dict, agents: dict, arena_shape=None):
        channel_size = self.num_groups * self.num_spins_per_group

        agent_channel  = np.zeros(channel_size)
        object_channel = np.zeros(channel_size)
        edge_counts    = np.zeros(channel_size, dtype=np.int32)
        agent_metadata = []
        arena_metadata = []

        hierarchy = self._resolve_hierarchy(agent, arena_shape)

        self._collect_agent_targets(
            agent_channel, agents, hierarchy,
            edge_counts, agent_metadata
        )
        self._collect_object_targets(object_channel, objects)

        # Bordi dell'arena: popolano object_channel e arena_metadata
        if arena_shape is not None:
            self._collect_arena_boundary(object_channel, arena_shape, arena_metadata)

        self._apply_global_inhibition(agent_channel)
        self._apply_global_inhibition(object_channel)

        combined = np.maximum(agent_channel, object_channel)
        combined = np.clip(combined, 0.0, 1.0)

        return {
            "objects":        object_channel,
            "agents":         agent_channel,
            "combined":       combined,
            "edge_counts":    edge_counts,
            "agent_metadata": agent_metadata,
            "arena_metadata": arena_metadata,
        }

    def _collect_arena_boundary(self, object_channel, arena_shape, arena_metadata):
        """
        Campiona num_boundary_samples punti equidistanti sul bordo dell'arena
        circolare. Per ciascun punto visibile (distanza <= max_detection_distance)
        accumula il campo binario in object_channel e aggiunge un entry in
        arena_metadata con lo stesso formato di agent_metadata (angle, distance,
        angular_width), compatibile con update_arena_repulsion_field in spin_model.py.

        Il raggio fittizio del punto-bordo è calcolato come la metà dell'arco
        tra due campioni adiacenti, così la copertura angolare è uniforme e
        indipendente dalla distanza.
        """
        TWO_PI = 2.0 * math.pi

        radius = getattr(arena_shape, "radius", None)
        if radius is None:
            # Fallback: prova a ricavarlo dal diametro
            diameter = getattr(arena_shape, "diameter", None)
            if diameter is not None:
                radius = diameter / 2.0
            else:
                logger.warning("arena_shape has no radius or diameter attribute; "
                               "arena boundary perception skipped.")
                return

        # Centro dell'arena — convenzione (0, 0, 0)
        cx = getattr(arena_shape, "center_x", 0.0)
        cy = getattr(arena_shape, "center_y", 0.0)

        # Arco tra campioni adiacenti: usato come "raggio fittizio" del punto
        # per calcolare l'ampiezza angolare percepita (half_subt).
        arc_half = (TWO_PI * radius / self.num_boundary_samples) / 2.0

        agent_x = self.agent.position.x
        agent_y = self.agent.position.y

        for k in range(self.num_boundary_samples):
            theta = TWO_PI * k / self.num_boundary_samples
            bx = cx + radius * math.cos(theta)
            by = cy + radius * math.sin(theta)

            dx = bx - agent_x
            dy = by - agent_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > self.max_detection_distance:
                continue

            # Angolo verso il punto-bordo
            angle_world = math.degrees(math.atan2(-dy, dx))

            if self.reference == "egocentric":
                angle = normalize_angle(angle_world - self.agent.orientation.z)
            else:
                angle = normalize_angle(angle_world)

            angle_rad = math.radians(angle)
            if angle_rad < 0:
                angle_rad += TWO_PI

            # Ampiezza angolare: tratta arc_half come raggio apparente del segmento
            half_subt = math.atan(arc_half / max(distance, 1e-6))
            obj_min   = angle_rad - half_subt
            obj_max   = angle_rad + half_subt

            self._accumulate_occlusion_interval(
                object_channel, obj_min, obj_max, strength=1.0
            )

            arena_metadata.append({
                "angle":         angle_rad,
                "distance":      distance,
                "angular_width": 2 * half_subt,
            })

    def _collect_agent_targets(self, perception, agents, hierarchy,
                                edge_counts, agent_metadata):
        TWO_PI = 2.0 * math.pi

        for club, agent_shapes in agents.items():
            for n, shape in enumerate(agent_shapes):
                meta = getattr(shape, "metadata", {}) if hasattr(shape, "metadata") else {}
                target_name = meta.get("entity_name")

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

                radius   = getattr(shape, "bounding_radius", 0.05)
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                if distance > self.max_detection_distance:
                    continue

                angle_world = math.degrees(math.atan2(-dy, dx))

                if self.reference == "egocentric":
                    angle = normalize_angle(angle_world - self.agent.orientation.z)
                else:
                    angle = normalize_angle(angle_world)

                angle_rad = math.radians(angle)
                if angle_rad < 0:
                    angle_rad += TWO_PI

                half_subt = math.atan(radius / max(distance, 1e-6))
                obj_min   = angle_rad - half_subt
                obj_max   = angle_rad + half_subt

                # campo binario — invariato
                self._accumulate_occlusion_interval(
                    perception, obj_min, obj_max, strength=1.0
                )

                # conteggio edge
                edge_left  = obj_min % TWO_PI
                edge_right = obj_max % TWO_PI
                self._accumulate_edge(edge_counts, edge_left)
                self._accumulate_edge(edge_counts, edge_right)

                # DEBUG EDGE — rimuovere dopo verifica
                active_edge_groups = sorted(set(
                    g for g in range(self.num_groups)
                    if edge_counts[g * self.num_spins_per_group] > 0
                ))
                # metadati
                agent_metadata.append({
                    "name":          target_name or f"{club}_{n}",
                    "angle":         angle_rad,
                    "distance":      distance,
                    "edge_left":     edge_left,
                    "edge_right":    edge_right,
                    "angular_width": 2 * half_subt,
                })

    def _accumulate_edge(self, edge_counts, edge_angle):
        """Incrementa di 1 gli spin il cui settore contiene edge_angle."""
        TWO_PI = 2.0 * math.pi
        edge_angle = edge_angle % TWO_PI

        for g, center in enumerate(self.group_angles):
            sec_min   = (center - self.perception_width / 2) % TWO_PI
            sec_max   = (center + self.perception_width / 2) % TWO_PI
            sec_wraps = sec_min > sec_max

            if sec_wraps:
                contains = (edge_angle >= sec_min) or (edge_angle <= sec_max)
            else:
                contains = sec_min <= edge_angle <= sec_max

            if contains:
                start = g * self.num_spins_per_group
                end   = start + self.num_spins_per_group
                edge_counts[start:end] += 1

    def _collect_object_targets(self, perception, objects):
        TWO_PI = 2.0 * math.pi
        for _, (shapes, positions, strengths, uncertainties) in objects.items():
            for i in range(len(shapes)):
                dx = positions[i].x - self.agent.position.x
                dy = positions[i].y - self.agent.position.y
                dz = positions[i].z - self.agent.position.z

                radius   = getattr(shapes[i], "bounding_radius", 0.05)
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                if distance > self.max_detection_distance:
                    continue

                angle_world = math.degrees(math.atan2(-dy, dx))

                if self.reference == "egocentric":
                    angle = normalize_angle(angle_world - self.agent.orientation.z)
                else:
                    angle = normalize_angle(angle_world)

                angle_rad = math.radians(angle)
                if angle_rad < 0:
                    angle_rad += TWO_PI

                half_subt = math.atan(radius / max(distance, 1e-6))
                obj_min   = angle_rad - half_subt
                obj_max   = angle_rad + half_subt

                self._accumulate_occlusion_interval(
                    perception, obj_min, obj_max, strength=1.0
                )

    def _apply_global_inhibition(self, perception_channel):
        if self.perception_global_inhibition == 0:
            return
        perception_channel -= self.perception_global_inhibition

    def _accumulate_occlusion_interval(self, perception, obj_min, obj_max, strength=1.0):
        TWO_PI = 2.0 * math.pi

        obj_min   = obj_min % TWO_PI
        obj_max   = obj_max % TWO_PI
        obj_wraps = obj_min > obj_max

        for g, center in enumerate(self.group_angles):
            sec_min   = (center - self.perception_width / 2) % TWO_PI
            sec_max   = (center + self.perception_width / 2) % TWO_PI
            sec_wraps = sec_min > sec_max

            if not obj_wraps and not sec_wraps:
                intersects = not (sec_max < obj_min or sec_min > obj_max)
            elif obj_wraps and not sec_wraps:
                intersects = (sec_max >= obj_min) or (sec_min <= obj_max)
            elif not obj_wraps and sec_wraps:
                intersects = (obj_max >= sec_min) or (obj_min <= sec_max)
            else:
                intersects = True

            if intersects:
                start = g * self.num_spins_per_group
                end   = start + self.num_spins_per_group
                perception[start:end] = strength

    @staticmethod
    def _interval_intersection(a1, a2, b1, b2):
        left  = max(a1, b1)
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
        checker = getattr(self.agent, "allows_hierarchical_link", None)
        if not callable(checker):
            return True
        return bool(checker(target_node, "detection", hierarchy))


register_detection_model("VISUAL", lambda agent, context=None: VisualDetectionModel(agent, context))