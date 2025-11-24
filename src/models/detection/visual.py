import math
import numpy as np
import logging

from plugin_base import DetectionModel
from plugin_registry import register_detection_model

logger = logging.getLogger("sim.detection.visual")


class VisualDetectionModel(DetectionModel):

    def __init__(self, agent, context=None):
        self.agent = agent
        context = context or {}

        # numero di raggi (risoluzione angolare)
        self.n_bins = context.get("n_bins", 360)

        # distanza massima di visione
        self.max_distance = float(
            context.get("max_detection_distance",
                        getattr(agent, "perception_distance", math.inf))
        )

        # raggio degli agenti, se utile
        self.agent_size = context.get("agent_size", 0.1)

        # precomputo angoli
        self.phi_array = np.linspace(-math.pi, math.pi, self.n_bins, endpoint=False)


    def sense(self, agent, objects, agents, arena_shape):
        """
        Ricostruisce:
          - V(phi): campo visivo binario
          - dV(phi): derivata angolare (bordi visivi)
        """

        V = np.zeros(self.n_bins)

        # posizione osservatore
        ax, ay = agent.position.x, agent.position.y
        heading = math.radians(agent.orientation.z)

        # raccogli lista di agenti bersaglio
        target_positions = []
        for club, shapes in agents.items():
            for shape in shapes:
                if shape is agent:  
                    continue
                pos = shape.center_of_mass()
                target_positions.append((pos.x, pos.y))

        # per ogni direzione φ -> ray casting
        for i, phi in enumerate(self.phi_array):

            # direzione assoluta del raggio
            world_phi = phi + heading

            # raggio
            rx = math.cos(world_phi)
            ry = math.sin(world_phi)

            # controlla intersezioni con tutti i bersagli
            for (tx, ty) in target_positions:
                dx = tx - ax
                dy = ty - ay

                # proiezione del vettore target sul raggio
                proj = dx * rx + dy * ry

                if proj < 0:
                    continue  # target dietro

                # distanza minima punto–raggio
                perp = abs(dx * ry - dy * rx)

                if perp < self.agent_size and proj < self.max_distance:
                    V[i] = 1
                    break

        # derivata angolare
        dV = np.roll(V, -1) - V

        return {
            "V": V,
            "dV": dV,
            "angles": self.phi_array
        }


# registra il modello nel sistema
register_detection_model("VISUAL", lambda agent, context=None: VisualDetectionModel(agent, context))