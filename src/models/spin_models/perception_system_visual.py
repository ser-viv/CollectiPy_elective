# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Perception system model for visual-based spin behaviours."""
from __future__ import annotations

import math
import numpy as np
from models.utility_functions import normalize_angle


class PerceptionModule:
    """Perception layer converting visual detections into spin-friendly channels.
    
    This module is a specialization for visual detection. If the visual detection
    model returns perception channels directly, this module acts as a passthrough.
    If additional processing is needed, it can be added here.
    """
    def __init__(
        self,
        num_groups: int,
        num_spins_per_group: int,
        perception_width: float,
        group_angles: np.ndarray,
        reference: str = "egocentric",
        max_detection_distance: float | None = None,
        agent_strength: float = 5.0,
    ):
        self.num_groups = num_groups
        self.num_spins_per_group = num_spins_per_group
        self.perception_width = perception_width
        self.group_angles = group_angles
        self.reference = reference
        self.max_detection_distance = (
            float(max_detection_distance) if max_detection_distance is not None else math.inf
        )
        self.agent_strength = agent_strength

    def build_channels(self, observer, detections: dict) -> dict[str, np.ndarray]:
        """Convert raw detections into object/agent/combined perception channels.
        
        For visual detection, if detections already contains perception channels
        as np.ndarray, return them directly. Otherwise, process them.
        """
        # Check if detections is already in the format of perception channels
        # (i.e., contains "agents", "objects", "combined" as np.ndarray)
        if isinstance(detections, dict):
            agents_data = detections.get("agents")
            objects_data = detections.get("objects")
            combined_data = detections.get("combined")
            
            # If all three are numpy arrays, we're already in the right format
            if (isinstance(agents_data, np.ndarray) and
                isinstance(objects_data, np.ndarray) and
                isinstance(combined_data, np.ndarray)):
                return detections
        
        # Fallback: empty channels (if visual detection produced raw data instead)
        channel_size = self.num_groups * self.num_spins_per_group
        return {
            "objects": np.zeros(channel_size, dtype=np.float32),
            "agents": np.zeros(channel_size, dtype=np.float32),
            "combined": np.zeros(channel_size, dtype=np.float32),
        }
