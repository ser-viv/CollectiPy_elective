# ------------------------------------------------------------------------------
#  CollectiPy
# Copyright (c) 2025 Fabio Oddi
#
#  Example plugin showing how to compute group-level statistics that can later
#  be exported through `results.group_specs`. Import this module (e.g. add
#  "plugins.examples.group_stats_plugin" to the `plugins` list in the config)
#  and set an agent's `logic_behavior` to `"heading_sampler"` to attach the
#  logic model provided below.
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
from statistics import mean
from typing import Any

from plugin_registry import register_logic_model

logger = logging.getLogger("sim.plugins.heading_sampler")


class HeadingSampler:
    """
    Simple logic plugin that tracks the rolling mean heading for each agent.

    The moving average is written to `agent.snapshot_metrics`, so data
    processors can look at it when building `group_specs` aggregates
    (e.g., mean heading per swarm). The plugin intentionally keeps the state
    lightweight: only the last 32 headings are retained.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._window = []

    def step(self, agent: Any, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        """Store the latest heading and expose a rolling average."""
        _ = (arena_shape, objects, agents)
        orientation = agent.get_orientation()
        heading = float(getattr(orientation, "z", 0.0)) if orientation is not None else 0.0
        self._window.append(heading)
        if len(self._window) > 32:
            self._window.pop(0)
        metrics = getattr(agent, "snapshot_metrics", {})
        metrics["heading_last_deg"] = heading
        metrics["heading_window_deg"] = float(mean(self._window))
        agent.snapshot_metrics = metrics
        if tick % 200 == 0:
            logger.debug("%s mean heading %.2f°", agent.get_name(), metrics["heading_window_deg"])


def _create_heading_sampler(agent: Any) -> HeadingSampler:
    """Factory registered in the plugin registry."""
    return HeadingSampler(agent)


register_logic_model("heading_sampler", _create_heading_sampler)
