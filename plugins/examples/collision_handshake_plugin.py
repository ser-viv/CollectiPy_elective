# ------------------------------------------------------------------------------
#  CollectiPy
# Copyright (c) 2025 Fabio Oddi
#
#  Example plugin showing how to defer handshake messages until agents collide.
#  Import this module (e.g. add
#  "plugins.examples.collision_handshake_plugin" to the `plugins` list in the
#  config) and set an agent's `logic_behavior` to `"collision_handshake"` to
#  attach the logic model provided below.
# ------------------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import Any, Dict, Set

from plugin_registry import register_logic_model

logger = logging.getLogger("sim.plugins.collision_handshake")


class CollisionHandshakeLogic:
    """
    Logic plugin that only allows handshakes after a physical collision.

    The plugin disables the automatic discovery flow built into the radio
    subsystem and toggles the acceptance window based on actual overlaps
    observed between the local shape and the per-agent shape snapshot provided
    by the entity manager. A collision keeps the handshake window open for a
    short period (`_contact_window_ticks`) so the next TX/RX exchange can
    complete the invite/accept cycle. When no collisions are detected for the
    configured window the plugin schedules a graceful termination.
    """

    def __init__(self, agent: Any, *, window_seconds: float = 1.0) -> None:
        self.agent = agent
        ticks_per_second = max(1, int(getattr(agent, "ticks_per_second", 5)))
        self._contact_window_ticks = max(1, int(round(window_seconds * ticks_per_second)))
        self._recent_contacts: Dict[str, int] = {}
        self._handshake_capable = getattr(agent, "msg_type", None) == "hand_shake"
        if not self._handshake_capable:
            logger.warning(
                "%s attached collision_handshake but messages.type != 'hand_shake'; plugin idle",
                agent.get_name()
            )
            return
        agent.set_handshake_autostart(False)
        agent.set_handshake_acceptance(False)
        logger.debug(
            "%s collision-handshake window set to %d ticks",
            agent.get_name(),
            self._contact_window_ticks
        )

    def step(self, agent: Any, tick: int, arena_shape, objects: dict, agents: dict) -> None:
        """Enable handshake flow only when a collision was observed recently."""
        _ = (agent, arena_shape, objects)
        if not self._handshake_capable:
            return
        collisions = self._detect_collisions(agents)
        for peer in collisions:
            self._recent_contacts[peer] = tick
        self._purge_stale_contacts(tick)
        has_contact = bool(self._recent_contacts)
        self.agent.set_handshake_acceptance(has_contact)
        if self.agent.handshake_active():
            if not has_contact:
                if self.agent.terminate_handshake():
                    logger.debug("%s ending handshake (no contact)", self.agent.get_name())
            return
        if has_contact:
            if self.agent.request_handshake():
                logger.debug("%s requesting collision-triggered handshake", self.agent.get_name())

    def _detect_collisions(self, agent_shapes: dict) -> Set[str]:
        """Return the identifiers of the agents currently overlapping with us."""
        own_shape = self.agent.get_shape()
        if own_shape is None:
            return set()
        own_name = self.agent.get_name()
        collisions: Set[str] = set()
        for shapes in agent_shapes.values():
            for shape in shapes:
                if shape is own_shape:
                    continue
                metadata = getattr(shape, "metadata", None)
                peer_name = None
                if isinstance(metadata, dict):
                    peer_name = metadata.get("entity_name") or metadata.get("name")
                if not peer_name or peer_name == own_name:
                    continue
                overlapping, _ = own_shape.check_overlap(shape)
                if overlapping:
                    collisions.add(peer_name)
        return collisions

    def _purge_stale_contacts(self, tick: int) -> None:
        """Drop stale contacts that exceeded the allowed window."""
        if not self._recent_contacts:
            return
        expired = [
            peer
            for peer, last_tick in self._recent_contacts.items()
            if tick - last_tick > self._contact_window_ticks
        ]
        for peer in expired:
            self._recent_contacts.pop(peer, None)


def _create_collision_handshake(agent: Any) -> CollisionHandshakeLogic:
    """Factory registered in the plugin registry."""
    return CollisionHandshakeLogic(agent)


register_logic_model("collision_handshake", _create_collision_handshake)
