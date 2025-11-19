# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Fabio Oddi
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
#  You may use, modify, and redistribute this file according to the terms of the
#  license. Attribution is required if this code is used in other works.
# ------------------------------------------------------------------------------

"""Messaging infrastructure between agents (protected core module)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from geometry_utils.spatialgrid import SpatialGrid
from plugin_registry import (
    available_message_buses,
    get_message_bus,
    register_message_bus,
)

logger = logging.getLogger("sim.messagebus")


class BaseMessageBus:
    """
    Base implementation shared by all buses.

    It keeps mailboxes for the participating agents and offers utility
    hooks so that derived classes only need to care about how recipients
    are selected (spatial neighbours, full broadcast, etc.).
    """

    def __init__(
        self,
        agent_entities: Iterable[Any],
        config: Optional[dict] = None,
        context: Optional[dict] = None
    ) -> None:
        """Initialize the instance."""
        self.global_config = config or {}
        self.context = context or {}
        self.comm_range = float(self.global_config.get("comm_range", 0.1))
        self.msg_type = str(self.global_config.get("type", "broadcast")).strip().lower()
        self.kind = str(self.global_config.get("kind", "anonymous")).strip().lower()
        if self.msg_type in {"hand_shake", "rebroadcast"} and self.kind == "anonymous":
            raise ValueError(f"Message type '{self.msg_type}' requires identifiable packets (kind must not be 'anonymous').")
        participants = list(agent_entities)
        self._allowed_names = {agent.get_name() for agent in participants}
        self.participants: Dict[str, Any] = {agent.get_name(): agent for agent in participants}
        self.mailboxes: Dict[str, List[dict]] = {name: [] for name in self._allowed_names}

    def reset_mailboxes(self) -> None:
        """Reset the mailboxes."""
        for name in list(self.mailboxes.keys()):
            self.mailboxes[name] = []

    def sync_agents(self, agents: Iterable[Any]) -> None:
        """Synchronise internal references with the provided agents."""
        self._filter_agents(agents)

    def send_message(self, sender: Any, msg: dict) -> None:
        """Send message."""
        if not isinstance(msg, dict):
            sender_name = sender.get_name() if hasattr(sender, "get_name") else "agent"
            logger.warning("%s attempted to send non-dict message '%s'; dropping", sender_name, msg)
            return
        recipients = list(self._iter_recipients(sender, msg))
        if not recipients:
            return
        if self.msg_type == "hand_shake":
            receiver_id = msg.get("to")
            if receiver_id:
                recipients = [agent for agent in recipients if agent.get_name() == receiver_id]
        base_payload = dict(msg)
        base_payload["from"] = sender.get_name() if self.kind == "id" else None
        visited = {sender.get_name()}
        self._deliver_to_recipients(recipients, base_payload, visited, relay_level=0)

    def receive_messages(self, receiver: Any, limit: Optional[int] = None) -> List[dict]:
        """Receive messages."""
        mailbox = self.mailboxes.setdefault(receiver.get_name(), [])
        if limit is None or limit >= len(mailbox):
            messages = mailbox[:]
            self.mailboxes[receiver.get_name()] = []
            return messages
        messages = mailbox[:limit]
        self.mailboxes[receiver.get_name()] = mailbox[limit:]
        return messages

    def close(self) -> None:
        """Close the component resources."""
        self.mailboxes.clear()
        self.participants.clear()
        self._allowed_names.clear()

    # ----- Internal utilities -------------------------------------------------

    def _filter_agents(self, agents: Iterable[Any]) -> List[Any]:
        """Filter the agents."""
        filtered = []
        for agent in agents:
            if agent is None:
                continue
            try:
                name = agent.get_name()
            except AttributeError:
                continue
            if name not in self._allowed_names:
                continue
            self.participants[name] = agent
            filtered.append(agent)
        return filtered

    def _select_recipients(self, sender: Any, msg: dict) -> Iterable[Any]:
        """Return iterable of agents that should receive the message."""
        _ = sender, msg
        return []

    def _deliver_to_recipients(self, recipients: Iterable[Any], payload: dict, visited: set[str], relay_level: int) -> List[Any]:
        """Push `payload` to recipient mailboxes, returning the agents that received it."""
        delivered = []
        for agent in recipients:
            if agent is None:
                continue
            agent_name = agent.get_name()
            if agent_name in visited:
                continue
            visited.add(agent_name)
            mailbox = self.mailboxes.get(agent_name)
            if mailbox is None:
                continue
            if not isinstance(payload, dict):
                logger.warning("Message bus attempted to deliver non-dict payload '%s' to %s; skipping", payload, agent_name)
                continue
            message = dict(payload)
            message["relay_level"] = relay_level
            mailbox.append(message)
            delivered.append(agent)
        return delivered

    def _iter_recipients(self, sender: Any, msg: dict) -> Iterable[Any]:
        """Yield recipients filtered by hierarchy-aware constraints."""
        for agent in self._select_recipients(sender, msg):
            if self._hierarchy_allows_message(sender, agent):
                yield agent

    def _hierarchy_allows_message(self, sender: Any, recipient: Any) -> bool:
        """Return True if hierarchy constraints allow this exchange."""
        hierarchy = self.context.get("hierarchy")
        if not hierarchy:
            return True
        sender_check = getattr(sender, "allows_hierarchical_link", None)
        if callable(sender_check):
            target_node = getattr(recipient, "hierarchy_node", None)
            if not sender_check(target_node, "messages", hierarchy):
                return False
        recipient_check = getattr(recipient, "allows_hierarchical_link", None)
        if callable(recipient_check):
            source_node = getattr(sender, "hierarchy_node", None)
            if not recipient_check(source_node, "messages", hierarchy):
                return False
        return True


class SpatialMessageBus(BaseMessageBus):
    """
    Default message bus using Euclidean distance between participants.
    """

    def __init__(
        self,
        agent_entities: Iterable[Any],
        config: Optional[dict] = None,
        context: Optional[dict] = None
    ) -> None:
        """Initialize the instance."""
        super().__init__(agent_entities, config, context)
        cell_size = max(self.comm_range, 0.01)
        self.grid = SpatialGrid(cell_size)

    def sync_agents(self, agents: Iterable[Any]) -> None:
        """Sync agents."""
        relevant = self._filter_agents(agents)
        self.grid.clear()
        for agent in relevant:
            self.grid.insert(agent)

    def _select_recipients(self, sender: Any, msg: dict) -> Iterable[Any]:
        """Select the recipients."""
        _ = msg
        return self.grid.neighbors(sender, self.comm_range)

    def close(self) -> None:
        """Close the component resources."""
        self.grid.close()
        super().close()


class GlobalMessageBus(BaseMessageBus):
    """
    Message bus for abstract arenas: delivers messages to every participant.
    """

    def _select_recipients(self, sender: Any, msg: dict) -> Iterable[Any]:
        """Select the recipients."""
        _ = msg
        sender_name = sender.get_name()
        for name, agent in self.participants.items():
            if name == sender_name:
                continue
            yield agent


class MessageBusFactory:
    """Helper responsible for instantiating the appropriate bus."""

    DEFAULT_BUS = "spatial"
    ABSTRACT_FALLBACK = "global"
    _AUTO_VALUES = {"", "auto"}

    @staticmethod
    def create(
        agent_entities: Iterable[Any],
        config: Optional[dict] = None,
        context: Optional[dict] = None
    ) -> BaseMessageBus:
        """Create value."""
        config = config or {}
        context = context or {}
        requested = config.get("bus", "auto")
        bus_name = MessageBusFactory._resolve_name(requested, context)
        bus = get_message_bus(bus_name, agent_entities, config, context)
        if bus is None:
            available = ", ".join(sorted(available_message_buses().keys()))
            raise ValueError(f"Message bus '{bus_name}' is not registered. Available: {available}")
        return bus

    @staticmethod
    def _resolve_name(requested: Optional[str], context: dict) -> str:
        """Resolve the name."""
        value = str(requested or "").strip().lower()
        if value in MessageBusFactory._AUTO_VALUES:
            if context.get("arena_shape") is None:
                return MessageBusFactory.ABSTRACT_FALLBACK
            return MessageBusFactory.DEFAULT_BUS
        return value


# Backwards compatibility: keep exporting the spatial implementation.
MessageBus = SpatialMessageBus


def _register_builtin_buses() -> None:
    """Register builtin buses."""
    register_message_bus("spatial", lambda agents, config, context: SpatialMessageBus(agents, config, context))
    register_message_bus("default", lambda agents, config, context: SpatialMessageBus(agents, config, context))
    register_message_bus("global", lambda agents, config, context: GlobalMessageBus(agents, config, context))
    register_message_bus("abstract", lambda agents, config, context: GlobalMessageBus(agents, config, context))


_register_builtin_buses()
