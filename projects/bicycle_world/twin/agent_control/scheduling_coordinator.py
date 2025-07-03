from __future__ import annotations
from typing import TYPE_CHECKING

from ofact.twin.agent_control.scheduling_coordinator import SchedulingCoordinatorAgent
from projects.bicycle_world.twin.agent_control.behaviours.scheduling.coordinator import (
    SchedulingCoordinatorBicycle)

if TYPE_CHECKING:
    from ofact.twin.agent_control.organization import Agents


class SchedulingCoordinatorBicycleAgent(SchedulingCoordinatorAgent):
    """
    The scheduling coordinator agent takes slices of process_executions_components that have overlapping time slots and
    resolve them.
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, state_model, processes):

        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server, ip_address_xmpp_server=ip_address_xmpp_server,
                         state_model=state_model, processes=processes)
        self.SchedulingCoordinator = SchedulingCoordinatorBicycle()

    def copy(self):
        agent_copy = super(SchedulingCoordinatorBicycleAgent, self).copy()

        agent_copy.SchedulingCoordinator = SchedulingCoordinatorBicycle()
        agent_copy.processes = agent_copy.processes.copy()

        return agent_copy
