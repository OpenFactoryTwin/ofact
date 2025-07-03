"""
The scheduling coordinator is a central instance to take scheduling snippets and create a feasible plan ...
@last update: 21.08.2023
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ofact.twin.agent_control.behaviours.basic import get_templates_tuple
from ofact.twin.agent_control.behaviours.scheduling.coordinator import SchedulingCoordinator
from ofact.twin.agent_control.basic import DigitalTwinAgent
from ofact.twin.agent_control.planning_services.routing import RoutingService

if TYPE_CHECKING:
    from ofact.twin.agent_control.organization import Agents


class SchedulingCoordinatorAgent(DigitalTwinAgent):
    """
    The scheduling coordinator agent takes slices of process_executions_components and
    resolve them to ensure a feasible schedule.
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, state_model, processes):

        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server, ip_address_xmpp_server=ip_address_xmpp_server)
        self.SchedulingCoordinator = SchedulingCoordinator()

        self.digital_twin = state_model
        self.processes = processes

        # instantiate routing_service
        resource_types = [self.digital_twin.stationary_resources,
                          self.digital_twin.active_moving_resources,
                          self.digital_twin.passive_moving_resources]

        possible_transport_transfer_processes = self.processes["transport_processes"] + \
                                                self.processes["loading_processes"]
        self.routing_service = RoutingService(resource_types, possible_transport_transfer_processes)

    def copy(self):
        agent_copy = super(SchedulingCoordinatorAgent, self).copy()

        agent_copy.SchedulingCoordinator = SchedulingCoordinator()
        agent_copy.processes = agent_copy.processes.copy()

        return agent_copy

    async def setup(self):
        print(f"[{self.name}] Hello World! I'm agent {str(self.jid)} \n {type(self).__name__} \n")

        scheduling_coordinator_initialization = (self.SchedulingCoordinator, self.SchedulingCoordinator.templates)

        behaviour_initializations = [scheduling_coordinator_initialization]
        for behaviour_instance, templates_raw in behaviour_initializations:
            if not templates_raw:
                self.add_behaviour(behaviour_instance)
            else:
                templates_created = [self.create_template(template) for template in templates_raw]

                templates = get_templates_tuple(templates_created)
                self.add_behaviour(behaviour_instance, templates)
