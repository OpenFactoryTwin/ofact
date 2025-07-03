"""
The information service agent is a central service, where information needed for planning can be requested.
@last update: 21.08.2023
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ofact.twin.agent_control.behaviours.basic import get_templates_tuple
from ofact.twin.agent_control.basic import DigitalTwinAgent
from ofact.twin.agent_control.behaviours.planning.availability_information import AvailableTimeSlotService

if TYPE_CHECKING:
    from ofact.twin.agent_control.organization import Agents
    from ofact.twin.state_model.model import StateModel


class InformationServiceAgent(DigitalTwinAgent):
    """
    The information service agent is a central service, where information needed for planning can be requested.
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, state_model, processes):

        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server, ip_address_xmpp_server=ip_address_xmpp_server)
        self.AvailableTimeSlotService = AvailableTimeSlotService()

        self.digital_twin: StateModel = state_model
        self.processes = processes

        # ToDo: to use?

    def copy(self):
        agent_copy = super(InformationServiceAgent, self).copy()

        agent_copy.AvailableTimeSlotService = AvailableTimeSlotService()
        agent_copy.processes = agent_copy.processes.copy()

        return agent_copy

    async def setup(self):
        print(f"[{self.name}] Hello World! I'm agent {str(self.jid)} \n {type(self).__name__} \n")

        scheduling_coordinator_initialization = (self.AvailableTimeSlotService, self.AvailableTimeSlotService.templates)

        behaviour_initializations = [scheduling_coordinator_initialization]
        for behaviour_instance, templates_raw in behaviour_initializations:
            if not templates_raw:
                self.add_behaviour(behaviour_instance)
            else:
                templates_created = [self.create_template(template)
                                     for template in templates_raw]

                templates = get_templates_tuple(templates_created)
                self.add_behaviour(behaviour_instance, templates)
