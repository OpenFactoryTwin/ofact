from __future__ import annotations

from projects.bicycle_world.twin.agent_control.behaviours.planning.part import LimitedPartRequest
from ofact.twin.agent_control.behaviours.planning.tree.preference import Preference
from ofact.twin.agent_control.organization import Agents
from ofact.twin.agent_control.resource import WarehouseAgent, WorkStationAgent

from ofact.twin.agent_control.planning_services.storage_reservation import (PartUnavailabilityStorageReservation)
from ofact.twin.state_model.entities import Resource
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.processes import Process
from projects.bicycle_world.env.simulation.event_discrete import PartUnavailabilityEnv
from projects.bicycle_world.twin.agent_control.behaviours.env_interface import WorkStationEnvInterface
from projects.bicycle_world.twin.agent_control.behaviours.process_executions_queue import (
    WorkStationProcessExecutionsOnHoldHandling)
from projects.bicycle_world.twin.agent_control.behaviours.scenario_specific import PartReservationBehaviour


class PartUnavailabilityWarehouseAgent(WarehouseAgent):

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str,
                 address_book: dict, transport_provider: dict, entity_provider: dict,
                 possible_processes: list[Process], processes, resources: list[Resource],
                 preferences: list[Preference],
                 state_model: StateModel, entity_types_to_store):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model,
                         entity_types_to_store=entity_types_to_store)

        # probabilities
        part_availability_probability = 0
        part_delay_time_mue = 60
        part_delay_time_sigma = 10
        hidden = True

        self.storage_reservations = \
            {resource: PartUnavailabilityStorageReservation(
                resource=resource, max_reservation_duration=None,
                part_availability_probability=part_availability_probability,
                part_delay_time_mue=part_delay_time_mue, part_delay_time_sigma=part_delay_time_sigma, hidden=hidden)
                for resource in self._resources}
        self.set_storage_reservation_to_simulation()

    def set_storage_reservation_to_simulation(self):
        # short cut
        self.change_handler._environment: PartUnavailabilityEnv
        self.change_handler._environment.set_storage_reservation(self.storage_reservations)


class LimitedWarehouseAgent(WarehouseAgent):
    """
    The TransportAgent can be responsible for a transport unit.
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str,
                 address_book: dict, transport_provider: dict, entity_provider: dict,
                 possible_processes: list[Process], processes, resources: list[Resource],
                 preferences: list[Preference],
                 state_model: StateModel, entity_types_to_store):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model,
                         entity_types_to_store=entity_types_to_store)

        self.PartRequest = LimitedPartRequest()
        self.PartReservation = PartReservationBehaviour()

    async def setup(self):
        await super().setup()

        reservation_templates = \
            [self.create_template(template) for template in self.PartReservation.templates]
        if len(reservation_templates) > 2:
            raise ValueError("template passing should be adapted")

        self.add_behaviour(self.PartReservation, (reservation_templates[0] ^ reservation_templates[1]))


class AdvancedWorkStationAgent(WorkStationAgent):

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, address_book: dict, entity_provider: dict, transport_provider,
                 possible_processes: list[Process], processes, resources: list[Resource],
                 preferences: list[Preference],
                 state_model: StateModel):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model)
        self.process_executions_on_hold_sub = {}

        self.ProcessExecutionsOnHoldHandling = WorkStationProcessExecutionsOnHoldHandling()
        self.EnvInterface = WorkStationEnvInterface()

    async def setup(self):
        await super().setup()
        self.ProcessExecutionsOnHoldHandling.initial_setting()
