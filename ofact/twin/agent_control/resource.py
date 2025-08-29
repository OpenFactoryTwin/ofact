"""
This module is used for the resource agents. Therefore, every specific resources like warehouse, work_station,
transport_agent has his own class and inherit from the class ResourceDigitalTwinAgent. The resource agents can represent
the digital representation form the real resources which can be existing in e.g. a real plant, a demonstrator or
a simulation.
@last update: 21.08.2023
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import asyncio
import time

from ofact.twin.utils import setup_dual_logger
logging=setup_dual_logger()
from datetime import timedelta, datetime
from typing import TYPE_CHECKING, Dict, Optional
# Imports Part 2: PIP Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.agent_control.basic import DigitalTwinAgent
from ofact.twin.agent_control.behaviours.basic import get_templates_tuple
from ofact.twin.agent_control.behaviours.env_release.interface import ResourceEnvInterfaceBehaviour
from ofact.twin.agent_control.behaviours.env_release.process_executions_queue import (
    ProcessExecutionsOnHoldHandling)
from ofact.twin.agent_control.behaviours.negotiation.CNET import CNETNegotiationBehaviour
from ofact.twin.agent_control.behaviours.planning.part import PartRequest, PartRequestIntern
from ofact.twin.agent_control.behaviours.planning.process import ProcessRequest
from ofact.twin.agent_control.behaviours.planning.process_group import ProcessGroupRequest
from ofact.twin.agent_control.behaviours.planning.resource import ResourceRequest
from ofact.twin.agent_control.behaviours.planning.resource_binding import ResourceBindingBehaviour
from ofact.twin.agent_control.planning_services.process_execution_projection import ProcessExecutionsProjection
from ofact.twin.agent_control.planning_services.routing import RoutingService
from ofact.twin.agent_control.planning_services.storage_reservation import StorageReservation
from ofact.twin.state_model.entities import (Resource, StationaryResource, Warehouse, WorkStation, Storage,
                                             NonStationaryResource, ActiveMovingResource)
from ofact.twin.state_model.helpers.helpers import convert_lst_of_lst_to_lst
from ofact.twin.state_model.model import StateModel
from ofact.twin.state_model.processes import Process, ProcessExecution

if TYPE_CHECKING:
    from ofact.twin.agent_control.behaviours.planning.tree.preference import Preference
    from ofact.twin.agent_control.organization import Agents
    from ofact.twin.state_model.entities import EntityType

    from ofact.twin.state_model.basic_elements import SourceApplicationTypes


class ResourceDigitalTwinAgent(DigitalTwinAgent):
    """
    The resource_agent provides all methods/ behaviours which are used from the warehouse, transport
    and work_station agent. They inherit from the agent and overwrites methods different to the resource agent.
    He is responsible for a resource and therefore to plan requests from the order agent.
    An agent can be responsible for 1 to n resources.
    He goes through the following phases in the simulation:
    >    1. planning of possible process_executions (executed by the PlanningBehaviours)
    >    2. coordination (executed by the coordinator/ scheduler)
    >    3. response to the order agent (managed over the negotiation behaviours)
    >    4. acceptance (managed through the negotiation behaviours)
    >    5. forwarding to the simulation environment (through the agent responsible for the main resource)

    General behaviours are the ...
    - ... planning behaviours: ProcessRequest, ProcessGroupRequest, PartRequest and ResourceRequest.
    - ... resource_binding (to avoid that planned/ bound resources cannot be used for another planning )
    - ... negotiation behaviours: Initiator, Participant.
    - ... env_interface behavior
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, address_book: dict, possible_processes: list[Process],
                 processes: dict[str: list[Process]], resources: list[Resource], entity_provider: dict,
                 transport_provider: dict, preferences: list[Preference], state_model: StateModel,
                 source_application: Optional[SourceApplicationTypes] = None):
        """

        :param name: the name of the agents
        :param organization: the organization used for communication without using the xmpp server
        :param change_handler: the interface to the environment (simulation and digital twin (for change requests))
        :param password_xmpp_server: password to register to the xmpp server
        :param ip_address_xmpp_server: ip_address to register to the xmpp server
        :param address_book: names of other agents to contact them
        :param possible_processes: processes that the agent can execute
        :param processes: all processes ?
        :param resources: resources the agent is responsible for
        :param entity_provider: agents that can provide a specific entity of an entity type
        :param transport_provider: agents that can provide transport for an entity type
        :param preferences: preference objects for the process_execution_plan's of the resources
        :param state_model: the digital twin object to look in
        """
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server)
        self.address_book = address_book
        self.entity_provider = entity_provider
        self.transport_provider = transport_provider
        self.possible_processes = possible_processes
        self.processes = processes

        self.digital_twin = state_model

        resources_dict: dict[EntityType, list[Resource]] = {}
        for resource in resources:
            try:
                resources_dict.setdefault(resource.entity_type,
                                          []).append(resource)
            except AttributeError:
                raise AttributeError(f"Should be a resource object: {resource}")
            for storages in (resource.get_storages().values()):
                for storage in storages:
                    try:
                        resources_dict.setdefault(storage.entity_type,
                                                  []).append(storage)
                    except:
                        print(storage)
        resources_dict: dict[EntityType, list[Resource]] = \
            {resource_entity_type: list(set(resources))
             for resource_entity_type, resources in resources_dict.items()}

        self._resources_by_entity_type = resources_dict.copy()
        for resource_et, resource_lst in resources_dict.items():
            if resource_et.super_entity_type is not None:
                self._resources_by_entity_type.setdefault(resource_et.super_entity_type,
                                                          []).extend(resource_lst)

        self._resources_without_storages = [resource
                                            for et, resources in resources_dict.items()
                                            for resource in resources
                                            if not isinstance(resource, Storage) or
                                            isinstance(resource, Storage) and resource.situated_in is None]
        self._resources = [resource
                           for et, resources in resources_dict.items()
                           for resource in resources]

        self.resource_reservation = {resource: None
                                     for resource in self._resources}

        if preferences:  # in the first instantiation, the preferences are still strings
            if type(preferences[0]) != str:
                transformed_preferences = {}
                for preference in preferences:
                    for reference_object in preference.reference_objects:
                        transformed_preferences[reference_object] = preference

                preferences = transformed_preferences

        self._preferences: Dict[Resource, Preference] = preferences

        self.storage_reservations = {resource: StorageReservation(resource=resource,
                                                                  max_reservation_duration=None)
                                     for resource in self._resources_without_storages}
        self.process_executions_projection = {resource: ProcessExecutionsProjection(resource=resource)
                                              for resource in self._resources
                                              if isinstance(resource, NonStationaryResource)}
        # instantiate routing_service
        resource_types = [self.digital_twin.stationary_resources,
                          self.digital_twin.active_moving_resources,
                          self.digital_twin.passive_moving_resources]

        possible_transport_transfer_processes = self.processes["transport_processes"] + \
                                                self.processes["loading_processes"]
        self.routing_service = RoutingService(resource_types, possible_transport_transfer_processes)
        self.agents.process_executions_projections |= self.process_executions_projection

        self.reaction_expected: bool = False

        self.deviation_tolerance_time_delta = timedelta(minutes=5)
        self.notification_time_delta = timedelta(minutes=5)
        self.plan_horizont = 2000
        self.negotiation_time_limit_period: int = 200  # ToDo: instantiation  # period
        self.source_application = source_application

        # working variables
        self.process_execution_service_provider: dict[ProcessExecution: str] = {}
        self.process_executions_on_hold: dict[ProcessExecution, str] = {}  # only for the main resources used ...

        self.new_process_executions_available = asyncio.Event()

        self.process_executions_queue: list[tuple[str, ProcessExecution]] = []
        self.process_executions_in_execution: dict[ProcessExecution, str | None] = {}
        self.process_executions_finished: list[ProcessExecution] = []

        self.NegotiationBehaviour = CNETNegotiationBehaviour(collection_phase_period=0.1,
                                                             collection_volume_limit=2,
                                                             negotiation_time_limit=self.negotiation_time_limit_period)

        self.PartRequest = PartRequest()
        self.ResourceRequest = ResourceRequest()
        self.ProcessRequest = ProcessRequest()
        self.ProcessGroupRequest = ProcessGroupRequest()
        self.EnvInterface = ResourceEnvInterfaceBehaviour()
        self.ResourceBinding = ResourceBindingBehaviour()

        self.log = pd.DataFrame()

    def set_source_application(self, source_application):
        self.source_application = source_application

    def update_resource_related_variables(self, new_resources: list[Resource], preferences: list[Preference]):

        resources = new_resources

        resources_dict: dict[EntityType, list[Resource]] = {}
        for resource in resources:
            try:
                resources_dict.setdefault(resource.entity_type,
                                          []).append(resource)
            except AttributeError:
                raise AttributeError(f"Should be a resource object: {resource}")
            for storages in (resource.get_storages().values()):
                for storage in storages:
                    try:
                        resources_dict.setdefault(storage.entity_type,
                                                  []).append(storage)
                    except:
                        print(storage)
        resources_dict: dict[EntityType, list[Resource]] = \
            {resource_entity_type: list(set(resources))
             for resource_entity_type, resources in resources_dict.items()}

        self._resources_by_entity_type = resources_dict.copy()
        for resource_et, resource_lst in resources_dict.items():
            if resource_et.super_entity_type is not None:
                self._resources_by_entity_type.setdefault(resource_et.super_entity_type,
                                                          []).extend(resource_lst)

        self._resources_without_storages = [resource
                                            for et, resources in resources_dict.items()
                                            for resource in resources
                                            if not isinstance(resource, Storage) or
                                            isinstance(resource, Storage) and resource.situated_in is None]
        self._resources = [resource
                           for et, resources in resources_dict.items()
                           for resource in resources]

        self.resource_reservation = {resource: None
                                     for resource in self._resources}

        if preferences:  # in the first instantiation, the preferences are still strings
            if type(preferences[0]) != str:
                transformed_preferences = {}
                for preference in preferences:
                    for reference_object in preference.reference_objects:
                        transformed_preferences[reference_object] = preference

                preferences = transformed_preferences

        self._preferences: Dict[Resource, Preference] = preferences

        self.storage_reservations = {resource: StorageReservation(resource=resource,
                                                                  max_reservation_duration=None)
                                     for resource in self._resources_without_storages}
        self.process_executions_projection = {resource: ProcessExecutionsProjection(resource=resource)
                                              for resource in self._resources
                                              if isinstance(resource, NonStationaryResource)}
        # instantiate routing_service
        resource_types = [self.digital_twin.stationary_resources,
                          self.digital_twin.active_moving_resources,
                          self.digital_twin.passive_moving_resources]

        possible_transport_transfer_processes = self.processes["transport_processes"] + \
                                                self.processes["loading_processes"]
        self.routing_service = RoutingService(resource_types, possible_transport_transfer_processes)

    def copy(self):
        agent_copy = super(ResourceDigitalTwinAgent, self).copy()

        agent_copy.address_book = agent_copy.address_book.copy()
        agent_copy.entity_provider = agent_copy.entity_provider.copy()
        agent_copy.transport_provider = agent_copy.transport_provider.copy()
        agent_copy.processes = agent_copy.processes.copy()
        agent_copy.possible_processes = agent_copy.possible_processes
        agent_copy._resources = agent_copy._resources.copy()
        agent_copy._resources_without_storages = agent_copy._resources_without_storages.copy()
        agent_copy.resource_reservation = {resource: None for resource in agent_copy._resources}
        agent_copy._resources_by_entity_type = agent_copy._resources_by_entity_type.copy()
        agent_copy._preferences = agent_copy._preferences.copy()

        agent_copy.storage_reservations = agent_copy.storage_reservations.copy()
        agent_copy.process_executions_projection = agent_copy.process_executions_projection.copy()

        # working variables
        agent_copy.process_execution_service_provider = agent_copy.process_execution_service_provider.copy()
        agent_copy.process_executions_on_hold = agent_copy.process_executions_on_hold.copy()
        agent_copy.new_process_executions_available = agent_copy.new_process_executions_available.copy()
        agent_copy.process_executions_queue = agent_copy.process_executions_queue.copy()
        agent_copy.process_executions_in_execution = agent_copy.process_executions_in_execution.copy()
        agent_copy.process_executions_finished = agent_copy.process_executions_finished.copy()
        agent_copy.reaction_expected = False

        # behaviours
        agent_copy.NegotiationBehaviour = \
            CNETNegotiationBehaviour(collection_phase_period=0.1,
                                     collection_volume_limit=2,
                                     negotiation_time_limit=self.negotiation_time_limit_period)

        agent_copy.PartRequest = PartRequest()
        agent_copy.ResourceRequest = ResourceRequest()
        agent_copy.ProcessRequest = ProcessRequest()
        agent_copy.ProcessExecutionsOnHoldHandling = ProcessExecutionsOnHoldHandling(
            agent_copy)  # ToDo: copy not possible
        agent_copy.EnvInterface = ResourceEnvInterfaceBehaviour()
        agent_copy.ResourceBinding = ResourceBindingBehaviour()
        agent_copy.log = pd.DataFrame()
        return agent_copy

    @property
    def resources(self):
        return self._resources_by_entity_type

    @resources.setter
    def resources(self, resources):
        if type(resources) != dict:
            return
        self._resources_by_entity_type = resources

    @property
    def preferences(self):
        return self._preferences

    @preferences.setter
    def preferences(self, preferences):
        if preferences:  # in the first instantiation the preferences are still strings
            if type(preferences[0]) == str:
                return
            transformed_preferences = {}
            for preference in preferences:
                for reference_object in preference.reference_objects:
                    transformed_preferences[reference_object] = preference

            preferences = transformed_preferences

        self._preferences = preferences

    def reserve_time_slots(self, resources_time_slots):
        """Reserve a time slot in the process_executions_plan of the Resource DigitalTwinObject"""

        last_issue_id = - 1
        resources_time_slots = sorted(resources_time_slots,
                                      key=lambda resources_time_slots: (resources_time_slots[-1][1],
                                                                        - (resources_time_slots[-1][1] -
                                                                           resources_time_slots[-1][0])),
                                      reverse=False)
        for resource, blocker_name, process_execution_id, order_id, issue_id, (start_time, end_time) \
                in resources_time_slots:
            if issue_id == last_issue_id:
                block_before = True
            else:
                block_before = False

            successful, clashing_blocker_names, clashing_process_execution_ids = \
                resource.block_period(start_time=start_time, end_time=end_time, blocker_name=blocker_name,
                                      process_execution_id=process_execution_id, work_order_id=order_id,
                                      issue_id=issue_id, block_before=block_before)


            if not successful:
                i = 100
                while i > 0:
                    print("Reservation was not successful ...")
                    i -= 1
                logging.error('not successful',extra={"obj_id": self.name})
                logging.error(f'start time : {start_time}, end time: {end_time}, blocker_name: {blocker_name}, '
                              f'process_execution_id: {process_execution_id}, work_order_id: {order_id}, '
                              f'issue_id: {issue_id}, block_before: {block_before}', extra={"obj_id": self.name})
                time.sleep(15)

                # debugging
                print(self.name, "Reservation was not successful")
                successful, clashing_blocker_names, clashing_process_execution_ids = \
                    resource.block_period(start_time=start_time, end_time=end_time, blocker_name=blocker_name,
                                          process_execution_id=process_execution_id, work_order_id=order_id,
                                          issue_id=issue_id, block_before=block_before)

                import numpy as np
                formatted_data = np.array2string(
                    resource.process_execution_plan._time_schedule,
                    separator=', ',
                    formatter={'float_kind': lambda x: f"{x:.2f}"}
                )
                exception_msg = (
                    f"Blocking failed for resource {resource.name}"
                    f"dtype:\n{str(resource.process_execution_plan._time_schedule.dtype)}\n\n"
                    f"Array:\n{formatted_data}\n\n"
                    f"Timeslot: {start_time}, {end_time}\n\n {resources_time_slots}"
                    f"Process Execution ID: {process_execution_id}, Issue ID: {issue_id}\n\n",
                )

                raise Exception(exception_msg)

            last_issue_id = issue_id

    def reserve_parts(self, parts_time_slots):
        """Reserve parts in the storage reservation tool to avoid using the same part twice"""
        for part, blocker_name, issue_id, (start_time, end_time) in parts_time_slots:
            self.reserve_part(part, issue_id)

    def reserve_parts_combined(self, parts_to_reserve):
        """Reserve parts in the storage reservation tool to avoid using the same part twice"""
        provider_resource_parts = {}
        for part in parts_to_reserve:
            provider_resource = self._get_entity_provider_resource(part)
            provider_resource_parts.setdefault(provider_resource,
                                               []).append(part)

        for provider_resource, parts in provider_resource_parts.items():
            if provider_resource in self.storage_reservations:
                self.storage_reservations[provider_resource].add_entity_reservations(
                    entities=parts, time_stamp=self.change_handler.get_current_time(), end_time=None,
                    process_execution_ids=None, issue_ids=None, long_time_reservation=False)
            else:
                raise Exception

    def reserve_part(self, part, issue_id=None):
        provider_resource = self._get_entity_provider_resource(part)  # ToDo: projection needed
        if provider_resource in self.storage_reservations:
            storage_reservation = self.storage_reservations[provider_resource]

            storage_reservation.add_entity_reservation(entity=part, time_stamp=self.change_handler.get_current_time(),
                                                       end_time=None, process_execution_id=None, issue_id=issue_id,
                                                       long_time_reservation=False)
        else:
            raise Exception(self.name, self.storage_reservations, provider_resource.name, part.name,
                            part.external_identifications)

    def cancel_parts_reservation(self, parts, process_execution_id):
        provider_resource_parts = {}
        for part in parts:
            provider_resource = self._get_entity_provider_resource(part)
            if provider_resource not in self.storage_reservations:
                continue

            provider_resource_parts.setdefault(provider_resource,
                                               []).append(part)

        for resource, parts in provider_resource_parts.items():
            self.storage_reservations[resource].remove_reservations(entities=parts,
                                                                    process_execution_id=process_execution_id)

    def _get_entity_provider_resource(self, entity):
        provider_resource = entity.situated_in  # ToDo: projection needed
        if provider_resource not in self.storage_reservations:
            if provider_resource is None:
                raise Exception(f"provider_resource {provider_resource} not in storage_reservations, entity {entity}, "
                                f"{entity.external_identifications}")
            provider_resource = provider_resource.situated_in

        return provider_resource

    def bind_resource(self, resource, order):
        # print(f"Order binding {order.identification}")
        self.resource_reservation[resource] = order
        print(f'bind resource {order.identification} to {resource.name}')
        # self.log = pd.concat([self.log, pd.DataFrame({'order': [order.identification],
        #                                               "resource": [resource.name],
        #                                               "now": [datetime.now()]})])
        # self.log.reset_index()

    def unbind_resource(self, resource):
        self.resource_reservation[resource] = None
        print(self.name, f'unbind resource {resource.name}')
        # self.log = pd.concat([self.log, pd.DataFrame({'order': [None],
        #                                              "resource": [resource.name],
        #                                              "now": [datetime.now()]})])
        # self.log.reset_index()

    async def stop(self):
        #if not self.log.empty:
        #    self.log.to_excel("binding.xlsx")
        await super().stop()

    def reset_planned_process_executions(self, unused_process_executions, blocking_agent_name):
        """Reset the blocks of the agents (time_slot_blocking and part reservations)"""
        unused_process_executions = convert_lst_of_lst_to_lst(unused_process_executions)

        for process_execution in unused_process_executions:
            print(f"Reset planned process_executions")  # {process_execution.identification}"
            # f", {process_execution.executed_start_time} {process_execution.executed_end_time}")
            # unblock reserved time_slots
            if process_execution.executed_start_time and process_execution.executed_end_time:
                for resource in process_execution.get_resources():

                    if resource in self._resources:
                        successful = resource.unblock_period(unblocker_name=blocking_agent_name,
                                                             process_execution_id=process_execution.identification)

                        if resource in self.process_executions_projection:
                            process_executions_projection = self.process_executions_projection[resource]
                            process_executions_projection.remove_planned_process_executions([process_execution])
                    else:
                        print(f"Not reset planned process_executions {process_execution.identification}, "
                              f"{process_execution.executed_start_time} {process_execution.executed_end_time}",
                              resource.name)

            # reset storage reservations
            parts = process_execution.get_parts()
            if not parts:
                continue

            self.cancel_parts_reservation(parts=process_execution.get_parts(),
                                          process_execution_id=process_execution.identification)

    def get_env_interface_behaviour(self):
        """Called from the change_handler to get his communication counter_part behaviour"""
        return self.EnvInterface

    async def setup(self):
        print(f"[{self.name}] Hello World! I'm agent {str(self.jid)} \n {type(self).__name__} \n")

        self.add_behaviour(self.PartRequest)
        self.add_behaviour(self.ResourceRequest)
        self.add_behaviour(self.ProcessRequest)
        self.add_behaviour(self.ProcessGroupRequest)

        negotiation_behaviours_with_templates = self.NegotiationBehaviour.get_behaviour_templates()
        for negotiation_behaviour in negotiation_behaviours_with_templates:
            templates_lst = [self.create_template(template) for template in negotiation_behaviour["templates"]]

            if not templates_lst:
                self.add_behaviour(negotiation_behaviour["behaviour"])
            else:
                templates = get_templates_tuple(templates_lst)
                self.add_behaviour(negotiation_behaviour["behaviour"], templates)

        release_templates = [self.create_template(template)
                             for template in self.EnvInterface.templates]
        if len(release_templates) > 2:
            raise ValueError("template passing should be adapted")

        self.add_behaviour(self.EnvInterface, (release_templates[0] ^ release_templates[1]))
        self.EnvInterface.initial_subscription()

        binding_templates = [self.create_template(template) for template in self.ResourceBinding.templates]
        if len(binding_templates) > 2:
            raise ValueError("template passing should be adapted")

        self.add_behaviour(self.ResourceBinding, binding_templates[0])


class TransportAgent(ResourceDigitalTwinAgent):
    """
    The TransportAgent can be responsible for a transport unit.
    Therefore, it is extended by location identification for non_stationary_resource to determine there current
    (resource) location.
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, address_book: dict, possible_processes: list[Process], processes,
                 entity_provider: dict, transport_provider,
                 resources: list[Resource], preferences: list[Preference], state_model: StateModel,
                 source_application: Optional[SourceApplicationTypes] = None):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model, source_application=source_application)

        self.resource_location_mapping: dict[Resource, StationaryResource] = \
            self.create_initial_resource_location_mapping()

    def copy(self):
        agent_copy = super(TransportAgent, self).copy()

        agent_copy.resource_location_mapping = agent_copy.resource_location_mapping.copy()
        # agent_copy.ResourceBindingBehaviour = agent_copy.ResourceBindingBehaviour()

        return agent_copy

    def create_initial_resource_location_mapping(self) -> dict[Resource, StationaryResource]:
        stationary_resources = \
            [stationary_resource
             for stationary_resources in list(self.digital_twin.stationary_resources.values())
             for stationary_resource in stationary_resources
             if stationary_resource.__class__.__name__ in ["WorkStation", "Warehouse", "ConveyorBelt"]
             or "Loading Station" in stationary_resource.name]

        resource_location_mapping = {}
        for resource in self._resources:
            possible_stationary_resources = [stationary_resource
                                             for stationary_resource in stationary_resources
                                             if stationary_resource.get_position() == resource.get_position()]
            if 0 == len(possible_stationary_resources) or 1 < len(possible_stationary_resources):
                continue
                # raise ValueError("Should not be possible")  # needed for (active moving) non stationary resources

            resource_location_mapping[resource] = possible_stationary_resources[0]

        return resource_location_mapping

    def get_resource_location(self, resource):
        # ToDo: for planning in the future, it should be updated by the process_execution projection
        return self.resource_location_mapping[resource]

    async def setup(self):
        await super().setup()


class WarehouseAgent(ResourceDigitalTwinAgent):
    """
    The WarehouseAgent can be responsible for a warehouse.
    The warehouse agent has additionally the capability to order new parts.
    ToDo: the order of new parts is currently not really possible (complete implementation needed)
     in some cases (reorder point needed) if the parts are requested, they should be available in the same round
    """

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str,
                 address_book: dict, transport_provider: dict, entity_provider: dict,
                 possible_processes: list[Process], processes, resources: list[Resource],
                 preferences: list[Preference],
                 state_model: StateModel, entity_types_to_store,
                 source_application: Optional[SourceApplicationTypes] = None):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model, source_application=source_application)

        self.entity_types_to_store = entity_types_to_store

    def copy(self):
        agent_copy = super(WarehouseAgent, self).copy()

        agent_copy.entity_types_to_store = agent_copy.entity_types_to_store.copy()

        return agent_copy

    def trigger_good_receipt(self, good_receipt_process_executions):
        self.EnvInterface.receive_goods(good_receipt_process_executions)


class WarehouseAgentIntern(WarehouseAgent):
    """Intern means that the warehouse agent does not request parts form other resources than Warehouses"""

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str,
                 address_book: dict, transport_provider: dict, entity_provider: dict,
                 possible_processes: list[Process], processes, resources: list[Resource],
                 preferences: list[Preference],
                 state_model: StateModel, entity_types_to_store,
                 source_application: Optional[SourceApplicationTypes] = None):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model,
                         entity_types_to_store=entity_types_to_store, source_application=source_application)

        self.entity_provider = {entity: [resource for resource in resources if isinstance(resource, Warehouse)]
                                for entity, resources in self.entity_provider.items()}
        self.PartRequest = PartRequestIntern()


class WorkStationAgent(ResourceDigitalTwinAgent):
    """The WorkStationAgent can be responsible for a bending unit."""

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, address_book: dict, entity_provider: dict, transport_provider,
                 possible_processes: list[Process], processes, resources: list[Resource],
                 preferences: list[Preference],
                 state_model: StateModel,
                 source_application: Optional[SourceApplicationTypes] = None):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model, source_application=source_application)

    def copy(self):
        agent_copy = super(WorkStationAgent, self).copy()

        return agent_copy


class WorkStationAgentIntern(WorkStationAgent):
    """Intern means that the work_station agent does not request parts from other resources than WorkStations"""

    def __init__(self, name: str, organization: Agents, change_handler, password_xmpp_server: str,
                 ip_address_xmpp_server: str, address_book: dict, entity_provider: dict, transport_provider,
                 possible_processes: list[Process], processes, resources: list[Resource],
                 preferences: list[Preference],
                 state_model: StateModel,
                 source_application: Optional[SourceApplicationTypes] = None):
        super().__init__(name=name, organization=organization, change_handler=change_handler,
                         password_xmpp_server=password_xmpp_server,
                         ip_address_xmpp_server=ip_address_xmpp_server, address_book=address_book,
                         entity_provider=entity_provider, transport_provider=transport_provider,
                         possible_processes=possible_processes, processes=processes, resources=resources,
                         preferences=preferences, state_model=state_model, source_application=source_application)

        self.entity_provider = {entity: [resource
                                         for resource in resources
                                         if isinstance(resource, WorkStation)]
                                for entity, resources in self.entity_provider.items()}
        self.PartRequest = PartRequestIntern()
