"""
#############################################################
This program and the accompanying materials are made available under the
terms of the Apache License, Version 2.0 which is available at
https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.

SPDX-License-Identifier: Apache-2.0
#############################################################

Instantiation of the digital_twin state model based on an Excel file and creating of the digital_twin state model.

@contact persons: Adrian Freiter
"""
from __future__ import annotations
from typing import TYPE_CHECKING

# Imports Part 3: Project Imports
from ofact.twin.repository_services.deserialization.basic_file_loader import (
    convert_str_to_list, convert_str_to_dict, convert_str_to_datetime, convert_str_to_int)
from ofact.twin.state_model.helpers.helpers import handle_numerical_value

from ofact.twin.agent_control.behaviours.planning.tree.preference import EntityPreference
from ofact.twin.agent_control.information_desk import InformationServiceAgent
from ofact.twin.agent_control.order import OrderDigitalTwinAgent, OrderPoolDigitalTwinAgent
from ofact.twin.agent_control.resource import (WorkStationAgent, WarehouseAgent, TransportAgent,
                                               ResourceDigitalTwinAgent)
from ofact.twin.agent_control.scheduling_coordinator import SchedulingCoordinatorAgent
from ofact.twin.state_model.entities import ActiveMovingResource, Warehouse, Storage, ConveyorBelt
from ofact.twin.state_model.processes import ValueAddedProcess

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.agent_control.organization import Agents
    from ofact.twin.change_handler.change_handler import ChangeHandler
    from ofact.twin.state_model.entities import EntityType, Resource

# Module-Specific Constants


# #### mappings  #######################################################################################################


agent_model_mapper = {'OrderPoolAgent': OrderPoolDigitalTwinAgent,
                      'OrderAgent': OrderDigitalTwinAgent,
                      'ResourceAgent': ResourceDigitalTwinAgent,
                      'WorkStationAgent': WorkStationAgent,
                      'WarehouseAgent': WarehouseAgent,
                      'TransportAgent': TransportAgent,
                      'SchedulingCoordinatorAgent': SchedulingCoordinatorAgent,
                      'InformationServiceAgent': InformationServiceAgent,
                      'Preference': EntityPreference}

class MapperMethods:
    state_model = None
    organization = None
    change_handler = None
    possible_processes_resource_memo = {}

    def __init__(self, state_model: StateModel, organization: Agents, change_handler: ChangeHandler):
        """
        Enable the access to the digital_twin model for the mapping methods

        Parameters
        ----------
        state_model: state_model object
        agents: the agents_model
        change_handler: interface between the agents, digital_twin and environment
        """
        type(self).state_model = state_model

        MapperMethods.possible_processes_resource_memo = {}

        type(self).organization = organization
        type(self).change_handler = change_handler

    @classmethod
    def get_resources(cls):
        stationary_resources = [item for sub_list in cls.state_model.stationary_resources.values()
                                for item in sub_list]
        active_moving_resources = [item for sub_list in list(cls.state_model.active_moving_resources.values())
                                   for item in sub_list]
        resources = stationary_resources + active_moving_resources
        return resources

    @classmethod
    def get_possible_processes_all(cls, x="ALL"):
        if x in cls.possible_processes_resource_memo:
            return cls.possible_processes_resource_memo[x]

        resources = cls.get_resources()

        entity_type_resource_match = {}  # dict[EntityType: Resource]
        for resource in resources:
            if resource.entity_type not in entity_type_resource_match:
                entity_type_resource_match[resource.entity_type] = [resource]
            else:
                entity_type_resource_match[resource.entity_type].append(resource)

        processes = [item for sub_list in list(cls.state_model.processes.values()) for item in sub_list]
        possible_processes_resource = {resource: []
                                       for resource in resources}  # dict[Resource: list[Process]]
        for process in processes:
            for resource_group in process.get_resource_groups():
                try:
                    for resource_entity_type in resource_group.resources:
                        if resource_entity_type not in entity_type_resource_match:
                            break
                        for resource in entity_type_resource_match[resource_entity_type]:
                            if process not in possible_processes_resource[resource]:
                                possible_processes_resource[resource].append(process)
                except:
                    raise Exception(f"Process: {process.name} \n"
                                    f"resource_group: {resource_group} "
                                    f"-> Check if the modelling in the excel file is correct.")

        cls.possible_processes_resource_memo[x] = possible_processes_resource

        return possible_processes_resource

    @classmethod
    def determine_digital_twin(cls, x):
        if x:
            return cls.state_model
        else:
            return None

    @classmethod
    def determine_agents(cls, x):
        if x:
            return cls.organization
        else:
            return None

    @classmethod
    def determine_change_handler(cls, x):
        if x:
            return cls.change_handler
        else:
            return None

    @classmethod
    def determine_possible_processes(cls, resource_s):
        if not isinstance(resource_s, list):
            resource_s = [resource_s]

        possible_processes_all = cls.get_possible_processes_all(x="ALL")
        possible_processes_all = {k.get_static_model_id()[1:]: v
                                       for k, v in possible_processes_all.items()}
        possible_processes = []
        for resource in resource_s:
            additional_resources = possible_processes_all[resource]
            possible_processes.extend(additional_resources)

        return possible_processes

    @classmethod
    def determine_process_provider(cls, x):
        resources = cls.get_resources()
        possible_processes_resource = cls.get_possible_processes_all(x="ALL")
        possible_processes_resource = {k.get_static_model_id()[1:]: v
                                       for k, v in possible_processes_resource.items()}

        process_provider = {}
        for resource, process_list in possible_processes_resource.items():
            resource_agent_name = [resource_
                                   for resource_ in resources
                                   if resource_.get_static_model_id()[1:] == resource][0]
            for process in process_list:
                process_provider.setdefault(process,
                                            []).append(resource_agent_name)

        return process_provider

    @classmethod
    def determine_entity_provider(cls, x) -> dict[EntityType: list[Resource]]:
        """entity_type: list of warehouses that can provide the part"""
        if x:
            resources = cls.get_resources()
            warehouses = [resource_
                          for resource_ in resources
                          if isinstance(resource_, Warehouse) or
                          isinstance(resource_, Storage) and resource_.situated_in is None]
            warehouse_possible_entity_types = [(warehouse, warehouse.get_possible_entity_types_to_store())
                                               for warehouse in warehouses]
            entity_provider = {}
            for warehouse, possible_entity_types in warehouse_possible_entity_types:
                for possible_entity_type in possible_entity_types:
                    entity_provider.setdefault(possible_entity_type, []).append(warehouse)
            entity_types_added = [entity_type.identification for entity_type in list(entity_provider.keys())]

            if len(entity_types_added) > len(list(set(entity_types_added))):
                raise ValueError(f"Problem: the same object is in the keys ...")

            return entity_provider
        else:
            return {}

    @classmethod
    def determine_transport_provider(cls, x):
        if x:
            resources = cls.get_resources()
            transport_provider = {}
            active_moving_transport_resources = [resource_
                                                 for resource_ in resources
                                                 if isinstance(resource_, ActiveMovingResource)]
            for active_moving_resource in active_moving_transport_resources:
                for storage_entity_type, storage_places in active_moving_resource.get_storages().items():
                    transport_provider.setdefault(storage_entity_type,
                                                  []).append(active_moving_resource)

            conveyor_belt_transport_resources = [resource_
                                                 for resource_ in resources
                                                 if isinstance(resource_, ConveyorBelt)]
            for conveyor_belt in conveyor_belt_transport_resources:
                for allowed_entity_type in conveyor_belt.allowed_entity_types:
                    transport_provider.setdefault(allowed_entity_type, []).append(conveyor_belt)

            return transport_provider
        else:
            return None

    @classmethod
    def determine_value_added_processes(cls, x):
        return cls.state_model.processes[ValueAddedProcess]

    @classmethod
    def get_order_pool(cls, x):
        return cls.state_model.get_orders()


format_function_mapper = {"string": None,
                          "integer": convert_str_to_int,
                          "string_list": convert_str_to_list,
                          "string_dict": convert_str_to_dict,
                          "datetime": convert_str_to_datetime,
                          "float": handle_numerical_value,
                          "tuple": None,
                          "boolean": None,
                          "selection": None,
                          "dt_object": None,
                          'state_model': MapperMethods.determine_digital_twin,
                          'organization': MapperMethods.determine_agents,
                          'change_handler': MapperMethods.determine_change_handler,
                          'possible_processes': MapperMethods.determine_possible_processes,  # ToDo
                          'process_provider': MapperMethods.determine_process_provider,
                          'entity_provider': MapperMethods.determine_entity_provider,
                          'transport_provider': MapperMethods.determine_transport_provider,
                          'value_added_processes': MapperMethods.determine_value_added_processes,
                          'order_pool': MapperMethods.get_order_pool}
