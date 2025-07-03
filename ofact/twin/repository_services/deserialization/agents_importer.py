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

Instantiation of the agents based on an Excel file and creating of the agents model.

@contact persons: Adrian Freiter
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.agent_control.basic import DigitalTwinAgent
from ofact.twin.agent_control.behaviours.planning.tree.preference import EntityPreference
from ofact.twin.agent_control.information_desk import InformationServiceAgent
# ToDo: later the highest hierarchy
# local packages
from ofact.twin.agent_control.order import OrderDigitalTwinAgent, OrderPoolDigitalTwinAgent
from ofact.twin.agent_control.resource import (WorkStationAgent, WarehouseAgent, TransportAgent,
                                               ResourceDigitalTwinAgent)
from ofact.twin.agent_control.scheduling_coordinator import SchedulingCoordinatorAgent
from ofact.twin.repository_services.deserialization.old_basic_file_loader import (
    ObjectInstantiation, Mapping, convert_str_to_list, convert_str_to_dict, split_df,
    execute_function_calls, combine_all_objects)
from ofact.twin.state_model.basic_elements import get_clean_attribute_name
from ofact.twin.state_model.entities import ActiveMovingResource, Warehouse, Storage, ConveyorBelt
from ofact.twin.state_model.processes import ValueAddedProcess

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel
    from ofact.twin.state_model.entities import EntityType, Resource
    from ofact.twin.change_handler.change_handler import ChangeHandler


class MapperMethods:
    digital_twin = None
    agents = None
    change_handler = None
    possible_processes_resource_memo = {}

    def __init__(self, digital_twin: StateModel, agents: Agents, change_handler: ChangeHandler):
        """
        Enable the access to the digital_twin model for the mapping methods

        Parameters
        ----------
        digital_twin: digital_twin model object
        agents: the agents_model
        change_handler: interface between the agents, digital_twin and environment
        """
        type(self).digital_twin = digital_twin

        MapperMethods.possible_processes_resource_memo = {}

        type(self).agents = agents
        type(self).change_handler = change_handler

    @classmethod
    def get_resources(cls):
        stationary_resources = [item for sub_list in cls.digital_twin.stationary_resources.values()
                                for item in sub_list]
        active_moving_resources = [item for sub_list in list(cls.digital_twin.active_moving_resources.values())
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

        processes = [item for sub_list in list(cls.digital_twin.processes.values()) for item in sub_list]
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
            return cls.digital_twin
        else:
            return None

    @classmethod
    def determine_agents(cls, x):
        if x:
            return cls.agents
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
        possible_processes = []
        for resource in resource_s:
            additional_resources = possible_processes_all[resource]
            possible_processes.extend(additional_resources)

        return possible_processes

    @classmethod
    def determine_process_provider(cls, x):
        resources = cls.get_resources()
        possible_processes_resource = cls.get_possible_processes_all(x="ALL")

        process_provider = {}
        for resource, process_list in possible_processes_resource.items():
            resource_agent_name = [resource_ for resource_ in resources if resource_ == resource][0]
            for process in process_list:
                process_provider.setdefault(process, []).append(resource_agent_name)

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
        return cls.digital_twin.processes[ValueAddedProcess]

    @classmethod
    def get_order_pool(cls, x):
        return cls.digital_twin.get_orders()


# #### mappings  #######################################################################################################


class MappingAgents(Mapping):
    """
    Used to transform the Factory elements from the Excel sheets to python classes.
    """

    mappings = {
        'OrderPoolAgent': OrderPoolDigitalTwinAgent,
        'OrderAgent': OrderDigitalTwinAgent,
        'ResourceAgent': ResourceDigitalTwinAgent,
        'WorkStationAgent': WorkStationAgent,
        'WarehouseAgent': WarehouseAgent,
        'TransportAgent': TransportAgent,
        'SchedulingCoordinatorAgent': SchedulingCoordinatorAgent,
        'InformationServiceAgent': InformationServiceAgent,
        'Preference': EntityPreference
    }

    object_columns = {
        'state_model': None,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,  # ToDo
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': convert_str_to_dict,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }

    to_be_defined_object_columns = {
        'state_model': MapperMethods.determine_digital_twin,
        'organization': MapperMethods.determine_agents,
        'change_handler': MapperMethods.determine_change_handler,
        'address_book': convert_str_to_dict,
        'processes': convert_str_to_dict,
        'resources': convert_str_to_list,
        'preferences': convert_str_to_list,
        'possible_processes': MapperMethods.determine_possible_processes,
        'entity_type_to_store': convert_str_to_list,
        'process_provider': MapperMethods.determine_process_provider,
        'entity_provider': MapperMethods.determine_entity_provider,
        'transport_provider': MapperMethods.determine_transport_provider,
        'value_added_processes': MapperMethods.determine_value_added_processes,
        'order_pool': MapperMethods.get_order_pool,
        'reference_objects': convert_str_to_list
    }


# #### creating agents_model  ##########################################################################################

def get_object_dicts_by_class(dict_, class_):
    return {object_name: object_
            for object_name, object_ in dict_.items()
            if isinstance(object_, class_)}


def get_required_objects(digital_twin_model: StateModel):

    all_resources = digital_twin_model.get_all_resources()
    entity_types = digital_twin_model.get_entity_types()
    processes = digital_twin_model.get_all_processes()

    required_objects_list = all_resources + entity_types + processes
    required_objects = {(dt_object.__class__.__name__, get_clean_attribute_name(static_attribute)): dt_object
                        for dt_object in required_objects_list
                        if "static_model" in dt_object.external_identifications
                        for static_attribute in dt_object.external_identifications["static_model"]}

    return required_objects


class AgentsObjects:

    def __init__(self, path, digital_twin: StateModel, empty_agents_model: Agents, mapping_class=MappingAgents,
                 order_agent_amount=None):
        """
        Create the agents_model from the Excel file.

        Parameters
        ----------
        path: path to the excel file
        digital_twin:
        empty_agents_model:
        order_agent_amount: for a subsequent adaption of the number order agents in the system usable
        """
        super().__init__()
        self.path = path
        self._all_objects_digital_twin = get_required_objects(digital_twin)

        self.mapping_class = mapping_class
        self.agents_model = empty_agents_model

        # instantiation of the Excel files
        self.object_instantiation = ObjectInstantiation()
        self.create_agents_from_excel(order_agent_amount)  # self.agent_objects
        print("agents from excel")
        self.execute_function_calls()

        # agents
        self.agent_objects = {(object_type, agent_name): agent_object
                              for (object_type, agent_name), agent_object in self.agent_objects.items()
                              if isinstance(agent_object, DigitalTwinAgent)}
        self.order_pool_agents: dict = get_object_dicts_by_class(self.agent_objects, OrderPoolDigitalTwinAgent)
        self.order_agents: dict = get_object_dicts_by_class(self.agent_objects, OrderDigitalTwinAgent)
        self.resource_agents: dict = get_object_dicts_by_class(self.agent_objects, ResourceDigitalTwinAgent)
        self.work_station_agents: dict = get_object_dicts_by_class(self.agent_objects, WorkStationAgent)
        self.warehouse_agents: dict = get_object_dicts_by_class(self.agent_objects, WarehouseAgent)
        self.transport_agents: dict = get_object_dicts_by_class(self.agent_objects, TransportAgent)
        self.coordinator_agents: dict = get_object_dicts_by_class(self.agent_objects, SchedulingCoordinatorAgent)
        self.information_service_agents: dict = get_object_dicts_by_class(self.agent_objects, InformationServiceAgent)

        self.agents = combine_all_objects(self.order_pool_agents, self.order_agents,
                                          self.resource_agents, self.work_station_agents,
                                          self.warehouse_agents, self.transport_agents, self.coordinator_agents,
                                          self.information_service_agents)

        self.handle_duplications()

        # agent_model
        self.create_agents_model()  # self.agents_model

        print("Agents model created")

    # ==== create_objects ==============================================================================================

    def create_agents_from_excel(self, order_agent_amount):
        """
        self.agent_objects: e.g. {(object_type, object_name), object_}
        """
        unprepared_df = self._read_agent_df_from_excel()

        # used to change the number of order agents subsequently
        if order_agent_amount is not None:
            order_agents = [agent_type for agent_type, agent_name in unprepared_df.index if agent_type == "OrderAgent"]
            if order_agents:
                unprepared_df.loc["OrderAgent", "amount"] = order_agent_amount

        # prepare the dataframe/ combine more than one cell with each other
        # because a cell in excel has a character limit
        if len(list(set(unprepared_df.index))) < len(list(unprepared_df.index)):
            df = pd.DataFrame()
            for idx in list(set(unprepared_df.index)):
                # merge the columns
                new_row = unprepared_df.loc[idx].head(1)
                advanced_data = unprepared_df.loc[idx].tail(-1)
                advanced_data = advanced_data.dropna(axis=1)
                if not advanced_data.empty:
                    for col in advanced_data.columns:
                        new_row[col] += advanced_data[col]
                df = pd.concat([df, new_row])
        else:
            df = unprepared_df

        if "Function calls" in df.columns:
            self.func_calls_agents_df, self.agents_df = split_df(df=df, intersection_point=-2)
        else:
            self.agents_df = df
            self.func_calls_agents_df = None

        if "amount" in self.agents_df.columns:
            correction_needed = True

        else:
            correction_needed = False

        self.agent_objects, self.corrected_agents_dict = \
            self.object_instantiation.load_dict(object_df=self.agents_df,
                                                mapping_class=self.mapping_class,
                                                input_objects=self._all_objects_digital_twin,
                                                correction_needed=correction_needed)
        print("correction_needed available")
        if not correction_needed:
            return

        # update the agent_objects (stored_entities)
        self.agent_objects = \
            self.object_instantiation.instantiate_dependent_objects(independent_objects=self.agent_objects,
                                                                    objects_df=self.agents_df,
                                                                    mapping_class=self.mapping_class,
                                                                    find_in=self.agent_objects,
                                                                    check_independent_objects=False)

    def _read_agent_df_from_excel(self):
        unprepared_df = pd.read_excel(self.path, index_col=[0, 1], sheet_name="Agents", skiprows=1)

        return unprepared_df

    def handle_duplications(self):
        if self.corrected_agents_dict is None:
            return

        # adapt the address_book of all agents - currently not needed
        names_to_duplicate = {}
        for agent, duplicated_agents in self.corrected_agents_dict.items():
            names_to_duplicate[agent.name] = []
            for duplicated_agent in duplicated_agents:
                names_to_duplicate[agent.name].append(duplicated_agent.name)

    def execute_function_calls(self):
        if self.func_calls_agents_df is not None:
            execute_function_calls(self.func_calls_agents_df, self.agent_objects)

    def create_agents_model(self):
        """create the agents_model object"""
        agents = {}
        for agent_name, agent_object in self.agents.items():
            agents.setdefault(agent_object.__class__, []).append(agent_object)
        self.agents_model.agents = agents

    # ==== agents_specific_methods =====================================================================================

    def get_agents(self):
        return self.agents_model


if __name__ == '__main__':
    import os
    from ofact.planning_services.model_generation.static_state_model_generator import StaticStateModelGenerator
    from ofact.twin.agent_control.organization import Agents
    from ofact.env.simulation.event_discrete import EventDiscreteSimulation
    from ofact.twin.change_handler.change_handler import ChangeHandlerSimulation
    from ofact.settings import ROOT_PATH
    # we need one path to the Excel file containing a sheet for all the following:
    # - plant, process and order

    digital_twin_path = os.path.join(ROOT_PATH, 'projects\\bicycle_world\\models\\bicycle_world.xlsx')

    state_model_generator = StaticStateModelGenerator(digital_twin_path)
    digital_twin = state_model_generator.get_state_model()

    simulation = EventDiscreteSimulation()
    empty_agents_model = Agents()
    change_handler = ChangeHandlerSimulation(digital_twin=digital_twin,
                                             environment=simulation,
                                             agents=empty_agents_model)
    # objects
    MapperMethods(digital_twin, empty_agents_model, change_handler)
    agent_objects = AgentsObjects(path=digital_twin_path, digital_twin=digital_twin,
                                  empty_agents_model=empty_agents_model)
    agents_model = agent_objects.get_agents()

    print('#' * 5, "AGENTS", '#' * 5)
    print(agents_model.agents)

