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

import json
from pathlib import Path
from typing import TYPE_CHECKING
import os

# Imports Part 2: PIP Imports
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.agent_control.basic import DigitalTwinAgent
from ofact.twin.agent_control.information_desk import InformationServiceAgent
from ofact.twin.agent_control.model import AgentsModel
from ofact.twin.agent_control.order import OrderDigitalTwinAgent, OrderPoolDigitalTwinAgent
from ofact.twin.agent_control.resource import (WorkStationAgent, WarehouseAgent, TransportAgent,
                                               ResourceDigitalTwinAgent)
from ofact.twin.agent_control.scheduling_coordinator import SchedulingCoordinatorAgent
from ofact.twin.repository_services.deserialization.agent_mapping import agent_model_mapper, format_function_mapper, \
    MapperMethods
from ofact.twin.repository_services.deserialization.basic_file_loader import ObjectInstantiation, combine_all_objects
from ofact.twin.state_model.basic_elements import get_clean_attribute_name
from ofact.settings import ROOT_PATH

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


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


def _get_mappings(mapping_file, state_model_format_function_mapper, agent_model_class_mapper) -> dict:
    """
    Get the mappings from the mapping file that are required to deserialize the state model.
    For example, the classes and in which excel sheet they are located.
    In addition, columns with the object (formats) expected for the attribute.

    Parameters
    ----------
    mapping_file : str
        path to the mapping file

    Returns
    -------
    dict
        dictionary of the mappings
    """

    sheet_mappings = {}
    mappings_dict = json.load(open(mapping_file, "r"))

    for sheet in mappings_dict["sheets"]:
        classes_dict = {class_name: agent_model_class_mapper[class_name]
                        for class_name in sheet["classes"]}
        columns_dict = {column_dict["name"]: state_model_format_function_mapper[column_dict["format"]]
                        for column_dict in sheet["columns"]}

        single_sheet_mappings = {"classes": classes_dict,
                                 "columns": columns_dict,
                                 "distributions": {},
                                 "source": "AgentsModel"}

        sheet_mappings["Mapping" + sheet["name"]] = single_sheet_mappings

    return sheet_mappings


class AgentsModelDeserialization:
    agents_model_format_function_mapper = format_function_mapper

    def __init__(self, path, state_model: StateModel, empty_agents_model: Agents,
                 mapping_file: str = "./agents_model_excel_mapping.json", agent_model_class_mapper = None,
                 order_agent_amount=None):
        """
        Create the agents_model from the Excel file.

        Parameters
        ----------
        path: path to the excel file
        state_model:
        empty_agents_model:
        order_agent_amount: for a subsequent adaption of the number order agents in the system usable
        """
        super().__init__()
        self.path = path
        self._all_objects_state_model = get_required_objects(state_model)

        mapping_file = Path(str(ROOT_PATH) + "/twin/repository_services", mapping_file)
        if not os.path.isfile(mapping_file):
            raise IOError(f"Mapping file {mapping_file} does not exist.")
        if agent_model_class_mapper is None:
            agent_model_class_mapper = agent_model_mapper
        self.mappings = _get_mappings(mapping_file, type(self).agents_model_format_function_mapper,
                                      agent_model_class_mapper)
        self.organization = empty_agents_model

        # instantiation of the Excel files
        self.object_instantiation = ObjectInstantiation()
        self.create_agents_from_excel(order_agent_amount)  # self.agent_objects
        print("agents from excel")

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

        self.create_agents_organization()
        # agent_model
        agents_model = self.get_agents_model(state_model)  # self.agents_model

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

        self.agents_df = df

        if "amount" in self.agents_df.columns:
            correction_needed = True

        else:
            correction_needed = False

        self.agent_objects, self.corrected_agents_dict = \
            self.object_instantiation.load_dict(object_df=self.agents_df,
                                                mapping_class=self.mappings["MappingAgents Model"],
                                                input_objects=self._all_objects_state_model,
                                                correction_needed=correction_needed)
        print("correction_needed available")
        if not correction_needed:
            return

        # update the agent_objects (stored_entities)
        self.agent_objects = \
            self.object_instantiation.instantiate_dependent_objects(independent_objects=self.agent_objects,
                                                                    objects_df=self.agents_df,
                                                                    mapping_class=self.mappings["MappingAgents Model"],
                                                                    find_in=self.agent_objects,
                                                                    check_independent_objects=False)

    def _read_agent_df_from_excel(self):
        unprepared_df = pd.read_excel(self.path, index_col=[0, 1], sheet_name="Agents Model", skiprows=1)

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

    def create_agents_organization(self):
        """create the agents_model object"""
        agents = {}
        for agent_name, agent_object in self.agents.items():
            agents.setdefault(agent_object.__class__, []).append(agent_object)
        self.organization.agents = agents

    def get_agents_model(self, state_model):
        """create the agents_model object"""

        agents_model = AgentsModel(order_pool_agents=self.order_pool_agents,
                    order_agents=self.order_agents,
                    resource_agents=self.resource_agents,
                    work_station_agents=self.work_station_agents,
                    warehouse_agents=self.warehouse_agents,
                    transport_agents=self.transport_agents,
                    scheduling_coordinator_agents=self.coordinator_agents,
                    information_service_agents=self.information_service_agents)

        agents_model.update_agent_model_through_state_model(state_model)

        return agents_model

    # ==== agents_specific_methods =====================================================================================

    def get_agents(self):
        return self.organization


if __name__ == '__main__':

    from ofact.planning_services.model_generation.static_state_model_generator import StaticStateModelGenerator
    from ofact.twin.agent_control.organization import Agents
    from ofact.env.simulation.event_discrete import EventDiscreteSimulation
    from ofact.twin.change_handler.change_handler import ChangeHandlerSimulation

    # we need one path to the Excel file containing a sheet for all the following:
    # - plant, process and order

    state_model_twin_path = "C:\\Users\\afreiter\\PycharmProjects\\ofact-intern\\projects\\dbs\\models\\twin\\learned_model_full_adapted.xlsx"
    agents_model_twin_path = "C:\\Users\\afreiter\\PycharmProjects\\ofact-intern\\projects\\dbs\\models\\agents\\container.xlsx"

    state_model_generator = StaticStateModelGenerator(state_model_twin_path,
                                                      CUSTOMER_GENERATION_FROM_EXCEL=True,
                                                      ORDER_GENERATION_FROM_EXCEL=True)
    state_model = state_model_generator.get_state_model()

    simulation = EventDiscreteSimulation(change_handler=None, start_time=None, processes_to_stop=set())
    empty_agents_model = Agents()
    change_handler = ChangeHandlerSimulation(digital_twin=state_model,
                                             environment=simulation,
                                             agents=empty_agents_model)
    # objects
    MapperMethods(state_model, empty_agents_model, change_handler)
    agent_objects = AgentsModelDeserialization(path=agents_model_twin_path, state_model=state_model,
                                               empty_agents_model=empty_agents_model)
    agents_model = agent_objects.get_agents()

    print('#' * 5, "AGENTS", '#' * 5)
    print(agents_model.agents)
