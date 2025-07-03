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

Used to update a digital twin state model with data coming from the environment such as the shop floor.

classes:
    DataTransformationManagement

@author: Adrian Freiter
@last update: 31.10.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

# Imports Part 2: PIP Imports
import pandas as pd

from ofact.env.data_integration.data_processing import DataProcessing
from ofact.env.model_administration.helper import get_attr_value

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Imports Part 3: Project Imports
from ofact.env.data_integration.state_model_updating import StateModelUpdating
from ofact.env.environment import Environment

from ofact.env.interfaces.data_integration.adapter import XLSXAdapter, MSSQLAdapter, CSVAdapter

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel

# Module-Specific Constants

class ExternalIDMapper:
    next_id = 0

    def __init__(self):
        self.external_id_mapper = {}

    def get_internal_id(self, name_space, external_id):
        if (name_space, external_id) not in self.external_id_mapper:
            new_id = self._get_next_id()
            self.external_id_mapper[(name_space, external_id)] = new_id
            return new_id

        else:
            return self.external_id_mapper[(name_space, external_id)]

    def _get_next_id(self):
        next_id = type(self).next_id
        type(self).next_id += 1

        return next_id


class ObjectCache:
    """
    Caches the objects used in the data transformation/ integration until they are instantiated.
    """

    def __init__(self, class_names: list[str] = None):
        if class_names is None:
            class_names = []

        self._class_names = class_names
        self._object_memory: dict[str: dict[int: dict]] = {}
        self._order_process_executions: dict = {}

        # objects like Parts that should be planned individually to did not choose for two processes the same part
        self._objects_already_planned: dict[str, list] = {}
        self.external_id_mapper = ExternalIDMapper()

        # instantiation sequence is needed because ProcessExecution needs EntityType, ActiveM.., Ass.., Part etc.
        # already initialized (to use them as attributes) -> adaptable for other objects
        self._instantiation_sequence = ['Feature', 'Customer', 'Part', 'Order', 'EntityType', 'PartType',
                                        'Resource', 'Storage', 'Warehouse',
                                        'ActiveMovingResource', 'PassiveMovingResource',
                                        'WorkStation', 'Process', 'ValueAddedProcess', 'ProcessExecution']

    def store_object_dict(self, class_name, name_space, external_id, object_dict):
        if class_name not in self._object_memory:
            self._object_memory[class_name] = {}
            self._class_names.append(class_name)
        if isinstance(external_id, list):
            external_id = external_id[0]
        internal_id = self.external_id_mapper.get_internal_id(name_space, external_id)
        self._object_memory[class_name][internal_id] = object_dict

        if class_name != "ProcessExecution":
            return

        self.map_process_executions_to_orders(object_dict)

    def remove_object_dict(self, class_name, name_space, external_id):
        """remove objects that are for example not filled enough to be loaded"""

        if class_name not in self._object_memory:
            return

        internal_id = self.external_id_mapper.get_internal_id(name_space, external_id)
        if internal_id not in self._object_memory[class_name]:
            return

        del self._object_memory[class_name][internal_id]

    def update_order_process_executions(self):

        if "ProcessExecution" not in self._object_memory:
            return

        process_executions_dicts = list(self._object_memory["ProcessExecution"].values())
        for object_dict in process_executions_dicts:
            self.map_process_executions_to_orders(object_dict)

    def map_process_executions_to_orders(self, object_dict):

        order = get_attr_value(object_dict, "order")
        if isinstance(order, dict):
            external_identifications = get_attr_value(order, "external_identifications")
            order_ids = list(*external_identifications.values())
        else:
            order_ids = []

        for order_id in order_ids:
            self._order_process_executions.setdefault(order_id,
                                                      []).append(object_dict)

    def get_object_by_external_identification(self, name_space, external_id, class_name=None) -> dict | None:
        if class_name not in self._object_memory:
            return None

        internal_id = self.external_id_mapper.get_internal_id(name_space, external_id)
        if internal_id not in self._object_memory[class_name]:
            return None

        object_ = self._object_memory[class_name][internal_id]

        return object_

    def get_process_executions_order(self, order_id):

        if order_id not in self._order_process_executions:
            self.update_order_process_executions()

        if order_id in self._order_process_executions:
            process_executions = self._order_process_executions[order_id]
        else:
            process_executions = []

        return process_executions

    def pop(self) -> tuple[str | None, list[dict] | None]:
        """pop element from object_cache"""
        for type_ in self._instantiation_sequence:
            if type_ not in self._object_memory:
                continue

            dict_of_objects = self._object_memory[type_]
            if not dict_of_objects:
                continue

            elem_batch = list(dict_of_objects.values())
            del self._object_memory[type_]

            elem_batch = [elem for elem in elem_batch if isinstance(elem, dict)]

            if not elem_batch:
                continue

            return type_, elem_batch

        self._order_process_executions = {}

        return None, None

    def empty(self):
        if any(list(self._object_memory.values())):
            return False
        else:
            return True

    def store_object_already_planned(self, type_: str, object_):
        self._objects_already_planned.setdefault(type_,
                                                 []).append(object_)

    def get_objects_already_planned(self, type_: str) -> list:
        if type_ in self._objects_already_planned:
            return self._objects_already_planned[type_]
        else:
            return []

    def get_passive_moving_resources(self):
        if "PassiveMovingResource" in self._object_memory:
            return list(self._object_memory["PassiveMovingResource"].values())
        else:
            return {}

    def get_active_moving_resources(self):
        if "ActiveMovingResource" in self._object_memory:
            return list(self._object_memory["ActiveMovingResource"].values())
        else:
            return {}


def _get_source_settings(data_source_entry, project_path, adapter_dict,
                         time_restrictions_df, column_mappings_df, split_df, clean_up_df, filters_df, sort_df,
                         static_refinements_df):
    """Create the adapter object with the information given from the data source model, created in an Excel file."""

    if data_source_entry["mapping"][0] == "*":
        data_source_entry["mapping"] = data_source_entry["mapping"][1:]
        general = False
    else:
        general = True

    if data_source_entry['Input'] == "MSSQL":
        external_source_path = data_source_entry['path']
    else:
        external_source_path = os.path.join(project_path, os.path.normpath(data_source_entry['path']))

    time_columns = ["columns", "None values accepted", "Min one column filled"]
    time_restriction_df = (
        time_restrictions_df.loc[time_restrictions_df["mapping"] == data_source_entry["mapping"], time_columns])

    mapping_columns = ["external", "mapping identification", "mapping reference",
                       "class", "attribute", "handling", "depends on"]
    column_mappings_batch_df = (
        column_mappings_df.loc[column_mappings_df["mapping"] == data_source_entry["mapping"], mapping_columns])

    split_columns = ["external", "separator", "action", "operation id"]
    split_batch_df = (split_df.loc[split_df["mapping"] == data_source_entry["mapping"], split_columns])

    clean_up_columns = ["external", "old value", "replacing value", "delete"]
    clean_up_batch_df = (
        clean_up_df.loc[clean_up_df["mapping"] == data_source_entry["mapping"], clean_up_columns])

    filter_columns = ["external", "needed entries", "not needed entries", "contains"]
    filters_batch_df = filters_df.loc[filters_df["mapping"] == data_source_entry["mapping"], filter_columns]

    sort_columns = ["external"]
    sort_batch_df = sort_df.loc[sort_df["mapping"] == data_source_entry["mapping"], sort_columns]

    adapter_class = adapter_dict[data_source_entry["Input"]]
    adapter = adapter_class(external_source_path=external_source_path,
                            time_restriction_df=time_restriction_df,
                            column_mappings_df=column_mappings_batch_df,
                            split_df=split_batch_df,
                            clean_up_df=clean_up_batch_df,
                            filters_df=filters_batch_df,
                            sort_df=sort_batch_df)

    static_refinement_columns = ["type", "class", "attribute", "value"]
    static_refinements_batch_df = static_refinements_df.loc[
        static_refinements_df["mapping"] == data_source_entry["mapping"], static_refinement_columns]

    source_settings = {"adapter": adapter,
                       "data_source_entry": data_source_entry,
                       "general": general,
                       "static_refinements_batch_df": static_refinements_batch_df}

    return source_settings


def _get_aggregation_combinations_with_connection(column_mappings_df, aggregation_df):
    """Define the aggregation combinations with the information given from the data source model,
    created in an Excel file."""

    aggregation_combinations = list(zip(aggregation_df["first"].to_list(), aggregation_df["second"].to_list()))
    aggregation_combinations_with_connection = {}
    for (source_1, source_2) in aggregation_combinations:
        source_1_df = column_mappings_df.groupby(by="mapping").get_group(source_1)
        source_2_df = column_mappings_df.groupby(by="mapping").get_group(source_2)

        possible_references_source_1 = (
            source_1_df.loc[(source_1_df["attribute"] == "identification") &
                            (source_1_df["class"] == source_1_df["class"]), ["class", "attribute"]])
        possible_references_source_2 = (
            source_2_df.loc)[(source_2_df["attribute"] == "identification") &
                             (source_2_df["class"] == source_2_df["class"]), ["class", "attribute"]]

        aggregation_combinations_with_connection[(source_1, source_2)] = (
            list(set(possible_references_source_1["class"]).intersection(
                set(possible_references_source_2["class"]))))

    return aggregation_combinations_with_connection


class DataTransformationManagement(Environment):
    """
    The data transformation management is used to integrate the data from the shop floor into the digital twin.
    Therefore, it takes the incoming changes from the shop floor hearing the adapters.
    After it, the data is available in pd.DataFrame format.
    In the first data transformation step, the data is mapped to digital twin objects.
    Two points are important here:
    - the chronological order of the data sources can be important
    - the mapping can be standardized or domain specific
    If the objects are mapped the aggregation is used to include the source overlapping information.
    In the last step, the objects are instantiated and passed to the change_handler, who is responsible for
    the digital twin.

    Access point:
    update_digital_twin method
    """

    data_processing_class = DataProcessing
    state_model_updating_class = StateModelUpdating

    adapter_dict = {"Excel": XLSXAdapter,
                    "csv": CSVAdapter,
                    "MSSQL": MSSQLAdapter}  # AdapterMapper.AdapterTypes.xlsx

    def __init__(self, root_path: str, project_path: str, data_source_model_path: str, state_model: StateModel,
                 change_handler, start_time=None, progress_tracker=None, artificial_simulation_need=False):
        super(DataTransformationManagement, self).__init__(change_handler=change_handler, start_time=start_time)
        self.progress_tracker = progress_tracker

        self.root_path = root_path
        self.\
            project_path = project_path
        self.data_source_model_path = data_source_model_path  # from the data_source_model

        self._state_model = state_model

        self.cache = {}

        object_cache = ObjectCache()
        self._data_processing = (
            type(self).data_processing_class(object_cache=object_cache, state_model=self._state_model,
                                             change_handler=self.change_handler,
                                             progress_tracker=self.progress_tracker, cache=self.cache,
                                             dtm=self))
        self._state_model_updating = (
            type(self).state_model_updating_class(object_cache=object_cache, change_handler=self.change_handler,
                                                  state_model=self._state_model,
                                                  artificial_simulation_need=artificial_simulation_need,
                                                  progress_tracker=self.progress_tracker, cache=self.cache,
                                                  dtm=self))

    def add_change_handler(self, change_handler):
        super().add_change_handler(change_handler)
        self._data_processing.add_change_handler(change_handler)
        self._state_model_updating.add_change_handler(change_handler)

    def update_state_model(self, start_datetime: Optional[datetime] = None, end_datetime: Optional[datetime] = None):
        """
        Used to update the digital_twin model with data coming from the environment like the shop floor
        will use a given Table project/name/models/adapter_allocation.xlsx to evaluate adapters (port-interfaces)
        Called for example by frontend button "update".

        path                                                      |Input Type     |
        ---------------------------------------------------------------------------
        192.168.0.45/server/ofact_data/order_table.xlsx           | Excel         |
        10.0.0.5                                                  | SQL           |

        Then it will call based on "input type" the specific adapter.get_data(path).
         These will receive current data from external planning_services and return derived DT-objects.
        (1) Frontend Button „Update“ is triggered
        (2) DTM.update_digital_twin() is called
        (3) DTM determines based on the data_source - interface mapping (gegeben, Modell) necessary adapters
        (4) Adapter.get_data() is/ are called
        Data Transformation procedure is executed by passing the following steps:
        Phase 1: Get Data Transformation Model (How to map the objects, etc.)
        Phase 2: Read Data
        Phase 3: Create state model objects
        Phase 4: Aggregate Data
        Phase 5: Update the digital twin state model with new event data
        Phase 6: Apply Changes
        """

        print(f"[{datetime.now()}] Data Transformation Process started from '{start_datetime}' until '{end_datetime}'.")

        print(f"[{datetime.now()}] Phase 1: Get Data Transformation Model (How to map the objects, etc.)")
        sources, aggregation_combinations_with_connection, domain_specific_refinements = (
            self._get_data_transformation_model())
        data_batches_received = {}
        priorities = {}

        # ToDo: batches also possible
        print(f"[{datetime.now()}] Phase 2: Read Data")
        self._data_processing.read_data(sources=sources, start_datetime=start_datetime, end_datetime=end_datetime)

        # ToDo: create order traces would be more intuitive and decentralized
        self.track_progress(5.0)

        print(f"[{datetime.now()}] Phase 3: Create state model objects")
        data_batches_received, priorities = (
            self._data_processing.create_state_model_objects(sources, data_batches_received, priorities))

        self.track_progress(50.0)

        print(f"[{datetime.now()}] Phase 3: Aggregate Data")
        self._data_processing.aggregate_data(data_batches_received, aggregation_combinations_with_connection,
                                             domain_specific_refinements)

        self.track_progress(75.0)

        process_models_updates = self._state_model_updating.get_process_model_updates()

        print(f"[{datetime.now()}] Phase 4: Update the digital twin state model with new event data")
        instantiated_resources = self._state_model_updating.store_into_state_model()

        print(f"[{datetime.now()}] Phase 5: Update the process models")
        self._state_model_updating.update_process_models(process_models_updates, instantiated_resources)

        self.track_progress(99.0)

        return self._state_model

    def track_progress(self, progress_level: float):
        if self.progress_tracker is None:
            return

        self.progress_tracker.announce(progress_level)

    def _get_data_transformation_model(self):
        """
        Define data source and transformation steps

        Returns
        -------
        a list with sources and planned transformation steps
        """
        adapter_allocation_df = pd.read_excel(self.data_source_model_path, sheet_name="adapter allocation")
        column_mappings_df = pd.read_excel(self.data_source_model_path, sheet_name="column mappings")

        split_df = pd.read_excel(self.data_source_model_path, sheet_name="split")
        clean_up_df = pd.read_excel(self.data_source_model_path, sheet_name="clean up")
        filters_df = pd.read_excel(self.data_source_model_path, sheet_name="filters")
        sort_df = pd.read_excel(self.data_source_model_path, sheet_name="sort by")
        static_refinements_df = pd.read_excel(self.data_source_model_path, sheet_name="static refinements")
        time_restrictions_df = pd.read_excel(self.data_source_model_path, sheet_name="time restriction columns")
        sources = [_get_source_settings(data_source_entry=data_source_entry, project_path=self.project_path,
                                        adapter_dict=type(self).adapter_dict, time_restrictions_df=time_restrictions_df,
                                        column_mappings_df=column_mappings_df, split_df=split_df,
                                        clean_up_df=clean_up_df, filters_df=filters_df, sort_df=sort_df,
                                        static_refinements_df=static_refinements_df)
                   for idx, data_source_entry in adapter_allocation_df.iterrows()]

        aggregation_df = pd.read_excel(self.data_source_model_path, sheet_name="aggregation")
        aggregation_combinations_with_connection = (
            _get_aggregation_combinations_with_connection(column_mappings_df, aggregation_df))

        domain_specific_refinements = {source["data_source_entry"]["mapping"]: source["static_refinements_batch_df"]
                                       for source in sources}

        return sources, aggregation_combinations_with_connection, domain_specific_refinements
