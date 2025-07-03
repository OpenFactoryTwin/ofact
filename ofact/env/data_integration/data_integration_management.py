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
raise NotImplementedError()

# Imports Part 1: Standard Imports
from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

# Imports Part 2: PIP Imports
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

# Imports Part 3: Project Imports
from ofact.env.environment import Environment
from ofact.env.model_administration.cache import ObjectCacheDataIntegration
from ofact.env.model_administration.mapper_initialization import get_data_transformation_model
from ofact.env.data_integration.data_processing import DataProcessing
from ofact.env.data_integration.state_model_updating import StateModelUpdating

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel

# Module-Specific Constants



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

    def __init__(self, root_path: str, project_path: str, data_source_model_path: str, state_model: StateModel,
                 change_handler, start_time=None, progress_tracker=None, artificial_simulation_need=False):
        super(DataTransformationManagement, self).__init__(change_handler=change_handler, start_time=start_time)
        self.progress_tracker = progress_tracker

        self.root_path = root_path
        self.project_path = project_path
        self.data_source_model_path = data_source_model_path  # from the data_source_model

        self._state_model = state_model

        self.cache = {}

        object_cache = ObjectCacheDataIntegration()
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
        sources, storages, aggregation_combinations_with_connection = (
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
        sources, storages, aggregation_combinations_with_connection = (
            get_data_transformation_model(self.data_source_model_path, self.project_path))

        return sources, storages, aggregation_combinations_with_connection
