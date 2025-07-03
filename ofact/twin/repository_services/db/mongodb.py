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

Used for the persistence in mongodb.
"""

from __future__ import annotations
from abc import abstractmethod, ABCMeta
from typing import Optional

try:
    from pymongo import MongoClient
except ModuleNotFoundError:
    pass

class StateModelDB(metaclass=ABCMeta):

    @abstractmethod
    def persist(self, data, scenario_name: Optional[str] = None):
        pass

    @abstractmethod
    def read(self, scenario_name: Optional[str] = None):
        pass


class StateModelMongoDB(StateModelDB):

    def __init__(self):
        # Provide the mongodb url to connect python to mongodb using pymongo
        CONNECTION_STRING = "mongodb://localhost:27017/"

        # Create a connection using MongoClient
        self.client = MongoClient(CONNECTION_STRING)

        # Create or access a database
        self.db = self.client['state_model_db']

    def persist(self, data, scenario_name: Optional[str] = None):
        if scenario_name is None:
            scenario_name = "state_model"

        # Create collection
        col = self.db[scenario_name]  # scenario dependent and individual

        data = col.insert_one(data)
        print(data.inserted_id)

    def read(self, scenario_name: Optional[str] = None):
        if scenario_name is None:
            scenario_name = "state_model"

        # Create collection
        col = self.db[scenario_name]  # scenario dependent and individual
        data = col.find()
        return data
