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

Module for storing and assigning data received by various inputs. For Example used by a resource-agent.
Todo: SenML - Sensor Measurement - maybe sensor information should be stored in a more unified way

Classes:
    Sensor: Store sensor data for any entity in the digital twin

@contact persons: Christian Schwede & Adrian Freiter
@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
import datetime
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.twin.state_model.basic_elements import DigitalTwinObject

if TYPE_CHECKING:
    from ofact.twin.state_model.entities import Entity


class Sensor(DigitalTwinObject):
    """
    Class to store sensor data for any entity in the digital twin
    Parameters
    name: designation of the sensor
    data: dictionary to save the data (e.g. key=timestamp and value=measured value)
    """

    def __init__(self,
                 identification: int,
                 name: str,
                 entity: Entity,
                 external_identifications: dict = {}):
        super().__init__(identification=identification, external_identifications=external_identifications)
        self.name: str = name
        self.entity: Entity = entity
        self.data = {}

    def copy(self):
        sensor_copy = super(Sensor, self).copy()
        sensor_copy.data = sensor_copy.data.copy()

        return sensor_copy

    def add_data_entry(self, key, value):
        self.data[key] = value

    def get_data_entries(self, start_time, end_time):
        requested_data = {}
        if start_time is None:
            start_time = datetime.datetime(1970, 1, 1)
        for timestamp, measured_value in self.data.items():
            if start_time <= timestamp <= end_time:
                requested_data[timestamp] = measured_value
        return requested_data
