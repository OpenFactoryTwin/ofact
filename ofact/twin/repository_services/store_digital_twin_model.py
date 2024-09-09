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

This file is used to store the digital twin or the kpi tables/ make them persistent beyond RAM memory
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union
import dill as pickle

from ofact.twin.repository_services.light_digital_twin_model_mapper import DigitalTwinModelMapper

if TYPE_CHECKING:
    from pathlib import Path
    from ofact.twin.state_model.model import StateModel


def store_state_model(digital_twin_model: StateModel, digital_twin_result_path: Union[str, Path]):
    digital_twin_model_mapper = DigitalTwinModelMapper(digital_twin_model=digital_twin_model)
    digital_twin_model_mapper.create_digital_twin_objects_as_key_value()
    digital_twin_model_mapper.to_pickle(digital_twin_result_path)
