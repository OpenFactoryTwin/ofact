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

This file is used to read the digital twin or the kpi tables/ read them from the pkl storages
"""
from __future__ import annotations
from typing import TYPE_CHECKING

import dill as pickle

from ofact.twin.repository_services.light_digital_twin_model_mapper import LightDigitalTwinModelMapper

if TYPE_CHECKING:
    from pathlib import Path
    from ofact.twin.state_model.model import StateModel



def read_state_model(pickle_path: str | Path) -> [StateModel, LightDigitalTwinModelMapper]:
    light_digital_twin_model_mapper = LightDigitalTwinModelMapper(pickle_path=pickle_path)
    digital_twin_model = light_digital_twin_model_mapper.get_digital_twin_model_by_key_value()

    return digital_twin_model
