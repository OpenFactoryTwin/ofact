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

Used to duplicate the digital twin model

@contact persons: Adrian Freiter
"""

from copy import deepcopy

from ofact.twin.repository_services.light_digital_twin_model_mapper import (
    DigitalTwinModelMapper, LightDigitalTwinModelMapper)


def get_digital_twin_model_duplicate(digital_twin_model):
    """The method is used to make a duplicate of the whole digital twin model."""

    try:
        # should be much faster
        digital_twin_duplicated = deepcopy(digital_twin_model)
    except:  # to many references
        digital_twin_model_mapper = DigitalTwinModelMapper(digital_twin_model=digital_twin_model)
        digital_twin_model_light_dict = digital_twin_model_mapper.create_digital_twin_objects_as_key_value()
        light_digital_twin_model_mapper = (
            LightDigitalTwinModelMapper(digital_twin_model_light_dict=digital_twin_model_light_dict))
        digital_twin_duplicated = light_digital_twin_model_mapper.get_digital_twin_model_by_key_value()

    return digital_twin_duplicated