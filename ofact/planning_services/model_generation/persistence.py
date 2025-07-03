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

This file is used to serialize and deserialize the digital twin state model from and to different persistence formats.

@last update: 24.10.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional
from pathlib import Path

# Imports Part 3: Project Imports
import ofact.twin.repository_services.persistence as persistence
from ofact.planning_services.model_generation.static_state_model_generator import StaticStateModelGenerator
from ofact.twin.repository_services.deserialization.order_types import OrderType

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel


def deserialize_state_model(source_file_path: Union[str, Path], persistence_format: str = "pkl", use_db: bool = False,
                            dynamics: bool = False, deserialization_required: bool = True,
                            state_model_generation_settings: Optional[dict] = None,
                            store_state_model: bool = True) \
        -> StateModel:
    """
    Deserialize the state model from the source file of the persistence_format.

    Parameters
    ----------
    source_file_path : Union[str, Path]
        The path to the source file
    persistence_format : str
        The persistence format to be used
    use_db : bool
        Whether to use the database or not
    dynamics : bool
        Whether to use dynamics or not (process executions available)
    deserialization_required : bool
        Whether to deserialize the state model or not
    state_model_generation_settings : Optional[dict]
        The settings for the state model generation
    store_state_model : bool
        Whether to store the state model or not
    """

    if persistence_format != "xlsx":
        state_model = persistence.deserialize_state_model(source_file_path=source_file_path,
                                                          persistence_format=persistence_format, use_db=use_db,
                                                          dynamics=dynamics,
                                                          deserialization_required=deserialization_required)
        return state_model

    if not state_model_generation_settings:
        state_model_generation_settings: dict = {}

    if "static_model_importer_class" not in state_model_generation_settings:
        static_model_importer_class = StaticStateModelGenerator
    else:
        static_model_importer_class = state_model_generation_settings["static_model_importer_class"]

    # ToDo: should be changed adapted
    if "customer_generation_from_excel" not in state_model_generation_settings:
        CUSTOMER_GENERATION_FROM_EXCEL = False
    else:
        CUSTOMER_GENERATION_FROM_EXCEL = state_model_generation_settings["customer_generation_from_excel"]
    if "order_generation_from_excel" not in state_model_generation_settings:
        ORDER_GENERATION_FROM_EXCEL = False
    else:
        ORDER_GENERATION_FROM_EXCEL = state_model_generation_settings["order_generation_from_excel"]
    if "customer_amount" not in state_model_generation_settings:
        CUSTOMER_AMOUNT = 0
    else:
        CUSTOMER_AMOUNT = state_model_generation_settings["customer_amount"]
    if "order_amount" not in state_model_generation_settings:
        ORDER_AMOUNT = 0
    else:
        ORDER_AMOUNT = state_model_generation_settings["order_amount"]
    if "order_type" not in state_model_generation_settings:
        ORDER_TYPE = OrderType.SHOPPING_BASKET
    else:
        ORDER_TYPE = state_model_generation_settings["order_type"]

    static_model_importer = \
        static_model_importer_class(source_file_path,
                                    CUSTOMER_GENERATION_FROM_EXCEL=CUSTOMER_GENERATION_FROM_EXCEL,
                                    ORDER_GENERATION_FROM_EXCEL=ORDER_GENERATION_FROM_EXCEL,
                                    CUSTOMER_AMOUNT=CUSTOMER_AMOUNT, ORDER_AMOUNT=ORDER_AMOUNT,
                                    ORDER_TYPE=ORDER_TYPE)

    state_model = static_model_importer.get_state_model()

    if store_state_model:
        if isinstance(source_file_path, str):
            source_file_path = source_file_path.split(".")[0] + ".pkl"
        else:
            source_file_path = str(source_file_path).split(".")[0] + ".pkl"

        serialize_state_model(state_model=state_model, target_file_path=source_file_path, dynamics=False,
                              serialization_required=False)

    return state_model


def serialize_state_model(state_model: StateModel, target_file_path: Union[str, Path],
                          mapping_file_path: Union[str, Path] = None,
                          use_db: bool = False, dynamics: bool = False, serialization_required: bool = True):
    """
    Serialize the state model to the target_file.
    The format can be found in the mapping file.

    Parameters
    ----------
    state_model : StateModel
        The state model to be exported
    target_file_path : Union[str, Path]
        The path to the target file
    mapping_file_path : Union[str, Path]
        The mapping_file_path refers to the settings file.
    use_db : bool
        Whether to use the database or not
    dynamics : bool
        Whether to use dynamics or not
    serialization_required : bool
        Whether to serialize the state model or not
    """

    persistence.serialize_state_model(state_model=state_model, target_file_path=target_file_path,
                                      mapping_file_path=mapping_file_path, use_db=use_db, dynamics=dynamics,
                                      serialization_required=serialization_required)


def get_state_model_file_path(state_model_file_path=None, state_model_file_name=None, project_path=None,
                              path_to_model="models/twin/"):
    """Return the state model file path"""

    if state_model_file_path is None:
        if state_model_file_name is None:
            raise Exception("Please provide either the digital_twin_file_path or the digital_twin_file_name")

        # digital_twin initialisation
        file_path = Path(str(project_path), path_to_model + state_model_file_name)

    else:
        file_path = state_model_file_path

    return file_path
