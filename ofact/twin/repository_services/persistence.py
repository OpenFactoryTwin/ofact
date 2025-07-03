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

This file is used to import and export the digital twin from and to different sources/sinks.

@last update: 14.05.2024
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Union

# Imports Part 3: Project Imports
from ofact.twin.repository_services.serialization.dynamic_state_model import DynamicStateModelSerialization
from ofact.twin.repository_services.db.mongodb import StateModelMongoDB
from ofact.twin.repository_services.serialization.static_state_model import StaticModelStateModelSerialization
from ofact.twin.repository_services.deserialization.static_state_model import StaticStateModelDeserialization
from ofact.twin.repository_services.deserialization.dynamic_state_model import DynamicStateModelDeserialization
from ofact.twin.state_model.seralized_model import SerializedStateModel
from ofact.twin.state_model.model import StateModel

if TYPE_CHECKING:
    from pathlib import Path


def deserialize_state_model(source_file_path: Union[str, Path], persistence_format: str = "pkl", use_db: bool = False,
                            dynamics: bool = False, deserialization_required: bool = True) \
        -> StateModel:
    """
    Deserialize the state model from the source file.

    Parameters
    ----------
    source_file_path : Union[str, Path]
        The path to the source file
    persistence_format : str
        The persistence format to be used
    use_db : bool
        Whether to use the database or not
    dynamics : bool
        Whether to use dynamics or not
    deserialization_required : bool
        Whether to deserialize the state model or not
    """

    if dynamics:
        if use_db:
            db = StateModelMongoDB()
            deserializer = DynamicStateModelDeserialization(db=db)
            state_model_dict = deserializer.from_db()

        else:
            match persistence_format:
                case "json":
                    state_model_dict = SerializedStateModel.load_from_json(source_file_path)
                case "parquet":
                    state_model_dict = SerializedStateModel.load_from_parquet_folder(source_file_path)
                case "pkl":
                    state_model_dict = SerializedStateModel.load_from_pickle(source_file_path)
                case _:
                    raise ValueError(f"Unknown persistence format: {persistence_format}")

            deserializer = DynamicStateModelDeserialization()

        state_model = deserializer.get_state_model(state_model_dict)

    else:
        if deserialization_required:
            deserializer = StaticStateModelDeserialization(source_file_path)
            state_model = deserializer.get_state_model()
        else:
            state_model = StateModel.from_pickle(source_file_path)

    return state_model

def serialize_state_model(state_model: StateModel, target_file_path: Union[str, Path],
                          mapping_file_path: Union[str, Path] = None,
                          use_db: bool = False, dynamics: bool = False, serialization_required: bool = True):
    """
    Serialize the state model to the target_file.

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

    if dynamics:
        if use_db:
            db = StateModelMongoDB()
        else:
            db = None

        if mapping_file_path is not None:
            exporter = DynamicStateModelSerialization(state_model=state_model, mapping_file=mapping_file_path, db=db)
        else:
            exporter = DynamicStateModelSerialization(state_model=state_model, db=db)
    else:
        if serialization_required:
            exporter = StaticModelStateModelSerialization(state_model=state_model)
        else:
            state_model.to_pickle(target_file_path)
            print(datetime.now(), "State Model Serialization finished.")
            return

    exporter.export(target_file_path)
    print(datetime.now(), "State Model Serialization finished.")
