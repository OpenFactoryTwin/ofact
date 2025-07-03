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

Used to export the digital twin state model to different persistence formats.

Classes:
    Exporter:

"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import inspect
import json
from datetime import datetime
from enum import Enum
from os import path
from pathlib import Path
from typing import TYPE_CHECKING

# Imports Part 2: PIP Imports
# Imports Part 3: Project Imports
from ofact.settings import ROOT_PATH
from ofact.twin.state_model.seralized_model import SerializedStateModel
from ofact.twin.state_model.serialization import further_serialization

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel

ROOT_PATH = str(ROOT_PATH).rsplit("ofact", 1)[0] + "ofact"


class TargetSchema(Enum):
    Xlsx = 1
    Pkl = 2
    Json = 3
    Parquet = 4
    Db = 5

    @staticmethod
    def from_string(input: str):
        match input.lower():
            case "xlsx":
                return TargetSchema.Xlsx
            case "pkl":
                return TargetSchema.Pkl
            case "json":
                return TargetSchema.Json
            case "parquet":
                return TargetSchema.Parquet
            case "db":
                return TargetSchema.Db
            case _:
                raise NotImplementedError(f"Target schema {input} not supported.")


class ReferenceType:

    @staticmethod
    def validate_string(input: str):
        match input.lower():
            case "label":
                return "label"
            case "identification":
                return "identification"
            case _:
                raise NotImplementedError(f"Reference type {input} not supported.")


def _get_all_class_inheritance_trees(state_model_classes):
    state_model_class_parent_classes: dict[str, list[str]] = {}  # Including the own class!
    irrelevant_classes = ["object", 'InstantiationFromDict', 'Serializable']
    for state_model_class in state_model_classes:
        parent_classes = inspect.getmro(state_model_class)
        state_model_class_parent_classes[state_model_class.__name__] = [class_.__name__
                                                                        for class_ in parent_classes
                                                                        if class_.__name__ not in irrelevant_classes]
    return state_model_class_parent_classes


def _get_enriched_drop_before_serialization(drop_before_serialization, state_model_class_parent_classes):
    drop_before_serialization_enriched = {}

    class_names_to_go_through = list(state_model_class_parent_classes.keys()) + list(drop_before_serialization.keys())
    for class_name in class_names_to_go_through:
        if class_name not in state_model_class_parent_classes:
            if class_name in drop_before_serialization:
                drop_before_serialization_enriched[class_name] = drop_before_serialization[class_name]
            continue

        drop_before_serialization_enriched[class_name] = []
        for inheritance_class_name in state_model_class_parent_classes[class_name]:
            if inheritance_class_name in drop_before_serialization:
                drop_before_serialization_enriched[class_name].extend(
                    drop_before_serialization[inheritance_class_name])

        drop_before_serialization_enriched[class_name] = list(set(drop_before_serialization_enriched[class_name]))

    return drop_before_serialization_enriched


def _get_drop_before_serialization(mapping):
    drop_before_serialization = {}
    for sheet_dict in mapping["sources"]:
        for class_ in sheet_dict["classes"]:
            drop_before_serialization.setdefault(class_, [])
            if "drop" in sheet_dict:
                drop_before_serialization[class_].extend(sheet_dict["drop"])

    return drop_before_serialization


def _get_enriched_further_serializable(state_model_class_parent_classes):
    further_serialization_enriched = {}

    class_names_to_go_through = list(state_model_class_parent_classes.keys()) + list(further_serialization.keys())
    for class_name in class_names_to_go_through:
        if class_name not in state_model_class_parent_classes:
            if class_name in further_serialization:
                further_serialization_enriched[class_name] = further_serialization[class_name]
            continue

        further_serialization_enriched[class_name] = {}
        for inheritance_class_name in state_model_class_parent_classes[class_name]:
            if inheritance_class_name in further_serialization:
                for serializable_type, attributes_list in further_serialization[inheritance_class_name].items():
                    if serializable_type in further_serialization_enriched[class_name]:
                        further_serialization_enriched[class_name][serializable_type].extend(attributes_list.copy())
                    else:
                        further_serialization_enriched[class_name][serializable_type] = attributes_list.copy()

        further_serialization_enriched[class_name] = \
            {serializable_type: list(set(attributes_list))
             for serializable_type, attributes_list in further_serialization_enriched[class_name].items()}

    return further_serialization_enriched


class StateModelSerialization:

    def __init__(self,
                 state_model: StateModel,
                 mapping_file: str):
        """
        The serialization persists the state model to the target file, based on the predefined mapping (file).

        Parameters
        ----------
        state_model: The source of the data to be exported.
        mapping_file: The path to the mapping file (json).
        """

        state_model.set_implicit_objects_explicit()

        mapping_file = Path(str(ROOT_PATH) + "/twin/repository_services", mapping_file)
        if not path.isfile(mapping_file):
            raise IOError(f"Mapping file {mapping_file} does not exist.")

        self.mapping = json.load(open(mapping_file, "r"))

        if "target_schema" not in self.mapping.keys():
            raise KeyError("Target schema not specified in mapping file.")

        self.target_schema = TargetSchema.from_string(self.mapping["target_schema"])

        state_model_class_parent_classes = _get_all_class_inheritance_trees(state_model.state_model_classes)

        drop_before_serialization = _get_drop_before_serialization(self.mapping)
        drop_before_serialization_enriched = (
            _get_enriched_drop_before_serialization(drop_before_serialization,
                                                    state_model_class_parent_classes))

        further_serializable_enriched = _get_enriched_further_serializable(state_model_class_parent_classes)

        if "reference_type" not in self.mapping.keys():
            raise KeyError("Reference type not specified in mapping file.")
        reference_type = ReferenceType.validate_string(self.mapping["reference_type"])

        serialized_state_model = SerializedStateModel.from_state_model(state_model)
        # serialize the source
        self.serialized_state_model_dict = (
            serialized_state_model.to_dict(drop_before_serialization=drop_before_serialization_enriched,
                                           further_serializable=further_serializable_enriched,
                                           reference_type=reference_type))

        state_model.delete_explicit_objects()

    def export(self, target_file: str):
        # Using the target schema and source to create an export to the target file.
        print(f"Exporting state model to {target_file}")

        match self.target_schema:
            case TargetSchema.Xlsx:
                if hasattr(self, "_export_xlsx"):
                    self._export_xlsx(target_file)
                else:
                    raise NotImplementedError(f"Target schema {self.target_schema} not supported.")

            case TargetSchema.Pkl:
                if hasattr(self, "_export_pkl"):
                    self._export_pkl(target_file)
                else:
                    raise NotImplementedError(f"Target schema {self.target_schema} not supported.")

            case TargetSchema.Json:
                if hasattr(self, "_export_json"):
                    self._export_json(target_file)
                else:
                    raise NotImplementedError(f"Target schema {self.target_schema} not supported.")

            case TargetSchema.Parquet:
                if hasattr(self, "_export_parquet"):
                    self._export_parquet(target_file)
                else:
                    raise NotImplementedError(f"Target schema {self.target_schema} not supported.")

            case TargetSchema.Db:
                if hasattr(self, "_export_db"):
                    self._export_db(target_file)
                else:
                    raise NotImplementedError(f"Target schema {self.target_schema} not supported.")
