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

This file is used to export the digital twin state model.
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from functools import reduce
from os import path

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.serialization import Serializable  # ToDo: Why from model and not from basic elements?


class SerializationKind(Enum):
    DictFlatten = 1
    List = 2
    SingleValue = 3
    Mixed = 4

    @staticmethod
    def from_string(input: str):
        match input.lower():
            case "dict_flatten":
                return SerializationKind.DictFlatten
            case "list":
                return SerializationKind.List
            case "single_value":
                return SerializationKind.SingleValue
            case "mixed":
                return SerializationKind.Mixed
            case _:
                raise NotImplementedError(f"Serialization kind {input} not supported.")


class ColumnKind(Enum):
    FixedValue = 1
    Type = 2
    Simple = 3
    Complex = 4
    Generate = 5

    @staticmethod
    def from_string(input: str):
        match input.lower():
            case "fixed_value":
                return ColumnKind.FixedValue
            case "type":
                return ColumnKind.Type
            case "simple":
                return ColumnKind.Simple
            case "complex":
                return ColumnKind.Complex
            case "generate":
                return ColumnKind.Generate
            case _:
                raise NotImplementedError(f"Column kind {input} not supported.")


class TargetSchema(Enum):
    Xlsx = 1

    @staticmethod
    def from_string(input: str):
        match input.lower():
            case "xlsx":
                return TargetSchema.Xlsx
            case _:
                raise NotImplementedError(f"Target schema {input} not supported.")


class Exporter:
    def __init__(
            self,
            source: Serializable,
            # TODO: change the default path of the aligned mapping file
            mapping_file: str = "./default_aligned_mapping.json",
    ) -> None:
        # Make sure the path is relative to the exporter.py file
        # otherwise it depends on, from where this file is called.
        mapping_file = path.join(path.dirname(__file__), mapping_file)
        if not path.isfile(mapping_file):
            raise IOError(f"Mapping file {mapping_file} does not exist.")

        self.mapping = json.load(open(mapping_file, "r"))

        if "target_schema" not in self.mapping.keys():
            raise KeyError("Target schema not specified in mapping file.")

        self.target_schema = TargetSchema.from_string(self.mapping["target_schema"])
        self.source = source.dict_serialize()

    @staticmethod
    def time(time: float) -> str:
        t = datetime.fromtimestamp(time)
        return t.strftime("(%Y, %m, %d, %H, %M)").replace(", 0", ", ")

    @staticmethod
    def fix_corners(corners: list) -> list:
        return [tuple(corner) for corner in corners]

    @staticmethod
    def provider(email_address: str) -> str:
        if not email_address:
            return ""
        return email_address.split("@")[1]

    @staticmethod
    def process_controllers(processes: list):
        controllers = []
        for process in processes:
            if "_lead_time_controller" in process.keys():
                controllers.append(process["_lead_time_controller"])

            if "_quality_controller" in process.keys():
                controllers.append(process["_quality_controller"])

            if "_resource_controller" in process.keys():
                controllers.append(process["_resource_controller"])

            if "_transition_controller" in process.keys():
                controllers.append(process["_transition_controller"])

            if "_transformation_controller" in process.keys():
                controllers.append(process["_transformation_controller"])

        return controllers

    @staticmethod
    def process_time_models(processes: list):
        models = []
        for process in processes:
            if "_lead_time_controller" in process.keys():
                models.append(process["_lead_time_controller"]["_process_time_model"])

        return Exporter.drop_duplicates_by_label(models)

    @staticmethod
    def quality_models(processes: list):
        models = []
        for process in processes:
            if "_quality_controller" in process.keys():
                models.append(process["_quality_controller"]["_quality_model"])

        return Exporter.drop_duplicates_by_label(models)

    @staticmethod
    def resource_models(processes: list):
        models = []
        for process in processes:
            if "_resource_controller" in process.keys():
                models.append(process["_resource_controller"]["_resource_model"])

        return Exporter.drop_duplicates_by_label(models)

    @staticmethod
    def resource_groups(processes: list):
        models = []
        for process in processes:
            if "_resource_controller" in process.keys():
                models.extend(process["_resource_controller"]["_resource_model"]["_resource_groups"])

        return Exporter.drop_duplicates_by_label(models)

    @staticmethod
    def joined_resource_models(processes: list):
        resource_models = Exporter.resource_models(processes)
        resource_groups = Exporter.resource_groups(processes)
        return resource_models + resource_groups

    @staticmethod
    def transition_models(processes: list):
        models = []
        for process in processes:
            if "_transition_controller" in process.keys():
                models.append(process["_transition_controller"]["_transition_model"])

        return Exporter.drop_duplicates_by_label(models)

    @staticmethod
    def transformation_models(processes: list):
        models = []
        for process in processes:
            if "_transformation_controller" in process.keys():
                models.append(process["_transformation_controller"]["_transformation_model"])

        return Exporter.drop_duplicates_by_label(models)

    @staticmethod
    def joined_transformation_models(processes: list):
        transformation_nodes = []
        transformation_models = []
        for item in processes:
            if Exporter._map_type(item["object_type"]) == "EntityTransformationNode":
                transformation_nodes.append(item)
            else:
                transformation_models.append(item)
        transformation_models = Exporter.transformation_models(transformation_models)
        return transformation_models + transformation_nodes

    @staticmethod
    def process_execution_plans(resources: list):
        peps = []
        for resource in resources:
            peps.append(resource["_process_execution_plan"])

        return Exporter.drop_duplicates_by_label(peps)

    @staticmethod
    def drop_duplicates_by_label(input_list: list) -> list:
        unique_labels = []
        unique_list = []
        for item in input_list:
            if item["label"] not in unique_labels:
                unique_labels.append(item["label"])
                unique_list.append(item)
        return unique_list

    @staticmethod
    def traverse_tree(root_nodes):
        nodes = []
        for node in root_nodes:
            # TODO: Why is the node a string sometimes?
            if type(node) == str:
                continue
            nodes.append(node)
            if "children" in node.keys():
                nodes.extend(Exporter.traverse_tree(node["children"]))

        return nodes

    @staticmethod
    def traverse_transformation_nodes(processes: list):
        models = Exporter.transformation_models(processes)
        nodes = []
        for model in models:
            nodes.extend(Exporter.traverse_tree(model["_root_nodes"]))
        return nodes

    def export(self, target_file: str):
        # Using the target schema and source to create an export to the target file. 
        match self.target_schema:
            case TargetSchema.Xlsx:
                self._export_xlsx(target_file)

    def _export_xlsx(self, target_file: str):
        if "sheets" not in self.mapping.keys():
            raise KeyError("No sheets specified in mapping file.")

        # Creating the target file which is subsequently filled with the specified sheets.
        target_writer = pd.ExcelWriter(target_file, engine='openpyxl')

        for sheet in self.mapping["sheets"]:
            if "name" not in sheet.keys():
                raise KeyError("Sheet name not specified.")

            if "source" not in sheet.keys():
                raise KeyError("Sheet source not specified.")

            if "columns" not in sheet.keys():
                raise KeyError("Sheet columns not specified.")

            if "serialization_kind" not in sheet.keys():
                raise KeyError("Sheet serialization kind not specified.")

            # I don't know whether this is mandatory for every serialization kind, like the one that 
            # exports a matrix, where the values are represented by a combination of the `ResourceController` and 
            # the `ResourceModel`. I don't quite know how to handle this matrix export yet. But probably we can 
            # just serialize a list as we were doing it before and leave the fields empty that don't apply to the 
            # specified controller.
            if "serialize_unique" not in sheet.keys():
                raise KeyError("Sheet serialization kind not specified.")

            target_df = pd.DataFrame()
            sheet_name = sheet["name"]
            columns = sheet["columns"]
            serialization_kind = SerializationKind.from_string(sheet["serialization_kind"])
            serialize_unique = sheet["serialize_unique"]

            sheet_source = sheet["source"]
            start_row = sheet["start_row"] if "start_row" in sheet.keys() else 0
            filter_attr = None

            if "filter" in sheet.keys():
                filter_name = sheet["filter"]
                if not hasattr(self, filter_name) or not callable(getattr(self, filter_name)):
                    raise NotImplementedError(f"Filter {filter_name} not supported.")

                filter_attr = getattr(self, filter_name)

            # TODO: How do we handle cases where there are multiple sources -> see `NonStationaryResource`.
            #       We could introduce a list of sources

            if isinstance(sheet_source, list):
                if any(s not in self.source.keys() for s in sheet_source):
                    raise KeyError(f"Source attribute {sheet_source} not found.")
                sources = [self.source[s] for s in sheet_source]
            else:
                if sheet_source not in self.source.keys():
                    raise KeyError(f"Source attribute {sheet_source} not found.")
                sources = [self.source[sheet_source]]

            match serialization_kind:
                case SerializationKind.DictFlatten:
                    values = []
                    for sheet_source in sources:
                        if not isinstance(sheet_source, dict):
                            raise TypeError(f"Source attribute {sheet_source} is not of type dict.")

                        if any(not isinstance(value, list) for value in sheet_source.values()):
                            raise TypeError(f"Source attribute {sheet_source} is not of type dict[str, list].")

                        temp = reduce(lambda x, y: x + y, sheet_source.values(), [])
                        values.append(temp)
                    values = reduce(lambda x, y: x + y, values, [])
                case SerializationKind.List:
                    if any(not isinstance(sheet_source, list) for sheet_source in sources):
                        raise TypeError(f"Source attribute {sources} is not of type list.")

                    values = reduce(lambda x, y: x + y, sources, [])

                case SerializationKind.SingleValue:
                    if len(sources) > 1:
                        raise ValueError("Source attribute is of length > 1 but serialization kind is `single_value`.")

                    values = sources

                case SerializationKind.Mixed:
                    values = []
                    for sheet_source in sources:
                        if isinstance(sheet_source, dict):
                            if any(not isinstance(value, list) for value in sheet_source.values()):
                                raise TypeError(f"Source attribute {sheet_source} is not of type dict[str, list].")

                            temp = reduce(lambda x, y: x + y, sheet_source.values(), [])
                            values.append(temp)
                        elif isinstance(sheet_source, list):
                            values.append(sheet_source)

                    values = reduce(lambda x, y: x + y, values, [])

            if "source_function" in sheet.keys():
                source_function = sheet["source_function"]
                if not hasattr(self, source_function) or not callable(getattr(self, source_function)):
                    raise NotImplementedError(f"Source function {source_function} not supported.")

                values = getattr(self, source_function)(values)

            if filter_attr:
                values = [value for value in values if filter_attr(value)]

            if "index" in sheet.keys():
                idx = sheet["index"]
                values = [value[idx] for value in values]

            headers = []
            has_header = False

            for column in columns:
                if "column_kind" not in column.keys():
                    raise KeyError("Column kind not specified.")

                column_kind = ColumnKind.from_string(column["column_kind"])

                if "name" not in column.keys():
                    raise KeyError("Column name not specified.")

                name = column["name"]
                header = column["header"] if "header" in column.keys() else ""
                has_header = has_header or (not header == "")

                match column_kind:
                    case ColumnKind.FixedValue:
                        if "value" not in column.keys():
                            raise KeyError("Column value not specified.")

                        target_df[name] = np.repeat(column["value"], len(values))
                    case ColumnKind.Type:
                        types = [self._map_type(value["object_type"]) for value in values]
                        target_df["index"] = np.array(types)
                    case ColumnKind.Simple:
                        if "indexing_strategy" not in column.keys():
                            raise KeyError("Column indexing strategy not specified.")

                        if not isinstance(column["indexing_strategy"], list):
                            raise TypeError("Column indexing strategy not of type list.")

                        indexing_strategy = column["indexing_strategy"]
                        target_df[name] = Exporter._map_values(values, indexing_strategy)
                    case ColumnKind.Complex:
                        if "indexing_strategy" not in column.keys():
                            raise KeyError("Column indexing strategy not specified.")

                        if "function" not in column.keys():
                            raise KeyError("Column function not specified.")

                        if not hasattr(self, column["function"]) or not callable(getattr(self, column["function"])):
                            raise NotImplementedError(f"Column function {column['function']} not supported.")

                        f = getattr(self, column["function"])
                        indexing_strategy = column["indexing_strategy"]

                        target_df[name] = Exporter._map_values(values, indexing_strategy, f=f)
                    case ColumnKind.Generate:
                        if "indexing_strategy" not in column.keys():
                            raise KeyError("Column source not specified.")

                        indexing_strategy = column["indexing_strategy"]
                        mapped_values = []
                        for value in values:
                            mapped_value = reduce(lambda acc, y: acc[y], indexing_strategy, value)
                            mapped_values.append(mapped_value)

                        if any(not isinstance(value, list) for value in mapped_values):
                            raise TypeError(f"The column kind `generate` requires the source to be of type `list`.")

                        columns = reduce(lambda x, y: x + y, mapped_values, [])
                        columns = set(Exporter.label_list(columns))

                        for col in columns:
                            counts = []
                            for value in mapped_values:
                                matches = list(filter(lambda x: x["label"] == col, value))
                                count = len(matches)
                                counts.append(count)
                            target_df[col] = np.array(counts)
                            headers.append((header, col))

                match column_kind:
                    case ColumnKind.Generate:
                        # already handled before
                        pass
                    case _:
                        if name not in {'index', 'label'}:
                            headers.append((header, name))

            if serialize_unique:
                # Group the `target_df` by all columns and count the size each gropu and add 
                # it as the `amount` column 
                grouped = target_df.groupby("label").size().reset_index(name="amount")
                target_df = pd.merge(target_df, grouped, on="label").drop_duplicates(subset="label", inplace=False)

            target_df.set_index(['index', 'label'], inplace=True)

            if has_header:
                target_df.columns = pd.MultiIndex.from_tuples(headers)
                # fix pandas bug: See also: https://github.com/pandas-dev/pandas/issues/27772

            target_df.to_excel(target_writer, sheet_name=sheet_name, index=True, merge_cells=True, startrow=start_row,
                               freeze_panes=(0, 2))

            if has_header:
                # fix pandas bug: See also: https://github.com/pandas-dev/pandas/issues/27772
                row_to_delete = target_df.columns.nlevels
                target_writer.sheets[sheet_name].delete_rows(row_to_delete + 1 + start_row)
        target_writer.close()

    @staticmethod
    def filter_type_storage(value: dict) -> bool:
        value_kind = Exporter._map_type(value["object_type"])
        return value_kind == "Storage"

    @staticmethod
    def filter_type_warehouse(value: dict) -> bool:
        value_kind = Exporter._map_type(value["object_type"])
        return value_kind == "Warehouse"

    @staticmethod
    def filter_type_work_station(value: dict) -> bool:
        value_kind = Exporter._map_type(value["object_type"])
        return value_kind == "WorkStation"

    @staticmethod
    def filter_stationary_resource_sheet(value: dict) -> bool:
        value_kind = Exporter._map_type(value["object_type"])
        return value_kind == "StationaryResource"

    @staticmethod
    def filter_non_stationary_resource_sheet(value: dict) -> bool:
        value_kind = Exporter._map_type(value["object_type"])
        return value_kind == "NonStationaryResource"

    # TODO: Clariy the meaning of `capacity_per_entity_type` attribute for the `ConveyorBelt` class

    @staticmethod
    def filter_type_conveyor_belt(value: dict) -> bool:
        value_kind = Exporter._map_type(value["object_type"])
        return value_kind == "ConveyorBelt"

    @staticmethod
    def name_to_pre_name(value: str) -> str:
        if value is None or len(value) == 0:
            value = ""
        try:
            pre_name = re.split(r'\s+', value)[0]
        except:
            pre_name = value
        return pre_name

    @staticmethod
    def name_to_last_name(value: str) -> str:
        if value is None or len(value) == 0:
            value = ""
        try:
            last_name = re.split(r'\s+', value)[1]
        except:
            last_name = value
        return last_name

    @staticmethod
    def label_list(entities: list[dict]) -> list[str]:
        return [entity["label"] for entity in entities]

    @staticmethod
    def label_list_tuple(entities: list[tuple[dict]]) -> list[str]:
        return [tuple(entity["label"] for entity in tup) for tup in entities]

    @staticmethod
    def list_tuple(entities: list[tuple[dict]]) -> list[str]:
        return [tuple(entity for entity in tup) for tup in entities]

    @staticmethod
    def label_dict(input: dict) -> dict[str, str]:
        labels = {}
        for key, values in input.items():
            labels[key] = [value["label"] for value in values]

        return labels

    @staticmethod
    def efficiency(efficiency: dict) -> str:
        # get the type as a string
        kind = Exporter._map_type(efficiency["object_type"])
        value = efficiency["value"]

        return f"{kind}({value})"

    @staticmethod
    def _map_type(kind: str) -> str:
        kind = reversed(kind.split("."))
        kind = next(kind)
        return kind[:-2]

    @staticmethod
    def _map_values(
            values: list,
            indexing_strategy: list,
            f=None
    ) -> np.ndarray:
        # The caller has to ensure that the `f` is supported by the exporter

        # TODO: Log a warning if the indexing strategy is invalid -> leads to a `None` value
        if f is None:
            res = []
            for value in values:
                try:
                    val = reduce(lambda acc, index: acc[index], indexing_strategy, value)
                    # TODO: When importing a digital twin through the UI they mess around with the type of the corners 
                    # attribute which gets converted from a list of tuples to a list of lists. This is why 
                    # `if val` does not work here and instead has to be replaced with `if val is not None`. This behavior 
                    # might break the importer again but has to be fixed from within the frontend.
                    val = str(val) if val is not None else ""
                    res.append(val)
                except (TypeError, KeyError):
                    res.append("")

            return np.array(res)
        else:
            res = []
            for value in values:
                try:
                    val = reduce(lambda acc, index: acc[index], indexing_strategy, value)
                    val = f(val)
                    val = str(val) if val else ""
                    # TODO: This might be not the most efficient way converting the value to a string 
                    res.append(val)
                except (TypeError, KeyError):
                    res.append("")

            return np.array(res)
