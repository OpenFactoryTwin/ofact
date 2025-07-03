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

Used to export the (static) digital twin state model to different persistence formats.
"""

from __future__ import annotations

import re
from copy import copy
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ofact.helpers import timestamp_to_datetime
from ofact.twin.repository_services.serialization.state_model import StateModelSerialization

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel

max_characters_per_cell = 32767  # Excel has a limit of 32k characters per cell
class SerializationKind(Enum):
    DictFlatten = 1
    List = 2
    SingleValue = 3

    @staticmethod
    def from_string(input: str):
        match input.lower():
            case "dict_flatten":
                return SerializationKind.DictFlatten
            case "list":
                return SerializationKind.List
            case "single_value":
                return SerializationKind.SingleValue
            case _:
                raise NotImplementedError(f"Serialization kind {input} not supported.")


class ColumnKind(Enum):
    FixedValue = 1
    Type = 2
    Simple = 3
    Complex = 4
    Selection = 5

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
            case "selection":
                return ColumnKind.Selection
            case _:
                raise NotImplementedError(f"Column kind {input} not supported.")


def _get_column_description_entries(column_description):
    description = {}
    if "description" in column_description:
        description['description'] = column_description['description']
    else:
        description['description'] = ""

    if "notation" in column_description:
        description['notation'] = column_description['notation']
    else:
        description['notation'] = ""

    if "example" in column_description:
        description['example'] = column_description['example']
    else:
        description['example'] = ""

    if "mandatory" in column_description:
        description['mandatory'] = column_description['mandatory']
    else:
        description['mandatory'] = ""

    return description


class StaticModelStateModelSerialization(StateModelSerialization):

    def __init__(self,
                 state_model: StateModel,
                 mapping_file: str = "./static_model_excel_mapping.json"):
        super().__init__(state_model=state_model, mapping_file=mapping_file)

    @staticmethod
    def time(time: float) -> str:
        t = timestamp_to_datetime(time)
        return t.strftime("(%Y, %m, %d, %H, %M)").replace(", 0", ", ")

    @staticmethod
    def fix_corners(corners: list) -> list:
        return [tuple(corner)
                for corner in corners]

    @staticmethod
    def provider(email_address: str) -> str:
        if not email_address:
            return ""
        return email_address.split("@")[1]

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
                nodes.extend(StaticModelStateModelSerialization.traverse_tree(node["children"]))

        return nodes

    def _export_xlsx(self, target_file: str):
        if "sources" not in self.mapping.keys():
            raise KeyError("No sources specified in mapping file.")

        if "sheets" not in self.mapping.keys():
            raise KeyError("No sheets specified in mapping file.")

        # Creating the target file which is subsequently filled with the specified sheets.
        target_writer = pd.ExcelWriter(target_file, engine='openpyxl')

        sources = self.mapping["sources"]
        sheets = self.mapping["sheets"]

        classes_sources_mapping = {class_: source
                                   for source in sources
                                   for class_ in source["classes"]}

        sheet_handling_generator = \
            [self._handle_sheet(classes_sources_mapping, sheet, target_writer)
             for sheet in sheets]
        for sheet_handling in sheet_handling_generator:
            pass

        target_writer.close()

    def _handle_sheet(self, classes_sources_mapping, sheet_mapping, target_writer):
        """
        Handles the sheet as specified in the mapping file.

        Parameters
        ----------
        classes_sources_mapping : dict
            Mapping of the sources that are used to create the sheet.
        sheet_mapping : dict
            Mapping of the sheet.
        target_writer : pd.ExcelWriter
            Writer to the target Excel file.
        """

        if "name" not in sheet_mapping.keys():
            raise KeyError("Sheet name not specified.")

        if "columns" not in sheet_mapping.keys():
            raise KeyError("Sheet columns not specified.")

        sheet_name = sheet_mapping["name"]
        columns = sheet_mapping["columns"]
        start_row = sheet_mapping["start_row"] if "start_row" in sheet_mapping.keys() else 0

        classes_required = sheet_mapping["classes"]

        sheet_sources = [classes_sources_mapping[class_required]
                         for class_required in classes_required]

        values = self._get_sheet_values(sheet_sources, sheet_mapping, sheet_name)

        target_df = pd.DataFrame()
        try:
            target_df, headers, order_configurator = self._handle_columns(columns, target_df, values, sheet_name)
        except:
            raise Exception(sheet_name)
        serialize_unique = sheet_sources[0]["serialize_unique"]  # assuming the same for all

        sheets_with_amount = ["StationaryResource", "Storage", "Warehouse", "WorkStation", "ConveyorBelt",
                              "ActiveMovingResource", "PassiveMovingResource", "Part"]

        if serialize_unique:
            # Group the `target_df` by all columns and count the size each gropu and add
            # it as the `amount` column
            if sheet_name in sheets_with_amount:
                grouped = target_df.groupby("label").size().reset_index(name="amount")  # ToDo: maybe in ...
                target_df = pd.merge(target_df, grouped, on="label")

            target_df = target_df.drop_duplicates(subset="label", inplace=False)

        target_df = self._add_description(target_df, sheet_mapping)
        target_df.set_index(['index', 'label'], inplace=True)

        if order_configurator:
            target_df.columns = pd.MultiIndex.from_tuples(headers)
            # fix pandas bug: See also: https://github.com/pandas-dev/pandas/issues/27772

        exceeded = target_df.map(lambda x: isinstance(x, str) and len(x) > max_characters_per_cell)
        rows, cols = exceeded.index[exceeded.any(axis=1)], exceeded.columns[exceeded.any(axis=0)]
        if any(list(rows)):
            target_df = self._handle_exceeded_cells(target_df, rows, cols)

        try:
            target_df.to_excel(target_writer, sheet_name=sheet_name, index=True, merge_cells=True,
                               startrow=start_row,
                               freeze_panes=(0, 2))
        except:
            print("Warning: Failed", sheet_name)

        if order_configurator:
            # fix pandas bug: See also: https://github.com/pandas-dev/pandas/issues/27772
            row_to_delete = target_df.columns.nlevels
            target_writer.sheets[sheet_name].delete_rows(row_to_delete + 1 + start_row)

    def _get_sheet_values(self, sheet_sources, sheet, sheet_name):
        """
        Get the values for the sheet.

        Parameters
        ----------
        sheet_sources : list
            List of sources that are used to create the sheet.
        sheet : dict
            Mapping of the sheet.
        sheet_name : str
            Name of the sheet.
        """

        values_sources = {}
        for source in sheet_sources:
            if source["source"] not in self.serialized_state_model_dict.keys():
                raise KeyError(f"Source attribute {source} not found.")

            sheet_source = self.serialized_state_model_dict[source["source"]]
            serialization_kind = SerializationKind.from_string(source["serialization_kind"])

            match serialization_kind:
                case SerializationKind.DictFlatten:

                    if not isinstance(sheet_source, dict):
                        raise TypeError(f"Source attribute {sheet_source} is not of type dict.")

                    if any(not isinstance(value, list) for value in sheet_source.values()):
                        raise TypeError(f"Source attribute {sheet_source} is not of type dict[str, list].")

                    values = reduce(lambda x, y: x + y, sheet_source.values(), [])

                case SerializationKind.List:
                    if not isinstance(sheet_source, list):
                        raise TypeError(f"Source attribute {sheet_source} is not of type list.")

                    values = sheet_source

                case SerializationKind.SingleValue:
                    if isinstance(sheet_source, list):
                        values = sheet_source
                    elif sheet_source is None:
                        values = []
                    else:
                        values = [sheet_source]
                case _:
                    raise NotImplementedError(f"Serialization kind {serialization_kind} not supported.")

            if "filter" in sheet.keys():
                filter_name = sheet["filter"]
                if not hasattr(self, filter_name) or not callable(getattr(self, filter_name)):
                    raise NotImplementedError(f"Filter {filter_name} not supported.")

                filter_attr = getattr(self, filter_name)

                values = [value
                          for value in values
                          if filter_attr(value, sheet_name=sheet_name)]

            # if "index" in sheet.keys():
            #     idx = sheet["index"]
            #     values = [value[idx]
            #               for value in values]
            values_with_label = {value["label"]: value for value in values}
            values_sources |= values_with_label

        sheet_entries = list(values_sources.values())

        return sheet_entries

    def _handle_columns(self, columns, target_df, values, sheet_name=None):
        print(f"Handling the columns of the sheet {sheet_name}.")

        headers = []
        order_configurator = False

        for column in columns:
            if "column_kind" not in column.keys():
                raise KeyError("Column kind not specified.")

            column_kind = ColumnKind.from_string(column["column_kind"])

            if "name" not in column.keys():
                raise KeyError("Column name not specified.")

            name = column["name"]
            header = column["indexing_strategy"][0] if "indexing_strategy" in column.keys() else ""
            header = header if header == "features_requested" else ""
            order_configurator = copy(header)

            match column_kind:
                case ColumnKind.FixedValue:
                    if "value" not in column.keys():
                        raise KeyError("Column value not specified.")

                    target_df[name] = np.repeat(column["value"], len(values))
                case ColumnKind.Type:
                    types = [self._map_type(value["object_type"])
                             for value in values]

                    target_df["index"] = np.array(types)
                case ColumnKind.Simple:
                    if "indexing_strategy" not in column.keys():
                        raise KeyError("Column indexing strategy not specified.")

                    if not isinstance(column["indexing_strategy"], list):
                        raise TypeError("Column indexing strategy not of type list.")

                    indexing_strategy = column["indexing_strategy"]
                    target_df[name] = StaticModelStateModelSerialization._map_values(values, indexing_strategy)
                case ColumnKind.Complex:
                    if "indexing_strategy" not in column.keys():
                        raise KeyError("Column indexing strategy not specified.")

                    if "function" not in column.keys():
                        raise KeyError("Column function not specified.")

                    if not hasattr(self, column["function"]) or not callable(getattr(self, column["function"])):
                        raise NotImplementedError(f"Column function {column['function']} not supported.")

                    f = getattr(self, column["function"])
                    indexing_strategy = column["indexing_strategy"]

                    target_df[name] = StaticModelStateModelSerialization._map_values(values, indexing_strategy, f=f)
                case ColumnKind.Selection:
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
                    columns = set(columns)

                    for col in columns:
                        counts = []
                        for value in mapped_values:
                            matches = list(filter(lambda x: x == col, value))
                            count = len(matches)
                            counts.append(count)
                        target_df[col] = np.array(counts)
                        headers.append((header, col))

            match column_kind:
                case ColumnKind.Selection:
                    # already handled before
                    pass
                case _:
                    if name not in {'index', 'label'}:
                        headers.append((header, name))

        return target_df, headers, order_configurator

    def _add_description(self, target_df, sheet_mapping):
        """
        Add description rows to the target dfs for better understanding in the modelling phase.
        """

        column_description = \
            [_get_column_description_entries(column_description)
             for column_description in sheet_mapping["columns"]]

        description_df = pd.DataFrame(column_description).T
        if len(description_df.columns) != len(target_df.columns):
            diff = len(target_df.columns) - len(description_df.columns)
            for i in range(diff):
                description_df = pd.concat([description_df, description_df[description_df.columns[-1]]],
                                           axis=1, ignore_index=True)

        description_df.columns = target_df.columns
        target_df = pd.concat([description_df, target_df],
                              ignore_index=True)
        return target_df

    def _handle_exceeded_cells(self, target_df, rows, cols):

        extension = {}
        for row in list(rows):
            for col in list(cols):
                cell_content = target_df.loc[row, col]
                if len(cell_content) <= max_characters_per_cell:
                    continue

                separated = False
                separated_list = []
                while not separated:
                    splitting_point = min(max_characters_per_cell, len(cell_content))
                    snippet, cell_content = (cell_content[:splitting_point],
                                             cell_content[splitting_point:])
                    separated_list.append(snippet)

                    if len(cell_content) == 0:
                        separated = True
                        separated_list = separated_list

                if row not in extension:
                    extension[row] = {}
                extension[row][col] = separated_list

        for row in extension:
            # check how many rows required
            additional_rows_required = 1
            for col, snippets in extension[row].items():
                if len(snippets) > additional_rows_required:
                    # insert new row
                    additional_rows_required = len(snippets)
                target_df.loc[row, col] = snippets[0]  # replace it with the first snippet

            # insert new rows
            new_row = target_df.loc[row]
            columns_to_reset = list(target_df.loc[row].index)
            for col in columns_to_reset:
                new_row.loc[col] = ""
            additional_rows = []
            for i in range(additional_rows_required - 1):
                for col, snippets in extension[row].items():
                    if len(snippets) > i:
                        new_row.loc[col] = snippets[i + 1]
                additional_rows.append(new_row)
            for additional_row in additional_rows:
                target_df = pd.concat([target_df, pd.DataFrame(additional_row).T])

        target_df.iloc[4:] = target_df.iloc[4:].sort_index()

        return target_df

    @staticmethod
    def filter_type_storage(value: dict, **kwargs) -> bool:
        value_kind = StaticModelStateModelSerialization._map_type(value["object_type"])
        return value_kind == "Storage"

    @staticmethod
    def filter_type_warehouse(value: dict, **kwargs) -> bool:
        value_kind = StaticModelStateModelSerialization._map_type(value["object_type"])
        return value_kind == "Warehouse"

    @staticmethod
    def filter_type_work_station(value: dict, **kwargs) -> bool:
        value_kind = StaticModelStateModelSerialization._map_type(value["object_type"])
        return value_kind == "WorkStation"

    @staticmethod
    def filter_stationary_resource_sheet(value: dict, **kwargs) -> bool:
        value_kind = StaticModelStateModelSerialization._map_type(value["object_type"])
        return value_kind == "StationaryResource"

    @staticmethod
    def filter_non_stationary_resource_sheet(value: dict, **kwargs) -> bool:
        value_kind = StaticModelStateModelSerialization._map_type(value["object_type"])
        return value_kind == "NonStationaryResource"

    @staticmethod
    def filter_type_conveyor_belt(value: dict, **kwargs) -> bool:
        value_kind = StaticModelStateModelSerialization._map_type(value["object_type"])
        return value_kind == "ConveyorBelt"

    @staticmethod
    def filter_process_models(value: dict, sheet_name) -> bool:
        value_kind = StaticModelStateModelSerialization._map_type(value["object_type"])
        if sheet_name == "ResourceModel":
            second_condition = "ResourceGroup" in value_kind
        elif sheet_name == "TransformationModel":
            second_condition = "EntityTransformationNode" in value_kind
        else:
            second_condition = False
        return sheet_name in value_kind or second_condition

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
        return [entity["label"]
                for entity in entities]

    @staticmethod
    def label_list_tuple(entities: list[tuple[dict]]) -> list[str]:
        return [tuple(entity["label"] for entity in tup)
                for tup in entities]

    @staticmethod
    def list_tuple(entities: list[tuple[dict]]) -> list[str]:
        return [tuple(entity for entity in tup)
                for tup in entities]

    @staticmethod
    def label_dict(input: dict) -> dict[str, str]:
        labels = {key.get_static_model_id()[1:]: [value.get_static_model_id()[1:]
                                              for value in values]
                  for key, values in input.items()}

        return labels

    @staticmethod
    def efficiency(efficiency: dict) -> str:
        # get the type as a string
        kind = StaticModelStateModelSerialization._map_type(efficiency["object_type"])
        if kind == "SingleValueDistribution":
            value = efficiency["value"]
        elif kind == "NormalDistribution":
            value = str(efficiency["mue"]) + ", " + str(efficiency["sigma"])
        else:
            raise NotImplementedError(f"Efficiency kind {kind} not supported.")

        return f"{kind}({value})"

    @staticmethod
    def _map_type(kind: str) -> str:
        return kind
        # kind = reversed(kind.split("."))
        # kind = next(kind)
        # return kind[:-2]

    @staticmethod
    def _map_values(
            values: list,
            indexing_strategy: list,
            f=None
    ) -> np.ndarray:
        # The caller has to ensure that the `f` is supported by the serialization

        # TODO: Log a warning if the indexing strategy is invalid -> leads to a `None` value
        if f is None:
            res = []
            for value in values:
                try:
                    val = reduce(lambda acc, index: acc[index], indexing_strategy, value)
                    # TODO: When importing a digital twin through the UI they mess around with the type of the corners
                    # attribute which gets converted from a list of tuples to a list of lists. This is why
                    # `if val` does not work here and instead has to be replaced with `if val is not None`. This behavior
                    # might break the deserialization again but has to be fixed from within the frontend.
                    if not isinstance(val, str):
                        val = str(val) if val is not None else ""

                    res.append(val)
                except (TypeError, KeyError):
                    res.append("")

            return np.array(res)
        else:
            res = []
            for value in values:
                try:
                    if len(indexing_strategy) == 2:
                        val = value[indexing_strategy[0]].__dict__[indexing_strategy[1]]
                    else:
                        val = reduce(lambda acc, index: acc[index], indexing_strategy, value)
                    if isinstance(val, (float, int)):
                        if val < 0:
                            raise Exception("Negative value", value)
                    val = f(val)
                    if not isinstance(val, str):
                        val = str(val) if val is not None else ""
                    # TODO: This might be not the most efficient way converting the value to a string
                    res.append(val)
                except (TypeError, KeyError):
                    res.append("")

            return np.array(res)


if __name__ == "__main__":
    from ofact.planning_services.model_generation.persistence import deserialize_state_model
    from pathlib import Path
    from ofact.settings import ROOT_PATH

    state_model = deserialize_state_model(source_file_path=Path(ROOT_PATH.rsplit("ofact", 1)[0],
                                                                "projects/tutorial/models/twin/mini.pkl"),
                                          persistence_format="pkl", deserialization_required=False)
    # state_model = (
    #     import_state_model(source_file_path=Path(ROOT_PATH.split("ofact")[0],
    #                                                  "projects/bicycle_world/scenarios/current/models/twin/base.pkl"),
    #                                      persistence_format="pkl")
    exporter = StaticModelStateModelSerialization(state_model)
    exporter.export("test.xlsx")
