from __future__ import annotations

from enum import Enum
from typing import Optional, TYPE_CHECKING, FrozenSet

import pandas as pd

from ofact.env.model_administration.pipeline_settings import PipelineType

if TYPE_CHECKING:
    pass


class FilterOptions(Enum):

    DATE_RANGE = "Date Range"
    REGEX = "Regex"
    CONTAINS = "Contains"
    EXACT_MATCH = "Exact Match"
    GREATER_THAN = "Greater Than"
    LESS_THAN = "Less Than"
    IN_LIST = "In List"
    START_WITH = "Start With"
    ENDS_WITH = "Ends With"

    @classmethod
    def from_string(cls, string: str):
        return FilterOptions[string]


class EntryType(Enum):
    """
    Toggle: toggle mean that the column header is set to the entries unequal nan values.
    """
    TOGGLE = "Toggle"

    @classmethod
    def from_string(cls, string: str):
        return EntryType[string]

class Filter:

    def __init__(self, settings: dict):
        """
        Filter the data frame columns based on the settings for each column
        Parameters
        ----------
        column_names (list[str]): List of column names associated to the dataframe to be filtered.
        settings (dict): Settings how to filter each column set (one set is seen as AND operation)  # ToDo
        """
        self.settings: dict[FrozenSet[str], tuple[FilterOptions, list]] = settings
        self.entry_mapping = None

    def add_entry_mapping(self, entry_mapping):
        self.entry_mapping = entry_mapping

    def filter(self, df: pd.DataFrame()):
        """
        Filter the data frame based on the settings.

        Parameters
        ----------
        df (pd.DataFrame()): Dataframe to be filtered
        """
        for origin_column_names, (filter_option, parameters) in self.settings.items():
            column_names = self.get_column_names(origin_column_names)
            filter_condition = None  # pd.Series([False] * len(df))
            for idx, column in enumerate(column_names):
                if column not in df.columns:
                    raise Exception(f"Column {column} not found in dataframe")

                match filter_option:
                    case FilterOptions.DATE_RANGE:
                        single_filter_condition = self._get_filter_condition_date_range(df, column, parameters)
                    case FilterOptions.REGEX:
                        single_filter_condition = self._get_filter_condition_regex(df, column, parameters)
                    case FilterOptions.CONTAINS:
                        single_filter_condition = self._get_filter_condition_contains(df, column, parameters)
                    case FilterOptions.EXACT_MATCH:
                        single_filter_condition = self._get_filter_condition_exact_match(df, column, parameters)
                    case FilterOptions.GREATER_THAN:
                        single_filter_condition = self._get_filter_condition_greater_than(df, column, parameters)
                    case FilterOptions.LESS_THAN:
                        single_filter_condition = self._get_filter_condition_less_than(df, column, parameters)
                    case FilterOptions.IN_LIST:
                        single_filter_condition = self._get_filter_condition_in_list(df, column, parameters)
                    case FilterOptions.START_WITH:
                        single_filter_condition = self._get_filter_condition_start_with(df, column, parameters)
                    case FilterOptions.ENDS_WITH:
                        single_filter_condition = self._get_filter_condition_ends_with(df, column, parameters)
                    case _:
                        raise ValueError(f"Invalid filter option: {filter_option}")

                if idx == 0:
                    filter_condition = single_filter_condition
                else:
                    filter_condition |= single_filter_condition

            df = df.loc[filter_condition]

        return df

    def get_column_names(self, column_names):
        # ToDo: if more than one is required
        return [self.entry_mapping.get_new_column_name()
                for origin_column_name in list(column_names)]

    def _get_filter_condition_date_range(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Date Range Filter"""
        filter_condition = ((df[column_name] >= parameters[0]) &
                            (df[column_name] <= parameters[1]))
        return filter_condition

    def _get_filter_condition_regex(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Regex Filter"""
        filter_condition = df[column_name].str.contains(parameters[0], regex=True, na=False)
        return filter_condition

    def _get_filter_condition_contains(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Contains Filter"""
        filter_condition = df[column_name].str.contains(parameters[0], na=False)
        return filter_condition

    def _get_filter_condition_exact_match(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Exact Match Filter"""
        filter_condition = df[column_name] == parameters[0]
        return filter_condition

    def _get_filter_condition_greater_than(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Greater Than Filter"""
        filter_condition = df[column_name] > parameters[0]
        return filter_condition

    def _get_filter_condition_less_than(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Less Than Filter"""
        filter_condition = df[column_name] < parameters[0]
        return filter_condition

    def _get_filter_condition_in_list(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """In List"""
        filter_condition = df[column_name].isin(parameters)

        return filter_condition

    def _get_filter_condition_start_with(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Starts With Filter"""
        filter_condition = df[column_name].str.startswith(parameters[0])
        return filter_condition

    def _get_filter_condition_ends_with(self, df: pd.DataFrame(), column_name: str, parameters: list[object]):
        """Ends With Filter"""
        filter_condition = df[column_name].str.endswith(parameters[0])
        return filter_condition


class DataEntryMapping:
    """next_id (int): Class-level counter for generating unique identification numbers."""
    next_id: int = int(0)

    @staticmethod
    def get_next_id() -> int:
        """Get the next available unique identifier.

        Returns:
            int: The next sequential ID number.
        """
        return DataEntryMapping.next_id

    def __init__(self,
                 external_name: str,
                 identification: Optional[int] = None,
                 reference_identification: Optional[int] = None,
                 correlated_entry_ids: Optional[list[int]] = None,
                 state_model_class: Optional[str] = None,
                 state_model_attribute: Optional[str] = None,
                 handling: Optional[str] = None,
                 mandatory: bool = False,
                 combination_part: Optional[tuple[int, bool]] = None,
                 splitting_element: Optional[tuple[str, int]] = None,
                 time_filter: Optional[str] = None,
                 filter: Optional[Filter] = None,
                 entry_type: Optional[EntryType] = None,
                 value: Optional[object] = None,
                 special_handling: Optional[list] = None,
                 required_for_model_generation: bool = False,
                 required_for_data_integration: bool = False):
        """
        A class that manages mappings between external data entries and internal state model objects.
        Each column of data point name should have at least one mapping if the content is required.
        since only one to one mappings are allowed, it is also normal that one column has more than one mapping.

        Parameters
        ----------
        identification (int): Unique identifier for this mapping instance.
        external_name (str): Name or identifier used in the external data source.
        reference_id (int): Optional reference ID for linking to other objects.
        correlated_entry_ids: ToDo: required?
        state_model_class: Class of state model object this maps to.
        state_model_attribute: Attribute of state model object this maps to (reference identification).
        handling: e.g. number of a part or resource
        mandatory (bool): Flag indicating if the entry is mandatory. If not, the whole entry is skipped/ deleted.
        combination_part (Optional[int]): ID for combining related parts if applicable.
        example given, combination_part 1 comes after 0 ''.join(1,2,3)
        splitting_element (Optional[tuple[str, int]]): Tuple containing splitting information (attribute name, value)
        if needed.
        filter (Optional[Filter]): Optional filter to apply to the data entry.
        entry_type (EntryType): Type of entry
        value: static refinement (not coming from the event logs)
        required_for_model_generation: states if the column is required for model generation
        required_for_data_integration: states if the column is required for data integration
        """
        if identification is None:
            self.identification: int = self.get_next_id()
            DataEntryMapping.next_id += 1
        else:
            self.identification: int = identification

        self.external_name: str = external_name
        if reference_identification is not None:
            reference_identification = int(reference_identification)
        self.reference_identification: Optional[int] = reference_identification
        if correlated_entry_ids is None:  # used???
            correlated_entry_ids = []
        self.correlated_entry_ids: list[int] = correlated_entry_ids

        self.state_model_class: str = state_model_class
        self.state_model_attribute: str = state_model_attribute
        self.handling = handling

        self.combination_part: tuple[int, bool] = combination_part
        self.splitting_element = splitting_element

        self.mandatory: bool = mandatory  # ToDo: either or relationships should be handleable

        self.time_filter = time_filter

        if filter is not None:
            filter.add_entry_mapping(self)
        self.filter: Filter = filter

        self.entry_type = entry_type

        self.value = value

        self.special_handling = special_handling

        self.required_for_model_generation = required_for_model_generation
        self.required_for_data_integration = required_for_data_integration

    def get_new_column_name(self):
        new_column_name = self.get_state_model_class_attribute()
        if self.handling is not None:
            if new_column_name != '':
                new_column_name += "_"
            new_column_name += self.handling
        if new_column_name != '':
            new_column_name += "_"
        new_column_name += str(int(self.identification))
        return new_column_name

    def get_state_model_class_attribute(self):
        new_column_name = ''
        if self.state_model_class is not None:
            new_column_name += self.state_model_class
        if self.state_model_attribute is not None:
            if new_column_name != '':
                new_column_name += "_"
            new_column_name += self.state_model_attribute

        return new_column_name

    def get_state_model_class(self):
        return self.state_model_class

    def get_extern_name(self):
        return self.external_name

    def get_value(self):
        return self.value

    def filter_required(self):
        if self.filter is None:
            return False
        return True

    def execute_filter(self, event_log: pd.DataFrame()):
        event_log = self.filter.filter(event_log)
        return event_log

    def handle_entry_type(self, event_log: pd.DataFrame()):
        merge = False
        if self.entry_type is None:
            return event_log, merge

        if self.entry_type == EntryType.TOGGLE:
            col_name = self.get_new_column_name()
            event_log[col_name] = event_log[col_name].astype("object")
            filter_condition = event_log[col_name] == event_log[col_name]
            event_log.loc[filter_condition, col_name] = self.external_name
            merge = True

        return event_log, merge

    def required(self, mode):
        if mode == PipelineType.MODEL_GENERATION:
            return self.required_for_model_generation
        elif mode == PipelineType.DATA_INTEGRATION:
            return self.required_for_data_integration
        else:
            return None
