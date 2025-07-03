"""
Standardize the event log in the OFacT standard, to enable the use of standard algorithms and processing steps.
"""
import numpy as np
import pandas as pd

from ofact.env.model_administration.adapter import DataAdapter
from ofact.env.model_administration.standardization.data_entry_mapping import DataEntryMapping
from ofact.env.model_administration.standardization.event_log_standard import EventLogStandardAttributes


def _merge_columns(event_log, intern_name_mapper: dict):
    """
    Find columns with more than one entry
    """

    # Dictionary to group by the prefix
    entries_to_combine = [data_entry_mapper
                          for intern_name, data_entry_mapper in intern_name_mapper.items()
                          if data_entry_mapper.combination_part is not None]
    combinations = {}
    for entry_to_combine in entries_to_combine:
        if entry_to_combine.get_state_model_class_attribute() is None:
            continue

        combinations.setdefault(entry_to_combine.get_state_model_class_attribute(),
                                []).append(entry_to_combine)

    for combination_column, entries in combinations.items():
        combination_columns = [entry.get_new_column_name()
                               for entry in entries]
        event_log_extract = _merge_column_set(event_log[combination_columns], entries)
        event_log[combination_columns[0]] = event_log_extract
        event_log = event_log.drop(columns=combination_columns[1:])

    return event_log


def _merge_column_set(event_log_extract, entries):
    # merging based on the last element of the column name

    if len(event_log_extract.columns) == 1:
        return event_log_extract
    ensure_wanted_combinations = {entry.get_new_column_name(): entry.combination_part[1]
                                  for entry in entries}  # ToDo
    all_columns = list(ensure_wanted_combinations.keys())
    for column_name, required in ensure_wanted_combinations.items():
        if required:
            event_log_extract.loc[event_log_extract[column_name] != event_log_extract[column_name], all_columns] = (
                np.nan)
    sorted_column_names_dict = {entry.combination_part[0]: entry.get_new_column_name()
                                for entry in entries}

    sorted_keys = sorted(sorted_column_names_dict.keys())
    sorted_column_names = [sorted_column_names_dict[key]
                           for key in sorted_keys]

    new_column_name = sorted_column_names[0]

    related_elements = event_log_extract[sorted_column_names].dropna(thresh=len(sorted_column_names) - 1)
    if related_elements.empty:
        adapted_event_log_extract = pd.DataFrame(event_log_extract[new_column_name])
        return adapted_event_log_extract

    aggregated_column = (
        event_log_extract.loc[related_elements.index, sorted_column_names].astype(str).agg(' '.join, axis=1))
    aggregated_column = aggregated_column.str.replace("nan ", "", regex=False)
    aggregated_column = aggregated_column.str.replace(" nan", "", regex=False)
    event_log_extract.loc[related_elements.index, new_column_name] = aggregated_column

    adapted_event_log_extract = pd.DataFrame(event_log_extract[new_column_name])
    return adapted_event_log_extract


def _handle_times(event_log):
    """
    1. Identify the format/type of the time entries
    2. Handle the time entries to get an uniform datetime output
    """
    time_columns = [EventLogStandardAttributes.EVENT_TIME_SINGLE.string,
                    EventLogStandardAttributes.EVENT_TIME_TRACE.string,
                    EventLogStandardAttributes.EXECUTION_START_TIME.string,
                    EventLogStandardAttributes.EXECUTION_END_TIME.string]
    for time_column in time_columns:
        if time_column not in event_log.columns:
            continue

        available_times_mask = event_log[time_column] == event_log[time_column]
        time_series = event_log[time_column].loc[available_times_mask]
        if time_series.empty:
            continue  # no adaption possible or required

        try:
            time_series = pd.to_datetime(time_series)  # .dt.tz_localize(None)
        except:
            def convert_to_datetime(value):
                try:
                    return pd.to_datetime(value)
                except Exception:
                    return pd.NaT  # Rückgabe von NaT, falls die Konvertierung fehlschlägt

            # Konvertieren Sie die Spalte in eine datetime-Serie
            time_series = time_series.apply(convert_to_datetime)

        event_log.loc[available_times_mask, time_column] = time_series

    return event_log


def _check_event_time_available(event_log_columns):
    columns_available = ["_".join(column.split("_")[:-1])
                         for column in event_log_columns
                         if "time" in column]

    if (EventLogStandardAttributes.EVENT_TIME_SINGLE.string not in columns_available and
            EventLogStandardAttributes.EVENT_TIME_TRACE.string not in columns_available and
            ((EventLogStandardAttributes.EXECUTION_START_TIME.string and
              EventLogStandardAttributes.EXECUTION_END_TIME.string) not in columns_available)):  # ToDo: use the enums
        print("Time column not found in event log {}.".format(event_log_columns))

class EventLogStandardization:

    def __init__(self, data_enty_mappers: list[DataEntryMapping]) -> None:
        """
        Take the event log form the system (shop floor, logistics system, ...)
        and convert them based on the column_header_mapper to the standard format.
        """
        self.data_enty_mappers = data_enty_mappers

    def standardize(self, event_log_adapter: DataAdapter, standardized_table_path: str = "../../data/standardized_table.csv",
                    store: bool = False) -> pd.DataFrame():
        adapted_event_log = pd.DataFrame()
        event_log = event_log_adapter.get_data(None, None)

        intern_name_mapper = {}
        static_refinements = []
        for data_entry_mapper in self.data_enty_mappers:
            intern_name = data_entry_mapper.get_new_column_name()
            extern_name = data_entry_mapper.get_extern_name()

            if extern_name in event_log:
                adapted_event_log[intern_name] = event_log[extern_name]

            else:
                # should be a static refinement
                value = data_entry_mapper.get_value()
                static_refinements.append((intern_name, value))

            intern_name_mapper[intern_name] = data_entry_mapper
            # else:
            #     # handle two entries of the same element (e.g., two resources)
            #     column_of_lists = event_log[intern_name].apply(lambda x: [x] if isinstance(x, str) else [])
            #     additional_lists = (
            #         pd.Series([[self.event_log[extern_name].iloc[i]]
            #                    if self.event_log[extern_name].iloc[i] == self.event_log[extern_name].iloc[i] else []
            #                    for i in range(len(event_log[intern_name]))]))
            #     event_log[intern_name] = column_of_lists + additional_lists

        # Merge columns that are not lists
        merged_event_log = _merge_columns(adapted_event_log, intern_name_mapper)

        _check_event_time_available(merged_event_log.columns)
        time_event_log = _handle_times(merged_event_log)
        time_event_log.dropna(thresh=1, axis=0, inplace=True)
        if store:
            time_event_log.to_csv(standardized_table_path)

        return time_event_log, static_refinements
