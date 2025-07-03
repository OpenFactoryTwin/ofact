"""
Preprocessing
"""

import pandas as pd

from ofact.env.model_administration.standardization.data_entry_mapping import DataEntryMapping


class Preprocessing:

    def __init__(self, data_entry_mappers: list[DataEntryMapping]) -> None:
        """
        Take the event log form the system (shop floor, logistics system, ...)
        and convert them based on the column_header_mapper to the standard format.
        """
        self.data_entry_mappers = data_entry_mappers

    def preprocess(self, event_log, preprocessed_table_path: str = "../../data/preprocessed_table.csv",
                    store: bool = False) -> pd.DataFrame():

        merge_entries = {}
        for data_entry_mapper in self.data_entry_mappers:
            intern_name = data_entry_mapper.get_new_column_name()

            if data_entry_mapper.mandatory:
                # drop all rows where the column is empty
                event_log = event_log.loc[~event_log[intern_name].isna()]  # ToDo: Executed_start_time is required

            if data_entry_mapper.filter_required():
                event_log = data_entry_mapper.execute_filter(event_log)

            event_log, merge = data_entry_mapper.handle_entry_type(event_log)
            if merge:
                merge_entries.setdefault((data_entry_mapper.reference_identification,
                                                       data_entry_mapper.state_model_class,
                                                       data_entry_mapper.state_model_attribute),
                                                      []).append(data_entry_mapper)

        if merge_entries:
            for (_, _, _), entries_mappers in merge_entries.items():
                if len(entries_mappers) > 1:
                    columns_concerned = [entry_mapper.get_new_column_name()
                                         for entry_mapper in entries_mappers]
                    first_colum, other_columns = columns_concerned[0], columns_concerned[1:]
                    for column_concerned in other_columns:
                        event_log[first_colum] = event_log[first_colum].combine_first(event_log[column_concerned])

                    event_log.drop(columns=other_columns,
                                   inplace=True)

        return event_log
