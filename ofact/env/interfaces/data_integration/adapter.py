"""
# This module handles the execution of data imports from an Excel file, a csv file or a MSSQL database.
Therefore, many different operations are performed, defined in the data_source_model (Excel file),
to transform the data given to a DataFrame, usable for the data integration into the digital twin.

Classes:
    DataTransformationAdapter
    XLSXAdapter
    CSVAdapter
    MSSQLAdapter

@last update: 27.01.2023
"""
# Imports Part 1: Standard Imports
import logging
import os
import csv
from ast import literal_eval
from copy import deepcopy
from enum import Enum

import numpy as np
# Imports Part 2: PIP Imports
import pandas as pd
# import pyodbc to ensure mssql connection
try:
    import pyodbc
except ModuleNotFoundError as e:
    pyodbc = None
    print(e)
except ImportError as e:
    pyodbc = None
    print(e)
# import dotenv to access .env file
try:
    from dotenv import load_dotenv

    # Init load_dotenv()
    load_dotenv()
except ModuleNotFoundError as e:
    load_dotenv = None
    print(e)


pd.options.mode.chained_assignment = None  # default='warn'
# Imports Part 3: Project Imports

# Module-Specific Constants
# logging.basicConfig(filename='data_transformation.log', level=logging.DEBUG)


def find_delimiter(filename) -> str:
    """Used to determine the delimiter dynamically based on the csv file itself"""
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter: str = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


class DataTransformationAdapter:

    def __init__(self, external_source_path, time_restriction_df: pd.DataFrame(), column_mappings_df: pd.DataFrame(),
                 split_df: pd.DataFrame(), clean_up_df: pd.DataFrame(), filters_df: pd.DataFrame(),
                 sort_df: pd.DataFrame()):
        """
        The adapter is used to provide a dataframe usable in the data transformation
        to integrate the information in a standard procedure in the digital twin.
        Different source types such as Excel, csv or MSSQL are supported in different specified adapters
        that inherit from the 'standard' DataTransformationAdapter.
        The standard access point is the 'get_data'-method.

        Parameters
        ----------
        external_source_path: used to access the source
        time_restriction_df: ensures the data is from the specified time window
        (defines the objects that can be used for to ensure the time window is respected)
        column_mappings_df: map the columns given from the source to columns understandable in the transformation
        split_df: split for example row in a defined manner
        clean_up_df: replace elements or delete them
        filters_df: filter elements not needed, respectively not handleable in the digital twin
        sort_df: sort the elements in the dataframe, for the right execution ...
        """

        self.external_source_path = external_source_path
        self.time_restriction_df: pd.DataFrame() = time_restriction_df
        self.column_mappings_df: pd.DataFrame() = column_mappings_df
        self.split_df: pd.DataFrame() = split_df
        self.clean_up_df: pd.DataFrame() = clean_up_df
        self.filters_df: pd.DataFrame() = filters_df
        self.sort_df: pd.DataFrame() = sort_df

    def get_data(self, data_batches: dict[str, tuple[pd.DataFrame(), str]], start_datetime, end_datetime) \
            -> pd.DataFrame:
        """a data source is read and some refinements are made upon the data to bring it into a standardized format"""
        raw_data_df = self._create_dataframe(start_datetime, end_datetime)
        if raw_data_df.empty:
            standardized_raw_data_df = self._transform_columns(raw_data_df)
            return standardized_raw_data_df

        raw_data_df = self._split_dataframe(raw_data_df)
        filtered_raw_data_df = self._filter_dataframe(raw_data_df, data_batches)
        cleaned_raw_data_df = self._clean_dataframe(filtered_raw_data_df)
        sorted_raw_data_df = self._sort_data(cleaned_raw_data_df)
        standardized_raw_data_df = self._transform_columns(sorted_raw_data_df)

        return standardized_raw_data_df

    def _create_dataframe(self, start_datetime, end_datetime) -> pd.DataFrame:
        """the adapter pull data from a data source, respectively they are push from the source itself
        This data is subsequently transformed into a dataframe ..."""
        pass

    def _split_dataframe(self, raw_data_df):
        split_operation_df = self.split_df.groupby(by="operation id")

        for _, split_action_df in split_operation_df:

            for _, split_action_s in split_action_df.iterrows():
                splitting_needed_detection_mask = (
                    raw_data_df[split_action_s["external"]].str.contains(split_action_s["separator"],
                                                                         na=False))
                rows_to_split = raw_data_df.loc[splitting_needed_detection_mask]

                raw_data_df = raw_data_df.drop(index=rows_to_split.index)
                if split_action_s["action"] == "choose" and not rows_to_split.empty:
                    rows_to_split = self._choose(rows_to_split, split_action_s)

                elif split_action_s["action"] == "add_row" and not rows_to_split.empty:
                    rows_to_split = self._add_row(rows_to_split, split_action_s)

                rows_to_split.set_index(np.arange(max(raw_data_df.index) + 1,
                                            max(raw_data_df.index) + 1 + len(rows_to_split)))
                raw_data_df = pd.concat([raw_data_df, rows_to_split])
                raw_data_df = raw_data_df.reset_index(drop=True)

        return raw_data_df

    def _choose(self, rows_affected_df, split_action_s) -> pd.DataFrame:
        # project specific

        return rows_affected_df

    def _add_row(self, rows_affected_df, split_action_s) -> pd.DataFrame:
        # project specific

        return rows_affected_df

    def _filter_dataframe(self, raw_data_df, data_batches: dict[str, tuple[pd.DataFrame(), str]]):
        """filter the raw dataframe"""
        for external, filter_ in self.filters_df.groupby(by="external"):

            # filter based on other data_batches
            other_data_source_needed = deepcopy(filter_['needed entries'].dropna().str.contains("ods*", case=True))
            other_data_source_not_needed = \
                deepcopy(filter_['not needed entries'].dropna().str.contains("ods*", case=True))
            other_data_source_contains = deepcopy(filter_["contains"].dropna())

            if other_data_source_needed.any():
                filter_other_data_source_needed = filter_.loc[other_data_source_needed]
                idx_to_delete = []
                # find the specific row
                for idx, entry in filter_other_data_source_needed['needed entries'].items():
                    if entry[0:4] == "ods*":
                        other_df_name, column_other_df_str = entry[5:-1].split('; ')
                        column_other_df = tuple(column_elem if column_elem != "np.nan" else np.nan
                                                for column_elem in literal_eval(column_other_df_str))

                        other_df = data_batches[other_df_name][0]
                        column_names = [column_name for column_name in other_df.columns
                                        if column_name[0] == column_other_df[0]]
                        other_df_elements = other_df[column_names[0]]
                        raw_data_df = raw_data_df.loc[raw_data_df[external].isin(other_df_elements)]
                        idx_to_delete.append(idx)
                #  delete the row from filter_
                filter_ = filter_.drop(idx_to_delete)

            if other_data_source_contains.any():
                filter_contains = filter_.loc[other_data_source_contains.index, ["external", "contains"]]
                for group, group_df in filter_contains.groupby(by="external"):
                    accepted_values = group_df["contains"].to_list()
                    raw_data_df = raw_data_df.loc[raw_data_df[group].str.contains('|'.join(accepted_values), na=False)]

            if other_data_source_not_needed.any():
                raise NotImplementedError

            if filter_.empty:
                continue

            # filter based on static entries
            if not filter_["needed entries"].isna().any():
                filter_needed_entries = filter_.loc[~filter_["needed entries"].isna()]
                filter_needed_entries["needed entries"] = filter_needed_entries["needed entries"].replace('""', np.nan)
                raw_data_df = raw_data_df.loc[raw_data_df[external].isin(filter_needed_entries["needed entries"])]
            if not filter_["not needed entries"].isna().any():
                filter_not_needed_entries = filter_.loc[~filter_["not needed entries"].isna()]
                filter_not_needed_entries["needed entries"] = \
                    filter_not_needed_entries["not needed entries"].replace('""', np.nan)
                raw_data_df = \
                    raw_data_df.loc[~raw_data_df[external].isin(filter_not_needed_entries["not needed entries"])]

        return raw_data_df

    def _clean_dataframe(self, raw_data_df):
        """Clean the dataframe by replacing and deleting rows that have a specified content (set in the excel file)"""
        for external, cleaner in self.clean_up_df.groupby(by="external"):
            entries_to_delete = cleaner.loc[cleaner["delete"] == True, "old value"]
            entries_to_replace = cleaner.loc[cleaner["delete"] == False, ["old value", "replacing value"]]

            for idx, entry_to_delete in entries_to_delete.items():
                if entry_to_delete != entry_to_delete:
                    raw_data_df = raw_data_df.loc[~raw_data_df[external].isnull()]
                else:
                    raw_data_df = raw_data_df.loc[raw_data_df[external] == entry_to_delete]

            for idx, entry_to_replace in entries_to_replace.iterrows():
                raw_data_df[external] = raw_data_df[external].replace(entry_to_replace["old value"],
                                                                      entry_to_replace["replacing value"])

        return raw_data_df

    def _sort_data(self, raw_data_df):
        raw_data_df = raw_data_df.sort_values(by=self.sort_df["external"].to_list(), ascending=True, na_position='last',
                                              ignore_index=True)
        return raw_data_df

    def _transform_columns(self, raw_data_df):
        """

        For example:
        Column-Mapping (Matching of external parameter names to the internal ones)
        "Freigegeben am" = Order_date, resource-external-id = resource-internal-id
        """

        old_column_names = self.column_mappings_df["external"]
        if len(list(old_column_names)) > len(set(old_column_names)):
            new_columns_needed = deepcopy(list(old_column_names))
            for column_name in set(old_column_names):
                new_columns_needed.remove(column_name)

            for column_name in set(new_columns_needed):
                for appendix in [str(x) for x in old_column_names.loc[old_column_names == column_name].index]:
                    # raw_data_df[column_name + appendix] = raw_data_df[column_name]
                    raw_data_df = raw_data_df.assign(**{str(column_name + appendix): raw_data_df[column_name]})
                del raw_data_df[column_name]

                old_column_names.loc[old_column_names == column_name] += \
                    [str(x) for x in old_column_names.loc[old_column_names == column_name].index]

        new_column_names = list(zip(self.column_mappings_df["mapping identification"].to_list(),
                                    self.column_mappings_df["mapping reference"].to_list(),
                                    self.column_mappings_df["class"].to_list(),
                                    self.column_mappings_df["attribute"].to_list(),
                                    self.column_mappings_df["handling"].to_list(),
                                    self.column_mappings_df["depends on"].to_list()))
        column_mapping = dict(zip(old_column_names, new_column_names))
        raw_data_df = raw_data_df.rename(columns=column_mapping)
        columns_not_needed = [column for column in raw_data_df.columns if len(column) != 6]
        if columns_not_needed:
            raw_data_df = raw_data_df.drop(columns=columns_not_needed)

        return raw_data_df


def _get_partial_dataframe(df, start_datetime, end_datetime, column, none_values_accepted):
    df_partial = deepcopy(df)

    none_values_accepted_mask = None
    if none_values_accepted:
        none_values_accepted_mask = (df_partial[column] != df_partial[column])

    datetime_series = pd.to_datetime(df_partial[column], format='%Y-%m-%d %H:%M:%S.%f')
    if start_datetime and end_datetime:
        start_end_datetime_mask = (start_datetime <= datetime_series) | (df_partial[column] != df_partial[column])
        if none_values_accepted_mask is not None:
            start_end_datetime_mask = start_end_datetime_mask | none_values_accepted_mask
        df_partial = df_partial.loc[start_end_datetime_mask]

    if start_datetime:
        # not determined times are also considered to ensure that planned positions are also possible
        start_datetime_mask = (start_datetime <= datetime_series)
        if none_values_accepted_mask is not None:
            start_datetime_mask = start_datetime_mask | none_values_accepted_mask
        df_partial = df_partial.loc[start_datetime_mask]

    if end_datetime:
        end_datetime_mask = datetime_series <= end_datetime
        if none_values_accepted_mask is not None:
            end_datetime_mask = end_datetime_mask | none_values_accepted_mask
        df_partial = df_partial.loc[end_datetime_mask]

    return list(df_partial.index)


class XLSXAdapter(DataTransformationAdapter):
    """"Adapter for XLSX files"""

    def __init__(self, external_source_path, time_restriction_df: pd.DataFrame(), column_mappings_df: pd.DataFrame(),
                 split_df: pd.DataFrame(), clean_up_df: pd.DataFrame(), filters_df: pd.DataFrame(),
                 sort_df: pd.DataFrame()):
        super(XLSXAdapter, self).__init__(external_source_path=external_source_path,
                                          time_restriction_df=time_restriction_df,
                                          column_mappings_df=column_mappings_df, split_df=split_df,
                                          clean_up_df=clean_up_df, filters_df=filters_df, sort_df=sort_df)

    def _create_dataframe(self, start_datetime, end_datetime) -> pd.DataFrame:
        """the adapter pull data from a data source, respectively they are push from the source itself
        These data is subsequently transformed into a dataframe ..."""
        df = pd.read_excel(self.external_source_path)

        if self.time_restriction_df.empty:
            return df

        df_indexes = []
        min_one_column_filled = True
        columns = []
        for idx, (column, none_values_accepted, min_one_column_filled) in self.time_restriction_df.iterrows():
            df_indexes += _get_partial_dataframe(df, start_datetime, end_datetime, column,
                                                 none_values_accepted)

            columns.append(column)

        if min_one_column_filled:
            possible_indexes = list(df[columns].dropna(thresh=1).index)
            df_indexes = list(set(df_indexes).intersection(set(possible_indexes)))

        df_indexes = list(set(df_indexes))

        filtered_df = df.loc[df_indexes]

        return filtered_df


class CSVAdapter(DataTransformationAdapter):
    """"Adapter for CSV files"""

    def __init__(self, external_source_path, time_restriction_df: pd.DataFrame(), column_mappings_df: pd.DataFrame(),
                 split_df: pd.DataFrame(), clean_up_df: pd.DataFrame(), filters_df: pd.DataFrame(),
                 sort_df: pd.DataFrame()):
        super(CSVAdapter, self).__init__(external_source_path=external_source_path,
                                         time_restriction_df=time_restriction_df,
                                         column_mappings_df=column_mappings_df, split_df=split_df,
                                         clean_up_df=clean_up_df, filters_df=filters_df, sort_df=sort_df)

    def _create_dataframe(self, start_datetime, end_datetime) -> pd.DataFrame:
        """the adapter pull data from a data source, respectively they are push from the source itself
        These data is subsequently transformed into a dataframe ..."""
        delimiter: str = find_delimiter(self.external_source_path)
        df = pd.read_csv(self.external_source_path, sep=delimiter)

        df_indexes = []
        min_one_column_filled = True
        columns = []
        for idx, (column, none_values_accepted, min_one_column_filled) in self.time_restriction_df.iterrows():
            df_indexes += _get_partial_dataframe(df, start_datetime, end_datetime, column,
                                                 none_values_accepted)

            columns.append(column)

        if min_one_column_filled:  # should be column dependent
            possible_indexes = list(df[columns].dropna(thresh=1).index)
            df_indexes = list(set(df_indexes).intersection(set(possible_indexes)))

        df_indexes = list(set(df_indexes))

        filtered_df = df.loc[df_indexes]

        return filtered_df


class MSSQLAdapter(DataTransformationAdapter):
    """An adapter class that allows the retrieval of data from a Microsoft SQL Server database,
    and transforming it into a dataframe that can be used by the DataTransformationAdapter."""

    db_connection = None

    def __init__(self, external_source_path, time_restriction_df: pd.DataFrame(), column_mappings_df: pd.DataFrame(),
                 split_df: pd.DataFrame(), clean_up_df: pd.DataFrame(), filters_df: pd.DataFrame(),
                 sort_df: pd.DataFrame()):
        # Initialize the parent class 'DataTransformationAdapter' with the provided arguments
        super(MSSQLAdapter, self).__init__(external_source_path=external_source_path,
                                           time_restriction_df=time_restriction_df,
                                           column_mappings_df=column_mappings_df, split_df=split_df,
                                           clean_up_df=clean_up_df, filters_df=filters_df, sort_df=sort_df)

        # check if the class variable 'db_connection' already exists. If not, create a new connection
        if MSSQLAdapter.db_connection is None:
            MSSQLAdapter.db_connection = self.mssql_connection()

    def mssql_connection(self):
        """
        This method fetches the user credentials from the.env file and creates a database connection string.

        Returns
        -------
        pypodbc connection object
        """
        # Define, if win_auth should be used to ensure the database connection
        use_win_auth = eval(os.environ.get('DB_USE_WIN_AUTH'))

        if use_win_auth:
            # Fetch db credentials from .env file
            connection_str = (
                f"DRIVER={os.environ.get('DB_DRIVER')};"  # Newer version works with  {ODBC Driver 18 for SQL Server}
                f"SERVER={os.environ.get('DB_SERVER')};"  # Fetch server name
                f"DATABASE={os.environ.get('DB_DB_NAME')};"  # Fetch database name
                f"ENCRYPT=no;"  # Define encrypted connection
                f"TrustServerCertificate=yes;"
                f"Trusted_Connection=yes;")  # Define trusted connection
        else:
            connection_str = (
                f"DRIVER={os.environ['DB_DRIVER']};"  # newer version works with {ODBC Driver 18 for SQL Server}
                f"SERVER={os.environ['DB_SERVER']};"
                f"DATABASE={os.environ['DB_DB_NAME']};"
                f"ENCRYPT=no;"
                f"UID={os.environ['DB_AUTH_USER']};"  # Fetch user name
                f"PWD={os.environ['DB_AUTH_PASSWORD']};"  # Fetch password
                f"TrustServerCertificate=yes;")  # Fetch password

        try:
            db_connection = pyodbc.connect(connection_str)
            logging.debug(f"Connected to the database with connection string {connection_str}")
        except pyodbc.Error as ex:

            logging.exception(f"Error connecting to the database: %s Connection-String: {connection_str}", ex)
            raise Exception(f"Error connecting to the database: %s Connection-String: {connection_str}")

        return db_connection

    def _create_dataframe(self, start_datetime, end_datetime) -> pd.DataFrame:
        """
        Define a sql_query and create a dataframe from the results.

        Parameters
        ----------
        start_datetime: The start date and time to use in the SQL statements.
        end_datetime: The end date and time to use in the SQL statements.

        Returns
        -------
        A tuple of dataframes containing the results of the SQL executions.
        """
        sql_query = (f"EXEC {self.external_source_path} "
                     f"'{self._get_datetime_str(start_datetime)}', '{self._get_datetime_str(end_datetime)}'")
        # Execute SQL statements and store results in dataframes
        df = pd.read_sql(sql_query, MSSQLAdapter.db_connection)

        return df

    def _get_datetime_str(self, datetime_):
        return datetime_.strftime("%d.%m.%Y %H:%M:%S")


class AdapterMapper:
    class AdapterTypes(Enum):
        xlsx = "xlsx"
        csv = "csv"
        mqtt = "MQTT"
        sql = "SQL"
        mssql = "MSSQL"

    adapter_mapper = {AdapterTypes.xlsx: XLSXAdapter,
                      AdapterTypes.csv: CSVAdapter,
                      AdapterTypes.mssql: MSSQLAdapter}

    @classmethod
    def get_adapter_class(cls, source_type: AdapterTypes):
        adapter_class = cls.adapter_mapper[source_type]
        return adapter_class
