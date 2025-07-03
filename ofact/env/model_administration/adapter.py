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
from abc import abstractmethod
from ast import literal_eval
from datetime import timedelta
from copy import deepcopy
from enum import Enum
from typing import Optional

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

timedelta()  # ensure import


class DataAdapter:

    def __init__(self, external_source_path, data_entry_mappings, settings: Optional[dict] = None):

        self.external_source_path = external_source_path
        self.data_entry_mappings: dict = data_entry_mappings
        self.settings = settings

    def get_data(self, start_datetime, end_datetime) \
            -> pd.DataFrame:
        """a data source is read and some refinements are made upon the data to bring it into a standardized format"""
        raw_data_df = self._create_dataframe(start_datetime, end_datetime)

        return raw_data_df

    @abstractmethod
    def _create_dataframe(self, start_datetime, end_datetime) -> pd.DataFrame:
        """the adapter pull data from a data source, respectively they are push from the source itself
        This data is subsequently transformed into a dataframe ..."""
        pass


def _get_partial_dataframe(df, start_datetime, end_datetime, column, allowed_rows_settings):
    df_partial = deepcopy(df)

    if allowed_rows_settings:
        none_values_accepted_mask = True
        for combi, mandatory_column_groups in allowed_rows_settings.items():
            if column in mandatory_column_groups:
                for mandatory_column in mandatory_column_groups:
                    # one of each group should be taken
                    none_values_accepted_mask |= (df_partial[mandatory_column] == df_partial[mandatory_column])

        if isinstance(none_values_accepted_mask, bool):
            none_values_accepted_mask = (df_partial[column] != df_partial[column])  # not mandatory
    else:
        none_values_accepted_mask = (df_partial[column] != df_partial[column])

    datetime_series = pd.to_datetime(df_partial[column],
                                     format='%Y-%m-%d %H:%M:%S.%f')
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


class XLSXAdapter(DataAdapter):
    """"Adapter for XLSX files"""

    def __init__(self, external_source_path, data_entry_mappings, settings: Optional[dict] = None):
        super(XLSXAdapter, self).__init__(external_source_path=external_source_path,
                                          data_entry_mappings=data_entry_mappings, settings=settings)

    def _create_dataframe(self, start_datetime, end_datetime) -> pd.DataFrame:
        """the adapter pull data from a data source, respectively they are push from the source itself
        These data is subsequently transformed into a dataframe ..."""
        if "skiprows" in self.settings:
            df = pd.read_excel(self.external_source_path, skiprows=self.settings["skiprows"], nrows=100)
        else:
            df = pd.read_excel(self.external_source_path, nrows=100)

        for data_entry_mapping in self.data_entry_mappings:
            if data_entry_mapping.special_handling is not None:
                for method in data_entry_mapping.special_handling:

                    external_name = data_entry_mapping.external_name
                    try:
                        df[external_name]
                    except KeyError:
                        substrings = external_name.split(" ")
                        for col in df.columns:  # in case of e.g. "\n" within the string
                            if all(sub in col for sub in substrings):
                                print(f"Column name {external_name} does not match and is replaced by {col}")
                                external_name = col
                                break


                    if method == "ffill":
                        df[external_name] = df[external_name].ffill()
                    elif "timedelta" in method:
                        if method[0] == "+":
                            df[external_name] = df[external_name] + eval(method[1:])
                        elif method[0] == "-":
                            df[external_name] = df[external_name] - eval(method[1:])

        df.columns = df.columns.str.replace('\n', ' ')  # Maybe to specific

        allowed_rows_settings = {}
        time_filters = {}
        for data_entry_mapping in self.data_entry_mappings:
            if data_entry_mapping.mandatory is not None:
                # from each group at least one item should be considered
                allowed_rows_settings.setdefault(data_entry_mapping.mandatory,
                                                 []).append(data_entry_mapping.external_name)
            if data_entry_mapping.time_filter is not None:
                time_filters.setdefault(data_entry_mapping.time_filter,
                                                 []).append(data_entry_mapping.external_name)

        if not allowed_rows_settings and not time_filters:
            return df

        df_indexes = []
        columns = []
        for combi, time_filter in time_filters.items():
            for column in time_filter:
                df_indexes += _get_partial_dataframe(df, start_datetime, end_datetime, column, allowed_rows_settings)

                columns.append(column)

        if allowed_rows_settings and time_filters:
            for combi, mandatory_columns in time_filters.items():
                possible_indexes = list(df[mandatory_columns].dropna(thresh=1).index)
                df_indexes = list(set(df_indexes).intersection(set(possible_indexes)))
        elif allowed_rows_settings:
            for combi, mandatory_columns in allowed_rows_settings.items():
                df_indexes += list(df.loc[df[mandatory_columns].notna().any(axis=1)].index)

        df_indexes = list(set(df_indexes))

        filtered_df = df.loc[df_indexes]

        return filtered_df


def find_delimiter(filename) -> str:
    """Used to determine the delimiter dynamically based on the csv file itself"""
    sniffer = csv.Sniffer()
    with open(filename) as fp:
        delimiter: str = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


class CSVAdapter(DataAdapter):
    """"Adapter for CSV files"""

    def __init__(self, external_source_path, data_entry_mappings, settings: Optional[dict] = None):
        super(CSVAdapter, self).__init__(external_source_path=external_source_path,
                                         data_entry_mappings=data_entry_mappings, settings=settings)

    def _create_dataframe(self, start_datetime, end_datetime) -> pd.DataFrame:
        """the adapter pull data from a data source, respectively they are push from the source itself
        These data is subsequently transformed into a dataframe ..."""
        delimiter: str = find_delimiter(self.external_source_path)
        df = pd.read_csv(self.external_source_path, sep=delimiter)

        allowed_rows_settings = {}
        time_filters = {}
        for data_entry_mapping in self.data_entry_mappings:
            if data_entry_mapping.mandatory == data_entry_mapping.mandatory:
                # from each group at least one item should be considered
                allowed_rows_settings.setdefault(data_entry_mapping.mandatory,
                                                 []).append(data_entry_mapping.external_name)
            if data_entry_mapping.time_filter == data_entry_mapping.time_filter:
                time_filters.setdefault(data_entry_mapping.time_filter,
                                        []).append(data_entry_mapping.external_name)

        if not allowed_rows_settings and not time_filters:
            return df

        df_indexes = []
        columns = []
        for combi, time_filter in time_filters.items():
            for column in time_filter:
                df_indexes += _get_partial_dataframe(df, start_datetime, end_datetime, column, allowed_rows_settings)

                columns.append(column)

        if allowed_rows_settings:
            for combi, mandatory_columns in time_filters.items():
                possible_indexes = list(df[mandatory_columns].dropna(thresh=1).index)
                df_indexes = list(set(df_indexes).intersection(set(possible_indexes)))

        df_indexes = list(set(df_indexes))

        filtered_df = df.loc[df_indexes]

        return filtered_df


class MSSQLAdapter(DataAdapter):
    """An adapter class that allows the retrieval of data from a Microsoft SQL Server database,
    and transforming it into a dataframe that can be used by the DataTransformationAdapter."""

    db_connection = None

    def __init__(self, external_source_path, data_entry_mappings, settings: Optional[dict] = None):
        # Initialize the parent class 'DataTransformationAdapter' with the provided arguments
        super(MSSQLAdapter, self).__init__(external_source_path=external_source_path,
                                           data_entry_mappings=data_entry_mappings, settings=settings)

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