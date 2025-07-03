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

Persist scenario analytics data base to a sql data_base (e.g. postgres)
"""
# Imports Part 1: Standard Imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional

# Imports Part 2: PIP Imports
try:  # if not installed and not required ...
    import psycopg2
    from psycopg2 import Error
    from psycopg2.extras import execute_values
except ModuleNotFoundError:
    pass

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

# Imports Part 3: Project Imports


def _get_query_statement(table_name, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None, scenario_identification: str = None):
    """
    Define the sql query to read from the database, restricted by the start time, end time and the scenario id.
    # ToDo: testing required ...
    """

    sql_query = (f'SELECT * FROM "{table_name}" '
                 f'WHERE "Scenario_ID" = \'{scenario_identification}\'')

    if table_name == 'PROCESS_EXECUTION':
        if start_time or end_time:
            sql_query += ' AND'

        if start_time is not None:
            sql_query += f' "Start Time" > \'{start_time}\' OR "Start Time" = \'{end_time}\''

        if end_time is not None:
            if start_time is not None:
                sql_query += ' OR'

            sql_query += f' "End Time" < \'{end_time}\' OR "End Time" = \'{end_time}\''

    # ToDo: the other tables should use the process executions ids and the order ids available
    #  from the PROCESS_EXECUTION table

    return sql_query


class DataBaseController:

    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 5432,
                 database="data_base",
                 user: str = "postgres",
                 password: str = "postgres",
                 table_data_types_list: dict[str, pd.DataFrame] = None):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None
        self.cur = None

        self.create_database(database)

        if table_data_types_list is not None:
            self.connect()
            for table_name, df in table_data_types_list.items():
                self.create_table(table_name, df)

            self.disconnect()

    def connect(self, database=None):
        if database is None:
            database = self.database
        self.conn = psycopg2.connect(host=self.host,
                                     port=self.port,
                                     database=database,
                                     user=self.user,
                                     password=self.password)
        self.cur = self.conn.cursor()

    def disconnect(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def check_database_exists(self, database):
        try:
            # Connect to the 'postgres' database to check for the existence of the target database
            self.connect(database='postgres')
            self.cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (database,))
            exists = self.cur.fetchone()
            self.disconnect()
            return exists is not None
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error checking if database exists: {error}")
            self.disconnect()
            return False

    def create_database(self, database):
        if self.check_database_exists(database):
            print(f"Database '{database}' already exists.")
            return

        try:
            # Connect to the 'postgres' database to create the new database
            self.connect(database='postgres')
            self.conn.autocommit = True
            self.cur.execute(f"CREATE DATABASE {database}")
            print(f"Database '{database}' created successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error creating database '{database}': {error}")
        finally:
            self.disconnect()

    def create_table(self, table_name, df):
        create_table_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
        for column in df.columns:
            column_type = df[column].dtype
            if column_type == "int32" or column_type == "int64":
                column_type = "bigint"
            elif column_type == "float64":
                column_type = "FLOAT"
            elif column_type == "object":
                column_type = "TEXT"
            elif column_type == "datetime64[ns]":
                column_type = "TIMESTAMP"
            create_table_query += f'"{"_".join(column.split(" "))}" {column_type}, '
        create_table_query = create_table_query[:-2]
        create_table_query += ")"

        self.cur = self.conn.cursor()
        self.cur.execute(create_table_query)
        self.conn.commit()

    def delete_all_tables(self):
        self.connect()
        try:
            # Connect to the PostgreSQL database
            self.conn.autocommit = True  # Enable autocommit mode

            # Query to list all tables in the public schema
            list_tables_query = """
            SELECT tablename
            FROM pg_catalog.pg_tables
            WHERE schemaname = 'public'
            """

            self.cur.execute(list_tables_query)
            tables = self.cur.fetchall()

            # Generate and execute DROP TABLE statements for each table
            for table in tables:
                drop_table_query = f'DROP TABLE IF EXISTS "{table[0]}" CASCADE'
                print(f"Executing: {drop_table_query}")
                self.cur.execute(drop_table_query)

            print("All tables deleted successfully.")

            self.cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if self.conn is not None:
                self.conn.close()

    def insert_data(self, table_name, df):

        data = [tuple(elem if elem == elem else None for elem in row)
                for row in df.itertuples(index=False)]

        columns = ", ".join(f'"{"_".join(col.split(" "))}"' for col in df.columns)

        insert_query = f'INSERT INTO "{table_name}" ({columns}) VALUES %s'
        self.cur = self.conn.cursor()

        try:
            execute_values(self.cur, insert_query, data)
        except Exception as e:
            print(f"Error occurred during bulk insertion: {e}")
            placeholders = ", ".join("%s" for _ in df.columns)
            insert_query = f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'

            try:
                for row in data:
                    try:
                        self.cur.execute(insert_query, row)
                        print("Successful:", row)
                    except Error as e:
                        print("Error:", e)
                        print("Data:", row)
                    except TypeError as e:
                        print("Error:", e)
                        print("Data:", row)
            except Error as e:
                print("Error occurred during insertion:", e)

        self.conn.commit()

    def store_dataframes(self, dict_df: dict[str, pd.DataFrame]):
        self.connect()
        for table_name, df in dict_df.items():
            self.insert_data(table_name, df)

        self.disconnect()

    def store_dataframe(self, df, table_name):
        self.connect()
        self.insert_data(table_name, df)

        self.disconnect()

    def read_tables(self, table_names: list[str], start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None, scenario_identification: str = None) -> (
            dict[str, pd.DataFrame]):

        table_output = {table_name: self.read_table(table_name, start_time, end_time, scenario_identification)
                        for table_name in table_names}

        return table_output

    def read_table(self, table_name, start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None, scenario_identification: str = None):
        self.connect()
        query = _get_query_statement(table_name, start_time, end_time, scenario_identification)
        df = pd.read_sql_query(query, self.conn)
        self.disconnect()
        return df
