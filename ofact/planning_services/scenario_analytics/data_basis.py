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

The file contains the data-basis for the kpi calculation/ scenario analytics.
To enable the kpi calculation, the information required is extracted from the digital twin state model,
especially the process_executions and partly from the orders.
The data provided is subsequently used for the constructive services, such as the kpi calculation and chart creation.
Relevant views: Order, Part, Process, Resource

classes:
    ScenarioAnalyticsDataBase

@contact persons: Jannik, Adrian Freiter
"""

# Imports Part 1: Standard Imports
from __future__ import annotations

import datetime
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Optional

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.twin.state_model.entities import Resource, Part, Storage, ConveyorBelt
from ofact.twin.state_model.process_models import EntityTransformationNode

# import polars as pl

if TYPE_CHECKING:
    from ofact.twin.state_model.model import StateModel

    from ofact.twin.state_model.processes import ProcessExecution, Process
    from ofact.twin.state_model.sales import Order

# Module-Specific Constants
MEMOIZATION_MAX = 20000
# Ignore Feature Warnings from pandas modul
warnings.simplefilter(action='ignore', category=FutureWarning)


no_part_id = 9999995
no_customer_id = 9999997
no_feature_id = 9999998
no_order_id = 9999999

def _get_id(object_):
    """
    Used to get the corresponding id to an object if not None.

    Parameters
    ----------
    object_: a digital_twin object

    Returns
    -------
    object_: if the object is not None an id else a nan value
    """
    if object_ is not None and object_ == object_:  # nan value also possible
        try:
            return object_.identification
        except AttributeError:
            return object_

    else:
        return np.nan


def _get_enum(enum_) -> np.nan | str:
    """
    Assuming that the enum_ is an instance of an enum.

    Parameters
    ----------
    enum_: an enum object

    Returns
    -------
    enum_: if the enum_ is not None, an id else nan value
    """

    if enum_ is None:
        return np.nan

    if enum_ is not None and enum_ == enum_:  # nan value also possible
        return enum_.name

    else:
        return np.nan


def _get_process_group(process: Optional[Process]):
    """
    Process groups are currently required to differentiate between parallel material flows
    (e.g., main product and material supply - lead times should be calculated separately)
    """

    if process.group is not None:
        if isinstance(process.group, str):
            return process.group
        else:
            return process.group.name
    else:
        return ""


def _get_order_df(order: Order):
    """
    Define an order dataframe row based on the order object

    Parameters
    ----------
    order: the order object that provides his attributes
    """

    if order.customer is not None:
        customer_name = order.get_customer_name()
        customer_id = _get_id(order.customer)

    else:
        customer_name = "Unknown"
        customer_id = no_customer_id

    or_row = np.empty(10, dtype=object)

    if order.products:
        order_products = [_get_id(product)
                          for product in order.products]
        order_product = order_products[0]
    else:
        order_product = np.nan

    if order.product_classes:
        order_product_classes = [_get_id(product_class)
                                 for product_class in order.product_classes]
        order_product_class = order_product_classes[0]
    else:
        order_product_class = np.nan

    or_row[0:10] = np.array([_get_id(order),  # "Order ID"
                             customer_name,  # "Order Name"
                             order.get_identifier(),  # Order Identifier
                             # order.get_progress_status(),
                             order.release_date_actual,  # "Release Date"
                             order.delivery_date_planned,  # "Delivery Date Planned"
                             order.delivery_date_actual,  # "Delivery Date Actual"
                             customer_id,  # "Customer ID"
                             customer_name,  # "Customer Name"
                             order_product,  # "Product ID"
                             order_product_class],  # "Product Entity Type ID"
                            dtype=object)

    return or_row


def _get_process_execution_row(process_execution: ProcessExecution):
    """
    Define for each process execution a single row in the dataframe

    Parameters
    ----------
    process_execution: the process_execution object that provides his process-specific attributes
    """

    pe_row = \
        np.array([_get_id(process_execution),  # "Process Execution ID"
                  _get_enum(process_execution.event_type),  # "Event Type"
                  process_execution.executed_start_time,  # "Start Time"
                  process_execution.executed_end_time,  # "End Time"
                  _get_id(process_execution.process),  # "Process ID":
                  process_execution.process.name,  # "Process Name":
                  process_execution.process.__class__.__name__,  # "Process Type":
                  _get_process_group(process_execution.process),  # Process Category:
                  process_execution.resulting_quality,  # "Resulting Quality":
                  _get_id(process_execution.main_resource),  # "Main Resource ID"
                  _get_id(process_execution.origin),  # "Origin ID":
                  _get_id(process_execution.destination),  # "Destination ID":
                  _get_id(process_execution.order),  # "Order ID":
                  process_execution.source_application,  # "Source"
                  _get_id(process_execution.connected_process_execution)],  # "Connected Process Execution ID"
                 dtype=object)

    # process (execution) df
    return pe_row


def _get_process_execution_part_row(process_execution: ProcessExecution,
                                    part_transformation_node_tuple: (tuple[Part, EntityTransformationNode] |
                                                                     tuple[Part,] | np.nan)):
    """
    Define for each part_transformation_node_tuple of the process execution a single row in the dataframe

    Parameters
    ----------
    process_execution: the process_execution object that provides his part-specific attributes
    part_transformation_node_tuple: the part tuple that provides his specific attributes
    """
    if part_transformation_node_tuple != part_transformation_node_tuple:
        pa_pe_row = np.array([_get_id(process_execution),  # "Process Execution ID"
                              no_part_id,  # "Part Involved ID"
                              "Kein Teil",  # "Part Involved Name":
                              np.nan,  # "Part Transformation Node ID"
                              no_part_id,  # "Entity Type ID"
                              "Kein Teil"],  # "Entity Type Name"
                             dtype=object)
        return pa_pe_row
    part = part_transformation_node_tuple[0]
    if len(part_transformation_node_tuple) == 1:
        part_transformation_node_id = None
    else:
        part_transformation_node_id = _get_id(part_transformation_node_tuple[1])

    pa_pe_row = np.array([_get_id(process_execution),  # "Process Execution ID"
                          _get_id(part),  # "Part Involved ID"
                          part.name,  # "Part Involved Name":
                          part_transformation_node_id,  # "Part Transformation Node ID"
                          _get_id(part.entity_type),  # "Entity Type ID"
                          part.entity_type.name], dtype=object)  # "Entity Type Name"
    return pa_pe_row


def _get_process_execution_resource_row(process_execution: ProcessExecution, resource: Resource):
    """
    Define for each resource of the process execution a single row in the dataframe

    Parameters
    ----------
    process_execution: the process_execution object that provides his resource-specific attributes
    resource: the resource object that provides his specific attributes
    """

    re_pe_row = np.array([_get_id(process_execution),  # "Process Execution ID":
                          _get_id(resource),  # "Resource Used ID"
                          resource.name,  # "Resource Used Name"
                          resource.__class__.__name__,  # "Resource Type"
                          _get_id(resource.situated_in)],  # "Superior Resource Used ID"
                         dtype=object)

    return re_pe_row


def get_inventory_resources(resources: list[Storage | ConveyorBelt],
                            process_execution_process_execution_id_mapper: dict[int, ProcessExecution]):
    timestamp_min = datetime(1900, 1, 1)
    timestamp_max = datetime(2099, 12, 31)

    inventory_rows = []
    for resource in resources:

        storage_resource_id = _get_id(resource)
        superior_storage_resource_id = _get_id(resource.situated_in)

        incoming_and_outgoing_entities_history = resource.get_incoming_and_outgoing_entities_history(timestamp_min,
                                                                                                     timestamp_max)

        def _process_row(stored_entity_row):
            process_execution_id = stored_entity_row["ProcessExecution"]

            if process_execution_id in process_execution_process_execution_id_mapper:
                process_execution = process_execution_process_execution_id_mapper[process_execution_id]
            elif process_execution_id == "INITIAL" or process_execution_id is None:
                process_execution_id = np.nan
                process_execution = None  # initial setting
            else:
                return None  # process execution is not relevant

            event_type = process_execution.event_type.name if process_execution else np.nan
            timestamp = np.datetime64(stored_entity_row["Timestamp"], "ns") if stored_entity_row["Timestamp"] \
                else np.nan
            quantity = stored_entity_row["ChangeType"]

            if isinstance(stored_entity_row["Value"], Part):
                part_id = _get_id(stored_entity_row["Value"])
                part_type_id = _get_id(stored_entity_row["Value"].entity_type)
                resource_id = np.nan
            else:
                part_id = np.nan
                part_type_id = np.nan
                resource_id = _get_id(stored_entity_row["Value"])

            row = [process_execution_id,  # "Process Execution ID"
                   event_type,  # "Event Type"
                   storage_resource_id,  # "Storage Resource ID"
                   superior_storage_resource_id,  # "Superior Storage Resource ID"
                   part_id,  # "Part ID"
                   part_type_id,  # "Part Type ID"
                   resource_id,  # "Resource ID"
                   timestamp,  # "Timestamp"
                   quantity  # "Quantity"
                   ]

            return row

        inventory_entries_resource = incoming_and_outgoing_entities_history.apply(_process_row,
                                                                                  axis=1).dropna().tolist()
        inventory_rows.extend(inventory_entries_resource)

    return inventory_rows


def _get_process_execution_order_row(process_execution: ProcessExecution, orders: list[Order]):
    """
    Define for the order of the process execution a single row in the dataframe

    Parameters
    ----------
    process_execution: the process_execution object that provides his resource-specific attributes
    orders: the orders that can be assigned to the process_execution
    """

    if process_execution.order is not None:
        order = process_execution.order
    else:
        return _get_process_execution_order_not_assignable_row(process_execution, orders)

    for feature_, pe_list in order.feature_process_execution_match.items():
        if feature_ is not None:
            feature_id = _get_id(feature_)
            feature_name = feature_.name
        else:
            feature_id = no_feature_id
            feature_name = "No feature assignable"

        if (process_execution in pe_list or
                process_execution.connected_process_execution in pe_list):

            customer_name = order.get_customer_name()
            if customer_name is None:
                order_name = "Unknown"

            elif customer_name != customer_name:
                order_name = "Unknown"

            else:
                order_name = customer_name

            or_pe_row = np.array([_get_id(process_execution),  # "Process Execution ID"
                                  _get_id(order),  # "Order ID"
                                  order_name,  # "Order Name"
                                  order.get_identifier(),  # Order Identifier
                                  feature_id,  # "Feature Requested ID"
                                  feature_name,  # "Feature Requested Name"
                                  _get_id(order.customer),  # "Customer ID"
                                  order_name], dtype=object)  # "Customer Name"

            return or_pe_row

    return _get_process_execution_order_not_assignable_row(process_execution, orders)


def _get_process_execution_order_not_assignable_row(process_execution: ProcessExecution,
                                                    orders: list[Order]):
    """
    Define a row for the order process execution relation ...

    Parameters
    ----------
    process_execution: the process_execution object that provides his order-specific attributes
    orders: the orders that can be assigned to the process_execution
    """

    # handle not assignable process_executions_components
    order = _get_order_for_process_execution(process_execution, orders)
    if order is not None:

        order_name = order.get_customer_name()
        if order_name != order_name or order_name is None:
            order_name = "Unknown"

        or_pe_row = np.array([_get_id(process_execution),  # "Process Execution ID"
                              _get_id(order),  # "Order ID"
                              order_name,  # "Order Name"
                              order.get_identifier(),
                              no_feature_id,  # "Feature Requested ID"
                              "No feature assignable",  # "Feature Requested Name"
                              _get_id(order.customer),  # "Customer ID"
                              order_name], dtype=object)  # "Customer Name"
    else:
        or_pe_row = np.array([_get_id(process_execution),  # "Process Execution ID"
                              no_order_id,  # "Order ID"
                              "Intern",  # "Order Name"
                              no_order_id,  # Order Identifier
                              no_feature_id,  # "Feature Requested ID"
                              "No feature assignable",  # "Feature Requested Name"
                              no_customer_id,  # "Customer ID"
                              "Intern"],  # "Customer Name"
                             dtype=object)

        print(f"Order not found for process: {process_execution.get_process_name()}")

    return or_pe_row


def _get_order_for_process_execution(process_execution: ProcessExecution, orders: list[Order]) -> (
        Optional[Order]):
    """
    Define a row for the order process execution relation ...

    Parameters
    ----------
    process_execution: the process_execution object that provides his order-specific attributes
    orders: the orders that can be assigned to the process_execution
    """

    if process_execution.order is not None:
        return process_execution.order

    # try to determine the order through the order product
    for order in orders:
        if not order.products:
            continue

        for part_transformation_node in process_execution.parts_involved:
            for product in order.products:
                if product.identification == part_transformation_node[0].identification:
                    return order

    return None


def _get_dummy_process_execution_row(order):
    """
    Define a row for the orders without a process execution

    Parameters
    ----------
    order: the order without any process execution
    """

    pe_row = \
        np.array([1000000,  # "Process Execution ID"
                  np.nan,  # "Event Type"
                  np.nan,  # "Start Time"
                  np.nan,  # "End Time"
                  np.nan,  # "Process ID":
                  np.nan,  # "Process Name":
                  np.nan,  # "Process Type":
                  np.nan,  # Process Category:
                  np.nan,  # "Resulting Quality":
                  np.nan,  # "Main Resource ID"
                  np.nan,  # "Origin ID":
                  np.nan,  # "Destination ID":
                  _get_id(order),  # "Order ID":
                  np.nan,  # "Source"
                  np.nan],  # "Connected Process Execution ID"
                 dtype=object)

    return pe_row


# vectorized functions for the row generation of the dataframes
v_get_process_execution_row = np.vectorize(_get_process_execution_row)
v_get_dummy_process_execution_row = np.vectorize(_get_dummy_process_execution_row)
v_get_process_execution_part_row = np.vectorize(_get_process_execution_part_row)
v_get_process_execution_resource_row = np.vectorize(_get_process_execution_resource_row)
v_get_order_df = np.vectorize(_get_order_df)


def _get_updated_df(old_df, new_rows, columns, drop_duplicates_subset: list[str]):
    """Update the dataframe """

    if not new_rows.any():
        return old_df

    new_df = pd.DataFrame(new_rows, columns=columns)
    old_df = pd.concat([old_df, new_df], ignore_index=True)
    old_df = old_df.drop_duplicates(subset=drop_duplicates_subset, keep="last")
    return old_df


class ScenarioAnalyticsDataBase:
    """
    The AnalyticsDataBase class serves as the data basis for the kpi calculation.
    To ensure the behavior, five dataframes are created.
    One for each data view (Order, Part, Process, Resource) and
    another one that contains the content of all dataframes before.
    """

    ####################################################################################################################
    # #### DATAFRAMES DEFINITION
    ####################################################################################################################

    # ToDo: Name Mapper df ??

    process_execution_df_default = pd.DataFrame(
        {"Process Execution ID": pd.Series([], dtype=np.dtype("int32")),
         "Event Type": pd.Series([], dtype=np.dtype("str")),
         "Start Time": pd.Series([], dtype=np.dtype('datetime64[ns]')),
         "End Time": pd.Series([], dtype=np.dtype('datetime64[ns]')),
         "Process ID": pd.Series([], dtype=np.dtype("int32")),
         "Process Name": pd.Series([], dtype=np.dtype("str")),
         "Process Type": pd.Series([], dtype=np.dtype("str")),
         "Process Category": pd.Series([], dtype=np.dtype('str')),
         "Resulting Quality": pd.Series([], dtype=np.dtype("float")),
         "Main Resource ID": pd.Series([], dtype=np.dtype("int32")),
         "Origin ID": pd.Series([], dtype=np.dtype("int32")),
         "Destination ID": pd.Series([], dtype=np.dtype("int32")),
         "Order ID": pd.Series([], dtype=np.dtype("int32")),
         "Source": pd.Series([], dtype=np.dtype("str")),
         "Connected Process Execution ID": pd.Series([], dtype=np.dtype("int32"))}
    )
    process_execution_part_df_default = pd.DataFrame(
        {"Process Execution ID": pd.Series([], dtype=np.dtype("int32")),
         "Part Involved ID": pd.Series([], dtype=np.dtype("int32")),
         "Part Involved Name": pd.Series([], dtype=np.dtype("str")),
         "Part Transformation Node ID": pd.Series([], dtype=np.dtype("int32")),
         "Entity Type ID": pd.Series([], dtype=np.dtype("int32")),
         "Entity Type Name": pd.Series([], dtype=np.dtype("str"))}
    )
    process_execution_resource_df_default = pd.DataFrame(
        {"Process Execution ID": pd.Series([], dtype=np.dtype("int32")),
         "Resource Used ID": pd.Series([], dtype=np.dtype("int32")),
         "Resource Used Name": pd.Series([], dtype=np.dtype("str")),
         "Resource Type": pd.Series([], dtype=np.dtype("str")),
         "Superior Resource Used ID": pd.Series([], dtype=np.dtype("int32"))}
    )
    process_execution_order_df_default = pd.DataFrame(
        {"Process Execution ID": pd.Series([], dtype=np.dtype("int32")),
         "Order ID": pd.Series([], dtype=np.dtype("int32")),
         "Order Name": pd.Series([], dtype=np.dtype("str")),
         "Order Identifier": pd.Series([], dtype=np.dtype("str")),
         "Feature Requested ID": pd.Series([], dtype=np.dtype("int32")),
         "Feature Requested Name": pd.Series([], dtype=np.dtype("str")),
         "Customer ID": pd.Series([], dtype=np.dtype("int32")),
         "Customer Name": pd.Series([], dtype=np.dtype("str"))}
    )
    inventory_df_default = pd.DataFrame(
        {"Process Execution ID": pd.Series([], dtype=np.dtype("int32")),
         "Event Type": pd.Series([], dtype=np.dtype("str")),
         "Storage Resource ID": pd.Series([], dtype=np.dtype("int32")),
         "Superior Storage Resource ID": pd.Series([], dtype=np.dtype("int32")),
         "Part ID": pd.Series([], dtype=np.dtype("int32")),
         "Part Type ID": pd.Series([], dtype=np.dtype("int32")),
         "Resource ID": pd.Series([], dtype=np.dtype("int32")),
         "Timestamp": pd.Series([], dtype=np.dtype('datetime64[ns]')),
         "Quantity": pd.Series([], dtype=np.dtype("int32"))}
    )
    order_df_default = pd.DataFrame(
        {"Order ID": pd.Series([], dtype=np.dtype("int32")),
         "Order Name": pd.Series([], dtype=np.dtype("str")),
         "Order Identifier": pd.Series([], dtype=np.dtype("str")),
         # "Order Progress": pd.Series([], dtype=np.dtype("int32")),
         "Release Date": pd.Series([], dtype=np.dtype('datetime64[ns]')),
         "Delivery Date Planned": pd.Series([], dtype=np.dtype('datetime64[ns]')),
         "Delivery Date Actual": pd.Series([], dtype=np.dtype('datetime64[ns]')),
         "Customer ID": pd.Series([], dtype=np.dtype("int32")),
         "Customer Name": pd.Series([], dtype=np.dtype("str")),
         "Product ID": pd.Series([], dtype=np.dtype("int32")),
         "Product Entity Type ID": pd.Series([], dtype=np.dtype("int32"))}
    )

    process_execution_df_columns = ["Process Execution ID", "Event Type", "Start Time", "End Time", "Process ID",
                                    "Process Name", "Process Type", "Process Category", "Resulting Quality",
                                    "Main Resource ID", "Origin ID", "Destination ID", "Order ID", "Source",
                                    "Connected Process Execution ID"]
    process_execution_part_df_columns = ["Process Execution ID", "Part Involved ID", "Part Involved Name",
                                         "Part Transformation Node ID", "Entity Type ID", "Entity Type Name"]
    process_execution_resource_df_columns = ["Process Execution ID", "Resource Used ID", "Resource Used Name",
                                             "Resource Type", "Superior Resource Used ID"]
    process_execution_order_df_columns = ["Process Execution ID", "Order ID", "Order Name", "Order Identifier",
                                          "Feature Requested ID", "Feature Requested Name",
                                          "Customer ID", "Customer Name"]
    inventory_df_columns = ["Process Execution ID", "Event Type", "Storage Resource ID", "Superior Storage Resource ID",
                            "Part ID", "Part Type ID", "Resource ID", "Timestamp", "Quantity"]
    order_df_columns = ["Order ID", "Order Name", "Order Identifier",  # "Order Progress",
                        "Release Date", "Delivery Date Planned", "Delivery Date Actual",
                        "Customer ID", "Customer Name", "Product ID", "Product Entity Type ID"]

    reference_value_mapper = [("Order ID", "Order Identifier"),
                              ("Entity Type ID", "Entity Type Name"),
                              ("Process ID", "Process Name"),
                              ("Resource Used ID", "Resource Used Name")]

    def __init__(self, state_model: StateModel):
        """
        Converts the dynamics in the state model/ digital twin to kpi (input) tables.
        These tables are used for the kpi calculation and the base for the scenario analytics.
        Relevant for the tables are the process executions and the orders.

        Parameters
        ----------
        state_model: the state model contains the state as well as the dynamics that are used for the table creation.
        """

        self._state_model: StateModel = state_model
        self._orders: list[Order] = []
        self._inventory_resources: list[Storage | ConveyorBelt] = []

        # initialize the data frames ...
        self.process_execution_df = type(self).process_execution_df_default.copy()
        self.process_execution_part_df = type(self).process_execution_part_df_default.copy()
        self.process_execution_resource_df = type(self).process_execution_resource_df_default.copy()
        self.process_execution_order_df = type(self).process_execution_order_df_default.copy()
        self.inventory_df = type(self).inventory_df_default.copy()
        self.order_df = type(self).order_df_default.copy()

        self.digital_twin_df = pd.DataFrame()
        # self.digital_twin_pl_df = pl.DataFrame()
        self.lead_time_dfs = {}  # managed from the kpi calc

        self.v_get_process_execution_order_row = np.vectorize(self._get_process_execution_order_row)

        self.last_update = datetime(1970, 1, 1)
        if self._state_model is not None:

            dynamics_available = self._state_model.get_process_executions_list()
            if dynamics_available:
                self.update_data()

        print("AnalyticsDataBase instantiation finished")

    ####################################################################################################################
    # #### IN MEMORY DATA_BASE INITIALIZATION
    ####################################################################################################################

    def update_data(self):
        """
        Used to update the dataframes for each view. (order, part, process and resource)
        """

        if self._state_model is None:
            self.process_execution_df = type(self).process_execution_df_default.copy()
            self.process_execution_part_df = type(self).process_execution_part_df_default.copy()
            self.process_execution_resource_df = type(self).process_execution_resource_df_default.copy()
            self.process_execution_order_df = type(self).process_execution_order_df_default.copy()
            self.order_df = type(self).order_df_default.copy()
            self.inventory_df = type(self).inventory_df_default.copy()
            self.update_digital_twin_df()

            return False

        update_process_executions = self._state_model.get_process_executions_list()
        self._orders: list[Order] = self._state_model.get_orders()
        self._inventory_resources: list[Storage | ConveyorBelt] = (self._state_model.get_storages() +
                                                                   self._state_model.get_conveyor_belts())

        if not update_process_executions:
            return False

        pe_rows, pa_pe_rows, re_pe_rows, or_pe_rows, or_rows, in_rows = self._get_rows(update_process_executions)
        self._merge_rows_into_dfs(pe_rows, pa_pe_rows, re_pe_rows, or_pe_rows, or_rows, in_rows)

        pe_rows_dummy = self.get_dummy_rows()
        if pe_rows_dummy is not None:
            self.process_execution_df = _get_updated_df(old_df=self.process_execution_df,
                                                        new_rows=pe_rows_dummy,
                                                        columns=type(self).process_execution_df_columns,
                                                        drop_duplicates_subset=["Process Execution ID", "Order ID"])

        # set working variable back
        self.update_digital_twin_df()

        self.last_update = datetime.now()

        return True

    def _get_rows(self, update_process_executions: list[ProcessExecution]):
        """
        Get for each process_execution in combination with each view a row

        Parameters
        ----------
        update_process_executions: a list of process executions that are used to update the dataframes
        """

        pe_rows = v_get_process_execution_row(update_process_executions)
        pe_rows = np.vstack(pe_rows)

        pa_pe_full = [(process_execution, part_transformation_node_tuple)
                          for process_execution in update_process_executions
                          for part_transformation_node_tuple in process_execution.parts_involved]
        pa_pe_null = [(process_execution, np.nan)
                      for process_execution in update_process_executions
                      if not process_execution.parts_involved]

        pa_pe = np.array(pa_pe_full + pa_pe_null,
                         dtype=object)
        pa_pe_rows = v_get_process_execution_part_row(pa_pe[:, 0], pa_pe[:, 1])
        pa_pe_rows = np.vstack(pa_pe_rows)
        pa_pe_rows = pa_pe_rows[~np.isnan(np.array(pa_pe_rows[:, 0], dtype=float))]

        pe_re = np.array([(process_execution, resource_transformation_node_tuple[0])
                          for process_execution in update_process_executions
                          for resource_transformation_node_tuple in process_execution.resources_used],
                         dtype=object)

        re_pe_rows = v_get_process_execution_resource_row(pe_re[:, 0], pe_re[:, 1])
        re_pe_rows = np.vstack(re_pe_rows)

        or_pe_rows = self.v_get_process_execution_order_row(update_process_executions)
        or_pe_rows = np.vstack(or_pe_rows)

        or_rows = v_get_order_df(self._orders)
        or_rows = np.vstack(or_rows)

        process_execution_process_execution_id_mapper = {process_execution.identification: process_execution
                                                         for process_execution in update_process_executions}
        if self._inventory_resources:
            in_rows = np.array(get_inventory_resources(self._inventory_resources,
                                                       process_execution_process_execution_id_mapper))
        else:
            in_rows = np.array([])

        return pe_rows, pa_pe_rows, re_pe_rows, or_pe_rows, or_rows, in_rows

    def _get_process_execution_order_row(self, process_execution: ProcessExecution):
        """
        Get the order row for the process_execution.

        Parameters
        ----------
        process_execution: the process_execution object that provides his order specific attributes
        """
        return _get_process_execution_order_row(process_execution, self._orders)

    def get_dummy_rows(self) -> Optional[np.array]:
        """Used for orders without process_executions to consider them on the dashboard"""
        order_ids_with_pe = set(self.process_execution_df["Order ID"].to_list())
        order_ids_all = set(self.order_df["Order ID"].to_list())
        order_ids_without_pe = list(order_ids_all.difference(order_ids_with_pe))
        if not order_ids_without_pe:
            return None

        pe_rows = v_get_dummy_process_execution_row(order_ids_without_pe)
        pe_rows = np.vstack(pe_rows)

        return pe_rows

    def _merge_rows_into_dfs(self, pe_rows, pa_pe_rows, re_pe_rows, or_pe_rows, or_rows, in_rows):
        """
        Merge the rows into the dfs
        Consider the existing entries and overwrite them if already existing

        Parameters
        ----------
        pe_rows: process_execution rows
        pa_pe_rows: part process_execution rows
        re_pe_rows: resource process_execution rows
        or_pe_rows: order process_execution rows
        or_rows: order rows
        in_rows: inventory rows
        """
        self.process_execution_df = _get_updated_df(old_df=self.process_execution_df,
                                                    new_rows=pe_rows,
                                                    columns=type(self).process_execution_df_columns,
                                                    drop_duplicates_subset=["Process Execution ID"])
        self.process_execution_part_df = (
            _get_updated_df(old_df=self.process_execution_part_df,
                            new_rows=pa_pe_rows,
                            columns=type(self).process_execution_part_df_columns,
                            drop_duplicates_subset=["Process Execution ID", "Part Involved ID"]))
        self.process_execution_resource_df = (
            _get_updated_df(old_df=self.process_execution_resource_df,
                            new_rows=re_pe_rows,
                            columns=type(self).process_execution_resource_df_columns,
                            drop_duplicates_subset=["Process Execution ID", "Resource Used ID"]))

        self.process_execution_order_df = (
            _get_updated_df(old_df=self.process_execution_order_df,
                            new_rows=or_pe_rows,
                            columns=type(self).process_execution_order_df_columns,
                            drop_duplicates_subset=["Process Execution ID", "Order ID"]))
        self.order_df = _get_updated_df(old_df=self.order_df,
                                        new_rows=or_rows,
                                        columns=type(self).order_df_columns,
                                        drop_duplicates_subset=["Order ID"])

        inventory_df_drop_duplicates_subset = \
            ["Timestamp", "Process Execution ID", "Resource ID", "Part ID", "Quantity"]
        self.inventory_df = _get_updated_df(old_df=self.inventory_df,
                                            new_rows=in_rows,
                                            columns=type(self).inventory_df_columns,
                                            drop_duplicates_subset=inventory_df_drop_duplicates_subset)

    def update_digital_twin_df(self):
        """
        Set up the digital twin df.
        Setup references of the digital_twin_df entries.
        """

        self.digital_twin_df = pd.concat([self.process_execution_df,
                                          self.process_execution_resource_df,
                                          self.process_execution_part_df,
                                          self.process_execution_order_df,
                                          self.order_df],
                                         ignore_index=True, sort=False)

        # self.digital_twin_pl_df = pl.from_pandas(self.digital_twin_df)

        if self.digital_twin_df.empty:
            return

        self.digital_twin_df.insert(0, "reference", " ")

        reference_value_mapper = type(self).reference_value_mapper

        for view_id, view_column_name in reference_value_mapper:
            view_df = self.digital_twin_df[view_id]
            view_mask = view_df.notnull()
            view_reference = self.digital_twin_df[view_mask][view_column_name]

            self.digital_twin_df.loc[view_mask, "reference"] = view_reference

    def update_dataframes_from_database(self, tables_from_database: dict[str, pd.DataFrame]):
        """
        Update the dataframes from the database,
        that can be used as an alternative to the extraction from the state model.

        Parameters
        ----------
        tables_from_database: a dictionary of tables from the database, that is mapped through the keys.
        """

        dict_df = {"PROCESS_EXECUTION": self.process_execution_df,
                   "PROCESS_EXECUTION_ORDER": self.process_execution_order_df,
                   "PROCESS_EXECUTION_PART": self.process_execution_part_df,
                   "PROCESS_EXECUTION_RESOURCE": self.process_execution_resource_df,
                   "INVENTORY": self.inventory_df,
                   "ORDER": self.order_df}

        for table_name, df in tables_from_database.items():
            dict_df[table_name] = df
