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

This module encapsulates all Services to Calculate KPIs from the digital Twin model
Relevant views: Order, Part, Process, Resource
@last update: ?.?.2024

TODO: Product Entity Type ID (goal conflict - only show products or also other processes...)
"""
from __future__ import annotations

# Imports Part 1: Standard Imports
# Ignore Feature Warnings from pandas modul
import warnings
from datetime import datetime, timedelta
from functools import wraps

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase, no_part_id

warnings.simplefilter(action='ignore', category=FutureWarning)
# Module-Specific Constants
MEMOIZATION_MAX = 20000


# ------------------------------------------------------------------------------------------------

def biased(value):
    """needed to avoid division by zero"""
    value += 1e-12
    return value


def memoize_get_amount_objects(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.
    """

    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_get_amount_pe(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.
    """
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_get_target_quantity(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.
    """
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_get_relevant_process_execution_ids(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.
    """
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_get_common_pe_ids(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.
    """
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


def memoize_get_orders_finished(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.
    """
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


# ------------------------------------------------------------------------------------------------------

class DisplaySequences:

    def get_process_sequence(self, digital_twin_df: pd.DataFrame):
        """
        Sort the process ids to ensure that the processes are represented in a predefined sequence
        for a better readability
        """
        process_available_mask = digital_twin_df["Process ID"].notnull()
        process_ids = set(digital_twin_df.loc[process_available_mask, "Process ID"])

        process_df = self._get_relevant_start_times_df(digital_twin_df)  # sorting according order start date

        process_df = process_df[process_df["Process ID"].isin(process_ids)]
        process_df = process_df.groupby(["Process ID"])["new Time"].agg("mean")
        process_df = process_df.sort_values()

        return process_df.index.drop_duplicates().to_list()

    def _get_relevant_start_times_df(self, digital_twin_df: pd.DataFrame):
        """Get the difference between order start time and the relative process start times of the orders"""

        order_mask = digital_twin_df["Order ID"].notnull()
        relevant_start_times_df = digital_twin_df.loc[order_mask, ["Process ID", "Start Time", "Order ID"]]
        order_min_df = relevant_start_times_df[["Start Time", "Order ID"]]
        order_min_df.loc[order_min_df["Start Time"] != order_min_df["Start Time"], "Start Time"] = (
                order_min_df["Start Time"].dropna().max() + pd.Timedelta(seconds=1))
        order_min_df = order_min_df.groupby(["Order ID"], as_index=True).min()

        order_df_min = order_min_df.rename({"Start Time": "Start Time min"}, axis='columns')
        relevant_start_times_df = relevant_start_times_df.merge(order_df_min, on="Order ID")

        relevant_start_times_df["new Time"] = (relevant_start_times_df["Start Time"] -
                                               relevant_start_times_df["Start Time min"])
        return relevant_start_times_df

    def get_resource_sequence(self, digital_twin_df):
        """
        Sort the resource ids to ensure that the resources are represented in a predefined sequence
        for a better readability
        """

        resource_available_mask = digital_twin_df["Resource Used ID"].notnull()
        relevant_columns = ["Resource Used ID", "Process Execution ID", "Resource Type"]
        resource_df = digital_twin_df.loc[resource_available_mask, relevant_columns]

        resource_mask = digital_twin_df["Process Execution ID"].isin(
            resource_df["Process Execution ID"])
        relevant_columns_process_executions = ["Process Execution ID", "Start Time"]
        relevant_start_times_new = digital_twin_df.loc[
            resource_mask, relevant_columns_process_executions].dropna(subset=["Start Time"])

        resource_df = resource_df.merge(relevant_start_times_new, on="Process Execution ID")
        resource_df = resource_df[["Resource Used ID", "Process Execution ID", "Start Time", "Resource Type"]]

        resource_id_start_time_mean_new = resource_df.groupby(["Resource Used ID"])["Start Time"].mean()
        resource_id_start_time_normalized = resource_id_start_time_mean_new - resource_id_start_time_mean_new.min()
        resource_merged_df = resource_df[["Resource Used ID", "Resource Type"]]
        resource_merged_df = resource_merged_df.drop_duplicates().set_index("Resource Used ID")
        resource_merged_df["Sort Time"] = resource_id_start_time_normalized

        resource_merged_df = resource_merged_df.sort_values(["Resource Type", "Sort Time"])

        return resource_merged_df.index.to_list()


class KPIAdministration:
    """
    The kpi administration provides some basic functions to the analytics,
    e.g., the determination of the consideration time period, the display sequence, ...
    """

    view_id_match = {"ORDER": "Order ID",
                     "PRODUCT": "Entity Type ID",
                     "PROCESS": "Process ID",
                     "RESOURCE": "Resource Used ID"}
    view_reference_match = {"ORDER": "Order Identifier",
                            "PRODUCT": "Entity Type Name",
                            "PROCESS": "Process Name",
                            "RESOURCE": "Resource Used Name"}

    start_time_observation_period = None
    end_time_observation_period = None

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        self.analytics_data_base: ScenarioAnalyticsDataBase = analytics_data_base
        self.process_sequence = None
        self.resource_sequence = None

        self.time_period = {}

    def get_filter_options(self, start_date: float, end_date: float):

        """
        Used to filter all orders, products/ parts, processes and resources with id and name which are
        in the digital twin listed linked with a process_execution.

        Parameters
        ----------
        start_date: start datetime
        end_date: end datetime

        Returns
        -------
        filter_options as dict
        """

        if int(start_date) == 0 and int(end_date) == 0 or self.analytics_data_base.digital_twin_df.empty:
            # avoid to get a problem with two get request at the same time
            filter_options = {"order": pd.Series([]),
                              "product": pd.Series([]),
                              "process": pd.Series([]),
                              "resource": pd.Series([])}
            return filter_options

        # find the respective ids
        order_mask = self.analytics_data_base.digital_twin_df["Order ID"].notnull()
        order_ids_list = list(set(self.analytics_data_base.digital_twin_df.loc[order_mask, "Order ID"]))

        part_mask = self.analytics_data_base.digital_twin_df["Entity Type ID"].notnull()
        part_ids_list = list(set(self.analytics_data_base.digital_twin_df.loc[part_mask, "Entity Type ID"]))

        display_sequences = DisplaySequences()

        self.process_sequence = display_sequences.get_process_sequence(self.analytics_data_base.digital_twin_df)

        process_mask = self.analytics_data_base.digital_twin_df["Process ID"].notnull()
        process_ids_set = set(self.analytics_data_base.digital_twin_df.loc[process_mask, "Process ID"])
        process_ids_list = [process_id
                            for process_id in self.process_sequence
                            if process_id in process_ids_set]

        self.resource_sequence = display_sequences.get_resource_sequence(self.analytics_data_base.digital_twin_df)
        resource_mask = self.analytics_data_base.digital_twin_df["Resource Used ID"].notnull()
        resource_ids_set = set(self.analytics_data_base.digital_twin_df.loc[resource_mask, "Resource Used ID"])
        resource_ids_list = [resource_id
                             for resource_id in self.resource_sequence
                             if resource_id in resource_ids_set]

        # matching id with name
        order_s = self.get_reference_value_by_id(order_ids_list, "ORDER")
        product_s = self.get_reference_value_by_id(part_ids_list, "PRODUCT")
        process_s = self.get_reference_value_by_id(process_ids_list, "PROCESS")
        resource_s = self.get_reference_value_by_id(resource_ids_list, "RESOURCE")

        filter_options = {"order": order_s,
                          "product": product_s,
                          "process": process_s,
                          "resource": resource_s}

        return filter_options

    def update_consideration_period(self, start_time, end_time):
        """
        Updates the consideration period.
        Required for the kpi calculation, e.g., for the availability calculation.
        Note The consideration period setting logic allows no consideration periods without data available.

        Parameters
        ----------
        start_time: start time stamp that restricts the consideration period.
        end_time: end time stamp that restricts the consideration period.
        """

        if not isinstance(start_time, datetime):
            start_time = datetime.fromtimestamp(start_time)

        if not isinstance(end_time, datetime):
            end_time = datetime.fromtimestamp(end_time)

        if self.analytics_data_base.digital_twin_df.shape[0] == 0:
            return datetime(1970, 1, 1), datetime(1970, 1, 1)

        min_ = self.analytics_data_base.digital_twin_df["Start Time"].dropna().min()
        max_end_time = self.analytics_data_base.digital_twin_df["End Time"].dropna().max()
        max_planned_delivery_time = self.analytics_data_base.digital_twin_df["Delivery Date Planned"].dropna().max()
        if max_end_time == max_end_time and max_planned_delivery_time == max_planned_delivery_time:
            max_ = max(max_end_time, max_planned_delivery_time)
        elif max_end_time == max_end_time:
            max_ = max_end_time
        elif max_planned_delivery_time == max_planned_delivery_time:
            max_ = max_planned_delivery_time
        else:
            max_ = min_

        if start_time < end_time:
            if max_ > start_time >= min_:
                type(self).start_time_observation_period = start_time
            else:
                type(self).start_time_observation_period = min_ - timedelta(seconds=1)
            if max_ >= end_time > type(self).start_time_observation_period:
                type(self).end_time_observation_period = end_time
            else:
                type(self).end_time_observation_period = max_ + timedelta(seconds=1)

        new_start_time, new_end_time = type(self).start_time_observation_period, type(self).end_time_observation_period

        self.time_period[(start_time, end_time)] = (new_start_time, new_end_time)

        return new_start_time, new_end_time

    def _get_observation_period(self):
        observation_period = biased((type(self).end_time_observation_period -
                                     type(self).start_time_observation_period).total_seconds())
        return observation_period

    def filter_view_ids(self, order_ids_list, product_ids_list, process_ids_list, resource_ids_list):
        """
        Searches the IDs from the lists in the data frame and stores them with the locations in a DF

        Parameters
        ----------
        order_ids_list: The Order IDs selected in the filter
        product_ids_list: The Part IDs selected in the filter
        process_ids_list: The Process IDs selected in the filter
        resource_ids_list: The Resource IDs selected in the filter

        Returns
        -------
        A series for each parameter, in which all IDs are entered and where they can be found
        in the digital twin data frame.
        """

        order_column_name = type(self).view_id_match["ORDER"]
        order_mask = self.analytics_data_base.digital_twin_df[order_column_name].isin(order_ids_list)
        order_ids_s = self.analytics_data_base.digital_twin_df.loc[order_mask, order_column_name]

        product_column_name = type(self).view_id_match["PRODUCT"]
        product_mask = self.analytics_data_base.digital_twin_df[product_column_name].isin(product_ids_list)
        product_ids_s = self.analytics_data_base.digital_twin_df.loc[product_mask, product_column_name]

        process_column_name = type(self).view_id_match["PROCESS"]
        process_mask = self.analytics_data_base.digital_twin_df[process_column_name].isin(process_ids_list)
        process_ids_s = self.analytics_data_base.digital_twin_df.loc[process_mask, process_column_name]

        resource_column_name = type(self).view_id_match["RESOURCE"]
        resource_mask = self.analytics_data_base.digital_twin_df[resource_column_name].isin(resource_ids_list)
        resource_ids_s = self.analytics_data_base.digital_twin_df.loc[resource_mask, resource_column_name]

        order_ids_list = list(set(order_ids_s.to_list()))
        product_ids_list = list(set(product_ids_s.to_list()))
        process_ids_list = list(set(process_ids_s.to_list()))
        resource_ids_list = list(set(resource_ids_s.to_list()))

        return order_ids_list, product_ids_list, process_ids_list, resource_ids_list

    def get_reference_value_by_id(self, ids_list, view) -> pd.Series():
        """
        Used to get the reference values based on their ids.

        Parameters
        ----------
        ids_list: ids of order, product, process, resource
        view: view can be "ORDER", "PRODUCT", "PROCESS", "RESOURCE"

        Returns
        -------
        reference_value list
        """

        view_id = type(self).view_id_match[view]
        view_reference = type(self).view_reference_match[view]

        # avoid a failure because the state model is not initialized
        if view_id not in self.analytics_data_base.digital_twin_df.columns:
            self.analytics_data_base.update_data()
        if view_reference not in self.analytics_data_base.digital_twin_df.columns:
            self.analytics_data_base.update_data()

        view_mask = ((self.analytics_data_base.digital_twin_df[view_id].isin(ids_list)) &
                     (self.analytics_data_base.digital_twin_df[view_reference] ==
                      self.analytics_data_base.digital_twin_df[view_reference]))
        reference_value_df = self.analytics_data_base.digital_twin_df.loc[view_mask, [view_id, view_reference]]

        # sort the results according the input
        reference_value_df = reference_value_df.drop_duplicates()
        reference_value_df = reference_value_df.set_index(view_id)

        # needed for more than one model (here: it can be the case that not all ids are in the model)
        common_ids_set = reference_value_df.index.intersection(set(ids_list))
        reference_value_s: pd.Series() = reference_value_df.loc[common_ids_set]

        return reference_value_s[view_reference]

    def get_customer_name_by_id(self, id_list: list[int], view: str):
        """
        Used to get the customer names based on their ids.

        Parameters
        ----------
        id_list: ids of order, product, process, resource
        view: view can be "ORDER", "PRODUCT", "PROCESS", "RESOURCE"

        Returns
        -------
        customer_name list
        """

        if view != "ORDER":
            raise NotImplementedError

        view_id_column = type(self).view_id_match[view]
        customer_name_mask = self.analytics_data_base.digital_twin_df[view_id_column].isin(id_list)
        customer_names_df: pd.DataFrame = self.analytics_data_base.digital_twin_df.loc[
            customer_name_mask, ['Order ID', "Customer Name"]]

        # results the results according the input
        customer_names_df.dropna(inplace=True)  # ToDo: if an id is dropped that should be imputed with None ...
        customer_names_df = customer_names_df.drop_duplicates()
        customer_names_df.set_index("Order ID", inplace=True)

        # integrate the orders that have a customer name np.nan
        orders_not_considered = list(set(id_list).difference(set(customer_names_df.index)))
        if orders_not_considered:
            orders_not_considered_df = pd.DataFrame([np.nan
                                                     for order_id in orders_not_considered],
                                                    columns=["Customer Name"], index=orders_not_considered)
            customer_names_df = pd.concat([customer_names_df, orders_not_considered_df])

        return customer_names_df

    @memoize_get_target_quantity
    def get_order_target_quantity(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                  scenario, all=False):

        """
        Used to determine the target quantities of products (relative to the orders).

        Returns
        -------
        a dictionary {product_entity_type_id: target_quantity}
        """
        # only for product_level

        # determine relevant pes
        relevant_process_execution_ids = \
            self.get_relevant_process_execution_ids(start_time, end_time,
                                                    order_ids, part_ids, process_ids, resource_ids,
                                                    "PLAN", scenario, from_actual=False)

        pe_order_df = self.analytics_data_base.process_execution_order_df
        possible_order_mask = pe_order_df["Process Execution ID"].isin(relevant_process_execution_ids)
        possible_order_ids = set(pe_order_df.loc[possible_order_mask, "Order ID"].dropna())

        product_target_quantity = self._get_product_target_quantity(start_time, end_time, possible_order_ids,
                                                                    part_ids)

        if all:
            product_target_quantity["All"] = product_target_quantity.sum()

        return product_target_quantity

    def _get_product_target_quantity(self, start_time, end_time, possible_order_ids, part_ids):
        """
        Quantity that should be produced in the consideration period/
        Has the delivery date planned between start_time and end_time

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        possible_order_ids: Previously calculated possible orders
        part_ids: Part IDs passed from frontend

        Returns
        -------
        Calculates the number of products produced
        """

        relevant_columns = ["Order ID", "Product Entity Type ID", "Delivery Date Planned", "Delivery Date Actual"]
        df = self.analytics_data_base.digital_twin_df[relevant_columns].dropna(
            subset=["Order ID", "Delivery Date Planned"])

        relevant_orders_mask = (df["Order ID"].isin(possible_order_ids) &  # filtered
                                (start_time <= df["Delivery Date Planned"]) &  # finished
                                (df["Delivery Date Planned"] <= end_time))
        finished_df = df.loc[relevant_orders_mask]

        product_entity_type_quantity = finished_df.groupby(by="Product Entity Type ID").count()
        product_entity_type_quantity = product_entity_type_quantity.rename({"Order ID": "target_quantity"},
                                                                           axis='columns')

        zero_df = pd.Series(np.zeros(len(part_ids)), part_ids,
                            name="target_quantity")
        product_target_quantity = (zero_df + product_entity_type_quantity["target_quantity"]).fillna(0)

        return product_target_quantity

    def get_difference_percentage(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                  event_type, view, scenario, all=False):
        """
        Used to get the difference between the target quantities of a product and their real production quantity
        :return: a dictionary {product_entity_type_id: percentage}
        """
        # existing quantities/ target_quantities

        real_production_quantities_sel = self.get_amount_objects(start_time, end_time, order_ids, part_ids,
                                                                 process_ids, resource_ids,
                                                                 event_type, view, True, scenario, all=all)
        target_production_quantities = self.get_order_target_quantity(start_time, end_time,
                                                                      order_ids, part_ids, process_ids,
                                                                      resource_ids, scenario, all=all)

        if not all:
            real_production_qty_s = pd.Series(np.zeros(len(part_ids)), index=part_ids)
            real_production_qty_s.update(real_production_quantities_sel)
            target_production_qty_s = pd.Series(np.zeros(len(part_ids)), index=part_ids)
            target_production_qty_s.update(target_production_quantities)
            qty_df = pd.DataFrame({"real": real_production_qty_s,
                                   "target": target_production_qty_s})

            qty_df = qty_df.fillna(0)
            qty_df["percentage"] = (((qty_df["real"] + 1e-12) / (qty_df["target"] + 1e-12)) - 1) * 100
            qty_df.loc[(qty_df["real"] != 0) & (qty_df["target"] <= 0), 'percentage'] = 100
            difference_percentage = qty_df['percentage'].to_dict()

        else:
            if not real_production_quantities_sel.empty and not target_production_quantities.empty:
                if (target_production_quantities["All"]) > 0 or real_production_quantities_sel[0] == 0:
                    difference_percentage = {"All": (biased(real_production_quantities_sel[0])
                                                     / biased(target_production_quantities["All"]) - 1) * 100}
                else:
                    difference_percentage = {"All": 100}

            else:
                difference_percentage = {"All": -100}

        return difference_percentage

    @memoize_get_amount_objects
    def get_amount_objects(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids, event_type,
                           view, selection, scenario, all=False):

        """
        Used to get the amount of actual objects per object id.
        Approach: count the finished orders

        Parameters
        ----------
        order_ids: id list
        part_ids: id list
        process_ids: id list
        resource_ids: id list
        event_type: PLAN or ACTUAL
        view: view can be "ORDER", "PRODUCT"
        selection: bool value - if True: only the selected objects/ their pes are considered
        scenario:
        all: True or False: True if the result is for the headline of the collected view. For the single view False

        Returns
        -------
        an amount list of objects (order, products) regarding the specific ids
        """

        if view != "ORDER" and view != "PRODUCT":
            raise NotImplementedError

        # search for the concerned pes in each view
        # find the pes which exists in all views
        relevant_process_executions = \
            self.get_relevant_process_execution_ids(start_time, end_time,
                                                    order_ids, part_ids, process_ids, resource_ids,
                                                    event_type, scenario)

        id_list_mapper = {"ORDER": order_ids,
                          "PRODUCT": part_ids,
                          "PROCESS": process_ids,
                          "RESOURCE": resource_ids}
        id_list = id_list_mapper[view]

        column_aggregated_mapper = {"ORDER": "Order ID",
                                    "PRODUCT": "Entity Type ID"}

        if view == "ORDER":
            counter_columns = ['Order ID', "Entity Type ID"]
        elif view == "PRODUCT":
            counter_columns = ['Entity Type ID', 'Part Involved ID']

        if not selection:
            amount_df = self.analytics_data_base.digital_twin_df[counter_columns]

        else:
            process_execution_mask = (
                self.analytics_data_base.digital_twin_df["Process Execution ID"].isin(
                    relevant_process_executions))
            amount_df = self.analytics_data_base.digital_twin_df.loc[process_execution_mask, counter_columns]

        if view == "ORDER":
            order_mask = self.analytics_data_base.digital_twin_df["Order ID"].isin(amount_df["Order ID"])
            amount_df = self.analytics_data_base.digital_twin_df.loc[order_mask, counter_columns]
        elif view == "PRODUCT":
            order_mask = self.analytics_data_base.digital_twin_df["Entity Type ID"].isin(amount_df["Entity Type ID"])
            amount_df = self.analytics_data_base.digital_twin_df.loc[order_mask, counter_columns]

        amount_df = amount_df.dropna()
        amount_df = amount_df.drop_duplicates()

        if view not in column_aggregated_mapper:
            raise NotImplementedError
        amount_s = amount_df[column_aggregated_mapper[view]].value_counts()

        if view == "PRODUCT":
            if no_part_id in amount_s:
                amount_s[no_part_id] = 0

        if all:
            amount_s = pd.Series({0: amount_s.sum()})
        else:
            amount_s_ground = pd.Series(np.zeros(len(id_list)),
                                        index=id_list)
            amount_s_ground.update(amount_s)
            amount_s = amount_s_ground
        return amount_s

    def get_relative_objects(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids, event_type,
                             view, scenario, all=False):
        """
        Used to get the relative part of all objects. (e.g. 10 objects are in the system, but because of the filtering
        only 5 are taken into account)

        Parameters
        ----------
        order_ids: list with order ids
        part_ids: list with part ids
        process_ids: list with process ids
        resource_ids: list with resource ids
        event_type: ACTUAL or PLAN
        view: view can be "ORDER", "PRODUCT"

        Returns
        -------
        a Data Frame with shares (percentage) of considered objects behind the specific ids
        """

        selection = self.get_amount_objects(start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                            event_type, view, True, scenario, all)
        whole = self.get_amount_objects(start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                        event_type, view, False, scenario, all)

        amount_objects_df = pd.concat([selection, whole], axis=1)
        amount_objects_df = amount_objects_df.replace(0, 1e-12)
        amount_objects_df.columns = ["selection", "whole"]
        amount_objects_df['relative'] = (amount_objects_df["selection"] / amount_objects_df["whole"]) * 100

        if all:
            amount_objects_df = amount_objects_df.fillna((0))
            relative_objects_s = pd.Series({0: amount_objects_df['relative'].values[0]})
        else:
            relative_objects_s = amount_objects_df['relative']

        return relative_objects_s

    @memoize_get_amount_pe
    def get_amount_pe(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids, event_type, view,
                      selection, scenario, all=False):
        """
        Used to get the amount of actual pe's per object id.

        Parameters
        ----------
        order_ids: list with order ids
        part_ids: list with part ids
        process_ids: list with process ids
        resource_ids: list with resource ids
        event_type: ACTUAL or PLAN
        view: view can be "PROCESS", "RESOURCE"
        selection: if the selection parameter is True, only the filter selection is considered

        Returns
        -------
        an amount dict of pe's regarding the specific ids
        """

        if not selection:
            order_ids_lst = self.analytics_data_base.process_execution_order_df["Order ID"].dropna().to_list()
            order_ids = list(set(order_ids_lst + order_ids))
            part_ids_lst = self.analytics_data_base.process_execution_part_df["Entity Type ID"].dropna().to_list()
            part_ids = list(set(part_ids_lst + part_ids))
            process_ids_lst = self.analytics_data_base.process_execution_df["Process ID"].dropna().to_list()
            process_ids = list(set(process_ids_lst + process_ids))
            resource_ids_lst = (
                self.analytics_data_base.process_execution_resource_df["Resource Used ID"].dropna().to_list())
            resource_ids = list(set(resource_ids_lst + resource_ids))

        # search for the concerned pes in each view
        # find the pes which exists in all views
        relevant_process_execution_ids = \
            self.get_relevant_process_execution_ids(start_time, end_time,
                                                    order_ids, part_ids, process_ids, resource_ids,
                                                    event_type, scenario)
        amount_s = pd.Series({0: np.nan})
        if view == "PROCESS":
            id_list = process_ids
            column = "Process ID"
        elif view == "RESOURCE":
            id_list = resource_ids
            column = "Resource Used ID"
        else:
            return amount_s


        if not (view == "PROCESS" or view == "RESOURCE"):
            return amount_s

        # mask = self.analytics_data_base.digital_twin_df[column].isin(id_list)
        object_dfs = self.analytics_data_base.digital_twin_df[['Process Execution ID', column]]
        # object_dfs = object_dfs.loc[mask]

        object_dfs = object_dfs.loc[object_dfs["Process Execution ID"].isin(relevant_process_execution_ids)]
        amount_s = object_dfs[column].value_counts()

        if all:
            amount = amount_s.loc[amount_s.index.isin(id_list)]
            amount_s = pd.Series({0: amount.sum()})

        else:
            missing_ids = set(id_list).difference(amount_s.index)
            if missing_ids:
                missing_amount_s = pd.Series({missing_id: 0
                                              for missing_id in list(missing_ids)})
                amount_s = pd.concat([amount_s, missing_amount_s])

        return amount_s

    @memoize_get_relevant_process_execution_ids
    def get_relevant_process_execution_ids(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                           event_type, scenario, from_actual=True):
        """
        Return the process executions that are relevant to the filter.


        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        event_type: Plan or Actual
        scenario: used for the scenario specific cache
        from_actual: True or False, True=Plan False=Actual
        """

        # Determine process executions that are relevant given the id restrictions and the time
        common_pe_ids = self._get_common_pe_ids(start_time, end_time,
                                                order_ids, part_ids, process_ids, resource_ids)
        # now plan and actual process executions are available

        mask = self.analytics_data_base.process_execution_df['Process Execution ID'].isin(common_pe_ids)
        pe_df = self.analytics_data_base.process_execution_df.loc[mask]

        # check the event_type/ filter according to the event type
        if not event_type:
            raise NotImplementedError

        if event_type == "ACTUAL" or not from_actual:
            df_with_event_type = pe_df.loc[pe_df['Event Type'] == event_type]
        elif event_type == "PLAN" and from_actual:
            df_with_event_type_actual = pe_df.loc[pe_df['Event Type'] == "ACTUAL"]
            if not df_with_event_type_actual.empty:
                df_with_event_type = pe_df.loc[pe_df["Process Execution ID"].isin(
                    df_with_event_type_actual["Connected Process Execution ID"])]  # relevant process_execution_ids
            else:
                df_with_event_type = pe_df
        else:
            raise NotImplementedError

        pe_s = df_with_event_type["Process Execution ID"]

        return pe_s

    @memoize_get_common_pe_ids
    def _get_common_pe_ids(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids):
        """
        Gets the PE ids from the ids from the filter

        Parameters
        ----------
        start_time: start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter

        Returns
        -------
        minimal set of process execution ids associated with the filter
        """

        order_mask = self.analytics_data_base.process_execution_order_df[
            type(self).view_id_match["ORDER"]].isin(order_ids)
        order_pe_ids = self.analytics_data_base.process_execution_order_df.loc[order_mask, "Process Execution ID"]

        part_mask = self.analytics_data_base.process_execution_part_df[
            type(self).view_id_match["PRODUCT"]].isin(part_ids)
        part_pe_ids = self.analytics_data_base.process_execution_part_df.loc[part_mask, "Process Execution ID"]

        process_mask = ((self.analytics_data_base.process_execution_df[
                             type(self).view_id_match["PROCESS"]].isin(process_ids)) &
                        (self.analytics_data_base.process_execution_df["Start Time"] >= start_time) &
                        (self.analytics_data_base.process_execution_df["End Time"] <= end_time))
        process_pe_ids = self.analytics_data_base.process_execution_df.loc[process_mask, "Process Execution ID"]

        resource_mask = self.analytics_data_base.process_execution_resource_df[
            type(self).view_id_match["RESOURCE"]].isin(resource_ids)
        resource_pe_ids = self.analytics_data_base.process_execution_resource_df.loc[
            resource_mask, "Process Execution ID"]

        common_pe_ids = list(set(order_pe_ids) & set(part_pe_ids) & set(process_pe_ids) & set(resource_pe_ids))

        return common_pe_ids

    def get_relative_pe(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids, event_type,
                        view, scenario, all=False):
        """
        Used to get the relative part of all pes. (e.g., 10 pes are in the system, but because of the filtering
        only 5 are taken into account)

        Parameters
        ----------
        order_ids: list with order ids
        part_ids: list with part ids
        process_ids: list with process ids
        resource_ids: list with resource ids
        view: view can be "PROCESS", "RESOURCE"

        Returns
        -------
        a list with shares (percentage) of considered pes behind the specific ids
        """
        # get filtered and unfiltered amounts
        selection = self.get_amount_pe(start_time=start_time, end_time=end_time,
                                       order_ids=order_ids, part_ids=part_ids,
                                       process_ids=process_ids, resource_ids=resource_ids,
                                       event_type=event_type, view=view, scenario=scenario, selection=True, all=all)
        whole = self.get_amount_pe(start_time=start_time, end_time=end_time,
                                   order_ids=order_ids, part_ids=part_ids,
                                   process_ids=process_ids, resource_ids=resource_ids,
                                   event_type=event_type, view=view, scenario=scenario, selection=False, all=all)

        share_df = pd.DataFrame({"selection": selection, "whole": whole, "relative": np.zeros(whole.shape[0])})
        share_df[["selection", "relative"]] = share_df[["selection", "relative"]].replace(0, 1e-12)
        share_df['relative'] = (share_df["selection"] / share_df["whole"]) * 100
        relative_s = share_df['relative']

        return relative_s

    @memoize_get_orders_finished
    def get_orders_finished(self, start_time, end_time, possible_order_ids, scenario):
        """
        Determine the orders that are finished in the consideration period.

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        possible_order_ids: calculated possible Customer ID
        scenario:

        Returns
        -------
        A Df in which the finished order Id with the produced product are indicated
        """

        relevant_columns = ["Order ID", "Release Date", "Delivery Date Planned", "Delivery Date Actual",
                            'Product Entity Type ID']
        df = self.analytics_data_base.order_df[relevant_columns].dropna(subset=["Order ID", "Delivery Date Actual"])

        possible_orders_mask = df["Order ID"].isin(possible_order_ids)
        time_frames_mask = (start_time <= df["Release Date"]) & (df["Release Date"] <= end_time) | \
                           (start_time <= df["Delivery Date Planned"]) & (df["Delivery Date Planned"] <= end_time) | \
                           (start_time <= df["Delivery Date Actual"]) & (df["Delivery Date Actual"] <= end_time)

        order_finished_mask = possible_orders_mask & time_frames_mask
        finished_order_ids = df.loc[order_finished_mask, ["Order ID", 'Product Entity Type ID']]

        return finished_order_ids

    def get_orders_in_progress(self, start_time, end_time, possible_order_ids, scenario):
        """Determine the orders that are in progress in the consideration period."""

        relevant_columns = ["Order ID", "Release Date", "Delivery Date Planned", "Delivery Date Actual",
                            'Product Entity Type ID']
        df = self.analytics_data_base.order_df[relevant_columns].dropna(subset=["Order ID", "Release Date"])
        possible_orders_mask = df["Order ID"].isin(possible_order_ids)

        # check if release_date met
        time_frames_mask = (df["Release Date"] <= end_time)

        released_finished_mask = possible_orders_mask & time_frames_mask
        df_released = df.loc[released_finished_mask]

        # check if not finished before
        open_orders_df_outside_period = df_released.dropna(subset=["Delivery Date Actual"])
        open_orders_df_outside_period = open_orders_df_outside_period.loc[
            open_orders_df_outside_period["Delivery Date Actual"] > end_time, ["Order ID", 'Product Entity Type ID']]
        open_orders_df = df_released.loc[
            df_released["Delivery Date Actual"] != df_released["Delivery Date Actual"],
            ["Order ID", 'Product Entity Type ID']]

        orders_in_progress_ids = pd.concat([open_orders_df_outside_period, open_orders_df])

        return orders_in_progress_ids

    def get_order_status(self, start_time, end_time, possible_order_ids, scenario):
        """
        Determine three states:
        1.) PLANNED
        2.) IN PROGRESS
        3.) FINISHED
        """

        finished_orders = self.get_orders_finished(start_time, end_time, possible_order_ids, scenario)["Order ID"]
        orders_in_progress = self.get_orders_in_progress(start_time, end_time, possible_order_ids, scenario)["Order ID"]
        order_progress_status = pd.Series(["PLANNED"
                                           for i in range(len(possible_order_ids))],
                                          index=possible_order_ids)
        order_progress_status.loc[orders_in_progress] = "IN PROGRESS"
        order_progress_status.loc[finished_orders] = "FINISHED"

        return order_progress_status
