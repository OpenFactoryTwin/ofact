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
import time
# Ignore Feature Warnings from pandas modul
import warnings
from datetime import datetime
from functools import wraps

warnings.simplefilter(action='ignore', category=FutureWarning)

# Imports Part 2: PIP Imports
import numpy as np
import pandas as pd

# Imports Part 3: Project Imports
from ofact.planning_services.scenario_analytics.data_basis import ScenarioAnalyticsDataBase
from ofact.planning_services.scenario_analytics.business_logic.kpi.table_basic import KPIAdministration, biased

# Module-Specific Constants
MEMOIZATION_MAX = 20000


def sort_df(df, order_ids, part_ids, process_ids, resource_ids, view, digital_twin_df):
    """
    Sorts the passed Df into the order of the passed parameter IDs

    Parameters
    ----------
    df: The result of a calculation, which is still unsorted
    order_ids: The Order IDs selected in the filter
    part_ids: The Part IDs selected in the filter
    process_ids: The Process IDs selected in the filter
    resource_ids: The Resource IDs selected in the filter
    view: The view displayed in the frontend
    digital_twin_df: The Digital Twin rendered in the DF

    Returns
    -------
    The sorted Df
    """

    if view == 'ORDER':
        sort_ids = order_ids
        sort_column = 'Order ID'
    elif view == 'PRODUCT':
        sort_ids = part_ids
        sort_column = 'Part Involved ID'
    elif view == 'PROCESS':
        sort_ids = process_ids
        sort_column = 'Process ID'
    elif view == 'RESOURCE':
        sort_ids = resource_ids
        sort_column = 'Resource Used ID'

    if len(df) == 1:
        return df
    if df.index.isin(digital_twin_df[sort_column]).all():
        df = df.loc[np.intersect1d(df.index, sort_ids)]

    return df


def memoize_get_pre_lead_time_dfs(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.

    Parameters
    ----------
    method: get_pre_lead_time_dfs

    Returns
    -------
    Result of Method
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


def memoize_prepared_dataframe(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.

    Parameters
    ----------
    method: get_prepared_dataframe

    Returns
    -------
    Result of Method
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


class KPICalc(KPIAdministration):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

        if not self.analytics_data_base.lead_time_dfs:
            self.analytics_data_base.lead_time_dfs = self.get_lead_time_dfs_updated()

    def get_lead_time_dfs_updated(self):
        return {"PLAN PROCESS": self.get_process_lead_time_process_chain_df("PLAN"),
                 "ACTUAL PROCESS": self.get_process_lead_time_process_chain_df("ACTUAL"),
                 "PLAN RESOURCE": self.get_process_lead_time_resource_df("PLAN"),
                 "ACTUAL RESOURCE": self.get_process_lead_time_resource_df("ACTUAL"),
                 "LAST UPDATE": datetime.now()}

    def get_process_lead_time_process_chain_df(self, event_type=None):
        """
        The preparation is needed for the calculation of the lead_times because they therefore the pes
        need to be ordered according time.

        Parameters
        ----------
        event_type: can be PLAN or ACTUAL

        Returns
        -------
        a prepared (sorted, ...) dataframe with lead_times
        """

        digital_twin_df = self.analytics_data_base.digital_twin_df.copy()
        if "Start Time" not in digital_twin_df.columns:
            return None

        digital_twin_df = digital_twin_df.loc[digital_twin_df["Event Type"] == event_type]
        digital_twin_df = self._add_lead_times(digital_twin_df)
        digital_twin_df = digital_twin_df[digital_twin_df["Start Time [s]"].notnull()]

        digital_twin_category_order_df, digital_twin_category_resource_df = self._split_value_streams(digital_twin_df)

        digital_twin_category_order_df = self._add_waiting_times_process_chain(digital_twin_category_order_df,
                                                                               relation_column="Order ID")
        digital_twin_category_resource_df = self._add_waiting_times_process_chain(digital_twin_category_resource_df,
                                                                                  relation_column="Main Resource ID")

        waiting_time_series = pd.concat([digital_twin_category_order_df,
                                         digital_twin_category_resource_df])
        digital_twin_df = digital_twin_df.merge(waiting_time_series, on='Process Execution ID')

        digital_twin_df.loc[digital_twin_df['Waiting Time [s]'] < 0, 'Waiting Time [s]'] = 0
        digital_twin_df['Lead Time with Waiting Time [s]'] = (digital_twin_df['Lead Time [s]'] +
                                                              digital_twin_df['Waiting Time [s]'])

        return digital_twin_df

    def get_process_lead_time_resource_df(self, event_type):

        digital_twin_df = self.analytics_data_base.digital_twin_df.copy()
        if "Start Time" not in digital_twin_df.columns:
            return None

        digital_twin_df = digital_twin_df.loc[digital_twin_df["Event Type"] == event_type]
        digital_twin_df = self._add_lead_times(digital_twin_df)
        digital_twin_df = digital_twin_df[digital_twin_df["Start Time [s]"].notnull()]

        waiting_time_series = self._add_waiting_times_process_chain(digital_twin_df, relation_column="Resource Used ID")

        digital_twin_df = digital_twin_df.merge(waiting_time_series, on='Process Execution ID')

        digital_twin_df.loc[digital_twin_df['Waiting Time [s]'] < 0, 'Waiting Time [s]'] = 0
        digital_twin_df['Lead Time with Waiting Time [s]'] = (digital_twin_df['Lead Time [s]'] +
                                                              digital_twin_df['Waiting Time [s]'])
        return digital_twin_df

    def _add_lead_times(self, digital_twin_df):
        """Add lead times to the digital twin df"""

        # timestamps
        to_time_stamp_func = lambda x: float(x.timestamp())
        start_time_mask = digital_twin_df["Start Time"].notnull()
        end_time_mask = digital_twin_df["End Time"].notnull()
        digital_twin_df["Start Time [s]"] = digital_twin_df.loc[start_time_mask, "Start Time"].apply(to_time_stamp_func)
        digital_twin_df["End Time [s]"] = digital_twin_df.loc[end_time_mask, "End Time"].apply(to_time_stamp_func)

        digital_twin_df.loc[:, "Lead Time [s]"] = (digital_twin_df.loc[end_time_mask, "End Time [s]"] -
                                                   digital_twin_df.loc[start_time_mask, "Start Time [s]"])

        return digital_twin_df

    def _split_value_streams(self, digital_twin_df):
        """
        Split different value streams of the digital twin df.
        Important since value streams can be parallel
        """

        allowed_order_process_types = ["Auftrag Prozesstyp", "order_et"]  # ToDo: general way needed ...
        allowed_resource_process_types = ["Ressource Prozesstyp", "resource_et"]
        digital_twin_category_order_df = (
            digital_twin_df.loc[digital_twin_df['Process Category'].isin(allowed_order_process_types)])
        digital_twin_category_resource_df = (
            digital_twin_df.loc[digital_twin_df['Process Category'].isin(allowed_resource_process_types)])

        return digital_twin_category_order_df, digital_twin_category_resource_df

    def _add_waiting_times_process_chain(self, digital_twin_part_df, relation_column):

        digital_twin_part_df = (
            digital_twin_part_df[['Process Execution ID', "Start Time [s]", "End Time [s]", relation_column]])

        digital_twin_part_df = digital_twin_part_df.sort_values(by=[relation_column, "Start Time [s]"])
        digital_twin_part_df = digital_twin_part_df.reset_index(drop=True)

        len_df = digital_twin_part_df.shape[0]
        waiting_time_part_s = (digital_twin_part_df[1:len_df]["Start Time [s]"].reset_index(drop=True) -
                               digital_twin_part_df[0:(len_df - 1)]["End Time [s]"].reset_index(drop=True))

        first_waiting_time = pd.Series([0])
        waiting_time_part_df = pd.concat([first_waiting_time, waiting_time_part_s])
        digital_twin_part_df['Waiting Time [s]'] = waiting_time_part_df[:].reset_index(drop=True)

        previous_order_id = pd.Series(0)
        previous_order_id = pd.concat([previous_order_id, digital_twin_part_df[relation_column][0:len_df - 1]])
        previous_order_id = previous_order_id.reset_index(drop=True)

        digital_twin_part_df['Calc Previous Relation ID'] = (previous_order_id -
                                                             digital_twin_part_df[relation_column])
        digital_twin_part_df['Waiting Time [s]'] = (
            digital_twin_part_df['Waiting Time [s]'].where(digital_twin_part_df['Calc Previous Relation ID'] == 0, 0))

        digital_twin_part_df = digital_twin_part_df[['Process Execution ID', 'Waiting Time [s]']]

        return digital_twin_part_df

    @memoize_get_pre_lead_time_dfs
    def get_pre_lead_time_dfs(self, start_time: datetime, end_time: datetime, scenario,
                              order_ids=None, part_ids=None, process_ids=None, resource_ids=None,
                              event_type=None, view=None) -> pd.DataFrame():
        """
        Filter the process_executions_df according the filter parameter also given as input.

        Parameters
        ----------
        start_time: start time stamp - defines the observation period
        # ToDo: (later) include the observation period
        end_time: end datetime timestamp
        order_ids: id list
        part_ids: id list
        process_ids: id list
        resource_ids: id list
        event_type: PLAN or ACTUAL
        view: ORDER, PRODUCT, PROCESS, RESOURCE

        Returns
        -------
        a df with the relevant data of the the respective filters
         (aggregated if necessary, sorted and lead_time calculated)
        """

        if resource_ids is None:
            resource_ids = []
        if process_ids is None:
            process_ids = []
        if part_ids is None:
            part_ids = []
        if order_ids is None:
            order_ids = []

        if self.analytics_data_base.last_update > self.analytics_data_base.lead_time_dfs["LAST UPDATE"]:
            self.analytics_data_base.lead_time_dfs = self.get_lead_time_dfs_updated()

        lead_time_view_mapper = {"ORDER": "PROCESS",
                                 "PRODUCT": "PROCESS",
                                 "PROCESS": "PROCESS",
                                 "RESOURCE": "RESOURCE"}
        lead_time_df_choice = event_type + " " + lead_time_view_mapper[view]
        if self.analytics_data_base.lead_time_dfs[lead_time_df_choice] is None:
            self.analytics_data_base.lead_time_dfs = self.get_lead_time_dfs_updated()

        lead_time_df = self.analytics_data_base.lead_time_dfs[lead_time_df_choice].copy()

        # search for the concerned pes in each view
        # find the process_executions which exists in all views
        relevant_process_executions = \
            self.get_relevant_process_execution_ids(start_time, end_time,
                                                    order_ids, part_ids, process_ids, resource_ids,
                                                    event_type, scenario)

        relevant_process_executions_mask = lead_time_df["Process Execution ID"].isin(relevant_process_executions)
        relevant_process_executions_df = lead_time_df.loc[relevant_process_executions_mask]

        if view == "PRODUCT":
            relevant_process_executions_df = self.aggregation(relevant_process_executions_df, part_ids, view)

        if relevant_process_executions_df.empty:
            relevant_process_executions_df = pd.DataFrame(columns=self.analytics_data_base.digital_twin_df.columns)

        return relevant_process_executions_df

    def aggregation(self, relevant_process_executions_df, product_ids, view):
        """
        Used to aggregate all process executions according to an object
        (e.g., all process executions associated to a product type)

        Parameters
        ----------
        relevant_process_executions_df: df of all relevant process_executions_components about an id
        product_ids: id list
        view: "ORDER", "PRODUCT", "PROCESS", "RESOURCE"

        Returns
        -------
        a dataframe with aggregated objects
        """

        if view != "PRODUCT":  # only for the product view currently an aggregation is executed
            # df not touched
            return relevant_process_executions_df

        ids = product_ids
        column_aggregated = "Entity Type ID"
        column_single = "Part Involved ID"

        df = self.analytics_data_base.digital_twin_df
        allowed_order_process_types = ["Auftrag Prozesstyp", "order_et"]

        # filter the main value stream
        order_process_category_mask = (
            relevant_process_executions_df['Process Category'].isin(allowed_order_process_types))
        relevant_process_executions_df = relevant_process_executions_df.loc[order_process_category_mask]

        # ToDo: (later) other point for the query of finished object
        single_ids_mask = df[column_single].isin(df.loc[df[column_aggregated].isin(ids), column_single])
        pe_ids = df.loc[single_ids_mask, ["Process Execution ID", column_single]]

        single_object_df = relevant_process_executions_df.loc[
            relevant_process_executions_df["Process Execution ID"].isin(pe_ids['Process Execution ID'])]
        single_object_df = single_object_df.drop(column_single, axis=1)
        single_object_df = single_object_df.merge(pe_ids, on='Process Execution ID')
        single_object_df = single_object_df.drop_duplicates()

        # aggregation based on the single object (e.g., part/ product)
        sum_df = single_object_df.groupby(column_single)[['Lead Time [s]', 'Waiting Time [s]',
                                                          'Lead Time with Waiting Time [s]']].agg('sum')

        mean_df = single_object_df[['Resulting Quality', column_single]].groupby(column_single).agg('mean')
        if mean_df.empty:
            mean_df = pd.DataFrame(columns=['Resulting Quality'])

        min_max_df = single_object_df.groupby(column_single)[['Start Time [s]', 'End Time [s]']].agg(['min', 'max'])
        event_type = single_object_df[['Event Type', column_single]]
        event_type = event_type.drop_duplicates()
        event_type = event_type.set_index(event_type[column_single])

        values = [event_type['Event Type'],
                  min_max_df['Start Time [s]', 'min'], min_max_df['End Time [s]', 'max'],
                  mean_df['Resulting Quality'],  # source['Source'],
                  sum_df['Lead Time [s]'], sum_df['Waiting Time [s]'],
                  sum_df['Lead Time with Waiting Time [s]']]

        aggregated_df = pd.DataFrame(values,
                                     index=['Event Type', 'Start Time [s]', 'End Time [s]',
                                            'Resulting Quality', 'Lead Time [s]', 'Waiting Time [s]',  # 'Source',
                                            'Lead Time with Waiting Time [s]']).T

        # sort
        if aggregated_df.shape[0] > 0:
            aggregated_df = aggregated_df.sort_values(by="Start Time [s]")

        return aggregated_df


def memoize_lead_time_dfs(method):
    """
    Saves the result of the method with the passing parameters. If this method is called again with the same parameters,
    the result is not recalculated, but passed from this method.

    Parameters
    ----------
    method: get_lead_time_dfs

    Returns
    -------
    Result of Method
    """
    cache = {}

    @wraps(method)
    def memoize(*args, **kwargs):
        start_time = time.process_time()
        if str(args) + str(kwargs) in cache:
            return cache[str(args) + str(kwargs)]

        result = method(*args, **kwargs)
        cache[str(args) + str(kwargs)] = result

        # pop an element if cache_max reached to prevent RAM overload
        if len(cache) > MEMOIZATION_MAX:
            del cache[list(cache.keys())[0]]

        return result

    return memoize


class LeadTimeBasedKPI(KPICalc):
    lead_time_df_columns = ["ID", "View", "Start Time [s]", "End Time [s]",
                            "avg_lead_time", "avg_waiting_time", "avg_lead_time_wt",
                            "total_lead_time", "total_waiting_time", "total_lead_time_wt",
                            "min_lead_time", "min_waiting_time", "min_lead_time_wt",
                            "max_lead_time", "max_waiting_time", "max_lead_time_wt",
                            "lead_time_var", "waiting_time_var", "var_lead_time_wt"]

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    @memoize_lead_time_dfs
    def get_lead_time_dfs(self, start_time: datetime, end_time: datetime, scenario,
                          order_ids=[], part_ids=[], process_ids=[], resource_ids=[], event_type=None, view=None,
                          all=False):
        """
        Create for each id of the view a key value pair with relevant process_executions_components and their lead_times
        respectively their waiting_times. (e.g. (1, "ORDER"): relevant_process_execution_df)

        Parameters
        ----------
        start_time: start_time as datetime timestamp
        end_time: end_time as datetime timestamp
        order_ids: id list
        part_ids: id list
        process_ids: id list
        resource_ids: id list
        event_type: can be "PLAN" or "ACTUAL"
        view: "ORDER", "PRODUCT", "PROCESS", "RESOURCE"
        """
        # order view: Average Lead_time per order + Waiting time (correct only if all processes, resources and parts are
        # chosen otherwise it includes times of non selected processes)
        # part view: Average lead_time per part
        # Process view: Average lead_time of Process, planned vs actual?
        # Resource View: Average lead_time in all process executions?
        # remove the "All ids"-id of the other view's

        # ToDo: check the polars df

        if view == "ORDER":
            search_string = 'Order ID'
            groupby_string = 'Order ID'
            relevant_ids = order_ids
        elif view == "PRODUCT":
            search_string = 'Entity Type ID'
            groupby_string = 'Part Involved ID'
            relevant_ids = part_ids
        elif view == "PROCESS":
            search_string = 'Process ID'
            groupby_string = 'Process ID'
            relevant_ids = process_ids
        elif view == "RESOURCE":
            search_string = 'Resource Used ID'
            groupby_string = 'Resource Used ID'
            relevant_ids = resource_ids

        relevant_df = self.get_pre_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids,
                                                 part_ids=part_ids, process_ids=process_ids, resource_ids=resource_ids,
                                                 event_type=event_type, view=view, scenario=scenario)

        if relevant_df.empty:
            lead_time_df = pd.DataFrame(np.zeros(19)[None, :],
                                        columns=type(self).lead_time_df_columns)
            return lead_time_df

        lead_time_df = self._calc_rows(relevant_ids, view, relevant_df, search_string, groupby_string)

        if not all:
            return lead_time_df

        lead_time_df["View"] = np.nan
        if view == 'ORDER':
            lead_time_df = lead_time_df.sum()
        elif view == 'PRODUCT':
            lead_time_df = lead_time_df.loc[lead_time_df['ID'].isin(
                self.analytics_data_base.digital_twin_df['Entity Type ID'])]
            lead_time_df = lead_time_df.mean()
        else:
            lead_time_df = lead_time_df.mean()
        lead_time_df = pd.DataFrame(lead_time_df)
        lead_time_df = lead_time_df.T
        lead_time_df['ID'] = 0
        lead_time_df['View'] = view

        return lead_time_df

    def _calc_rows(self, ids: list, view, dig_twin, search_string, groupby_string):
        """
        Calculates the start, end, avg, min, max, var of the order together. Then these are combined for each order
        and result in the order view.

        Parameters
        ----------
        ids: Set an Order ID
        view: View of the frontend
        dig_twin: a calculated df with relevant PE

        Returns
        -------
        start, end, avg, min, max, var Lead Time for every view
        """
        if not ids:
            return

        if ids != [0]:
            mask = self.analytics_data_base.digital_twin_df[search_string].isin(ids)
            if search_string == groupby_string:
                dig_twin_two = self.analytics_data_base.digital_twin_df.loc[
                    mask, ["Process Execution ID", groupby_string]]
            else:
                dig_twin_two = self.analytics_data_base.digital_twin_df.loc[mask, [search_string, groupby_string]]

            dig_twin_two = dig_twin_two.drop_duplicates()

            if view == "PROCESS":
                dig_twin = dig_twin.merge(dig_twin_two, on="Process Execution ID")
                dig_twin["Process ID"] = dig_twin["Process ID_y"]
                dig_twin = dig_twin.drop(columns=["Process ID_x", "Process ID_y"])
                dig_twin_two = dig_twin_two.drop(columns="Process Execution ID").dropna().drop_duplicates()

            elif view == "RESOURCE":
                dig_twin = dig_twin.merge(dig_twin_two, on="Process Execution ID")
                dig_twin["Resource Used ID"] = dig_twin["Resource Used ID_y"]
                dig_twin = dig_twin.drop(columns=["Resource Used ID_x", "Resource Used ID_y"])
                dig_twin_two = dig_twin_two.drop(columns="Process Execution ID").dropna().drop_duplicates()

            if view == 'ORDER' or view == 'PRODUCT':
                if not dig_twin_two.empty:
                    if "Process Execution ID" in dig_twin and "Process Execution ID" in dig_twin_two.columns:
                        order_group = dig_twin.merge(dig_twin_two, on=[groupby_string, "Process Execution ID"])
                    else:
                        order_group = dig_twin.merge(dig_twin_two, on=[groupby_string])
                else:
                    order_group = dig_twin

            else:
                order_group = dig_twin.merge(dig_twin_two, on=[groupby_string])

                # order_group['new time'] = order_group['End Time [s]'] - order_group['Start Time [s]']
                # umrechnen 86400

            lead_waiting_df = order_group.groupby(search_string)

            row = pd.DataFrame(index=lead_waiting_df.indices.keys())
            row["ID"] = lead_waiting_df.indices.keys()
            row.loc[:, "View"] = view

        else:
            lead_waiting_df = dig_twin
            row = pd.DataFrame(index=ids)
            row["ID"] = 0
            row["View"] = view

        row["Start Time [s]"] = lead_waiting_df['Start Time [s]'].min()
        row["End Time [s]"] = lead_waiting_df['End Time [s]'].max()
        row["avg_lead_time"] = lead_waiting_df['Lead Time [s]'].mean()
        row["avg_waiting_time"] = lead_waiting_df['Waiting Time [s]'].mean()
        row["avg_lead_time_wt"] = lead_waiting_df['Lead Time with Waiting Time [s]'].mean()
        row["total_lead_time"] = lead_waiting_df['Lead Time [s]'].sum()
        row["total_waiting_time"] = lead_waiting_df['Waiting Time [s]'].sum()
        row["total_lead_time_wt"] = lead_waiting_df['Lead Time with Waiting Time [s]'].sum()
        row["min_lead_time"] = lead_waiting_df['Lead Time [s]'].min()
        row["min_waiting_time"] = lead_waiting_df['Waiting Time [s]'].min()
        row["min_lead_time_wt"] = lead_waiting_df['Lead Time with Waiting Time [s]'].min()
        row["max_lead_time"] = lead_waiting_df['Lead Time [s]'].max()
        row["max_waiting_time"] = lead_waiting_df['Waiting Time [s]'].max()
        row["max_lead_time_wt"] = lead_waiting_df['Lead Time with Waiting Time [s]'].max()
        row["var_lead_time"] = lead_waiting_df['Lead Time [s]'].var()
        row["var_waiting_time"] = lead_waiting_df['Waiting Time [s]'].var()
        row["var_lead_time_wt"] = lead_waiting_df['Lead Time with Waiting Time [s]'].var()

        return row


class LeadTime(LeadTimeBasedKPI):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    def get_lead_time(self, start_time: datetime, end_time: datetime,
                      order_ids, part_ids, process_ids, resource_ids, event_type, view, scenario, all=False):
        """
        Create a list with a dict for each id of the view. In the dict are the AnalyticsDataBase's
        [lead_time, waiting_time] - total, average, min, max, variance

        Parameters
        ----------
        start_time: start_time as datetime timestamp
        end_time: end_time as datetime timestamp
        order_ids: id list
        part_ids: id list
        process_ids: id list
        resource_ids: id list
        event_type: can be "PLAN" or "ACTUAL"
        view: "ORDER", "PRODUCT", "PROCESS", "RESOURCE"

        Returns
        -------
        a list of dicts with kpis
        """
        lead_time_df = self.get_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids,
                                              part_ids=part_ids, process_ids=process_ids, resource_ids=resource_ids,
                                              event_type=event_type, view=view, scenario=scenario, all=all)

        return lead_time_df

    def get_start_end_time_order(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                 event_type, view, scenario, all=False):
        """
        get a df with start and endtime for every order

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        Used to get the start and end_time of aggregated order sets from an order

        Returns
        -------
        df with start and end time for every order
        """
        relevant_orders_mask = self.analytics_data_base.order_df["Order ID"].isin(order_ids)
        relevant_columns = ["Order ID", "Release Date", "Delivery Date Planned", "Delivery Date Actual"]
        relevant_orders_df = self.analytics_data_base.order_df.loc[relevant_orders_mask, relevant_columns]
        start_time_df = self.analytics_data_base.digital_twin_df.groupby("Order ID").agg({"Start Time": "min"})
        start_end_time_order_df = pd.merge(relevant_orders_df, start_time_df, how="left", on="Order ID")
        start_end_time_order_df.reset_index(inplace=True)

        columns = ["Release Date", "Start Time", "Delivery Date Actual"]
        for col in columns:
            mask = start_end_time_order_df.loc[start_end_time_order_df[col].notnull(), col] > end_time
            start_end_time_order_df.loc[np.where(mask == True)[0], col] = np.nan

        start_end_time_order_df.rename(columns={"Release Date": "Release Time [s]",
                                                "Start Time": "Start Time [s]",
                                                "Delivery Date Planned": "Planned End Time [s]",
                                                "Delivery Date Actual": "End Time [s]"},
                                       inplace=True)

        cols = ["Release Time [s]", "Start Time [s]", "Planned End Time [s]", "End Time [s]"]
        to_time_stamp_func = lambda x: float(x.timestamp())
        for col in cols:
            mask = start_end_time_order_df[col].notnull()
            start_end_time_order_df.loc[mask, col] = (
                start_end_time_order_df.loc[mask, col].apply(to_time_stamp_func))

        # planned_end_time
        start_end_time_order_df.set_index("Order ID", inplace=True)

        if all:
            release_time = start_end_time_order_df["Release Time [s]"].min()
            start_time = start_end_time_order_df["Start Time [s]"].min()
            end_time = start_end_time_order_df["End Time [s]"].max()
            planned_end_time = start_end_time_order_df["Planned End Time [s]"].max()
            start_end_time_order_df = pd.DataFrame({"Release Time [s]": release_time,
                                                    "Start Time [s]": start_time,
                                                    "End Time [s]": end_time,
                                                    "Planned End Time [s]": planned_end_time},
                                                   index=[0])

        return start_end_time_order_df


class Performance(LeadTimeBasedKPI):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    def get_performance(self, start_time: datetime, end_time: datetime,
                        order_ids, part_ids, process_ids, resource_ids, view, scenario, all=False):
        """
        Used to get the performance. Can be used for different views.

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        view: view passed from frontend
        scenario:

        Returns
        -------
        df of performance_percentages
        """

        lead_time_actual = self.get_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids,
                                                  part_ids=part_ids, process_ids=process_ids, resource_ids=resource_ids,
                                                  event_type="ACTUAL", view=view, scenario=scenario, all=all)
        lead_time_actual = lead_time_actual['avg_lead_time']
        lead_time_planned = self.get_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids,
                                                   part_ids=part_ids, process_ids=process_ids,
                                                   resource_ids=resource_ids,
                                                   event_type="PLAN", view=view, scenario=scenario, all=all)
        lead_time_planned = lead_time_planned['avg_lead_time']

        # mask the not calculated elements
        m_actual = lead_time_actual.isna().replace({True: False, False: True})
        m_plan = lead_time_planned.isna().replace({True: False, False: True})
        lead_time_planned = lead_time_planned.where(m_actual, float("NaN"))
        lead_time_actual = lead_time_actual.where(m_plan, float("NaN"))

        # ToDo: avoid that the performance is over 100 - factor 0.9
        lead_time_actual['performance'] = (biased(lead_time_planned * 0.9)) / (biased(lead_time_actual))
        lead_time_actual['performance'] = lead_time_actual['performance'].fillna(0)
        lead_time_actual['performance'] *= 100
        kpi_series = lead_time_actual['performance']
        kpi_series.name = "performance"
        # kpi_list_int = sort_df(kpi_list_int, order_ids, part_ids, process_ids, resource_ids, view,
        #                                self.kpi.digital_twin_df)
        return kpi_series


class Availability(LeadTimeBasedKPI):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    def get_utilisation(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                        event_type, view, scenario, all=False):
        """
        Used to calculate the utilisation (of a resource).  # ToDo: (later) filtering

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        event_type: Plan or Actual
        view: view passed from frontend
        scenario:
        all: True or False; True calculated for headline, False: calculate for every (.. view)

        Returns
        -------
        df with utilisation for (Order, part...)
        """

        kpis_utilization = self.calc_utilisation(start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                                 event_type, view, scenario, all)

        if not all:
            return kpis_utilization

        if not kpis_utilization.empty:
            utilisation = kpis_utilization.sum() / kpis_utilization.size
            kpis = pd.Series({0: utilisation})

        else:
            kpis = pd.Series()

        return kpis

    def calc_utilisation(self, start_time, end_time, order_ids, part_ids, process_ids,
                         resource_ids, event_type, view, scenario, all) -> pd.Series():

        """
        Calculate the utilisation for the df

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        event_type: Plan or Actual
        view: view passed from frontend
        scenario:
        all: True or False; True calculated for headline, False: calcualte for every (.. view)

        Returns
        -------
        a Series ((View(Order ID.....); Utilisation)
        """

        lead_time_df = \
            self.get_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids,
                                   part_ids=part_ids, process_ids=process_ids, resource_ids=resource_ids,
                                   event_type=event_type, view=view, scenario=scenario, all=all)

        # 1. calc consideration_period
        observation_period = self._get_observation_period()

        # calc the availabilities - total_lead_time for each resource / observation_period
        utilisation_df = pd.DataFrame(columns=["ID", "utilisation"])
        utilisation_df["ID"] = lead_time_df["ID"]
        utilisation_df["utilisation"] = (lead_time_df["total_lead_time"] / observation_period) * 100

        # convert the availability_time df to a list of dicts
        utilisation_df = sort_df(utilisation_df, order_ids, part_ids, process_ids, resource_ids, view,
                                 self.analytics_data_base.digital_twin_df)
        kpis = utilisation_df["utilisation"]

        return kpis

    def get_availability(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                         event_type, view, scenario, all=False):
        """
        used to get the availability for view

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        event_type: Plan or Actual
        view: view passed from frontend
        scenario:
        all: True or False; True calculated for headline, False: calculate for every (.. view)

        Returns
        -------
        a Series ((View(Order ID.....); availability)
        """

        kpi_values = self.calc_availability(start_time, end_time, order_ids, part_ids, process_ids,
                                            resource_ids, event_type, view, scenario, all=False)

        if all:
            kpi_list = pd.Series(kpi_values.mean())

        else:
            kpi_list = kpi_values

        return kpi_list

    def calc_availability(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                          event_type, view, scenario, all):
        """
        Calculates the availability

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        event_type: Plan or Actual
        view: view passed from frontend
        scenario:
        all: True or False; True calculated for headline, False: calcualte for every (.. view)

        Returns
        -------
        a Series ((View(Order ID.....); availability)
        """

        # self.kpi.
        df = self.analytics_data_base.digital_twin_df

        mask = (df['Resource Used ID'].isin(resource_ids)) & (df['Resource Type'] == "ActiveMovingResource")
        utilization_s = df.loc[~mask, "Resource Used ID"].dropna().drop_duplicates()
        availability_s = df.loc[mask, "Resource Used ID"].dropna().drop_duplicates()

        if not availability_s.empty:
            observation_period = self._get_observation_period()

            # How does it work?
            # 1. process_executions and resource ids -> process_executions by resources

            relevant_resource_mask = df["Resource Used ID"].isin(availability_s)
            relevant_process_executions_df = df.loc[relevant_resource_mask, ["Process Execution ID", 'Resource Used ID',
                                                                             'Order ID', 'Start Time', 'End Time']]
            resource_used_df = \
                relevant_process_executions_df.dropna(subset=["Resource Used ID"])[["Process Execution ID",
                                                                                    'Resource Used ID']]

            order_df = df.dropna(subset=["Order ID"])[["Process Execution ID", 'Order ID']]

            relevant_process_executions_s = \
                df["Process Execution ID"].isin(relevant_process_executions_df["Process Execution ID"])
            relevant_columns = ["Process Execution ID", 'Start Time', 'End Time']
            time_df = df.loc[relevant_process_executions_s, relevant_columns].dropna(subset=["Start Time", "End Time"])

            order_resource_df = resource_used_df.merge(order_df, on='Process Execution ID')
            order_resource_start_df = order_resource_df.merge(time_df, on='Process Execution ID')
            relevant_process_executions = order_resource_start_df.dropna()

            resource_order_group = relevant_process_executions.groupby(['Resource Used ID', 'Order ID'])

            order_start_times = resource_order_group["Start Time"].min()
            order_end_times = resource_order_group["End Time"].max()
            relevant_process_execution_group = pd.DataFrame({"Order Start Time": order_start_times,
                                                             "Order End Time": order_end_times})

            mask1 = relevant_process_execution_group > type(self).end_time_observation_period
            relevant_process_execution_group[mask1] = type(self).end_time_observation_period
            mask2 = relevant_process_execution_group < type(self).start_time_observation_period
            relevant_process_execution_group[mask2] = type(self).start_time_observation_period

            relevant_process_execution_group['order_time'] = relevant_process_execution_group["Order End Time"] - \
                                                             relevant_process_execution_group["Order Start Time"]

            relevant_process_execution_group['order_time'] = \
                relevant_process_execution_group['order_time'].dt.total_seconds()
            relevant_process_execution_group = \
                relevant_process_execution_group.groupby('Resource Used ID')["order_time"].sum()

            availability_final_s = (relevant_process_execution_group / observation_period) * 100
            availability_df = availability_final_s.to_frame()

        if not utilization_s.empty:
            resource_ids_utilization = utilization_s.to_list()
            avail_utilization_s = self.calc_utilisation(start_time, end_time, order_ids, part_ids, process_ids,
                                                        resource_ids_utilization, event_type, view, scenario, all)

            if avail_utilization_s.shape[0] < len(resource_ids_utilization):
                zero_df = pd.Series(np.zeros(len(resource_ids_utilization)), index=resource_ids_utilization)
                avail_utilization_s = zero_df + avail_utilization_s
                avail_utilization_s = avail_utilization_s.fillna(0)
            avail_utilization_df = pd.DataFrame({"order_time": avail_utilization_s})

        if availability_s.empty:
            availability = avail_utilization_df.loc[resource_ids, 'order_time']

        elif utilization_s.empty:
            availability = availability_df.loc[resource_ids, 'order_time']

        else:
            availability_df = pd.concat([avail_utilization_df, availability_df])
            availability = availability_df.loc[resource_ids, 'order_time']

        return availability


class Inventory(KPICalc):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    def get_inventory(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                      event_type, view, scenario, all=False) -> pd.Series():
        """
        The inventory can be used on all four views (order, product, process, resource).
        Views and their inventory calculation:
        - PRODUCT: based on the product entity types.
        ToDo: maybe also on Order level (with relation to the order products)
        - RESOURCE: based on the resources (entity type) associated with orders.

        Parameters
        ----------
        start_time: datetime
        end_time: datetime
        order_ids: list
        part_ids: list
        process_ids: list
        resource_ids: list
        event_type: str
        view: str
        scenario: str
        all: bool

        Returns
        -------
        pd.Series
        """

        if view == "ORDER" and view == "PROCESS":
            raise NotImplementedError("The order and process views are not implemented yet.")

        relevant_process_execution_ids = (
            self.get_relevant_process_execution_ids(start_time, end_time,
                                                    order_ids, part_ids, process_ids, resource_ids,
                                                    event_type, scenario))

        inventory_mask = self.analytics_data_base.inventory_df["Event Type"] == event_type
        inventory_mask_raw =  (inventory_mask |
                               (self.analytics_data_base.inventory_df["Event Type"] !=
                               self.analytics_data_base.inventory_df["Event Type"]))
        inventory_df = self.analytics_data_base.inventory_df.loc[inventory_mask]
        inventory_df_raw = self.analytics_data_base.inventory_df.loc[inventory_mask_raw]

        # present the state at the start time stamp
        initial_inventory = inventory_df_raw.loc[(inventory_df_raw["Timestamp"] != inventory_df_raw["Timestamp"]) |
                                                 (start_time > inventory_df_raw["Timestamp"])]

        # ToDo: consider in the initial inventory only the relevant entries (resource ids or product entity type ids)
        #  start with the resource view

        # represent the changes between the start time stamp and the end time stamp
        changes_inventory_df = (
            inventory_df.loc[(inventory_df["Process Execution ID"].isin(relevant_process_execution_ids)) &
                             (start_time <= inventory_df["Timestamp"]) &
                             (inventory_df["Timestamp"] <= end_time)])

        combined_df = pd.concat([initial_inventory, changes_inventory_df],
                                ignore_index=True)

        if not all:

            if view == "RESOURCE":
                inventory_s = pd.Series(np.zeros(len(resource_ids)), resource_ids)

                inventory_storage_level_s = (
                    combined_df[["Storage Resource ID", "Quantity"]].groupby("Storage Resource ID")["Quantity"].sum())

                inventory_superior_level_s = (
                    combined_df[["Superior Storage Resource ID", "Quantity"]].groupby(
                        "Superior Storage Resource ID")["Quantity"].sum())

                inventory_s.update(inventory_storage_level_s)
                inventory_s.update(inventory_superior_level_s)

            elif view == "PRODUCT":
                # determine relevant part ids
                inventory_s = pd.Series(np.zeros(len(part_ids)), part_ids)
                part_inventory_s = combined_df[["Part Type ID", "Quantity"]].groupby("Part Type ID")["Quantity"].sum()
                inventory_s.update(part_inventory_s)

            else:
                raise NotImplementedError("The view {} is not implemented yet.".format(view))

        else:
            inventory_all = combined_df["Quantity"].sum()
            inventory_s = pd.Series([inventory_all], index=["All"])

        return inventory_s


class Quality(KPICalc):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    # all levels
    def get_quality(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                    event_type, view, scenario, all=False):
        """
        Used to get quality

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        event_type: Plan or Actual
        view: view passed from frontend
        scenario:
        all: True or False; True calculated for headline, False: calculate for every (.. view)

        Returns
        -------
        a Series ((View(Order ID.....); quality)
        """

        if view:
            kpi_quality_s = \
                self._get_resulting_quality(start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                            event_type, view, scenario)

        else:
            return None

        if all:
            kpi_quality = kpi_quality_s['Resulting Quality'].mean()
            kpi_quality_s = pd.DataFrame({"Resulting Quality": [kpi_quality]})

        return kpi_quality_s

    def _get_resulting_quality(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids, event_type,
                               view, scenario):
        """
        Calculates the quality

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        event_type: Plan or Actual
        view: view passed from frontend
        scenario:

        Returns
        -------
        a Series ((View(Order ID.....); Quality)
        """
        relevant_process_executions_df = \
            self.get_pre_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids, part_ids=part_ids,
                                       process_ids=process_ids, resource_ids=resource_ids, event_type=event_type,
                                       view=view, scenario=scenario)

        if view == "ORDER":
            search_string = 'Order ID'
            groupby_string = 'Order ID'
            relevant_ids = order_ids
        elif view == "PRODUCT":
            search_string = 'Entity Type ID'
            groupby_string = 'Part Involved ID'
            relevant_ids = part_ids
        elif view == "PROCESS":
            search_string = 'Process ID'
            groupby_string = 'Process Execution ID'
            relevant_ids = process_ids
        elif view == "RESOURCE":
            search_string = 'Resource Used ID'
            groupby_string = 'Process Execution ID'
            relevant_ids = resource_ids
        else:
            raise Exception

        if relevant_process_executions_df.empty:
            return pd.DataFrame(np.zeros(2)[None, :], columns=['Resulting Quality', search_string])

        mask = self.analytics_data_base.digital_twin_df[search_string].isin(relevant_ids)
        if search_string != groupby_string:
            columns_list = [search_string, groupby_string]
        else:
            columns_list = [search_string]
        dig_twin_two = self.analytics_data_base.digital_twin_df.loc[mask, columns_list]
        dig_twin_two = dig_twin_two.drop_duplicates()

        if view == 'ORDER' or view == 'PRODUCT':
            if not dig_twin_two.empty:
                order_group = relevant_process_executions_df.merge(dig_twin_two, on=groupby_string)
                if order_group.empty:
                    return pd.Series(np.zeros(len(relevant_ids)), index=relevant_ids)
                result_quality = order_group[['Resulting Quality', search_string]].groupby(search_string).agg('mean')

            else:
                return pd.DataFrame(np.zeros(2)[None, :], columns=['Resulting Quality', search_string])

        else:
            dig_twin = relevant_process_executions_df.drop(search_string, axis=1)
            order_group = dig_twin.merge(dig_twin_two, on=groupby_string)
            result_quality = order_group[['Resulting Quality', search_string]].groupby(search_string).agg('mean')

        result_quality *= 100
        result_quality = result_quality.T
        relevant_ids = list(set(relevant_ids).intersection(set(result_quality.columns)))
        quality_s = result_quality[relevant_ids].T

        return quality_s


class DeliveryReliability(KPICalc):

    def __init__(self, analytics_data_base: ScenarioAnalyticsDataBase):
        super().__init__(analytics_data_base=analytics_data_base)

    # all levels
    def get_delivery_reliability(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                                 view, scenario, all=False):
        """
        Calculates the delivery reliability

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        view: view passed from frontend
        scenario:
        all: True or False; True calculated for headline, False: calcualte for every (.. view)

        Returns
        -------
        a Series ((View(Order ID.....); delivery_reliability)
        """

        delivery_delays_raw = self.get_delivery_delay_raw(start_time, end_time, order_ids, part_ids, process_ids,
                                                          resource_ids, view, scenario,
                                                          consider_unfinished_orders=False)
        if view == "ORDER":
            id_list = order_ids
        elif view == "PRODUCT":
            id_list = part_ids
        elif view == "PROCESS":
            id_list = process_ids
        elif view == "RESOURCE":
            id_list = resource_ids

        if not delivery_delays_raw.size:
            return pd.Series([0.0])

        if not all:
            kpi_part = delivery_delays_raw[delivery_delays_raw >= pd.Timedelta(seconds=0)].groupby(level=0).agg('count')
            kpi_part *= 100
            kpi_sum = delivery_delays_raw.groupby(level=0).agg('count')
            kpi_delivery_reliability = kpi_part / kpi_sum

            all_id_df = pd.Series(index=id_list)
            common_ids = kpi_delivery_reliability.index.intersection(all_id_df.index)
            all_id_df = all_id_df.drop(common_ids)  # ToDo: error
            if not all_id_df.empty:
                kpi_delivery_reliability = pd.concat([kpi_delivery_reliability, all_id_df])

            kpi_delivery_reliability = sort_df(kpi_delivery_reliability, order_ids, part_ids, process_ids, resource_ids,
                                               view, self.analytics_data_base.digital_twin_df)
            kpi_delivery_reliability_s = kpi_delivery_reliability.fillna(0.0)

        else:
            kpi_part = delivery_delays_raw[delivery_delays_raw >= pd.Timedelta(seconds=0)].agg('count')
            kpi_part *= 100
            kpi_delivery_reliability_s = pd.Series([kpi_part / delivery_delays_raw.agg('count')])

        return kpi_delivery_reliability_s

    def get_delivery_delay(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                           view, scenario, all=False):
        """
        Calculates the delivery_delay

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        view: view passed from frontend
        scenario: scenario name
        all: True or False; True calculated for headline, False: calculate for every (.. view)

        Returns
        -------
        a Series ((View(Order ID.....); delivery_delay)
        """

        if view == "ORDER":
            id_list = order_ids
        elif view == "PRODUCT":
            id_list = part_ids
        elif view == "PROCESS":
            id_list = process_ids
        elif view == "RESOURCE":
            id_list = resource_ids

        delivery_delays_raw = self.get_delivery_delay_raw(start_time, end_time, order_ids, part_ids, process_ids,
                                                          resource_ids, view, scenario,
                                                          consider_unfinished_orders=True)

        delivery_delay = delivery_delays_raw[delivery_delays_raw < pd.Timedelta(seconds=0)].groupby(level=0).agg('mean')
        kpi_delivery_delay = delivery_delay.fillna(pd.Timedelta(seconds=0))
        if kpi_delivery_delay.any():
            kpi_delivery_delay = kpi_delivery_delay.dt.total_seconds() * (-1)

        if not all:
            undefined_ids = set(id_list).difference(kpi_delivery_delay.index)
            missing_series = pd.Series(["-" for i in range(len(undefined_ids))],
                                       index=undefined_ids)
            kpi_delivery_delay = pd.concat([kpi_delivery_delay, missing_series])

        kpi_delivery_delay = sort_df(kpi_delivery_delay, order_ids, part_ids, process_ids,
                                     resource_ids, view, self.analytics_data_base.digital_twin_df)

        if all:
            kpi_delivery_delay = pd.Series(kpi_delivery_delay.loc[kpi_delivery_delay != "-"].mean())

        return kpi_delivery_delay

    # def get_order_status(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
    #                      view, all):
    #     delivery_delays_raw = self.get_delivery_delay_raw(start_time, end_time, order_ids, part_ids, process_ids,
    #                                                       resource_ids, view, all)
    #
    #
    #     return

    def get_delivery_delay_raw(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids,
                               view, scenario, consider_unfinished_orders=False):
        """
        Calculates the delivery_delay for every raw in df

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        view: view passed from frontend
        scenario:

        Parameters
        ----------
        a Series ((View(Order ID.....); delivery_delay)
        """

        if view == "ORDER" or view == "PRODUCT":
            kpi_delivery_delay = (
                self.calculate_order_delivery_delay(start_time, end_time, order_ids, part_ids,
                                                    consider_unfinished_orders=consider_unfinished_orders,
                                                    view=view, scenario=scenario))

        elif view == "PROCESS" or view == "RESOURCE":
            kpi_delivery_delay = \
                self.calculate_delivery_delay(start_time, end_time, order_ids, part_ids, process_ids,
                                              resource_ids, view=view, scenario=scenario)
        else:
            raise NotImplementedError()

        if kpi_delivery_delay.isna:
            kpi_delivery_delay = kpi_delivery_delay.fillna(pd.Timedelta(seconds=0))

        return kpi_delivery_delay

    def calculate_delivery_delay(self, start_time, end_time, order_ids, part_ids, process_ids, resource_ids, view,
                                 scenario):
        """
        Calculates the delivery_delay

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        process_ids: The Process IDs selected in the filter
        resource_ids: The Resource IDs selected in the filter
        view: view passed from frontend
        scenario:

        Returns
        -------
        a Series ((View(Order ID.....); delivery_delay)
        """

        if view == 'PROCESS':
            group_column = 'Process ID'
        elif view == 'RESOURCE':
            group_column = 'Resource Used ID'
        else:
            raise Exception("View not supported")

        relevant_process_executions_plan_df = \
            self.get_pre_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids, part_ids=part_ids,
                                       process_ids=process_ids, resource_ids=resource_ids, event_type="PLAN",
                                       view=view, scenario=scenario)
        # relevant_process_executions_plan_df = relevant_process_executions_plan_df.copy().reset_index(drop=True)

        relevant_process_executions_actual_df = \
            self.get_pre_lead_time_dfs(start_time=start_time, end_time=end_time, order_ids=order_ids, part_ids=part_ids,
                                       process_ids=process_ids, resource_ids=resource_ids, event_type="ACTUAL",
                                       view=view, scenario=scenario)

        if not (relevant_process_executions_plan_df.size and relevant_process_executions_actual_df.size):
            # process_executions only available of type PLAN or ACTUAL not both

            if relevant_process_executions_plan_df.size:
                kpi_delivery_delay = pd.Series([pd.Timedelta(seconds=0)
                                                for _ in range(relevant_process_executions_plan_df.shape[0])],
                                               index=relevant_process_executions_plan_df[group_column])
            elif relevant_process_executions_actual_df.size:
                kpi_delivery_delay = pd.Series([pd.Timedelta(seconds=0)
                                                for _ in range(relevant_process_executions_actual_df.shape[0])],
                                               index=relevant_process_executions_actual_df[group_column])
            else:
                kpi_delivery_delay = pd.Series()

            return kpi_delivery_delay

        merge_df = self.analytics_data_base.digital_twin_df[['Process Execution ID', group_column]]
        merge_df = merge_df.dropna()

        # relevant_process_executions_actual_df = relevant_process_executions_actual_df.copy().reset_index(drop=True)
        if view == 'RESOURCE':
            relevant_process_executions_plan_df = relevant_process_executions_plan_df.drop(group_column, axis=1)
            relevant_process_executions_plan_df = relevant_process_executions_plan_df.merge(merge_df,
                                                                                            on='Process Execution ID')

            relevant_process_executions_actual_df = relevant_process_executions_actual_df.drop(group_column, axis=1)
            relevant_process_executions_actual_df = \
                relevant_process_executions_actual_df.merge(merge_df, on='Process Execution ID')

        relevant_process_executions_plan_df = relevant_process_executions_plan_df.sort_values('Process Execution ID')
        relevant_process_executions_plan_df = relevant_process_executions_plan_df.set_index('Process Execution ID')

        relevant_process_executions_actual_df = \
            relevant_process_executions_actual_df.sort_values('Connected Process Execution ID')

        if not relevant_process_executions_plan_df.empty and not relevant_process_executions_actual_df.empty:

            kpi_delivery_delay_s = relevant_process_executions_plan_df['End Time'] - \
                                   relevant_process_executions_actual_df['End Time']
            kpi_delivery_delay_s = merge_df.merge(kpi_delivery_delay_s, left_on='Process Execution ID',
                                                  right_index=True)
            kpi_delivery_delay_s = kpi_delivery_delay_s.set_index(group_column)
            kpi_delivery_delay_s = kpi_delivery_delay_s['End Time']
            kpi_delivery_delay = kpi_delivery_delay_s

        else:
            kpi_delivery_delay = pd.Series()

        return kpi_delivery_delay

    def calculate_order_delivery_delay(self, start_time, end_time, order_ids, part_ids, consider_unfinished_orders,
                                       view, scenario):
        """
        Calculates the delivery_delay

        Parameters
        ----------
        start_time: Start time passed from frontend
        end_time: End time passed from frontend
        order_ids: The Order IDs selected in the filter
        part_ids: The Part IDs selected in the filter
        view: view passed from frontend
        scenario:

        Returns
        -------
        a Series ((View(Order ID.....); delivery_delay)
        """

        finished_orders = self.get_orders_finished(start_time=start_time, end_time=end_time,
                                                   possible_order_ids=order_ids, scenario=scenario)
        finished_order_ids = finished_orders['Order ID']

        order_product_mask = (self.analytics_data_base.digital_twin_df["Order ID"].isin(finished_order_ids) &
                              self.analytics_data_base.digital_twin_df["Product Entity Type ID"].isin(part_ids))
        if view == "ORDER":
            relevant_columns = ["Delivery Date Planned", "Delivery Date Actual", 'Order ID']
        else:
            relevant_columns = ["Delivery Date Planned", "Delivery Date Actual", 'Product Entity Type ID']

        if consider_unfinished_orders:
            unfinished_order_ids = list(set(order_ids).difference(finished_order_ids))
            unfinished_order_product_mask = (
                    self.analytics_data_base.digital_twin_df["Order ID"].isin(unfinished_order_ids) &
                    self.analytics_data_base.digital_twin_df["Product Entity Type ID"].isin(part_ids))

            unfinished_order_df = self.analytics_data_base.digital_twin_df.loc[
                unfinished_order_product_mask, relevant_columns]
            last_process_execution_time_stamp = self.analytics_data_base.digital_twin_df["End Time"].max()
            # ToDo: maybe alternative timestamp also valid
            unfinished_order_df.loc[:, "Delivery Date Actual"] = last_process_execution_time_stamp

            finished_order_df = self.analytics_data_base.digital_twin_df.loc[order_product_mask, relevant_columns]
            order_df = pd.concat([finished_order_df, unfinished_order_df])

        else:
            order_df = self.analytics_data_base.digital_twin_df.loc[order_product_mask, relevant_columns]

        order_finished_df = order_df[["Delivery Date Planned", "Delivery Date Actual", relevant_columns[2]]]
        order_finished_df = order_finished_df.set_index(relevant_columns[2])

        kpi_delivery_delay_s = order_finished_df["Delivery Date Planned"] - order_finished_df["Delivery Date Actual"]

        # ToDo: kpi_delivery_delay_s[~finished_orders["Order ID"].to_list()]

        return kpi_delivery_delay_s


class ORE:
    """ToDo: a other way should be better - but this is better for the run time """

    def get_ore(self, quality, performance, availability):
        """The ORE is calculated based on the multiplication of the factor quality, performance and availability"""

        # kpi_ore = [(q * p * a) / 10000 for q, p, a in zip(quality, performance, availability)]
        if quality.size == 1:
            quality = pd.DataFrame(quality, columns=['Resulting Quality'])
        ore_df = quality.merge(performance, left_index=True, right_index=True)
        if availability.size == 1:
            availability = pd.DataFrame(availability, columns=['order_time'])
        ore_df = ore_df.merge(availability, left_index=True, right_index=True)

        # series column 0 =order time! do not know why
        ore_df['sum'] = ore_df['Resulting Quality'] * ore_df['performance'] * ore_df['order_time']
        kpi_ore = ore_df['sum'] / 10000

        return kpi_ore
